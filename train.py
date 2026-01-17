import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from models.resnet_defocus import ResNetDefocus
from loss import defocus_total_loss, DifferentiableBlurLayer, ranking_loss
from focal_stack_dataset import DefocusSceneDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from eval import eval_defocus_curve, eval_error_defocus
import torch.optim as optim

# 保存路径
SAVE_DIR = "./checkpoint"
FIG_DIR = "./figure"

# 设置设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device: {device}')

# 设置超参数
epochs = 30
learn_rate = 1e-4
w_radii = 10.0
w_rank = 1.0    # 强制学习模糊程度的顺序
w_smooth = 0.05  # 保证曲线平滑
w_uni = 0.05     # 保证单峰性
w_recon = 0.005   # 重构权重，按需使用，仅作为辅助正则项
NORM_FACTOR = 100.0

# 设置模型，优化器
model = ResNetDefocus(pretrained=True).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

# 设置PSF约束
blur_layer = DifferentiableBlurLayer(kernel_size=31).to(device)

# 创建数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = DefocusSceneDataset(
    root_dir="./dataset/train",
    transform=transform,
    is_train=True #是训练集
)

validate_dataset = DefocusSceneDataset(
    root_dir="./dataset/validate",
    transform=transform,
)

test_dataset = DefocusSceneDataset(
    root_dir="./dataset/test",
    transform=transform,
)

# 创建数据加载器
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
validate_dataloader = DataLoader(validate_dataset, batch_size=1)
test_dataloader = DataLoader(test_dataset, batch_size=1)

# 训练函数
def train(epoch):
    loss_epoch = 0.0
    model.train()
    for batch_idx, (images, gt_radii, focus_pos) in enumerate(train_dataloader):
        """
        images.shape: (batch_size, k, 3, H, W)
        focus_pos.shape: (batch_size, k)
        """
        images = images.squeeze(0).to(device)        # (k, 3, H, W)
        gt_radii = gt_radii.squeeze(0).to(device)    # (k,)
        focus_pos = focus_pos.squeeze(0).to(device)  # (k,)

        # 前向预测 (预测归一化半径)
        pred_radii = model(images).squeeze(1)  # (k,)

        # PSF重构
        z0_idx = torch.argmin(torch.abs(focus_pos))
        img_clear = images[z0_idx:z0_idx + 1].expand(images.size(0), -1, -1, -1).clone()
        pred_radii = pred_radii + 1e-4 #增加一个小的偏移量 (EPS)，防止半径完全为 0
        recon_images = blur_layer(img_clear, pred_radii)

        loss,loss_dict = defocus_total_loss(pred_radii, gt_radii, focus_pos, recon_images, images,
                                            w_radii, w_rank, w_smooth, w_uni, w_recon)
        loss_epoch += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(
            f"Epoch[{epoch}] "
            f"[{batch_idx+1}/{len(train_dataloader)}] "
            f"Loss={loss.item():.4f} | "
            f"Radii_Loss={loss_dict['radii']:.4f} "
            f"Rank={loss_dict['rank']:.4f} "
            f"Smooth={loss_dict['smooth']:.4f} "
            f"Uni={loss_dict['unimodal']:.4f} "
            f"Recon={loss_dict['recon']:.4f}"
        )

    print(
        f"===> Epoch {epoch} Complete: "
        f"Avg Loss = {loss_epoch / len(train_dataloader):.4f}"
    )


def validate(epoch):
    model.eval()  # 切换到评估模式 (关闭 Dropout, 锁定 BatchNorm)

    total_loss_meter = 0.0
    mae_meter = 0.0  # Mean Absolute Error (平均绝对误差)
    steps = 0

    # 验证阶段必须关闭梯度计算，节省显存并加速
    with torch.no_grad():
        for batch_idx, (images, gt_radii, focus_pos) in enumerate(validate_dataloader):

            # 1. 数据搬运
            images = images.squeeze(0).to(device)  # Shape: (K, 3, 256, 256)
            gt_radii = gt_radii.squeeze(0).to(device)  # Shape: (K,)
            focus_pos = focus_pos.squeeze(0).to(device)  # Shape: (K,)

            # 2. 前向传播
            # ResNet 输出是 (K, 1)，需要 squeeze 成 (K,) 与 gt_radii 对齐
            pred_radii = model(images).squeeze(1)

            # 3. 计算验证损失
            # 主要关注 MSE (准确度)，Ranking Loss 仅作参考
            # 与训练 Loss 保持量级一致（防止 Scheduler 误判），保留少量 Rank 权重
            loss_mse = F.mse_loss(pred_radii, gt_radii)
            loss_rank = ranking_loss(pred_radii, focus_pos)
            val_loss = loss_mse + 0.1 * loss_rank

            # 4. 统计指标
            total_loss_meter += val_loss.item()

            # 计算 MAE (归一化后的误差，0.0-1.0)
            batch_mae = torch.abs(pred_radii - gt_radii).mean().item()
            mae_meter += batch_mae

            steps += 1

    # 计算 Epoch 平均值
    avg_loss = total_loss_meter / steps
    avg_mae = mae_meter / steps

    # 将误差还原回物理像素，比如 MAE=0.05, 乘以 100 后，表示平均预测误差是 5 个像素
    real_pixel_error = avg_mae * NORM_FACTOR

    print(f"\n[Validation] Epoch {epoch} Report:")
    print(f"  > Avg Loss : {avg_loss:.6f} (用于调度器)")
    print(f"  > Avg MAE  : {avg_mae:.6f} (归一化后)")
    print(f"  > Real Err : {real_pixel_error:.2f} pixels (真实物理误差)")

    return avg_loss

# 训练过程记录
train_loss_history = []
val_loss_history = []
lr_history = []

best_val_loss = float("inf")
best_epoch = -1

# 训练
for epoch in range(1, epochs + 1):
    train(epoch)
    val_loss = validate(epoch)
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch} | Current LR: {current_lr}")

    val_loss_history.append(val_loss)
    lr_history.append(current_lr)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch

        save_path = os.path.join(SAVE_DIR, "best_model.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss
        }, save_path)

        print(f"Best model updated at epoch {epoch}, val_loss = {val_loss:.6f}")

    if epoch % 5 == 0:
        images, _, focus_pos = next(iter(test_dataloader))
        images = images.squeeze(0)
        focus_pos = focus_pos.squeeze(0)
        save_name = f'test_output/output_img_{epoch / 5}.png'
        output_path=os.path.join(FIG_DIR, save_name)
        eval_defocus_curve(
            model,
            images,
            focus_pos,
            save_path=None
        )

print("\n================ Training Finished ================")
print(f"Best validation loss : {best_val_loss:.6f}")
print(f"Best epoch           : {best_epoch}")

# --------- Error vs Defocus Analysis ----------
eval_error_defocus(
    model=model,
    dataloader=validate_dataloader,
    device=device,
    norm_factor=NORM_FACTOR,
    save_path="./figure/error_vs_defocus.png"
)

# --------- Plot Validation Loss ----------
plt.figure(figsize=(6, 4))
plt.plot(range(1, len(val_loss_history) + 1), val_loss_history, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title("Validation Loss Curve")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "val_loss_curve.png"), dpi=150)
plt.close()

# --------- Plot Learning Rate ----------
plt.figure(figsize=(6, 4))
plt.plot(range(1, len(lr_history) + 1), lr_history, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("Learning Rate Schedule")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "lr_curve.png"), dpi=150)
plt.close()

print(f"Curves saved to {FIG_DIR}")
