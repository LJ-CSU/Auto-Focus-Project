import torch
from models.resnet_defocus import ResNetDefocus
from loss import defocus_total_loss, DifferentiableBlurLayer
from focal_stack_dataset import DefocusSceneDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from eval import eval_defocus_curve

# 设置设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device: {device}')

# 设置超参数
epochs = 20
learn_rate = 1e-4
w_radii = 10.0
w_rank = 1.0    # 强制学习模糊程度的顺序
w_smooth = 0.1  # 保证曲线平滑
w_uni = 0.1     # 保证单峰性
w_recon = 0.0   # 重构权重，按需使用，仅作为辅助正则项

# 设置模型，优化器
model = ResNetDefocus(pretrained=True).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

# 设置PSF约束
blur_layer = DifferentiableBlurLayer(kernel_size=31).to(device)

# 创建数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = DefocusSceneDataset(
    root_dir="./datasets/train_aug",
    transform=transform,
    is_train=True #是训练集
)

validate_dataset = DefocusSceneDataset(
    root_dir="./datasets/validate",
    transform=transform,
)

test_dataset = DefocusSceneDataset(
    root_dir="./datasets/test",
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

# 训练过程
for epoch in range(1, epochs + 1):
    train(epoch)
    if epoch % 5 == 0:
        images, _, focus_pos = next(iter(validate_dataloader))
        images = images.squeeze(0)
        focus_pos = focus_pos.squeeze(0)
        eval_defocus_curve(
            model,
            images,
            focus_pos,
            save_path=None
        )