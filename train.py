import torch
from models.resnet_defocus import ResNetDefocus
from loss import defocus_total_loss
from focal_stack_dataset import DefocusSceneDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from eval import eval_defocus_curve

# 设置设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device: {device}')

# 设置超参数
epochs = 10
learn_rate = 1e-4
w_rank = 1.0
w_smooth = 0.8
w_uni = 0.5

# 设置模型，优化器
model = ResNetDefocus(pretrained=True).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

# 创建数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = DefocusSceneDataset(
    root_dir="./datasets/train",
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
    for batch_idx, (images, focus_pos) in enumerate(train_dataloader):
        """
        images.shape: (batch_size, k, 3, H, W)
        focus_pos.shape: (batch_size, k)
        """
        images = images.squeeze(0).to(device)        # (k, 3, H, W)
        focus_pos = focus_pos.squeeze(0).to(device)  # (k,)
        pred = model(images).squeeze(1)  # (k,)

        loss,loss_dict = defocus_total_loss(pred, focus_pos, w_rank, w_smooth, w_uni)
        loss_epoch += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(
            f"Epoch[{epoch}] "
            f"[{batch_idx+1}/{len(train_dataloader)}] "
            f"Loss={loss.item():.4f} | "
            f"Rank={loss_dict['rank']:.4f} "
            f"Smooth={loss_dict['smooth']:.4f} "
            f"Uni={loss_dict['unimodal']:.4f}"
        )

    print(
        f"===> Epoch {epoch} Complete: "
        f"Avg Loss = {loss_epoch / len(train_dataloader):.4f}"
    )

# 训练过程
for epoch in range(1, epochs + 1):
    train(epoch)
    if epoch % 5 == 0:
        images, focus_pos = next(iter(test_dataloader))
        images = images.squeeze(0)
        focus_pos = focus_pos.squeeze(0)
        eval_defocus_curve(
            model,
            images,
            focus_pos,
            save_path=None
        )