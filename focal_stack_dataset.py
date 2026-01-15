import os
import re
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms  # 用于获取 get_params
import random


class DefocusSceneDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_train=False):
        self.root_dir = root_dir
        self.transform = transform
        self.scenes = sorted(os.listdir(root_dir))
        self.is_train = is_train

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        scene_path = os.path.join(self.root_dir, self.scenes[idx])
        files = sorted(os.listdir(scene_path), key=lambda x: int(re.search(r'z(-?\d+)', x).group(1)))

        images = []
        radii = []  # 半径标签
        focus_pos = []  # 离焦位置标签，用于排序或绘图

        # 同一组的尺寸参数一致，读取第一张图来获得原始尺寸
        temp_img = Image.open(os.path.join(scene_path, files[0])).convert('RGB')
        w_orig, h_orig = temp_img.size

        target_size = (256, 256) # 缩放尺寸

        if self.is_train:
            # 定义：随机生成这一组 Focal Stack 共用的裁剪参数
            i, j, h, w = transforms.RandomCrop.get_params(
                temp_img, output_size=target_size
            )
            # 定义：随机决定这一组图像是否翻转
            do_flip = random.random() > 0.5
        else:
            # 验证集始终切中心
            i = (h_orig - target_size[0]) // 2
            j = (w_orig - target_size[1]) // 2
            h = target_size[0]
            w = target_size[1]
            do_flip = False

        # 读取图片
        for fname in files:
            # 解析 z 和 radius
            # 训练图像文件名示例: img_z5_r32.50.png
            # 验证/测试文件名示例： img_z5.png
            z_match = re.search(r'z(-?\d+)', fname)
            z = int(z_match.group(1)) if z_match else 0
            r_match = re.search(r'r(\d+\.?\d*)', fname)
            r = float(r_match.group(1)) if r_match else 0.0

            img = Image.open(os.path.join(scene_path, fname)).convert('RGB')

            # 使用 TF 模块手动应用上面生成的相同参数 (i, j, h, w)
            # 这样保证了 z 和 -z 裁掉的是同一个位置
            img = TF.crop(img, i, j, h, w)
            if do_flip:
                img = TF.hflip(img)

            # 最后调用 train.py 传进来的 transforms.ToTensor() 和 Normalize()
            if self.transform:
                img = self.transform(img)

            images.append(img)
            radii.append(r)
            focus_pos.append(z)

        # 归一化 Radius 标签
        # 假设最大可能半径是 Alpha_max(9.0) * z_max(11) ≈ 100
        # 将半径归一化到 [0, 1] 区间方便网络回归，预测时再乘回来
        MAX_RADIUS = 100.0
        radii_tensor = torch.tensor(radii, dtype=torch.float32) / MAX_RADIUS

        return torch.stack(images), radii_tensor, torch.tensor(focus_pos)