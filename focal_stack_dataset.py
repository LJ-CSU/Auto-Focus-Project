import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms  # 用于获取 get_params
import torchvision.transforms.functional as TF  # 用于执行同步变换
import random


class DefocusSceneDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_train=False):
        self.root_dir = root_dir
        self.transform = transform
        self.scenes = sorted(os.listdir(root_dir))
        self.is_train = is_train  # 增加训练模式标志

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        scene_path = os.path.join(self.root_dir, self.scenes[idx])
        files = sorted(os.listdir(scene_path))

        images = []
        focus_pos = []

        # 尺寸变换参数
        target_size = (256, 256)

        if self.is_train:
            # 定义：随机生成这一组 Focal Stack 共用的裁剪参数
            # 这里的 dummy 图像只要尺寸对就行，用来生成随机坐标
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                Image.new('RGB', target_size),
                scale=(0.8, 1.0),
                ratio=(1.0, 1.0)
            )
            # 定义：随机决定这一组图像是否翻转
            do_flip = random.random() > 0.5

        for fname in files:
            z = int(fname.split('z')[-1].split('.')[0])
            img = Image.open(os.path.join(scene_path, fname)).convert('RGB')

            # 基础预处理：先统一大小
            img = TF.resize(img, target_size)

            if self.is_train:
                # 使用 TF 模块手动应用上面生成的相同参数 (i, j, h, w)
                # 这样保证了 z=11 和 z=-11 裁掉的是同一个位置
                img = TF.resized_crop(img, i, j, h, w, size=target_size)
                if do_flip:
                    img = TF.hflip(img)

            # 最后调用 train.py 传进来的 transforms.ToTensor() 和 Normalize()
            if self.transform:
                img = self.transform(img)

            images.append(img)
            focus_pos.append(z)

        return torch.stack(images), torch.tensor(focus_pos)