import os
from PIL import Image
import torch
from torch.utils.data import Dataset


class DefocusSceneDataset(Dataset):
    """
    Each item = one scene (focal sweep)
    """

    def __init__(self, root_dir, transform=None, k=None):
        """
        root_dir/
            scene_xxxx/
                img_z-4.png
                img_z0.png
                ...
        k: number of images sampled per scene (None = use all)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.k = k
        self.scenes = sorted(os.listdir(root_dir))

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        scene_path = os.path.join(self.root_dir, self.scenes[idx])
        files = sorted(os.listdir(scene_path))

        images = []
        focus_pos = []

        for fname in files:
            # parse z from filename, e.g. img_z-2.png
            z = int(fname.split('z')[-1].split('.')[0])

            img = Image.open(os.path.join(scene_path, fname)).convert('RGB')
            if self.transform:
                img = self.transform(img)

            images.append(img)
            focus_pos.append(z)

        images = torch.stack(images, dim=0)      # (k, C, H, W)
        focus_pos = torch.tensor(focus_pos)      # (k,)

        return images, focus_pos
