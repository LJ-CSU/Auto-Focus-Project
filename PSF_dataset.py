import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

alpha = 4.0
defocus_levels = [-11, -9, -7, -5, -3, -1, 0, 1, 3, 5, 7, 9, 11] #最大 PSF 半径 ≈ 图像中等结构尺寸的 1/3～1/2

def load_images(folder_path):

    image_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
    ])
    images = []
    for fname in image_files:
        img = cv2.imread(os.path.join(folder_path, fname), cv2.IMREAD_COLOR)
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img.astype(np.float32)/255.0)

    return images

def disk_psf(radius):

    size = int(2 * radius + 1)
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x**2 + y**2 <= radius**2
    psf = np.zeros((size, size), dtype=np.float32)
    psf[mask] = 1.0
    psf /= psf.sum()
    return psf


def apply_psf(image, psf):

    return cv2.filter2D(image, -1, psf)


def blur_rgb(image, psf):

    return np.stack(
        [apply_psf(image[:, :, c], psf) for c in range(3)],
        axis=2
    )


if __name__ == "__main__":
    images = load_images('./PSF_Source_data')
    num_pic = len(images)
    base_dir = './datasets/train'
    for i in range(1, num_pic + 1):
        img = images[i-1]
        folder_name = f'scene_{i:04d}'
        scene_dir = os.path.join(base_dir, folder_name)
        os.makedirs(scene_dir, exist_ok=True)
        for z in defocus_levels:
            if z == 0:
                blurred = img.copy()
            else:
                r = int(alpha * abs(z))
                psf = disk_psf(r)
                blurred = blur_rgb(img, psf)

            # ---------- 保存前处理 ----------
            blurred_uint8 = np.clip(blurred * 255.0, 0, 255).astype(np.uint8)

            # RGB → BGR（cv2.imwrite 要求）
            blurred_uint8 = cv2.cvtColor(blurred_uint8, cv2.COLOR_RGB2BGR)
            pic_path = os.path.join(scene_dir, f'img_z{z}.png')
            cv2.imwrite(
                pic_path,
                blurred_uint8
            )

