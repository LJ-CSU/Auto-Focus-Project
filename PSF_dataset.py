import os
import numpy as np
import cv2
import random

# --- 增强参数配置 ---
# 调大 alpha 确保 256x256 下仍有明显模糊感
ALPHA_RANGE = (5.0, 8.0)
DEFOCUS_LEVELS = [-11, -9, -7, -5, -3, -1, 0, 1, 3, 5, 7, 9, 11]

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


def disk_psf_advanced(radius, z):
    """
    带物理非对称性的 PSF 生成
    """
    # 确保 size 始终为奇数，利于中心对称
    size = int(2 * radius + 1)
    if size % 2 == 0:
        size += 1

    if size < 3:
        psf = np.zeros((3, 3), dtype=np.float32)
        psf[1, 1] = 1.0
        return psf

    # 使用 size 直接生成坐标轴，确保形状与 psf 矩阵完全一致
    center = (size - 1) / 2
    y, x = np.indices((size, size))
    dist_sq = (x - center) ** 2 + (y - center) ** 2

    mask = dist_sq <= radius ** 2
    psf = np.zeros((size, size), dtype=np.float32)

    if z >= 0:
        # 正离焦：均匀分布
        psf[mask] = 1.0
    else:
        # 负离焦：模拟球差 (Spherical Aberration)
        # 边缘亮度高于中心，打破视觉对称性
        # 避免半径为 0 导致除以 0
        safe_radius_sq = max(radius ** 2, 1e-6)
        edge_weight = np.exp(dist_sq / (2 * safe_radius_sq))
        psf[mask] = edge_weight[mask]

    # 防止全黑或全零的情况
    sum_val = psf.sum()
    if sum_val == 0:
        psf[int(center), int(center)] = 1.0
    else:
        psf /= sum_val

    return psf


def apply_blur_with_noise(image, z, alpha):
    """
    卷积并添加轻微噪声，模拟真实传感器特性
    """
    if z == 0:
        return image

    radius = alpha * abs(z)
    psf = disk_psf_advanced(radius, z)

    # 卷积
    blurred = cv2.filter2D(image, -1, psf)

    # 添加极微量的噪声，防止网络过拟合理想卷积
    noise = np.random.normal(0, 0.002, blurred.shape).astype(np.float32)
    blurred = np.clip(blurred + noise, 0, 1)

    return blurred


def process_and_save():
    source_images = load_images('./PSF_Source_data')
    base_dir = './datasets/train'

    for i, img in enumerate(source_images):
        scene_dir = os.path.join(base_dir, f'scene_{i + 1:04d}')
        os.makedirs(scene_dir, exist_ok=True)

        # 每个场景随机一个 alpha，让网络学习“相对关系”而非“固定像素半径”
        current_alpha = random.uniform(*ALPHA_RANGE)

        for z in DEFOCUS_LEVELS:
            blurred = apply_blur_with_noise(img, z, current_alpha)

            # 转为 uint8 保存
            blurred_uint8 = (blurred * 255.0).clip(0, 255).astype(np.uint8)
            blurred_bgr = cv2.cvtColor(blurred_uint8, cv2.COLOR_RGB2BGR)

            cv2.imwrite(os.path.join(scene_dir, f'img_z{z}.png'), blurred_bgr)


if __name__ == "__main__":
    process_and_save()