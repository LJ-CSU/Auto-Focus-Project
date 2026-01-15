import os
import numpy as np
import cv2
import random

# --- 配置参数 ---
# Alpha 范围 (Radius = Alpha * |z|)
ALPHA_RANGE = (4.0, 9.0)
# 离焦层级
DEFOCUS_LEVELS = [-11, -9, -7, -5, -3, -1, 0, 1, 3, 5, 7, 9, 11]


def load_images(folder_path):
    image_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
    ])
    images = []
    for fname in image_files:
        img = cv2.imread(os.path.join(folder_path, fname), cv2.IMREAD_COLOR)
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img.astype(np.float32) / 255.0)
    return images


# --- 1. 多样化的 PSF 生成 ---
def generate_random_kernel(radius):
    """
    生成不同类型的模糊核，模拟不同的镜头特性
    """
    size = int(2 * radius + 1)
    if size % 2 == 0: size += 1
    if size < 3: return np.zeros((3, 3), dtype=np.float32)  # Dummy

    # 随机选择核类型
    kernel_type = random.choice(['disk', 'disk_soft', 'gaussian'])

    kernel = np.zeros((size, size), dtype=np.float32)
    center = (size - 1) / 2
    y, x = np.indices((size, size))
    dist_sq = (x - center) ** 2 + (y - center) ** 2

    if kernel_type == 'disk':
        # 理想圆盘 (Hard Edge)
        mask = dist_sq <= radius ** 2
        kernel[mask] = 1.0

    elif kernel_type == 'disk_soft':
        # 软边圆盘 (模拟球差)
        sigma = radius / 2.0
        mask = dist_sq <= radius ** 2
        kernel[mask] = np.exp(-dist_sq[mask] / (2 * sigma ** 2))
        # 保持边缘有一定硬度
        kernel[dist_sq > radius ** 2] = 0

    elif kernel_type == 'gaussian':
        # 高斯模糊 (模拟变迹或衍射)
        # 使得高斯的视觉宽度近似于半径为 radius 的圆盘
        sigma = radius / 2.0
        kernel = np.exp(-dist_sq / (2 * sigma ** 2))

    # 归一化
    return kernel / (kernel.sum() + 1e-8)


# --- 2. 运动模糊 (模拟相机抖动) ---
def apply_motion_blur(image, intensity=0.3):
    """
    随机添加轻微的运动模糊
    """
    if random.random() > 0.5: return image  # 50% 概率不加

    kernel_size = random.randint(3, 7)
    kernel = np.zeros((kernel_size, kernel_size))

    # 随机角度
    angle = random.randint(0, 180)
    M = cv2.getRotationMatrix2D((kernel_size / 2 - 0.5, kernel_size / 2 - 0.5), angle, 1)

    # 生成线性核
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
    kernel = kernel / (kernel.sum() + 1e-8)

    return cv2.filter2D(image, -1, kernel)


# --- 3. 噪声注入 ---
def add_noise(image):
    noise_type = random.choice(['gaussian', 'poisson', 'none'])
    if noise_type == 'none':
        return image

    if noise_type == 'gaussian':
        sigma = random.uniform(0.005, 0.02)  # 随机噪声强度
        noise = np.random.normal(0, sigma, image.shape)
        return np.clip(image + noise, 0, 1)

    elif noise_type == 'poisson':
        # 模拟光子噪声
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return np.clip(noisy, 0, 1)

    return image


# --- 4. JPEG 压缩伪影 ---
def apply_jpeg_compression(image_uint8):
    if random.random() > 0.7: return image_uint8  # 30% 概率不加

    quality = random.randint(50, 95)  # 随机质量因子
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', image_uint8, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg


def process_and_save():
    source_images = load_images('./PSF_Source_data')    # 原清晰图片
    base_dir = './datasets/train_aug'  # 生成训练图片保存路径

    print(f"Loaded {len(source_images)} source images.")

    for i, img in enumerate(source_images):
        scene_dir = os.path.join(base_dir, f'scene_{i + 1:04d}')
        os.makedirs(scene_dir, exist_ok=True)

        # 随机设定该场景的 Alpha (模拟不同镜头)
        current_alpha = random.uniform(*ALPHA_RANGE)

        # 记录该场景所有图片的 radius 和 z
        # 网络将学习预测 radius，但保留 z 以便维持 dataset 的结构

        for z in DEFOCUS_LEVELS:
            if z == 0:
                radius = 0.0
                blurred = img
            else:
                # 1. 计算物理半径
                radius = current_alpha * abs(z)

                # 2. 生成随机 PSF 并卷积
                kernel = generate_random_kernel(radius)
                blurred = cv2.filter2D(img, -1, kernel)

            # 3. 叠加运动模糊 (干扰项)
            blurred = apply_motion_blur(blurred)

            # 4. 叠加噪声
            blurred = add_noise(blurred)

            # 转 uint8 准备做 JPEG 压缩
            blurred_uint8 = (blurred * 255.0).clip(0, 255).astype(np.uint8)
            blurred_bgr = cv2.cvtColor(blurred_uint8, cv2.COLOR_RGB2BGR)

            # 5. JPEG 压缩
            final_img = apply_jpeg_compression(blurred_bgr)

            # ---文件名保存真实半径 ---
            # 格式: img_z{z}_r{radius}.png
            # 例如: img_z5_r32.50.png
            save_name = f'img_z{z}_r{radius:.2f}.png'
            cv2.imwrite(os.path.join(scene_dir, save_name), final_img)

        print(f"Scene {i + 1} processed with Alpha={current_alpha:.2f}")


if __name__ == "__main__":
    process_and_save()