import os
import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt
from scipy.fftpack import dct


def load_images_from_folder(folder_path):
    image_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
    ])
    focus_pos = []
    images = []
    for fname in image_files:
        z = int(fname.split('z')[-1].split('.')[0])
        img = cv2.imread(os.path.join(folder_path, fname), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        images.append(img.astype(np.float32))
        focus_pos.append(z)
    pos_idx = np.argsort(focus_pos)
    focus_pos = [focus_pos[i] for i in pos_idx]
    images = [images[i] for i in pos_idx]
    return images, focus_pos


# -------------------------
# DCT 高频能量
# -------------------------
def compute_dct_focus(image, block_size=8, hf_ratio=0.5):
    h, w = image.shape
    h_crop = h - h % block_size
    w_crop = w - w % block_size
    img = image[:h_crop, :w_crop]

    energy = 0.0
    hf_start = int(block_size * hf_ratio)

    for i in range(0, h_crop, block_size):
        for j in range(0, w_crop, block_size):
            block = img[i:i + block_size, j:j + block_size]
            block_dct = dct(dct(block.T, norm='ortho').T, norm='ortho')

            hf_coeffs = block_dct[hf_start:, hf_start:]
            energy += np.sum(hf_coeffs ** 2)

    return energy


# -------------------------
# 小波高频子带能量
# -------------------------
def compute_wavelet_focus(image, wavelet='db2', level=1):
    coeffs = pywt.wavedec2(image, wavelet=wavelet, level=level)
    energy = 0.0

    for i in range(1, len(coeffs)):
        LH, HL, HH = coeffs[i]
        energy += np.sum(LH ** 2) + np.sum(HL ** 2) + np.sum(HH ** 2)

    return energy


def normalize_curve(curve):
    curve = np.array(curve)
    return (curve - curve.min()) / (curve.max() - curve.min() + 1e-8)


def main(image_folder):
    images, focus_pos = load_images_from_folder(image_folder)

    dct_scores = []
    wave_scores = []

    for img in images:
        dct_scores.append(compute_dct_focus(img))
        wave_scores.append(compute_wavelet_focus(img))

    dct_norm = normalize_curve(dct_scores)
    wave_norm = normalize_curve(wave_scores)

    plt.figure(figsize=(8, 5))
    plt.plot(focus_pos, dct_norm, marker='o', label='DCT High-Frequency Energy')
    plt.plot(focus_pos, wave_norm, marker='s', label='Wavelet High-Frequency Energy')

    plt.xlabel('Focus Position / Image Index')
    plt.ylabel('Normalized Focus Measure')
    plt.title('Frequency-Domain Focus Measure Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    image_folder = './datasets/scene_0001'  # 修改为你的图像文件夹路径
    main(image_folder)
