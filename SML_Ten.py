import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

Threshold_SML = 40
Threshold_Ten = 5

def load_images_and_focuspos(folder_path):
    """
    按文件名顺序读取文件夹内所有图像
    """
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

def load_images(folder_path):
    image_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
    ])
    images = []
    for fname in image_files:
        img = cv2.imread(os.path.join(folder_path, fname), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        images.append(img.astype(np.float32))

    return images

def compute_sml(image, threshold=0.0):
    """
    SML (Sum of Modified Laplacian)
    """
    # 水平方向二阶差分
    lap_x = np.abs(
        2 * image[:, 1:-1] - image[:, :-2] - image[:, 2:]
    )

    # 垂直方向二阶差分
    lap_y = np.abs(
        2 * image[1:-1, :] - image[:-2, :] - image[2:, :]
    )

    # 对齐尺寸
    lap_x = lap_x[1:-1, :]
    lap_y = lap_y[:, 1:-1]

    ml = lap_x + lap_y

    if threshold > 0:
        ml = ml[ml >= threshold]

    return np.sum(ml)


def compute_tenengrad(image, threshold=0.0):
    """
    Tenengrad (Sobel Gradient Energy)
    """
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)

    grad_mag = np.sqrt(gx ** 2 + gy ** 2)

    if threshold > 0:
        grad_mag = grad_mag[grad_mag >= threshold]

    return np.sum(grad_mag ** 2)


def normalize_curve(curve):
    """
    归一化到 [0, 1]，便于对比
    """
    curve = np.array(curve)
    return (curve - curve.min()) / (curve.max() - curve.min() + 1e-8)


def focus_measure_curve(image_folder):
    images, focus_pos = load_images_and_focuspos(image_folder)

    sml_scores = []
    ten_scores = []

    for img in images:
        sml_scores.append(compute_sml(img, Threshold_SML))
        ten_scores.append(compute_tenengrad(img, Threshold_Ten))

    sml_norm = normalize_curve(sml_scores)
    ten_norm = normalize_curve(ten_scores)

    plt.figure(figsize=(8, 5))
    plt.plot(focus_pos, sml_norm, marker='o', label='SML')
    plt.plot(focus_pos, ten_norm, marker='s', label='Tenengrad')

    plt.xlabel('Focus Position')
    plt.ylabel('Normalized Focus Measure')
    plt.title('Focus Measure Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def focus_compare_curve(image_folder):
    images = load_images(image_folder)
    x = list(range(1,len(images)+1))
    h_data = [0.25,0.734,0.29,0.36,0.742,0.792,0.77,0.716,0.74,0.682,0.726,
              0.776,0.886,0.806,0.774,0.66,0.666,0.28,0.476,0.8]
    sml_scores = []
    ten_scores = []

    for img in images:
        sml_scores.append(compute_sml(img, Threshold_SML))
        ten_scores.append(compute_tenengrad(img, Threshold_Ten))

    sml_norm = normalize_curve(sml_scores)
    ten_norm = normalize_curve(ten_scores)

    plt.figure(figsize=(8, 5))
    plt.plot(x, sml_norm, marker='o', label='SML')
    plt.plot(x, ten_norm, marker='s', label='Tenengrad')
    plt.plot(x, h_data, marker='*', label='subjective data')

    plt.xlabel('Image Index')
    plt.ylabel('Normalized Focus Measure')
    plt.title('Focus Measure Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image_folder_1 = "./datasets/scene_0001"  #离焦图像序列
    image_folder_2 = "./test_data"
    focus_measure_curve(image_folder_1)
    focus_compare_curve(image_folder_2)
