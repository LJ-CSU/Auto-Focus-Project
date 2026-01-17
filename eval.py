import matplotlib.pyplot as plt
import numpy as np
import torch

@torch.no_grad()
def eval_defocus_curve(model, images, focus_pos, save_path=None):
    """
    model     : trained model
    images    : (k, 3, H, W)
    focus_pos : (k,)
    """

    model.eval()

    images = images.to(next(model.parameters()).device)
    focus_pos = focus_pos.to(images.device)

    # forward
    pred_radii = model(images).squeeze(1)  # (k,)

    # sort by focus position
    idx = torch.argsort(focus_pos)
    focus_sorted = focus_pos[idx].cpu().numpy()
    pred_sorted = pred_radii[idx].cpu().numpy()

    # plot
    plt.figure(figsize=(6, 4))
    plt.plot(focus_sorted, pred_sorted, marker='o')
    plt.xlabel('Focus Position (proxy)')
    plt.ylabel('Predicted Defocus State')
    plt.title('Defocus Prediction Curve')
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

@torch.no_grad()
def eval_error_defocus(
    model,
    dataloader,
    device,
    norm_factor=100.0,
    save_path="./figure/error_vs_defocus.png"
):
    """
    Evaluate prediction error vs defocus degree |z|

    X-axis: |focus_pos|
    Y-axis: MAE of predicted radius (in pixel domain)
    """

    model.eval()

    # 用 dict 按 |z| 聚合误差
    error_dict = {}   # { |z| : [errors...] }

    for images, gt_radii, focus_pos in dataloader:
        # shapes: (1, K, 3, H, W)
        images = images.squeeze(0).to(device)
        gt_radii = gt_radii.squeeze(0).to(device)
        focus_pos = focus_pos.squeeze(0).to(device)

        # forward
        pred_radii = model(images).squeeze(1)

        # 还原到真实物理尺度（像素）
        pred_radii_real = pred_radii * norm_factor
        gt_radii_real = gt_radii * norm_factor

        abs_error = torch.abs(pred_radii_real - gt_radii_real)

        for z, err in zip(focus_pos, abs_error):
            z_abs = int(torch.abs(z).item())
            if z_abs not in error_dict:
                error_dict[z_abs] = []
            error_dict[z_abs].append(err.item())

    # 计算每个 |z| 的平均误差
    z_vals = sorted(error_dict.keys())
    mean_errors = [np.mean(error_dict[z]) for z in z_vals]

    # --------- Plot ----------
    plt.figure(figsize=(6, 4))
    plt.plot(z_vals, mean_errors, marker='o')
    plt.xlabel("Defocus Degree |z|")
    plt.ylabel("Mean Absolute Error (pixels)")
    plt.title("Prediction Error vs Defocus Degree")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print("Error vs Defocus evaluation finished.")
    print("Saved to:", save_path)

    # 返回数据
    return z_vals, mean_errors
