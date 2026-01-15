import torch
import torch.nn.functional as F
import torch.nn as nn

# --- 可微模糊层：放在 loss.py 中方便调用 ---
class DifferentiableBlurLayer(nn.Module):
    def __init__(self, kernel_size=31):
        super().__init__()
        self.kernel_size = kernel_size
        # 生成坐标网格 [-1, 1]
        range_vec = torch.linspace(-1, 1, kernel_size)
        y, x = torch.meshgrid(range_vec, range_vec, indexing='ij')
        self.register_buffer('grid', x ** 2 + y ** 2)  # 存储距离平方

    def forward(self, x, radius_normalized):
        """
        x: 清晰图像 (B, 3, H, W)
        radius_normalized: 预测的模糊半径 (B,)
        """
        B, C, H, W = x.shape
        # 使用 Sigmoid 模拟硬边缘，使其可微。temperature 越小越接近硬圆盘
        temperature = 0.1
        # radius_normalized 需要映射到 [0, 1]
        kernels = torch.sigmoid((radius_normalized.view(B, 1, 1) - torch.sqrt(self.grid)) / temperature)

        # 归一化内核
        kernels = kernels / (kernels.sum(dim=(1, 2), keepdim=True) + 1e-8)

        # 组卷积实现 Batch 内不同图片用不同内核
        x_reshaped = x.reshape(1, B * C, H, W)
        kernels = kernels.reshape(B, 1, 1, self.kernel_size, self.kernel_size)
        kernels = kernels.repeat(1, C, 1, 1, 1).reshape(B * C, 1, self.kernel_size, self.kernel_size)

        out = F.conv2d(x_reshaped, kernels, groups=B * C, padding=self.kernel_size // 2)
        return out.view(B, C, H, W)

# --- SSIM 损失函数 ---
def ssim_loss(img1, img2, window_size=11):
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size // 2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size // 2)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2

    sigma1_sq = F.avg_pool2d(img1 * img1, window_size, stride=1, padding=window_size // 2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, window_size, stride=1, padding=window_size // 2) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size // 2) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return 1 - ssim_map.mean()

def ranking_loss(pred, focus_pos):
    """
    pred      : (k,) predicted defocus states
    focus_pos: (k,) focus index or defocus proxy (0 = best focus)

    Enforces correct ordering w.r.t. defocus magnitude
    """
    k = pred.size(0)
    loss = 0.0
    cnt = 0

    for i in range(k):
        for j in range(k):
            if i == j:
                continue

            # ground-truth ordering
            gt_sign = torch.sign(
                torch.abs(focus_pos[j]) - torch.abs(focus_pos[i])
            )

            if gt_sign == 0:
                continue

            loss += F.relu(-gt_sign * (pred[j] - pred[i]))
            cnt += 1

    if cnt > 0:
        loss = loss / cnt

    return loss

def unimodal_loss(pred, focus_pos):
    """
    pred      : (k,)
    focus_pos: (k,)

    Encourage single minimum at focus_pos == 0
    """
    # sort by focus position
    idx = torch.argsort(focus_pos)
    pred_sorted = pred[idx]
    pos_sorted = focus_pos[idx]

    # find focus index (closest to zero)
    focus_idx = torch.argmin(torch.abs(pos_sorted))

    # left side: should decrease toward focus
    left = pred_sorted[:focus_idx + 1]
    if left.numel() > 1:
        left_diff = left[1:] - left[:-1]
        loss_left = torch.mean(F.relu(left_diff))  # penalize increase
    else:
        loss_left = 0.0

    # right side: should increase away from focus
    right = pred_sorted[focus_idx:]
    if right.numel() > 1:
        right_diff = right[1:] - right[:-1]
        loss_right = torch.mean(F.relu(-right_diff))  # penalize decrease
    else:
        loss_right = 0.0

    return loss_left + loss_right

# 暂时不用
def anchor_loss(pred, focus_pos, target_val=0.0):
    """
    锚定损失：强制 z=0 时的预测值为 target_val
    """
    # 找到 focus_pos 中最接近 0 的索引
    idx_zero = torch.argmin(torch.abs(focus_pos))
    pred_zero = pred[idx_zero]

    # 使用 MSE 损失锚定中心点
    return F.mse_loss(pred_zero, torch.tensor(target_val).to(pred.device))

def smoothness_loss_v2(pred, focus_pos):
    """
    二阶平滑损失：惩罚斜率的变化率，使曲线更加圆滑
    """
    # 按焦距顺序排列预测值
    idx = torch.argsort(focus_pos)
    pred_sorted = pred[idx]

    # 一阶差分（保持基本平滑）
    diff1 = pred_sorted[1:] - pred_sorted[:-1]
    l1_smooth = torch.mean(torch.abs(diff1))

    # 二阶差分（惩罚突变和锯齿）
    # 公式: pred[i-1] - 2*pred[i] + pred[i+1]
    if pred_sorted.numel() > 2:
        diff2 = pred_sorted[2:] - 2 * pred_sorted[1:-1] + pred_sorted[:-2]
        l2_smooth = torch.mean(diff2 ** 2)
    else:
        l2_smooth = 0.0

    return l1_smooth + l2_smooth


def defocus_total_loss(pred, focus_pos, recon_img, target_img,
                       w_rank=1.0,
                       w_smooth=0.8,
                       w_uni=0.5,
                       w_recon=2.0,
                       recon_type='ssim'):
    """
    组合损失函数
    """
    l_rank = ranking_loss(pred, focus_pos)
    l_smooth = smoothness_loss_v2(pred, focus_pos)
    l_uni = unimodal_loss(pred, focus_pos)

    if recon_type == 'ssim':
        l_recon = ssim_loss(recon_img, target_img)
    else:
        l_recon = F.mse_loss(recon_img, target_img)

    total = (
            w_rank * l_rank +
            w_smooth * l_smooth +
            w_uni * l_uni +
            w_recon * l_recon
    )

    return total, {
        'rank': l_rank.item(),
        'smooth': l_smooth.item(),
        'unimodal': l_uni.item(),
        'recon': l_recon.item()
    }