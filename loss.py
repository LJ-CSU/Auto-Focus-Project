import torch
import torch.nn.functional as F

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


def defocus_total_loss(pred, focus_pos,
                       w_rank=1.0,
                       w_smooth=0.8,
                       w_uni=0.5):
    """
    组合损失函数
    """
    l_rank = ranking_loss(pred, focus_pos)
    l_smooth = smoothness_loss_v2(pred, focus_pos)
    l_uni = unimodal_loss(pred, focus_pos)
    l_anchor = anchor_loss(pred, focus_pos)

    total = (
            w_rank * l_rank +
            w_smooth * l_smooth +
            w_uni * l_uni
    )

    return total, {
        'rank': l_rank.item(),
        'smooth': l_smooth.item(),
        'unimodal': l_uni.item(),
        'anchor': l_anchor.item()
    }