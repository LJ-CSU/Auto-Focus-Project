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

def smoothness_loss(pred, focus_pos):
    """
    pred      : (k,)
    focus_pos: (k,)
    Penalize abrupt changes along focal sweep
    """
    # sort by focus position
    idx = torch.argsort(focus_pos)
    pred_sorted = pred[idx]

    # first-order difference
    diff = pred_sorted[1:] - pred_sorted[:-1]

    loss = torch.mean(torch.abs(diff))
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

def defocus_total_loss(pred, focus_pos,
                       w_rank=1.0,
                       w_smooth=0.1,
                       w_uni=0.05):
    """
    Combined loss for defocus state learning
    """
    l_rank = ranking_loss(pred, focus_pos)
    l_smooth = smoothness_loss(pred, focus_pos)
    l_uni = unimodal_loss(pred, focus_pos)

    total = (
        w_rank * l_rank +
        w_smooth * l_smooth +
        w_uni * l_uni
    )

    return total, {
        'rank': l_rank.item(),
        'smooth': l_smooth.item(),
        'unimodal': l_uni.item()
    }
