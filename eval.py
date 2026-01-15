import matplotlib.pyplot as plt
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
