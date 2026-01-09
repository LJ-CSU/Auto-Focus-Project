import torch.nn as nn
import torchvision.models as models


class ResNetDefocus(nn.Module):
    """
    Single-image defocus state predictor
    Input : (B, 3, H, W)
    Output: (B, 1) continuous defocus state
    """

    def __init__(self, pretrained=True):
        super().__init__()

        backbone = models.resnet34(pretrained=pretrained)

        # Remove classification head
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])  # GAP included
        self.regressor = nn.Linear(512, 1)

    def forward(self, x):
        feat = self.encoder(x)          # (B, 512, 1, 1)
        feat = feat.view(x.size(0), -1) # (B, 512)
        out = self.regressor(feat)      # (B, 1)
        return out
