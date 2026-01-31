import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        probs = torch.sigmoid(logits)

        probs = probs.view(probs.size(0), -1)
        target = target.view(target.size(0), -1)

        intersection = (probs * target).sum(dim=1)
        union = probs.sum(dim=1) + target.sum(dim=1)

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight=0.5, bce_weight=0.5, pos_weight=5.0):
        super().__init__()
        self.dice = DiceLoss()
        self.pos_weight = pos_weight
        self.dw = dice_weight
        self.bw = bce_weight

    def forward(self, logits, target):
        pos_weight = torch.tensor([self.pos_weight], device=logits.device)
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        dice_loss = self.dice(logits, target)
        bce_loss = bce(logits, target)

        return self.dw * dice_loss + self.bw * bce_loss
