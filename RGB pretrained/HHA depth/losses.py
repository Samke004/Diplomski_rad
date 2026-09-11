"""
losses.py — Multi-class segmentation losses
────────────────────────────────────────────────────────────────────────────
  U-Net / DeepLabv3 : CrossEntropy + multi-class Dice
  Pix2Pix GAN       : Adversarial (BCEWithLogits) + λ·CrossEntropy
  Pix2Pix Gen only  : CrossEntropy only
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Multi-class Dice Loss ─────────────────────────────────────────────────────

class MultiClassDiceLoss(nn.Module):
    """
    Soft Dice loss averaged over all foreground classes (ignores background).
    pred   : [B, C, H, W] raw logits
    target : [B, H, W]    long class indices
    """
    def __init__(self, num_classes: int = 4, smooth: float = 1.0, ignore_bg: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.smooth      = smooth
        self.ignore_bg   = ignore_bg

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs  = F.softmax(pred, dim=1)                   # [B, C, H, W]
        target_one_hot = F.one_hot(target, self.num_classes)  # [B, H, W, C]
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()  # [B, C, H, W]

        start_cls = 1 if self.ignore_bg else 0
        dice_per_class = []
        for c in range(start_cls, self.num_classes):
            p = probs[:, c].reshape(-1)
            t = target_one_hot[:, c].reshape(-1)
            intersection = (p * t).sum()
            dice = (2 * intersection + self.smooth) / (p.sum() + t.sum() + self.smooth)
            dice_per_class.append(1.0 - dice)

        return torch.stack(dice_per_class).mean()


# ── CrossEntropy + Dice (main loss for U-Net and DeepLabv3) ──────────────────

class CrossEntropyDiceLoss(nn.Module):
    """
    Weighted combination of CrossEntropy and multi-class Dice loss.

    Parameters
    ----------
    num_classes  : number of output classes
    ce_weight    : weight for CE term
    dice_weight  : weight for Dice term
    class_weights: optional per-class weights tensor for CE (handles imbalance)
    """
    def __init__(
        self,
        num_classes:   int   = 4,
        ce_weight:     float = 0.5,
        dice_weight:   float = 0.5,
        class_weights: torch.Tensor = None,
    ):
        super().__init__()
        self.dice_weight  = dice_weight
        self.focal_weight = ce_weight
        self.dice  = MultiClassDiceLoss(num_classes=num_classes, ignore_bg=True)
        try:
            import segmentation_models_pytorch as smp
            self.focal   = smp.losses.FocalLoss(mode="multiclass", gamma=2.0)
        except Exception:
            self.focal   = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (self.focal_weight * self.focal(logits, target) +
                self.dice_weight  * self.dice(logits, target))


# ── GAN Loss ──────────────────────────────────────────────────────────────────

class GANLoss(nn.Module):
    def __init__(self, mode: str = "vanilla"):
        super().__init__()
        self.criterion = (nn.BCEWithLogitsLoss() if mode == "vanilla"
                          else nn.MSELoss())

    def _label(self, pred, is_real):
        return torch.ones_like(pred) if is_real else torch.zeros_like(pred)

    def forward(self, prediction: torch.Tensor, is_real: bool) -> torch.Tensor:
        return self.criterion(prediction, self._label(prediction, is_real))


# ── Pix2Pix Generator Loss ────────────────────────────────────────────────────

class Pix2PixGeneratorLoss(nn.Module):
    """
    Generator loss = adversarial + λ * CrossEntropy
    When use_gan=False: CrossEntropy only
    """
    def __init__(self, num_classes: int = 4, lambda_ce: float = 100.0, use_gan: bool = True):
        super().__init__()
        self.lambda_ce = lambda_ce
        self.use_gan   = use_gan
        self.gan_loss  = GANLoss("vanilla") if use_gan else None
        self.ce        = nn.CrossEntropyLoss()

    def forward(self, fake_pred, fake_logits, real_target):
        """
        fake_pred    : discriminator output on fake [B, 1, H, W] (or None)
        fake_logits  : generator output [B, C, H, W]
        real_target  : ground truth labels [B, H, W] long
        """
        ce_loss = self.ce(fake_logits, real_target) * self.lambda_ce
        if self.use_gan and fake_pred is not None:
            return self.gan_loss(fake_pred, is_real=True) + ce_loss
        return ce_loss
