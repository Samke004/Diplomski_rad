"""
models/pix2pix.py
────────────────────────────────────────────────────────────────────────────
Pix2Pix for semantic segmentation as used in the paper.

Architecture (original Isola et al. 2017):
  Generator  : U-Net with 8 encoder / 8 decoder blocks.
               Input  [B, 3, 256, 256] → Output [B, 1, 256, 256] (sigmoid)
  Discriminator : 70×70 PatchGAN.
               Input  [B, 4, 256, 256] (image cat mask) → patch logits

Two training modes (both are evaluated in the paper):
  "gan" → full adversarial training (Generator + Discriminator)
  "gen" → generator only, trained with L1 loss (no Discriminator)
"""

import torch
import torch.nn as nn


# ── Building Blocks ──────────────────────────────────────────────────────────

class _DownBlock(nn.Module):
    """Encoder block: Conv(stride=2) → [BN] → LeakyReLU"""
    def __init__(self, cin: int, cout: int, use_bn: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(cin, cout, 4, stride=2, padding=1, bias=not use_bn)
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(cout))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class _UpBlock(nn.Module):
    """Decoder block: ConvTranspose(stride=2) → BN → [Dropout] → ReLU"""
    def __init__(self, cin: int, cout: int, use_dropout: bool = False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(cin, cout, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(cout),
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# ── Generator (U-Net, 8 blocks) ──────────────────────────────────────────────

class Pix2PixGenerator(nn.Module):
    """
    U-Net Generator for Pix2Pix.
    For 256×256 input: 8 downsampling → 1×1 bottleneck → 8 upsampling.

    Parameters
    ----------
    in_channels  : input image channels (3 = RGB, 4 = RGB + depth)
    out_channels : 1 for binary segmentation
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 4):
        super().__init__()

        # ── Encoder ─────────────────────────────────────────────────────────
        # No BN on the first block (standard Pix2Pix)
        self.d1 = _DownBlock(in_channels, 64,   use_bn=False)  # 256→128
        self.d2 = _DownBlock(64,          128)                  # 128→64
        self.d3 = _DownBlock(128,         256)                  # 64→32
        self.d4 = _DownBlock(256,         512)                  # 32→16
        self.d5 = _DownBlock(512,         512)                  # 16→8
        self.d6 = _DownBlock(512,         512)                  # 8→4
        self.d7 = _DownBlock(512,         512)                  # 4→2
        self.d8 = _DownBlock(512,         512)                  # 2→1 (bottleneck)

        # ── Decoder (with skip connections → concat doubles channels) ────────
        # First 3 decoder blocks use Dropout (paper standard)
        self.u1 = _UpBlock(512,  512, use_dropout=True)   # 1→2,   cat d7 → 1024
        self.u2 = _UpBlock(1024, 512, use_dropout=True)   # 2→4,   cat d6 → 1024
        self.u3 = _UpBlock(1024, 512, use_dropout=True)   # 4→8,   cat d5 → 1024
        self.u4 = _UpBlock(1024, 512)                     # 8→16,  cat d4 → 1024
        self.u5 = _UpBlock(1024, 256)                     # 16→32, cat d3 → 512
        self.u6 = _UpBlock(512,  128)                     # 32→64, cat d2 → 256
        self.u7 = _UpBlock(256,  64)                      # 64→128,cat d1 → 128

        # Final output: raw logits for CrossEntropyLoss
        self.final = nn.ConvTranspose2d(128, out_channels, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.d1(x)
        e2 = self.d2(e1)
        e3 = self.d3(e2)
        e4 = self.d4(e3)
        e5 = self.d5(e4)
        e6 = self.d6(e5)
        e7 = self.d7(e6)
        e8 = self.d8(e7)   # bottleneck

        # Decoder with skip connections
        d = self.u1(e8)
        d = self.u2(torch.cat([d, e7], dim=1))
        d = self.u3(torch.cat([d, e6], dim=1))
        d = self.u4(torch.cat([d, e5], dim=1))
        d = self.u5(torch.cat([d, e4], dim=1))
        d = self.u6(torch.cat([d, e3], dim=1))
        d = self.u7(torch.cat([d, e2], dim=1))

        return self.final(torch.cat([d, e1], dim=1))   # raw logits [B, 4, H, W]


# ── Discriminator (70×70 PatchGAN) ──────────────────────────────────────────

class Pix2PixDiscriminator(nn.Module):
    """
    PatchGAN discriminator.
    Classifies overlapping 70×70 patches as real or fake.

    Input  : concatenation of image [B, C, H, W] and mask [B, 1, H, W]
    Output : patch-level logit map (no sigmoid — use BCEWithLogitsLoss)
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()

        # in_channels (image) + num_classes (mask logits)
        cin = in_channels + 4

        def disc_block(cin, cout, stride, use_bn=True):
            layers = [nn.Conv2d(cin, cout, 4, stride=stride, padding=1, bias=not use_bn)]
            if use_bn:
                layers.append(nn.BatchNorm2d(cout))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            disc_block(cin, 64,  stride=2, use_bn=False),  # 256→128
            disc_block(64,  128, stride=2),                  # 128→64
            disc_block(128, 256, stride=2),                  # 64→32
            disc_block(256, 512, stride=1),                  # 32→32 (stride=1 for last)
            nn.Conv2d(512, 1, 4, stride=1, padding=1),      # → patch logit map
        )

    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        image : [B, C, H, W]
        mask  : [B, 1, H, W]  (ground truth or generated)
        """
        x = torch.cat([image, mask], dim=1)
        return self.model(x)
