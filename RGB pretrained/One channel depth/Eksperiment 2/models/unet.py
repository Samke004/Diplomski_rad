"""
models/unet.py
────────────────────────────────────────────────────────────────────────────
U-Net with a ResNet34 encoder pre-trained on ImageNet.
Uses the segmentation_models_pytorch (smp) library.

Paper:  ResNet34 encoder + U-Net decoder, trained with BCE loss.
Output: sigmoid probability map [B, 1, H, W].
"""

import torch
import torch.nn as nn

try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False


class _FallbackUNet(nn.Module):
    """Minimal fallback U-Net for multi-class segmentation."""

    def __init__(self, in_channels: int = 3, num_classes: int = 4):
        super().__init__()

        def double_conv(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            )

        self.enc1 = double_conv(in_channels, 64)
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), double_conv(64,  128))
        self.enc3 = nn.Sequential(nn.MaxPool2d(2), double_conv(128, 256))
        self.enc4 = nn.Sequential(nn.MaxPool2d(2), double_conv(256, 512))
        self.bot  = nn.Sequential(nn.MaxPool2d(2), double_conv(512, 1024))
        self.up4  = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = double_conv(1024, 512)
        self.up3  = nn.ConvTranspose2d(512,  256, 2, stride=2)
        self.dec3 = double_conv(512,  256)
        self.up2  = nn.ConvTranspose2d(256,  128, 2, stride=2)
        self.dec2 = double_conv(256,  128)
        self.up1  = nn.ConvTranspose2d(128,  64,  2, stride=2)
        self.dec1 = double_conv(128,  64)
        self.head = nn.Conv2d(64, num_classes, 1)   # multi-class output

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b  = self.bot(e4)
        d  = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d  = self.dec3(torch.cat([self.up3(d), e3], dim=1))
        d  = self.dec2(torch.cat([self.up2(d), e2], dim=1))
        d  = self.dec1(torch.cat([self.up1(d), e1], dim=1))
        return self.head(d)   # raw logits [B, num_classes, H, W]


def build_unet(
    encoder:     str  = "resnet34",
    weights:     str  = "imagenet",
    in_channels: int  = 3,
    num_classes: int  = 4,
) -> nn.Module:
    """
    Build U-Net with ResNet34 encoder.
    Returns raw logits [B, num_classes, H, W] — apply softmax externally.
    """
    if SMP_AVAILABLE:
        model = smp.Unet(
            encoder_name=encoder,
            encoder_weights=weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=None,          # raw logits for CrossEntropyLoss
        )
        print(f"[U-Net] Built with smp: encoder={encoder}, weights={weights}, classes={num_classes}")
    else:
        print("[U-Net] segmentation_models_pytorch not found → using fallback U-Net")
        model = _FallbackUNet(in_channels=in_channels, num_classes=num_classes)

    return model
