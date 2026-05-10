"""
esanet.py — ESANet: Efficient RGB-D Semantic Segmentation
────────────────────────────────────────────────────────────────────────────
Paper: "Efficient RGB-D Semantic Segmentation for Indoor Scene Analysis"
       Seichter et al., ICRA 2021

Arhitektura:
  - Dva odvojena enkodera: RGB stream (ResNet34) + Depth stream (ResNet18)
  - Fusion na 4 razine enkodiranja (SE-weighted addition)
  - Decoder s skip connections i upsampling
  - Ulaz: [B, 4, H, W] — automatski se dijeli na RGB [B,3] i Depth [B,1]
  - Izlaz: raw logits [B, num_classes, H, W]

Konzistentno s ostatkom projekta:
  - Isti interface kao build_unet, build_deeplabv3 itd.
  - Radi s postojećim SegmentationTrainer, CrossEntropyDiceLoss, metrics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, resnet18, ResNet34_Weights, ResNet18_Weights


# ── Squeeze-and-Excitation Fusion ─────────────────────────────────────────────

class SEFusion(nn.Module):
    """
    SE-based channel attention fusion.
    Uči koliko RGB vs. Depth doprinosi na svakoj razini.
    """
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels * 2, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels * 2),
            nn.Sigmoid(),
        )

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([rgb, depth], dim=1)          # [B, 2C, H, W]
        B, C2, H, W = combined.shape
        weights = self.se(combined).view(B, C2, 1, 1)      # [B, 2C, 1, 1]
        w_rgb   = weights[:, :C2//2]
        w_depth = weights[:, C2//2:]
        return rgb * w_rgb + depth * w_depth               # [B, C, H, W]


# ── Decoder Block ─────────────────────────────────────────────────────────────

class DecoderBlock(nn.Module):
    """
    Upsample x2 + skip connection + conv.
    """
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Poravnaj dimenzije ako se ne slažu (zbog padding/stride artefakata)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ── Depth Encoder (1-kanal → isti channel dims kao RGB encoder) ───────────────

class DepthEncoder(nn.Module):
    """
    ResNet18 modificiran za 1-kanalni (depth) ulaz.
    Manji od RGB enkodera — paper koristi asimetričan dizajn.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base    = resnet18(weights=weights)

        # Prilagodi prvi conv za 1-kanalni ulaz
        old_conv = base.conv1
        new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            # Inicijaliziraj kao prosjek RGB kanala
            new_conv.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)
        base.conv1 = new_conv

        self.layer0 = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1   # 64
        self.layer2 = base.layer2   # 128
        self.layer3 = base.layer3   # 256
        self.layer4 = base.layer4   # 512

    def forward(self, x):
        e0 = self.layer0(x)
        e1 = self.layer1(e0)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)
        return e1, e2, e3, e4


# ── RGB Encoder ───────────────────────────────────────────────────────────────

class RGBEncoder(nn.Module):
    """
    ResNet34 za 3-kanalni RGB ulaz.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        base    = resnet34(weights=weights)

        self.layer0 = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1   # 64
        self.layer2 = base.layer2   # 128
        self.layer3 = base.layer3   # 256
        self.layer4 = base.layer4   # 512

    def forward(self, x):
        e0 = self.layer0(x)
        e1 = self.layer1(e0)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)
        return e1, e2, e3, e4


# ── Channel alignment (ResNet18 → ResNet34 dims) ──────────────────────────────

class ChannelAlign(nn.Module):
    """Poravnava depth feature channels na RGB channel dims."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        if in_ch != out_ch:
            self.proj = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        else:
            self.proj = nn.Identity()

    def forward(self, x):
        return self.proj(x)


# ── ESANet ────────────────────────────────────────────────────────────────────

class ESANet(nn.Module):
    """
    ESANet — Efficient RGB-D Semantic Segmentation.

    Ulaz : [B, 4, H, W] — RGB (kanali 0-2) + Depth (kanal 3)
    Izlaz: [B, num_classes, H, W] — raw logits
    """

    # ResNet34 channel dims: [64, 128, 256, 512]
    # ResNet18 channel dims: [64, 128, 256, 512]
    RGB_DIMS   = [64, 128, 256, 512]
    DEPTH_DIMS = [64, 128, 256, 512]

    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super().__init__()

        # ── Enkoderi ──────────────────────────────────────────────────────────
        self.rgb_encoder   = RGBEncoder(pretrained=pretrained)
        self.depth_encoder = DepthEncoder(pretrained=pretrained)

        # ── Channel alignment (depth → iste dims kao RGB) ─────────────────────
        self.align = nn.ModuleList([
            ChannelAlign(d, r)
            for d, r in zip(self.DEPTH_DIMS, self.RGB_DIMS)
        ])

        # ── SE Fusion na svakoj od 4 razine ───────────────────────────────────
        self.fusions = nn.ModuleList([
            SEFusion(c) for c in self.RGB_DIMS
        ])

        # ── Context module na bottlenecku ─────────────────────────────────────
        self.context = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # ── Dekoder ───────────────────────────────────────────────────────────
        # Ulaz u dekoder: 512 (bottleneck) + skip connections
        self.dec4 = DecoderBlock(512, 256, 256)   # + skip3 (256)
        self.dec3 = DecoderBlock(256, 128, 128)   # + skip2 (128)
        self.dec2 = DecoderBlock(128, 64,  64)    # + skip1 (64)
        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # ── Glava ─────────────────────────────────────────────────────────────
        self.head = nn.Conv2d(64, num_classes, 1)

        self._init_decoder_weights()

    def _init_decoder_weights(self):
        for m in [self.dec4, self.dec3, self.dec2, self.dec1, self.head,
                  self.context, self.fusions, self.align]:
            for layer in (m.modules() if hasattr(m, 'modules') else [m]):
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode="fan_out")
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Razdvoji RGB i Depth
        rgb   = x[:, :3, :, :]   # [B, 3, H, W]
        depth = x[:, 3:4, :, :]  # [B, 1, H, W]

        # Enkoderi
        rgb_feats   = self.rgb_encoder(rgb)     # (e1, e2, e3, e4)
        depth_feats = self.depth_encoder(depth) # (e1, e2, e3, e4)

        # Fusion na svakoj razini
        fused = []
        for i, (rf, df) in enumerate(zip(rgb_feats, depth_feats)):
            df_aligned = self.align[i](df)
            fused.append(self.fusions[i](rf, df_aligned))

        f1, f2, f3, f4 = fused

        # Context na bottlenecku
        f4 = self.context(f4)

        # Dekoder
        d = self.dec4(f4, f3)
        d = self.dec3(d,  f2)
        d = self.dec2(d,  f1)
        d = self.dec1(d)

        # Finalna predikcija — poravnaj na ulaznu veličinu
        if d.shape[-2:] != x.shape[-2:]:
            d = F.interpolate(d, size=x.shape[-2:], mode="bilinear", align_corners=False)

        return self.head(d)


# ── Builder ───────────────────────────────────────────────────────────────────

def build_esanet(
    pretrained:  bool = True,
    in_channels: int  = 4,
    num_classes: int  = 4,
) -> nn.Module:
    """
    Build ESANet.
    in_channels mora biti 4 (RGB + Depth) — model interno dijeli kanale.
    """
    assert in_channels == 4, "ESANet zahtijeva točno 4 ulazna kanala (RGB + Depth)"
    model = ESANet(num_classes=num_classes, pretrained=pretrained)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[ESANet] Built: pretrained={pretrained}, "
          f"in_channels={in_channels}, classes={num_classes}, "
          f"params={n/1e6:.1f}M")
    return model


if __name__ == "__main__":
    # Brzi test
    model = build_esanet(pretrained=False, num_classes=4)
    x     = torch.randn(2, 4, 640, 640)
    with torch.no_grad():
        out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")   # [2, 4, 640, 640]
    assert out.shape == (2, 4, 640, 640), "Shape mismatch!"
    print("✓ ESANet test prošao")