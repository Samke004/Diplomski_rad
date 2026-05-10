"""
models/deeplabv3plus.py
────────────────────────────────────────────────────────────────────────────
DeepLabV3+ with MobileNetV2 backbone via segmentation_models_pytorch.

Key improvement over DeepLabV3:
  - Encoder-decoder with skip connections → better edge / thin structure detail
  - MobileNetV2 backbone → only 5.8M parameters, fast training
  - Better for thin elongated structures (branches)

Output: raw logits [B, num_classes, H, W]
"""

import torch.nn as nn

try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False


def build_deeplabv3plus(
    encoder:     str  = "mobilenet_v2",
    weights:     str  = "imagenet",
    in_channels: int  = 3,
    num_classes: int  = 4,
) -> nn.Module:
    """
    Build DeepLabV3+ with MobileNetV2 encoder.

    Parameters
    ----------
    encoder     : backbone (default mobilenet_v2 — 5.8M params)
    weights     : pretrained weights ('imagenet' or None)
    in_channels : 3 (RGB) or 4 (RGBD)
    num_classes : number of output classes
    """
    if not SMP_AVAILABLE:
        raise ImportError(
            "segmentation_models_pytorch is required for DeepLabV3+.\n"
            "Install with: pip install segmentation-models-pytorch"
        )

    model = smp.DeepLabV3Plus(
        encoder_name=encoder,
        encoder_weights=weights,
        in_channels=in_channels,
        classes=num_classes,
        activation=None,   # raw logits for CrossEntropyLoss
    )

    print(f"[DeepLabV3+] Built: encoder={encoder}, "
          f"weights={weights}, in_channels={in_channels}, classes={num_classes}")
    return model
