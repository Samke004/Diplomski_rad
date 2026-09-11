import segmentation_models_pytorch as smp

def build_manet(pretrained=True, in_channels=4, num_classes=4):
    weights = "imagenet" if pretrained else None
    model = smp.MAnet(
        encoder_name="resnext50_32x4d",
        encoder_weights=weights,
        in_channels=in_channels,
        classes=num_classes,
        activation=None,
    )
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MA-Net ResNeXt50] Built: weights={weights}, in_channels={in_channels}, classes={num_classes}, params={n/1e6:.1f}M")
    return model
