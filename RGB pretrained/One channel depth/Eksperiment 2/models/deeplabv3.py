import segmentation_models_pytorch as smp

def build_deeplabv3(pretrained=True, in_channels=3, num_classes=4):
    weights = "imagenet" if pretrained else None
    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights=weights,
        in_channels=in_channels,
        classes=num_classes,
        activation=None,
    )
    print(f"[DeepLabV3+ResNet50] Built: weights={weights}, in_channels={in_channels}, classes={num_classes}")
    return model
