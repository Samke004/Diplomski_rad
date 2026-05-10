from transformers import SegformerForSemanticSegmentation
import torch.nn as nn
import torch

class SegFormerWrapper(nn.Module):
    def __init__(self, variant="b1", in_channels=4, num_classes=4):
        super().__init__()
        name = f"nvidia/segformer-{variant}-finetuned-ade-512-512"
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
        # Adapt first conv for 4-channel input
        if in_channels != 3:
            old = self.model.segformer.encoder.patch_embeddings[0].proj
            new = nn.Conv2d(in_channels, old.out_channels,
                           kernel_size=old.kernel_size, stride=old.stride,
                           padding=old.padding, bias=old.bias is not None)
            with torch.no_grad():
                new.weight[:, :3] = old.weight.clone()
                nn.init.kaiming_normal_(new.weight[:, 3:])
            self.model.segformer.encoder.patch_embeddings[0].proj = new

    def forward(self, x):
        out = self.model(pixel_values=x)
        logits = out.logits  # [B, num_classes, H/4, W/4]
        # Upsample to input size
        logits = torch.nn.functional.interpolate(
            logits, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        return logits

def build_segformer(variant="b1", weights="imagenet", in_channels=4, num_classes=4):
    model = SegFormerWrapper(variant=variant, in_channels=in_channels, num_classes=num_classes)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[SegFormer-{variant.upper()}] Built: in_channels={in_channels}, classes={num_classes}, params={n/1e6:.1f}M")
    return model
