"""
helpmain.py
Pokrece samo preostale modele (bez unet_resnet34 i unet_efficientnet_b4)
"""

import os
import torch
from torch.utils.data import DataLoader

from config import Config as cfg
from dataset import AppleTreeDataset
from augmentations import get_train_transforms, get_val_transforms
from models import (build_unet, build_deeplabv3, build_deeplabv3plus, build_segformer)
from trainer import SegmentationTrainer
from evaluate import evaluate_all


def build_loaders():
    train_tf = get_train_transforms(cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH)
    val_tf   = get_val_transforms(cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH)

    train_ds = AppleTreeDataset(
        img_dir=cfg.TRAIN_IMG_DIR, mask_dir=cfg.TRAIN_MASK_DIR,
        transform=train_tf,
        img_size=(cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH),
    )
    val_ds = AppleTreeDataset(
        img_dir=cfg.VAL_IMG_DIR, mask_dir=cfg.VAL_MASK_DIR,
        transform=val_tf,
        img_size=(cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH),
    )
    print(f"Train: {len(train_ds)} images  |  Val: {len(val_ds)} images")

    loader_kw = dict(
        batch_size  = cfg.BATCH_SIZE,
        num_workers = cfg.NUM_WORKERS,
        pin_memory  = cfg.PIN_MEMORY,
    )
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kw)
    return train_loader, val_loader


def main():
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    print(f"\nDevice : {cfg.DEVICE}")
    print(f"Epochs : {cfg.NUM_EPOCHS}")
    print(f"Batch  : {cfg.BATCH_SIZE}")
    print(f"Size   : {cfg.IMAGE_HEIGHT}x{cfg.IMAGE_WIDTH}")
    print(f"Channels: {cfg.IN_CHANNELS}")

    train_loader, val_loader = build_loaders()
    histories = {}

    # ── DeepLabV3 ─────────────────────────────────────────────────────────────
    print(f"\n{'#'*60}\n  MODEL: DEEPLABV3_RESNET50\n{'#'*60}")
    from models import build_deeplabv3
    model = build_deeplabv3(
        pretrained=cfg.DEEPLABV3_PRETRAINED,
        in_channels=cfg.IN_CHANNELS,
        num_classes=cfg.NUM_CLASSES,
    )
    histories["deeplabv3_resnet50"] = SegmentationTrainer(
        model=model, model_name="deeplabv3_resnet50", cfg=cfg,
        train_loader=train_loader, val_loader=val_loader,
        lr=cfg.DEEPLABV3_LR,
    ).train()

    # ── DeepLabV3+ ────────────────────────────────────────────────────────────
    print(f"\n{'#'*60}\n  MODEL: DEEPLABV3PLUS_MOBILENETV2\n{'#'*60}")
    from models import build_deeplabv3plus
    model = build_deeplabv3plus(
        encoder="mobilenet_v2",
        weights="imagenet" if cfg.DEEPLABV3PLUS_PRETRAINED else None,
        in_channels=cfg.IN_CHANNELS,
        num_classes=cfg.NUM_CLASSES,
    )
    histories["deeplabv3plus_mobilenetv2"] = SegmentationTrainer(
        model=model, model_name="deeplabv3plus_mobilenetv2", cfg=cfg,
        train_loader=train_loader, val_loader=val_loader,
        lr=cfg.DEEPLABV3PLUS_LR,
    ).train()

    # ── SegFormer ─────────────────────────────────────────────────────────────
    print(f"\n{'#'*60}\n  MODEL: SEGFORMER_B1\n{'#'*60}")
    from models import build_segformer
    model = build_segformer(
        variant=cfg.SEGFORMER_VARIANT,
        weights="imagenet",
        in_channels=cfg.IN_CHANNELS,
        num_classes=cfg.NUM_CLASSES,
    )
    histories["segformer_b1"] = SegmentationTrainer(
        model=model, model_name="segformer_b1", cfg=cfg,
        train_loader=train_loader, val_loader=val_loader,
        lr=cfg.SEGFORMER_LR,
    ).train()

    # ── Evaluacija ────────────────────────────────────────────────────────────
    print(f"\n{'#'*60}\n  EVALUATION\n{'#'*60}")
    evaluate_all(histories)


if __name__ == "__main__":
    main()