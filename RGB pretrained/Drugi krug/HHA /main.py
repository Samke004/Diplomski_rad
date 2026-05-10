"""
main.py
────────────────────────────────────────────────────────────────────────────
Orchestrates the full pipeline:
  1. Build datasets and data loaders
  2. Train all models
  3. Evaluate on the test set with all metrics
  4. Save results and visualisations

Usage
─────
    # Train and evaluate ALL models
    python main.py

    # Train a single model only
    python main.py --model unet_resnet34
    python main.py --model unet_efficientnet_b4
    python main.py --model deeplabv3_resnet50
    python main.py --model deeplabv3plus_mobilenet
    python main.py --model segformer_b1

    # Skip training, run evaluation only (requires saved checkpoints)
    python main.py --eval_only
"""

import argparse
import os
import torch
from torch.utils.data import DataLoader

from config import Config as cfg
from dataset import AppleTreeDataset
from augmentations import get_train_transforms, get_val_transforms
from models import (build_unet, build_deeplabv3, build_deeplabv3plus, build_segformer)
from trainer import SegmentationTrainer
from evaluate import evaluate_all


# ─────────────────────────────────────────────────────────────────────────────
# Build data loaders
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Individual trainers
# ─────────────────────────────────────────────────────────────────────────────

def train_unet(train_loader, val_loader) -> dict:
    model = build_unet(
        encoder=cfg.UNET_ENCODER,
        weights=cfg.UNET_WEIGHTS,
        in_channels=cfg.IN_CHANNELS,
        num_classes=cfg.NUM_CLASSES,
    )
    return SegmentationTrainer(
        model=model, model_name="unet_resnet34", cfg=cfg,
        train_loader=train_loader, val_loader=val_loader,
        lr=cfg.UNET_LR,
    ).train()


def train_unet_efficientnet(train_loader, val_loader) -> dict:
    model = build_unet(
        encoder=cfg.UNET_EFF_ENCODER,
        weights=cfg.UNET_EFF_WEIGHTS,
        in_channels=cfg.IN_CHANNELS,
        num_classes=cfg.NUM_CLASSES,
    )
    return SegmentationTrainer(
        model=model, model_name="unet_efficientnet_b4", cfg=cfg,
        train_loader=train_loader, val_loader=val_loader,
        lr=cfg.UNET_EFF_LR,
    ).train()


def train_deeplabv3(train_loader, val_loader) -> dict:
    model = build_deeplabv3(
        pretrained=cfg.DEEPLABV3_PRETRAINED,
        in_channels=cfg.IN_CHANNELS,
        num_classes=cfg.NUM_CLASSES,
    )
    return SegmentationTrainer(
        model=model, model_name="deeplabv3_resnet50", cfg=cfg,
        train_loader=train_loader, val_loader=val_loader,
        lr=cfg.DEEPLABV3_LR,
    ).train()


def train_deeplabv3plus(train_loader, val_loader) -> dict:
    model = build_deeplabv3plus(
        encoder=cfg.DEEPLABV3PLUS_ENCODER if hasattr(cfg, 'DEEPLABV3PLUS_ENCODER')
                else "mobilenet_v2",
        weights="imagenet" if cfg.DEEPLABV3PLUS_PRETRAINED else None,
        in_channels=cfg.IN_CHANNELS,
        num_classes=cfg.NUM_CLASSES,
    )
    return SegmentationTrainer(
        model=model, model_name="deeplabv3plus_mobilenetv2", cfg=cfg,
        train_loader=train_loader, val_loader=val_loader,
        lr=cfg.DEEPLABV3PLUS_LR,
    ).train()


def train_segformer(train_loader, val_loader) -> dict:
    model = build_segformer(
        variant=cfg.SEGFORMER_VARIANT,
        weights="imagenet",
        in_channels=cfg.IN_CHANNELS,
        num_classes=cfg.NUM_CLASSES,
    )
    return SegmentationTrainer(
        model=model, model_name=f"segformer_{cfg.SEGFORMER_VARIANT}", cfg=cfg,
        train_loader=train_loader, val_loader=val_loader,
        lr=cfg.SEGFORMER_LR,
    ).train()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

MODEL_TRAINERS = {
    "unet_resnet34":           train_unet,
    "unet_efficientnet_b4":    train_unet_efficientnet,
    "deeplabv3_resnet50":      train_deeplabv3,
    "deeplabv3plus_mobilenet": train_deeplabv3plus,
    "segformer_b1":            train_segformer,
}


def main():
    parser = argparse.ArgumentParser(description="Apple Tree Segmentation Pipeline")
    parser.add_argument(
        "--model",
        choices=["all"] + list(MODEL_TRAINERS.keys()),
        default="all",
        help="Which model to train (default: all)",
    )
    parser.add_argument(
        "--eval_only", action="store_true",
        help="Skip training and run evaluation only",
    )
    args = parser.parse_args()

    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    print(f"\nDevice : {cfg.DEVICE}")
    print(f"Epochs : {cfg.NUM_EPOCHS}")
    print(f"Batch  : {cfg.BATCH_SIZE}")
    print(f"Size   : {cfg.IMAGE_HEIGHT}×{cfg.IMAGE_WIDTH}")
    print(f"Channels: {cfg.IN_CHANNELS}")

    histories = {}

    if not args.eval_only:
        train_loader, val_loader = build_loaders()

        to_train = (
            list(MODEL_TRAINERS.keys()) if args.model == "all"
            else [args.model]
        )

        for name in to_train:
            print(f"\n{'#'*60}")
            print(f"  MODEL: {name.upper()}")
            print(f"{'#'*60}")
            histories[name] = MODEL_TRAINERS[name](train_loader, val_loader)

    print(f"\n{'#'*60}")
    print(f"  EVALUATION")
    print(f"{'#'*60}")
    evaluate_all(histories if histories else None)


if __name__ == "__main__":
    main()