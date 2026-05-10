"""
config.py
Central configuration for the apple tree segmentation pipeline.
Edit this file to match your system and dataset before running.
"""

import os
import torch


class Config:
    # ── Data Paths ──────────────────────────────────────────────────────────
    # Images are *_rgbd.npy files (4-channel: RGB + depth)
    # Masks  are *_mask.png files (0=bg, 1=trunk, 2=branches, 3=support)
    DATA_DIR        = "data"
    TRAIN_IMG_DIR   = os.path.join(DATA_DIR, "train", "images")
    TRAIN_MASK_DIR  = os.path.join(DATA_DIR, "train", "masks")
    VAL_IMG_DIR     = os.path.join(DATA_DIR, "val",   "images")
    VAL_MASK_DIR    = os.path.join(DATA_DIR, "val",   "masks")
    TEST_IMG_DIR    = os.path.join(DATA_DIR, "test",  "images")
    TEST_MASK_DIR   = os.path.join(DATA_DIR, "test",  "masks")

    # ── Image ───────────────────────────────────────────────────────────────
    # Original PLY projection size is 640x480
    # Resized to 512x512 for model input (power of 2, works well for all models)
    IMAGE_HEIGHT    = 640
    IMAGE_WIDTH     = 640
    IN_CHANNELS     = 6   # RGB + Depth (all packed in _rgbd.npy)

    # ── Training ────────────────────────────────────────────────────────────
    BATCH_SIZE           = 4
    NUM_EPOCHS           = 100
    EARLY_STOP_PATIENCE  = 15
    DEVICE               = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS          = 2
    PIN_MEMORY           = True if torch.cuda.is_available() else False

    # ── U-Net (ResNet34 backbone, ImageNet pretrained) ───────────────────────
    UNET_LR         = 1e-4
    UNET_ENCODER    = "resnet34"
    UNET_WEIGHTS    = "imagenet"

    # ── U-Net (EfficientNet-B4 backbone) ─────────────────────────────────────
    UNET_EFF_LR      = 1e-4
    UNET_EFF_ENCODER = "efficientnet-b4"
    UNET_EFF_WEIGHTS = "imagenet"

    # ── DeepLabv3 (ResNet50 backbone) ────────────────────────────────────────
    DEEPLABV3_LR         = 1e-4
    DEEPLABV3_PRETRAINED = True

    # ── DeepLabv3+ (MobileNetV2 backbone) ────────────────────────────────────
    DEEPLABV3PLUS_LR         = 1e-4
    DEEPLABV3PLUS_PRETRAINED = True

    # ── SegFormer-B1 ──────────────────────────────────────────────────────────
    SEGFORMER_LR      = 6e-5
    SEGFORMER_VARIANT = "b1"

    # ── Pix2Pix ──────────────────────────────────────────────────────────────
    PIX2PIX_LR_G    = 2e-4
    PIX2PIX_LR_D    = 2e-4
    PIX2PIX_BETA1   = 0.5
    PIX2PIX_LAMBDA  = 100.0

    # ── Classes ──────────────────────────────────────────────────────────────
    NUM_CLASSES  = 4
    CLASS_NAMES  = ["background", "trunk", "branches", "support"]

    # ── Metrics ──────────────────────────────────────────────────────────────
    BOUNDARY_THETA  = 2

    # ── Checkpoints & Results ────────────────────────────────────────────────
    CHECKPOINT_DIR  = "checkpoints_hha"
    RESULTS_DIR     = "results_hha"

    # ── Reproducibility ──────────────────────────────────────────────────────
    RANDOM_SEED     = 42
    USE_DEPTH       = False
    USE_OCCLUSION   = False