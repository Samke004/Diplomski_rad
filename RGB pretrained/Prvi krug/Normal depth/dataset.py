"""
dataset.py
────────────────────────────────────────────────────────────────────────────
PyTorch Dataset za RGBD segmentaciju stabala.
Geometrijske augmentacije se sada primjenjuju identično
na RGB i Depth kanal istovremeno — sprječava neusklađenost kanala.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Clip dubine — 4.0m daje bolji kontrast za stabla na ~1-2m udaljenosti
DEPTH_CLIP_MAX = 4.0


class AppleTreeDataset(Dataset):

    def __init__(self, img_dir, mask_dir, transform=None, img_size=(512, 512)):
        self.img_dir   = img_dir
        self.mask_dir  = mask_dir
        self.transform = transform
        self.img_size  = img_size

        img_stems  = {Path(f).stem.replace("_rgbd", "")
                      for f in os.listdir(img_dir) if f.endswith("_rgbd.npy")}
        mask_stems = {Path(f).stem.replace("_mask", "")
                      for f in os.listdir(mask_dir) if f.endswith("_mask.png")}

        self.stems = sorted(img_stems & mask_stems)

        if len(self.stems) == 0:
            raise FileNotFoundError(
                f"No matching pairs found.\n  images: {img_dir}\n  masks: {mask_dir}"
            )
        print(f"Dataset loaded: {len(self.stems)} samples from {img_dir}")

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        stem = self.stems[idx]

        # ── Load RGBD ────────────────────────────────────────────────────────
        rgbd = np.load(os.path.join(self.img_dir, f"{stem}_rgbd.npy"))

        H, W = self.img_size
        if rgbd.shape[:2] != (H, W):
            rgb   = cv2.resize(rgbd[:, :, :3], (W, H), interpolation=cv2.INTER_LINEAR)
            depth = cv2.resize(rgbd[:, :, 3],  (W, H), interpolation=cv2.INTER_LINEAR)
            rgbd  = np.dstack([rgb, depth])

        rgb   = rgbd[:, :, :3].astype(np.float32)  # [H, W, 3] in [0,1]
        depth = rgbd[:, :, 3].astype(np.float32)   # [H, W]

        # ── Load mask ────────────────────────────────────────────────────────
        mask = cv2.imread(
            os.path.join(self.mask_dir, f"{stem}_mask.png"), cv2.IMREAD_GRAYSCALE
        )
        if mask.shape[:2] != (H, W):
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        mask = np.clip(mask, 0, 3).astype(np.uint8)

        # ── Augmentacija — RGB + Depth + Mask ZAJEDNO ────────────────────────
        if self.transform is not None:
            # Pretvori RGB u uint8 za albumentations
            rgb_uint8 = (rgb * 255).clip(0, 255).astype(np.uint8)

            # Normaliziraj depth u [0, 255] uint8 za albumentations
            # (samo za geometrijske transformacije — vraćamo nazad u float)
            depth_norm = np.clip(depth, 0, DEPTH_CLIP_MAX) / DEPTH_CLIP_MAX
            depth_uint8 = (depth_norm * 255).astype(np.uint8)

            # Albumentations prima: image (RGB), mask, i additional_target 'depth'
            result = self.transform(
                image=rgb_uint8,
                mask=mask,
                depth=depth_uint8,   # ← depth transformiran IDENTIČNO kao RGB
            )

            rgb       = result["image"].astype(np.float32) / 255.0
            mask      = result["mask"].astype(np.int64)
            depth_uint8 = result["depth"]

            # Vrati depth u [0, DEPTH_CLIP_MAX] float raspon
            depth = depth_uint8.astype(np.float32) / 255.0 * DEPTH_CLIP_MAX

        # ── Normalizacija RGB (ImageNet) ──────────────────────────────────────
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD

        # ── Normalizacija depth u [0, 1] ──────────────────────────────────────
        depth = np.clip(depth, 0, DEPTH_CLIP_MAX) / DEPTH_CLIP_MAX

        # ── Tenzori ───────────────────────────────────────────────────────────
        rgb_t   = torch.from_numpy(rgb).permute(2, 0, 1).float()     # [3, H, W]
        depth_t = torch.from_numpy(depth).unsqueeze(0).float()        # [1, H, W]
        mask_t  = torch.from_numpy(mask.astype(np.int64)).long()      # [H, W]

        image_t = torch.cat([rgb_t, depth_t], dim=0)   # [4, H, W]

        return {"image": image_t, "mask": mask_t, "stem": stem}
