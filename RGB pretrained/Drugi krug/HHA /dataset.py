"""
dataset.py
────────────────────────────────────────────────────────────────────────────
PyTorch Dataset za RGB+HHA segmentaciju stabala.
Ulaz: RGB (3ch) + HHA (3ch) = 6 kanala
Svi fajlovi su u istom folderu: _rgbd.npy, _hha.npy, _mask.png
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class AppleTreeDataset(Dataset):

    def __init__(self, img_dir, mask_dir, transform=None, img_size=(512, 512)):
        self.img_dir   = img_dir
        self.mask_dir  = mask_dir
        self.transform = transform
        self.img_size  = img_size

        img_stems  = {Path(f).stem.replace("_hha", "")
                      for f in os.listdir(img_dir) if f.endswith("_hha.npy")}
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
        H, W = self.img_size

        # ── Load RGB iz _rgbd.npy ─────────────────────────────────────────────
        rgbd = np.load(os.path.join(self.img_dir, f"{stem}_rgbd.npy"))
        rgb  = rgbd[:, :, :3].astype(np.float32)
        if rgb.shape[:2] != (H, W):
            rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)

        # ── Load HHA iz _hha.npy ──────────────────────────────────────────────
        hha = np.load(os.path.join(self.img_dir, f"{stem}_hha.npy"))
        hha = hha.astype(np.float32) / 255.0   # normalizacija u [0,1]
        if hha.shape[:2] != (H, W):
            hha = cv2.resize(hha, (W, H), interpolation=cv2.INTER_LINEAR)

        # ── Load mask ─────────────────────────────────────────────────────────
        mask = cv2.imread(
            os.path.join(self.mask_dir, f"{stem}_mask.png"), cv2.IMREAD_GRAYSCALE
        )
        if mask.shape[:2] != (H, W):
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        mask = np.clip(mask, 0, 3).astype(np.uint8)

        # ── Augmentacija — RGB + HHA + Mask ZAJEDNO ───────────────────────────
        if self.transform is not None:
            rgb_uint8 = (rgb * 255).clip(0, 255).astype(np.uint8)
            hha_uint8 = (hha * 255).clip(0, 255).astype(np.uint8)

            result = self.transform(
                image=rgb_uint8,
                mask=mask,
                depth=hha_uint8,
            )

            rgb  = result["image"].astype(np.float32) / 255.0
            mask = result["mask"].astype(np.int64)
            hha  = result["depth"].astype(np.float32) / 255.0

        # ── Normalizacija RGB (ImageNet) ───────────────────────────────────────
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD

        # ── Tenzori ───────────────────────────────────────────────────────────
        rgb_t  = torch.from_numpy(rgb).permute(2, 0, 1).float()   # [3, H, W]
        hha_t  = torch.from_numpy(hha).permute(2, 0, 1).float()   # [3, H, W]
        mask_t = torch.from_numpy(mask.astype(np.int64)).long()    # [H, W]

        image_t = torch.cat([rgb_t, hha_t], dim=0)   # [6, H, W]

        return {"image": image_t, "mask": mask_t, "stem": stem}