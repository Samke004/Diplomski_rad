"""
trainer.py
────────────────────────────────────────────────────────────────────────────
Training engines with detailed CSV logging.

Logs saved to: logs/<model_name>_log.csv
  - One row per epoch
  - All metrics: loss, mIoU, per-class IoU, BF1, recall, LR, time

Also saves: logs/<model_name>_info.txt
  - Model summary, dataset info, hyperparameters
"""

import os
import csv
import time
import datetime
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from losses import CrossEntropyDiceLoss, GANLoss, Pix2PixGeneratorLoss
from metrics import compute_all_metrics, aggregate_metrics, CLASS_NAMES


# ─────────────────────────────────────────────────────────────────────────────
# CSV Logger
# ─────────────────────────────────────────────────────────────────────────────

class CSVLogger:
    """
    Logs training metrics to a CSV file, one row per epoch.
    All metrics are logged — useful for plotting training curves.
    """

    # All columns that will be written to CSV
    COLUMNS = [
        "epoch", "timestamp", "elapsed_s",
        "train_loss", "val_loss",
        "lr",
        "mean_iou", "mean_boundary_f1", "overall_accuracy", "branch_recall",
        # Per-class IoU
        "iou_background", "iou_trunk", "iou_branches", "iou_support",
        # Per-class F1
        "f1_background", "f1_trunk", "f1_branches", "f1_support",
        # Per-class Precision
        "precision_background", "precision_trunk", "precision_branches", "precision_support",
        # Per-class Recall
        "recall_background", "recall_trunk", "recall_branches", "recall_support",
        # Per-class Boundary F1
        "boundary_f1_background", "boundary_f1_trunk", "boundary_f1_branches", "boundary_f1_support",
        # Best tracking
        "is_best",
    ]

    def __init__(self, model_name: str, log_dir: str = "logs"):
        self.model_name = model_name
        self.log_dir    = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.csv_path  = os.path.join(log_dir, f"{model_name}_log.csv")
        self.info_path = os.path.join(log_dir, f"{model_name}_info.txt")

        # Open CSV and write header
        self.csv_file = open(self.csv_path, "w", newline="")
        self.writer   = csv.DictWriter(self.csv_file, fieldnames=self.COLUMNS)
        self.writer.writeheader()
        self.csv_file.flush()

        self.start_time = time.time()
        print(f"  CSV log → {self.csv_path}")

    def write_info(self, info: dict):
        """Write model/training info to a text file."""
        with open(self.info_path, "w") as f:
            f.write(f"{'='*60}\n")
            f.write(f"  TRAINING INFO — {self.model_name}\n")
            f.write(f"  Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*60}\n\n")
            for k, v in info.items():
                f.write(f"  {k:<25}: {v}\n")
            f.write(f"\n{'='*60}\n")
        print(f"  Info    → {self.info_path}")

    def log_epoch(
        self,
        epoch:      int,
        train_loss: float,
        val_loss:   float,
        val_metrics: dict,
        lr:         float,
        elapsed_s:  float,
        is_best:    bool,
    ):
        row = {
            "epoch":      epoch,
            "timestamp":  datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_s":  round(elapsed_s, 1),
            "train_loss": round(train_loss, 6),
            "val_loss":   round(val_loss,   6),
            "lr":         f"{lr:.2e}",
            "is_best":    int(is_best),
        }

        # Fill all metric columns
        for col in self.COLUMNS:
            if col not in row:
                key = col  # metric key matches column name
                row[col] = round(float(val_metrics.get(key, float("nan"))), 6)

        self.writer.writerow(row)
        self.csv_file.flush()

    def close(self):
        self.csv_file.close()


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"  Checkpoint saved → {path}")


def _count_params(model: nn.Module) -> str:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"{trainable/1e6:.2f}M trainable / {total/1e6:.2f}M total"


# ─────────────────────────────────────────────────────────────────────────────
# Segmentation Trainer (U-Net / DeepLabv3 / DeepLabv3+ / SegFormer)
# ─────────────────────────────────────────────────────────────────────────────

class SegmentationTrainer:

    def __init__(
        self,
        model:        nn.Module,
        model_name:   str,
        cfg,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        lr:           float = 1e-4,
    ):
        self.model        = model.to(cfg.DEVICE)
        self.model_name   = model_name
        self.cfg          = cfg
        self.train_loader = train_loader
        self.val_loader   = val_loader

        self.criterion = CrossEntropyDiceLoss(num_classes=cfg.NUM_CLASSES)
        self.optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=cfg.NUM_EPOCHS, eta_min=1e-6
        )

        self.best_val_iou   = 0.0
        self.early_stop_ctr = 0
        self.logger         = CSVLogger(model_name, log_dir="logs_exp2")
        self.history        = {
            "train_loss": [], "val_loss": [], "val_iou": [], "val_bf1": []
        }

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        for batch in tqdm(self.train_loader, desc="  [Train]", leave=False):
            images = batch["image"].to(self.cfg.DEVICE)
            masks  = batch["mask"].to(self.cfg.DEVICE)
            self.optimizer.zero_grad()
            logits = self.model(images)
            loss   = self.criterion(logits, masks)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * images.size(0)
        return total_loss / len(self.train_loader.dataset)

    @torch.no_grad()
    def _val_epoch(self):
        self.model.eval()
        total_loss  = 0.0
        all_metrics = []

        for batch in tqdm(self.val_loader, desc="  [Val]  ", leave=False):
            images = batch["image"].to(self.cfg.DEVICE)
            masks  = batch["mask"].to(self.cfg.DEVICE)

            logits = self.model(images)
            loss   = self.criterion(logits, masks)
            total_loss += loss.item() * images.size(0)

            preds_np = logits.argmax(dim=1).cpu().numpy()
            masks_np = masks.cpu().numpy()

            for i in range(len(preds_np)):
                m = compute_all_metrics(
                    preds_np[i], masks_np[i],
                    boundary_theta=self.cfg.BOUNDARY_THETA,
                )
                all_metrics.append(m)

        val_loss = total_loss / len(self.val_loader.dataset)
        avg      = aggregate_metrics(all_metrics)
        return val_loss, avg

    def train(self):
        print(f"\n{'='*60}")
        print(f" Training: {self.model_name}")
        print(f" Device:   {self.cfg.DEVICE}")
        print(f" Epochs:   {self.cfg.NUM_EPOCHS}")
        print(f"{'='*60}")

        # Write info file
        self.logger.write_info({
            "model_name":    self.model_name,
            "parameters":    _count_params(self.model),
            "device":        self.cfg.DEVICE,
            "epochs":        self.cfg.NUM_EPOCHS,
            "early_stop":    self.cfg.EARLY_STOP_PATIENCE,
            "batch_size":    self.cfg.BATCH_SIZE,
            "image_size":    f"{self.cfg.IMAGE_HEIGHT}x{self.cfg.IMAGE_WIDTH}",
            "in_channels":   self.cfg.IN_CHANNELS,
            "num_classes":   self.cfg.NUM_CLASSES,
            "class_names":   str(self.cfg.CLASS_NAMES),
            "train_samples": len(self.train_loader.dataset),
            "val_samples":   len(self.val_loader.dataset),
            "optimizer":     "Adam",
            "scheduler":     "CosineAnnealingLR",
            "loss":          "CrossEntropy + Dice",
        })

        ckpt_path = os.path.join(
            self.cfg.CHECKPOINT_DIR, f"{self.model_name}_best.pth"
        )

        for epoch in range(1, self.cfg.NUM_EPOCHS + 1):
            t0 = time.time()
            train_loss         = self._train_epoch()
            val_loss, val_m    = self._val_epoch()
            self.scheduler.step()

            val_iou = val_m.get("mean_iou",         0.0)
            val_bf1 = val_m.get("mean_boundary_f1", 0.0)
            lr      = self.optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0
            is_best = val_iou > self.best_val_iou

            # History
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_iou"].append(val_iou)
            self.history["val_bf1"].append(val_bf1)

            # CSV log — detailed per-class metrics
            self.logger.log_epoch(
                epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                val_metrics=val_m, lr=lr, elapsed_s=elapsed, is_best=is_best,
            )

            # Console print
            print(
                f"Ep {epoch:03d}/{self.cfg.NUM_EPOCHS}  "
                f"train={train_loss:.4f}  val={val_loss:.4f}  "
                f"mIoU={val_iou:.4f}  BF1={val_bf1:.4f}  "
                f"lr={lr:.1e}  ({elapsed:.1f}s)"
                + (" ★" if is_best else "")
            )

            # Per-class IoU summary every 10 epochs
            if epoch % 10 == 0:
                print(f"         Per-class IoU: ", end="")
                for name in CLASS_NAMES:
                    v = val_m.get(f"iou_{name}", float("nan"))
                    print(f"{name}={v:.3f}  ", end="")
                print()

            # Checkpoint + early stopping
            if is_best:
                self.best_val_iou   = val_iou
                self.early_stop_ctr = 0
                _save_checkpoint(
                    {"epoch": epoch, "model_state": self.model.state_dict(),
                     "optimizer_state": self.optimizer.state_dict(),
                     "val_iou": val_iou, "val_metrics": val_m},
                    ckpt_path,
                )
            else:
                self.early_stop_ctr += 1
                if self.early_stop_ctr >= self.cfg.EARLY_STOP_PATIENCE:
                    print(f"\n  Early stopping at epoch {epoch} "
                          f"(no improvement for {self.cfg.EARLY_STOP_PATIENCE} epochs)")
                    break

        self.logger.close()
        print(f"\nBest val mIoU: {self.best_val_iou:.4f}")
        print(f"Log saved:     logs/{self.model_name}_log.csv")
        return self.history


# ─────────────────────────────────────────────────────────────────────────────
# Pix2Pix Trainer
# ─────────────────────────────────────────────────────────────────────────────

class Pix2PixTrainer:

    def __init__(
        self,
        generator:     nn.Module,
        cfg,
        train_loader:  DataLoader,
        val_loader:    DataLoader,
        discriminator: Optional[nn.Module] = None,
        use_gan:       bool = True,
    ):
        self.G          = generator.to(cfg.DEVICE)
        self.D          = discriminator.to(cfg.DEVICE) if discriminator else None
        self.cfg        = cfg
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.use_gan    = use_gan and (discriminator is not None)
        self.name       = "pix2pix_gan" if self.use_gan else "pix2pix_gen"

        self.gan_loss  = GANLoss("vanilla") if self.use_gan else None
        self.g_loss_fn = Pix2PixGeneratorLoss(
            num_classes=cfg.NUM_CLASSES,
            lambda_ce=cfg.PIX2PIX_LAMBDA,
            use_gan=self.use_gan,
        )

        self.opt_G = Adam(
            self.G.parameters(),
            lr=cfg.PIX2PIX_LR_G,
            betas=(cfg.PIX2PIX_BETA1, 0.999),
        )
        if self.use_gan:
            self.opt_D = Adam(
                self.D.parameters(),
                lr=cfg.PIX2PIX_LR_D,
                betas=(cfg.PIX2PIX_BETA1, 0.999),
            )

        self.best_val_iou   = 0.0
        self.early_stop_ctr = 0
        self.logger         = CSVLogger(self.name, log_dir="logs_exp2")
        self.history        = {
            "g_loss": [], "d_loss": [], "val_iou": [], "val_bf1": []
        }

    def _train_epoch(self):
        self.G.train()
        if self.D:
            self.D.train()
        total_g, total_d, n = 0.0, 0.0, 0

        for batch in tqdm(self.train_loader, desc="  [Train]", leave=False):
            images = batch["image"].to(self.cfg.DEVICE)
            masks  = batch["mask"].to(self.cfg.DEVICE)

            fake_logits = self.G(images)
            fake_probs  = torch.softmax(fake_logits, dim=1)
            real_onehot = torch.nn.functional.one_hot(masks, self.cfg.NUM_CLASSES) \
                              .permute(0, 3, 1, 2).float()

            if self.use_gan:
                self.opt_D.zero_grad()
                d_real  = self.gan_loss(self.D(images, real_onehot),      is_real=True)
                d_fake  = self.gan_loss(self.D(images, fake_probs.detach()), is_real=False)
                d_loss  = (d_real + d_fake) * 0.5
                d_loss.backward()
                self.opt_D.step()
                total_d += d_loss.item() * images.size(0)

            self.opt_G.zero_grad()
            fake_pred_g = self.D(images, fake_probs) if self.use_gan else None
            g_loss = self.g_loss_fn(fake_pred_g, fake_logits, masks)
            g_loss.backward()
            self.opt_G.step()
            total_g += g_loss.item() * images.size(0)
            n += images.size(0)

        return total_g / n, (total_d / n if self.use_gan else 0.0)

    @torch.no_grad()
    def _val_epoch(self):
        self.G.eval()
        all_metrics = []
        for batch in self.val_loader:
            images = batch["image"].to(self.cfg.DEVICE)
            masks  = batch["mask"].to(self.cfg.DEVICE)
            logits = self.G(images)
            preds_np = logits.argmax(dim=1).cpu().numpy()
            masks_np = masks.cpu().numpy()
            for i in range(len(preds_np)):
                m = compute_all_metrics(
                    preds_np[i], masks_np[i],
                    boundary_theta=self.cfg.BOUNDARY_THETA,
                )
                all_metrics.append(m)
        return aggregate_metrics(all_metrics)

    def train(self):
        print(f"\n{'='*60}")
        print(f" Training: {self.name}  (use_gan={self.use_gan})")
        print(f" Device:   {self.cfg.DEVICE}")
        print(f" Epochs:   {self.cfg.NUM_EPOCHS}")
        print(f"{'='*60}")

        self.logger.write_info({
            "model_name":    self.name,
            "generator":     _count_params(self.G),
            "discriminator": _count_params(self.D) if self.D else "none",
            "use_gan":       self.use_gan,
            "device":        self.cfg.DEVICE,
            "epochs":        self.cfg.NUM_EPOCHS,
            "batch_size":    self.cfg.BATCH_SIZE,
            "lambda":        self.cfg.PIX2PIX_LAMBDA,
            "train_samples": len(self.train_loader.dataset),
            "val_samples":   len(self.val_loader.dataset),
        })

        ckpt_path = os.path.join(
            self.cfg.CHECKPOINT_DIR, f"{self.name}_best.pth"
        )

        for epoch in range(1, self.cfg.NUM_EPOCHS + 1):
            t0 = time.time()
            g_loss, d_loss = self._train_epoch()
            val_m          = self._val_epoch()

            val_iou = val_m.get("mean_iou",         0.0)
            val_bf1 = val_m.get("mean_boundary_f1", 0.0)
            elapsed = time.time() - t0
            is_best = val_iou > self.best_val_iou

            self.history["g_loss"].append(g_loss)
            self.history["d_loss"].append(d_loss)
            self.history["val_iou"].append(val_iou)
            self.history["val_bf1"].append(val_bf1)

            # Use g_loss as train_loss for logging consistency
            self.logger.log_epoch(
                epoch=epoch, train_loss=g_loss, val_loss=d_loss,
                val_metrics=val_m, lr=self.cfg.PIX2PIX_LR_G,
                elapsed_s=elapsed, is_best=is_best,
            )

            print(
                f"Ep {epoch:03d}/{self.cfg.NUM_EPOCHS}  "
                f"G={g_loss:.4f}  D={d_loss:.4f}  "
                f"mIoU={val_iou:.4f}  BF1={val_bf1:.4f}  "
                f"({elapsed:.1f}s)"
                + (" ★" if is_best else "")
            )

            if is_best:
                self.best_val_iou   = val_iou
                self.early_stop_ctr = 0
                _save_checkpoint(
                    {"epoch": epoch, "G_state": self.G.state_dict(),
                     "val_iou": val_iou, "val_metrics": val_m},
                    ckpt_path,
                )
            else:
                self.early_stop_ctr += 1
                if self.early_stop_ctr >= self.cfg.EARLY_STOP_PATIENCE:
                    print(f"\n  Early stopping at epoch {epoch} "
                          f"(no improvement for {self.cfg.EARLY_STOP_PATIENCE} epochs)")
                    break

        self.logger.close()
        print(f"\nBest val mIoU: {self.best_val_iou:.4f}")
        print(f"Log saved:     logs/{self.name}_log.csv")
        return self.history