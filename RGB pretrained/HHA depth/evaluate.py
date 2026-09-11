"""
evaluate.py — Multi-class evaluation on test set
────────────────────────────────────────────────────────────────────────────
Outputs:
  1. Per-image metric CSV per model
  2. Aggregated comparison table (CSV + console)
  3. Worst-10 images by difficulty index
  4. Qualitative visualisation (colour-coded per class)
  5. Training curves
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config as cfg
from dataset import AppleTreeDataset
from augmentations import get_val_transforms
from models.esanet import build_esanet
from models import (build_unet, build_deeplabv3, build_deeplabv3plus, build_manet,
                    build_segformer, Pix2PixGenerator)
from metrics import compute_all_metrics, aggregate_metrics, CLASS_NAMES

# Colour map for visualisation: bg=black, trunk=red, branches=blue, support=green
CLASS_COLORS = np.array([
    [0,   0,   0  ],   # 0 background
    [255, 0,   0  ],   # 1 trunk
    [0,   0,   255],   # 2 branches
    [0,   255, 0  ],   # 3 support
], dtype=np.uint8)


# ─────────────────────────────────────────────────────────────────────────────

def _load_model(model, ckpt_path, device, key="model_state"):
    if not os.path.isfile(ckpt_path):
        print(f"  [SKIP] Checkpoint not found: {ckpt_path}")
        return None
    state = torch.load(ckpt_path, map_location=device)
    actual_key = key if key in state else "G_state"
    model.load_state_dict(state[actual_key])
    model.eval()
    return model.to(device)


@torch.no_grad()
def _run_inference(model, test_loader, device):
    results = []
    for batch in tqdm(test_loader, desc="  Inference"):
        images = batch["image"].to(device)
        logits = model(images)                          # [B, 4, H, W]
        preds  = logits.argmax(dim=1).cpu().numpy()     # [B, H, W] int
        gts    = batch["mask"].cpu().numpy()            # [B, H, W] int
        stems  = batch["stem"]

        occs = batch["occlusion"].squeeze(1).numpy() if "occlusion" in batch else [None]*len(stems)
        deps = batch["depth"].squeeze(1).numpy()     if "depth"     in batch else [None]*len(stems)

        for i, stem in enumerate(stems):
            results.append({
                "stem": stem, "pred": preds[i], "gt": gts[i],
                "occlusion": occs[i], "depth": deps[i],
            })
    return results


def _compute_per_image_metrics(inference_results):
    rows = []
    for r in inference_results:
        m = compute_all_metrics(
            pred=r["pred"], gt=r["gt"],
            occlusion=r["occlusion"], depth=r["depth"],
            boundary_theta=cfg.BOUNDARY_THETA,
        )
        m["stem"] = r["stem"]
        rows.append(m)
    return pd.DataFrame(rows)


def _visualise(model_name, inference_results, out_dir, n_show=8):
    os.makedirs(out_dir, exist_ok=True)
    samples = inference_results[:n_show]
    n = len(samples)

    fig, axes = plt.subplots(n, 3, figsize=(12, n * 4))
    if n == 1:
        axes = axes[np.newaxis, :]

    for j, t in enumerate(["Ground Truth", "Prediction", "Overlay (TP/FP/FN branches)"]):
        axes[0, j].set_title(t, fontsize=11, fontweight="bold")

    for i, r in enumerate(samples):
        pred = r["pred"]
        gt   = r["gt"]

        # Colour-code by class
        gt_rgb   = CLASS_COLORS[gt]
        pred_rgb = CLASS_COLORS[pred]

        # Overlay: correct=green, wrong=red (only for branch class)
        overlay = np.zeros((*gt.shape, 3), dtype=np.uint8)
        tp = (pred == 2) & (gt == 2)
        fp = (pred == 2) & (gt != 2)
        fn = (pred != 2) & (gt == 2)
        overlay[tp] = [0,   255, 0  ]
        overlay[fp] = [255, 0,   0  ]
        overlay[fn] = [0,   0,   255]

        axes[i, 0].imshow(gt_rgb);   axes[i, 0].axis("off")
        axes[i, 1].imshow(pred_rgb); axes[i, 1].axis("off")
        axes[i, 2].imshow(overlay);  axes[i, 2].axis("off")
        axes[i, 0].set_ylabel(r["stem"][:20], fontsize=7)

    fig.suptitle(f"{model_name} — Multi-class Results\n"
                 "Colors: black=bg  red=trunk  blue=branch  green=support",
                 fontsize=12)
    plt.tight_layout()
    path = os.path.join(out_dir, f"{model_name}_qual.png")
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"  Saved: {path}")


def plot_training_curves(histories: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name, h in histories.items():
        if "val_iou" in h:
            axes[0].plot(h["val_iou"], label=name)
        if "val_bf1" in h:
            axes[1].plot(h["val_bf1"], label=name)
    for ax, title, ylabel in zip(axes,
                                  ["Validation Mean IoU (fg classes)",
                                   "Validation Mean Boundary F1"],
                                  ["mIoU", "BF1"]):
        ax.set_title(title); ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel); ax.legend(); ax.grid(True)
    plt.tight_layout()
    path = os.path.join(out_dir, "training_curves.png")
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"Saved: {path}")


def worst_n_analysis(df, model_name, out_dir, n=10):
    rows = []
    for col, label in [("occlusion_difficulty_index", "Occlusion DI"),
                        ("depth_difficulty_index",     "Depth DI")]:
        if col not in df.columns:
            continue
        sub = df.dropna(subset=[col]).nlargest(n, col)
        for _, row in sub.iterrows():
            rows.append({
                "model": model_name, "difficulty_by": label,
                "stem":  row["stem"], col: row.get(col, float("nan")),
                "branch_recall":         row.get("branch_recall", float("nan")),
                "occluded_branch_recall":row.get("occluded_branch_recall", float("nan")),
            })
    if rows:
        path = os.path.join(out_dir, f"{model_name}_worst{n}.csv")
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"  Worst-{n} saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────

def evaluate_all(histories: dict = None):
    device = cfg.DEVICE
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    test_dataset = AppleTreeDataset(
        img_dir   = cfg.TEST_IMG_DIR,
        mask_dir  = cfg.TEST_MASK_DIR,
        
        
        transform = get_val_transforms(cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH),
        img_size  = (cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH),
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             num_workers=cfg.NUM_WORKERS, pin_memory=False)
    print(f"\nTest set: {len(test_dataset)} images")

    model_configs = [
        ("unet_resnet34",
         build_unet("resnet34", weights=None,
                    in_channels=cfg.IN_CHANNELS, num_classes=cfg.NUM_CLASSES),
         os.path.join(cfg.CHECKPOINT_DIR, "unet_resnet34_best.pth"), "model_state"),

        ("unet_efficientnet_b4_bs2",
         build_unet("efficientnet-b4", weights=None,
                    in_channels=cfg.IN_CHANNELS, num_classes=cfg.NUM_CLASSES),
         os.path.join(cfg.CHECKPOINT_DIR, "unet_efficientnet_b4_best.pth"), "model_state"),

        ("unet_efficientnet_b4_bs4",
         build_unet("efficientnet-b4", weights=None,
                    in_channels=cfg.IN_CHANNELS, num_classes=cfg.NUM_CLASSES),
         os.path.join(cfg.CHECKPOINT_DIR, "unet_efficientnet_b4_best_4.pth"), "model_state"),

        ("deeplabv3_resnet50",
         build_deeplabv3(pretrained=False,
                         in_channels=cfg.IN_CHANNELS, num_classes=cfg.NUM_CLASSES),
         os.path.join(cfg.CHECKPOINT_DIR, "deeplabv3_resnet50_best.pth"), "model_state"),

        ("segformer_b1",
         build_segformer(variant=cfg.SEGFORMER_VARIANT, weights=None,
                         in_channels=cfg.IN_CHANNELS, num_classes=cfg.NUM_CLASSES),
         os.path.join(cfg.CHECKPOINT_DIR, "segformer_b1_best.pth"), "model_state"),

        ("manet_resnext50",
         build_manet(pretrained=False,
                     in_channels=cfg.IN_CHANNELS, num_classes=cfg.NUM_CLASSES),
         os.path.join(cfg.CHECKPOINT_DIR, "manet_resnext50_best.pth"), "model_state"),

        ("deeplabv3plus_mobilenetv2",
         build_deeplabv3plus(encoder="mobilenet_v2", weights=None,
                             in_channels=cfg.IN_CHANNELS, num_classes=cfg.NUM_CLASSES),
         os.path.join(cfg.CHECKPOINT_DIR, "deeplabv3plus_mobilenetv2_best.pth"), "model_state"),

        ("segformer_b1",
         build_segformer(variant=cfg.SEGFORMER_VARIANT, weights=None,
                         in_channels=cfg.IN_CHANNELS, num_classes=cfg.NUM_CLASSES),
         os.path.join(cfg.CHECKPOINT_DIR, "segformer_b1_best.pth"), "model_state"),

        ("esanet_resnet34",
         build_esanet(pretrained=False, in_channels=cfg.IN_CHANNELS, num_classes=cfg.NUM_CLASSES),
         os.path.join(cfg.CHECKPOINT_DIR, "esanet_resnet34_best.pth"), "model_state"),

        ("pix2pix_gan",
         Pix2PixGenerator(in_channels=cfg.IN_CHANNELS, out_channels=cfg.NUM_CLASSES),
         os.path.join(cfg.CHECKPOINT_DIR, "pix2pix_gan_best.pth"), "G_state"),

        ("pix2pix_gen",
         Pix2PixGenerator(in_channels=cfg.IN_CHANNELS, out_channels=cfg.NUM_CLASSES),
         os.path.join(cfg.CHECKPOINT_DIR, "pix2pix_gen_best.pth"), "G_state"),
    ]

    aggregated_all = {}

    # Key metrics for comparison table
    KEY_METRICS = (
        ["overall_accuracy", "pixel_accuracy", "pixel_accuracy_fg",
         "frequency_weighted_iou", "frequency_weighted_iou_fg",
         "mean_iou", "mean_boundary_f1", "branch_recall"] +
        [f"iou_{c}"       for c in CLASS_NAMES] +
        [f"recall_{c}"    for c in CLASS_NAMES] +
        [f"boundary_f1_{c}" for c in CLASS_NAMES]
    )

    for model_name, model, ckpt_path, state_key in model_configs:
        print(f"\n── Evaluating: {model_name} ──")
        model = _load_model(model, ckpt_path, device, key=state_key)
        if model is None:
            continue

        inf_results = _run_inference(model, test_loader, device)
        df = _compute_per_image_metrics(inf_results)
        csv_path = os.path.join(cfg.RESULTS_DIR, f"{model_name}_per_image.csv")
        df.to_csv(csv_path, index=False)
        print(f"  Per-image CSV: {csv_path}")

        agg = aggregate_metrics(df.drop(columns=["stem"]).to_dict("records"))
        aggregated_all[model_name] = agg

        worst_n_analysis(df, model_name, cfg.RESULTS_DIR)
        _visualise(model_name, inf_results, cfg.RESULTS_DIR)

    # ── Comparison table ────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("  COMPARISON TABLE")
    print("="*80)

    compare_rows = []
    for mname, agg in aggregated_all.items():
        row = {"model": mname}
        for k in KEY_METRICS:
            row[k] = round(agg.get(k, float("nan")), 4)
        compare_rows.append(row)

    df_compare = pd.DataFrame(compare_rows).set_index("model")
    print(df_compare.to_string())
    path = os.path.join(cfg.RESULTS_DIR, "comparison_table.csv")
    df_compare.to_csv(path)
    print(f"\nComparison table: {path}")

    if histories:
        plot_training_curves(histories, cfg.RESULTS_DIR)

    return df_compare


if __name__ == "__main__":
    evaluate_all()
