"""
Dataset Mask Statistics
=======================
Prolazi kroz sve splitove (train/val/test), čita PNG maske,
broji piksele po klasi i ispisuje statistiku.

Uporaba:
    python dataset_stats.py --data_root /path/to/data
    python dataset_stats.py  # koristi trenutni direktorij
"""

import os
import argparse
from collections import defaultdict

import numpy as np
from PIL import Image


# ── Helpers ──────────────────────────────────────────────────────────────────

def is_backup(filename: str) -> bool:
    return "BACKUP" in filename.upper()


def load_mask(path: str) -> np.ndarray:
    """Učitaj masku kao 2D numpy array (grayscale vrijednosti = klase)."""
    img = Image.open(path)
    arr = np.array(img)
    # Ako je RGB/RGBA, uzmi samo prvi kanal (klase su iste u svim kanalima)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr


def collect_mask_paths(split_dir: str):
    """Vrati listu (path, stem) maski unutar split_dir/masks/, preskači BACKUP."""
    masks_dir = os.path.join(split_dir, "masks")
    if not os.path.isdir(masks_dir):
        return []
    paths = []
    for fname in sorted(os.listdir(masks_dir)):
        if not fname.lower().endswith(".png"):
            continue
        if is_backup(fname):
            continue
        paths.append(os.path.join(masks_dir, fname))
    return paths


def format_row(label, count, total, bar_width=30):
    pct = 100.0 * count / total if total > 0 else 0.0
    filled = int(round(bar_width * pct / 100))
    bar = "█" * filled + "░" * (bar_width - filled)
    return f"  {label:<20} {count:>12,}  ({pct:6.2f}%)  [{bar}]"


# ── Main ─────────────────────────────────────────────────────────────────────

def analyse(data_root: str):
    splits = ["train", "val", "test"]

    # global accumulators
    global_class_counts  = defaultdict(int)
    global_total_pixels  = 0
    global_total_images  = 0
    global_skipped       = 0

    print("=" * 72)
    print(f"  Dataset root: {os.path.abspath(data_root)}")
    print("=" * 72)

    for split in splits:
        split_dir = os.path.join(data_root, split)
        if not os.path.isdir(split_dir):
            print(f"\n[SKIP] Split '{split}' not found.\n")
            continue

        mask_paths = collect_mask_paths(split_dir)
        if not mask_paths:
            print(f"\n[SKIP] No masks found in '{split}'.\n")
            continue

        split_class_counts = defaultdict(int)
        split_total_pixels = 0

        # Count BACKUP files separately (just for info)
        all_files = [f for f in os.listdir(os.path.join(split_dir, "masks"))
                     if f.lower().endswith(".png")]
        backup_count = sum(1 for f in all_files if is_backup(f))

        print(f"\n{'─'*72}")
        print(f"  SPLIT: {split.upper()}")
        print(f"{'─'*72}")
        print(f"  Maske pronađene : {len(mask_paths)}")
        if backup_count:
            print(f"  BACKUP (preskočeno): {backup_count}")

        for path in mask_paths:
            try:
                mask = load_mask(path)
            except Exception as e:
                print(f"  [WARN] Ne mogu čitati {os.path.basename(path)}: {e}")
                global_skipped += 1
                continue

            unique, counts = np.unique(mask, return_counts=True)
            for cls, cnt in zip(unique, counts):
                split_class_counts[int(cls)] += int(cnt)
            split_total_pixels += mask.size

        # ── Per-split report ──────────────────────────────────────────────
        all_classes = sorted(split_class_counts.keys())
        print(f"\n  Ukupno piksela  : {split_total_pixels:,}")
        print(f"  Klase pronađene : {all_classes}\n")
        print(f"  {'Klasa':<20} {'Broj piksela':>12}   {'%':>6}   Vizualizacija")
        print(f"  {'─'*20}  {'─'*12}   {'─'*6}   {'─'*30}")
        for cls in all_classes:
            cnt = split_class_counts[cls]
            print(format_row(f"Klasa {cls}", cnt, split_total_pixels))

        # accumulate globals
        for cls, cnt in split_class_counts.items():
            global_class_counts[cls] += cnt
        global_total_pixels += split_total_pixels
        global_total_images += len(mask_paths)

    # ── Global report ─────────────────────────────────────────────────────
    print(f"\n{'═'*72}")
    print(f"  UKUPNA STATISTIKA (svi splitovi)")
    print(f"{'═'*72}")
    print(f"  Ukupno slika    : {global_total_images:,}")
    print(f"  Ukupno piksela  : {global_total_pixels:,}")
    if global_skipped:
        print(f"  Preskočeno (greška): {global_skipped}")

    all_classes = sorted(global_class_counts.keys())
    print(f"  Klase pronađene : {all_classes}\n")
    print(f"  {'Klasa':<20} {'Broj piksela':>12}   {'%':>6}   Vizualizacija")
    print(f"  {'─'*20}  {'─'*12}   {'─'*6}   {'─'*30}")
    for cls in all_classes:
        cnt = global_class_counts[cls]
        print(format_row(f"Klasa {cls}", cnt, global_total_pixels))

    print(f"\n{'═'*72}\n")

    # ── Class imbalance warning ────────────────────────────────────────────
    if global_total_pixels > 0 and len(all_classes) > 1:
        max_pct = max(global_class_counts[c] / global_total_pixels for c in all_classes) * 100
        min_pct = min(global_class_counts[c] / global_total_pixels for c in all_classes) * 100
        ratio = max_pct / min_pct if min_pct > 0 else float("inf")
        if ratio > 10:
            print(f"  ⚠️  Jako neuravnotežen dataset! Omjer najveće/najmanje klase: {ratio:.1f}x")
        elif ratio > 3:
            print(f"  ⚠️  Umjerena neravnoteža klasa. Omjer: {ratio:.1f}x")
        else:
            print(f"  ✅  Klase su relativno uravnotežene. Omjer: {ratio:.1f}x")
        print()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Statistika maski dataseta")
    parser.add_argument(
        "--data_root",
        type=str,
        default=".",
        help="Putanja do root direktorija dataseta (sadrži train/val/test)",
    )
    args = parser.parse_args()
    analyse(args.data_root)