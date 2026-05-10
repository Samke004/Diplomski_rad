"""
prepare_data.py
────────────────────────────────────────────────────────────────────────────
Splits the converted RGBD dataset into train / val / test.

Strategy:
  - Split is done PER TREE (not per PLY file) to avoid data leakage
  - Val and Test contain ONLY manually annotated data (highest quality)
  - Train contains all model-annotated (reviewed) + remaining manual data
  - All files from one tree go to the same split

Dataset groups:
  GROUP A — Samuel's manual annotations:    Drvo0000-Drvo0015   (108 files)
  GROUP B — Valentin's manual annotations:  tree_1_V_0179-0183  (91 files)
  GROUP C — Model annotations (reviewed):   tree_nove_od_valeta (200 files)

Final split:
  TRAIN (~80%): All of GROUP C + 70% of GROUP A + 70% of GROUP B
  VAL   (~10%): 15% of GROUP A + 15% of GROUP B
  TEST  (~10%): 15% of GROUP A + 15% of GROUP B

Usage:
    python prepare_data.py --anotirane /home/samuel/Desktop/Anotirane \
                           --out_dir   /home/samuel/Desktop/data
"""

import os
import argparse
import random
import shutil
from pathlib import Path


# ── Tree folder definitions ───────────────────────────────────────────────────

def get_tree_groups(anotirane_root):
    """
    Returns three lists of tree folder paths:
      group_a — Samuel's manual annotations  (Drvo00xx)
      group_b — Valentin's manual annotations (tree_1_V_0179-0183)
      group_c — Model annotations reviewed    (tree_nove_od_valeta/*)
    """
    group_a, group_b, group_c = [], [], []

    for d in sorted(os.listdir(anotirane_root)):
        full = os.path.join(anotirane_root, d)
        if not os.path.isdir(full):
            continue

        # ── Group C: tree_nove_od_valeta contains subfolders ─────────────────
        if d == "tree_nove_od_valeta":
            for sub in sorted(os.listdir(full)):
                sub_full  = os.path.join(full, sub)
                if not os.path.isdir(sub_full):
                    continue
                sub_depth = os.path.join(sub_full, "Depth")
                if not os.path.isdir(sub_depth):
                    continue
                npy = [f for f in os.listdir(sub_depth)
                       if f.endswith("_rgbd.npy")]
                if npy:
                    group_c.append(sub_full)
            continue

        # ── Groups A and B: Depth/ is directly inside the tree folder ────────
        depth_dir = os.path.join(full, "Depth")
        if not os.path.isdir(depth_dir):
            continue
        npy_files = [f for f in os.listdir(depth_dir)
                     if f.endswith("_rgbd.npy")]
        if not npy_files:
            continue

        if d.startswith("Drvo"):
            group_a.append(full)
        elif d.startswith("tree_1_V_"):
            group_b.append(full)

    return group_a, group_b, group_c


# ── Collect file pairs from a tree folder ─────────────────────────────────────

def collect_pairs(tree_folder):
    """
    Returns list of (rgbd_path, mask_path) pairs from a tree's Depth/ folder.
    """
    depth_dir = os.path.join(tree_folder, "Depth")
    pairs = []
    for f in sorted(os.listdir(depth_dir)):
        if not f.endswith("_rgbd.npy"):
            continue
        stem     = f.replace("_rgbd.npy", "")
        rgbd     = os.path.join(depth_dir, f)
        mask     = os.path.join(depth_dir, f"{stem}_mask.png")
        if os.path.isfile(mask):
            pairs.append((rgbd, mask))
        else:
            print(f"  [WARN] No mask for {f}")
    return pairs


# ── Split trees into train/val/test ──────────────────────────────────────────

def split_trees(trees, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Splits a list of tree folders into train/val/test.
    Split is done at tree level to avoid data leakage.
    """
    random.seed(seed)
    trees = trees.copy()
    random.shuffle(trees)

    n       = len(trees)
    n_val   = max(1, round(n * val_ratio))
    n_test  = max(1, round(n * test_ratio))
    n_train = n - n_val - n_test

    return (
        trees[:n_train],
        trees[n_train:n_train + n_val],
        trees[n_train + n_val:]
    )


# ── Copy files to output split folders ───────────────────────────────────────

def copy_pairs(pairs, split_dir, prefix=""):
    """
    Copies (rgbd, mask) pairs to split_dir/images/ and split_dir/masks/.
    Optionally adds a prefix to avoid filename collisions.
    """
    img_dir  = os.path.join(split_dir, "images")
    mask_dir = os.path.join(split_dir, "masks")
    os.makedirs(img_dir,  exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    for rgbd_src, mask_src in pairs:
        stem     = Path(rgbd_src).stem.replace("_rgbd", "")
        new_name = f"{prefix}{stem}" if prefix else stem

        shutil.copy2(rgbd_src, os.path.join(img_dir,  f"{new_name}_rgbd.npy"))
        shutil.copy2(mask_src, os.path.join(mask_dir, f"{new_name}_mask.png"))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anotirane", default="/home/samuel/Desktop/Anotirane",
                        help="Root folder with all annotated tree folders")
    parser.add_argument("--out_dir",   default="/home/samuel/Desktop/data",
                        help="Output folder for train/val/test splits")
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"\nScanning: {args.anotirane}")
    group_a, group_b, group_c = get_tree_groups(args.anotirane)

    print(f"\nFound tree groups:")
    print(f"  Group A (Samuel manual):    {len(group_a)} trees")
    print(f"  Group B (Valentin manual):  {len(group_b)} trees")
    print(f"  Group C (Model reviewed):   {len(group_c)} trees")

    # ── Split manual groups (A and B) into train/val/test ────────────────────
    a_train, a_val, a_test = split_trees(group_a, val_ratio=0.15, test_ratio=0.15)
    b_train, b_val, b_test = split_trees(group_b, val_ratio=0.15, test_ratio=0.15)

    # All of Group C goes to train
    c_train = group_c

    print(f"\nTree-level split:")
    print(f"  TRAIN: {len(a_train)} A-trees + {len(b_train)} B-trees + {len(c_train)} C-trees")
    print(f"  VAL:   {len(a_val)} A-trees   + {len(b_val)} B-trees  (manual only)")
    print(f"  TEST:  {len(a_test)} A-trees  + {len(b_test)} B-trees  (manual only)")

    # ── Collect file pairs ────────────────────────────────────────────────────
    def collect_all(tree_list, prefix=""):
        pairs = []
        for i, tree in enumerate(tree_list):
            tree_prefix = f"{prefix}{i:03d}_" if prefix else ""
            tree_pairs  = collect_pairs(tree)
            # Add tree index to avoid filename collisions
            pairs.extend([(r, m, tree_prefix) for r, m in tree_pairs])
        return pairs

    train_data = (
        collect_all(a_train, "A") +
        collect_all(b_train, "B") +
        collect_all(c_train, "C")
    )
    val_data  = collect_all(a_val,  "A") + collect_all(b_val,  "B")
    test_data = collect_all(a_test, "A") + collect_all(b_test, "B")

    # Count actual files
    n_train = len(train_data)
    n_val   = len(val_data)
    n_test  = len(test_data)
    total   = n_train + n_val + n_test

    print(f"\nFile-level split:")
    print(f"  TRAIN: {n_train} files ({100*n_train/total:.1f}%)")
    print(f"  VAL:   {n_val}   files ({100*n_val/total:.1f}%)")
    print(f"  TEST:  {n_test}  files ({100*n_test/total:.1f}%)")
    print(f"  TOTAL: {total}   files")

    # ── Copy to output folders ────────────────────────────────────────────────
    print(f"\nCopying to: {args.out_dir}")

    for split, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        split_dir = os.path.join(args.out_dir, split)
        img_dir   = os.path.join(split_dir, "images")
        mask_dir  = os.path.join(split_dir, "masks")
        os.makedirs(img_dir,  exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        for rgbd_src, mask_src, prefix in data:
            stem     = Path(rgbd_src).stem.replace("_rgbd", "")
            new_name = f"{prefix}{stem}"

            shutil.copy2(rgbd_src, os.path.join(img_dir,  f"{new_name}_rgbd.npy"))
            shutil.copy2(mask_src, os.path.join(mask_dir, f"{new_name}_mask.png"))

        print(f"  {split:5s}: {len(data)} files → {split_dir}")

    # ── Summary report ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  DATASET SUMMARY FOR MENTOR")
    print(f"{'='*60}")
    print(f"  Total annotated images:  {total}")
    print(f"  ├─ Manual (Samuel):      108 images from 16 trees")
    print(f"  ├─ Manual (Valentin):    91 images from 5 trees")
    print(f"  └─ Model+reviewed:       200 images")
    print(f"")
    print(f"  Split strategy: per-tree (no data leakage)")
    print(f"  ├─ TRAIN: {n_train} images ({100*n_train/total:.1f}%) — all groups")
    print(f"  ├─ VAL:   {n_val} images ({100*n_val/total:.1f}%)  — manual only")
    print(f"  └─ TEST:  {n_test} images ({100*n_test/total:.1f}%)  — manual only")
    print(f"")
    print(f"  Input:  4-channel RGBD [640×480] (RGB + depth)")
    print(f"  Target: label mask [640×480] (0=bg, 1=trunk, 2=branches, 3=support)")
    print(f"  Seed:   {args.seed} (reproducible)")
    print(f"{'='*60}")
    print(f"\nDone! Data ready in: {args.out_dir}")


if __name__ == "__main__":
    main()