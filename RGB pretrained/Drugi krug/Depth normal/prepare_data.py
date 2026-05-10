"""
prepare_data_v2.py
────────────────────────────────────────────────────────────────────────────
Split dataset into train/val/test.

VAL + TEST: samo ručno anotirani:
  - Drvo0000-Drvo0015        (Samuel ručno)
  - tree_1_V_0179-0183       (Valentin ručno)
  - tree_najnovije_od_valeta/tree_1_V_0115, 0125, 0135, 0145, 0155 (ručno korigirani)

TRAIN: sve ostalo:
  - tree_nove_od_valeta      (sva stabla, model+pregled)
  - tree_najnovije_od_valeta (sva osim 0115,0125,0135,0145,0155)
  - Dio Drvo + tree_1_V_0179-0183 koji ne uđu u val/test

Split po stablu (no data leakage).
Target: ~500 train / ~75 val / ~75 test

Usage:
    python prepare_data_v2.py \
        --anotirane /home/samuel/Desktop/Anotirane \
        --out_dir   /home/samuel/Desktop/Pokusaj_Chen_MultiDepth/data_v2
"""

import os
import argparse
import random
import shutil
from pathlib import Path


# ── Stabla koja MOGU ići u val/test ──────────────────────────────────────────

# Samuel ručno
MANUAL_SAMUEL = [f"Drvo{i:04d}" for i in range(16)]  # Drvo0000-Drvo0015

# Valentin ručno
MANUAL_VALENTIN = [f"tree_1_V_0{i}" for i in [179, 180, 181, 182, 183]]

# tree_najnovije_od_valeta — ručno korigirani
MANUAL_NAJNOVIJE = ["tree_1_V_0115", "tree_1_V_0125", "tree_1_V_0135",
                    "tree_1_V_0145", "tree_1_V_0155"]


def collect_pairs(depth_dir):
    """Vrati listu (rgbd_path, mask_path) iz Depth/ foldera."""
    pairs = []
    for f in sorted(os.listdir(depth_dir)):
        if not f.endswith("_rgbd.npy"):
            continue
        stem = f.replace("_rgbd.npy", "")
        rgbd = os.path.join(depth_dir, f)
        mask = os.path.join(depth_dir, f"{stem}_mask.png")
        if os.path.isfile(mask):
            pairs.append((rgbd, mask))
    return pairs


def get_depth_dir(tree_path):
    """Vrati Depth/ folder ako postoji i ima fajlove."""
    d = os.path.join(tree_path, "Depth")
    if os.path.isdir(d) and any(f.endswith("_rgbd.npy") for f in os.listdir(d)):
        return d
    return None


def split_trees(trees, val_ratio, test_ratio, seed=42):
    rng = random.Random(seed)
    trees = trees.copy()
    rng.shuffle(trees)
    n       = len(trees)
    n_val   = max(1, round(n * val_ratio))
    n_test  = max(1, round(n * test_ratio))
    n_train = n - n_val - n_test
    return trees[:n_train], trees[n_train:n_train+n_val], trees[n_train+n_val:]


def copy_pairs(pairs, split_dir, prefix=""):
    img_dir  = os.path.join(split_dir, "images")
    mask_dir = os.path.join(split_dir, "masks")
    os.makedirs(img_dir,  exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    for rgbd_src, mask_src in pairs:
        stem     = Path(rgbd_src).stem.replace("_rgbd", "")
        new_name = f"{prefix}{stem}"
        shutil.copy2(rgbd_src, os.path.join(img_dir,  f"{new_name}_rgbd.npy"))
        shutil.copy2(mask_src, os.path.join(mask_dir, f"{new_name}_mask.png"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anotirane", default="/home/samuel/Desktop/Anotirane")
    parser.add_argument("--out_dir",   default="/home/samuel/Desktop/Pokusaj_Chen_MultiDepth/data_v2")
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    base = args.anotirane
    random.seed(args.seed)

    # ── Prikupi sve grupe stabala ─────────────────────────────────────────────

    # Grupa A: Samuel ručno (Drvo0000-Drvo0015)
    group_a = []
    for name in MANUAL_SAMUEL:
        p = os.path.join(base, name)
        if os.path.isdir(p) and get_depth_dir(p):
            group_a.append((name, p, "A"))

    # Grupa B: Valentin ručno (tree_1_V_0179-0183)
    group_b = []
    for name in MANUAL_VALENTIN:
        p = os.path.join(base, name)
        if os.path.isdir(p) and get_depth_dir(p):
            group_b.append((name, p, "B"))

    # Grupa C: tree_najnovije_od_valeta — RUČNO (za val/test)
    group_c_manual = []
    najnovije_base = os.path.join(base, "tree_najnovije_od_valeta")
    for name in MANUAL_NAJNOVIJE:
        p = os.path.join(najnovije_base, name)
        if os.path.isdir(p) and get_depth_dir(p):
            group_c_manual.append((name, p, "CM"))

    # Grupa D: tree_najnovije_od_valeta — ostala (samo train)
    group_d_train = []
    if os.path.isdir(najnovije_base):
        for name in sorted(os.listdir(najnovije_base)):
            if name in MANUAL_NAJNOVIJE:
                continue
            p = os.path.join(najnovije_base, name)
            if os.path.isdir(p) and get_depth_dir(p):
                group_d_train.append((name, p, "D"))

    # Grupa E: tree_nove_od_valeta — sve samo train
    group_e_train = []
    nove_base = os.path.join(base, "tree_nove_od_valeta")
    if os.path.isdir(nove_base):
        for name in sorted(os.listdir(nove_base)):
            p = os.path.join(nove_base, name)
            if os.path.isdir(p) and get_depth_dir(p):
                group_e_train.append((name, p, "E"))

    print(f"\nGrupe stabala:")
    print(f"  A  Samuel ručno:               {len(group_a)} stabala")
    print(f"  B  Valentin ručno:             {len(group_b)} stabala")
    print(f"  CM Najnovije ručno (val/test): {len(group_c_manual)} stabala")
    print(f"  D  Najnovije ostalo (train):   {len(group_d_train)} stabala")
    print(f"  E  tree_nove_od_valeta (train):{len(group_e_train)} stabala")

    # ── Split ručnih stabala (A + B + CM) → train/val/test ───────────────────
    all_manual = group_a + group_b + group_c_manual
    total_manual = len(all_manual)

    # Sve ručne grupe zajedno → split pool
    all_manual_trees = group_a + group_b + group_c_manual
    manual_train, manual_val, manual_test = split_trees(
        all_manual_trees, val_ratio=0.30, test_ratio=0.30, seed=args.seed)
    a_train, a_val, a_test = manual_train, manual_val, manual_test
    b_train, b_val, b_test = [], [], []
    c_train, c_val, c_test = [], [], []

    # TRAIN = ručni dio + svi D i E
    train_trees = manual_train + group_d_train + group_e_train
    val_trees   = a_val   + b_val   + c_val
    test_trees  = a_test  + b_test  + c_test

    print(f"\nSplit po stablu:")
    print(f"  TRAIN: {len(train_trees)} stabala")
    print(f"  VAL:   {len(val_trees)} stabala")
    print(f"  TEST:  {len(test_trees)} stabala")

    # ── Prikupi parove fajlova ────────────────────────────────────────────────
    def collect_all(tree_list):
        result = []
        for i, (name, path, grp) in enumerate(tree_list):
            d = get_depth_dir(path)
            if d:
                prefix = f"{grp}_{name}_"
                for rgbd, mask in collect_pairs(d):
                    result.append((rgbd, mask, prefix))
        return result

    train_data = collect_all(train_trees)
    val_data   = collect_all(val_trees)
    test_data  = collect_all(test_trees)

    total = len(train_data) + len(val_data) + len(test_data)

    print(f"\nFajlovi:")
    print(f"  TRAIN: {len(train_data)} ({100*len(train_data)/total:.1f}%)")
    print(f"  VAL:   {len(val_data)}  ({100*len(val_data)/total:.1f}%)")
    print(f"  TEST:  {len(test_data)}  ({100*len(test_data)/total:.1f}%)")
    print(f"  UKUPNO:{total}")

    # ── Kopiraj fajlove ───────────────────────────────────────────────────────
    print(f"\nKopiranje u: {args.out_dir}")
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
        print(f"  {split:5s}: {len(data)} fajlova → {split_dir}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"  Ukupno stabala:   {len(train_trees)+len(val_trees)+len(test_trees)}")
    print(f"  Ukupno fajlova:   {total}")
    print(f"  TRAIN: {len(train_data)} ({100*len(train_data)/total:.1f}%) — ručno + model+pregled")
    print(f"  VAL:   {len(val_data)} ({100*len(val_data)/total:.1f}%)  — samo ručno anotirano")
    print(f"  TEST:  {len(test_data)} ({100*len(test_data)/total:.1f}%)  — samo ručno anotirano")
    print(f"  Split po stablu (no data leakage)")
    print(f"  Seed: {args.seed}")
    print(f"{'='*60}")
    print(f"\nGotovo! Data u: {args.out_dir}")


if __name__ == "__main__":
    main()