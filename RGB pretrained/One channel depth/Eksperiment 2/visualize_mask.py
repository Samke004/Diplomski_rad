"""
Mask Visualizer
===============
Vizualizira ~10 maski po splitu, boji klase i sprema u output_dir.

Boje:
  Klasa 0 → Plava    (background)
  Klasa 1 → Crvena   (trunk / deblo)
  Klasa 2 → Zelena   (branches / grane)
  Klasa 3 → Indigo   (fine branches / vrhovi)

Uporaba:
    python visualize_masks.py --data_root data
    python visualize_masks.py --data_root data --n 10 --output_dir mask_vis
"""

import os
import argparse
import random

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Konfiguracija boja ────────────────────────────────────────────────────────
CLASS_COLORS = {
    0: (  0,   0,   0),   # Crna      – background
    1: ( 30, 100, 220),   # Plava     – deblo
    2: (220,  30,  30),   # Crvena    – grane
    3: ( 50, 180,  50),   # Zelena    – sitne grane / vrhovi
}

CLASS_LABELS = {
    0: "Klasa 0 – Pozadina",
    1: "Klasa 1 – Deblo",
    2: "Klasa 2 – Grane",
    3: "Klasa 3 – Sitne grane",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def is_backup(fname):
    return "BACKUP" in fname.upper()


def load_mask(path):
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr


def mask_to_rgb(mask):
    """Pretvori 2D mask array u RGB sliku prema CLASS_COLORS."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in CLASS_COLORS.items():
        rgb[mask == cls] = color
    return rgb


def collect_masks(data_root, splits=("train", "val", "test")):
    """Vrati dict {split: [paths]} za sve non-BACKUP maske."""
    result = {}
    for split in splits:
        masks_dir = os.path.join(data_root, split, "masks")
        if not os.path.isdir(masks_dir):
            continue
        paths = sorted([
            os.path.join(masks_dir, f)
            for f in os.listdir(masks_dir)
            if f.lower().endswith(".png") and not is_backup(f)
        ])
        if paths:
            result[split] = paths
    return result


def legend_patches():
    return [
        mpatches.Patch(color=np.array(c) / 255, label=CLASS_LABELS[k])
        for k, c in CLASS_COLORS.items()
    ]


# ── Vizualizacija ─────────────────────────────────────────────────────────────

def visualize_split(split, paths, n_samples, output_dir):
    """Odaberi n_samples random maski iz splita i spremi grid sliku."""
    sample = random.sample(paths, min(n_samples, len(paths)))

    cols = min(5, len(sample))
    rows = (len(sample) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4 + 1))
    fig.suptitle(f"Split: {split.upper()}  ({len(sample)} maski)", fontsize=14, fontweight="bold", y=1.01)

    # Osiguraj da axes uvijek bude 2D lista
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    for idx, path in enumerate(sample):
        r, c = divmod(idx, cols)
        ax = axes[r][c]

        mask = load_mask(path)
        rgb  = mask_to_rgb(mask)

        ax.imshow(rgb)
        ax.set_title(os.path.basename(path).replace("_mask.png", ""), fontsize=7, pad=3)
        ax.axis("off")

        # Klase prisutne u ovoj masci
        unique = np.unique(mask)
        stats = "  ".join(
            f"C{cls}:{(mask == cls).sum() / mask.size * 100:.1f}%"
            for cls in unique
        )
        ax.set_xlabel(stats, fontsize=6, labelpad=2)
        ax.xaxis.set_label_position("bottom")

    # Sakrij prazne subplotove
    for idx in range(len(sample), rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].axis("off")

    # Legenda ispod
    fig.legend(
        handles=legend_patches(),
        loc="lower center",
        ncol=4,
        fontsize=9,
        frameon=True,
        bbox_to_anchor=(0.5, -0.03),
    )

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"masks_{split}.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Spreman: {out_path}")
    return out_path


def visualize_montage(all_samples, output_dir):
    """Jedna velika slika s po 3–4 uzorka iz svakog splita."""
    per_split = 4
    splits = list(all_samples.keys())
    n_cols = per_split
    n_rows = len(splits)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 3.5 + 1))
    fig.suptitle("Pregled maski – svi splitovi", fontsize=14, fontweight="bold")

    if n_rows == 1:
        axes = [axes]

    for row_idx, split in enumerate(splits):
        sample = random.sample(all_samples[split], min(per_split, len(all_samples[split])))
        for col_idx in range(n_cols):
            ax = axes[row_idx][col_idx]
            if col_idx < len(sample):
                mask = load_mask(sample[col_idx])
                rgb  = mask_to_rgb(mask)
                ax.imshow(rgb)
                ax.set_title(os.path.basename(sample[col_idx]).replace("_mask.png", ""), fontsize=6, pad=2)
                if col_idx == 0:
                    ax.set_ylabel(split.upper(), fontsize=10, fontweight="bold", rotation=90, labelpad=5)
            ax.axis("off")

    fig.legend(
        handles=legend_patches(),
        loc="lower center",
        ncol=4,
        fontsize=9,
        frameon=True,
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.tight_layout()
    out_path = os.path.join(output_dir, "masks_overview.png")
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Spreman overview: {out_path}")
    return out_path


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Vizualizacija maski dataseta")
    parser.add_argument("--data_root",  type=str, default="data")
    parser.add_argument("--n",          type=int, default=10,          help="Broj maski po splitu")
    parser.add_argument("--output_dir", type=str, default="mask_vis",  help="Izlazni direktorij")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Vizualizacija maski")
    print(f"  data_root  : {os.path.abspath(args.data_root)}")
    print(f"  output_dir : {os.path.abspath(args.output_dir)}")
    print(f"  n po splitu: {args.n}")
    print(f"{'='*60}\n")

    all_masks = collect_masks(args.data_root)
    if not all_masks:
        print("[ERROR] Nisu pronađene nijedna maska. Provjeri --data_root.")
        return

    # Slika po splitu
    for split, paths in all_masks.items():
        print(f"[{split.upper()}] {len(paths)} maski, vizualiziram {min(args.n, len(paths))}...")
        visualize_split(split, paths, args.n, args.output_dir)

    # Overview montaža
    print("\n[OVERVIEW] Generiranje montaže...")
    visualize_montage(all_masks, args.output_dir)

    print(f"\n✅ Gotovo! Slike su u: {os.path.abspath(args.output_dir)}/\n")


if __name__ == "__main__":
    main()