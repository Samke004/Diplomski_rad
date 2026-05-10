"""
ply_to_rgbd.py
────────────────────────────────────────────────────────────────────────────
Converts every PLY file to proper 4-channel RGBD images using REAL RGB colours.

Input  : PLY file with x,y,z + red,green,blue + annotation scalars
Output per PLY (saved in <tree_folder>/Depth/):
  <stem>_rgb.png      — real RGB image (3-channel, 8-bit)
  <stem>_depth.png    — 16-bit depth in mm
  <stem>_mask.png     — label mask (0=bg, 1=trunk, 2=branches, 3=support)
  <stem>_rgbd.npy     — 4-channel float32 [H,W,4] (RGB + depth)
  <stem>_visual.png   — 4-panel: Real RGB | Depth | Annotation | RGBD overlay

Usage
─────
  # Single tree folder
  python ply_to_rgbd.py --folder /home/samuel/Desktop/Anotirane/Drvo0000

  # ALL trees at once
  python ply_to_rgbd.py --batch /home/samuel/Desktop/Anotirane
"""

import argparse
import os
import numpy as np
from plyfile import PlyData
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Asus camera intrinsics ────────────────────────────────────────────────────
FX_DEPTH = 570.3422241210938
FY_DEPTH = 570.3422241210938
CX_DEPTH = 314.5
CY_DEPTH = 235.5

IMAGE_HEIGHT = 480
IMAGE_WIDTH  = 640

# Annotation colours (for visualisation panel only)
LABEL_COLORS = np.array([
    [0,   0,   0  ],   # 0 background — black
    [255, 0,   0  ],   # 1 trunk      — red
    [0,   0,   255],   # 2 branches   — blue
    [0,   255, 0  ],   # 3 support    — green
], dtype=np.uint8)


# ── Unified PLY loader ────────────────────────────────────────────────────────

def load_ply(path):
    """
    Returns:
        pts    : [N, 3] float32  — XYZ coordinates
        colors : [N, 3] uint8    — real RGB from camera
        labels : [N]   int64     — 0=bg, 1=trunk, 2=branches, 3=support
    """
    ply   = PlyData.read(path)
    v     = ply.elements[0].data
    props = {p.name for p in ply.elements[0].properties}

    pts = np.column_stack([v['x'], v['y'], v['z']]).astype(np.float32)

    # Real RGB colours
    if 'red' in props and 'green' in props and 'blue' in props:
        colors = np.column_stack([
            v['red'].astype(np.uint8),
            v['green'].astype(np.uint8),
            v['blue'].astype(np.uint8),
        ])
    else:
        colors = np.full((len(pts), 3), 128, dtype=np.uint8)
        print(f"    [WARN] No RGB in {os.path.basename(path)} — using grey")

    # Labels — normalised to 0=bg, 1=trunk, 2=branches, 3=support
    labels = np.zeros(len(pts), dtype=np.int64)

    # Samuel's format
    if 'scalar_Grane' in props or 'scalar_Deblo' in props:
        if 'scalar_Deblo' in props:
            mask = ~np.isnan(v['scalar_Deblo'].astype(np.float32))
            labels[mask & (v['scalar_Deblo'] == 1)] = 1
        if 'scalar_Grane' in props:
            mask = ~np.isnan(v['scalar_Grane'].astype(np.float32))
            labels[mask & (v['scalar_Grane'] == 2)] = 2
        if 'scalar_Potpora' in props:
            mask = ~np.isnan(v['scalar_Potpora'].astype(np.float32))
            labels[mask & (v['scalar_Potpora'] == 3)] = 3

    # Valentin's formats
    elif 'scalar_PTv3_pred' in props:
        vals  = v['scalar_PTv3_pred'].astype(np.float32)
        valid = ~np.isnan(vals)
        labels[valid & (vals == 0)] = 1
        labels[valid & (vals == 1)] = 2
        labels[valid & (vals == 2)] = 3
    elif 'scalar_segment' in props:
        vals  = v['scalar_segment'].astype(np.float32)
        valid = ~np.isnan(vals)
        labels[valid & (vals == 0)] = 1
        labels[valid & (vals == 1)] = 2
        labels[valid & (vals == 2)] = 3
    elif 'scalar_Constant' in props:
        vals  = v['scalar_Constant'].astype(np.float32)
        valid = ~np.isnan(vals)
        labels[valid & (vals == 0)] = 1
        labels[valid & (vals == 1)] = 2
        labels[valid & (vals == 2)] = 3
    else:
        print(f"    [WARN] Unknown annotation format — available: {props}")

    return pts, colors, labels


# ── Project to image ──────────────────────────────────────────────────────────

def project_to_image(pts, colors, labels):
    """
    Projects all points to 640×480 image space.
    Returns:
        rgb_image   [H, W, 3] uint8   — real RGB
        depth_image [H, W]    float32 — depth in metres
        label_image [H, W]    int32   — class labels (-1 = empty)
    """
    rgb_image   = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    depth_image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float32)
    label_image = np.full((IMAGE_HEIGHT, IMAGE_WIDTH), -1, dtype=np.int32)

    for (x, y, z), (r, g, b), label in zip(pts, colors, labels):
        z = abs(z)
        if z <= 0:
            continue

        u   = int((x * FX_DEPTH / z) + CX_DEPTH)
        v_p = int((y * FY_DEPTH / z) + CY_DEPTH)

        if 0 <= u < IMAGE_WIDTH and 0 <= v_p < IMAGE_HEIGHT:
            if depth_image[v_p, u] == 0 or z < depth_image[v_p, u]:
                depth_image[v_p, u] = z
                label_image[v_p, u] = label
                rgb_image[v_p, u]   = [r, g, b]

    # Flip vertically
    rgb_image   = np.flip(rgb_image,   axis=0).copy()
    depth_image = np.flip(depth_image, axis=0).copy()
    label_image = np.flip(label_image, axis=0).copy()

    return rgb_image, depth_image, label_image


# ── Build output files ────────────────────────────────────────────────────────

def build_outputs(rgb_image, depth_image, label_image):
    # 16-bit depth PNG (metres → mm)
    depth_16bit = (depth_image * 1000).astype(np.uint16)

    # Label mask: -1 (empty) → 0 (background)
    mask = np.where(label_image >= 0, label_image, 0).astype(np.uint8)

    # Annotation colour image (for visualisation)
    annot = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    for lbl in range(4):
        annot[label_image == lbl] = LABEL_COLORS[lbl]

    # 4-channel RGBD float32 [H, W, 4]
    rgbd = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 4), dtype=np.float32)
    rgbd[:, :, :3] = rgb_image.astype(np.float32) / 255.0
    rgbd[:, :,  3] = depth_image

    return depth_16bit, mask, annot, rgbd


# ── Visualisation — 4 panels ──────────────────────────────────────────────────

def save_visual(rgb_image, depth_image, label_image, annot, out_path, title=""):
    valid = depth_image > 0

    # Normalised depth for display
    depth_vis = np.zeros_like(depth_image)
    if np.any(valid):
        d = depth_image[valid]
        depth_vis[valid] = (depth_image[valid] - d.min()) / (d.max() - d.min() + 1e-10)

    # RGBD overlay: annotation colours on top of depth background
    overlay = plt.cm.plasma(depth_vis)[:, :, :3]
    for lbl in range(1, 4):
        m = label_image == lbl
        overlay[m] = LABEL_COLORS[lbl] / 255.0

    # Annotation overlaid on real RGB (semi-transparent)
    rgb_float = rgb_image.astype(np.float32) / 255.0
    annot_overlay = rgb_float.copy()
    for lbl in range(1, 4):
        m = label_image == lbl
        annot_overlay[m] = (
            rgb_float[m] * 0.4 + (LABEL_COLORS[lbl] / 255.0) * 0.6
        )

    fig, axes = plt.subplots(1, 5, figsize=(28, 5))

    axes[0].imshow(rgb_image)
    axes[0].set_title("Real RGB\n(from PLY camera colours)", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(depth_vis, cmap="plasma")
    axes[1].set_title("Depth\n(closer = brighter)", fontsize=10)
    axes[1].axis("off")

    axes[2].imshow(annot)
    axes[2].set_title("Annotation\n(red=trunk  blue=branches  green=support)", fontsize=10)
    axes[2].axis("off")

    axes[3].imshow(annot_overlay)
    axes[3].set_title("Annotation on RGB\n(labels overlaid on real image)", fontsize=10)
    axes[3].axis("off")

    axes[4].imshow(overlay)
    axes[4].set_title("RGBD Overlay\n(annotation on depth)", fontsize=10)
    axes[4].axis("off")

    n = {lbl: int((label_image == lbl).sum()) for lbl in range(4)}
    depth_mm = depth_image[valid]
    fig.suptitle(
        f"{title}  |  "
        f"trunk={n[1]:,}px  branches={n[2]:,}px  support={n[3]:,}px  |  "
        f"depth={int(depth_mm.min()*1000) if np.any(valid) else 0}–"
        f"{int(depth_mm.max()*1000) if np.any(valid) else 0} mm",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=90)
    plt.close()


# ── Process one PLY file ──────────────────────────────────────────────────────

def process_ply(ply_path, out_dir):
    stem = os.path.splitext(os.path.basename(ply_path))[0]

    pts, colors, labels = load_ply(ply_path)

    counts = {lbl: int((labels == lbl).sum()) for lbl in range(4)}
    print(f"    Points: {len(pts):,}  "
          f"trunk={counts[1]:,}  branches={counts[2]:,}  "
          f"support={counts[3]:,}  bg={counts[0]:,}")

    rgb_image, depth_image, label_image = project_to_image(pts, colors, labels)

    valid_px = (depth_image > 0).sum()
    print(f"    Valid pixels: {valid_px:,}")

    if valid_px == 0:
        print(f"    [SKIP] Empty projection")
        return False

    depth_16bit, mask, annot, rgbd = build_outputs(rgb_image, depth_image, label_image)

    # Save all outputs
    Image.fromarray(rgb_image).save(  os.path.join(out_dir, f"{stem}_rgb.png"))
    Image.fromarray(depth_16bit).save(os.path.join(out_dir, f"{stem}_depth.png"))
    Image.fromarray(mask).save(       os.path.join(out_dir, f"{stem}_mask.png"))
    Image.fromarray(annot).save(      os.path.join(out_dir, f"{stem}_annot.png"))
    np.save(                          os.path.join(out_dir, f"{stem}_rgbd.npy"), rgbd)
    save_visual(rgb_image, depth_image, label_image, annot,
                os.path.join(out_dir, f"{stem}_visual.png"),
                title=os.path.basename(ply_path))

    return True


# ── Process one tree folder ───────────────────────────────────────────────────

def process_folder(folder):
    tree_name = os.path.basename(folder)
    out_dir   = os.path.join(folder, "Depth")
    os.makedirs(out_dir, exist_ok=True)

    ply_files = sorted([f for f in os.listdir(folder) if f.endswith(".ply")])
    if not ply_files:
        print(f"  [SKIP] No PLY files in {folder}")
        return 0, 0

    print(f"\n{'─'*60}")
    print(f"  {tree_name}  ({len(ply_files)} PLY files)  →  {out_dir}")
    print(f"{'─'*60}")

    ok, failed = 0, 0
    for fname in ply_files:
        print(f"\n  {fname}")
        if process_ply(os.path.join(folder, fname), out_dir):
            ok += 1
        else:
            failed += 1

    print(f"\n  Done: {ok} OK  |  {failed} failed")
    return ok, failed


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--folder", help="Single tree folder (e.g. Drvo0000)")
    group.add_argument("--batch",  help="Root folder with all tree folders")
    args = parser.parse_args()

    total_ok, total_failed = 0, 0

    if args.folder:
        ok, failed = process_folder(args.folder)
        total_ok, total_failed = ok, failed

    elif args.batch:
        root = args.batch
        tree_folders = sorted([
            os.path.join(root, d)
            for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
            and any(f.endswith(".ply")
                    for f in os.listdir(os.path.join(root, d)))
        ])

        if not tree_folders:
            print(f"No tree folders with PLY files found in {root}")
            return

        print(f"\nFound {len(tree_folders)} tree folders to process")
        for folder in tree_folders:
            ok, failed = process_folder(folder)
            total_ok     += ok
            total_failed += failed

    print(f"\n{'='*60}")
    print(f"  BATCH COMPLETE")
    print(f"  Total converted: {total_ok}")
    print(f"  Total failed:    {total_failed}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()