"""
Vizualizacija predikcija DFormer modela:
  - Stupac 1: Ground Truth
  - Stupac 2: Predikcija modela
  - Stupac 3: Overlay TP/FP/FN za klasu "grane" (branches, klasa 2)

Pokretanje:
  cd ~/Downloads/DFormerv1/DFormer
  python3 visualize_dformer.py
"""

import os
import argparse
import math
from importlib import import_module
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

# Paleta boja za segmentacijske maske (BGR za OpenCV)
# klase: 0=ostalo, 1=deblo, 2=grane, 3=potpora
CLASS_COLORS_BGR = [
    (0,   0,   0),    # 0 ostalo    - crna
    (0,   0,   255),  # 1 deblo     - crvena
    (255, 0,   0),    # 2 grane     - plava
    (0,   255, 0),    # 3 potpora   - zelena
]

# Boje za TP/FP/FN overlay (BGR)
TP_COLOR  = (0,   255, 0)   # zelena
FP_COLOR  = (0,   0,   255) # crvena
FN_COLOR  = (255, 0,   0)   # plava


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Pretvori 2D masku klasa u BGR sliku."""
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, color in enumerate(CLASS_COLORS_BGR):
        out[mask == cls_idx] = color
    return out


def make_overlay(gt: np.ndarray, pred: np.ndarray, cls: int = 2) -> np.ndarray:
    """
    Stvori TP/FP/FN overlay za jednu klasu.
    gt, pred: 2D numpy int arraji s vrijednostima klasa.
    cls: indeks klase za analizu (default 2 = grane)
    """
    h, w = gt.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)

    gt_cls   = (gt   == cls)
    pred_cls = (pred == cls)

    tp = gt_cls  & pred_cls
    fp = pred_cls & ~gt_cls
    fn = gt_cls  & ~pred_cls

    overlay[tp] = TP_COLOR
    overlay[fp] = FP_COLOR
    overlay[fn] = FN_COLOR

    return overlay


def load_model(config, checkpoint_path, device):
    from models.builder import EncoderDecoder as segmodel

    criterion = nn.CrossEntropyLoss(reduction="mean", ignore_index=config.background)
    model = segmodel(
        cfg=config,
        criterion=criterion,
        norm_layer=nn.SyncBatchNorm,
        syncbn=True,
    )

    weight = torch.load(checkpoint_path, map_location="cpu")
    if "model" in weight:
        weight = weight["model"]
    elif "state_dict" in weight:
        weight = weight["state_dict"]

    result = model.load_state_dict(weight, strict=False)
    print(f"Loaded checkpoint: {result}")
    model.to(device)
    model.eval()
    return model


def preprocess(rgb_path: str, depth_path: str, config):
    """Učitaj i preprocess RGB + Depth sliku za model."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    H, W = config.image_height, config.image_width

    # RGB
    rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (W, H)).astype(np.float32) / 255.0
    rgb = (rgb - mean) / std
    rgb_t = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).float()

    # Depth (single channel)
    depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    depth = cv2.resize(depth, (W, H)).astype(np.float32) / 255.0
    depth = (depth - 0.5) / 0.5
    depth_t = torch.from_numpy(depth[np.newaxis, np.newaxis]).float()

    return rgb_t, depth_t


def predict(model, rgb_t, depth_t, device):
    rgb_t   = rgb_t.to(device)
    depth_t = depth_t.to(device)
    with torch.no_grad():
        logits = model(rgb_t, depth_t)
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int32)
    return pred


def load_gt(label_path: str, config) -> np.ndarray:
    H, W = config.image_height, config.image_width
    gt = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    gt = cv2.resize(gt, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.int32)
    # postavi ignore label na 0 za vizualizaciju
    gt[gt == config.background] = 0
    return gt


def make_grid(images_row: list, pad: int = 4) -> np.ndarray:
    """
    Prima listu [GT_bgr, Pred_bgr, Overlay_bgr] i spoji ih horizontalno s paddingom.
    """
    pad_col = np.zeros((images_row[0].shape[0], pad, 3), dtype=np.uint8)
    row = images_row[0]
    for img in images_row[1:]:
        row = np.concatenate([row, pad_col, img], axis=1)
    return row


def add_header(grid: np.ndarray, labels: list, font_scale: float = 0.5) -> np.ndarray:
    """Dodaj naslovni redak iznad grida."""
    h, w = grid.shape[:2]
    col_w = w // len(labels)
    header = np.zeros((28, w, 3), dtype=np.uint8)
    for i, label in enumerate(labels):
        x = i * col_w + 5
        cv2.putText(header, label, (x, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([header, grid], axis=0)


def run(args):
    # Učitaj config
    config = getattr(import_module(args.config), "C")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Učitaj model
    model = load_model(config, args.checkpoint, device)

    # Prikupi parove slika iz eval_source
    eval_source = config.eval_source  # putanja do test.txt
    rgb_root    = config.rgb_root_folder
    depth_root  = config.x_root_folder
    gt_root     = config.gt_root_folder
    rgb_ext     = config.rgb_format
    depth_ext   = config.x_format
    gt_ext      = config.gt_format

    with open(eval_source) as f:
        names = [line.strip() for line in f if line.strip()]

    rows = []
    for name in names[:args.n]:
        rgb_path   = os.path.join(rgb_root,   name + rgb_ext)
        depth_path = os.path.join(depth_root, name + depth_ext)
        gt_path    = os.path.join(gt_root,    name + gt_ext)

        if not os.path.exists(rgb_path):
            print(f"Nema RGB: {rgb_path}, preskačem.")
            continue

        # Preprocess
        rgb_t, depth_t = preprocess(rgb_path, depth_path, config)

        # Predikcija
        pred = predict(model, rgb_t, depth_t, device)

        # GT
        if os.path.exists(gt_path):
            gt = load_gt(gt_path, config)
        else:
            gt = np.zeros_like(pred)

        # Vizualizacija
        gt_vis      = colorize_mask(gt)
        pred_vis    = colorize_mask(pred)
        overlay_vis = make_overlay(gt, pred, cls=args.overlay_class)

        row = make_grid([gt_vis, pred_vis, overlay_vis], pad=4)
        rows.append(row)
        print(f"Obrađeno: {name}")

    if not rows:
        print("Nema slika za prikaz!")
        return

    # Spoji sve retke vertikalno
    pad_row = np.zeros((4, rows[0].shape[1], 3), dtype=np.uint8)
    grid = rows[0]
    for row in rows[1:]:
        grid = np.concatenate([grid, pad_row, row], axis=0)

    # Dodaj header
    class_name = config.class_names[args.overlay_class]
    headers = ["Ground Truth", "Prediction", f"Overlay (TP/FP/FN {class_name})"]
    grid = add_header(grid, headers)

    # Spremi
    out_path = args.output
    cv2.imwrite(out_path, grid)
    print(f"\nSpravljeno u: {out_path}")
    print(f"Boje overlaya: Zelena=TP, Crvena=FP, Plava=FN (klasa: {class_name})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vizualizacija DFormer predikcija")
    parser.add_argument("--config",
                        default="local_configs.BranchDataset.DFormer_Base",
                        help="Config modul (npr. local_configs.BranchDataset.DFormer_Base)")
    parser.add_argument("--checkpoint",
                        default="checkpoints/BranchDataset_DFormer-Base_20260507-181200/epoch-165_miou_88.24.pth",
                        help="Putanja do checkpoint .pth fajla")
    parser.add_argument("--n", type=int, default=8,
                        help="Broj slika za prikaz (default: 8)")
    parser.add_argument("--overlay_class", type=int, default=2,
                        help="Klasa za TP/FP/FN overlay (0=ostalo,1=deblo,2=grane,3=potpora)")
    parser.add_argument("--output", default="visualization2.png",
                        help="Izlazna slika (default: visualization2.png)")
    args = parser.parse_args()

    run(args)