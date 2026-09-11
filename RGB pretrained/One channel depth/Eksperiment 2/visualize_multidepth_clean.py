"""
visualize_multidepth_clean.py
────────────────────────────────────────────────────────────────────────────
Vizualizacija predikcija svih modela iz Pokusaj_Chen_MultiDepthV2 projekta.
Sprema tri ciste visoke slike spremne za LaTeX subfigure (bez teksta i okvira):
  <prefix>-mask.png     — Ground Truth maske (crvena=deblo, plava=grane, zelena=potpora)
  <prefix>-pred.png     — Predikcije odabranog modela
  <prefix>-overlay.png  — TP/FP/FN overlay za klasu grane (zelena=TP, crvena=FP, plava=FN)
"""

import argparse
import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config as cfg
from dataset import AppleTreeDataset
from augmentations import get_val_transforms
from evaluate import _load_model
from models.esanet import build_esanet
from models import (build_unet, build_deeplabv3, build_deeplabv3plus, build_manet,
                    build_segformer, Pix2PixGenerator)

# ── Boje (BGR za OpenCV export) ──────────────────────────────────────────────
# Klase: 0=ostalo/pozadina, 1=deblo, 2=grane, 3=potpora
CLASS_COLORS_BGR = np.array([
    [0,   0,   0  ],   # 0 pozadina — crna
    [0,   0,   255],   # 1 deblo    — crvena (BGR: 0,0,255)
    [255, 0,   0  ],   # 2 grane    — plava  (BGR: 255,0,0)
    [0,   255, 0  ],   # 3 potpora  — zelena (BGR: 0,255,0)
], dtype=np.uint8)

# Overlay boje za klasu grana (BGR)
TP_COLOR = (0,   255, 0)   # zelena (Točno detektirana grana)
FP_COLOR = (0,   0,   255) # crvena (Lažno detektirana grana)
FN_COLOR = (255, 0,   0)   # plava  (Propuštena grana)


# ── Pomoćne funkcije ────────────────────────────────────────────────────────

def colorize_mask(mask):
    """Pretvori 2D masku klasa u BGR sliku."""
    return CLASS_COLORS_BGR[mask]


def make_overlay(gt, pred, cls=2):
    """Izradi TP/FP/FN overlay za specificiranu klasu."""
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


def stack_vertical(images, pad=6):
    """Spaja listu slika vertikalno s crnim razmakom izmedu."""
    if not images:
        return None
    w = images[0].shape[1]
    pad_row = np.zeros((pad, w, 3), dtype=np.uint8)
    stacked = images[0]
    for img in images[1:]:
        stacked = np.concatenate([stacked, pad_row, img], axis=0)
    return stacked


def get_model_builder(model_key):
    """Mapira naziv modela na njegovu instancu i checkpoint."""
    models_registry = {
        "unet_resnet34": (
            build_unet("resnet34", weights=None, in_channels=cfg.IN_CHANNELS, num_classes=cfg.NUM_CLASSES),
            os.path.join(cfg.CHECKPOINT_DIR, "unet_resnet34_best.pth"), "model_state"
        ),
        "unet_effb4": (
            build_unet("efficientnet-b4", weights=None, in_channels=cfg.IN_CHANNELS, num_classes=cfg.NUM_CLASSES),
            os.path.join(cfg.CHECKPOINT_DIR, "unet_efficientnet_b4_best.pth"), "model_state"
        ),
        "deeplabv3": (
            build_deeplabv3(pretrained=False, in_channels=cfg.IN_CHANNELS, num_classes=cfg.NUM_CLASSES),
            os.path.join(cfg.CHECKPOINT_DIR, "deeplabv3_resnet50_best.pth"), "model_state"
        ),
        "deeplabv3plus": (
            build_deeplabv3plus(encoder="mobilenet_v2", weights=None, in_channels=cfg.IN_CHANNELS, num_classes=cfg.NUM_CLASSES),
            os.path.join(cfg.CHECKPOINT_DIR, "deeplabv3plus_mobilenetv2_best.pth"), "model_state"
        ),
        "segformer": (
            build_segformer(variant=cfg.SEGFORMER_VARIANT, weights=None, in_channels=cfg.IN_CHANNELS, num_classes=cfg.NUM_CLASSES),
            os.path.join(cfg.CHECKPOINT_DIR, "segformer_b1_best.pth"), "model_state"
        ),
        "manet": (
            build_manet(pretrained=False, in_channels=cfg.IN_CHANNELS, num_classes=cfg.NUM_CLASSES),
            os.path.join(cfg.CHECKPOINT_DIR, "manet_resnext50_best.pth"), "model_state"
        ),
        "esanet": (
            build_esanet(pretrained=False, in_channels=cfg.IN_CHANNELS, num_classes=cfg.NUM_CLASSES),
            os.path.join(cfg.CHECKPOINT_DIR, "esanet_resnet34_best.pth"), "model_state"
        ),
    }

    if model_key not in models_registry:
        raise ValueError(f"Nepoznat model '{model_key}'. Dostupni modeli: {list(models_registry.keys())}")

    return models_registry[model_key]


# ── Glavna logika ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generiranje cistih slika za LaTeX subfigure")
    parser.add_argument("--model", type=str, required=True,
                        help="Kratko ime modela (npr. unet_resnet34, segformer, deeplabv3plus, esanet, manet, deeplabv3, unet_effb4)")
    parser.add_argument("--samples", nargs="+", required=True,
                        help="Popis imena testnih uzoraka bez ekstenzije")
    parser.add_argument("--overlay_class", type=int, default=2,
                        help="Klasa za overlay (default: 2 = grane)")
    parser.add_argument("--output_dir", type=str, default="./qualitative_dformer",
                        help="Direktorij za spremanje slika")
    parser.add_argument("--pad", type=int, default=6,
                        help="Crni razmak izmedu slika u pikselima")
    args = parser.parse_args()

    device = cfg.DEVICE

    # 1. Učitavanje modela
    model_inst, ckpt_path, state_key = get_model_builder(args.model)
    print(f"\nUčitavam model: {args.model} iz {ckpt_path}")
    model = _load_model(model_inst, ckpt_path, device, key=state_key)
    if model is None:
        print("Greška pri učitavanju checkpointa. Prekidam.")
        return

    # 2. Priprema skupa podataka
    test_dataset = AppleTreeDataset(
        img_dir=cfg.TEST_IMG_DIR,
        mask_dir=cfg.TEST_MASK_DIR,
        transform=get_val_transforms(cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH),
        img_size=(cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH),
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    # 3. Dohvaćanje samo traženih uzoraka u zadanom redoslijedu
    samples_dict = {}
    print(f"Tražim {len(args.samples)} zadanih uzoraka u test skupu...")
    
    for batch in test_loader:
        stem = batch["stem"][0]
        if stem in args.samples:
            samples_dict[stem] = batch
            if len(samples_dict) == len(args.samples):
                break

    # Provjera jesu li svi uzorci pronađeni
    missing = [s for s in args.samples if s not in samples_dict]
    if missing:
        print(f"Upozorenje: Sljedeći uzorci nisu pronađeni u test skupu: {missing}")

    mask_rows, pred_rows, overlay_rows = [], [], []

    # 4. Inferencija i izrada vizualizacija u točnom redoslijedu s CLI-ja
    for stem in args.samples:
        if stem not in samples_dict:
            continue

        batch = samples_dict[stem]
        image = batch["image"].to(device)
        gt = batch["mask"].squeeze(0).numpy().astype(np.int32)

        with torch.no_grad():
            logits = model(image)
            pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int32)

        mask_rows.append(colorize_mask(gt))
        pred_rows.append(colorize_mask(pred))
        overlay_rows.append(make_overlay(gt, pred, cls=args.overlay_class))
        print(f"  Obrađeno: {stem}")

    if not mask_rows:
        print("Nema obrađenih uzoraka!")
        return

    # 5. Vertikalno spajanje slika
    mask_img = stack_vertical(mask_rows, pad=args.pad)
    pred_img = stack_vertical(pred_rows, pad=args.pad)
    overlay_img = stack_vertical(overlay_rows, pad=args.pad)

    # 6. Spremanje datoteka s traženom konvencijom imenovanja: <model>-mask, prediction, overlay
    os.makedirs(args.output_dir, exist_ok=True)
    prefix = args.model

    mask_path = os.path.join(args.output_dir, f"{prefix}-mask.png")
    pred_path = os.path.join(args.output_dir, f"{prefix}-prediction.png")
    overlay_path = os.path.join(args.output_dir, f"{prefix}-overlay.png")

    cv2.imwrite(mask_path, mask_img)
    cv2.imwrite(pred_path, pred_img)
    cv2.imwrite(overlay_path, overlay_img)

    print(f"\nUspješno spremljeno u '{args.output_dir}/':")
    print(f"  1. {prefix}-mask.png")
    print(f"  2. {prefix}-prediction.png")
    print(f"  3. {prefix}-overlay.png")


if __name__ == "__main__":
    main()