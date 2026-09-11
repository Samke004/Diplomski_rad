"""
visualize_dformer_clean.py
────────────────────────────────────────────────────────────────────────────
Vizualizacija predikcija DFormer modela u tri odvojene ciste slike
spremne za subfigure u znanstvenom radu (FERIT stil).

Za svaki odabrani testni uzorak generira jedan redak. Sve retke slaze
vertikalno u tri odvojene visoke slike:

  <prefix>_mask.png     — Ground Truth maske (crveno=deblo, plavo=grane, zeleno=potpora)
  <prefix>_pred.png     — Predikcije modela
  <prefix>_overlay.png  — TP/FP/FN overlay za klasu grane
                          (zeleno=TP, crveno=FP, plavo=FN)

Sve slike su BEZ TEKSTA, bez naslova, bez okvira — samo piksel-na-piksel
sadrzaj, spremno za umetanje u LaTeX subfigure. LaTeX opis figure
pokriva sve tekstualne oznake.

Pokretanje:
  cd ~/Downloads/DFormerv1/DFormer

  # 6 slika iz testnog skupa, spremljeno u ./qualitative/
  python3 visualize_dformer_clean.py --n 6 --output_dir ./qualitative

  # Odaberi konkretne testne slike po imenu
  python3 visualize_dformer_clean.py \\
      --samples tree_1_V_0025_0 tree_1_V_0145_1 \\
      --output_dir ./qualitative

  # Promijeni klasu za overlay (default: 2 = grane)
  python3 visualize_dformer_clean.py --n 5 --overlay_class 3
"""

import argparse
import os
from importlib import import_module

import cv2
import numpy as np
import torch
import torch.nn as nn


# ── Boje ────────────────────────────────────────────────────────────────────
# Klase (BGR za OpenCV):  0=ostalo, 1=deblo, 2=grane, 3=potpora
CLASS_COLORS_BGR = [
    (0,   0,   0),    # 0 ostalo   — crna
    (0,   0,   255),  # 1 deblo    — crvena
    (255, 0,   0),    # 2 grane    — plava
    (0,   255, 0),    # 3 potpora  — zelena
]

# Overlay boje (BGR)
TP_COLOR = (0,   255, 0)   # zelena
FP_COLOR = (0,   0,   255) # crvena
FN_COLOR = (255, 0,   0)   # plava


# ── Pomocne funkcije ────────────────────────────────────────────────────────

def colorize_mask(mask):
    """2D masku klasa pretvori u BGR sliku."""
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, color in enumerate(CLASS_COLORS_BGR):
        out[mask == cls_idx] = color
    return out


def make_overlay(gt, pred, cls=2):
    """TP/FP/FN overlay za jednu klasu."""
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

    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=config.background)
    model = segmodel(cfg=config, criterion=criterion,
                     norm_layer=nn.SyncBatchNorm, syncbn=True)

    weight = torch.load(checkpoint_path, map_location='cpu')
    if 'model' in weight:
        weight = weight['model']
    elif 'state_dict' in weight:
        weight = weight['state_dict']

    result = model.load_state_dict(weight, strict=False)
    print(f'Loaded checkpoint: {result}')
    model.to(device)
    model.eval()
    return model


def preprocess(rgb_path, depth_path, config):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    H, W = config.image_height, config.image_width

    rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (W, H)).astype(np.float32) / 255.0
    rgb = (rgb - mean) / std
    rgb_t = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).float()

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


def load_gt(label_path, config):
    H, W = config.image_height, config.image_width
    gt = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    gt = cv2.resize(gt, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.int32)
    gt[gt == config.background] = 0
    return gt


def stack_vertical(images, pad=6):
    """Slozi listu slika vertikalno s crnim paddingom izmedu."""
    if not images:
        return None
    w = images[0].shape[1]
    pad_row = np.zeros((pad, w, 3), dtype=np.uint8)
    stacked = images[0]
    for img in images[1:]:
        stacked = np.concatenate([stacked, pad_row, img], axis=0)
    return stacked


# ── Glavna funkcija ────────────────────────────────────────────────────────

def run(args):
    config = getattr(import_module(args.config), 'C')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Uredaj: {device}')

    model = load_model(config, args.checkpoint, device)

    # Ucitaj popis test uzoraka
    with open(config.eval_source) as f:
        all_names = [line.strip() for line in f if line.strip()]

    # Odaberi uzorke
    if args.samples:
        # Explicitno navedeni uzorci
        chosen = []
        for name in args.samples:
            if name in all_names:
                chosen.append(name)
            else:
                print(f'  Upozorenje: {name} nije u test skupu, preskacem')
    else:
        # Prvih N
        chosen = all_names[:args.n]

    if not chosen:
        print('Nema odabranih uzoraka!')
        return

    print(f'\nObradjujem {len(chosen)} uzoraka...')

    mask_rows    = []
    pred_rows    = []
    overlay_rows = []

    for name in chosen:
        rgb_path   = os.path.join(config.rgb_root_folder, name + config.rgb_format)
        depth_path = os.path.join(config.x_root_folder,   name + config.x_format)
        gt_path    = os.path.join(config.gt_root_folder,  name + config.gt_format)

        if not os.path.exists(rgb_path):
            print(f'  Preskacem (nema RGB): {name}')
            continue

        # Inference
        rgb_t, depth_t = preprocess(rgb_path, depth_path, config)
        pred = predict(model, rgb_t, depth_t, device)

        # GT
        gt = load_gt(gt_path, config) if os.path.exists(gt_path) else np.zeros_like(pred)

        # Vizualizacije
        mask_rows.append(colorize_mask(gt))
        pred_rows.append(colorize_mask(pred))
        overlay_rows.append(make_overlay(gt, pred, cls=args.overlay_class))

        print(f'  OK: {name}')

    if not mask_rows:
        print('Nema obradjenih uzoraka!')
        return

    # Slozi svaku vrstu vertikalno
    mask_img    = stack_vertical(mask_rows,    pad=args.pad)
    pred_img    = stack_vertical(pred_rows,    pad=args.pad)
    overlay_img = stack_vertical(overlay_rows, pad=args.pad)

    # Spremi tri odvojene slike
    os.makedirs(args.output_dir, exist_ok=True)

    mask_path    = os.path.join(args.output_dir, f'{args.prefix}_mask.png')
    pred_path    = os.path.join(args.output_dir, f'{args.prefix}_pred.png')
    overlay_path = os.path.join(args.output_dir, f'{args.prefix}_overlay.png')

    cv2.imwrite(mask_path,    mask_img)
    cv2.imwrite(pred_path,    pred_img)
    cv2.imwrite(overlay_path, overlay_img)

    print(f'\n{"="*60}')
    print(f'  Spremljeno u {args.output_dir}/')
    print(f'{"="*60}')
    print(f'  {args.prefix}_mask.png     — GT maske ({len(mask_rows)} redaka)')
    print(f'  {args.prefix}_pred.png     — Predikcije')
    print(f'  {args.prefix}_overlay.png  — TP/FP/FN za klasu "{config.class_names[args.overlay_class]}"')
    print(f'\n  Boje maski:   crvena=deblo, plava=grane, zelena=potpora')
    print(f'  Boje overlay: zelena=TP, crvena=FP, plava=FN')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ciste vizualizacije DFormer predikcija za subfigure')
    parser.add_argument('--config',
                        default='local_configs.BranchDataset.DFormer_Base',
                        help='Config modul')
    parser.add_argument('--checkpoint',
                        default='checkpoints/BranchDataset_DFormer-Base_20260507-181200/epoch-165_miou_88.24.pth',
                        help='Putanja do .pth checkpointa')
    parser.add_argument('--n', type=int, default=6,
                        help='Broj slika iz testnog skupa (default: 6). Ignorira se ako je zadano --samples.')
    parser.add_argument('--samples', nargs='+', default=None,
                        help='Konkretni testni uzorci po imenu (bez ekstenzije). Primjer: --samples tree_1_V_0025_0 tree_1_V_0145_1')
    parser.add_argument('--overlay_class', type=int, default=2,
                        help='Klasa za TP/FP/FN overlay (0=ostalo, 1=deblo, 2=grane, 3=potpora). Default: 2 (grane)')
    parser.add_argument('--output_dir', default='./qualitative_dformer',
                        help='Direktorij u koji se spremaju tri slike')
    parser.add_argument('--prefix', default='dformer_base_qual',
                        help='Prefiks naziva izlaznih slika')
    parser.add_argument('--pad', type=int, default=6,
                        help='Broj crnih piksela izmedju redaka (default: 6)')
    args = parser.parse_args()

    run(args)
