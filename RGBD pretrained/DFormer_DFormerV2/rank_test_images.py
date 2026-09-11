"""
rank_test_images.py
────────────────────────────────────────────────────────────────────────────
Prolazi cijeli testni skup i za svaki uzorak izracunava per-image mIoU
i IoU po klasama za DFormer model. Sortira uzorke od najboljeg do najgoreg
i identificira BEST i WORST slucajeve.

Opcionalno generira kvalitativne vizualizacije za odabrane best/worst
uzorke, u istom formatu kao visualize_dformer_clean.py (3 odvojene slike
mask/pred/overlay bez teksta, spremne za subfigure u LaTeX-u).

NAPOMENA: Ova skripta koristi single-scale inferenciju (bez MST) radi
brzine. To znaci da apsolutne brojke per-image nece biti identicne onima
iz utils/eval.py sa MST, ali RELATIVNI poredak (koji je bolji, koji je
losiji) ostaje jednak.

Usage:
  # Samo rangiranje (spremi u CSV) - default
  python rank_test_images.py

  # Rangiranje + odmah generiraj best/worst slike (3 top + 3 bot)
  python rank_test_images.py --generate_visuals --n_best 3 --n_worst 3

  # Sortiraj po IoU za problematicnu klasu potpore
  python rank_test_images.py --metric iou_potpora
"""

import argparse
import os
from importlib import import_module

import cv2
import numpy as np
import torch
import torch.nn as nn


# ── Boje ────────────────────────────────────────────────────────────────────
CLASS_COLORS_BGR = [
    (0,   0,   0),    # 0 ostalo   — crna
    (0,   0,   255),  # 1 deblo    — crvena
    (255, 0,   0),    # 2 grane    — plava
    (0,   255, 0),    # 3 potpora  — zelena
]
TP_COLOR = (0,   255, 0)
FP_COLOR = (0,   0,   255)
FN_COLOR = (255, 0,   0)

CLASS_NAMES = ['ostalo', 'deblo', 'grane', 'potpora']


# ── Pomocne funkcije ────────────────────────────────────────────────────────

def colorize_mask(mask):
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, color in enumerate(CLASS_COLORS_BGR):
        out[mask == cls_idx] = color
    return out


def make_overlay(gt, pred, cls=2):
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
    model.load_state_dict(weight, strict=False)
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


def compute_per_image_metrics(pred, gt, num_classes=4):
    """Izracunaj per-image IoU po klasi + agregatne metrike."""
    ious = []
    present = []
    for c in range(num_classes):
        gt_c   = (gt   == c)
        pred_c = (pred == c)
        inter = np.logical_and(gt_c, pred_c).sum()
        union = np.logical_or(gt_c,  pred_c).sum()
        present.append(bool(gt_c.sum() > 0))
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(float(inter) / float(union))

    ious_arr = np.array(ious)
    miou    = float(np.nanmean(ious_arr))        # svih klasa
    miou_fg = float(np.nanmean(ious_arr[1:]))    # samo foreground
    acc     = float((pred == gt).sum()) / float(gt.size)

    return {
        'ious':    ious,
        'present': present,
        'miou':    miou,
        'miou_fg': miou_fg,
        'acc':     acc,
    }


def stack_vertical(images, pad=6):
    if not images:
        return None
    w = images[0].shape[1]
    pad_row = np.zeros((pad, w, 3), dtype=np.uint8)
    stacked = images[0]
    for img in images[1:]:
        stacked = np.concatenate([stacked, pad_row, img], axis=0)
    return stacked


def generate_subfigure_images(names, config, model, device,
                              output_dir, prefix, overlay_class):
    """Za listu uzoraka generira 3 odvojene stupne slike (mask/pred/overlay)."""
    mask_rows, pred_rows, overlay_rows = [], [], []

    for name in names:
        rgb_path   = os.path.join(config.rgb_root_folder, name + config.rgb_format)
        depth_path = os.path.join(config.x_root_folder,   name + config.x_format)
        gt_path    = os.path.join(config.gt_root_folder,  name + config.gt_format)

        rgb_t, depth_t = preprocess(rgb_path, depth_path, config)
        pred = predict(model, rgb_t, depth_t, device)
        gt   = load_gt(gt_path, config) if os.path.exists(gt_path) else np.zeros_like(pred)

        mask_rows.append(colorize_mask(gt))
        pred_rows.append(colorize_mask(pred))
        overlay_rows.append(make_overlay(gt, pred, cls=overlay_class))

    os.makedirs(output_dir, exist_ok=True)
    for kind, imgs in [('mask', mask_rows), ('pred', pred_rows), ('overlay', overlay_rows)]:
        stacked = stack_vertical(imgs)
        path = os.path.join(output_dir, f'{prefix}_{kind}.png')
        cv2.imwrite(path, stacked)
        print(f'  Sacuvano: {path}')


def _fmt(v):
    if isinstance(v, float):
        if np.isnan(v):
            return 'NaN'
        return f'{v:.4f}'
    return str(v)


def _print_ranking_row(rank, r, metric_key):
    name = r['name']
    metric_val = r[metric_key]
    metric_str = f'{metric_val:.4f}' if not np.isnan(metric_val) else 'NaN'
    print(f'  #{rank:>2}  {name:55s}  '
          f'{metric_key}={metric_str}  |  '
          f'deblo={_fmt(r["iou_deblo"])}  '
          f'grane={_fmt(r["iou_grane"])}  '
          f'potpora={_fmt(r["iou_potpora"])}')


def run(args):
    config = getattr(import_module(args.config), 'C')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Uredaj: {device}')

    model = load_model(config, args.checkpoint, device)
    print(f'Model ucitan iz: {args.checkpoint}')

    with open(config.eval_source) as f:
        all_names = [line.strip() for line in f if line.strip()]

    print(f'\nRacunam per-image metrike za {len(all_names)} testnih uzoraka...')

    results = []
    for i, name in enumerate(all_names):
        rgb_path   = os.path.join(config.rgb_root_folder, name + config.rgb_format)
        depth_path = os.path.join(config.x_root_folder,   name + config.x_format)
        gt_path    = os.path.join(config.gt_root_folder,  name + config.gt_format)

        if not os.path.exists(rgb_path) or not os.path.exists(gt_path):
            print(f'  Preskacem (nedostaje datoteka): {name}')
            continue

        rgb_t, depth_t = preprocess(rgb_path, depth_path, config)
        pred = predict(model, rgb_t, depth_t, device)
        gt   = load_gt(gt_path, config)

        m = compute_per_image_metrics(pred, gt)
        results.append({
            'name':            name,
            'miou':            m['miou'],
            'miou_fg':         m['miou_fg'],
            'acc':             m['acc'],
            'iou_ostalo':      m['ious'][0],
            'iou_deblo':       m['ious'][1],
            'iou_grane':       m['ious'][2],
            'iou_potpora':     m['ious'][3],
            'present_deblo':   m['present'][1],
            'present_grane':   m['present'][2],
            'present_potpora': m['present'][3],
        })

        if (i + 1) % 10 == 0 or i == len(all_names) - 1:
            print(f'  {i + 1}/{len(all_names)}...')

    if not results:
        print('Nema obradjenih uzoraka!')
        return

    # Filtriraj slucajeve gdje metrika nije definirana (NaN)
    metric_key = args.metric
    valid_results = [r for r in results if not np.isnan(r[metric_key])]
    invalid_count = len(results) - len(valid_results)
    if invalid_count > 0:
        print(f'\n  Napomena: {invalid_count} uzoraka ima NaN za "{metric_key}" - iskljuceni iz rangiranja')

    # Sortiraj od najgoreg (najniza metrika) do najboljeg (najvisa)
    valid_results.sort(key=lambda x: x[metric_key])

    # Spremi CSV
    if args.output_csv:
        header = ['rank', 'name', 'miou', 'miou_fg', 'acc',
                  'iou_ostalo', 'iou_deblo', 'iou_grane', 'iou_potpora',
                  'present_deblo', 'present_grane', 'present_potpora']
        with open(args.output_csv, 'w') as f:
            f.write(','.join(header) + '\n')
            # Rang 1 = najbolji, zato reversamo pri pisanju
            for rank, r in enumerate(reversed(valid_results), 1):
                row = [str(rank), r['name']]
                for col in header[2:]:
                    v = r[col]
                    if isinstance(v, bool):
                        row.append('1' if v else '0')
                    elif isinstance(v, float):
                        row.append(f'{v:.6f}' if not np.isnan(v) else 'NaN')
                    else:
                        row.append(str(v))
                f.write(','.join(row) + '\n')
        print(f'\nCSV rangiranje spremljeno: {args.output_csv}')

    # Ispisi najbolje i najgore
    print(f'\n{"=" * 100}')
    print(f'  TOP {args.n_best} NAJBOLJIH  (rangirano po "{metric_key}")')
    print(f'{"=" * 100}')
    best = list(reversed(valid_results[-args.n_best:]))  # od najboljeg
    for i, r in enumerate(best, 1):
        _print_ranking_row(i, r, metric_key)

    print(f'\n{"=" * 100}')
    print(f'  TOP {args.n_worst} NAJGORIH  (rangirano po "{metric_key}")')
    print(f'{"=" * 100}')
    worst = valid_results[:args.n_worst]  # od najgoreg
    for i, r in enumerate(worst, 1):
        _print_ranking_row(i, r, metric_key)

    # Opcionalno generiraj vizualizacije
    if args.generate_visuals:
        best_names  = [r['name'] for r in best]
        worst_names = [r['name'] for r in worst]

        print(f'\n{"=" * 100}')
        print(f'  Generiram BEST vizualizacije...')
        print(f'{"=" * 100}')
        generate_subfigure_images(best_names, config, model, device,
                                   args.output_dir, f'{args.prefix}_best',
                                   args.overlay_class)

        print(f'\n{"=" * 100}')
        print(f'  Generiram WORST vizualizacije...')
        print(f'{"=" * 100}')
        generate_subfigure_images(worst_names, config, model, device,
                                   args.output_dir, f'{args.prefix}_worst',
                                   args.overlay_class)

        print(f'\n  Boje maski:   crvena=deblo, plava=grane, zelena=potpora')
        print(f'  Boje overlay: zelena=TP, crvena=FP, plava=FN')

    print(f'\nGotovo.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Rangiraj testne slike po per-image metrici i identificiraj best/worst.'
    )
    parser.add_argument('--config',
                        default='local_configs.BranchDataset.DFormer_Base',
                        help='Config modul')
    parser.add_argument('--checkpoint',
                        default='checkpoints/BranchDataset_DFormer-Base_20260507-181200/epoch-165_miou_88.24.pth',
                        help='Putanja do .pth checkpointa')
    parser.add_argument('--metric', default='miou_fg',
                        choices=['miou', 'miou_fg', 'iou_deblo', 'iou_grane', 'iou_potpora', 'acc'],
                        help='Metrika za rangiranje. Default: miou_fg (deblo+grane+potpora)')
    parser.add_argument('--n_best',  type=int, default=3,
                        help='Broj najboljih uzoraka za prikaz/vizualizaciju')
    parser.add_argument('--n_worst', type=int, default=3,
                        help='Broj najgorih uzoraka za prikaz/vizualizaciju')
    parser.add_argument('--output_csv', default='./test_ranking.csv',
                        help='Putanja za CSV rangiranja svih uzoraka')
    parser.add_argument('--generate_visuals', action='store_true',
                        help='Ako je zadano, generira mask/pred/overlay slike za best i worst')
    parser.add_argument('--output_dir', default='./best_worst_visuals',
                        help='Direktorij za vizualizacije')
    parser.add_argument('--prefix', default='dformer_base',
                        help='Prefiks naziva izlaznih slika')
    parser.add_argument('--overlay_class', type=int, default=2,
                        help='Klasa za TP/FP/FN overlay (0=ostalo, 1=deblo, 2=grane, 3=potpora)')
    args = parser.parse_args()

    run(args)
