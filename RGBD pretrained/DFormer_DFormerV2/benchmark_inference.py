"""
Mjerenje inference vremena DFormer modela (vrijeme obrade jedne slike).

Pokretanje (jedan model):
  cd ~/Downloads/DFormerv1/DFormer
  python3 benchmark_inference.py \
      --config local_configs.BranchDataset.DFormer_Base \
      --checkpoint checkpoints/BranchDataset_DFormer-Base_20260507-181200/epoch-165_miou_88.24.pth \
      --model_name DFormer_Base \
      --output results_inference.csv

Skripta:
  1. Učita model i checkpoint (isto kao visualize_dformer.py).
  2. Preprocesira sve slike iz eval_source unaprijed (da I/O ne uđe u mjerenje).
  3. Odradi N "zagrijavajućih" (warmup) inferencija koje se NE broje
     (prva/prve inferencije na GPU-u su sporije zbog CUDA/cuDNN inicijalizacije).
  4. Mjeri vrijeme svake sljedeće inferencije (batch=1, tj. "jedna slika"),
     uz torch.cuda.synchronize() prije/poslije da mjerenje bude točno na GPU-u.
  5. Računa prosjek/std/min/max/FPS i dopisuje jedan red u CSV.
"""

import argparse
import csv
import os
import statistics
import time
from importlib import import_module

import cv2
import numpy as np
import torch
import torch.nn as nn


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
    """Učitaj i preprocess RGB + Depth sliku za model (rezultat ostaje na CPU-u)."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

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


def timed_inference(model, rgb_cpu, depth_cpu, device, use_amp) -> float:
    """
    Odradi jednu inferenciju i vrati proteklo vrijeme u sekundama.
    Mjerenje uključuje prijenos podataka na GPU (to je dio stvarne obrade "jedne slike").
    """
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    rgb_t = rgb_cpu.to(device, non_blocking=True)
    depth_t = depth_cpu.to(device, non_blocking=True)

    with torch.no_grad():
        if use_amp and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _ = model(rgb_t, depth_t)
        else:
            _ = model(rgb_t, depth_t)

    if device.type == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()

    return end - start


def append_to_csv(csv_path: str, row: dict, fieldnames: list):
    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Mjerenje inference vremena DFormer modela")
    parser.add_argument("--config", required=True,
                         help="Config modul (npr. local_configs.BranchDataset.DFormer_Base)")
    parser.add_argument("--checkpoint", required=True,
                         help="Putanja do checkpoint .pth fajla")
    parser.add_argument("--model_name", default=None,
                         help="Naziv modela za CSV (default: zadnji dio --config)")
    parser.add_argument("--n", type=int, default=None,
                         help="Broj slika za mjerenje (default: sve iz eval_source)")
    parser.add_argument("--warmup", type=int, default=10,
                         help="Broj zagrijavajućih inferencija koje se ne broje (default: 10)")
    parser.add_argument("--amp", default=True, action=argparse.BooleanOptionalAction,
                         help="Koristi mixed precision (autocast), isto kao u eval.py (default: uključeno)")
    parser.add_argument("--output", default="results_inference.csv",
                         help="Izlazni CSV (default: results_inference.csv)")
    args = parser.parse_args()

    config = getattr(import_module(args.config), "C")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = args.model_name or args.config.split(".")[-1]

    model = load_model(config, args.checkpoint, device)

    eval_source = config.eval_source
    rgb_root = config.rgb_root_folder
    depth_root = config.x_root_folder
    rgb_ext = config.rgb_format
    depth_ext = config.x_format

    with open(eval_source) as f:
        names = [line.strip() for line in f if line.strip()]

    if args.n is not None:
        names = names[: args.n]

    # Preprocesiraj sve slike unaprijed da I/O/decode ne uđe u mjerenje inferencije.
    pairs = []
    for name in names:
        rgb_path = os.path.join(rgb_root, name + rgb_ext)
        depth_path = os.path.join(depth_root, name + depth_ext)
        if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
            print(f"Preskačem {name}, nedostaje RGB ili Depth.")
            continue
        pairs.append(preprocess(rgb_path, depth_path, config))

    if not pairs:
        print("Nema slika za mjerenje!")
        return

    n_warmup = min(args.warmup, len(pairs))
    print(f"Zagrijavanje ({n_warmup} slika, ne broji se u rezultat)...")
    for rgb_t, depth_t in pairs[:n_warmup]:
        timed_inference(model, rgb_t, depth_t, device, args.amp)

    print(f"Mjerenje inferencije na {len(pairs)} slika (batch=1, jedna slika po mjerenju)...")
    times = []
    for rgb_t, depth_t in pairs:
        elapsed = timed_inference(model, rgb_t, depth_t, device, args.amp)
        times.append(elapsed)

    times_ms = [t * 1000 for t in times]
    avg_ms = statistics.mean(times_ms)
    std_ms = statistics.pstdev(times_ms) if len(times_ms) > 1 else 0.0
    min_ms = min(times_ms)
    max_ms = max(times_ms)
    fps = 1000.0 / avg_ms if avg_ms > 0 else float("nan")

    print(f"Prosječno vrijeme inferencije: {avg_ms:.2f} ms "
          f"(std: {std_ms:.2f}, min: {min_ms:.2f}, max: {max_ms:.2f})")
    print(f"FPS (slika/s): {fps:.2f}")

    row = {
        "model": model_name,
        "device": str(device),
        "num_images": len(times_ms),
        "warmup_images": n_warmup,
        "amp": args.amp,
        "avg_inference_time_ms": round(avg_ms, 4),
        "std_inference_time_ms": round(std_ms, 4),
        "min_inference_time_ms": round(min_ms, 4),
        "max_inference_time_ms": round(max_ms, 4),
        "fps": round(fps, 4),
    }
    fieldnames = list(row.keys())
    append_to_csv(args.output, row, fieldnames)
    print(f"Zapisano u: {args.output}")


if __name__ == "__main__":
    main()
