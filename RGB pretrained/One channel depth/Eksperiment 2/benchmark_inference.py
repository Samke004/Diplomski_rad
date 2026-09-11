"""
benchmark_inference.py
────────────────────────────────────────────────────────────────────────────
Mjeri inference vrijeme (obrada JEDNE slike) za sve segmentacijske modele
u ovom projektu (UNet, DeepLabV3, DeepLabV3+, SegFormer, MANet, ESANet,
Pix2Pix) i zapisuje rezultate u CSV.

Pokretanje:
  cd ~/Desktop/Pokusaj_Chen_MultiDepthV2
  python3 benchmark_inference.py

Rezultat: results/inference_results.csv (jedan red po modelu)

Kako radi:
  1. Učita testni set JEDNOM (isti transform/resize kao evaluate.py) i drži
     ga u memoriji, dijeli se za sve modele (izbjegava ponovno čitanje s
     diska za svaki model).
  2. Za svaki model: učita checkpoint, odradi N_WARMUP inferencija koje se
     NE broje (prva/prve inferencije na GPU-u su sporije zbog CUDA/cuDNN
     inicijalizacije), zatim mjeri svaku sljedeću inferenciju (batch=1,
     tj. "jedna slika") uz torch.cuda.synchronize() prije/poslije (bez
     toga bi mjerenje na GPU-u bilo pogrešno jer su CUDA pozivi asinkroni).
  3. Računa prosjek/std/min/max/FPS i dopisuje jedan red u CSV.

Checkpointi za koje ne postoji .pth fajl u checkpoints/ se automatski
preskaču (npr. pix2pix ako ga nisi trenirao).
"""

import os
import csv
import time
import statistics

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config as cfg
from dataset import AppleTreeDataset
from augmentations import get_val_transforms
from models.esanet import build_esanet
from models import (build_unet, build_deeplabv3, build_deeplabv3plus, build_manet,
                    build_segformer, Pix2PixGenerator)

# ── Podesivo ──────────────────────────────────────────────────────────────
N_WARMUP = 10       # broj zagrijavajucih inferencija koje se NE broje u rezultat
N_IMAGES = None     # None = sve slike iz test seta; ili npr. 50 za brze mjerenje
OUT_CSV  = os.path.join(cfg.RESULTS_DIR, "inference_results.csv")


def _load_model(model, ckpt_path, device, key="model_state"):
    if not os.path.isfile(ckpt_path):
        print(f"  [SKIP] Checkpoint not found: {ckpt_path}")
        return None
    state = torch.load(ckpt_path, map_location=device)
    actual_key = key if key in state else "G_state"
    model.load_state_dict(state[actual_key])
    model.eval()
    return model.to(device)


@torch.no_grad()
def _timed_inference(model, image, device) -> float:
    """Odradi jednu inferenciju i vrati proteklo vrijeme u sekundama."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    image = image.to(device, non_blocking=True)
    _ = model(image)

    if device.type == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()

    return end - start


def _append_to_csv(csv_path, row, fieldnames):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def benchmark_model(model_name, model, ckpt_path, state_key, images, device):
    print(f"\n── Benchmarking: {model_name} ──")
    model = _load_model(model, ckpt_path, device, key=state_key)
    if model is None:
        return

    if len(images) <= N_WARMUP:
        print(f"  Nedovoljno slika za warmup+mjerenje ({len(images)} dostupno).")
        return

    n_warmup = min(N_WARMUP, len(images))
    print(f"  Zagrijavanje ({n_warmup} slika, ne broji se u rezultat)...")
    for img in images[:n_warmup]:
        _timed_inference(model, img, device)

    measure_images = images[n_warmup:]
    print(f"  Mjerenje na {len(measure_images)} slika (batch=1, jedna slika po mjerenju)...")
    times = []
    for img in tqdm(measure_images, desc="  Inference"):
        elapsed = _timed_inference(model, img, device)
        times.append(elapsed)

    times_ms = [t * 1000 for t in times]
    avg_ms = statistics.mean(times_ms)
    std_ms = statistics.pstdev(times_ms) if len(times_ms) > 1 else 0.0
    min_ms = min(times_ms)
    max_ms = max(times_ms)
    fps = 1000.0 / avg_ms if avg_ms > 0 else float("nan")

    print(f"  Prosječno vrijeme inferencije: {avg_ms:.2f} ms "
          f"(std: {std_ms:.2f}, min: {min_ms:.2f}, max: {max_ms:.2f})")
    print(f"  FPS (slika/s): {fps:.2f}")

    row = {
        "model": model_name,
        "device": str(device),
        "num_images": len(times_ms),
        "warmup_images": n_warmup,
        "avg_inference_time_ms": round(avg_ms, 4),
        "std_inference_time_ms": round(std_ms, 4),
        "min_inference_time_ms": round(min_ms, 4),
        "max_inference_time_ms": round(max_ms, 4),
        "fps": round(fps, 4),
    }
    _append_to_csv(OUT_CSV, row, list(row.keys()))
    print(f"  Zapisano u: {OUT_CSV}")


def main():
    device = torch.device(cfg.DEVICE)

    test_dataset = AppleTreeDataset(
        img_dir   = cfg.TEST_IMG_DIR,
        mask_dir  = cfg.TEST_MASK_DIR,
        transform = get_val_transforms(cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH),
        img_size  = (cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH),
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             num_workers=cfg.NUM_WORKERS, pin_memory=False)
    print(f"Test set: {len(test_dataset)} images")

    # Preprocesiraj sve slike JEDNOM i drži u memoriji - dijeli se za sve
    # modele umjesto da se test set ponovno cita s diska za svaku arhitekturu.
    print("Učitavanje i preprocesiranje test slika...")
    images = []
    for batch in test_loader:
        images.append(batch["image"])
        if N_IMAGES is not None and len(images) >= (N_IMAGES + N_WARMUP):
            break
    print(f"Pripremljeno {len(images)} slika za mjerenje.")

    # obrisi stari CSV da krenemo iz cista
    if os.path.exists(OUT_CSV):
        os.remove(OUT_CSV)

    model_configs = [
        ("unet_resnet34",
         build_unet("resnet34", weights=None,
                    in_channels=cfg.IN_CHANNELS, num_classes=cfg.NUM_CLASSES),
         os.path.join(cfg.CHECKPOINT_DIR, "unet_resnet34_best.pth"), "model_state"),

        ("unet_efficientnet_b4",
         build_unet("efficientnet-b4", weights=None,
                    in_channels=cfg.IN_CHANNELS, num_classes=cfg.NUM_CLASSES),
         os.path.join(cfg.CHECKPOINT_DIR, "unet_efficientnet_b4_best.pth"), "model_state"),

        ("deeplabv3_resnet50",
         build_deeplabv3(pretrained=False,
                         in_channels=cfg.IN_CHANNELS, num_classes=cfg.NUM_CLASSES),
         os.path.join(cfg.CHECKPOINT_DIR, "deeplabv3_resnet50_best.pth"), "model_state"),

        ("segformer_b1",
         build_segformer(variant=cfg.SEGFORMER_VARIANT, weights=None,
                         in_channels=cfg.IN_CHANNELS, num_classes=cfg.NUM_CLASSES),
         os.path.join(cfg.CHECKPOINT_DIR, "segformer_b1_best.pth"), "model_state"),

        ("manet_resnext50",
         build_manet(pretrained=False,
                     in_channels=cfg.IN_CHANNELS, num_classes=cfg.NUM_CLASSES),
         os.path.join(cfg.CHECKPOINT_DIR, "manet_resnext50_best.pth"), "model_state"),

        ("deeplabv3plus_mobilenetv2",
         build_deeplabv3plus(encoder="mobilenet_v2", weights=None,
                             in_channels=cfg.IN_CHANNELS, num_classes=cfg.NUM_CLASSES),
         os.path.join(cfg.CHECKPOINT_DIR, "deeplabv3plus_mobilenetv2_best.pth"), "model_state"),

        ("esanet_resnet34",
         build_esanet(pretrained=False, in_channels=cfg.IN_CHANNELS, num_classes=cfg.NUM_CLASSES),
         os.path.join(cfg.CHECKPOINT_DIR, "esanet_resnet34_best.pth"), "model_state"),

        ("pix2pix_gan",
         Pix2PixGenerator(in_channels=cfg.IN_CHANNELS, out_channels=cfg.NUM_CLASSES),
         os.path.join(cfg.CHECKPOINT_DIR, "pix2pix_gan_best.pth"), "G_state"),

        ("pix2pix_gen",
         Pix2PixGenerator(in_channels=cfg.IN_CHANNELS, out_channels=cfg.NUM_CLASSES),
         os.path.join(cfg.CHECKPOINT_DIR, "pix2pix_gen_best.pth"), "G_state"),
    ]

    for model_name, model, ckpt_path, state_key in model_configs:
        benchmark_model(model_name, model, ckpt_path, state_key, images, device)

    print(f"\n=== SVE MJERENJE INFERENCIJE GOTOVO === Rezultati u {OUT_CSV}")


if __name__ == "__main__":
    main()