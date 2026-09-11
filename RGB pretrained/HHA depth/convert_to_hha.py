import os, sys, numpy as np
from pathlib import Path
from tqdm import tqdm
sys.path.insert(0, "/home/samuel/Desktop/DiplomksiHHA/Depth2HHA-python")
from getHHA import getHHA

FX = FY = 570.3422241210938
CX, CY = 314.5, 235.5
CAMERA_MATRIX = np.array([
    [FX,  0, CX],
    [ 0, FY, CY],
    [ 0,  0,  1],
], dtype=np.float64)

DATA_ROOT = "/home/samuel/Desktop/Pokusaj_Chen_HHA/data"
SPLITS = ["train", "val", "test"]

def process_split(split):
    img_dir = os.path.join(DATA_ROOT, split, "images")
    os.makedirs(img_dir, exist_ok=True)

    npy_files = sorted(Path(img_dir).glob("*_rgbd.npy"))
    print(f"[{split}] {len(npy_files)} fajlova → {img_dir}")

    ok, failed = 0, 0
    for npy_path in tqdm(npy_files, desc=f"  {split}"):
        hha_path = str(npy_path).replace("_rgbd.npy", "_hha.npy")
        if os.path.exists(hha_path):
            ok += 1
            continue
        try:
            rgbd  = np.load(str(npy_path))
            depth = rgbd[:, :, 3].astype(np.float64)  # metri
            depth[depth <= 0] = 0
            hha = getHHA(CAMERA_MATRIX, depth, depth)  # ispravno, bez *1000
            np.save(hha_path, hha.astype(np.float32))
            ok += 1
        except Exception as e:
            print(f"  FAIL: {npy_path.name} — {e}")
            failed += 1
    print(f"  OK: {ok} | Failed: {failed}")

for split in SPLITS:
    process_split(split)
print("Gotovo!")