import os
import numpy as np
from PIL import Image
from pathlib import Path

src_base = "/home/samuel/Desktop/Pokusaj_Chen_MultiDepthV2/data"
dst_base = "/home/samuel/Desktop/DFormer/datasets/BranchDataset"

os.makedirs(f"{dst_base}/RGB", exist_ok=True)
os.makedirs(f"{dst_base}/Depth", exist_ok=True)
os.makedirs(f"{dst_base}/Label", exist_ok=True)

for split in ["train", "val", "test"]:
    img_dir  = f"{src_base}/{split}/images"
    mask_dir = f"{src_base}/{split}/masks"
    names = []

    for f in sorted(os.listdir(img_dir)):
        if not f.endswith("_rgbd.npy"):
            continue
        if "BACKUP" in f:
            continue

        stem = f.replace("_rgbd.npy", "")
        rgbd = np.load(os.path.join(img_dir, f))  # [H,W,4]
        mask_path = os.path.join(mask_dir, f"{stem}_mask.png")

        if not os.path.exists(mask_path):
            print(f"SKIP (no mask): {f}")
            continue

        # RGB — kanali 0,1,2 (float 0-1 → uint8)
        rgb = (rgbd[:, :, :3] * 255).astype(np.uint8)
        Image.fromarray(rgb).save(f"{dst_base}/RGB/{stem}.png")

        # Depth — kanal 3 (metar → 16bit mm)
        depth = (rgbd[:, :, 3] * 1000).astype(np.uint16)
        Image.fromarray(depth).save(f"{dst_base}/Depth/{stem}.png")

        # Label mask
        mask = Image.open(mask_path)
        mask.save(f"{dst_base}/Label/{stem}.png")

        names.append(stem)

    # Spremi txt listu
    txt_file = "test.txt" if split == "test" else f"{split}.txt"
    with open(f"{dst_base}/{txt_file}", "w") as f:
        f.write("\n".join(names))

    print(f"{split}: {len(names)} slika")

print("Gotovo!")