# Eksperiment 2 — RGB + jednokanalna dubina (rana fuzija), ImageNet pretrenirani modeli

Odgovara poglavlju 5.2 rada (tablica 5.2, slike 5.1/5.2). 7 arhitektura 
(U-Net ResNet-34/EfficientNet-B4, DeepLabV3, DeepLabV3+, SegFormer-B1, 
MA-Net, ESANet), 657 uzoraka BRANCH v2, dubina spojena kao 4. ulazni kanal.

`reference_results/comparison_table.csv` = Tablica 5.2 u radu.

## Okruženje

Sve potrebne biblioteke su u `requirements.txt`. torch/torchvision instalirati 
s `--index-url https://download.pytorch.org/whl/cu121` (verzije nose `+cu121` 
sufiks, nisu na standardnom PyPI-ju).

## Priprema podataka


python prepare_data.py --anotirane <put_do_anotiranih_PLY_oblaka> --out_dir data/


**Obavezno proslijediti oba argumenta** — defaultne vrijednosti su moje lokalne 
putanje.

Očekivana struktura nakon pripreme:

data/
train/images/_rgbd.npy train/masks/_mask.png
val/images/_rgbd.npy val/masks/_mask.png
test/images/_rgbd.npy test/masks/_mask.png


Provjera broja uzoraka/distribucije klasa 

python datasetstats.py --data_root data/


## Trening i evaluacija

python main.py # svih 7 modela + evaluacija
python main.py --model unet_resnet34 # jedan model
python main.py --eval_only # samo evaluacija postojećih checkpointa


`--model`: `unet_resnet34`, `unet_efficientnet_b4`, `deeplabv3_resnet50`, 
`deeplabv3plus_mobilenetv2`, `segformer_b1`, `manet_resnext50`, `esanet_resnet34`. 
(`pix2pix_gan`/`pix2pix_gen` ostali u kodu iz Eksperimenta 1, nisu dio 
Eksperimenta 2 u radu.)

`evaluate.py` traži checkpointove pod fiksnim imenima u `checkpoints/` 
(`{model}_best.pth`) i preskače model ako checkpoint ne postoji.

Izlaz:
- `results/comparison_table.csv` → Tablica 5.2
- `results/{model}_per_image.csv` → metrika po slici
- `results/{model}_qual.png` → kvalitativni prikaz
- `results/training_curves.png` → krivulje učenja

## Brzina inferencije

python benchmark_inference.py



## Kvalitativne vizualizacije (Slike 5.1, 5.2)

python visualize_mask.py
bash run_all_vis.sh # svi modeli


## `reference_results/`

Finalni logs/results/overlays iz treninga korištenog u radu, bez checkpointa 
(prevelik za git).

## Implementacijske napomene

- `augmentations.py`: `additional_targets={"depth": "image"}` znači da 
  `RandomBrightnessContrast`/`GaussianBlur`/`CLAHE` trenutno cure i na depth 
  kanal, ne samo RGB. Za razdvajanje treba eksplicitno filtrirati transformacije 
  po tipu, albumentations to ne radi automatski.
- Inicijalizacija 4. kanala kod smp-baziranih modela (U-Net/DeepLabV3/DeepLabV3+/MA-Net) 
  ide preko `patch_first_conv()` iz `segmentation_models_pytorch` — ciklički 
  kopira RGB težine i skalira cijeli tenzor s 3/N, ne prosjekuje. ESANet i 
  SegFormer imaju ručnu inicijalizaciju (`models/esanet.py`, `models/segformer.py`).
