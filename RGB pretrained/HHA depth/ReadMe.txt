# Eksperiment 3 — RGB + HHA enkodiranje dubine (6 kanala)

Odgovara poglavlju 5.3 rada (tablica 5.5, poglavlje 5.3.1 — negativan nalaz o 
HHA enkodiranju). 5 arhitektura (U-Net ResNet-34/EfficientNet-B4, DeepLabV3, 
DeepLabV3+, SegFormer-B1), isti 657-uzorkovni split kao Eksperiment 2, dubina 
zamijenjena trokanalnim HHA enkodiranjem (RGB + HHA = 6 kanala).

`reference_results/comparison_table.csv` = Tablica 5.5 u radu.

## Okruženje

Isti `requirements.txt` kao Eksperiment 1/2 (Python 3.8, torch 2.1.0+cu121, 
segmentation-models-pytorch 0.3.3, transformers 4.46.3...).

**Dodatna ovisnost — nije pip paket, treba ručno klonirati:**

git clone https://github.com/charlesCXK/Depth2HHA-python

`convert_to_hha.py` trenutno ima **hardkodiranu putanju u kodu** 
(`sys.path.insert(0, "/home/samuel/Desktop/DiplomksiHHA/Depth2HHA-python")`) 
— tu liniju treba ručno izmijeniti da pokazuje na vaš lokalni klon prije 
pokretanja.

## Priprema podataka

Koristi **isti** train/val/test split kao Eksperiment 2 — pripremite `data/` 
prema uputama za Eksperiment 2 (`prepare_data.py`), zatim generirajte HHA 
kanale iz postojećih `_rgbd.npy` datoteka:

python convert_to_hha.py


Ovo za svaki `*_rgbd.npy` generira odgovarajući `*_hha.npy` (3-kanalni, 
normaliziran u [0,1]) u istom folderu. Kamera parametri (FX=FY=570.34, 
CX=314.5, CY=235.5) hardkodirani su u skripti — odgovaraju Asus Xtion Pro 
senzoru korištenom pri snimanju.


## Trening i evaluacija

python main.py # svih 5 modela + evaluacija
python main.py --model unet_resnet34 # jedan model
python main.py --eval_only


`--model`: `unet_resnet34`, `unet_efficientnet_b4`, `deeplabv3_resnet50`, 
`deeplabv3plus_mobilenetv2`, `segformer_b1`.

Izlaz:
- `results_hha/comparison_table.csv` → Tablica 5.5
- `results_hha/{model}_per_image.csv`, `{model}_qual.png`, `training_curves.png`
- `checkpoints_hha/{model}_best.pth`
- `logs/{model}_log.csv`, `{model}_info.txt`

## ESANet — pokušano, isključeno iz rada

`models/esanet.py` u ovom folderu sadrži modificiranu verziju koja prihvaća 
6-kanalni HHA ulaz (za razliku od Eksperimenta 2, gdje ESANet očekuje isključivo 
1-kanalnu dubinu). Pokušaj treniranja rezultirao je kolapsom modela na 
predviđanje isključivo pozadinske klase (mIoU ≈ 0,32%, `recall_background ≈ 1.0` 
na gotovo svim testnim slikama) — vidljivo u `results_hha/esanet_resnet34_per_image.csv`. 
Ovo nije nužno arhitekturalno ograničenje, nego trening koji nije konvergirao 
(mogući uzrok: neuparen learning rate ili SE-fuzijska neravnoteža kad depth 
grana odjednom prima puni 3-kanalni pretrenirani ResNet-18 umjesto prosječenih 
1-kanalnih težina). Kod je ostavljen radi transparentnosti i kao polazna 
točka ako se netko odluči ovo dalje istražiti.

## Implementacijske napomene

- **Asimetrična normalizacija RGB naspram HHA kanala** — dodatna varijabla za 
  buduće istraživanje, istog statusa kao četiri objašnjenja u poglavlju 5.3.1 
  rada, ali dosad neisprobana: `dataset.py` normalizira HHA kanale dijeljenjem 
  s 255, dok RGB kanali prolaze ImageNet mean/std normalizaciju. Budući da 
  prva konvolucija dijeli ImageNet-pretrenirane težine preko oba tipa kanala, 
  ovaj mismatch u rasponu vrijednosti mogao bi biti dio objašnjenja za 
  degradaciju iz 5.3.1 — nije testirano, ali je jednostavno provjeriti: 
  primijeniti ImageNet mean/std i na HHA kanale i ponoviti Eksperiment 3.
- **Redoslijed kanala u `_hha.npy`**: `getHHA()` iz Depth2HHA-python vraća 
  kanale u redoslijedu [kut prema gravitaciji, visina, disparitet] (BGR 
  konvencija iz izvorne implementacije), što se ovdje sprema bez zamjene 
  redoslijeda. Ne utječe na treniranje (mreža uči proizvoljan raspored ulaznih 
  kanala), ali vrijedi imati na umu ako se HHA kanali ikad vizualiziraju ili 
  interpretiraju pojedinačno.
- **Isti mehanizam inicijalizacije dodatnih kanala** kao Eksperiment 2 
  (`patch_first_conv` za smp-modele, ručna Kaiming inicijalizacija za SegFormer) 
  — ovdje se skalira faktorom 3/6 = 0,5 umjesto 3/4.
