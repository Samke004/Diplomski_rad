=============================================================
  SEGMENTACIJA STABALA JABUKE — RGBD PIPELINE
=============================================================

PREGLED PROJEKTA
----------------

DATASET
-------
Ručno anotirane PLY datoteke dvaju anotatora:

  - Samuel  — 16 stabala, ~108 slika (Drvo0000-Drvo0015)       [Grupa A]
  - Valentin — 5 stabala, ~91 slika (tree_1_V_0179-0183)       [Grupa B]
  - Model    — ~200 slika, modelom generirane + ručno pregledane [Grupa C]

Format anotacija:
  Samuel:   NaN = neoznačeno; zasebna scalar_ polja po klasi
            (scalar_Deblo, scalar_Grane, scalar_Potpora, scalar_Ostalo)
  Valentin: cjelobrojna vrijednost u jednom scalar stupcu
            (0=Deblo, 1=Grane, 2=Potpora, NaN=Ostalo)

Podjela na razini stabla (bez data leakagea):
  TRAIN (~80%): Grupa C (sve) + 70% Grupe A + 70% Grupe B
  VAL   (~10%): 15% Grupe A + 15% Grupe B  -- samo ručne anotacije
  TEST  (~10%): 15% Grupe A + 15% Grupe B  -- samo ručne anotacije


ULAZNI FORMAT
-------------
  Slike : *_rgbd.npy  -- 4-kanalni NumPy array [H, W, 4]
                         (RGB normaliziran na [0,1] + dubina u metrima)
  Maske : *_mask.png  -- sivotonska slika [H, W], vrijednosti 0-3
                         (0=pozadina, 1=deblo, 2=grane, 3=potpora)
  Rezolucija: 640x640 piksela (resize iz originalnih 640x480)


STRUKTURA PROJEKTA
------------------
  config.py          -- Centralna konfiguracija
  dataset.py         -- PyTorch Dataset za RGBD + maske
  augmentations.py   -- Albumentations augmentacije
  losses.py          -- Funkcije gubitka
  metrics.py         -- Evaluacijske metrike
  trainer.py         -- Petlje za treniranje
  evaluate.py        -- Evaluacija, vizualizacija, CSV izvještaji
  main.py            -- Glavna skripta
  prepare_data.py    -- Priprema i podjela dataseta
  dataset_report.py  -- Izvještaj i validacija PLY datoteka
  models/
    unet.py, deeplabv3.py, deeplabv3plus.py,
    segformer.py, manet.py, esanet.py, pix2pix.py


MODELI
------
  U-Net          ResNet-34       ImageNet pretrained
  U-Net          EfficientNet-B4 Testirano s batch=2 i batch=4
  DeepLabV3      ResNet-50
  DeepLabV3+     MobileNetV2     Lagani model
  SegFormer      B1              Transformer arhitektura
  MANet          ResNeXt-50      Multi-scale attention
  ESANet         ResNet-34       RGB-D specijalizirana mreža
  Pix2Pix GAN                   Generator + Diskriminator
  Pix2Pix Gen                   Samo generator (bez GAN gubitka)


EKSPERIMENTI
------------
  Eksperiment 1 -- bez augmentacija
    Treniranje samo s Resize.
    Rezultati: checkpoints_exp1/, results_exp1/

  Eksperiment 2 -- s augmentacijama
    Geometrijske transformacije primjenjuju se identično na
    RGB, depth i masku kako bi se spriječila neusklađenost kanala.

    Geometrijske (RGB + Depth + Maska):
      HorizontalFlip    p=0.5
      VerticalFlip      p=0.2
      RandomRotate90    p=0.5
      ShiftScaleRotate  p=0.5
      ElasticTransform  p=0.3
      GridDistortion    p=0.2

    Fotometrijske (samo RGB):
      RandomBrightnessContrast  p=0.5
      GaussianBlur              p=0.2
      CLAHE                     p=0.2

    Rezultati: checkpoints_exp2/, results_exp2/


FUNKCIJE GUBITKA
----------------
  SegmentationTrainer : CrossEntropy + Dice (kombinirani, težinski)
  Pix2PixTrainer GAN  : Adversarial (BCEWithLogits) + lambda * CrossEntropy
  Pix2PixTrainer Gen  : samo CrossEntropy


METRIKE
-------
  mean_iou              -- Srednji IoU (foreground klase, bez pozadine)
  mean_boundary_f1      -- Srednji Boundary F1 (kvaliteta rubova)
  overall_accuracy      -- Ukupna točnost po pikselu
  pixel_accuracy_fg     -- Točnost samo na foreground pikselima
  frequency_weighted_iou-- fwIoU (klase s više piksela imaju veći utjecaj)
  branch_recall         -- Recall za klasu Grane -- KLJUCNA METRIKA
  iou_{klasa}           -- IoU po pojedinoj klasi
  boundary_f1_{klasa}   -- Boundary F1 po pojedinoj klasi


POKRETANJE
----------
  1. Priprema podataka:
     python prepare_data.py --anotirane /path/to/Anotirane --out_dir /path/to/data

  2. Provjera dataseta:
     python dataset_report.py

  3. Treniranje:
     python main.py                          -- svi modeli
     python main.py --model unet_resnet34    -- jedan model

     Dostupni modeli:
       unet_resnet34, unet_efficientnet_b4, deeplabv3_resnet50,
       deeplabv3plus_mobilenet, segformer_b1, manet_resnext50,
       esanet_resnet34, pix2pix_gan, pix2pix_gen

  4. Samo evaluacija (bez ponovnog treniranja):
     python main.py --eval_only


IZLAZNI REZULTATI
-----------------
  checkpoints_exp2/
    {model}_best.pth          -- Checkpoint s najboljim val mIoU

  results_exp2/
    comparison_table.csv      -- Usporedna tablica svih modela
    training_curves.png       -- Krivulje ucenja (mIoU i BF1)
    {model}_per_image.csv     -- Metrike po svakoj slici
    {model}_qual.png          -- Vizualizacija segmentacije
    {model}_worst10.csv       -- Najtezih 10 slika po difficulty indexu

  logs_exp2/
    {model}_log.csv           -- Detaljan log po epohi
    {model}_info.txt          -- Informacije o modelu i treniranju


OVISNOSTI
---------
  pip install torch torchvision
  pip install segmentation-models-pytorch
  pip install albumentations
  pip install plyfile numpy opencv-python
  pip install pandas matplotlib tqdm scipy
  pip install transformers


KONFIGURACIJA (config.py)
-------------------------
  DATA_DIR             = "data"
  IMAGE_HEIGHT         = 640
  IMAGE_WIDTH          = 640
  IN_CHANNELS          = 4
  NUM_CLASSES          = 4
  BATCH_SIZE           = 4
  NUM_EPOCHS           = 100
  EARLY_STOP_PATIENCE  = 15
  CHECKPOINT_DIR       = "checkpoints_exp2"
  RESULTS_DIR          = "results_exp2"


=============================================================