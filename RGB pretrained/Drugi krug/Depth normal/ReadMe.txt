=============================================================
  SEGMENTACIJA STABALA JABUKE — DRUGI KRUG TRENIRANJA
  (657 PLY datoteka, prošireni dataset)
=============================================================

PROMJENE U ODNOSU NA PRVI KRUG
-------------------------------
  - Dataset proširen s ~400 na 657 PLY datoteka
  - Nova skripta za podjelu: prepare_data_v2.py
  - Dodan novi skup ručno korigiranih stabala (tree_najnovije_od_valeta)
  - MANet pokrenut s batch=2 zbog nedostatka grafičke memorije
  - Pix2Pix pokrenut na rezoluciji 512x512 (umjesto 640x640) zbog memorije


DATASET (v2)
------------
  Grupe stabala:

    A  -- Samuel ručno         Drvo0000-Drvo0015             ~108 slika
    B  -- Valentin ručno       tree_1_V_0179-0183             ~91 slika
    CM -- Najnovije ručno      tree_1_V_0115/0125/0135/
                               0145/0155 (val/test eligible)   ~50 slika
    D  -- Najnovije ostalo     tree_najnovije_od_valeta        ~train only
    E  -- Model + pregled      tree_nove_od_valeta             ~200 slika

  Podjela na razini stabla (bez data leakagea):
    TRAIN (~80%): Grupe D + E + 40% od A+B+CM
    VAL   (~10%): 30% od A+B+CM  -- samo ručno anotirano
    TEST  (~10%): 30% od A+B+CM  -- samo ručno anotirano

  Target: ~500 train / ~75 val / ~75 test


POKRETANJE — POSEBNI SLUCAJEVI
-------------------------------

  MANet (batch=2 zbog OOM na 640x640):
  -------------------------------------
  python3 -c "
  from config import Config as cfg
  cfg.BATCH_SIZE = 2
  from dataset import AppleTreeDataset
  from augmentations import get_train_transforms, get_val_transforms
  from torch.utils.data import DataLoader
  from models import build_manet
  from trainer import SegmentationTrainer
  train_ds = AppleTreeDataset(cfg.TRAIN_IMG_DIR, cfg.TRAIN_MASK_DIR,
      transform=get_train_transforms(640, 640), img_size=(640, 640))
  val_ds = AppleTreeDataset(cfg.VAL_IMG_DIR, cfg.VAL_MASK_DIR,
      transform=get_val_transforms(640, 640), img_size=(640, 640))
  train_loader = DataLoader(train_ds, batch_size=2, shuffle=True,
      drop_last=True, num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY)
  val_loader = DataLoader(val_ds, batch_size=2, shuffle=False,
      num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY)
  model = build_manet(pretrained=True, in_channels=4, num_classes=4)
  SegmentationTrainer(model=model, model_name='manet_resnext50', cfg=cfg,
      train_loader=train_loader, val_loader=val_loader, lr=1e-4).train()
  "

  Pix2Pix GAN (512x512 zbog OOM):
  --------------------------------
  python3 -c "
  from config import Config as cfg
  cfg.IMAGE_HEIGHT = 512
  cfg.IMAGE_WIDTH  = 512
  from dataset import AppleTreeDataset
  from augmentations import get_train_transforms, get_val_transforms
  from torch.utils.data import DataLoader
  from models import Pix2PixGenerator, Pix2PixDiscriminator
  from trainer import Pix2PixTrainer
  train_ds = AppleTreeDataset(cfg.TRAIN_IMG_DIR, cfg.TRAIN_MASK_DIR,
      transform=get_train_transforms(512, 512), img_size=(512, 512))
  val_ds = AppleTreeDataset(cfg.VAL_IMG_DIR, cfg.VAL_MASK_DIR,
      transform=get_val_transforms(512, 512), img_size=(512, 512))
  train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,
      drop_last=True, num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY)
  val_loader = DataLoader(val_ds, batch_size=4, shuffle=False,
      num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY)
  G = Pix2PixGenerator(in_channels=4, out_channels=4)
  D = Pix2PixDiscriminator(in_channels=4)
  Pix2PixTrainer(generator=G, discriminator=D, use_gan=True,
      cfg=cfg, train_loader=train_loader, val_loader=val_loader).train()
  "

  Svi ostali modeli — standardno pokretanje:
  -------------------------------------------
  python main.py                          -- svi modeli
  python main.py --model unet_resnet34    -- jedan model
  python main.py --eval_only              -- samo evaluacija


KONFIGURACIJA ZA OVAJ KRUG
---------------------------
  Svi modeli (osim iznimki gore):
    IMAGE_HEIGHT  = 640
    IMAGE_WIDTH   = 640
    BATCH_SIZE    = 4
    IN_CHANNELS   = 4

  MANet iznimka:
    BATCH_SIZE    = 2   (OOM pri batch=4 na 640x640)

  Pix2Pix iznimka:
    IMAGE_HEIGHT  = 512
    IMAGE_WIDTH   = 512
    BATCH_SIZE    = 4   (OOM pri 640x640)


NAPOMENE O MEMORIJI (GPU OOM)
------------------------------
  - MANet/ResNeXt-50 je memorijski najzahtjevniji encoder
    -> smanjen batch na 2, rezolucija ostavljena na 640
  - Pix2Pix ima i Generator i Diskriminator u memoriji istovremeno
    -> smanjena rezolucija na 512x512, batch ostaje 4
  - Usporedba s prvim krugom može biti djelomično narušena
    za ova dva modela zbog različitih uvjeta treniranja


STRUKTURA REZULTATA
-------------------
  checkpoints/
    {model}_best.pth        -- Checkpoint s najboljim val mIoU

  results/
    comparison_table.csv    -- Usporedna tablica svih modela
    training_curves.png     -- Krivulje ucenja
    {model}_per_image.csv   -- Metrike po svakoj slici
    {model}_qual.png        -- Vizualizacija segmentacije
    {model}_worst10.csv     -- Najtezih 10 slika

  logs/
    {model}_log.csv         -- Detaljan log po epohi
    {model}_info.txt        -- Info o modelu + ukupno trajanje


=============================================================