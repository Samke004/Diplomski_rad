=============================================================
  SEGMENTACIJA STABALA JABUKE — HHA EKSPERIMENT
  (RGB + HHA enkodiranje dubine, 6 kanala)
=============================================================

MOTIVACIJA
----------
Digumarti et al. (ICRA 2018) pokazuju da HHA enkodiranje dubine
daje bolje rezultate od raw depth u segmentaciji stabala.
HHA (Horizontal disparity, Height above ground, Angle with gravity)
daje geometrijski bogatiji opis scene od jednog kanalnog depth.

Za razliku od originalnog papera koji koristi late fusion
(dva odvojena enkodera), ovaj eksperiment istrazuje early fusion
pristup gdje se RGB i HHA konkateniraju na ulazu u 6-kanalni
tensor i prolaze kroz jedan enkoder. Prednost je jednostavnija
implementacija i kompatibilnost s modernim arhitekturama.

Usporedba s prethodnim eksperimentima:
  Exp 1 — RGB + raw depth (4 kanala, bez augmentacija)
  Exp 2 — RGB + raw depth (4 kanala, s augmentacijama)
  Exp 3 — RGB + HHA       (6 kanala, s augmentacijama)  <- OVAJ


HHA GENERIRANJE
---------------
Algoritam: Gupta et al. "Learning Rich Features from RGB-D
           Images for Object Detection and Segmentation" (ECCV 2014)
Implementacija: github.com/charlesCXK/Depth2HHA-python

Parametri kamere (Kinect v1):
  FX = FY = 570.3422241210938
  CX = 314.5
  CY = 235.5

Ulaz:  depth kanal iz _rgbd.npy (u METRIMA, max ~1.85m)
Izlaz: 3-kanalni HHA array uint8 [H, W, 3], sprema se kao _hha.npy

Napomena: depth se prosljeduje getHHA() u metrima bez mnozenja
s 1000 — u skladu s uputama originalnog repozitorija.


ULAZNI FORMAT
-------------
  _rgbd.npy  — 4-kanalni array, koristi se RGB dio (kanali 0-2)
  _hha.npy   — 3-kanalni HHA array, normaliziran u [0,1]
  _mask.png  — maska klasa (0=pozadina, 1=deblo, 2=grane, 3=potpora)

  Ulazni tenzor: [6, H, W] = RGB(3) + HHA(3)
  Rezolucija: 640x640


DATASET
-------
Isti split kao Exp 2 (Pokusaj_Chen_MultiDepthV2), rucno kopiran:
  /home/samuel/Desktop/Pokusaj_Chen_HHA/data/

  Train: 501 slika
  Val:    80 slika
  Test:   79 slika

HHA fajlovi generirani skriptom convert_to_hha.py i smjesteni
u isti folder pored _rgbd.npy i _mask.png fajlova.


MODELI
------
  unet_resnet34          — U-Net, ResNet-34 backbone
  unet_efficientnet_b4   — U-Net, EfficientNet-B4 backbone
  deeplabv3_resnet50     — DeepLabV3, ResNet-50 backbone
  deeplabv3plus_mobilenet— DeepLabV3+, MobileNetV2 backbone
  segformer_b1           — SegFormer-B1, transformer arhitektura

Svi modeli pod identičnim uvjetima:
  IN_CHANNELS = 6
  IMAGE_HEIGHT = IMAGE_WIDTH = 640
  BATCH_SIZE = 4

Inicijalizacija 4. i 5. kanala (HHA kanali 1 i 2):
  smp modeli: automatska adaptacija prvog conv sloja
  SegFormer:  prva 3 kanala = pretrained, kanali 3-5 = Kaiming init


POKRETANJE
----------
  # Generiraj HHA fajlove (jednom):
  python3 convert_to_hha.py

  # Treniraj sve modele:
  python3 main.py

  # Treniraj jedan model:
  python3 main.py --model unet_resnet34

  # Samo evaluacija:
  python3 main.py --eval_only


REZULTATI
---------
  checkpoints_hha/
    {model}_best.pth       — checkpoint s najboljim val mIoU

  results_hha/
    comparison_table.csv   — usporedna tablica svih modela
    training_curves.png    — krivulje ucenja
    {model}_per_image.csv  — metrike po svakoj slici
    {model}_qual.png       — vizualizacija segmentacije

  logs/
    {model}_log.csv        — detaljan log po epohi
    {model}_info.txt       — info o modelu i treniranju


LITERATURA
----------
  Digumarti et al., "An Approach for Semantic Segmentation of
  Tree-like Vegetation", ICRA 2018.

  Gupta et al., "Learning Rich Features from RGB-D Images for
  Object Detection and Segmentation", ECCV 2014.

  charlesCXK, Depth2HHA-python,
  github.com/charlesCXK/Depth2HHA-python

=============================================================