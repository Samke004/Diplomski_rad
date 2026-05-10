MODELI I HYPERPARAMETRI
-----------------------
  Optimizer:    AdamW
  LR:           6e-5
  Epohe:        200
  Augmentacije: [0.5, 0.75, 1.0, 1.25, 1.5]
  Warm-up:      5 epoha

  DFormer-Small/Base:   batch_size=4
  DFormer-Large:        batch_size=2
  DFormerv2-Small:      batch_size=4
  DFormerv2-Base:       batch_size=2

Hardware:
  Treniranje: NVIDIA RTX 3080 Ti (12GB VRAM)
  Evaluacija: NVIDIA RTX 4060 Ti (8GB VRAM)

REZULTATI NA TEST SKUPU (77 slika)
mIoU = mean samo po foreground klasama (Deblo, Grane, Potpora)
IoUs = [Ostalo, Deblo, Grane, Potpora]
------------------------------------
Model           | mIoU  | mAcc  | mF1   |   IoU Deblo | IoU Grane | IoU Potpora
----------------|-------|-------|-------|------------|-----------|-----------|------------
DFormer-Small   | 86.77 | 98.14 | 94.45 |    80.26   |   87.75   |   92.30
DFormer-Base    | 87.14 | 98.20 | 94.62 |   81.04   |   88.09   |   92.29
DFormer-Large   | 87.07 | 98.18 | 94.58 |  80.76   |   87.98   |   92.46
DFormerv2-Small | 84.30 | 97.88 | 93.35 |    77.26   |   86.02   |   89.62
DFormerv2-Base  | 84.84 | 97.97 | 93.60 |    77.78   |   86.63   |   90.11

ZAKLJUČCI
---------
- DFormer-Base postiže najbolje rezultate (mIoU 87.14%)
- Razlike između Small/Base/Large su minimalne (~0.4%)
- Originalni DFormer (ICLR 2024) nadmašuje DFormerv2 (CVPR 2025) na ovom datasetu
- Klasa Deblo (~77-81%) najteža za segmentaciju
- Klasa Ostalo (~98%) najlakša — nije uključena u mIoU
- DFormerv2-Large nije testiran (OOM na 12GB VRAM)
