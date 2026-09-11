# Eksperiment 4 — DFormer / DFormerv2 (RGB-D pretrenirani bazni model)


Fork službenog DFormer/DFormerv2 repozitorija (VCIP-RGBD, MIT/ICLR2024+CVPR2025 
radovi — vidi `README_upstream.md` za originalnu dokumentaciju i citiranje). 
6 varijanti: DFormer-{Small,Base,Large}, DFormerv2-{Small,Base,Large}.

`results.csv` = Tablica 5.8. `results_inference.csv` = Tablica 5.11.

## Okruženje

Python 3.10.20, CUDA 11.8 (drugi env od RGB-pretrained dijela):

    conda create -n dformer python=3.10.20 -y
    conda activate dformer
    conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
    pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
    pip install -r requirements.txt

`requirements.txt` (prema upustvu autora DFormer-a + provjereno protiv 
stvarno instaliranih verzija):

    tqdm
    opencv-python==4.13.0.92
    scipy==1.15.3
    tensorboardX==2.6.5
    tabulate
    easydict
    ftfy
    regex
    numpy==1.26.4
    timm==1.0.27
    addict


## Pretrenirani bazni modeli

Preuzeti ImageNet/RGB-D-pretrenirane backbone checkpointove sa službenog 
DFormer/DFormerv2 repozitorija (linkovi u `README_upstream.md`, sekcija 
"Checkpoints" → "Pretrained"), staviti u `checkpoints/pretrained/` prema 
putanjama u `local_configs/BranchDataset/*.py` (`C.pretrained_model`).

## Priprema podataka

Koristi isti split kao Eksperiment 2, konvertiran u DFormer-ov format 
(`RGB/`, `Depth/`, `Label/` + `train.txt`/`val.txt`/`test.txt`):

    python data_prepare_DFormer.py

**PRIJE POKRETANJA ručno izmijeniti `src_base` i `dst_base` na vrhu skripte** 
— hardkodirane su moje lokalne putanje, nema CLI argumenata. Skripta automatski 
preskače datoteke s "BACKUP" u imenu.

## Brzi test prije punog treninga

Preporučeno prije pokretanja punog 200-epoha treninga — provjeri da 
config/podaci/okruženje rade, bez čekanja:

    bash test_models.sh   # DFormer v1: Small, Base, Large — po 120s svaki
    bash test_v2.sh       # DFormerv2: Small, Base, Large — po 60s svaki

Svaki pokrene trening u pozadini, pričeka, ubije proces. Ako se nešto sruši 
u tih 60-120s (import greška, OOM, kriva putanja podataka), vidjet ćete odmah.

## Trening

    bash train_all.sh   # DFormer v1: Small → Base → Large, redom
    bash train_v2.sh     # DFormerv2: Small → Base (NE Large, vidi napomenu ispod)

Oba koriste AdamW, lr=6×10⁻⁵, 200 epoha, `--no-amp --val_amp --no-use_seed`.

**Napomena — AMP**: skripte eksplicitno isključuju AMP za sam trening 
(`--no-amp`), dok `--val_amp` ostaje uključen (default) za periodičku 
validaciju tijekom treninga. Ovo se razlikuje od opisa u poglavlju 4.9 rada, 
koji navodi da je AMP korišten za ubrzavanje/uštedu memorije **tijekom 
treninga**, ali ovo su bili naknadni eksperimenti, pa moze se vratiti ili ostaviti.

**DFormerv2-Large nije dio `train_v2.sh`** — treniran je zasebno, ručno 
(kasnije, na RTX 4080, nakon što je prvi pokušaj na manjoj kartici pukao 
zbog OOM-a). 

`batch_size` nije jednak za sve varijante (Small/Base v1 = 4, Large v1 i sve 
tri v2 osim Small = 2) — vidi tablicu 4.10 rada i `local_configs/BranchDataset/*.py`.
Zbog eksperimenata na 4060TI koja nemoze pokretat sve kao 4080...


## Evaluacija

    bash eval_all.sh

Automatski petlja kroz sve foldere u `checkpoints/BranchDataset_*`, prepoznaje 
tip modela iz naziva foldera, uzima checkpoint s najvišim mIoU u imenu, 
evaluira sa svih 5 skala (MST, uključen po defaultu — `eval_scale_array` u 
pojedinačnim configima se u praksi ne koristi, MST skale su hardkodirane u 
`utils/eval.py`), i piše red u `results.csv`.

`eval_all_v2.sh` i `eval_v2_novi.sh` su raniji, ručni pokušaji specifičnih 
checkpointa za Small/Base — **ne koristiti za finalnu reprodukciju**, zadržani 
samo kao povijesni trag. `eval.sh` je nepromijenjeni upstream template 
(NYUDepthv2 config) — nije korišten za ovaj rad.

## Brzina inferencije 

    bash benchmark_all_inference.sh

Isti mehanizam kao `eval_all.sh` — petlja kroz sve checkpoint foldere, mjeri 
prosjek/std/min/max vremena inferencije (batch=1, 10 zagrijavajućih + ostatak 
mjeren) preko `benchmark_inference.py`, piše u `results_inference.csv`.

Za DFormerv2-Small/Base tablica 5.11 prikazuje prosjek više mjerenja (više 
checkpoint foldera je postojalo za te varijante) — vidi pojedinačne redove 
u CSV-u.

Napomena: `benchmark_inference.py` čita dubinsku PNG preko 
`cv2.imread(path, cv2.IMREAD_GRAYSCALE)` bez `IMREAD_ANYDEPTH` zastavice, što 
za 16-bitne dubinske slike gubi preciznost/skalu u odnosu na službeni 
`RGBXDataset` loader korišten u `eval.py`. Ovo **ne utječe na točnost 
izmjerenog vremena** (broj FLOP-ova u forward prolazu ne ovisi o sadržaju 
ulaza), ali ako se skripta ikad koristi i za provjeru točnosti predikcija, 
ovo treba ispraviti.

## Kvalitativni rezultati 

    python visualize_dformer_clean.py --n 6 --output_dir ./qualitative_dformer

Default checkpoint u skripti (`checkpoints/BranchDataset_DFormer-Base_20260507-181200/epoch-165_miou_88.24.pth`) 
je onaj korišten za Sliku 5.3 u radu 

`visualizacija_Dformer.py` (bez "_clean") je raniji nacrt iste vizualizacije 
(s tekstualnim naslovima, jedna spojena slika) — zamijenjen ovim, nije 
korišten za finalne slike u radu.


## Rangiranje testnih slika i "najgori primjer" (Slika 5.5)

    python rank_test_images.py \
      --checkpoint checkpoints/BranchDataset_DFormer-Base_20260507-181200/epoch-165_miou_88.24.pth \
      --metric miou_fg --generate_visuals --n_best 3 --n_worst 3

Proizvodi `test_ranking.csv` (kompletno rangiranje) i best/worst vizualizacije 
u `best_worst_visuals/`. **Napomena iz same skripte**: koristi single-scale 
inferenciju (bez MST) radi brzine — apsolutne brojke se stoga neće poklopiti 
s Tablicom 5.8, ali relativni poredak (koja je slika najgora/najbolja) ostaje 
vrijedeći i to je ono što određuje Sliku 5.5.

## Implementacijske napomene

- `eval_scale_array` u `local_configs/BranchDataset/*.py` razlikuje se po 
  varijanti (samo DFormer-Base ima punih 5 skala, ostalih pet imaju `[1]`) — 
  ovo **ne utječe na stvarne rezultate**: `evaluate_msf()` u `utils/val_mm.py` 
  prima skale kao eksplicitan parametar hardkodiran u `eval.py`, nikad ne čita 
  ovo polje configa. Vrijedno ujednačiti radi jasnoće ili ukloniti polje.
- AMP je isključen za sam trening unatoč opisu u poglavlju 4.9 rada — vidi 
  napomenu u sekciji Trening.

