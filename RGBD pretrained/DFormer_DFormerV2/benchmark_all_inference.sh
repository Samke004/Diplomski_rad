#!/bin/bash
GPUS=1
export CUDA_VISIBLE_DEVICES="0"

BASE="/home/samuel/Downloads/DFormerv1/DFormer"
CKPT_DIR="$BASE/checkpoints"
OUT_CSV="$BASE/results_inference.csv"

N_IMAGES=""     # broj slika za mjerenje po modelu; ostavi prazno za sve iz eval_source
WARMUP=10       # broj zagrijavajucih inferencija koje se ne broje u rezultat

cd "$BASE" || exit 1
export PYTHONPATH="$BASE:$BASE/..:$PYTHONPATH"

# obrisi stari CSV, benchmark_inference.py sam ispisuje header
rm -f "$OUT_CSV"

for dir in "$CKPT_DIR"/BranchDataset_*; do
    name=$(basename "$dir")

    # nadji checkpoint s najvecim miou u nazivu
    ckpt=$(ls "$dir"/epoch-*_miou_*.pth 2>/dev/null | \
        awk -F'miou_' '{print $2, $0}' | sed 's/\.pth$//' | \
        sort -k1 -n -r | head -n1 | awk '{print $2".pth"}')

    if [ -z "$ckpt" ]; then
        echo "Nema checkpointa u $name, skip"
        continue
    fi

    ckpt_rel="${ckpt#$BASE/}"

    if [[ "$name" == *"DFormerv2_S"* ]]; then
        cfg="local_configs.BranchDataset.DFormerv2_S"
    elif [[ "$name" == *"DFormerv2_B"* ]]; then
        cfg="local_configs.BranchDataset.DFormerv2_B"
    elif [[ "$name" == *"DFormerv2_L"* ]]; then
        cfg="local_configs.BranchDataset.DFormerv2_L"
    elif [[ "$name" == *"DFormer-Small"* ]]; then
        cfg="local_configs.BranchDataset.DFormer_Small"
    elif [[ "$name" == *"DFormer-Base"* ]]; then
        cfg="local_configs.BranchDataset.DFormer_Base"
    elif [[ "$name" == *"DFormer-Large"* ]]; then
        cfg="local_configs.BranchDataset.DFormer_Large"
    else
        echo "Nepoznat tip za $name, skip"
        continue
    fi

    # puni naziv foldera (arhitektura + datum treninga) - jedinstven u CSV-u
    # cak i ako postoji vise treninga iste arhitekture
    model_name="$name"

    echo "=== Mjerenje inferencije: $name ($ckpt_rel) ==="

    N_ARG=()
    if [ -n "$N_IMAGES" ]; then
        N_ARG=(--n "$N_IMAGES")
    fi

    python3 benchmark_inference.py \
        --config="$cfg" \
        --checkpoint="$ckpt_rel" \
        --model_name="$model_name" \
        --warmup="$WARMUP" \
        --output="$OUT_CSV" \
        "${N_ARG[@]}"

    echo "=== Gotovo: $name ==="
done

echo "=== SVE MJERENJE INFERENCIJE GOTOVO === Rezultati u $OUT_CSV"