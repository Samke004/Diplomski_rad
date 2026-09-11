#!/bin/bash
GPUS=1
export CUDA_VISIBLE_DEVICES="0"
BASE="/home/samuel/Downloads/DFormerv1/DFormer"
CKPT_DIR="$BASE/checkpoints"
OUT_CSV="$BASE/results.csv"
PORT=29200

cd "$BASE" || exit 1
export PYTHONPATH="$BASE:$BASE/..:$PYTHONPATH"

echo "model,overall_accuracy,pixel_accuracy,pixel_accuracy_fg,frequency_weighted_iou,frequency_weighted_iou_fg,mean_iou,mean_boundary_f1,branch_recall,iou_background,iou_trunk,iou_branches,iou_support,recall_background,recall_trunk,recall_branches,recall_support,boundary_f1_background,boundary_f1_trunk,boundary_f1_branches,boundary_f1_support" > "$OUT_CSV"

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

    # relativni put od BASE (eval.py konstruira putanje relativno na cwd)
    ckpt_rel="${ckpt#$BASE/}"

    if [[ "$name" == *"DFormerv2_S"* ]]; then
        cfg="local_configs.BranchDataset.DFormerv2_S"; model_name="DFormerv2_S"
    elif [[ "$name" == *"DFormerv2_B"* ]]; then
        cfg="local_configs.BranchDataset.DFormerv2_B"; model_name="DFormerv2_B"
    elif [[ "$name" == *"DFormerv2_L"* ]]; then
        cfg="local_configs.BranchDataset.DFormerv2_L"; model_name="DFormerv2_L"
    elif [[ "$name" == *"DFormer-Small"* ]]; then
        cfg="local_configs.BranchDataset.DFormer_Small"; model_name="DFormer_Small"
    elif [[ "$name" == *"DFormer-Base"* ]]; then
        cfg="local_configs.BranchDataset.DFormer_Base"; model_name="DFormer_Base"
    elif [[ "$name" == *"DFormer-Large"* ]]; then
        cfg="local_configs.BranchDataset.DFormer_Large"; model_name="DFormer_Large"
    else
        echo "Nepoznat tip za $name, skip"
        continue
    fi

    echo "=== Evaluacija $name ($ckpt_rel) ==="
    PORT=$((PORT+1))

    torchrun --nproc_per_node=$GPUS --master_port=$PORT utils/eval.py \
        --config="$cfg" \
        --gpus=$GPUS --no-sliding --no-compile --no-amp \
        --continue_fpath="$ckpt_rel" > "/tmp/eval_${name}.log" 2>&1

    log="/tmp/eval_${name}.log"

    # izvuci RESULT_JSON liniju i pretvori je u CSV redak istog redoslijeda kolona kao header
    python3 - "$log" "$model_name" >> "$OUT_CSV" <<'PYEOF'
import sys, re, json

log_path = sys.argv[1]
model_name = sys.argv[2]

result = None
with open(log_path) as f:
    for line in f:
        m = re.search(r"RESULT_JSON:\s*(\{.*\})", line)
        if m:
            result = json.loads(m.group(1))

cols = [
    "overall_accuracy","pixel_accuracy","pixel_accuracy_fg",
    "frequency_weighted_iou","frequency_weighted_iou_fg",
    "mean_iou","mean_boundary_f1","branch_recall",
    "iou_background","iou_trunk","iou_branches","iou_support",
    "recall_background","recall_trunk","recall_branches","recall_support",
    "boundary_f1_background","boundary_f1_trunk","boundary_f1_branches","boundary_f1_support",
]

if result is None:
    print(f"{model_name}," + ",".join(["NA"] * len(cols)))
else:
    vals = [result.get(c, "NA") for c in cols]
    print(f"{model_name}," + ",".join(str(v) for v in vals))
PYEOF

    echo "=== Gotovo: $name ==="
done

echo "=== SVE EVALUACIJE GOTOVE === Rezultati u $OUT_CSV"