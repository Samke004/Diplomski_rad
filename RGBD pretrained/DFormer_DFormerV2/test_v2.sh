#!/bin/bash
GPUS=1
export CUDA_VISIBLE_DEVICES="0"

echo "=== TEST DFormerv2-Small ==="
PYTHONPATH="$(dirname $0)/..":"$(dirname $0)":$PYTHONPATH \
torchrun --nproc_per_node=$GPUS --master_port=29158 utils/train.py \
--config=local_configs.BranchDataset.DFormerv2_S --gpus=$GPUS \
--no-sliding --no-compile --no-amp --val_amp --no-use_seed &
PID=$!
sleep 60
kill $PID

echo "=== TEST DFormerv2-Base ==="
PYTHONPATH="$(dirname $0)/..":"$(dirname $0)":$PYTHONPATH \
torchrun --nproc_per_node=$GPUS --master_port=29159 utils/train.py \
--config=local_configs.BranchDataset.DFormerv2_B --gpus=$GPUS \
--no-sliding --no-compile --no-amp --val_amp --no-use_seed &
PID=$!
sleep 60
kill $PID

echo "=== TEST DFormerv2-Large ==="
PYTHONPATH="$(dirname $0)/..":"$(dirname $0)":$PYTHONPATH \
torchrun --nproc_per_node=$GPUS --master_port=29160 utils/train.py \
--config=local_configs.BranchDataset.DFormerv2_L --gpus=$GPUS \
--no-sliding --no-compile --no-amp --val_amp --no-use_seed &
PID=$!
sleep 60
kill $PID

echo "=== SVE OK ==="
