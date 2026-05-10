#!/bin/bash
GPUS=1
export CUDA_VISIBLE_DEVICES="0"

echo "=== TEST Small ==="
PYTHONPATH="$(dirname $0)/..":"$(dirname $0)":$PYTHONPATH \
torchrun --nproc_per_node=$GPUS --master_port=29158 utils/train.py \
--config=local_configs.BranchDataset.DFormer_Small --gpus=$GPUS \
--no-sliding --no-compile --no-amp --val_amp --no-use_seed &
PID=$!
sleep 120
kill $PID
echo "=== TEST Base ==="
PYTHONPATH="$(dirname $0)/..":"$(dirname $0)":$PYTHONPATH \
torchrun --nproc_per_node=$GPUS --master_port=29159 utils/train.py \
--config=local_configs.BranchDataset.DFormer_Base --gpus=$GPUS \
--no-sliding --no-compile --no-amp --val_amp --no-use_seed &
PID=$!
sleep 120
kill $PID
echo "=== TEST Large ==="
PYTHONPATH="$(dirname $0)/..":"$(dirname $0)":$PYTHONPATH \
torchrun --nproc_per_node=$GPUS --master_port=29160 utils/train.py \
--config=local_configs.BranchDataset.DFormer_Large --gpus=$GPUS \
--no-sliding --no-compile --no-amp --val_amp --no-use_seed &
PID=$!
sleep 120
kill $PID
echo "=== SVE OK ==="
