#!/bin/bash
GPUS=1
export CUDA_VISIBLE_DEVICES="0"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Treniranje DFormerv2-Small ==="
PYTHONPATH="$(dirname $0)/..":"$(dirname $0)":$PYTHONPATH \
torchrun --nproc_per_node=$GPUS --master_port=29158 utils/train.py \
--config=local_configs.BranchDataset.DFormerv2_S --gpus=$GPUS \
--no-sliding --no-compile --no-amp --val_amp --no-use_seed

echo "=== Treniranje DFormerv2-Base ==="
PYTHONPATH="$(dirname $0)/..":"$(dirname $0)":$PYTHONPATH \
torchrun --nproc_per_node=$GPUS --master_port=29159 utils/train.py \
--config=local_configs.BranchDataset.DFormerv2_B --gpus=$GPUS \
--no-sliding --no-compile --no-amp --val_amp --no-use_seed

echo "=== SVE GOTOVO ==="
