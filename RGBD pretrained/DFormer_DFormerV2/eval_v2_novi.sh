#!/bin/bash
GPUS=1
export CUDA_VISIBLE_DEVICES="0"

echo "=== DFormerv2-Small NOVI ==="
PYTHONPATH="$(dirname $0)/..":"$(dirname $0)":$PYTHONPATH \
torchrun --nproc_per_node=$GPUS --master_port=29158 utils/eval.py \
--config=local_configs.BranchDataset.DFormerv2_S \
--gpus=$GPUS --no-sliding --no-compile --no-amp \
--continue_fpath="checkpoints/BranchDataset_DFormerv2_S_20260509-162607/epoch-215_miou_86.4.pth"

echo "=== DFormerv2-Base NOVI ==="
PYTHONPATH="$(dirname $0)/..":"$(dirname $0)":$PYTHONPATH \
torchrun --nproc_per_node=$GPUS --master_port=29159 utils/eval.py \
--config=local_configs.BranchDataset.DFormerv2_B \
--gpus=$GPUS --no-sliding --no-compile --no-amp \
--continue_fpath="checkpoints/BranchDataset_DFormerv2_B_20260509-205923/epoch-184_miou_86.96.pth"

echo "=== GOTOVO ==="
