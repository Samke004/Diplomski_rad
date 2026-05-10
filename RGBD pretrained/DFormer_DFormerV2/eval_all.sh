#!/bin/bash
GPUS=1
export CUDA_VISIBLE_DEVICES="0"

echo "=== Evaluacija DFormer-Small ==="
PYTHONPATH="$(dirname $0)/..":"$(dirname $0)":$PYTHONPATH \
torchrun --nproc_per_node=$GPUS --master_port=29158 utils/eval.py \
--config=local_configs.BranchDataset.DFormer_Small \
--gpus=$GPUS --no-sliding --no-compile --no-amp \
--continue_fpath="checkpoints/BranchDataset_DFormer-Small_20260507-164410/epoch-184_miou_87.87.pth"

echo "=== Evaluacija DFormer-Base ==="
PYTHONPATH="$(dirname $0)/..":"$(dirname $0)":$PYTHONPATH \
torchrun --nproc_per_node=$GPUS --master_port=29159 utils/eval.py \
--config=local_configs.BranchDataset.DFormer_Base \
--gpus=$GPUS --no-sliding --no-compile --no-amp \
--continue_fpath="checkpoints/BranchDataset_DFormer-Base_20260507-181200/epoch-165_miou_88.24.pth"

echo "=== Evaluacija DFormer-Large ==="
PYTHONPATH="$(dirname $0)/..":"$(dirname $0)":$PYTHONPATH \
torchrun --nproc_per_node=$GPUS --master_port=29160 utils/eval.py \
--config=local_configs.BranchDataset.DFormer_Large \
--gpus=$GPUS --no-sliding --no-compile --no-amp \
--continue_fpath="checkpoints/BranchDataset_DFormer-Large_20260507-203128/epoch-65_miou_87.81.pth"

echo "=== SVE EVALUACIJE GOTOVE ==="
