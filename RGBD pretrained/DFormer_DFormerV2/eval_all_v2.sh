#!/bin/bash
GPUS=1
export CUDA_VISIBLE_DEVICES="0"

echo "=== Evaluacija DFormerv2-Small ==="
PYTHONPATH="$(dirname $0)/..":"$(dirname $0)":$PYTHONPATH \
torchrun --nproc_per_node=$GPUS --master_port=29158 utils/eval.py \
--config=local_configs.BranchDataset.DFormerv2_S \
--gpus=$GPUS --no-sliding --no-compile --no-amp \
--continue_fpath="checkpoints/BranchDataset_DFormerv2_S_20260508-134047/epoch-151_miou_87.83.pth"

echo "=== Evaluacija DFormerv2-Base ==="
PYTHONPATH="$(dirname $0)/..":"$(dirname $0)":$PYTHONPATH \
torchrun --nproc_per_node=$GPUS --master_port=29159 utils/eval.py \
--config=local_configs.BranchDataset.DFormerv2_B \
--gpus=$GPUS --no-sliding --no-compile --no-amp \
--continue_fpath="checkpoints/BranchDataset_DFormerv2_B_20260508-164158/epoch-146_miou_88.25.pth"

echo "=== SVE EVALUACIJE GOTOVE ==="
