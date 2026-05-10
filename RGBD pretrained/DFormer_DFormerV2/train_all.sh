#!/bin/bash

echo "=== Treniranje DFormer-Small ==="
GPUS=1
NNODES=1
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29158}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export CUDA_VISIBLE_DEVICES="0"

PYTHONPATH="$(dirname $0)/..":"$(dirname $0)":$PYTHONPATH \
    torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    utils/train.py \
    --config=local_configs.BranchDataset.DFormer_Small --gpus=$GPUS \
    --no-sliding --no-compile --no-amp --val_amp --no-use_seed

echo "=== Small gotov, krece Base ==="

PYTHONPATH="$(dirname $0)/..":"$(dirname $0)":$PYTHONPATH \
    torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=29159 \
    utils/train.py \
    --config=local_configs.BranchDataset.DFormer_Base --gpus=$GPUS \
    --no-sliding --no-compile --no-amp --val_amp --no-use_seed

echo "=== Base gotov, krece Large ==="

PYTHONPATH="$(dirname $0)/..":"$(dirname $0)":$PYTHONPATH \
    torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=29160 \
    utils/train.py \
    --config=local_configs.BranchDataset.DFormer_Large --gpus=$GPUS \
    --no-sliding --no-compile --no-amp --val_amp --no-use_seed

echo "=== SVE GOTOVO ==="