#!/bin/bash

# Zadani uzorci
SAMPLES="A_Drvo0000_0000-0 A_Drvo0003_0003-2 A_Drvo0007_0007-3 B_tree_1_V_0183_5 CM_tree_1_V_0145_tree_1_V_0145_1_pred CM_tree_1_V_0115_tree_1_V_0115_5_pred"
OUT_DIR="./qualitative_dformer"

# Popis svih modela
MODELS=(
    "deeplabv3plus"
    "deeplabv3"
    "esanet"
    "manet"
    "segformer"
    "unet_resnet34"
    "unet_effb4"
)

for MODEL in "${MODELS[@]}"; do
    echo "================================================================"
    echo " Generiram slike za model: $MODEL"
    echo "================================================================"
    
    python3 visualize_multidepth_clean.py \
        --model "$MODEL" \
        --samples $SAMPLES \
        --output_dir "$OUT_DIR"
done

echo "Gotovo! Sve slike se nalaze u $OUT_DIR"