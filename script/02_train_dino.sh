#!/bin/bash
# Step 2: Train DINOv2 ViT-Base model on BEV images
# Requires 4 GPUs and the dataset prepared in train/test splits

IDEPS_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DINOV2_DIR="$IDEPS_ROOT/source/dinov2"
OUTPUT_DIR="$IDEPS_ROOT/logs/DINO_pretraining/train_run_vit-base"
DATA_ROOT="$IDEPS_ROOT/data/bev_images_split"
LABEL_FILE="<PATH_TO_LABEL_TSV>"  # TSV with eid and population_density_urban_or_rural columns

cd "$DINOV2_DIR"

CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. torchrun \
    --nproc_per_node=4 \
    --master_port=29502 \
    dinov2/train/train.py \
    --config-file dinov2/configs/train/vitb14_short_ukbbev.yaml \
    --output-dir "$OUTPUT_DIR" \
    train.dataset_path="UKBBEV:split=TRAIN:root=${DATA_ROOT}:extra=${DATA_ROOT}/extra:label_file=${LABEL_FILE}"
