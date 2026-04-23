#!/bin/bash
# Step 3: Extract features from BEV images using trained DINOv2 model

IDEPS_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CHECKPOINT="$IDEPS_ROOT/logs/DINO_pretraining/train_run_vit-base/eval/training_124999/teacher_checkpoint.pth"

# Extract features at zoom level 15
python "$IDEPS_ROOT/source/feature_extraction/feature_extractor.py" \
    --image_dir "$IDEPS_ROOT/data/bev_images/" \
    --checkpoint "$CHECKPOINT" \
    --level_of_interest 15 \
    --input_size 224 \
    --feat_dim 768 \
    --output_file "$IDEPS_ROOT/data/features_level-15.npz" \
    --batch_size 64 \
    --num_workers 32

# Extract features at zoom level 17
python "$IDEPS_ROOT/source/feature_extraction/feature_extractor.py" \
    --image_dir "$IDEPS_ROOT/data/bev_images/" \
    --checkpoint "$CHECKPOINT" \
    --level_of_interest 17 \
    --input_size 224 \
    --feat_dim 768 \
    --output_file "$IDEPS_ROOT/data/features_level-17.npz" \
    --batch_size 64 \
    --num_workers 32
