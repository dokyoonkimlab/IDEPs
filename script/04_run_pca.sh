#!/bin/bash
# Step 4: Apply PCA to reduce DINOv2 features to 30 principal components

IDEPS_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

python "$IDEPS_ROOT/source/pca/pca_transform.py" \
    --input_npz "$IDEPS_ROOT/data/features_level-15.npz" \
    --n_components 30 \
    --output_csv "$IDEPS_ROOT/data/df_IDEPs_level_15.csv"
