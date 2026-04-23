#!/bin/bash
# Step 5: Run prevalence prediction using Random Forest
# Compares base model (Cov+PRS) vs full model (Cov+PRS+SES+Lifestyle+IDEP_residual)

IDEPS_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

python "$IDEPS_ROOT/source/prevalence_prediction/RF_model.py" \
    --labels_file "$IDEPS_ROOT/data/df_labels.csv" \
    --prs_file "$IDEPS_ROOT/data/df_prs.csv" \
    --pc_file "$IDEPS_ROOT/data/df_IDEPs_level_15.csv" \
    --cov_file "$IDEPS_ROOT/data/df_cov.csv" \
    --ses_file "$IDEPS_ROOT/data/df_ses.csv" \
    --phenotype T2D \
    --output_dir "$IDEPS_ROOT/results/prevalence/" \
    --max_pcs 30
