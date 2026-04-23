#!/bin/bash
# Step 6: Run incidence prediction using Cox Proportional Hazards model
# Compares base model (Cov+PRS) vs full model (Cov+PRS+SES+Lifestyle+IDEP_residual)
# Includes NNS (Number Needed to Screen) contextualization metrics

IDEPS_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

python "$IDEPS_ROOT/source/incidence_prediction/ML_model.py" \
    --incidence_file "$IDEPS_ROOT/data/incidence_labels.csv" \
    --prs_file "$IDEPS_ROOT/data/df_prs.csv" \
    --pc_file "$IDEPS_ROOT/data/df_IDEPs_level_15.csv" \
    --cov_file "$IDEPS_ROOT/data/df_cov.csv" \
    --ses_file "$IDEPS_ROOT/data/df_ses.csv" \
    --phenotype T2D \
    --output_dir "$IDEPS_ROOT/results/incidence/" \
    --max_pcs 30 \
    --n_splits 10
