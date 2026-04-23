#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import os
import argparse
import numpy as np
import pandas as pd
import scipy.stats as st

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings("ignore")

# ==============================================================================
# DeLong Test Implementation
# ==============================================================================
def compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float64)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=np.float64)
    T2[J] = T + 1
    return T2

def fast_delong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float64)
    ty = np.empty([k, n], dtype=np.float64)
    tz = np.empty([k, m + n], dtype=np.float64)

    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov

def calc_pvalue(aucs, sigma):
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    return 2 * (1 - st.norm.cdf(np.abs(z)))

def delong_roc_test(ground_truth, predictions_one, predictions_two):
    order = np.argsort(ground_truth)[::-1]
    ground_truth = ground_truth[order]
    predictions_one = predictions_one[order]
    predictions_two = predictions_two[order]

    predictions_sorted_transposed = np.vstack((predictions_one, predictions_two))
    label_1_count = int(np.sum(ground_truth))

    aucs, delongcov = fast_delong(predictions_sorted_transposed, label_1_count)
    return calc_pvalue(aucs, delongcov)[0][0]

# ---------------------------------------------------
# Argument parser
# ---------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Incremental value of IDEPs on top of PRS, SES, and lifestyle (RF + DeLong)'
    )
    parser.add_argument('--labels_file', type=str, required=True)
    parser.add_argument('--prs_file', type=str, required=True)
    parser.add_argument('--pc_file', type=str, required=True)
    parser.add_argument('--cov_file', type=str, required=True)
    parser.add_argument('--ses_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--phenotype', type=str, default='T2D')

    # RF hyperparams
    parser.add_argument('--n_estimators', type=int, default=500)
    parser.add_argument('--max_depth', type=int, default=None)
    parser.add_argument('--min_samples_leaf', type=int, default=5)
    parser.add_argument('--max_features', type=str, default="sqrt")
    parser.add_argument('--class_weight', type=str, default="balanced")

    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--max_pcs', type=int, default=None)
    return parser.parse_args()

# ---------------------------------------------------
# RandomForest CV with OOF Prediction Storage
# ---------------------------------------------------
def train_and_get_predictions(X, y,
                              n_estimators=500,
                              max_depth=None,
                              min_samples_leaf=5,
                              max_features="sqrt",
                              class_weight="balanced",
                              random_state=42,
                              n_splits=5):

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    y_probs_full = np.zeros(len(y))
    y_true_full = np.zeros(len(y))

    auroc_scores = []
    auprc_scores = []
    accuracy_scores = []

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight=(None if class_weight == "None" else class_weight),
            n_jobs=-1,
            random_state=random_state
        )

        rf.fit(X_train, y_train)

        y_prob = rf.predict_proba(X_test)[:, 1]
        y_pred = rf.predict(X_test)

        auroc_scores.append(roc_auc_score(y_test, y_prob))
        auprc_scores.append(average_precision_score(y_test, y_prob))
        accuracy_scores.append(accuracy_score(y_test, y_pred))

        y_probs_full[test_idx] = y_prob
        y_true_full[test_idx] = y_test

    return {
        'AUROC': float(np.mean(auroc_scores)),
        'AUPRC': float(np.mean(auprc_scores)),
        'Accuracy': float(np.mean(accuracy_scores)),
        'y_prob_full': y_probs_full,
        'y_true_full': y_true_full
    }

# ---------------------------------------------------
# main
# ---------------------------------------------------
def main():
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load data ---
    df_labels = pd.read_csv(args.labels_file)
    df_prs    = pd.read_csv(args.prs_file)
    df_pc     = pd.read_csv(args.pc_file)
    df_cov    = pd.read_csv(args.cov_file)
    df_ses    = pd.read_csv(args.ses_file)

    y = df_labels[['eid', args.phenotype]].rename(columns={args.phenotype: 'label'})
    x = df_prs[['eid', args.phenotype]].rename(columns={args.phenotype: 'prs'})

    if 'lifestyle_label' not in df_cov.columns:
        raise KeyError("df_cov must contain a column named 'lifestyle_label'")
    cov = df_cov[['eid', 'age', 'sex', 'lifestyle_label']]

    ses_cols = [c for c in df_ses.columns if c != 'eid']
    pc_cols = [c for c in df_pc.columns if re.match(r"^PC\d+$", c)]
    pc_cols = sorted(pc_cols, key=lambda x: int(re.findall(r"\d+", x)[0]))
    if args.max_pcs is not None:
        pc_cols = pc_cols[:args.max_pcs]

    # --- Merge ---
    df = pd.merge(x, y, on='eid')
    df = pd.merge(df, cov, on='eid')
    df = pd.merge(df, df_pc[['eid'] + pc_cols], on='eid')
    df = pd.merge(df, df_ses[['eid'] + ses_cols], on='eid')

    df = df.dropna()
    y_final = df['label'].astype(int)

    base_cov   = ['age', 'sex']
    prs_feat   = ['prs']
    lifestyle  = ['lifestyle_label']
    ses_feats  = ses_cols
    idep_feats = pc_cols

    # --- Residualize: IDEP ~ SES + Lifestyle ---
    print("Fitting IDEP ~ SES + Lifestyle and computing residuals...")
    X_env = df[lifestyle + ses_feats].values
    linreg = LinearRegression()

    idep_resid_feats = []
    for pc in idep_feats:
        y_pc = df[pc].values
        linreg.fit(X_env, y_pc)
        resid = y_pc - linreg.predict(X_env)
        col_name = pc + "_resid"
        df[col_name] = resid
        idep_resid_feats.append(col_name)

    # --- Model definitions (base + full with residualized IDEPs) ---
    model_configs = [
        ("Cov_PRS", base_cov + prs_feat),
        ("Cov_PRS_SES_LS_IDEP_residual", base_cov + prs_feat + lifestyle + ses_feats + idep_resid_feats),
    ]

    results_store = {}
    final_metrics = []

    # --- Train Loop ---
    for model_name, features in model_configs:
        print(f"\nEvaluating RF model: {model_name}")
        X = df[features]

        res = train_and_get_predictions(
            X, y_final,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features,
            class_weight=args.class_weight,
            random_state=args.random_state,
            n_splits=args.n_splits
        )

        results_store[model_name] = res
        print(f"AUROC: {res['AUROC']:.4f}")

        final_metrics.append({
            'Model': model_name,
            'AUROC': res['AUROC'],
            'AUPRC': res['AUPRC'],
            'Accuracy': res['Accuracy'],
            'DeLong_P_Value': np.nan
        })

    # --- DeLong Test: Cov_PRS vs Cov_PRS_SES_LS_IDEP_residual ---
    print("\nCalculating DeLong P-value...")
    if "Cov_PRS" in results_store and "Cov_PRS_SES_LS_IDEP_residual" in results_store:
        p_val = delong_roc_test(
            results_store["Cov_PRS"]['y_true_full'],
            results_store["Cov_PRS"]['y_prob_full'],
            results_store["Cov_PRS_SES_LS_IDEP_residual"]['y_prob_full']
        )
        for rec in final_metrics:
            if rec['Model'] == 'Cov_PRS_SES_LS_IDEP_residual':
                rec['DeLong_P_Value'] = p_val

    # --- Save ---
    results_df = pd.DataFrame(final_metrics)
    output_file = os.path.join(
        args.output_dir,
        f"{args.phenotype}_RF_CovPRS_SES_Lifestyle_IDEP_comparison_DeLong.csv"
    )
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
