#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import os
import argparse
import numpy as np
import pandas as pd
import warnings
from scipy.stats import ttest_rel

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

warnings.filterwarnings("ignore")

# ---------------------------------------------------
# Argument parser
# ---------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Incremental value of IDEPs (Cox model, K-Fold CV Mean/SD + Paired T-test + NNS)'
    )
    parser.add_argument('--incidence_file', type=str, required=True)
    parser.add_argument('--prs_file',       type=str, required=True)
    parser.add_argument('--pc_file',        type=str, required=True)
    parser.add_argument('--cov_file',       type=str, required=True)
    parser.add_argument('--ses_file',       type=str, required=True)
    parser.add_argument('--output_dir',     type=str, default='./results')
    parser.add_argument('--phenotype',      type=str, default='T2D')
    parser.add_argument('--l1_ratio',       type=float, default=0.5)
    parser.add_argument('--C',              type=float, default=1.0)
    parser.add_argument('--n_splits',       type=int,   default=10)
    parser.add_argument('--random_state',   type=int,   default=42)
    parser.add_argument('--max_pcs',        type=int,   default=None)
    parser.add_argument('--eval_time_pct',  type=float, default=0.75,
                        help='Follow-up time quantile for survival probability evaluation (default: 75th pct)')
    parser.add_argument('--screen_top_pct', type=float, default=0.10,
                        help='Top-risk fraction selected for NNS calculation (default: top 10%%)')
    return parser.parse_args()


# ---------------------------------------------------
# Cox cross-validation (Return List of C-indices)
# ---------------------------------------------------
def train_and_evaluate_cox_cv_list(
        X, time, event,
        l1_ratio=0.5, C=1.0,
        n_splits=10, random_state=42
):
    n_events = int(event.sum())
    if n_events < 10:
        return [np.nan] * n_splits

    cindex_list = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        X_train = X.iloc[train_idx].copy()
        X_test  = X.iloc[test_idx].copy()
        t_train = time.iloc[train_idx].copy()
        e_train = event.iloc[train_idx].copy()
        t_test  = time.iloc[test_idx].copy()
        e_test  = event.iloc[test_idx].copy()

        pc_cols_local    = [c for c in X_train.columns if re.match(r"^PC\d+($|_resid$)", c)]
        non_pc_features  = [c for c in X_train.columns if c not in pc_cols_local]

        if len(non_pc_features) > 0:
            scaler = StandardScaler()
            X_train[non_pc_features] = scaler.fit_transform(X_train[non_pc_features])
            X_test[non_pc_features]  = scaler.transform(X_test[non_pc_features])

        penalizer = 0.0 if C <= 0 else 1.0 / C
        cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)

        df_train = X_train.copy()
        df_train["time"]  = t_train.values
        df_train["event"] = e_train.values

        try:
            cph.fit(df_train, duration_col="time", event_col="event", show_progress=False)
            risk_scores = cph.predict_partial_hazard(X_test)
            c_idx = concordance_index(t_test.values, -risk_scores.values, e_test.values)
            cindex_list.append(c_idx)
        except Exception:
            cindex_list.append(np.nan)

    return cindex_list


# ---------------------------------------------------
# Helper: Perform Paired T-test
# ---------------------------------------------------
def perform_paired_ttest(scores_a, scores_b):
    valid_indices = [i for i in range(len(scores_a))
                     if not np.isnan(scores_a[i]) and not np.isnan(scores_b[i])]
    if len(valid_indices) < 3:
        return np.nan
    a_clean = [scores_a[i] for i in valid_indices]
    b_clean = [scores_b[i] for i in valid_indices]
    try:
        _, p_val = ttest_rel(a_clean, b_clean)
        return p_val
    except Exception:
        return np.nan


# ===================================================================
# NNS (Number Needed to Screen)
# ===================================================================

def _get_survival_prob_at_time(cph, X_test, eval_time):
    try:
        surv_df = cph.predict_survival_function(X_test, times=[eval_time])
        prob_survive = surv_df.iloc[0].values
        risk_prob    = 1.0 - prob_survive
        return risk_prob
    except Exception:
        return None


def compute_nns(risk_base, risk_idep, event, screen_top_pct=0.10):
    """
    Compute NNS (Number Needed to Screen) for base and IDEP models.
    NNS = n_screen / cases_found (lower is better).
    """
    risk_base = np.array(risk_base, dtype=float)
    risk_idep = np.array(risk_idep, dtype=float)
    event     = np.array(event,     dtype=int)

    n_total  = len(event)
    n_screen = max(1, int(np.ceil(n_total * screen_top_pct)))

    idx_base = np.argsort(risk_base)[::-1][:n_screen]
    idx_idep = np.argsort(risk_idep)[::-1][:n_screen]

    cases_base = int(event[idx_base].sum())
    cases_idep = int(event[idx_idep].sum())

    additional_cases = cases_idep - cases_base
    total_events = int(event.sum())

    NNS_base = round(n_screen / cases_base, 2) if cases_base > 0 else np.inf
    NNS_idep = round(n_screen / cases_idep, 2) if cases_idep > 0 else np.inf

    if np.isfinite(NNS_base) and np.isfinite(NNS_idep) and NNS_base > 0:
        NNS_improvement_pct = round((NNS_base - NNS_idep) / NNS_base * 100, 4)
    else:
        NNS_improvement_pct = np.nan

    return {
        "NNS_base":             NNS_base,
        "NNS_idep":             NNS_idep,
        "NNS_improvement_pct":  NNS_improvement_pct,
        "cases_base":           cases_base,
        "cases_idep":           cases_idep,
        "additional_cases":     additional_cases,
        "n_screen":             n_screen,
        "cohort_size":          n_total,
        "screen_pct":           screen_top_pct,
        "event_rate":           round(event.mean(), 4),
        "total_events":         total_events,
    }


def compute_nns_cv(
        X_base, X_idep,
        time, event,
        eval_time_pct=0.75,
        screen_top_pct=0.10,
        l1_ratio=0.5, C=1.0,
        n_splits=10, random_state=42
):
    """
    Cross-validation loop to compute fold-level NNS metrics.
    """
    n_events = int(event.sum())
    if n_events < 20:
        empty = {"NNS_base": np.nan, "NNS_idep": np.nan, "NNS_improvement_pct": np.nan}
        return {k: {"mean": np.nan, "sd": np.nan, "folds": []} for k in empty}

    event_times = time[event == 1]
    eval_time   = float(np.quantile(event_times, eval_time_pct))

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_NNS_base            = []
    fold_NNS_idep            = []
    fold_NNS_improvement_pct = []

    penalizer = 0.0 if C <= 0 else 1.0 / C

    for train_idx, test_idx in kf.split(X_base):
        Xb_train = X_base.iloc[train_idx].copy()
        Xb_test  = X_base.iloc[test_idx].copy()
        Xi_train = X_idep.iloc[train_idx].copy()
        Xi_test  = X_idep.iloc[test_idx].copy()

        t_train  = time.iloc[train_idx].copy()
        e_train  = event.iloc[train_idx].copy()
        t_test   = time.iloc[test_idx].copy()
        e_test   = event.iloc[test_idx].copy()

        def scale_features(X_tr, X_te):
            pc_pattern = [c for c in X_tr.columns if re.match(r"^PC\d+($|_resid$)", c)]
            non_pc     = [c for c in X_tr.columns if c not in pc_pattern]
            if non_pc:
                sc = StandardScaler()
                X_tr[non_pc] = sc.fit_transform(X_tr[non_pc])
                X_te[non_pc] = sc.transform(X_te[non_pc])
            return X_tr, X_te

        Xb_train, Xb_test = scale_features(Xb_train, Xb_test)
        Xi_train, Xi_test = scale_features(Xi_train, Xi_test)

        df_base_train = Xb_train.copy()
        df_base_train["time"]  = t_train.values
        df_base_train["event"] = e_train.values
        cph_base = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)

        df_idep_train = Xi_train.copy()
        df_idep_train["time"]  = t_train.values
        df_idep_train["event"] = e_train.values
        cph_idep = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)

        try:
            cph_base.fit(df_base_train, duration_col="time",
                         event_col="event", show_progress=False)
            cph_idep.fit(df_idep_train, duration_col="time",
                         event_col="event", show_progress=False)
        except Exception:
            for lst in [fold_NNS_base, fold_NNS_idep, fold_NNS_improvement_pct]:
                lst.append(np.nan)
            continue

        risk_base = _get_survival_prob_at_time(cph_base, Xb_test, eval_time)
        risk_idep = _get_survival_prob_at_time(cph_idep, Xi_test, eval_time)

        if risk_base is None or risk_idep is None:
            continue

        e_arr = e_test.values.astype(int)
        t_arr = t_test.values

        is_case = (e_arr == 1) & (t_arr <= eval_time)
        is_control = (t_arr > eval_time)
        valid_mask = is_case | is_control

        risk_base_valid = risk_base[valid_mask]
        risk_idep_valid = risk_idep[valid_mask]
        e_arr_valid     = is_case[valid_mask].astype(int)

        nns_res = compute_nns(risk_base_valid, risk_idep_valid, e_arr_valid,
                              screen_top_pct=screen_top_pct)

        fold_NNS_base.append(nns_res["NNS_base"])
        fold_NNS_idep.append(nns_res["NNS_idep"])
        fold_NNS_improvement_pct.append(nns_res["NNS_improvement_pct"])

    def summarize(lst):
        clean = [v for v in lst if not np.isnan(v) and not np.isinf(v)]
        if len(clean) == 0:
            return {"mean": np.nan, "sd": np.nan, "folds": lst}
        return {
            "mean":  round(float(np.mean(clean)),  6),
            "sd":    round(float(np.std(clean, ddof=1)), 6),
            "folds": lst
        }

    return {
        "NNS_base":              summarize(fold_NNS_base),
        "NNS_idep":              summarize(fold_NNS_idep),
        "NNS_improvement_pct":   summarize(fold_NNS_improvement_pct),
        "eval_time_days":   eval_time,
        "eval_time_pct":    eval_time_pct,
        "screen_top_pct":   screen_top_pct,
    }


# ===================================================================
# main
# ===================================================================
def main():
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load Data ---
    df_inc = pd.read_csv(args.incidence_file)
    df_inc = df_inc[df_inc["phenotype"] == args.phenotype].copy()

    if df_inc.empty:
        pd.DataFrame().to_csv(
            os.path.join(args.output_dir, f"{args.phenotype}_Cox_CV_Stats.csv"))
        return

    df_inc = df_inc[["eid", "label", "followup_days"]].rename(
        columns={"label": "event", "followup_days": "time"}
    )
    df_inc = df_inc[df_inc["time"] > 0].copy()

    df_prs = pd.read_csv(args.prs_file)
    df_pc  = pd.read_csv(args.pc_file)
    df_cov = pd.read_csv(args.cov_file)
    df_ses = pd.read_csv(args.ses_file)

    if args.phenotype not in df_prs.columns:
        raise KeyError(f"Phenotype '{args.phenotype}' not found in prs_file")

    prs      = df_prs[["eid", args.phenotype]].rename(columns={args.phenotype: "prs"})
    cov      = df_cov[["eid", "age", "sex", "lifestyle_label"]]
    ses_cols = [c for c in df_ses.columns if c != "eid"]
    pc_cols  = sorted([c for c in df_pc.columns if re.match(r"^PC\d+$", c)],
                      key=lambda x: int(re.findall(r"\d+", x)[0]))
    if args.max_pcs:
        pc_cols = pc_cols[:args.max_pcs]

    # --- Merge ---
    df = df_inc.merge(prs, on="eid", how="inner") \
               .merge(cov, on="eid", how="inner") \
               .merge(df_pc[["eid"] + pc_cols], on="eid", how="inner") \
               .merge(df_ses[["eid"] + ses_cols], on="eid", how="inner")

    df    = df.dropna().reset_index(drop=True)
    time  = df["time"]
    event = df["event"]

    print(f"Phenotype: {args.phenotype}, N: {len(df)}, Events: {int(event.sum())}")

    # --- Residualization ---
    base_cov        = ["age", "sex"]
    prs_feat        = ["prs"]
    lifestyle       = ["lifestyle_label"]
    ses_feats       = ses_cols
    idep_feats      = pc_cols
    idep_resid_feats = []

    X_env  = df[lifestyle + ses_feats].values
    linreg = LinearRegression()
    for pc in idep_feats:
        y_pc = df[pc].values
        linreg.fit(X_env, y_pc)
        df[pc + "_resid"] = y_pc - linreg.predict(X_env)
        idep_resid_feats.append(pc + "_resid")

    # --- Model Configs (base + full with residualized IDEPs) ---
    model_configs = [
        ("Cov_PRS",                       base_cov + prs_feat),
        ("Cov_PRS_SES_LS_IDEP_residual",  base_cov + prs_feat + lifestyle + ses_feats + idep_resid_feats),
    ]

    # --- Train & Collect C-index Stats ---
    score_store  = {}
    results_list = []

    for model_name, features in model_configs:
        print(f"Evaluating C-index: {model_name}")
        scores = train_and_evaluate_cox_cv_list(
            df[features].copy(), time, event,
            l1_ratio=args.l1_ratio, C=args.C,
            n_splits=args.n_splits, random_state=args.random_state
        )
        score_store[model_name] = scores
        mean_c = np.nanmean(scores)
        sd_c   = np.nanstd(scores, ddof=1)
        print(f"  > C-index: {mean_c:.4f} +/- {sd_c:.4f}")

        results_list.append({
            "Phenotype":      args.phenotype,
            "Model":          model_name,
            "C_index_Mean":   mean_c,
            "C_index_SD":     sd_c,
            "Ttest_P_Value":  np.nan,
        })

    # --- Paired T-test ---
    print("\nCalculating Paired T-test...")
    if "Cov_PRS" in score_store and "Cov_PRS_SES_LS_IDEP_residual" in score_store:
        p_val = perform_paired_ttest(score_store["Cov_PRS"],
                                     score_store["Cov_PRS_SES_LS_IDEP_residual"])
        for rec in results_list:
            if rec["Model"] == "Cov_PRS_SES_LS_IDEP_residual":
                rec["Ttest_P_Value"] = p_val
        print(f"  > Cov_PRS vs Cov_PRS_SES_LS_IDEP_residual: p={p_val:.4e}")

    # --- Save C-index & T-test results ---
    results_df = pd.DataFrame(results_list)
    out_cindex = os.path.join(args.output_dir, f"{args.phenotype}_Cox_CV_Stats.csv")
    results_df.to_csv(out_cindex, index=False)
    print(f"\nSaved C-index & T-test results to: {out_cindex}")

    # ===========================================================
    # NNS (Number Needed to Screen)
    # ===========================================================
    print("\n" + "=" * 60)
    print("[Contextualization] Computing NNS ...")
    print("=" * 60)

    context_comparisons = [
        ("Cov_PRS  vs  Cov_PRS_SES_LS_IDEP_residual",
         base_cov + prs_feat,
         base_cov + prs_feat + lifestyle + ses_feats + idep_resid_feats),
    ]

    context_rows = []

    for label, feat_base, feat_idep in context_comparisons:
        print(f"\n  -> {label}")
        ctx = compute_nns_cv(
            X_base        = df[feat_base].copy(),
            X_idep        = df[feat_idep].copy(),
            time          = time,
            event         = event,
            eval_time_pct = args.eval_time_pct,
            screen_top_pct= args.screen_top_pct,
            l1_ratio      = args.l1_ratio,
            C             = args.C,
            n_splits      = args.n_splits,
            random_state  = args.random_state,
        )

        print(f"     Eval time     : {ctx['eval_time_days']:.1f} days "
              f"(~{ctx['eval_time_days']/365:.1f} yrs, "
              f"{int(ctx['eval_time_pct']*100)}th pct of event times)")
        print(f"     NNS (base)    : {ctx['NNS_base']['mean']:.2f} "
              f"+/- {ctx['NNS_base']['sd']:.2f}")
        print(f"     NNS (IDEP)    : {ctx['NNS_idep']['mean']:.2f} "
              f"+/- {ctx['NNS_idep']['sd']:.2f}")
        print(f"     NNS improve%  : {ctx['NNS_improvement_pct']['mean']:.2f}%")

        context_rows.append({
            "Phenotype":                 args.phenotype,
            "Comparison":                label,
            "Eval_time_days":            round(ctx["eval_time_days"], 1),
            "Screen_top_pct":            ctx["screen_top_pct"],
            "NNS_base_mean":             ctx["NNS_base"]["mean"],
            "NNS_base_sd":               ctx["NNS_base"]["sd"],
            "NNS_idep_mean":             ctx["NNS_idep"]["mean"],
            "NNS_idep_sd":               ctx["NNS_idep"]["sd"],
            "NNS_improvement_pct_mean":  ctx["NNS_improvement_pct"]["mean"],
            "NNS_improvement_pct_sd":    ctx["NNS_improvement_pct"]["sd"],
        })

    # --- Save NNS results ---
    ctx_df  = pd.DataFrame(context_rows)
    out_ctx = os.path.join(args.output_dir,
                           f"{args.phenotype}_NNS.csv")
    ctx_df.to_csv(out_ctx, index=False)
    print(f"\nSaved NNS results to: {out_ctx}")


if __name__ == "__main__":
    main()
