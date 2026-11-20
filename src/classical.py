"""Classical baselines with nested, patient-grouped cross-validation.

Outer 5-fold GroupKFold gives unbiased out-of-fold predictions; an inner 3-fold
GroupKFold tunes hyperparameters within each training fold, so no tuning
information leaks into the outer estimate. Pooled out-of-fold predictions are
saved so that quantum and classical models can later be compared with DeLong's
test on identical samples.
"""
import os, json, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from sklearn.model_selection import GroupKFold, ParameterGrid
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

import splits as S
from metrics import point_metrics, bootstrap_ci

SEED = 20260826
HORIZONS = [12, 24, 48]
BLOCKS = ["static", "delta", "combined"]

GRIDS = {
    "LogisticRegression": {"C": [0.01, 0.1, 1.0, 10.0]},
    "RandomForest": {"n_estimators": [300], "max_depth": [3, 5, None],
                     "min_samples_leaf": [1, 5, 20]},
    "XGBoost": {"n_estimators": [200], "max_depth": [2, 3, 5],
                "learning_rate": [0.05, 0.1], "subsample": [0.8],
                "colsample_bytree": [0.8], "min_child_weight": [1, 5]},
}


def make(name, params, pos_weight):
    if name == "LogisticRegression":
        return LogisticRegression(max_iter=2000, class_weight="balanced",
                                  random_state=SEED, **params)
    if name == "RandomForest":
        return RandomForestClassifier(class_weight="balanced", random_state=SEED,
                                      n_jobs=-1, **params)
    if name == "XGBoost":
        return XGBClassifier(eval_metric="logloss", random_state=SEED,
                             scale_pos_weight=pos_weight, n_jobs=-1,
                             tree_method="hist", **params)
    raise ValueError(name)


def _prep(tr_df, te_df, cols):
    """Median-impute and standardise using training-fold statistics only."""
    med = tr_df[cols].median()
    Xtr = tr_df[cols].fillna(med).values.astype(float)
    Xte = te_df[cols].fillna(med).values.astype(float)
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1.0
    return (Xtr - mu) / sd, (Xte - mu) / sd


def nested_cv(df, horizon, block, model_name, n_outer=5, n_inner=3):
    cols = S.BLOCKS[block]
    y_col = f"y_{horizon}h"
    y_all = df[y_col].values.astype(int)
    groups = df.patient.values
    oof = np.full(len(df), np.nan)
    chosen = []

    outer = GroupKFold(n_splits=n_outer)
    for tr_idx, te_idx in outer.split(df, y_all, groups):
        tr_df, te_df = df.iloc[tr_idx], df.iloc[te_idx]
        ytr = y_all[tr_idx]
        pw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))

        # ---- inner loop: select hyperparameters on training fold only
        best, best_score = None, -np.inf
        inner = GroupKFold(n_splits=n_inner)
        for params in ParameterGrid(GRIDS[model_name]):
            scores = []
            for i_tr, i_va in inner.split(tr_df, ytr, tr_df.patient.values):
                a, b = tr_df.iloc[i_tr], tr_df.iloc[i_va]
                if len(np.unique(ytr[i_tr])) < 2 or len(np.unique(ytr[i_va])) < 2:
                    continue
                Xa, Xb = _prep(a, b, cols)
                m = make(model_name, params, pw).fit(Xa, ytr[i_tr])
                scores.append(average_precision_score(ytr[i_va], m.predict_proba(Xb)[:, 1]))
            if scores and np.mean(scores) > best_score:
                best_score, best = np.mean(scores), params
        best = best or {}
        chosen.append(best)

        # ---- refit on the full training fold, predict the held-out outer fold
        Xtr, Xte = _prep(tr_df, te_df, cols)
        m = make(model_name, best, pw).fit(Xtr, ytr)
        oof[te_idx] = m.predict_proba(Xte)[:, 1]

    return oof, chosen


def main():
    os.makedirs("results/preds", exist_ok=True)
    rows = []
    for horizon in HORIZONS:
        df = S.load(horizon)
        y = df[f"y_{horizon}h"].values.astype(int)
        groups = df.patient.values
        for block in BLOCKS:
            for name in GRIDS:
                oof, chosen = nested_cv(df, horizon, block, name)
                mt = point_metrics(y, oof)
                lo, hi = bootstrap_ci(y, oof, groups=groups, n_boot=1000)
                lo_pr, hi_pr = bootstrap_ci(y, oof, fn=average_precision_score,
                                            groups=groups, n_boot=1000)
                rec = {"horizon": horizon, "block": block, "model": name,
                       **mt, "auroc_lo": lo, "auroc_hi": hi,
                       "auprc_lo": lo_pr, "auprc_hi": hi_pr}
                rows.append(rec)
                np.save(f"results/preds/{name}_{block}_{horizon}h.npy", oof)
                print(f"h={horizon:2d} {block:8s} {name:18s} "
                      f"AUROC={mt['auroc']:.3f} [{lo:.3f},{hi:.3f}]  "
                      f"AUPRC={mt['auprc']:.3f} (prev {mt['prevalence']:.3f})", flush=True)
        # labels and groups saved once per horizon for later paired tests
        np.save(f"results/preds/_y_{horizon}h.npy", y)
        np.save(f"results/preds/_g_{horizon}h.npy", groups)
    pd.DataFrame(rows).to_csv("results/logs/classical_results.csv", index=False)
    print("\nsaved results/logs/classical_results.csv")


if __name__ == "__main__":
    main()
