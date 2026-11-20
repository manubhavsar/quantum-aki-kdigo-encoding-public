"""Learning curve: does the quantum model's relative standing improve at small N?

Pre-specified hypothesis, motivated by Caro et al. (2022), who show QML models can
generalize from few training data, and by Kubler et al. (2021), who argue quantum
advantage requires a low-dimensional, well-matched hypothesis class -- a bias that
should matter most when data are scarce.

Protocol fixed in advance: patient-grouped 5-fold CV; within each training fold,
subsample *patients* (not rows) to a fraction f; evaluate on the untouched test
fold. Three seeds per fraction. The FULL curve is reported whatever it shows.
"""
import os, sys, time, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier

import splits as S
from quantum import VQC

SEED = 20260826
HOR = 24
BLOCK = "delta"
FRACTIONS = [0.10, 0.20, 0.35, 0.50, 0.75, 1.00]
SEEDS = [int(x) for x in os.environ.get("LC_SEEDS", "0,1,2").split(",")]
APPEND = os.environ.get("LC_APPEND", "") == "1"


def prep(tr, te, cols):
    med = tr[cols].median()
    Xtr = tr[cols].fillna(med).values.astype(float)
    Xte = te[cols].fillna(med).values.astype(float)
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
    return (Xtr - mu) / sd, (Xte - mu) / sd


def subsample_patients(tr_df, y_tr, frac, rng):
    """Subsample patients, keeping at least a few positives so training is possible."""
    pats = np.unique(tr_df.patient.values)
    n = max(int(round(len(pats) * frac)), 8)
    chosen = rng.choice(pats, size=min(n, len(pats)), replace=False)
    m = tr_df.patient.isin(chosen).values
    if y_tr[m].sum() < 3:                       # ensure the minority class exists
        pos_pats = np.unique(tr_df.patient.values[y_tr == 1])
        extra = rng.choice(pos_pats, size=min(3, len(pos_pats)), replace=False)
        m = m | tr_df.patient.isin(extra).values
    return m


def main():
    df = S.load(HOR)
    cols = S.BLOCKS[BLOCK]
    y = df[f"y_{HOR}h"].values.astype(int)
    groups = df.patient.values
    rows = []

    for frac in FRACTIONS:
        for seed in SEEDS:
            rng = np.random.default_rng(SEED + seed)
            oof = {k: np.full(len(df), np.nan)
                   for k in ("VQC", "LogisticRegression", "XGBoost", "RandomForest")}
            for tr_i, te_i in GroupKFold(n_splits=5).split(df, y, groups):
                tr_df, te_df = df.iloc[tr_i], df.iloc[te_i]
                ytr = y[tr_i]
                m = subsample_patients(tr_df, ytr, frac, rng)
                sub, ysub = tr_df[m], ytr[m]
                if len(np.unique(ysub)) < 2:
                    continue
                Xtr, Xte = prep(sub, te_df, cols)
                pw = float((ysub == 0).sum() / max((ysub == 1).sum(), 1))
                oof["LogisticRegression"][te_i] = LogisticRegression(
                    max_iter=3000, C=1.0, class_weight="balanced",
                    random_state=SEED).fit(Xtr, ysub).predict_proba(Xte)[:, 1]
                oof["RandomForest"][te_i] = RandomForestClassifier(
                    n_estimators=300, max_depth=5, min_samples_leaf=5,
                    class_weight="balanced", random_state=SEED,
                    n_jobs=-1).fit(Xtr, ysub).predict_proba(Xte)[:, 1]
                oof["XGBoost"][te_i] = XGBClassifier(
                    n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                    colsample_bytree=0.8, scale_pos_weight=pw, eval_metric="logloss",
                    random_state=SEED, n_jobs=-1).fit(Xtr, ysub).predict_proba(Xte)[:, 1]
                oof["VQC"][te_i] = VQC(len(cols), n_layers=2, epochs=40, lr=0.1,
                                       scale=1.0, seed=SEED).fit(
                    Xtr, ysub).predict_proba(Xte)
            for k, p in oof.items():
                ok = np.isfinite(p)
                if ok.sum() < 50 or len(np.unique(y[ok])) < 2:
                    continue
                rows.append({"frac": frac, "seed": seed, "model": k,
                             "n_train_patients": int(round(
                                 len(np.unique(groups)) * 0.8 * frac)),
                             "auroc": roc_auc_score(y[ok], p[ok]),
                             "auprc": average_precision_score(y[ok], p[ok])})
            out = pd.DataFrame(rows)
            if APPEND and os.path.exists("results/logs/learning_curve.csv"):
                prev = pd.read_csv("results/logs/learning_curve.csv")
                prev = prev[~prev.seed.isin(SEEDS)]
                out = pd.concat([prev, out], ignore_index=True)
            out.to_csv("results/logs/learning_curve.csv", index=False)
        d = pd.DataFrame(rows)
        d = d[d.frac == frac].groupby("model").auroc.mean()
        print(f"frac={frac:.2f}  " + "  ".join(f"{k}={v:.3f}" for k, v in d.items()),
              flush=True)

    d = pd.read_csv("results/logs/learning_curve.csv")
    piv = d.groupby(["frac", "model"]).auroc.mean().unstack()
    print("\n=== mean AUROC over 3 seeds ===")
    print(piv.round(3).to_string())
    best_classical = piv[["LogisticRegression", "XGBoost", "RandomForest"]].max(axis=1)
    print("\n=== VQC minus best classical (positive = quantum ahead) ===")
    print((piv["VQC"] - best_classical).round(3).to_string())
    print("\nsaved results/logs/learning_curve.csv")


if __name__ == "__main__":
    main()
