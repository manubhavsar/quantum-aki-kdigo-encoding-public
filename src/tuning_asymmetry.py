"""Artifact: comparing a tuned quantum model against an untuned classical one.

Bowles et al. identify unequal tuning budgets as a common source of apparent
quantum advantage. We quantify it directly: the same VQC is compared against
classical baselines at library-default settings and again after the nested
hyperparameter search used throughout this paper, on identical folds.
"""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import splits as S

HOR, BLOCK, SEED = 24, "delta", 20260826


def prep(tr, te, cols):
    med = tr[cols].median()
    Xtr = tr[cols].fillna(med).values.astype(float)
    Xte = te[cols].fillna(med).values.astype(float)
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
    return (Xtr - mu) / sd, (Xte - mu) / sd


def default_models():
    """Library defaults: no class weighting, no tuning -- the 'untuned baseline'."""
    return {"LogisticRegression": LogisticRegression(random_state=SEED),
            "RandomForest": RandomForestClassifier(random_state=SEED, n_jobs=-1),
            "XGBoost": XGBClassifier(eval_metric="logloss", random_state=SEED, n_jobs=-1)}


def main():
    df = S.load(HOR); cols = S.BLOCKS[BLOCK]
    y = df[f"y_{HOR}h"].values.astype(int)
    folds = list(GroupKFold(n_splits=5).split(df, y, df.patient.values))

    untuned = {k: np.full(len(df), np.nan) for k in default_models()}
    for tr_i, te_i in folds:
        Xtr, Xte = prep(df.iloc[tr_i], df.iloc[te_i], cols)
        for name, m in default_models().items():
            untuned[name][te_i] = m.fit(Xtr, y[tr_i]).predict_proba(Xte)[:, 1]

    vqc = np.load(f"results/preds/VQC_{BLOCK}_{HOR}h.npy")
    a_vqc = roc_auc_score(y, vqc)
    rows = []
    for name in default_models():
        a_un = roc_auc_score(y, untuned[name])
        a_tu = roc_auc_score(y, np.load(f"results/preds/{name}_{BLOCK}_{HOR}h.npy"))
        rows.append({"classical": name, "untuned_auroc": a_un, "tuned_auroc": a_tu,
                     "vqc_auroc": a_vqc,
                     "apparent_gap_vs_untuned": a_vqc - a_un,
                     "true_gap_vs_tuned": a_vqc - a_tu,
                     "inflation": (a_vqc - a_un) - (a_vqc - a_tu)})
        print(f"{name:20s} untuned={a_un:.3f} tuned={a_tu:.3f} | "
              f"VQC-vs-untuned={a_vqc-a_un:+.3f}  VQC-vs-tuned={a_vqc-a_tu:+.3f}  "
              f"inflation={(a_tu-a_un):+.3f}", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv("results/logs/tuning_asymmetry.csv", index=False)
    print(f"\nVQC (tuned) = {a_vqc:.3f}")
    print(f"apparent advantage over the best UNTUNED classical: "
          f"{a_vqc - max(r['untuned_auroc'] for r in rows):+.3f}")
    print(f"actual result vs the best TUNED classical:          "
          f"{a_vqc - max(r['tuned_auroc'] for r in rows):+.3f}")


if __name__ == "__main__":
    main()
