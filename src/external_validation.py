"""External validation on MIMIC-IV demo (independent hospital system).

Models are trained on the *entire* eICU cohort and evaluated on MIMIC-IV without
any refitting, so this is a true external test rather than a transfer-learned one.
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

import splits as S
from metrics import point_metrics, bootstrap_ci
from quantum import VQC

SEED = 20260826
HORIZONS = [12, 24, 48]
BLOCKS = ["static", "delta", "combined"]


def prep(tr, te, cols):
    med = tr[cols].median()
    Xtr = tr[cols].fillna(med).values.astype(float)
    Xte = te[cols].fillna(med).values.astype(float)
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
    return (Xtr - mu) / sd, (Xte - mu) / sd


def main():
    rows = []
    for h in HORIZONS:
        tr = S.load(h, "data/processed/eicu_samples.csv")
        te = S.load(h, "data/processed/mimic_samples.csv")
        ycol = f"y_{h}h"
        ytr, yte = tr[ycol].values.astype(int), te[ycol].values.astype(int)
        if len(np.unique(yte)) < 2:
            print(f"h={h}: external set has one class only, skipping"); continue
        gte = te.patient.values
        for block in BLOCKS:
            cols = S.BLOCKS[block]
            Xtr, Xte = prep(tr, te, cols)
            pw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
            models = {
                "LogisticRegression": LogisticRegression(max_iter=2000, C=1.0,
                                                         class_weight="balanced",
                                                         random_state=SEED),
                "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=5,
                                                       min_samples_leaf=5,
                                                       class_weight="balanced",
                                                       random_state=SEED, n_jobs=-1),
                "XGBoost": XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                         subsample=0.8, colsample_bytree=0.8,
                                         scale_pos_weight=pw, eval_metric="logloss",
                                         random_state=SEED, n_jobs=-1),
            }
            for name, m in models.items():
                m.fit(Xtr, ytr)
                p = m.predict_proba(Xte)[:, 1]
                _rec(rows, h, block, name, yte, p, gte)
            if block in ("static", "delta"):
                q = VQC(len(cols), n_layers=3, epochs=60, lr=0.1, scale=0.6,
                        seed=SEED).fit(Xtr, ytr)
                _rec(rows, h, block, "VQC", yte, q.predict_proba(Xte), gte)
    pd.DataFrame(rows).to_csv("results/logs/external_validation.csv", index=False)
    print("\nsaved results/logs/external_validation.csv")


def _rec(rows, h, block, name, y, p, g):
    mt = point_metrics(y, p)
    lo, hi = bootstrap_ci(y, p, groups=g, n_boot=1000)
    rows.append({"horizon": h, "block": block, "model": name, **mt,
                 "auroc_lo": lo, "auroc_hi": hi})
    print(f"EXT h={h:2d} {block:9s} {name:18s} AUROC={mt['auroc']:.3f} "
          f"[{lo:.3f},{hi:.3f}] AUPRC={mt['auprc']:.3f} (n={mt['n']}, "
          f"prev={mt['prevalence']:.3f})", flush=True)


if __name__ == "__main__":
    main()
