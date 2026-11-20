"""Urine-inclusive endpoint for the quantum model as well (reviewer point R1.5).

urine_sensitivity.py re-evaluated logistic regression under the C3-inclusive
endpoint. This does the same for the variational classifier, and refits an
untuned logistic regression alongside it, so the endpoint comparison covers both
arms under one protocol (fixed configuration, identical folds).
"""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import roc_auc_score

import splits as S
from quantum import VQC
from urine_sensitivity import build

SEED = 20260826


def evaluate(df, y, cols):
    g = df.patient.values
    out = {}
    oof_lr, oof_q = np.full(len(df), np.nan), np.full(len(df), np.nan)
    for tr, te in GroupKFold(n_splits=5).split(df, y, g):
        med = df.iloc[tr][cols].median()
        X = df[cols].fillna(med).fillna(0.0).values.astype(float)
        mu, sd = X[tr].mean(0), X[tr].std(0); sd[sd == 0] = 1
        X = (X - mu) / sd
        oof_lr[te] = LogisticRegression(max_iter=2000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        tr_df = df.iloc[tr]
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
        _, i_va = next(gss.split(tr_df, y[tr], tr_df.patient.values))
        b = tr[i_va]
        m = VQC(len(cols), n_layers=2, epochs=60, lr=0.1, scale=1.0,
                seed=SEED).fit(X[tr], y[tr], X[b], y[b])
        oof_q[te] = m.predict_proba(X[te])
    return roc_auc_score(y, oof_lr), roc_auc_score(y, oof_q)


def main():
    rows = []
    for h in (12, 24, 48):
        base = S.load(h).reset_index(drop=True)
        cols = S.BLOCKS["delta"]
        uo = build(h).dropna(subset=["patient"]).reset_index(drop=True)
        for name, df, y in (("creatinine-only", base, base[f"y_{h}h"].values.astype(int)),
                            ("creatinine+urine", uo, uo._y.values.astype(int))):
            lr, q = evaluate(df, y, cols)
            rows.append({"horizon": h, "endpoint": name, "n": len(df),
                         "prevalence": float(y.mean()), "lr": lr, "vqc": q,
                         "gap": q - lr})
            print(f"h={h:2d} {name:16s} LR={lr:.3f} VQC={q:.3f} G={q-lr:+.3f}", flush=True)
            pd.DataFrame(rows).to_csv("results/logs/urine_sensitivity_vqc.csv", index=False)


if __name__ == "__main__":
    main()
