"""Urine-inclusive endpoint for the kernel SVM and the tabular transformer, so the
sensitivity analysis covers all seven tabular model families (reviewer point R1.5).
Fixed, untuned protocol as in urine_sensitivity_vqc.py; the kernel SVM trains on at
most 700 rows per fold, all positives kept, as in the main analysis."""
import os, sys, warnings
import numpy as np, pandas as pd, torch
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC

import splits as S
import tab_transformer as tabt
from quantum import quantum_kernel_matrix
from urine_sensitivity import build

SEED = 20260826


def evaluate(df, y, cols):
    g = df.patient.values
    o = {k: np.full(len(df), np.nan) for k in ("tabt", "qsvm")}
    rng = np.random.default_rng(SEED)
    for tr, te in GroupKFold(n_splits=5).split(df, y, g):
        med = df.iloc[tr][cols].median()
        X = df[cols].fillna(med).fillna(0.0).values.astype(float)
        mu, sd = X[tr].mean(0), X[tr].std(0); sd[sd == 0] = 1; X = (X - mu) / sd
        tr_df = df.iloc[tr]
        a_i, b_i = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
                        .split(tr_df, y[tr], tr_df.patient.values))
        m = tabt.fit(X[tr[a_i]], y[tr[a_i]], X[tr[b_i]], y[tr[b_i]])
        with torch.no_grad():
            o["tabt"][te] = torch.sigmoid(m(torch.tensor(X[te], dtype=torch.float32))).numpy()
        pos, neg = np.where(y[tr] == 1)[0], np.where(y[tr] == 0)[0]
        sub = np.sort(np.concatenate([pos, rng.choice(neg, size=min(max(700 - len(pos), 50), len(neg)), replace=False)]))
        Xs, ys = X[tr][sub], y[tr][sub]
        svc = SVC(kernel="precomputed", C=1.0).fit(quantum_kernel_matrix(Xs, Xs, len(cols), scale=1.0), ys)
        o["qsvm"][te] = svc.decision_function(quantum_kernel_matrix(X[te], Xs, len(cols), scale=1.0))
    return {k: roc_auc_score(y, v) for k, v in o.items()}


def main():
    rows = []
    for h in (12, 24, 48):
        base = S.load(h).reset_index(drop=True)
        uo = build(h).dropna(subset=["patient"]).reset_index(drop=True)
        for name, df, y in (("creatinine-only", base, base[f"y_{h}h"].values.astype(int)),
                            ("creatinine+urine", uo, uo._y.values.astype(int))):
            r = evaluate(df, y, S.BLOCKS["delta"])
            rows.append({"horizon": h, "endpoint": name, **r})
            print(f"h={h:2d} {name:16s} TabT={r['tabt']:.3f} QSVM={r['qsvm']:.3f}", flush=True)
            pd.DataFrame(rows).to_csv("results/logs/urine_sensitivity_rest.csv", index=False)


if __name__ == "__main__":
    main()
