"""Label leakage expressed on the common estimand.

Every audited choice must move the SAME quantity if the magnitudes are to be
compared. The estimand is the quantum-classical gap

    G = AUROC(quantum) - AUROC(classical)

evaluated on identical samples. This script recomputes the leakage entry in that
space by fitting both a VQC and a logistic regression under the incident-only
labelling used throughout and under the leaky labelling that retains post-onset
windows, and reporting how far G moves.
"""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import splits as S
from quantum import VQC
from leakage_probe import build

SEED = 20260826


def fit_both(df, cols):
    y = df.y.values.astype(int)
    g = df.patient.values
    oq = np.full(len(df), np.nan); oc = np.full(len(df), np.nan)
    for tr, te in GroupKFold(n_splits=5).split(df, y, g):
        a, b = df.iloc[tr], df.iloc[te]
        med = a[cols].median()
        Xa = a[cols].fillna(med).values.astype(float)
        Xb = b[cols].fillna(med).values.astype(float)
        mu, sd = Xa.mean(0), Xa.std(0); sd[sd == 0] = 1
        Xa, Xb = (Xa - mu) / sd, (Xb - mu) / sd
        oc[te] = LogisticRegression(max_iter=3000, class_weight="balanced",
                                    random_state=SEED).fit(Xa, y[tr]).predict_proba(Xb)[:, 1]
        oq[te] = VQC(len(cols), n_layers=2, epochs=40, lr=0.1, scale=1.0,
                     seed=SEED).fit(Xa, y[tr]).predict_proba(Xb)
    aq, ac = roc_auc_score(y, oq), roc_auc_score(y, oc)
    return aq, ac, aq - ac, y.mean(), len(y)


def main():
    cols = S.BLOCKS["delta"]
    rows = []
    for inc, lab in ((False, "incident only (primary)"), (True, "post-onset windows retained")):
        df = build(inc)
        aq, ac, gap, prev, n = fit_both(df, cols)
        rows.append({"labelling": lab, "vqc": aq, "logreg": ac, "gap_G": gap,
                     "prevalence": prev, "n": n})
        print(f"{lab:30s} n={n:5d} prev={prev:.3f} "
              f"VQC={aq:.3f} LR={ac:.3f} G={gap:+.3f}", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv("results/logs/leakage_gap.csv", index=False)
    shift = d.gap_G.iloc[1] - d.gap_G.iloc[0]
    print(f"\nleakage shifts the quantum-classical gap G by {shift:+.3f} AUROC")
    print(f"(single-model AUROC inflation was {d.vqc.iloc[1]-d.vqc.iloc[0]:+.3f} for the VQC, "
          f"{d.logreg.iloc[1]-d.logreg.iloc[0]:+.3f} for logistic regression)")


if __name__ == "__main__":
    main()
