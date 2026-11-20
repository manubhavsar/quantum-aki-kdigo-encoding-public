"""Artifact 6: how large an apparent advantage can seed selection alone produce?

If a quantum-classical comparison is run once and reported, the number obtained
depends on the random seed. We refit both models across seeds on identical folds
and report the full spread, so the size of a 'best-seed' claim can be read off
directly and compared with the genuine controlled effect.
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

HOR, BLOCK = 24, "delta"
SEEDS = list(range(8))


def prep(tr, te, cols):
    med = tr[cols].median()
    Xtr = tr[cols].fillna(med).values.astype(float)
    Xte = te[cols].fillna(med).values.astype(float)
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
    return (Xtr - mu) / sd, (Xte - mu) / sd


def main():
    df = S.load(HOR); cols = S.BLOCKS[BLOCK]
    y = df[f"y_{HOR}h"].values.astype(int)
    folds = list(GroupKFold(n_splits=5).split(df, y, df.patient.values))
    rows = []
    for s in SEEDS:
        oq = np.full(len(df), np.nan); oc = np.full(len(df), np.nan)
        for tr_i, te_i in folds:
            Xtr, Xte = prep(df.iloc[tr_i], df.iloc[te_i], cols)
            oq[te_i] = VQC(len(cols), n_layers=2, epochs=40, lr=0.1, scale=1.0,
                           seed=s).fit(Xtr, y[tr_i]).predict_proba(Xte)
            oc[te_i] = LogisticRegression(max_iter=3000, C=1.0,
                                          class_weight="balanced",
                                          random_state=s).fit(
                Xtr, y[tr_i]).predict_proba(Xte)[:, 1]
        aq, ac = roc_auc_score(y, oq), roc_auc_score(y, oc)
        rows.append({"seed": s, "vqc": aq, "logreg": ac, "gap": aq - ac})
        print(f"seed={s}  VQC={aq:.3f}  LR={ac:.3f}  gap={aq-ac:+.3f}", flush=True)
        pd.DataFrame(rows).to_csv("results/logs/seed_sensitivity.csv", index=False)
    d = pd.DataFrame(rows)
    print(f"\ngap across seeds: mean {d.gap.mean():+.3f}  "
          f"best {d.gap.max():+.3f}  worst {d.gap.min():+.3f}  "
          f"range {d.gap.max()-d.gap.min():.3f}")
    print(f"a best-seed report would claim {d.gap.max():+.3f}; "
          f"the seed-averaged truth is {d.gap.mean():+.3f}")


if __name__ == "__main__":
    main()
