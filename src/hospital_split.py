"""Hospital-grouped cross-validation: does the effect survive centre shift?

Patient-grouped splits control leakage between prediction points of one patient
but leave each fold drawing from the same set of hospitals. Grouping folds by
HOSPITAL instead forces every evaluation fold to come from centres unseen in
training, which is the shift a deployed model actually faces.
"""
import os, sys, warnings
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
from metrics import bootstrap_ci

SEED = 20260826


def prep(tr, te, cols):
    med = tr[cols].median()
    Xtr = tr[cols].fillna(med).values.astype(float)
    Xte = te[cols].fillna(med).values.astype(float)
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
    return (Xtr - mu) / sd, (Xte - mu) / sd


def main():
    rows = []
    for h in (24, 48):
        df = S.load(h)
        y = df[f"y_{h}h"].values.astype(int)
        hosp = df.hospital.values
        print(f"h={h}: {len(np.unique(hosp))} hospitals across {len(df)} points",
              flush=True)
        for block in ("static", "delta"):
            cols = S.BLOCKS[block]
            oof = {k: np.full(len(df), np.nan)
                   for k in ("VQC", "LogisticRegression", "RandomForest", "XGBoost")}
            for tr_i, te_i in GroupKFold(n_splits=5).split(df, y, hosp):
                tr, te = df.iloc[tr_i], df.iloc[te_i]
                Xtr, Xte = prep(tr, te, cols)
                ytr = y[tr_i]
                if len(np.unique(ytr)) < 2:
                    continue
                pw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
                oof["LogisticRegression"][te_i] = LogisticRegression(
                    max_iter=3000, C=1.0, class_weight="balanced",
                    random_state=SEED).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
                oof["RandomForest"][te_i] = RandomForestClassifier(
                    n_estimators=300, max_depth=5, min_samples_leaf=5,
                    class_weight="balanced", random_state=SEED,
                    n_jobs=-1).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
                oof["XGBoost"][te_i] = XGBClassifier(
                    n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                    colsample_bytree=0.8, scale_pos_weight=pw, eval_metric="logloss",
                    random_state=SEED, n_jobs=-1).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
                oof["VQC"][te_i] = VQC(len(cols), n_layers=2, epochs=40, lr=0.1,
                                       scale=1.0, seed=SEED).fit(
                    Xtr, ytr).predict_proba(Xte)
            for name, p in oof.items():
                ok = np.isfinite(p)
                if ok.sum() < 50:
                    continue
                lo, hi = bootstrap_ci(y[ok], p[ok], groups=df.patient.values[ok],
                                      n_boot=800)
                rows.append({"horizon": h, "block": block, "model": name,
                             "auroc": roc_auc_score(y[ok], p[ok]),
                             "auprc": average_precision_score(y[ok], p[ok]),
                             "auroc_lo": lo, "auroc_hi": hi, "n": int(ok.sum())})
                print(f"  {block:7s} {name:18s} AUROC={rows[-1]['auroc']:.3f} "
                      f"[{lo:.3f},{hi:.3f}]", flush=True)
            pd.DataFrame(rows).to_csv("results/logs/hospital_split.csv", index=False)
    d = pd.DataFrame(rows)
    print("\n=== delta-minus-static gain under hospital-grouped CV ===")
    for h in d.horizon.unique():
        for m in d.model.unique():
            s = d[(d.horizon == h) & (d.model == m)]
            if len(s) == 2:
                g = (s[s.block == "delta"].auroc.iloc[0]
                     - s[s.block == "static"].auroc.iloc[0])
                print(f"  h={h} {m:18s} {g:+.3f}")


if __name__ == "__main__":
    main()
