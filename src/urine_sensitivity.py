"""Discrimination when the KDIGO urine-output criterion (C3) is included (R1.5).

The primary endpoint is creatinine-only, because charting density cannot support
the six-hour oliguria rule on this cohort. Reviewers reasonably asked what that
exclusion costs clinically, so this rebuilds the labels with C3 included and
re-evaluates, holding the feature blocks, folds, seeds and models fixed.

Prediction points are re-derived against the urine-inclusive onset: a point is
kept only if no onset (by either criterion) has occurred at or before it, so the
target stays incident AKI under the wider definition.
"""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score

import splits as S
from metrics import bootstrap_ci

HORIZONS = [12, 24, 48]
SEED = 20260826


def build(horizon):
    samples = pd.read_csv("data/processed/eicu_samples.csv")
    stays = pd.read_csv("data/processed/eicu_stays.csv")
    m = samples.merge(stays[["stay", "onset_min", "onset_uo_min", "los_min"]],
                      on="stay", how="left")
    m = m[m.cohort == "eicu"] if "cohort" in m.columns else m

    # earliest onset under either criterion, in hours
    on_cr = m.onset_min / 60.0
    on_uo = m.onset_uo_min / 60.0
    onset = pd.concat([on_cr, on_uo], axis=1).min(axis=1)

    t = m.t_hours.values
    o = onset.values
    keep = np.isnan(o) | (o > t)                       # incident only
    lab = (~np.isnan(o)) & (o > t) & (o <= t + horizon)  # onset inside the horizon
    # administratively censor points whose horizon runs past the end of the stay
    # without an onset, exactly as the primary endpoint does
    los_h = m.los_min.values / 60.0
    censored = (~lab) & ((t + horizon) > los_h)
    m = m.assign(_y=lab.astype(int))[keep & ~censored]
    return m


def main():
    rows = []
    for h in HORIZONS:
        base = S.load(h)
        cols = [c for c in S.BLOCKS["delta"] if c in base.columns]
        uo = build(h)
        uo = uo.dropna(subset=["patient"])
        for name, df, y in (("creatinine-only", base, base[f"y_{h}h"].values.astype(int)),
                            ("creatinine+urine", uo, uo._y.values.astype(int))):
            use = [c for c in cols if c in df.columns]
            X = df[use].copy()
            X = X.fillna(X.median()).values.astype(float)
            g = df.patient.values
            oof = np.full(len(df), np.nan)
            for tr, te in GroupKFold(n_splits=5).split(X, y, g):
                mu, sd = X[tr].mean(0), X[tr].std(0); sd[sd == 0] = 1
                Z = (X - mu) / sd
                lr = LogisticRegression(max_iter=2000, class_weight="balanced")
                lr.fit(Z[tr], y[tr])
                oof[te] = lr.predict_proba(Z[te])[:, 1]
            au = roc_auc_score(y, oof)
            ap = average_precision_score(y, oof)
            lo, hi = bootstrap_ci(y, oof, groups=g, n_boot=1000)
            rows.append({"horizon": h, "endpoint": name, "n": len(df),
                         "n_pos": int(y.sum()), "prevalence": float(y.mean()),
                         "auroc": au, "auroc_lo": lo, "auroc_hi": hi, "auprc": ap})
            print(f"h={h:2d} {name:16s} n={len(df):5d} prev={y.mean():.3f} "
                  f"AUROC={au:.3f} [{lo:.3f},{hi:.3f}] AUPRC={ap:.3f}", flush=True)
        pd.DataFrame(rows).to_csv("results/logs/urine_sensitivity.csv", index=False)
    print("\nsaved results/logs/urine_sensitivity.csv")


if __name__ == "__main__":
    main()
