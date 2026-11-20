"""Death handling and a competing-risk sensitivity analysis.

The primary analysis censors a prediction point administratively when the stay
ends before the horizon closes without an AKI onset. Stays ending in death are
treated identically to stays ending in discharge. Death is a competing event: a
patient who dies cannot subsequently develop AKI, so censoring assumes the
unobserved outcome is missing rather than precluded.

This script quantifies how many points are affected and re-runs the primary
comparison under the alternative convention in which death before the horizon is
labelled a non-event, which is the other standard choice.
"""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import data_pipeline as DP
import splits as S
from kdigo import creatinine_aki_onset

H, HOR = 60.0, 24


def build(death_as_nonevent):
    stays, cr, bun, ur = DP.load_eicu()
    raw = pd.read_csv(f"{DP.EICU}/patient.csv.gz",
                      usecols=["patientunitstayid", "unitdischargestatus"])
    died = dict(zip(raw.patientunitstayid, raw.unitdischargestatus == "Expired"))
    crg = {k: (g.labresultoffset.values, g.labresult.values)
           for k, g in cr.groupby("patientunitstayid")}
    bung = {k: (g.labresultoffset.values, g.labresult.values)
            for k, g in bun.groupby("patientunitstayid")}
    urg = {k: (g.intakeoutputoffset.values, g.cellvaluenumeric.values)
           for k, g in ur.groupby("patientunitstayid")}
    rows = []
    for r in stays.itertuples(index=False):
        ct, cv = crg.get(r.patientunitstayid, (np.array([]), np.array([])))
        ct, cv = np.asarray(ct, float), np.asarray(cv, float)
        m = np.isfinite(ct) & np.isfinite(cv)
        ct, cv = ct[m], cv[m]
        o = np.argsort(ct); ct, cv = ct[o], cv[o]
        if len(ct) < 2:
            continue
        onset, _ = creatinine_aki_onset(ct, cv)
        bt, bv = bung.get(r.patientunitstayid, (np.array([]), np.array([])))
        ut, uv = urg.get(r.patientunitstayid, (np.array([]), np.array([])))
        los = r.unitdischargeoffset if np.isfinite(r.unitdischargeoffset) else np.inf
        d = bool(died.get(r.patientunitstayid, False))
        for th in range(12, 97, 12):
            t = th * H
            if t > los:
                break
            if (ct <= t).sum() < 2:
                continue
            if onset is not None and onset <= t:
                break
            end = t + HOR * H
            lab = DP._label(onset, t, end, los)
            censored_by_end = (onset is None and los < end)
            if censored_by_end and d and death_as_nonevent:
                lab = 0                      # died before horizon -> non-event
            if lab is None or not np.isfinite(lab):
                continue
            f = DP._features(ct[(ct <= t)], cv[(ct <= t)], bt, bv, ut, uv, t, r)
            rows.append({"stay": r.patientunitstayid, "patient": r.uniquepid,
                         "y": int(lab), "died": d,
                         "censored_by_stay_end": bool(censored_by_end), **f})
    return pd.DataFrame(rows)


def run(df, cols):
    y = df.y.values.astype(int); g = df.patient.values
    oof = np.full(len(df), np.nan)
    for tr, te in GroupKFold(n_splits=5).split(df, y, g):
        a, b = df.iloc[tr], df.iloc[te]
        med = a[cols].median()
        Xa = a[cols].fillna(med).values.astype(float)
        Xb = b[cols].fillna(med).values.astype(float)
        mu, sd = Xa.mean(0), Xa.std(0); sd[sd == 0] = 1
        oof[te] = LogisticRegression(max_iter=3000, class_weight="balanced",
                                     random_state=20260826).fit(
            (Xa - mu) / sd, y[tr]).predict_proba((Xb - mu) / sd)[:, 1]
    return roc_auc_score(y, oof), y.mean(), len(y)


def main():
    cols = S.BLOCKS["delta"]
    out = []
    base = build(False)
    n_death_censored = int((base.censored_by_stay_end & base.died).sum())
    print(f"prediction points at 24 h horizon (primary censoring): {len(base)}")
    print(f"  of which censored by stay end:        "
          f"{int(base.censored_by_stay_end.sum())}")
    print(f"  of which the stay ended in death:     {n_death_censored} "
          f"({100*n_death_censored/max(len(base),1):.2f}% of points)")
    for flag, lab in ((False, "death censored (primary)"),
                      (True, "death as non-event (sensitivity)")):
        d = build(flag)
        a, prev, n = run(d, cols)
        out.append({"convention": lab, "auroc": a, "prevalence": prev, "n": n})
        print(f"  {lab:34s} n={n:5d} prev={prev:.3f} AUROC={a:.3f}", flush=True)
    r = pd.DataFrame(out)
    r.to_csv("results/logs/competing_risk.csv", index=False)
    print(f"\nAUROC difference between conventions: "
          f"{r.auroc.iloc[1]-r.auroc.iloc[0]:+.4f}")


if __name__ == "__main__":
    main()
