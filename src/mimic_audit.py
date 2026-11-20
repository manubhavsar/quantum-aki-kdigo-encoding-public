"""Does the leakage magnitude replicate on a second cohort?

The audit is measured on eICU, which invites the question of whether the
magnitudes are specific to that dataset. The MIMIC-IV demo is small (98 patients),
so most audited choices cannot be resolved there. Label leakage can: it is a
property of the labelling rule rather than of model comparison, and its effect on
absolute discrimination is large enough to measure on a small cohort.
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

H, HOR, SEED = 60.0, 24, 20260826


def build(include_prevalent):
    stays, cr, bun, ur = DP.load_mimic()
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
        for th in range(12, 97, 12):
            t = th * H
            if t > los:
                break
            if (ct <= t).sum() < 2:
                continue
            already = onset is not None and onset <= t
            if already and not include_prevalent:
                break
            end = t + HOR * H
            y = 1 if already else DP._label(onset, t, end, los)
            if y is None or not np.isfinite(y):
                continue
            f = DP._features(ct[ct <= t], cv[ct <= t], bt, bv, ut, uv, t, r)
            rows.append({"patient": r.uniquepid, "y": int(y),
                         "after_onset": bool(already), **f})
    return pd.DataFrame(rows)


def run(df, cols):
    y = df.y.values.astype(int); g = df.patient.values
    oof = np.full(len(df), np.nan)
    n = min(5, len(np.unique(g)))
    for tr, te in GroupKFold(n_splits=n).split(df, y, g):
        a, b = df.iloc[tr], df.iloc[te]
        if len(np.unique(y[tr])) < 2:
            continue
        med = a[cols].median()
        Xa = a[cols].fillna(med).values.astype(float)
        Xb = b[cols].fillna(med).values.astype(float)
        mu, sd = Xa.mean(0), Xa.std(0); sd[sd == 0] = 1
        oof[te] = LogisticRegression(max_iter=3000, class_weight="balanced",
                                     random_state=SEED).fit(
            (Xa - mu) / sd, y[tr]).predict_proba((Xb - mu) / sd)[:, 1]
    ok = np.isfinite(oof)
    return roc_auc_score(y[ok], oof[ok]), y.mean(), len(y), df.after_onset.mean()


def main():
    cols = S.BLOCKS["delta"]
    rows = []
    for inc, lab in ((False, "incident only"), (True, "post-onset windows retained")):
        d = build(inc)
        a, prev, n, frac = run(d, cols)
        rows.append({"cohort": "MIMIC-IV demo", "labelling": lab, "auroc": a,
                     "prevalence": prev, "n": n, "frac_after_onset": frac})
        print(f"  {lab:30s} n={n:4d} prev={prev:.3f} after-onset={frac:.1%} "
              f"AUROC={a:.3f}", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv("results/logs/mimic_audit.csv", index=False)
    print(f"\nleakage inflates absolute AUROC on MIMIC-IV by "
          f"{d.auroc.iloc[1]-d.auroc.iloc[0]:+.3f} (eICU: +0.193)")


if __name__ == "__main__":
    main()
