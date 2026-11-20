"""Quantify the effect of NOT excluding already-AKI (prevalent) windows.

Our pipeline predicts *incident* AKI: once a stay meets KDIGO criteria we stop
emitting prediction points for it. A common alternative in AKI pipelines is to
slide a window across the whole stay and label "AKI within the next H hours"
without removing windows where AKI is already established. Those windows contain
an already-elevated creatinine and a label that is essentially a continuation of
the current state, so the task collapses towards "is this patient currently in
AKI" rather than "will they develop it".

This script builds both label sets from the same cohort and the same features and
reports the difference, so the size of that effect is measured rather than
asserted.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score

import data_pipeline as DP
from kdigo import creatinine_aki_onset
import splits as S

H = 60.0
HORIZON = 48


def build(include_prevalent):
    stays, cr, bun, ur = DP.load_eicu()
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
            hist = ct <= t
            if hist.sum() < 2:
                continue
            already = onset is not None and onset <= t
            if already and not include_prevalent:
                break                      # incident-only: stop at onset
            end = t + HORIZON * H
            if already:
                # window sits after onset; the "will they have AKI" label stays 1
                y = 1
            else:
                y = DP._label(onset, t, end, los)
            if not np.isfinite(y if y is not None else np.nan):
                continue
            f = DP._features(ct[hist], cv[hist], bt, bv, ut, uv, t, r)
            rows.append({"stay": r.patientunitstayid, "patient": r.uniquepid,
                         "t_hours": th, "y": int(y), "already_aki": bool(already), **f})
    return pd.DataFrame(rows)


def run(df, cols):
    y = df.y.values.astype(int)
    g = df.patient.values
    oof = np.full(len(df), np.nan)
    for tr, te in GroupKFold(n_splits=5).split(df, y, g):
        a, b = df.iloc[tr], df.iloc[te]
        med = a[cols].median()
        Xa = a[cols].fillna(med).values.astype(float)
        Xb = b[cols].fillna(med).values.astype(float)
        mu, sd = Xa.mean(0), Xa.std(0); sd[sd == 0] = 1
        m = LogisticRegression(max_iter=3000, class_weight="balanced",
                               random_state=20260826).fit((Xa - mu) / sd, y[tr])
        oof[te] = m.predict_proba((Xb - mu) / sd)[:, 1]
    return roc_auc_score(y, oof), average_precision_score(y, oof), y.mean(), len(y)


def main():
    cols = S.BLOCKS["combined"]
    out = []
    for inc in (False, True):
        df = build(inc)
        auroc, auprc, prev, n = run(df, cols)
        frac_prev = df.already_aki.mean()
        out.append({"include_prevalent": inc, "n": n, "prevalence": prev,
                    "frac_windows_after_onset": frac_prev,
                    "auroc": auroc, "auprc": auprc})
        print(f"include_prevalent={str(inc):5s}  n={n:5d}  prevalence={prev:.3f}  "
              f"windows_after_onset={frac_prev:.1%}  AUROC={auroc:.3f}  AUPRC={auprc:.3f}",
              flush=True)
    d = pd.DataFrame(out)
    d.to_csv("results/logs/leakage_probe.csv", index=False)
    delta = d.auroc.iloc[1] - d.auroc.iloc[0]
    print(f"\nAUROC inflation from including prevalent-AKI windows: {delta:+.3f}")

if __name__ == "__main__":
    main()
