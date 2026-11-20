"""Give the quantum arm the same curvature features as the classical arm.

The curvature baseline supplies explicit trajectory-curvature features to the
classical models only, so "classical + curvature beats every quantum model"
compares arms with different inputs. This closes that gap: one representative
feature from each curvature family named in the paper (second difference,
acceleration, direction changes, dispersion) is appended to the delta block, and
the SAME twelve features go to a logistic regression and to a 12-qubit VQC on
identical folds. A fixed, untuned configuration is used for both arms and for the
8-feature reference, so the comparison isolates what the curvature features add
to each arm rather than differences in tuning.
"""
import os, sys, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import roc_auc_score

import splits as S
from quantum import VQC
from curvature_baseline import curvature_features
from metrics import bootstrap_ci

SEED = 20260826
CURV = ["curv_last", "accel", "n_sign_changes", "traj_std"]


def prep(df, tr, cols):
    med = df.iloc[tr][cols].median()
    X = df[cols].fillna(med).fillna(0.0).values.astype(float)
    mu, sd = X[tr].mean(0), X[tr].std(0); sd[sd == 0] = 1
    return (X - mu) / sd


def run(df, y, cols, kind):
    g = df.patient.values
    oof = np.full(len(df), np.nan)
    for tr, te in GroupKFold(n_splits=5).split(df, y, g):
        X = prep(df, tr, cols)
        if kind == "LR":
            m = LogisticRegression(max_iter=2000).fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
        else:
            tr_df = df.iloc[tr]
            gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
            _, i_va = next(gss.split(tr_df, y[tr], tr_df.patient.values))
            b = tr[i_va]
            m = VQC(len(cols), n_layers=2, epochs=60, lr=0.1, scale=1.0,
                    seed=SEED).fit(X[tr], y[tr], X[b], y[b])
            oof[te] = m.predict_proba(X[te])
    return oof


def main():
    seq_all = np.load("data/processed/eicu_seq.npy")
    full = pd.read_csv("data/processed/eicu_samples.csv")
    curv_all = curvature_features(seq_all)
    rows = []
    for h in (24, 48):
        df = S.load(h).reset_index(drop=True)
        sel = full.dropna(subset=[f"y_{h}h"]).index.values
        df = pd.concat([df, curv_all.iloc[sel].reset_index(drop=True)[CURV]], axis=1)
        y = df[f"y_{h}h"].values.astype(int)
        g = df.patient.values
        blocks = {"delta": S.BLOCKS["delta"], "delta+curv4": S.BLOCKS["delta"] + CURV}
        for kind in ("LR", "VQC"):
            for bname, cols in blocks.items():
                t0 = time.time()
                oof = run(df, y, cols, kind)
                au = roc_auc_score(y, oof)
                lo, hi = bootstrap_ci(y, oof, groups=g, n_boot=1000)
                np.save(f"results/preds/CQ_{kind}_{bname.replace('+','')}_{h}h.npy", oof)
                rows.append({"horizon": h, "model": kind, "block": bname,
                             "n_features": len(cols), "auroc": au, "lo": lo, "hi": hi})
                print(f"h={h} {kind:3s} {bname:12s} ({len(cols)} feat) AUROC={au:.3f} "
                      f"[{lo:.3f},{hi:.3f}]  {time.time()-t0:.0f}s", flush=True)
                pd.DataFrame(rows).to_csv("results/logs/curvature_quantum.csv", index=False)


if __name__ == "__main__":
    main()
