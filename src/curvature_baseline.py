"""The adversarial classical control: give classical models explicit curvature.

The paper argues the interleaved circuit represents trajectory structure beyond a
first difference. The obvious objection is that such structure can be computed
classically. This builds that competitor directly -- second differences,
acceleration, direction changes and dispersion over the same trajectory, supplied
to the same classical families -- and asks whether any quantum advantage survives
once the classical arm is given the very structure the circuit is claimed to
capture. If it does not, the honest conclusion is that the structure is classically
accessible, which is worth reporting either way.
"""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

import splits as S
from metrics import bootstrap_ci

SEED = 20260826


def curvature_features(seq):
    c = seq[..., 0]
    d1 = np.diff(c, axis=1)
    d2 = np.diff(d1, axis=1)
    return pd.DataFrame({
        "curv_mean": np.nanmean(d2, axis=1),
        "curv_last": d2[:, -1],
        "curv_absmax": np.nanmax(np.abs(d2), axis=1),
        "accel": d1[:, -1] - d1[:, 0],
        "n_sign_changes": np.sum(np.diff(np.sign(d1), axis=1) != 0, axis=1),
        "traj_std": np.nanstd(c, axis=1),
        "traj_range": np.nanmax(c, axis=1) - np.nanmin(c, axis=1),
        "peak_position": np.argmax(c, axis=1) / max(c.shape[1] - 1, 1),
    })


def prep(tr, te, cols):
    med = tr[cols].median()
    Xtr = tr[cols].fillna(med).values.astype(float)
    Xte = te[cols].fillna(med).values.astype(float)
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
    return (Xtr - mu) / sd, (Xte - mu) / sd


def main():
    seq = np.load("data/processed/eicu_seq.npy")
    full = pd.read_csv("data/processed/eicu_samples.csv")
    curv_all = curvature_features(seq)
    rows = []
    for h in (24, 48):
        df = S.load(h).reset_index(drop=True)
        sel = full.dropna(subset=[f"y_{h}h"]).index.values
        df = pd.concat([df, curv_all.iloc[sel].reset_index(drop=True)], axis=1)
        y = df[f"y_{h}h"].values.astype(int)
        g = df.patient.values
        blocks = {"delta": S.BLOCKS["delta"],
                  "delta+curvature": S.BLOCKS["delta"] + list(curv_all.columns)}
        for bname, cols in blocks.items():
            for mname in ("LogisticRegression", "RandomForest", "XGBoost"):
                oof = np.full(len(df), np.nan)
                for tr_i, te_i in GroupKFold(n_splits=5).split(df, y, g):
                    Xtr, Xte = prep(df.iloc[tr_i], df.iloc[te_i], cols)
                    ytr = y[tr_i]
                    pw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
                    m = {"LogisticRegression": LogisticRegression(
                             max_iter=3000, class_weight="balanced", random_state=SEED),
                         "RandomForest": RandomForestClassifier(
                             n_estimators=300, max_depth=5, min_samples_leaf=5,
                             class_weight="balanced", random_state=SEED, n_jobs=-1),
                         "XGBoost": XGBClassifier(
                             n_estimators=200, max_depth=3, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, scale_pos_weight=pw,
                             eval_metric="logloss", random_state=SEED, n_jobs=-1)}[mname]
                    oof[te_i] = m.fit(Xtr, ytr).predict_proba(Xte)[:, 1]
                lo, hi = bootstrap_ci(y, oof, groups=g, n_boot=800)
                rows.append({"horizon": h, "block": bname, "model": mname,
                             "auroc": roc_auc_score(y, oof), "lo": lo, "hi": hi})
                np.save(f"results/preds/CB_{mname}_{bname.replace('+','')}_{h}h.npy", oof)  # CB_ prefix: never overwrite the tuned Table 1 predictions
                print(f"h={h:2d} {bname:16s} {mname:18s} "
                      f"AUROC={rows[-1]['auroc']:.3f} [{lo:.3f},{hi:.3f}]", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv("results/logs/curvature_baseline.csv", index=False)
    print("\n=== gain to the classical arm from explicit curvature ===")
    for h in (24, 48):
        for m in d.model.unique():
            s = d[(d.horizon == h) & (d.model == m)]
            if len(s) == 2:
                print(f"  h={h} {m:18s} "
                      f"{s[s.block=='delta+curvature'].auroc.iloc[0]-s[s.block=='delta'].auroc.iloc[0]:+.3f}")
    print("\nbest classical WITH curvature: " + ", ".join(
        f"h={h}: {v:.3f}" for h, v in
        d[d.block == 'delta+curvature'].groupby('horizon').auroc.max().items()))


if __name__ == "__main__":
    main()
