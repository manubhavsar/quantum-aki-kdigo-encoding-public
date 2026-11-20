"""'Best achievable' arm: classical and (PCA-compressed) quantum models on the
rich feature set, under the same patient-grouped nested CV as everything else.

This arm exists to establish the absolute discrimination attainable on this
cohort. It is deliberately NOT part of the 2x2 factorial, whose validity depends
on the two feature blocks being matched and narrow.
"""
import os, sys, json, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from sklearn.model_selection import GroupKFold, ParameterGrid, GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

import splits as S
from metrics import point_metrics, bootstrap_ci
from quantum import VQC

SEED = 20260826
HORIZONS = [12, 24, 48]
DEMOG = ["age", "sex_male", "weight_kg"]
DROP_PREFIX = ("albumin",)          # ~45% missing

GRIDS = {
    "LogisticRegression": {"C": [0.01, 0.1, 1.0]},
    "RandomForest": {"n_estimators": [400], "max_depth": [5, None],
                     "min_samples_leaf": [5, 20]},
    "XGBoost": {"n_estimators": [300], "max_depth": [3, 5],
                "learning_rate": [0.05], "subsample": [0.8],
                "colsample_bytree": [0.6], "min_child_weight": [5]},
}


def load_rich(horizon):
    base = S.load(horizon)
    rich = pd.read_csv("data/processed/eicu_rich.csv")
    df = base.merge(rich, on=["stay", "t_hours"], how="left")
    assert len(df) == len(base)
    feats = [c for c in rich.columns if c not in ("stay", "t_hours")
             and not c.startswith(DROP_PREFIX)]
    static = [c for c in feats if c.endswith(("_last", "_mean", "_min", "_max"))
              or c == "bun_cr_ratio"]
    delta = [c for c in feats if c.endswith(("_delta", "_rate"))]
    blocks = {"rich_static": DEMOG + static,
              "rich_delta": DEMOG + delta,
              "rich_all": DEMOG + feats}
    return df, blocks


def make(name, params, pw):
    if name == "LogisticRegression":
        return LogisticRegression(max_iter=4000, class_weight="balanced",
                                  random_state=SEED, **params)
    if name == "RandomForest":
        return RandomForestClassifier(class_weight="balanced", random_state=SEED,
                                      n_jobs=-1, **params)
    return XGBClassifier(eval_metric="logloss", random_state=SEED,
                         scale_pos_weight=pw, n_jobs=-1, tree_method="hist", **params)


def prep(tr, te, cols, k=None):
    med = tr[cols].median()
    Xtr = tr[cols].fillna(med).values.astype(float)
    Xte = te[cols].fillna(med).values.astype(float)
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    if k:
        p = PCA(n_components=k, random_state=SEED).fit(Xtr)
        Xtr, Xte = p.transform(Xtr), p.transform(Xte)
        s = Xtr.std(0); s[s == 0] = 1
        Xtr, Xte = Xtr / s, Xte / s
    return Xtr, Xte


def nested(df, y, cols, name):
    oof = np.full(len(df), np.nan)
    for tr_i, te_i in GroupKFold(n_splits=5).split(df, y, df.patient.values):
        tr, te = df.iloc[tr_i], df.iloc[te_i]
        ytr = y[tr_i]
        pw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
        best, bs = None, -np.inf
        for params in ParameterGrid(GRIDS[name]):
            sc = []
            for a_i, b_i in GroupKFold(n_splits=3).split(tr, ytr, tr.patient.values):
                if len(np.unique(ytr[a_i])) < 2 or len(np.unique(ytr[b_i])) < 2:
                    continue
                Xa, Xb = prep(tr.iloc[a_i], tr.iloc[b_i], cols)
                m = make(name, params, pw).fit(Xa, ytr[a_i])
                sc.append(average_precision_score(ytr[b_i], m.predict_proba(Xb)[:, 1]))
            if sc and np.mean(sc) > bs:
                bs, best = np.mean(sc), params
        Xtr, Xte = prep(tr, te, cols)
        m = make(name, best or {}, pw).fit(Xtr, ytr)
        oof[te_i] = m.predict_proba(Xte)[:, 1]
    return oof


def quantum_pca(df, y, cols, k=8):
    oof = np.full(len(df), np.nan)
    for tr_i, te_i in GroupKFold(n_splits=5).split(df, y, df.patient.values):
        tr, te = df.iloc[tr_i], df.iloc[te_i]
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
        a_i, b_i = next(gss.split(tr, y[tr_i], tr.patient.values))
        best, bs = 0.6, -np.inf
        for bw in (0.3, 0.6, 1.0, 1.5):
            Xa, Xb = prep(tr.iloc[a_i], tr.iloc[b_i], cols, k=k)
            m = VQC(k, n_layers=2, epochs=40, lr=0.1, scale=bw, seed=SEED).fit(
                Xa, y[tr_i][a_i], Xb, y[tr_i][b_i])
            s = average_precision_score(y[tr_i][b_i], m.predict_proba(Xb))
            if s > bs:
                bs, best = s, bw
        Xtr, Xte = prep(tr, te, cols, k=k)
        m = VQC(k, n_layers=2, epochs=40, lr=0.1, scale=best, seed=SEED).fit(Xtr, y[tr_i])
        oof[te_i] = m.predict_proba(Xte)
    return oof


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "classical"
    rows = []
    for h in HORIZONS:
        df, blocks = load_rich(h)
        y = df[f"y_{h}h"].values.astype(int)
        g = df.patient.values
        jobs = []
        if which in ("classical", "all"):
            jobs += [(n, b) for b in blocks for n in GRIDS]
        if which in ("quantum", "all"):
            jobs += [("VQC_pca8", b) for b in ("rich_static", "rich_delta", "rich_all")]
        for name, block in jobs:
            t0 = time.time()
            cols = blocks[block]
            oof = (quantum_pca(df, y, cols) if name == "VQC_pca8"
                   else nested(df, y, cols, name))
            mt = point_metrics(y, oof)
            lo, hi = bootstrap_ci(y, oof, groups=g, n_boot=1000)
            lo2, hi2 = bootstrap_ci(y, oof, fn=average_precision_score, groups=g, n_boot=1000)
            rows.append({"horizon": h, "block": block, "model": name, **mt,
                         "auroc_lo": lo, "auroc_hi": hi,
                         "auprc_lo": lo2, "auprc_hi": hi2, "n_features": len(cols)})
            np.save(f"results/preds/{name}_{block}_{h}h.npy", oof)
            print(f"h={h:2d} {block:12s} {name:18s} AUROC={mt['auroc']:.3f} "
                  f"[{lo:.3f},{hi:.3f}] AUPRC={mt['auprc']:.3f} "
                  f"({len(cols)} feats, {time.time()-t0:.0f}s)", flush=True)
            out = f"results/logs/rich_{which}_results.csv"
            pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nsaved results/logs/rich_{which}_results.csv")


if __name__ == "__main__":
    main()
