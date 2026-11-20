"""Enhanced VQC search: removes a handicap the original protocol imposed.

The first-pass VQC fixed the data re-uploading depth at 1 and searched only
bandwidth and layer count. Data re-uploading is known to raise the expressivity
of a variational classifier substantially (Perez-Salinas et al., 2020), so
leaving it untuned understates what the quantum model can do. Here the search
covers re-uploading depth as well, on identical folds, so results remain paired
with every other model.

This widens the search for the quantum side only; the conclusion it can support
is therefore a *stronger* quantum result, not a weaker classical one. Classical
baselines already received fully nested tuning.
"""
import os, sys, time, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import average_precision_score

import splits as S
from metrics import point_metrics, bootstrap_ci
from quantum import VQC

SEED = 20260826
HORIZONS = [12, 24, 48]
BLOCKS = ["static", "delta"]
BANDWIDTHS = [0.3, 0.6, 1.0, 1.5]
LAYERS = [2, 3]
REUPLOADS = [1, 2, 3]


def prep(tr, te, cols):
    med = tr[cols].median()
    Xtr = tr[cols].fillna(med).values.astype(float)
    Xte = te[cols].fillna(med).values.astype(float)
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
    return (Xtr - mu) / sd, (Xte - mu) / sd


def run(df, horizon, block, epochs=40):
    cols = S.BLOCKS[block]
    y = df[f"y_{horizon}h"].values.astype(int)
    oof = np.full(len(df), np.nan)
    picks = []
    for tr_i, te_i in GroupKFold(n_splits=5).split(df, y, df.patient.values):
        tr_df, te_df = df.iloc[tr_i], df.iloc[te_i]
        ytr = y[tr_i]
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
        a_i, b_i = next(gss.split(tr_df, ytr, tr_df.patient.values))
        Xa, Xb = prep(tr_df.iloc[a_i], tr_df.iloc[b_i], cols)
        best, bs = None, -np.inf
        for bw in BANDWIDTHS:
            for nl in LAYERS:
                for ru in REUPLOADS:
                    m = VQC(len(cols), n_layers=nl, reupload=ru, epochs=epochs,
                            lr=0.1, scale=bw, seed=SEED).fit(Xa, ytr[a_i], Xb, ytr[b_i])
                    s = average_precision_score(ytr[b_i], m.predict_proba(Xb))
                    if s > bs:
                        bs, best = s, (bw, nl, ru)
        picks.append(best)
        Xtr, Xte = prep(tr_df, te_df, cols)
        m = VQC(len(cols), n_layers=best[1], reupload=best[2], epochs=epochs,
                lr=0.1, scale=best[0], seed=SEED).fit(Xtr, ytr)
        oof[te_i] = m.predict_proba(Xte)
    return oof, picks


def main():
    rows = []
    for h in HORIZONS:
        df = S.load(h)
        y = df[f"y_{h}h"].values.astype(int)
        g = df.patient.values
        for block in BLOCKS:
            t0 = time.time()
            oof, picks = run(df, h, block)
            mt = point_metrics(y, oof)
            lo, hi = bootstrap_ci(y, oof, groups=g, n_boot=1000)
            lo2, hi2 = bootstrap_ci(y, oof, fn=average_precision_score, groups=g, n_boot=1000)
            rows.append({"horizon": h, "block": block, "model": "VQC_ru", **mt,
                         "auroc_lo": lo, "auroc_hi": hi,
                         "auprc_lo": lo2, "auprc_hi": hi2, "picks": json.dumps(picks)})
            np.save(f"results/preds/VQC_ru_{block}_{h}h.npy", oof)
            ru = [p[2] for p in picks]
            print(f"h={h:2d} {block:7s} VQC+reupload AUROC={mt['auroc']:.3f} "
                  f"[{lo:.3f},{hi:.3f}] AUPRC={mt['auprc']:.3f} "
                  f"reupload_picks={ru} ({time.time()-t0:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv("results/logs/quantum_enhanced_results.csv", index=False)
    print("\nsaved results/logs/quantum_enhanced_results.csv")


if __name__ == "__main__":
    main()
