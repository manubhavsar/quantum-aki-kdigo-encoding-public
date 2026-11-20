"""Minimal-QLSTM on identical folds and identical inputs as TRE and the LSTM."""
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import average_precision_score

import splits as S
from metrics import point_metrics, bootstrap_ci
from qlstm import fit_qlstm, predict

SEED = 20260826
DEMOG = ["age", "sex_male", "weight_kg"]
HORIZONS = [12, 24, 48]


def prep_static(df, tr_idx):
    med = df.iloc[tr_idx][DEMOG].median()
    X = df[DEMOG].fillna(med).values.astype(float)
    mu, sd = X[tr_idx].mean(0), X[tr_idx].std(0); sd[sd == 0] = 1
    return np.tanh((X - mu) / sd)


def prep_seq(seq, mask, tr_idx):
    s = seq.copy()
    flat = s[tr_idx].reshape(-1, s.shape[-1])
    mu, sd = flat.mean(0), flat.std(0); sd[sd == 0] = 1
    return ((s - mu) / sd) * mask[..., None]


def run(df, seq, mask, horizon, epochs=40):
    y = df[f"y_{horizon}h"].values.astype(int)
    oof = np.full(len(df), np.nan)
    picks = []
    for tr_i, te_i in GroupKFold(n_splits=5).split(df, y, df.patient.values):
        St, Sq = prep_static(df, tr_i), prep_seq(seq, mask, tr_i)
        tr_df = df.iloc[tr_i]
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
        a_i, b_i = next(gss.split(tr_df, y[tr_i], tr_df.patient.values))
        a, b = tr_i[a_i], tr_i[b_i]
        best, bs = 1.0, -np.inf
        for sc in (0.6, 1.0, 1.5):
            m = fit_qlstm(Sq[a], St[a], mask[a], y[a], Sq[b], St[b], mask[b], y[b],
                          epochs=epochs, scale=sc)
            s = average_precision_score(y[b], predict(m, Sq[b], St[b], mask[b]))
            if s > bs:
                bs, best = s, sc
        picks.append(best)
        m = fit_qlstm(Sq[tr_i], St[tr_i], mask[tr_i], y[tr_i],
                      Sq[b], St[b], mask[b], y[b], epochs=epochs, scale=best)
        oof[te_i] = predict(m, Sq[te_i], St[te_i], mask[te_i])
    return oof, picks


def main():
    seq = np.load("data/processed/eicu_seq.npy")
    mask = np.load("data/processed/eicu_mask.npy")
    full = pd.read_csv("data/processed/eicu_samples.csv")
    rows = []
    for h in HORIZONS:
        df = S.load(h)
        sel = full.dropna(subset=[f"y_{h}h"]).index.values
        sq, mk = seq[sel], mask[sel]
        assert len(sq) == len(df)
        y = df[f"y_{h}h"].values.astype(int)
        g = df.patient.values
        t0 = time.time()
        oof, picks = run(df, sq, mk, h)
        mt = point_metrics(y, oof)
        lo, hi = bootstrap_ci(y, oof, groups=g, n_boot=1000)
        lo2, hi2 = bootstrap_ci(y, oof, fn=average_precision_score, groups=g, n_boot=1000)
        rows.append({"horizon": h, "block": "trajectory", "model": "QLSTM", **mt,
                     "auroc_lo": lo, "auroc_hi": hi, "auprc_lo": lo2, "auprc_hi": hi2})
        np.save(f"results/preds/QLSTM_trajectory_{h}h.npy", oof)
        print(f"h={h:2d} QLSTM AUROC={mt['auroc']:.3f} [{lo:.3f},{hi:.3f}] "
              f"AUPRC={mt['auprc']:.3f} picks={picks} ({time.time()-t0:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv("results/logs/qlstm_results.csv", index=False)
    print("\nsaved results/logs/qlstm_results.csv")


if __name__ == "__main__":
    main()
