"""LSTM baseline and the proposed TRE quantum encoding, on identical CV folds.

Also runs the key architectural ablation for TRE: with entangling layers
interleaved between per-timepoint rotation blocks (the proposal) versus without
(which provably collapses to a delta encoding).
"""
import os, sys, time, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import average_precision_score

import splits as S
from metrics import point_metrics, bootstrap_ci
from quantum import TRE
from lstm_baseline import run as run_lstm

SEED = 20260826
HORIZONS = [12, 24, 48]
DEMOG = ["age", "sex_male", "weight_kg"]
N_QUBITS = 6


def _static_mat(df, tr_idx, cols=DEMOG):
    med = df.iloc[tr_idx][cols].median()
    X = df[cols].fillna(med).values.astype(float)
    mu, sd = X[tr_idx].mean(0), X[tr_idx].std(0); sd[sd == 0] = 1
    X = (X - mu) / sd
    # pad demographics up to the qubit count
    if X.shape[1] < N_QUBITS:
        X = np.hstack([X, np.zeros((len(X), N_QUBITS - X.shape[1]))])
    return X


def _seq_norm(seq, mask, tr_idx):
    s = seq.copy()
    flat = s[tr_idx].reshape(-1, 2)
    mu, sd = flat.mean(0), flat.std(0); sd[sd == 0] = 1
    s = (s - mu) / sd
    return s * mask[..., None]


def run_tre(df, seq, mask, horizon, entangle=True, epochs=40):
    y = df[f"y_{horizon}h"].values.astype(int)
    groups = df.patient.values
    oof = np.full(len(df), np.nan)
    for tr_idx, te_idx in GroupKFold(n_splits=5).split(df, y, groups):
        St = _static_mat(df, tr_idx)
        Sq = _seq_norm(seq, mask, tr_idx)
        tr_df = df.iloc[tr_idx]
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
        i_tr, i_va = next(gss.split(tr_df, y[tr_idx], tr_df.patient.values))
        a, b = tr_idx[i_tr], tr_idx[i_va]
        best, best_s = None, -np.inf
        for bw in (0.6, 1.0, 1.5):
            m = TRE(n_qubits=N_QUBITS, K=seq.shape[1], epochs=epochs, scale=bw,
                    entangle_between=entangle, seed=SEED).fit(
                Sq[a], St[a], y[a], Sq[b], St[b], y[b])
            s = average_precision_score(y[b], m.predict_proba(Sq[b], St[b]))
            if s > best_s:
                best_s, best = s, bw
        m = TRE(n_qubits=N_QUBITS, K=seq.shape[1], epochs=epochs, scale=best,
                entangle_between=entangle, seed=SEED).fit(
            Sq[tr_idx], St[tr_idx], y[tr_idx], Sq[b], St[b], y[b])
        oof[te_idx] = m.predict_proba(Sq[te_idx], St[te_idx])
    return oof


def main():
    seq = np.load("data/processed/eicu_seq.npy")
    mask = np.load("data/processed/eicu_mask.npy")
    full = pd.read_csv("data/processed/eicu_samples.csv")
    rows = []
    for horizon in HORIZONS:
        df = S.load(horizon)
        # align trajectory rows to the (dropna-filtered) sample rows
        key = full.index.to_series()
        sel = full.dropna(subset=[f"y_{horizon}h"]).index.values
        sq, mk = seq[sel], mask[sel]
        y = df[f"y_{horizon}h"].values.astype(int)
        groups = df.patient.values
        assert len(sq) == len(df), (len(sq), len(df))

        jobs = [("LSTM", lambda: run_lstm(df, sq, mk, horizon)),
                ("TRE", lambda: run_tre(df, sq, mk, horizon, entangle=True)),
                ("TRE_noent", lambda: run_tre(df, sq, mk, horizon, entangle=False))]
        for name, fn in jobs:
            t = time.time()
            oof = fn()
            mt = point_metrics(y, oof)
            lo, hi = bootstrap_ci(y, oof, groups=groups, n_boot=1000)
            lo_pr, hi_pr = bootstrap_ci(y, oof, fn=average_precision_score,
                                        groups=groups, n_boot=1000)
            rows.append({"horizon": horizon, "block": "trajectory", "model": name,
                         **mt, "auroc_lo": lo, "auroc_hi": hi,
                         "auprc_lo": lo_pr, "auprc_hi": hi_pr})
            np.save(f"results/preds/{name}_trajectory_{horizon}h.npy", oof)
            print(f"h={horizon:2d} {name:10s} AUROC={mt['auroc']:.3f} [{lo:.3f},{hi:.3f}] "
                  f"AUPRC={mt['auprc']:.3f} ({time.time()-t:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv("results/logs/temporal_results.csv", index=False)


if __name__ == "__main__":
    main()
