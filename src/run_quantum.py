"""Quantum experiments: VQC and quantum-kernel SVM under static vs delta encoding.

Uses the same deterministic GroupKFold folds as src/classical.py, so pooled
out-of-fold predictions are paired sample-for-sample with the classical models
and can be compared with DeLong's test.

Hyperparameters (encoding bandwidth, circuit depth) are selected inside each
training fold on a held-out validation split; the outer fold is never seen
during selection.
"""
import os, sys, json, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.svm import SVC
from sklearn.metrics import average_precision_score

import splits as S
from metrics import point_metrics, bootstrap_ci
from quantum import VQC, quantum_kernel_matrix

SEED = 20260826
HORIZONS = [12, 24, 48]
BLOCKS = ["static", "delta"]
BANDWIDTHS = [0.3, 0.6, 1.0, 1.5]
LAYERS = [2, 3]
QSVM_MAX_TRAIN = 700          # caps the O(n^2) kernel cost on a simulator


def _prep(tr_df, te_df, cols):
    med = tr_df[cols].median()
    Xtr = tr_df[cols].fillna(med).values.astype(float)
    Xte = te_df[cols].fillna(med).values.astype(float)
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1.0
    return (Xtr - mu) / sd, (Xte - mu) / sd


def run_vqc(df, horizon, block, epochs=60):
    cols = S.BLOCKS[block]
    y = df[f"y_{horizon}h"].values.astype(int)
    groups = df.patient.values
    oof = np.full(len(df), np.nan)
    picks = []
    for tr_idx, te_idx in GroupKFold(n_splits=5).split(df, y, groups):
        tr_df, te_df = df.iloc[tr_idx], df.iloc[te_idx]
        ytr = y[tr_idx]
        # inner validation split (patient-grouped) for hyperparameter choice
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
        i_tr, i_va = next(gss.split(tr_df, ytr, tr_df.patient.values))
        a_df, b_df = tr_df.iloc[i_tr], tr_df.iloc[i_va]
        Xa, Xb = _prep(a_df, b_df, cols)
        best, best_s = None, -np.inf
        for bw in BANDWIDTHS:
            for nl in LAYERS:
                m = VQC(len(cols), n_layers=nl, epochs=epochs, lr=0.1, scale=bw,
                        seed=SEED).fit(Xa, ytr[i_tr], Xb, ytr[i_va])
                s = average_precision_score(ytr[i_va], m.predict_proba(Xb))
                if s > best_s:
                    best_s, best = s, (bw, nl)
        picks.append(best)
        Xtr, Xte = _prep(tr_df, te_df, cols)
        m = VQC(len(cols), n_layers=best[1], epochs=epochs, lr=0.1, scale=best[0],
                seed=SEED).fit(Xtr, ytr)
        oof[te_idx] = m.predict_proba(Xte)
    return oof, picks


def run_qsvm(df, horizon, block):
    cols = S.BLOCKS[block]
    y = df[f"y_{horizon}h"].values.astype(int)
    groups = df.patient.values
    oof = np.full(len(df), np.nan)
    picks = []
    rng = np.random.default_rng(SEED)
    for tr_idx, te_idx in GroupKFold(n_splits=5).split(df, y, groups):
        tr_df, te_df = df.iloc[tr_idx], df.iloc[te_idx]
        ytr = y[tr_idx]
        # subsample training rows to bound kernel cost, keeping all positives
        if len(tr_df) > QSVM_MAX_TRAIN:
            pos = np.where(ytr == 1)[0]
            neg = np.where(ytr == 0)[0]
            k = max(QSVM_MAX_TRAIN - len(pos), 50)
            sel = np.concatenate([pos, rng.choice(neg, size=min(k, len(neg)), replace=False)])
            sel.sort()
        else:
            sel = np.arange(len(tr_df))
        sub = tr_df.iloc[sel]; ysub = ytr[sel]
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
        i_tr, i_va = next(gss.split(sub, ysub, sub.patient.values))
        a_df, b_df = sub.iloc[i_tr], sub.iloc[i_va]
        Xa, Xb = _prep(a_df, b_df, cols)
        best, best_s = None, -np.inf
        for bw in BANDWIDTHS:
            Kaa = quantum_kernel_matrix(Xa, Xa, len(cols), scale=bw)
            Kba = quantum_kernel_matrix(Xb, Xa, len(cols), scale=bw)
            for C in [0.1, 1.0, 10.0]:
                sv = SVC(kernel="precomputed", C=C, class_weight="balanced")
                sv.fit(Kaa, ysub[i_tr])
                s = average_precision_score(ysub[i_va], sv.decision_function(Kba))
                if s > best_s:
                    best_s, best = s, (bw, C)
        picks.append(best)
        Xtr, Xte = _prep(sub, te_df, cols)
        K = quantum_kernel_matrix(Xtr, Xtr, len(cols), scale=best[0])
        Kt = quantum_kernel_matrix(Xte, Xtr, len(cols), scale=best[0])
        sv = SVC(kernel="precomputed", C=best[1], class_weight="balanced").fit(K, ysub)
        oof[te_idx] = sv.decision_function(Kt)
    return oof, picks


def main():
    os.makedirs("results/preds", exist_ok=True)
    rows = []
    for horizon in HORIZONS:
        df = S.load(horizon)
        y = df[f"y_{horizon}h"].values.astype(int)
        groups = df.patient.values
        for block in BLOCKS:
            for name, fn in (("VQC", run_vqc), ("QSVM", run_qsvm)):
                t = time.time()
                oof, picks = fn(df, horizon, block)
                mt = point_metrics(y, oof)
                lo, hi = bootstrap_ci(y, oof, groups=groups, n_boot=1000)
                lo_pr, hi_pr = bootstrap_ci(y, oof, fn=average_precision_score,
                                            groups=groups, n_boot=1000)
                rows.append({"horizon": horizon, "block": block, "model": name, **mt,
                             "auroc_lo": lo, "auroc_hi": hi,
                             "auprc_lo": lo_pr, "auprc_hi": hi_pr,
                             "picks": json.dumps(picks)})
                np.save(f"results/preds/{name}_{block}_{horizon}h.npy", oof)
                print(f"h={horizon:2d} {block:7s} {name:5s} AUROC={mt['auroc']:.3f} "
                      f"[{lo:.3f},{hi:.3f}] AUPRC={mt['auprc']:.3f} "
                      f"picks={picks[:2]} ({time.time()-t:.0f}s)", flush=True)
                pd.DataFrame(rows).to_csv("results/logs/quantum_results.csv", index=False)
    print("\nsaved results/logs/quantum_results.csv")


if __name__ == "__main__":
    main()
