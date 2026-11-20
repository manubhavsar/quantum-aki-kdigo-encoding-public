"""Matched quantum-versus-classical pairs with clustered inference (R3.4).

Each row pairs a quantum model with the classical model that shares its inputs,
folds and protocol, and reports the AUROC difference with a patient-level
bootstrap interval and a patient-clustered permutation p-value, per horizon.
Everything is computed from the stored pooled out-of-fold predictions.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from sklearn.metrics import roc_auc_score

import splits as S
from clustered_inference import clustered_permutation_p

SEED = 20260826
PAIRS = [  # (label, quantum prediction file stem, classical prediction file stem)
    ("VQC vs MLP control (delta)", "VQC_delta", "MLP_delta"),
    ("VQC vs logistic regression (delta)", "VQC_delta", "LogisticRegression_delta"),
    ("VQC vs tabular transformer (delta)", "VQC_delta", "TabTransformer_delta"),
    ("Kernel SVM vs logistic regression (delta)", "QSVM_delta", "LogisticRegression_delta"),
    ("Kernel SVM vs tabular transformer (delta)", "QSVM_delta", "TabTransformer_delta"),
    ("Minimal-QLSTM vs classical-gate LSTM", "QLSTM_trajectory", "QLSTM_classicalgate_trajectory"),
    ("Minimal-QLSTM vs transformer", "QLSTM_trajectory", "Transformer_trajectory"),
    ("TRE vs additive control", "TRE_trajectory", "TRE_noent_trajectory"),
    ("TRE vs LSTM", "TRE_trajectory", "LSTM_trajectory"),
    ("TRE vs transformer", "TRE_trajectory", "Transformer_trajectory"),
]


def boot_ci(y, p1, p2, groups, n_boot=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in uniq}
    out = []
    for _ in range(n_boot):
        ii = np.concatenate([idx[g] for g in rng.choice(uniq, len(uniq))])
        if y[ii].min() == y[ii].max():
            continue
        out.append(roc_auc_score(y[ii], p1[ii]) - roc_auc_score(y[ii], p2[ii]))
    return np.percentile(out, [2.5, 97.5])


def main():
    rows = []
    for h in (12, 24, 48):
        df = S.load(h)
        y = df[f"y_{h}h"].values.astype(int)
        g = df.patient.values
        for label, q, c in PAIRS:
            pq = np.load(f"results/preds/{q}_{h}h.npy")
            pc = np.load(f"results/preds/{c}_{h}h.npy")
            d = roc_auc_score(y, pq) - roc_auc_score(y, pc)
            lo, hi = boot_ci(y, pq, pc, g)
            res = clustered_permutation_p(y, pq, pc, g)
            p = res[1] if isinstance(res, (tuple, list)) else res
            rows.append({"pair": label, "horizon": h, "diff": d, "lo": lo, "hi": hi, "p": float(p)})
            print(f"h={h:2d} {label:42s} {d:+.3f} [{lo:+.3f},{hi:+.3f}] p={float(p):.3f}", flush=True)
    pd.DataFrame(rows).to_csv("results/logs/paired_comparisons.csv", index=False)


if __name__ == "__main__":
    main()
