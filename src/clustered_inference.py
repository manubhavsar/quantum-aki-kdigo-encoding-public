"""Patient-clustered paired inference for model comparison.

Prediction points are repeated within patients (4,955 points from 1,609 patients),
so they are not independent. DeLong's test assumes independent observations in the
ROC comparison and can be anti-conservative under this clustering. Every reported
comparison is therefore recomputed with a paired, patient-clustered procedure:

  * confidence interval -- resample patients with replacement, retain all of a
    sampled patient's prediction points, recompute both AUROCs on the resample and
    take the paired difference;
  * p-value -- a permutation test in which the two models' predictions are
    exchanged for a random subset of PATIENTS (all points of a patient move
    together), preserving the clustering under the null of no difference.

Both respect the unit of analysis. DeLong values are retained alongside for
comparison so the effect of the correction is visible.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from sklearn.metrics import roc_auc_score
from metrics import delong_test

SEED = 20260826


def clustered_ci(y, p1, p2, groups, n_boot=4000, seed=SEED):
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in uniq}
    diffs = []
    for _ in range(n_boot):
        gs = rng.choice(uniq, size=len(uniq), replace=True)
        ii = np.concatenate([idx[g] for g in gs])
        if len(np.unique(y[ii])) < 2:
            continue
        diffs.append(roc_auc_score(y[ii], p1[ii]) - roc_auc_score(y[ii], p2[ii]))
    d = np.asarray(diffs)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), d


def clustered_permutation_p(y, p1, p2, groups, n_perm=4000, seed=SEED):
    """Exchange the two models' predictions within randomly chosen patients."""
    rng = np.random.default_rng(seed + 1)
    uniq = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in uniq}
    obs = roc_auc_score(y, p1) - roc_auc_score(y, p2)
    null = np.empty(n_perm)
    for b in range(n_perm):
        flip = rng.random(len(uniq)) < 0.5
        a, c = p1.copy(), p2.copy()
        for g, f in zip(uniq, flip):
            if f:
                ii = idx[g]
                a[ii], c[ii] = p2[ii], p1[ii]
        null[b] = roc_auc_score(y, a) - roc_auc_score(y, c)
    # two-sided, with the +1 correction so p is never exactly zero
    p = (1 + np.sum(np.abs(null) >= abs(obs))) / (n_perm + 1)
    return float(obs), float(min(p, 1.0))


def compare(y, p1, p2, groups, label=""):
    lo, hi, _ = clustered_ci(y, p1, p2, groups)
    obs, pp = clustered_permutation_p(y, p1, p2, groups)
    _, _, pd_ = delong_test(y, p1, p2)
    return {"comparison": label, "diff": obs, "ci_lo": lo, "ci_hi": hi,
            "p_clustered": pp, "p_delong": pd_}


PAIRS = [
    ("TRE_trajectory", "TRE_noent_trajectory", "TRE interleaving vs no interleaving"),
    ("QLSTM_trajectory", "QLSTM_classicalgate_trajectory", "QLSTM quantum gate vs classical gate"),
    ("VQC_delta", "VQC_static", "VQC delta vs static encoding"),
    ("VQC_delta", "LogisticRegression_delta", "VQC vs logistic regression (delta)"),
    ("VQC_delta", "LogisticRegression_combined", "VQC delta vs logistic regression (combined)"),
    ("QLSTM_trajectory", "LSTM_trajectory", "QLSTM vs classical LSTM"),
]


def main():
    rows = []
    for h in (12, 24, 48):
        y = np.load(f"results/preds/_y_{h}h.npy")
        g = np.load(f"results/preds/_g_{h}h.npy", allow_pickle=True)
        for a, b, lab in PAIRS:
            fa, fb = f"results/preds/{a}_{h}h.npy", f"results/preds/{b}_{h}h.npy"
            if not (os.path.exists(fa) and os.path.exists(fb)):
                continue
            r = compare(y, np.load(fa), np.load(fb), g, lab)
            r["horizon"] = h
            rows.append(r)
            print(f"h={h:2d} {lab:44s} d={r['diff']:+.3f} "
                  f"[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}] "
                  f"p_clust={r['p_clustered']:.4f} (DeLong {r['p_delong']:.4f})",
                  flush=True)
    pd.DataFrame(rows).to_csv("results/logs/clustered_inference.csv", index=False)
    d = pd.DataFrame(rows)
    print(f"\nsign changes vs DeLong at alpha=0.05: "
          f"{int(((d.p_clustered<0.05) != (d.p_delong<0.05)).sum())}/{len(d)}")


if __name__ == "__main__":
    main()
