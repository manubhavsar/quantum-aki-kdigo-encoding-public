"""Evaluation metrics, bootstrap CIs and DeLong's test for correlated AUROCs."""
import numpy as np
from scipy import stats
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             roc_curve, confusion_matrix)

SEED = 20260826


def point_metrics(y, p, thresh=None):
    """AUROC, AUPRC, sensitivity/specificity at the Youden-optimal threshold, Brier."""
    y = np.asarray(y).astype(int); p = np.asarray(p, float)
    out = {"n": int(len(y)), "n_pos": int(y.sum()), "prevalence": float(y.mean())}
    if y.sum() == 0 or y.sum() == len(y):
        return {**out, "auroc": np.nan, "auprc": np.nan, "sensitivity": np.nan,
                "specificity": np.nan, "brier": np.nan, "threshold": np.nan}
    out["auroc"] = float(roc_auc_score(y, p))
    out["auprc"] = float(average_precision_score(y, p))
    out["brier"] = float(brier_score_loss(y, np.clip(p, 0, 1)))
    if thresh is None:
        fpr, tpr, thr = roc_curve(y, p)
        thresh = float(thr[np.argmax(tpr - fpr)])
    out["threshold"] = float(thresh)
    tn, fp, fn, tp = confusion_matrix(y, (p >= thresh).astype(int), labels=[0, 1]).ravel()
    out["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) else np.nan
    out["specificity"] = float(tn / (tn + fp)) if (tn + fp) else np.nan
    out["ppv"] = float(tp / (tp + fp)) if (tp + fp) else np.nan
    return out


def bootstrap_ci(y, p, fn=roc_auc_score, n_boot=2000, alpha=0.05, groups=None, seed=SEED):
    """Percentile bootstrap CI. If `groups` is given, resample *groups* (patients)
    rather than rows, so correlated repeated measures are handled correctly."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int); p = np.asarray(p, float)
    stats_ = []
    if groups is not None:
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        idx_by_g = {g: np.where(groups == g)[0] for g in uniq}
        for _ in range(n_boot):
            gs = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([idx_by_g[g] for g in gs])
            if len(np.unique(y[idx])) < 2:
                continue
            stats_.append(fn(y[idx], p[idx]))
    else:
        n = len(y)
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            if len(np.unique(y[idx])) < 2:
                continue
            stats_.append(fn(y[idx], p[idx]))
    if not stats_:
        return (np.nan, np.nan)
    return (float(np.percentile(stats_, 100 * alpha / 2)),
            float(np.percentile(stats_, 100 * (1 - alpha / 2))))


# ------------------------------------------------------------------ DeLong test
def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x)
    T = np.zeros(N, float); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, float); T2[J] = T
    return T2


def _fast_delong(preds_sorted, m):
    """Structural components for the DeLong covariance (Sun & Xu 2014 formulation)."""
    n = preds_sorted.shape[1] - m
    k = preds_sorted.shape[0]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r] = _midrank(preds_sorted[r, :m])
        ty[r] = _midrank(preds_sorted[r, m:])
        tz[r] = _midrank(preds_sorted[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01); sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_test(y, p1, p2):
    """Two-sided DeLong test for two correlated ROC curves on the same samples.

    Returns (auc1, auc2, p_value).
    """
    y = np.asarray(y).astype(int)
    p1 = np.asarray(p1, float); p2 = np.asarray(p2, float)
    order = (-y).argsort(kind="mergesort")          # positives first
    label_1_count = int(y.sum())
    preds_sorted = np.vstack((p1[order], p2[order]))
    aucs, cov = _fast_delong(preds_sorted, label_1_count)
    l = np.array([[1, -1]])
    var = float(np.squeeze(l @ cov @ l.T))
    if not np.isfinite(var) or var <= 0:
        return float(aucs[0]), float(aucs[1]), 1.0
    z = (float(aucs[0]) - float(aucs[1])) / np.sqrt(var)
    # survival function avoids underflow to exactly 0 for large |z|
    pval = 2.0 * float(stats.norm.sf(abs(z)))
    return float(aucs[0]), float(aucs[1]), pval


def mcnemar_test(y, p1, p2, thresh1=0.5, thresh2=0.5):
    """McNemar's test on binarised predictions (exact binomial for small counts)."""
    y = np.asarray(y).astype(int)
    c1 = (np.asarray(p1) >= thresh1).astype(int) == y
    c2 = (np.asarray(p2) >= thresh2).astype(int) == y
    b = int(np.sum(c1 & ~c2)); c = int(np.sum(~c1 & c2))
    if b + c == 0:
        return b, c, 1.0
    p = float(stats.binomtest(b, b + c, 0.5).pvalue)
    return b, c, p
