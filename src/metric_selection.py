"""Artifact: does the winner change with the reported metric?

Selective metric reporting is the cheapest way to manufacture a favourable
comparison, because it requires no new experiment -- only a choice about what to
put in the table. We rank every model under five common metrics on identical
pooled out-of-fold predictions and record how far each model's rank moves.
"""
import os, sys, glob
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             accuracy_score, balanced_accuracy_score,
                             brier_score_loss, roc_curve)

HOR = 24


def binarise(y, p):
    """Youden-optimal threshold, the usual choice when a paper reports F1/accuracy."""
    fpr, tpr, thr = roc_curve(y, p)
    t = thr[np.argmax(tpr - fpr)]
    return (p >= t).astype(int)


def main():
    y = np.load(f"results/preds/_y_{HOR}h.npy")
    rows = []
    for f in sorted(glob.glob(f"results/preds/*_{HOR}h.npy")):
        name = os.path.basename(f).replace(f"_{HOR}h.npy", "")
        if name.startswith("_"):
            continue
        p = np.load(f)
        if not np.isfinite(p).all():
            continue
        pn = (p - p.min()) / (p.max() - p.min() + 1e-12)
        b = binarise(y, p)
        rows.append({"model": name,
                     "AUROC": roc_auc_score(y, p),
                     "AUPRC": average_precision_score(y, p),
                     "F1": f1_score(y, b),
                     "Accuracy": accuracy_score(y, b),
                     "BalancedAcc": balanced_accuracy_score(y, b),
                     "Brier(neg)": -brier_score_loss(y, pn)})
    d = pd.DataFrame(rows).set_index("model")
    ranks = d.rank(ascending=False).astype(int)
    d.to_csv("results/logs/metric_selection_scores.csv")
    ranks.to_csv("results/logs/metric_selection_ranks.csv")
    print("=== rank of each model under each metric (1 = best) ===")
    print(ranks.to_string())
    spread = (ranks.max(axis=1) - ranks.min(axis=1)).sort_values(ascending=False)
    print("\n=== how far each model's rank moves across metrics ===")
    print(spread.to_string())
    print(f"\nlargest rank swing: {spread.max()} places ({spread.idxmax()})")
    for m in ranks.columns:
        print(f"  best under {m:12s}: {ranks[m].idxmin()}")


if __name__ == "__main__":
    main()
