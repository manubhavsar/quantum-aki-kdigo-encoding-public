"""Horizon selection as an audited evaluation choice.

The study reports three prediction horizons. A reviewer may reasonably ask what
would be gained by reporting only the most favourable one, since selecting the
horizon after seeing results is the same class of choice as selecting a seed or a
metric. This measures it on the same estimand as the other entries: the
quantum-classical AUROC gap, G = AUROC(quantum) - AUROC(classical).
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from sklearn.metrics import roc_auc_score

HORIZONS = [12, 24, 48]
PAIRS = [("VQC", "delta", "LogisticRegression", "delta"),
         ("VQC", "delta", "MLP", "delta"),
         ("TRE", "trajectory", "TRE_noent", "trajectory"),
         ("QLSTM", "trajectory", "QLSTM_classicalgate", "trajectory")]


def main():
    rows = []
    for qm, qb, cm, cb in PAIRS:
        gaps = {}
        for h in HORIZONS:
            fa = f"results/preds/{qm}_{qb}_{h}h.npy"
            fb = f"results/preds/{cm}_{cb}_{h}h.npy"
            if not (os.path.exists(fa) and os.path.exists(fb)):
                continue
            y = np.load(f"results/preds/_y_{h}h.npy")
            gaps[h] = roc_auc_score(y, np.load(fa)) - roc_auc_score(y, np.load(fb))
        if len(gaps) < 2:
            continue
        best = max(gaps.values()); mean = float(np.mean(list(gaps.values())))
        rows.append({"comparison": f"{qm}|{qb} vs {cm}|{cb}",
                     **{f"gap_h{h}": v for h, v in gaps.items()},
                     "best_horizon_gap": best, "mean_gap": mean,
                     "selection_gain": best - mean,
                     "best_horizon": max(gaps, key=gaps.get)})
        print(f"{rows[-1]['comparison']:46s} "
              + " ".join(f"h{h}={v:+.3f}" for h, v in gaps.items())
              + f" | best={best:+.3f} mean={mean:+.3f} "
                f"selection gain={best-mean:+.3f}")
    d = pd.DataFrame(rows)
    d.to_csv("results/logs/horizon_artifact.csv", index=False)
    print(f"\nmean selection gain across comparisons: {d.selection_gain.mean():+.3f}")
    print(f"max  selection gain:                    {d.selection_gain.max():+.3f}")


if __name__ == "__main__":
    main()
