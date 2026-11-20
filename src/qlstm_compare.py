"""Paired comparison of the two temporal quantum architectures.

TRE places the whole trajectory in a single circuit via interleaved
non-commuting blocks; the Minimal-QLSTM applies a small circuit recurrently at
each timestep. Both receive identical inputs on identical folds, so DeLong's
test on pooled out-of-fold predictions isolates architecture.
"""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from metrics import delong_test

PAIRS = [("QLSTM", "trajectory", "TRE", "trajectory"),
         ("QLSTM", "trajectory", "LSTM", "trajectory"),
         ("QLSTM", "trajectory", "VQC", "delta"),
         ("QLSTM", "trajectory", "LogisticRegression", "delta"),
         ("QLSTM", "trajectory", "LogisticRegression", "combined")]


def main():
    rows = []
    for h in (12, 24, 48):
        y = np.load(f"results/preds/_y_{h}h.npy")
        for ma, ba, mb, bb in PAIRS:
            fa = f"results/preds/{ma}_{ba}_{h}h.npy"
            fb = f"results/preds/{mb}_{bb}_{h}h.npy"
            if not (os.path.exists(fa) and os.path.exists(fb)):
                continue
            a, b, p = delong_test(y, np.load(fa), np.load(fb))
            rows.append({"horizon": h, "model_a": f"{ma}|{ba}", "auc_a": a,
                         "model_b": f"{mb}|{bb}", "auc_b": b,
                         "diff": a - b, "delong_p": p})
            star = "*" if p < 0.05 else " "
            print(f"h={h:2d} {ma}|{ba:10s} {a:.3f}  vs  {mb}|{bb:10s} {b:.3f}   "
                  f"diff={a-b:+.3f}  p={p:.4f}{star}")
    pd.DataFrame(rows).to_csv("results/logs/qlstm_comparison.csv", index=False)
    d = pd.DataFrame(rows)
    tre = d[d.model_b == "TRE|trajectory"]
    print(f"\nQLSTM vs TRE: QLSTM ahead at {int((tre['diff']>0).sum())}/{len(tre)} horizons; "
          f"significant at {int((tre.delong_p<0.05).sum())}")


if __name__ == "__main__":
    main()
