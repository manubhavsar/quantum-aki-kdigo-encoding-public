"""Robustness check: is the quantum result an artefact of under-tuning?

The main protocol fixed the data re-uploading depth at 1. Re-uploading is known to
increase variational expressivity (Perez-Salinas et al., 2020), so leaving it
untuned could have understated the quantum models. We therefore re-ran the VQC
with re-uploading depth in the search space, on identical folds, and compare
paired out-of-fold predictions with DeLong's test.
"""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from metrics import delong_test

def main():
    rows = []
    for h in (12, 24, 48):
        y = np.load(f"results/preds/_y_{h}h.npy")
        for block in ("static", "delta"):
            base = np.load(f"results/preds/VQC_{block}_{h}h.npy")
            enh = np.load(f"results/preds/VQC_ru_{block}_{h}h.npy")
            a, b, p = delong_test(y, base, enh)
            rows.append({"horizon": h, "block": block, "vqc_base": a,
                         "vqc_reupload": b, "diff": b - a, "delong_p": p})
            print(f"h={h:2d} {block:7s} base={a:.3f} +reupload={b:.3f} "
                  f"diff={b-a:+.3f} p={p:.4f}")
    d = pd.DataFrame(rows)
    d.to_csv("results/logs/reupload_check.csv", index=False)
    print(f"\nre-uploading improved {int((d['diff'] > 0).sum())}/{len(d)} configurations; "
          f"mean change {d['diff'].mean():+.3f}, worst {d['diff'].min():+.3f}")

if __name__ == "__main__":
    main()
