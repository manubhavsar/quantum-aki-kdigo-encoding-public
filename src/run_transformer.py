"""Evaluate the transformer trajectory baseline across all horizons (R2.4)."""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from sklearn.metrics import average_precision_score

import splits as S
from metrics import point_metrics, bootstrap_ci
from transformer_baseline import run as run_tf

HORIZONS = [12, 24, 48]


def main():
    seq_all = np.load("data/processed/eicu_seq.npy")
    mask_all = np.load("data/processed/eicu_mask.npy")
    full = pd.read_csv("data/processed/eicu_samples.csv")
    rows = []
    for h in HORIZONS:
        df = S.load(h)
        # align trajectory rows to the dropna-filtered sample rows, as run_temporal does
        sel = full.dropna(subset=[f"y_{h}h"]).index.values
        seq, mask = seq_all[sel], mask_all[sel]
        assert len(seq) == len(df), (len(seq), len(df))
        y = df[f"y_{h}h"].values.astype(int)
        g = df.patient.values
        oof, npar = run_tf(df, seq, mask, h, return_params=True)
        mt = point_metrics(y, oof)
        lo, hi = bootstrap_ci(y, oof, groups=g, n_boot=1000)
        rows.append({"horizon": h, "block": "trajectory", "model": "Transformer",
                     "n": len(df), "n_pos": int(y.sum()),
                     "prevalence": float(y.mean()), **mt,
                     "auroc_lo": lo, "auroc_hi": hi, "params": npar})
        np.save(f"results/preds/Transformer_trajectory_{h}h.npy", oof)
        print(f"h={h:2d} Transformer AUROC={mt['auroc']:.3f} [{lo:.3f},{hi:.3f}] "
              f"AUPRC={mt['auprc']:.3f} params={npar}", flush=True)
        pd.DataFrame(rows).to_csv("results/logs/transformer_baseline.csv", index=False)
    print("\nsaved results/logs/transformer_baseline.csv")


if __name__ == "__main__":
    main()
