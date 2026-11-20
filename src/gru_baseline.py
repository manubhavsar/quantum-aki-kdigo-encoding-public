"""GRU over raw creatinine trajectories: a second recurrent comparator.

Identical to lstm_baseline in inputs, splits, standardisation, optimiser and
early stopping; only the recurrent cell differs. Run directly to evaluate all
three horizons.
"""
import os, sys, warnings
import numpy as np, pandas as pd, torch, torch.nn as nn
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import splits as S
import transformer_baseline as tb
from metrics import point_metrics, bootstrap_ci


class GRUNet(nn.Module):
    def __init__(self, n_static, hidden=32, seq_feat=2):
        super().__init__()
        self.gru = nn.GRU(seq_feat, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + n_static, 32), nn.ReLU(),
                                  nn.Dropout(0.2), nn.Linear(32, 1))

    def forward(self, seq, static, mask):
        out, _ = self.gru(seq)
        lens = mask.sum(1).clamp(min=1).long() - 1
        last = out[torch.arange(len(out)), lens]
        return self.head(torch.cat([last, static], dim=1)).squeeze(-1)


def run(df, seq, mask, horizon):
    """Same training and cross-validation loop as the transformer baseline, with
    the encoder swapped for the GRU."""
    original = tb.TransformerNet
    tb.TransformerNet = GRUNet
    try:
        return tb.run(df, seq, mask, horizon, return_params=True)
    finally:
        tb.TransformerNet = original


def main():
    seq_all = np.load("data/processed/eicu_seq.npy")
    mask_all = np.load("data/processed/eicu_mask.npy")
    full = pd.read_csv("data/processed/eicu_samples.csv")
    rows = []
    for h in (12, 24, 48):
        df = S.load(h)
        sel = full.dropna(subset=[f"y_{h}h"]).index.values
        seq, mask = seq_all[sel], mask_all[sel]
        y = df[f"y_{h}h"].values.astype(int)
        oof, npar = run(df, seq, mask, h)
        mt = point_metrics(y, oof)
        lo, hi = bootstrap_ci(y, oof, groups=df.patient.values, n_boot=1000)
        rows.append({"horizon": h, "block": "trajectory", "model": "GRU", **mt,
                     "auroc_lo": lo, "auroc_hi": hi, "params": npar})
        np.save(f"results/preds/GRU_trajectory_{h}h.npy", oof)
        print(f"h={h:2d} GRU AUROC={mt['auroc']:.3f} [{lo:.3f},{hi:.3f}] "
              f"AUPRC={mt['auprc']:.3f} params={npar}", flush=True)
        pd.DataFrame(rows).to_csv("results/logs/gru_baseline.csv", index=False)


if __name__ == "__main__":
    main()
