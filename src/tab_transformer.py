"""Transformer on the tabular feature blocks (reviewer point R2.4).

The sequence baselines (LSTM, GRU, transformer) consume the raw creatinine
trajectory, the same input as TRE and the Minimal-QLSTM. This adds a transformer on
exactly the static and delta feature blocks the VQC and kernel receive, so the
quantum feature-block models also face an attention-based classical model on
identical inputs. Each feature becomes a token (a learned per-feature embedding
scaled by its value) and a class token is pooled, following the feature-tokeniser
design of Gorishniy et al. (2021). Training matches the sequence baselines: Adam
1e-3, batch 64, up to 120 epochs, patience 15, patient-grouped folds.
"""
import os, sys, warnings
import numpy as np, pandas as pd, torch, torch.nn as nn
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

import splits as S
from metrics import point_metrics, bootstrap_ci

SEED = 20260826


class FTTransformer(nn.Module):
    def __init__(self, n_feat, d=16, heads=2):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_feat, d) * 0.1)
        self.b = nn.Parameter(torch.zeros(n_feat, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=heads, dim_feedforward=2 * d,
                                           dropout=0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, x):
        tok = x.unsqueeze(-1) * self.w + self.b
        tok = torch.cat([self.cls.expand(len(x), -1, -1), tok], dim=1)
        return self.head(self.enc(tok)[:, 0]).squeeze(-1)


def fit(Xa, ya, Xb, yb, seed=SEED, epochs=120, patience=15):
    torch.manual_seed(seed)
    m = FTTransformer(Xa.shape[1])
    pw = torch.tensor((ya == 0).sum() / max((ya == 1).sum(), 1), dtype=torch.float32)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    T = lambda a: torch.tensor(a, dtype=torch.float32)
    xa, ta, xb, tb = T(Xa), T(ya), T(Xb), T(yb)
    best, state, bad = np.inf, None, 0
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        m.train()
        perm = torch.randperm(len(xa), generator=g)
        for i in range(0, len(xa), 64):
            idx = perm[i:i + 64]
            opt.zero_grad(); lossf(m(xa[idx]), ta[idx]).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            vl = lossf(m(xb), tb).item()
        if vl < best - 1e-4:
            best, bad, state = vl, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    m.load_state_dict(state); m.eval()
    return m


def main():
    rows = []
    for h in (12, 24, 48):
        df = S.load(h)
        y = df[f"y_{h}h"].values.astype(int)
        g = df.patient.values
        for block in ("static", "delta"):
            cols = S.BLOCKS[block]
            oof = np.full(len(df), np.nan)
            for tr, te in GroupKFold(n_splits=5).split(df, y, g):
                med = df.iloc[tr][cols].median()
                X = df[cols].fillna(med).values.astype(float)
                mu, sd = X[tr].mean(0), X[tr].std(0); sd[sd == 0] = 1
                X = (X - mu) / sd
                tr_df = df.iloc[tr]
                gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
                a_i, b_i = next(gss.split(tr_df, y[tr], tr_df.patient.values))
                m = fit(X[tr[a_i]], y[tr[a_i]], X[tr[b_i]], y[tr[b_i]])
                with torch.no_grad():
                    oof[te] = torch.sigmoid(m(torch.tensor(X[te], dtype=torch.float32))).numpy()
            mt = point_metrics(y, oof)
            lo, hi = bootstrap_ci(y, oof, groups=g, n_boot=1000)
            npar = sum(p.numel() for p in m.parameters())
            rows.append({"horizon": h, "block": block, "model": "TabTransformer", **mt,
                         "auroc_lo": lo, "auroc_hi": hi, "params": npar})
            np.save(f"results/preds/TabTransformer_{block}_{h}h.npy", oof)
            print(f"h={h:2d} {block:6s} TabTransformer AUROC={mt['auroc']:.3f} [{lo:.3f},{hi:.3f}] "
                  f"AUPRC={mt['auprc']:.3f} params={npar}", flush=True)
            pd.DataFrame(rows).to_csv("results/logs/tab_transformer.csv", index=False)


if __name__ == "__main__":
    main()
