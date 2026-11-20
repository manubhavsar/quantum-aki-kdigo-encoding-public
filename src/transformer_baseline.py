"""Transformer encoder over raw creatinine trajectories (reviewer request R2.4).

Reviewer 2 asked for a benchmark against classical recurrent *and* transformer
models on the same features. The LSTM comparator already covers the recurrent
case; this adds the attention-based one on identical sequences, splits,
standardisation, optimiser and early-stopping rule, so the encoder is the only
thing that differs.

Attention over K=4 timepoints is a genuinely unfavourable setting for a
transformer: self-attention has little to exploit in a four-element sequence and
the architecture carries more parameters than the recurrent baseline. Reporting
that honestly is the point, exactly as for the LSTM.
"""
import os, sys, math
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(__file__))
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

SEED = 20260826
DEMOG = ["age", "sex_male", "weight_kg"]


class TransformerNet(nn.Module):
    def __init__(self, n_static, d_model=16, nhead=2, layers=1, seq_feat=2):
        super().__init__()
        self.proj = nn.Linear(seq_feat, d_model)
        self.pos = nn.Parameter(torch.zeros(1, 64, d_model))
        nn.init.normal_(self.pos, std=0.02)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=2 * d_model,
            dropout=0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, num_layers=layers)
        self.head = nn.Sequential(nn.Linear(d_model + n_static, 32), nn.ReLU(),
                                  nn.Dropout(0.2), nn.Linear(32, 1))

    def forward(self, seq, static, mask):
        h = self.proj(seq) + self.pos[:, :seq.shape[1]]
        pad = mask < 0.5
        pad = torch.where(pad.all(dim=1, keepdim=True), torch.zeros_like(pad), pad)
        h = self.enc(h, src_key_padding_mask=pad)
        w = mask.clamp(min=0).unsqueeze(-1)
        pooled = (h * w).sum(1) / w.sum(1).clamp(min=1.0)
        return self.head(torch.cat([pooled, static], dim=1)).squeeze(-1)


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _fit(seq_tr, st_tr, mk_tr, y_tr, seq_va, st_va, mk_va, y_va,
         epochs=120, lr=1e-3, seed=SEED):
    torch.manual_seed(seed)
    model = TransformerNet(st_tr.shape[1])
    pw = torch.tensor((y_tr == 0).sum() / max((y_tr == 1).sum(), 1), dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    T = lambda a: torch.tensor(a, dtype=torch.float32)
    xs, ss, ms, ys = T(seq_tr), T(st_tr), T(mk_tr), T(y_tr)
    xv, sv, mv, yv = T(seq_va), T(st_va), T(mk_va), T(y_va)
    best, best_state, patience = np.inf, None, 0
    n = len(xs)
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, 64):
            idx = perm[i:i + 64]
            opt.zero_grad()
            lossf(model(xs[idx], ss[idx], ms[idx]), ys[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = lossf(model(xv, sv, mv), yv).item()
        if vl < best - 1e-4:
            best, best_state, patience = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 15:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model


def run(df, seq, mask, horizon, return_params=False):
    y = df[f"y_{horizon}h"].values.astype(int)
    groups = df.patient.values
    oof = np.full(len(df), np.nan)
    npar = None
    for tr_idx, te_idx in GroupKFold(n_splits=5).split(df, y, groups):
        tr_df = df.iloc[tr_idx]
        med = tr_df[DEMOG].median()
        st_all = df[DEMOG].fillna(med).values.astype(float)
        mu, sd = st_all[tr_idx].mean(0), st_all[tr_idx].std(0); sd[sd == 0] = 1
        st_all = (st_all - mu) / sd
        s_all = seq.copy()
        smu = s_all[tr_idx].reshape(-1, 2).mean(0)
        ssd = s_all[tr_idx].reshape(-1, 2).std(0); ssd[ssd == 0] = 1
        s_all = (s_all - smu) / ssd
        s_all = s_all * mask[..., None]

        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
        i_tr, i_va = next(gss.split(tr_df, y[tr_idx], tr_df.patient.values))
        a, b = tr_idx[i_tr], tr_idx[i_va]
        m = _fit(s_all[a], st_all[a], mask[a], y[a],
                 s_all[b], st_all[b], mask[b], y[b])
        if npar is None:
            npar = n_params(m)
        with torch.no_grad():
            T = lambda x: torch.tensor(x, dtype=torch.float32)
            oof[te_idx] = torch.sigmoid(
                m(T(s_all[te_idx]), T(st_all[te_idx]), T(mask[te_idx]))).numpy()
    return (oof, npar) if return_params else oof
