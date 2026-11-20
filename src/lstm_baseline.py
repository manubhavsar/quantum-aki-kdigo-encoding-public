"""LSTM baseline over raw creatinine trajectories.

This is a deliberately strong comparator: it receives the raw sequence rather
than engineered rate-of-change summaries, so if it matches the delta-encoded
models then the contribution of explicit KDIGO-aligned delta features is weaker
than claimed. Reporting it honestly is the point.
"""
import os, sys, numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(__file__))
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
import splits as S

SEED = 20260826
DEMOG = ["age", "sex_male", "weight_kg"]


class LSTMNet(nn.Module):
    def __init__(self, n_static, hidden=32, seq_feat=2):
        super().__init__()
        self.lstm = nn.LSTM(seq_feat, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + n_static, 32), nn.ReLU(),
                                  nn.Dropout(0.2), nn.Linear(32, 1))

    def forward(self, seq, static, mask):
        out, _ = self.lstm(seq)
        lens = mask.sum(1).clamp(min=1).long() - 1
        last = out[torch.arange(len(out)), lens]
        return self.head(torch.cat([last, static], dim=1)).squeeze(-1)


def _fit(seq_tr, st_tr, mk_tr, y_tr, seq_va, st_va, mk_va, y_va,
         epochs=120, lr=1e-3, seed=SEED):
    torch.manual_seed(seed)
    model = LSTMNet(st_tr.shape[1])
    pw = torch.tensor((y_tr == 0).sum() / max((y_tr == 1).sum(), 1), dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    T = lambda a, d=torch.float32: torch.tensor(a, dtype=d)
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
            loss = lossf(model(xs[idx], ss[idx], ms[idx]), ys[idx])
            loss.backward(); opt.step()
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


def run(df, seq, mask, horizon):
    y = df[f"y_{horizon}h"].values.astype(int)
    groups = df.patient.values
    oof = np.full(len(df), np.nan)
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
        with torch.no_grad():
            T = lambda x: torch.tensor(x, dtype=torch.float32)
            oof[te_idx] = torch.sigmoid(
                m(T(s_all[te_idx]), T(st_all[te_idx]), T(mask[te_idx]))).numpy()
    return oof
