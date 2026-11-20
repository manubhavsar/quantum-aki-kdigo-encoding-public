"""Parameter-matched classical neural control for the variational quantum classifier.

The classical baselines used elsewhere (logistic regression, random forest,
gradient boosting) differ from the VQC in model family, so they cannot separate
"quantum circuit" from "trainable nonlinear map of the same inputs". This adds a
multilayer perceptron matched to the VQC on input, optimiser, schedule, folds and
seeds, with a hidden width chosen so its parameter count slightly EXCEEDS the
circuit's, making the comparison conservative towards the classical control.

VQC parameter count: reupload * n_layers * n_qubits * 3 weights, plus a bias.
"""
import os, sys, warnings
import numpy as np, pandas as pd, torch, torch.nn as nn
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import average_precision_score

import splits as S
from metrics import point_metrics, bootstrap_ci

SEED = 20260826
HORIZONS = [12, 24, 48]


def vqc_param_count(n_features, n_layers=2, reupload=1):
    return reupload * n_layers * n_features * 3 + 1


def make_mlp(n_in, hidden):
    return nn.Sequential(nn.Linear(n_in, hidden), nn.Tanh(),
                         nn.Linear(hidden, hidden), nn.Tanh(),
                         nn.Linear(hidden, 1))


def mlp_params(n_in, hidden):
    return (n_in + 1) * hidden + (hidden + 1) * hidden + (hidden + 1)


def pick_hidden(n_in, target):
    """Smallest hidden width whose parameter count exceeds the circuit's."""
    h = 2
    while mlp_params(n_in, h) <= target:
        h += 1
    return h


def fit_mlp(Xtr, ytr, Xva, yva, hidden, epochs=200, lr=0.01, seed=SEED, patience=20):
    torch.manual_seed(seed)
    m = make_mlp(Xtr.shape[1], hidden).double()
    pos = float(ytr.sum())
    pw = torch.tensor((len(ytr) - pos) / max(pos, 1.0), dtype=torch.float64)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    T = lambda a: torch.tensor(np.asarray(a), dtype=torch.float64)
    xs, ys, xv, yv = T(Xtr), T(ytr), T(Xva), T(yva)
    best, state, bad = np.inf, None, 0
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        m.train()
        perm = torch.randperm(len(xs), generator=g)
        for i in range(0, len(xs), 64):
            idx = perm[i:i + 64]
            opt.zero_grad()
            lossf(m(xs[idx]).squeeze(-1), ys[idx]).backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            vl = lossf(m(xv).squeeze(-1), yv).item()
        if vl < best - 1e-4:
            best, bad = vl, 0
            state = {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if state:
        m.load_state_dict(state)
    m.eval()
    return m


def prep(tr, te, cols):
    med = tr[cols].median()
    Xtr = tr[cols].fillna(med).values.astype(float)
    Xte = te[cols].fillna(med).values.astype(float)
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
    return (Xtr - mu) / sd, (Xte - mu) / sd


def main():
    rows = []
    for h in HORIZONS:
        df = S.load(h)
        y = df[f"y_{h}h"].values.astype(int)
        g = df.patient.values
        for block in ("static", "delta"):
            cols = S.BLOCKS[block]
            target = vqc_param_count(len(cols))
            hidden = pick_hidden(len(cols), target)
            oof = np.full(len(df), np.nan)
            for tr_i, te_i in GroupKFold(n_splits=5).split(df, y, g):
                tr, te = df.iloc[tr_i], df.iloc[te_i]
                gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
                a_i, b_i = next(gss.split(tr, y[tr_i], tr.patient.values))
                Xa, Xb = prep(tr.iloc[a_i], tr.iloc[b_i], cols)
                Xtr, Xte = prep(tr, te, cols)
                m = fit_mlp(Xa, y[tr_i][a_i], Xb, y[tr_i][b_i], hidden)
                m2 = fit_mlp(Xtr, y[tr_i], Xb, y[tr_i][b_i], hidden)
                with torch.no_grad():
                    oof[te_i] = torch.sigmoid(
                        m2(torch.tensor(Xte, dtype=torch.float64)).squeeze(-1)).numpy()
            mt = point_metrics(y, oof)
            lo, hi = bootstrap_ci(y, oof, groups=g, n_boot=1000)
            rows.append({"horizon": h, "block": block, "model": "MLP",
                         **mt, "auroc_lo": lo, "auroc_hi": hi,
                         "hidden": hidden, "mlp_params": mlp_params(len(cols), hidden),
                         "vqc_params": target})
            np.save(f"results/preds/MLP_{block}_{h}h.npy", oof)
            print(f"h={h:2d} {block:7s} MLP AUROC={mt['auroc']:.3f} [{lo:.3f},{hi:.3f}] "
                  f"hidden={hidden} params={mlp_params(len(cols),hidden)} "
                  f"(VQC {target})", flush=True)
            pd.DataFrame(rows).to_csv("results/logs/mlp_control.csv", index=False)
    print("\nsaved results/logs/mlp_control.csv")


if __name__ == "__main__":
    main()
