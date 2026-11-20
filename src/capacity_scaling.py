"""Does the quantum-component contribution survive as the classical control grows?

The MLP control in mlp_control.py is fixed at the smallest width whose parameter
count exceeds the circuit's. That answers "is a same-size classical network
enough?" but not "is a *richer* one enough?". Here the classical control is swept
across a range of parameter budgets while the VQC stays fixed, and the estimand is
the interaction

    G(capacity) = [AUROC_VQC(delta) - AUROC_VQC(static)]
                - [AUROC_MLP(delta) - AUROC_MLP(static)]

evaluated at each budget. A contribution that persists across the range is not an
artifact of an undersized classical control; one that vanishes is.
"""
import os, sys, warnings
import numpy as np, pandas as pd, torch
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

import splits as S
from metrics import point_metrics
from mlp_control import (SEED, HORIZONS, vqc_param_count, mlp_params,
                         fit_mlp, prep)

BUDGETS = [100, 200, 500, 1000, 2000]


def hidden_for(n_in, budget):
    """Largest hidden width whose parameter count stays within the budget."""
    h = 2
    while mlp_params(n_in, h + 1) <= budget:
        h += 1
    return h


def oof_auroc(df, y, g, cols, hidden):
    oof = np.full(len(df), np.nan)
    for tr_i, te_i in GroupKFold(n_splits=5).split(df, y, g):
        tr, te = df.iloc[tr_i], df.iloc[te_i]
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
        a_i, b_i = next(gss.split(tr, y[tr_i], tr.patient.values))
        Xa, Xb = prep(tr.iloc[a_i], tr.iloc[b_i], cols)
        Xtr, Xte = prep(tr, te, cols)
        fit_mlp(Xa, y[tr_i][a_i], Xb, y[tr_i][b_i], hidden)
        m = fit_mlp(Xtr, y[tr_i], Xb, y[tr_i][b_i], hidden)
        with torch.no_grad():
            oof[te_i] = torch.sigmoid(
                m(torch.tensor(Xte, dtype=torch.float64)).squeeze(-1)).numpy()
    return point_metrics(y, oof)["auroc"], oof


def main():
    q = pd.read_csv("results/logs/quantum_results.csv")
    rows = []
    for h in HORIZONS:
        df = S.load(h)
        y = df[f"y_{h}h"].values.astype(int)
        g = df.patient.values
        vq = {b: q[(q.horizon == h) & (q.block == b) & (q.model == "VQC")].iloc[0].auroc
              for b in ("static", "delta")}
        vqc_delta = vq["delta"] - vq["static"]
        for budget in BUDGETS:
            aur = {}
            for block in ("static", "delta"):
                cols = S.BLOCKS[block]
                hid = hidden_for(len(cols), budget)
                aur[block], oof = oof_auroc(df, y, g, cols, hid)
                np.save(f"results/preds/MLPcap{budget}_{block}_{h}h.npy", oof)
                npar = mlp_params(len(cols), hid)
            mlp_delta = aur["delta"] - aur["static"]
            rows.append({"horizon": h, "budget": budget, "hidden": hid,
                         "mlp_params": npar,
                         "vqc_params": vqc_param_count(len(S.BLOCKS["delta"])),
                         "mlp_static": aur["static"], "mlp_delta": aur["delta"],
                         "mlp_gain": mlp_delta, "vqc_gain": vqc_delta,
                         "interaction": vqc_delta - mlp_delta})
            print(f"h={h:2d} budget={budget:5d} params={npar:5d} "
                  f"MLP {aur['static']:.3f}->{aur['delta']:.3f} "
                  f"gain={mlp_delta:+.3f} vs VQC {vqc_delta:+.3f} "
                  f"interaction={vqc_delta-mlp_delta:+.3f}", flush=True)
            pd.DataFrame(rows).to_csv("results/logs/capacity_scaling.csv", index=False)
    print("\nsaved results/logs/capacity_scaling.csv")


if __name__ == "__main__":
    main()
