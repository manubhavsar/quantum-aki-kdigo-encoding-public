"""Seed robustness of the TRE interleaving ablation (reviewer point H).

Reported tables use one pre-specified seed. This repeats the paired ablation --
interleaved circuit against its additive no-interleaving control -- across five
initialisation seeds at 24 h and 48 h, so the effect can be stated as mean +/- SD
instead of a single draw. Folds are held fixed; only the model seed varies, and
both arms share each seed, so the difference is paired within seed.

To keep five seeds tractable the encoding bandwidth is fixed at 1.0 instead of
tuned per fold, so absolute values differ slightly from the tuned figures in the
main results table; the quantity of interest is the within-seed difference.
"""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import roc_auc_score

import splits as S
from quantum import TRE
from run_temporal import _static_mat, _seq_norm, N_QUBITS

SEEDS = [0, 1, 2, 3, 4]
HORIZONS = [24, 48]
SPLIT_SEED = 20260826


def oof_tre(df, seq, mask, y, entangle, seed):
    oof = np.full(len(df), np.nan)
    for tr, te in GroupKFold(n_splits=5).split(df, y, df.patient.values):
        St, Sq = _static_mat(df, tr), _seq_norm(seq, mask, tr)
        tr_df = df.iloc[tr]
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SPLIT_SEED)
        _, i_va = next(gss.split(tr_df, y[tr], tr_df.patient.values))
        b = tr[i_va]
        m = TRE(n_qubits=N_QUBITS, K=seq.shape[1], epochs=40, scale=1.0,
                entangle_between=entangle, seed=seed).fit(
            Sq[tr], St[tr], y[tr], Sq[b], St[b], y[b])
        oof[te] = m.predict_proba(Sq[te], St[te])
    return roc_auc_score(y, oof)


def main():
    seq_all = np.load("data/processed/eicu_seq.npy")
    mask_all = np.load("data/processed/eicu_mask.npy")
    full = pd.read_csv("data/processed/eicu_samples.csv")
    rows = []
    for h in HORIZONS:
        df = S.load(h)
        sel = full.dropna(subset=[f"y_{h}h"]).index.values
        seq, mask = seq_all[sel], mask_all[sel]
        y = df[f"y_{h}h"].values.astype(int)
        for s in SEEDS:
            a_int = oof_tre(df, seq, mask, y, True, s)
            a_add = oof_tre(df, seq, mask, y, False, s)
            rows.append({"horizon": h, "seed": s, "tre": a_int, "tre_noent": a_add,
                         "diff": a_int - a_add})
            print(f"h={h} seed={s} TRE={a_int:.3f} additive={a_add:.3f} "
                  f"diff={a_int-a_add:+.3f}", flush=True)
            pd.DataFrame(rows).to_csv("results/logs/seed_robustness_tre.csv", index=False)
    d = pd.DataFrame(rows)
    for h in HORIZONS:
        x = d[d.horizon == h]["diff"]
        print(f"h={h}: diff mean={x.mean():+.3f} SD={x.std(ddof=1):.3f} "
              f"positive in {(x > 0).sum()}/{len(x)} seeds")


if __name__ == "__main__":
    main()
