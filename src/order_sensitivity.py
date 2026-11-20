"""Does the interleaved circuit use the ORDER of measurements, or only their values?

The mechanism claimed for TRE is that non-commuting interleaved blocks make the
model sensitive to the sequence in which measurements arrive. That is testable
directly on real trajectories: train normally, then evaluate on inputs whose
timepoints have been permuted. The multiset of values is unchanged, so any drop in
discrimination is attributable to order alone.

A model that ignores order should be unaffected. The prediction is that the
interleaved circuit degrades most, the additive ablation less, and that a
degradation is evidence the mechanism operates as described rather than being an
aggregate AUROC artefact.
"""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import roc_auc_score

import splits as S
from qlstm import MinimalQLSTM, ClassicalGateLSTM, fit_generic, predict
from quantum import TRE
from run_qlstm import prep_static, prep_seq

SEED = 20260826


def evaluate(df, seq, mask, h, kind, rng_seed=SEED):
    """Train on true order; evaluate on true, reversed and permuted order."""
    y = df[f"y_{h}h"].values.astype(int)
    out = {k: np.full(len(df), np.nan) for k in ("true", "reversed", "permuted")}
    rng = np.random.default_rng(rng_seed)
    for tr_i, te_i in GroupKFold(n_splits=5).split(df, y, df.patient.values):
        St, Sq = prep_static(df, tr_i), prep_seq(seq, mask, tr_i)
        tr_df = df.iloc[tr_i]
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
        a_i, b_i = next(gss.split(tr_df, y[tr_i], tr_df.patient.values))
        a, b = tr_i[a_i], tr_i[b_i]
        if kind in ("TRE", "TRE_noent"):
            m = TRE(n_qubits=6, K=seq.shape[1], epochs=30,
                    entangle_between=(kind == "TRE"), seed=SEED).fit(
                Sq[tr_i], St[tr_i], y[tr_i], Sq[b], St[b], y[b])
            pred = lambda X: m.predict_proba(X, St[te_i])
        else:
            cls = MinimalQLSTM if kind == "QLSTM" else ClassicalGateLSTM
            m = fit_generic(cls, Sq[tr_i], St[tr_i], mask[tr_i], y[tr_i],
                            Sq[b], St[b], mask[b], y[b], epochs=30)
            pred = lambda X: predict(m, X, St[te_i], mask[te_i])
        Xte = Sq[te_i]
        out["true"][te_i] = pred(Xte)
        out["reversed"][te_i] = pred(Xte[:, ::-1, :].copy())
        perm = Xte.copy()
        for i in range(len(perm)):
            perm[i] = perm[i][rng.permutation(perm.shape[1])]
        out["permuted"][te_i] = pred(perm)
    return {k: roc_auc_score(y, v) for k, v in out.items()}


def main():
    seq = np.load("data/processed/eicu_seq.npy")
    mask = np.load("data/processed/eicu_mask.npy")
    full = pd.read_csv("data/processed/eicu_samples.csv")
    rows = []
    for h in (24,):
        df = S.load(h)
        sel = full.dropna(subset=[f"y_{h}h"]).index.values
        sq, mk = seq[sel], mask[sel]
        for kind in ("TRE", "TRE_noent", "QLSTM", "QLSTM_classicalgate"):
            r = evaluate(df, sq, mk, h, kind)
            rows.append({"horizon": h, "model": kind, **r,
                         "drop_reversed": r["true"] - r["reversed"],
                         "drop_permuted": r["true"] - r["permuted"]})
            print(f"h={h} {kind:22s} true={r['true']:.3f} "
                  f"reversed={r['reversed']:.3f} permuted={r['permuted']:.3f} "
                  f"| drop {r['true']-r['permuted']:+.3f}", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv("results/logs/order_sensitivity.csv", index=False)
    print("\nA larger drop means the model relies more on the order in which")
    print("measurements arrive, rather than only on their values.")


if __name__ == "__main__":
    main()
