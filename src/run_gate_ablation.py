"""Controlled test of the quantum candidate gate.

MinimalQLSTM vs an identical cell whose quantum gate is replaced by a classical
sub-network of the same input/output width (and slightly MORE parameters, so the
comparison is conservative towards classical). Same folds, same inputs, same
training loop, same head. Any difference is attributable to the gate.
"""
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import average_precision_score

import splits as S
from metrics import point_metrics, bootstrap_ci, delong_test
from qlstm import MinimalQLSTM, ClassicalGateLSTM, fit_generic, predict
from run_qlstm import prep_static, prep_seq

SEED = 20260826


def run(df, seq, mask, horizon, model_cls, epochs=40):
    y = df[f"y_{horizon}h"].values.astype(int)
    oof = np.full(len(df), np.nan)
    for tr_i, te_i in GroupKFold(n_splits=5).split(df, y, df.patient.values):
        St, Sq = prep_static(df, tr_i), prep_seq(seq, mask, tr_i)
        tr_df = df.iloc[tr_i]
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
        a_i, b_i = next(gss.split(tr_df, y[tr_i], tr_df.patient.values))
        a, b = tr_i[a_i], tr_i[b_i]
        best, bs = 1.0, -np.inf
        for sc in (0.6, 1.0, 1.5):
            m = fit_generic(model_cls, Sq[a], St[a], mask[a], y[a],
                            Sq[b], St[b], mask[b], y[b], epochs=epochs, scale=sc)
            s = average_precision_score(y[b], predict(m, Sq[b], St[b], mask[b]))
            if s > bs:
                bs, best = s, sc
        m = fit_generic(model_cls, Sq[tr_i], St[tr_i], mask[tr_i], y[tr_i],
                        Sq[b], St[b], mask[b], y[b], epochs=epochs, scale=best)
        oof[te_i] = predict(m, Sq[te_i], St[te_i], mask[te_i])
    return oof


def main():
    seq = np.load("data/processed/eicu_seq.npy")
    mask = np.load("data/processed/eicu_mask.npy")
    full = pd.read_csv("data/processed/eicu_samples.csv")
    rows = []
    for h in (12, 24, 48):
        df = S.load(h)
        sel = full.dropna(subset=[f"y_{h}h"]).index.values
        sq, mk = seq[sel], mask[sel]
        y = df[f"y_{h}h"].values.astype(int)
        g = df.patient.values
        t0 = time.time()
        oof_c = run(df, sq, mk, h, ClassicalGateLSTM)
        np.save(f"results/preds/QLSTM_classicalgate_trajectory_{h}h.npy", oof_c)
        oof_q = np.load(f"results/preds/QLSTM_trajectory_{h}h.npy")
        mq, mc = point_metrics(y, oof_q), point_metrics(y, oof_c)
        lo, hi = bootstrap_ci(y, oof_c, groups=g, n_boot=1000)
        aq, ac, p = delong_test(y, oof_q, oof_c)
        rows.append({"horizon": h, "qlstm_auroc": aq, "classical_gate_auroc": ac,
                     "diff": aq - ac, "delong_p": p,
                     "classical_lo": lo, "classical_hi": hi,
                     "qlstm_auprc": mq["auprc"], "classical_auprc": mc["auprc"]})
        star = "*" if p < 0.05 else " "
        print(f"h={h:2d}  quantum gate={aq:.3f}   classical gate={ac:.3f} "
              f"[{lo:.3f},{hi:.3f}]   diff={aq-ac:+.3f}  p={p:.4f}{star} "
              f"({time.time()-t0:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv("results/logs/gate_ablation.csv", index=False)
    d = pd.DataFrame(rows)
    print(f"\nquantum gate ahead at {int((d['diff']>0).sum())}/3 horizons; "
          f"significant at {int((d.delong_p<0.05).sum())}")


if __name__ == "__main__":
    main()
