"""Phase 5 evaluation: results tables, the 2x2 interaction test, DeLong
comparisons and figures.

The headline scientific quantity is the *interaction*:

    Delta_q  = AUROC(quantum, delta)   - AUROC(quantum, static)
    Delta_c  = AUROC(classical, delta) - AUROC(classical, static)
    interaction = Delta_q - Delta_c

A positive interaction means the KDIGO-aligned temporal encoding helps the
quantum model *more* than the same features help a classical model -- i.e. the
benefit is attributable to the encoding rather than to feature engineering.
Its uncertainty is obtained by resampling patients and recomputing all four
AUROCs on each resample, which preserves the pairing between models.
"""
import os, sys, json, itertools
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from sklearn.metrics import roc_auc_score, average_precision_score
from metrics import delong_test, point_metrics, mcnemar_test

HORIZONS = [12, 24, 48]
P = "results/preds"


def load_pred(model, block, h):
    f = f"{P}/{model}_{block}_{h}h.npy"
    return np.load(f) if os.path.exists(f) else None


def load_y(h):
    return np.load(f"{P}/_y_{h}h.npy"), np.load(f"{P}/_g_{h}h.npy", allow_pickle=True)


def interaction_test(y, groups, q_static, q_delta, c_static, c_delta,
                     n_boot=2000, seed=20260826):
    """Bootstrap the 2x2 interaction, resampling patients (not rows)."""
    preds = [q_static, q_delta, c_static, c_delta]
    obs = ((roc_auc_score(y, q_delta) - roc_auc_score(y, q_static)) -
           (roc_auc_score(y, c_delta) - roc_auc_score(y, c_static)))
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in uniq}
    vals = []
    for _ in range(n_boot):
        gs = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_g[g] for g in gs])
        if len(np.unique(y[idx])) < 2:
            continue
        a = [roc_auc_score(y[idx], p[idx]) for p in preds]
        vals.append((a[1] - a[0]) - (a[3] - a[2]))
    vals = np.array(vals)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    # two-sided bootstrap p-value for interaction != 0
    p = 2 * min((vals <= 0).mean(), (vals >= 0).mean())
    return {"interaction": float(obs), "lo": float(lo), "hi": float(hi),
            "p": float(min(p, 1.0)),
            "delta_quantum": float(roc_auc_score(y, q_delta) - roc_auc_score(y, q_static)),
            "delta_classical": float(roc_auc_score(y, c_delta) - roc_auc_score(y, c_static))}


def main():
    frames = []
    for f in ("classical_results.csv", "quantum_results.csv", "temporal_results.csv",
              "qlstm_results.csv", "mlp_control.csv"):
        p = f"results/logs/{f}"
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
    allr = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    allr.to_csv("results/logs/all_results.csv", index=False)

    print("=" * 96)
    print("MAIN RESULTS (pooled out-of-fold, patient-grouped 5-fold CV, eICU)")
    print("=" * 96)
    for h in HORIZONS:
        sub = allr[allr.horizon == h].sort_values("auroc", ascending=False)
        if not len(sub):
            continue
        print(f"\n--- horizon {h} h  (n={int(sub.iloc[0]['n'])}, "
              f"prevalence {sub.iloc[0]['prevalence']:.3f}) ---")
        print(f"{'model':12s} {'block':11s} {'AUROC':>20s} {'AUPRC':>20s} {'Brier':>7s}")
        for _, r in sub.iterrows():
            print(f"{r['model']:12s} {r['block']:11s} "
                  f"{r['auroc']:.3f} [{r['auroc_lo']:.3f},{r['auroc_hi']:.3f}]  "
                  f"{r['auprc']:.3f} [{r['auprc_lo']:.3f},{r['auprc_hi']:.3f}]  "
                  f"{r['brier']:.3f}")

    # ---------------------------------------------------------- interaction test
    print("\n" + "=" * 96)
    print("2x2 INTERACTION: does delta encoding help the QUANTUM model more than classical?")
    print("=" * 96)
    inter_rows = []
    for h in HORIZONS:
        y, g = load_y(h)
        for qm in ("VQC", "QSVM"):
            qs, qd = load_pred(qm, "static", h), load_pred(qm, "delta", h)
            if qs is None or qd is None:
                continue
            for cm in ("LogisticRegression", "XGBoost", "RandomForest"):
                cs, cd = load_pred(cm, "static", h), load_pred(cm, "delta", h)
                if cs is None or cd is None:
                    continue
                r = interaction_test(y, g, qs, qd, cs, cd)
                r.update({"horizon": h, "quantum": qm, "classical": cm})
                inter_rows.append(r)
                print(f"h={h:2d} {qm:5s} vs {cm:18s} "
                      f"dQ={r['delta_quantum']:+.3f} dC={r['delta_classical']:+.3f} "
                      f"interaction={r['interaction']:+.3f} "
                      f"[{r['lo']:+.3f},{r['hi']:+.3f}] p={r['p']:.3f}")
    pd.DataFrame(inter_rows).to_csv("results/logs/interaction_tests.csv", index=False)

    # -------------------------------------------------- pairwise DeLong per horizon
    print("\n" + "=" * 96)
    print("DeLong tests (best quantum vs best classical, same samples)")
    print("=" * 96)
    dl = []
    for h in HORIZONS:
        y, g = load_y(h)
        avail = {}
        for m in ("VQC", "QSVM", "TRE", "TRE_noent", "LSTM",
                  "LogisticRegression", "XGBoost", "RandomForest"):
            for b in ("static", "delta", "combined", "trajectory"):
                p = load_pred(m, b, h)
                if p is not None and np.isfinite(p).all():
                    avail[f"{m}|{b}"] = p
        for a, bkey in itertools.combinations(sorted(avail), 2):
            au1, au2, pv = delong_test(y, avail[a], avail[bkey])
            dl.append({"horizon": h, "model_a": a, "model_b": bkey,
                       "auc_a": au1, "auc_b": au2, "p": pv})
    pd.DataFrame(dl).to_csv("results/logs/delong_tests.csv", index=False)
    d = pd.DataFrame(dl)
    for h in HORIZONS:
        s = d[(d.horizon == h)].copy()
        if not len(s):
            continue
        s["gap"] = (s.auc_a - s.auc_b).abs()
        print(f"\nh={h}: {len(s)} pairwise comparisons, "
              f"{(s.p < 0.05).sum()} significant at p<0.05")
        for _, r in s.nlargest(4, "gap").iterrows():
            print(f"   {r['model_a']:28s} {r['auc_a']:.3f} vs "
                  f"{r['model_b']:28s} {r['auc_b']:.3f}  p={r['p']:.4f}")
    print("\nsaved results/logs/{all_results,interaction_tests,delong_tests}.csv")


if __name__ == "__main__":
    main()
