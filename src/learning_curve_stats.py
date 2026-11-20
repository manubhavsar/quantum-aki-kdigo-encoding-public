"""Summarise the learning-curve experiment with paired CIs and a trend test."""
import numpy as np, pandas as pd
from scipy import stats

def main():
    d = pd.read_csv("results/logs/learning_curve.csv")
    rows = []
    for (f, s), g in d.groupby(["frac", "seed"]):
        m = g.set_index("model").auroc
        if "VQC" not in m.index:
            continue
        bc = m[[c for c in ("LogisticRegression", "XGBoost", "RandomForest")
                if c in m.index]].max()
        rows.append({"frac": f, "seed": s, "gap": m["VQC"] - bc,
                     "vqc": m["VQC"], "best_classical": bc})
    g = pd.DataFrame(rows)
    g.to_csv("results/logs/learning_curve_gap.csv", index=False)

    summ = g.groupby("frac").gap.agg(["mean", "std", "count"]).reset_index()
    summ["ci_lo"] = summ["mean"] - 1.96 * summ["std"] / np.sqrt(summ["count"])
    summ["ci_hi"] = summ["mean"] + 1.96 * summ["std"] / np.sqrt(summ["count"])
    # At frac = 1.0 no subsampling occurs, so every seed gives the identical
    # split and the paired t-test is degenerate (zero variance). Mark it rather
    # than reporting a meaningless p-value of 1.
    ps, deg = [], []
    for f in summ.frac:
        v = g[g.frac == f].gap.values
        if np.allclose(v, v[0]):
            ps.append(np.nan); deg.append(True)
        else:
            ps.append(stats.ttest_1samp(v, 0).pvalue); deg.append(False)
    summ["p"] = ps
    summ["degenerate_no_subsampling"] = deg
    rho, pv = stats.spearmanr(g.frac, g.gap)
    summ["spearman_rho"] = rho
    summ["spearman_p"] = pv
    summ.to_csv("results/logs/learning_curve_stats.csv", index=False)
    print(summ.round(4).to_string(index=False))
    print(f"\ntrend: Spearman rho = {rho:.3f}, p = {pv:.2e}")
    par = summ[(summ.p >= 0.05) & (~summ.degenerate_no_subsampling)].frac.tolist()
    print(f"fractions at statistical parity (p>=0.05): {par}")
    print("note: frac=1.00 has no seed variance (no subsampling); its gap is a "
          "single deterministic value and is tested by DeLong in the main analysis.")

if __name__ == "__main__":
    main()
