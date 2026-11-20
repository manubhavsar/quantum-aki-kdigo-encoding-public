"""Derived quantities quoted in the revised manuscript, computed from logs.

Keeps every number the revision quotes traceable: seed means and SDs, the
Holm-adjusted ablation p-values, the matched-input lead, and the fit-time ratio.
Writes results/logs/round2_summary.csv (name, value).
"""
import pandas as pd

L = "results/logs/"
out = {}

r = pd.read_csv(L + "seed_robustness_tre.csv")
for h in (24, 48):
    x = r[r.horizon == h]["diff"]
    out[f"tre_seed_mean_{h}"] = x.mean()
    out[f"tre_seed_sd_{h}"] = x.std(ddof=1)
    out[f"tre_seed_positive_{h}"] = int((x > 0).sum())

s = pd.read_csv(L + "seed_sensitivity.csv")
out["vqc_seed_mean_24"] = s.vqc.mean()
out["vqc_seed_sd_24"] = s.vqc.std(ddof=1)

# Holm across the six pre-specified ablation tests (two controls x three horizons)
pc = pd.read_csv(L + "paired_comparisons.csv")
abl = pc[pc.pair.isin(["TRE vs additive control", "Minimal-QLSTM vs classical-gate LSTM"])]
abl = abl.sort_values("p").reset_index(drop=True)
m, prev = len(abl), 0.0
for i, row in abl.iterrows():
    adj = max(prev, min(1.0, (m - i) * row.p)); prev = adj
    tag = ("tre" if row.pair.startswith("TRE") else "gate") + f"_{int(row.horizon)}"
    out[f"holm_{tag}"] = adj

# Holm across every independent quantum-vs-classical test in Table 3(a)
ind = pc[~pc.pair.isin(["TRE vs additive control", "Minimal-QLSTM vs classical-gate LSTM"])]
ind = ind.sort_values("p").reset_index(drop=True)
m, prev = len(ind), 0.0
adj = []
for i, row in ind.iterrows():
    prev = max(prev, min(1.0, (m - i) * row.p)); adj.append(prev)
out["independent_tests"] = m
out["independent_min_p"] = ind.p.min()
out["independent_min_holm"] = min(adj)

c = pd.read_csv(L + "curvature_quantum.csv")
for h in (24, 48):
    g = c[(c.horizon == h) & (c.block == "delta+curv4")].set_index("model").auroc
    out[f"matched_curvature_lead_{h}"] = g["LR"] - g["VQC"]

k = pd.read_csv(L + "complexity.csv").set_index("model")
out["tre_over_lstm_fit_ratio"] = k.loc["TRE (6 qubits)", "fit_seconds"] / k.loc["LSTM", "fit_seconds"]

pd.Series(out, name="value").rename_axis("name").to_csv(L + "round2_summary.csv")
for a, b in out.items():
    print(f"{a:32s} {b:.4f}")
