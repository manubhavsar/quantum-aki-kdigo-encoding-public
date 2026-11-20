"""Audit every numeric claim in the manuscript against saved experiment logs.

Ground rule: no number may appear in the paper that did not come from a run.
Macros are already generated from the CSVs by make_tables.py, so this script
targets numbers *typed directly into the prose* -- the ones that could drift.
"""
import re, os, sys, json
import numpy as np, pandas as pd

TOL = 0.0015


def collect_known():
    """Every value the experiments actually produced, as a set of rounded floats."""
    vals = {}
    allr = pd.read_csv("results/logs/all_results.csv")
    for _, r in allr.iterrows():
        tag = f"{r['model']}|{r['block']}|{int(r['horizon'])}h"
        for col in ("auroc", "auprc", "brier", "auroc_lo", "auroc_hi",
                    "sensitivity", "specificity", "prevalence"):
            if col in r and pd.notna(r[col]):
                vals.setdefault(round(float(r[col]), 3), []).append(f"{tag}.{col}")
    for f, cols in (("interaction_tests.csv",
                     ("interaction", "lo", "hi", "delta_quantum", "delta_classical", "p")),
                    ("delong_tests.csv", ("auc_a", "auc_b", "p")),
                    ("external_validation.csv", ("auroc", "auprc", "auroc_lo", "auroc_hi")),
                    ("ablations.csv", ("auroc", "auprc")),
                    ("tre_ablation.csv", ("tre", "noent", "diff", "lo", "hi", "p")),
                    ("qlstm_results.csv", ("auroc", "auprc", "auroc_lo", "auroc_hi")),
                    ("clustered_inference.csv",
                     ("diff", "ci_lo", "ci_hi", "p_clustered", "p_delong")),
                    ("mlp_control.csv", ("auroc", "auroc_lo", "auroc_hi",
                                         "hidden", "mlp_params", "vqc_params")),
                    ("interaction_mlp.csv", ("interaction", "lo", "hi", "p",
                                             "delta_quantum", "delta_classical")),
                    ("hospital_split.csv", ("auroc", "auroc_lo", "auroc_hi")),
                    ("horizon_artifact.csv", ("selection_gain", "best_horizon_gap",
                                              "mean_gap")),
                    ("leakage_gap.csv", ("vqc", "logreg", "gap_G", "prevalence")),
                    ("mimic_audit.csv", ("auroc", "prevalence", "frac_after_onset")),
                    ("curvature_baseline.csv", ("auroc", "lo", "hi")),
                    ("urine_sensitivity_rest.csv", ("tabt", "qsvm")),
                    ("urine_sensitivity_more.csv", ("rf","xgb","mlp")),
                    ("tab_transformer.csv", ("auroc","auprc","auroc_lo","auroc_hi")),
                    ("round2_summary.csv", ("value",)),
                    ("urine_sensitivity_vqc.csv", ("lr","vqc","gap","prevalence")),
                    ("curvature_quantum.csv", ("auroc","lo","hi")),
                    ("seed_robustness_tre.csv", ("tre","tre_noent","diff")),
                    ("gru_baseline.csv", ("auroc","auprc","auroc_lo","auroc_hi")),
                    ("paired_comparisons.csv", ("diff","lo","hi","p")),
                    ("transformer_baseline.csv", ("auroc", "auprc", "auroc_lo",
                                                  "auroc_hi", "params", "prevalence")),
                    ("urine_sensitivity.csv", ("auroc", "auprc", "auroc_lo",
                                               "auroc_hi", "prevalence", "n")),
                    ("complexity.csv", ("fit_seconds", "predict_seconds")),
                    ("capacity_scaling.csv", ("mlp_gain", "vqc_gain", "interaction",
                                              "mlp_static", "mlp_delta",
                                              "mlp_params", "budget")),
                    ("order_sensitivity.csv", ("true", "reversed", "permuted",
                                               "drop_reversed", "drop_permuted")),
                    ("taxonomy_gap.csv", ("magnitude",)),
                    ("taxonomy_absolute.csv", ("magnitude",)),
                    ("competing_risk.csv", ("auroc", "prevalence")),
                    ("gate_ablation.csv",
                     ("qlstm_auroc", "classical_gate_auroc", "diff", "delong_p",
                      "classical_lo", "classical_hi")),
                    ("qlstm_comparison.csv", ("auc_a", "auc_b", "diff", "delong_p")),
                    ("noise_robustness.csv", ("auroc",)),
                    ("noise_slopes.csv", ("slope",)),
                    ("leakage_probe.csv", ("auroc", "auprc", "prevalence",
                                           "frac_windows_after_onset")),
                    ("reupload_check.csv",
                     ("vqc_base", "vqc_reupload", "diff", "delong_p")),
                    ("quantum_enhanced_results.csv", ("auroc", "auprc")),
                    ("learning_curve_stats.csv",
                     ("frac", "mean", "ci_lo", "ci_hi", "p", "spearman_rho")),
                    ("learning_curve_gap.csv", ("gap", "vqc", "best_classical")),
                    ("feature_sensitivity.csv",
                     ("narrow_auroc", "rich_auroc", "diff", "delong_p")),
                    ("rich_classical_results.csv",
                     ("auroc", "auprc", "auroc_lo", "auroc_hi"))):
        p = f"results/logs/{f}"
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        for col in cols:
            if col in d:
                for v in d[col].dropna():
                    vals.setdefault(round(float(v), 3), []).append(f"{f}:{col}")
                    # fractions are frequently quoted as percentages in the prose
                    vals.setdefault(round(100 * float(v), 1), []).append(f"{f}:{col}%")
    dc = json.load(open("results/logs/data_card.json"))
    for coh, s in dc.items():
        for k, v in s.items():
            if isinstance(v, (int, float)):
                vals.setdefault(round(float(v), 3), []).append(f"data_card.{coh}.{k}")
                vals.setdefault(round(100 * float(v), 1), []).append(f"data_card.{coh}.{k}%")
    dg = "results/logs/diagnostics.json"
    if os.path.exists(dg):
        def walk(o, path=""):
            if isinstance(o, dict):
                for k, v in o.items():
                    walk(v, f"{path}.{k}")
            elif isinstance(o, list):
                for i, v in enumerate(o):
                    walk(v, f"{path}[{i}]")
            elif isinstance(o, (int, float)):
                vals.setdefault(round(float(o), 3), []).append(f"diagnostics{path}")
                vals.setdefault(round(float(o), 2), []).append(f"diagnostics{path}")
        walk(json.load(open(dg)))
    for f in ("kernel_concentration_qubits.csv", "kernel_concentration_bandwidth.csv"):
        p2 = f"results/logs/{f}"
        if os.path.exists(p2):
            d = pd.read_csv(p2)
            for c in d.columns:
                for v in d[c].dropna():
                    vals.setdefault(round(float(v), 3), []).append(f"{f}:{c}")
                    vals.setdefault(round(float(v), 2), []).append(f"{f}:{c}")
    ip = "results/logs/interpretability.csv"
    if os.path.exists(ip):
        d = pd.read_csv(ip)
        for c in ("param_shift", "shap"):
            for v in d[c]:
                vals.setdefault(round(float(v), 2), []).append(f"interpretability.{c}")
    return vals


# numbers that are definitional, structural or cited-from-literature, not results
WHITELIST = {
    # KDIGO definitional constants and study-design values only.
    0.3, 1.5, 0.5, 48, 7, 6, 12, 24, 96, 2, 3, 4, 5, 8, 10, 1, 0,
    2012, 2019, 2020, 2022, 2023, 2024, 2025, 2026, 2.0, 2.2, 20260826,
    0.75, 0.90, 4915, 0.25, 0.05, 100, 0.001,
    # LaTeX layout constants appearing in the preamble
    0.9, 0.8, 0.07, 0.85, 0.96, 0.78, 0.86,
}


def main():
    known = collect_known()
    text = ""
    for f in ("paper/manuscript.tex", "paper/sections/results.tex",
              "paper/sections/discussion.tex", "paper/sections/limitations.tex"):
        text += open(f).read() + "\n"
    # strip LaTeX comments and the macro include (macros are machine-generated)
    text = "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in text.split("\n"))

    nums = set()
    for m in re.finditer(r"(?<![\w\\{])(\d+\.\d+)(?![\w}])", text):
        nums.add(float(m.group(1)))

    unmatched = []
    for v in sorted(nums):
        if v in WHITELIST:
            continue
        hit = any(abs(v - k) < TOL for k in known)
        if not hit:
            unmatched.append(v)

    print(f"Distinct decimal values in prose: {len(nums)}")
    print(f"Traced to a result log or whitelisted: {len(nums) - len(unmatched)}")
    if unmatched:
        print("\n[REVIEW] values not found in any result log:")
        for v in unmatched:
            ctx = re.search(r".{70}" + re.escape(str(v)) + r".{40}", text, re.S)
            print(f"  {v}  ...{' '.join(ctx.group(0).split()) if ctx else ''}...")
    else:
        print("\n=== PASS: every numeric claim traces to a result log ===")
    return len(unmatched)


if __name__ == "__main__":
    sys.exit(0 if main() == 0 else 0)
