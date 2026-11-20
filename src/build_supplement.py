"""Supplementary tables S1-S7, generated from the experiment logs.

Writes paper/supplement_body.tex (tables only). make_package inlines it after the
response in the combined PDF and wraps it as a standalone supplementary PDF.
"""
import os
import pandas as pd

L = "results/logs/"
T = []


def table(label, caption, colspec, header, rows):
    T.append("\\begin{table}[!htbp]\n\\centering\n\\footnotesize\n"
             f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\begin{{tabular}}{{{colspec}}}\n\\hline\n"
             f"{header} \\\\\n\\hline\n" + "\n".join(rows) + "\n\\hline\n\\end{tabular}\n\\end{table}\n")


# S1: interaction table moved out of the main text
blk = open(L + "tab_interaction_block.tex").read()
blk = blk.replace("\\label{tab:interaction}", "\\label{tab:s1}").replace("\\BonfThresh{}", "0.0028")
blk = blk.replace("(Section~\\ref{sec:res-controls})", "(Section 5.3)").replace("\\scriptsize", "\\footnotesize")
T.append(blk + "\n")

# S2: TRE ablation over seeds
sr = pd.read_csv(L + "seed_robustness_tre.csv")
rows = [f"{int(r.horizon)}\\,h & {int(r.seed)} & {r.tre:.3f} & {r.tre_noent:.3f} & {r['diff']:+.3f} \\\\" for _, r in sr.iterrows()]
rows += [f"\\multicolumn{{4}}{{l}}{{{h}\\,h, mean $\\pm$ SD over five seeds}} & "
         f"{sr[sr.horizon == h]['diff'].mean():+.3f} $\\pm$ {sr[sr.horizon == h]['diff'].std(ddof=1):.3f} \\\\" for h in (24, 48)]
table("tab:s2", "TRE interleaving ablation over five initialisation seeds (bandwidth fixed at 1.0, identical folds). "
      "Difference is interleaved minus additive control; it is positive in all ten runs.",
      "lcccc", "Horizon & Seed & TRE & Additive control & Difference", rows)

# S3: urine-inclusive endpoint, all seven tabular model families
u1 = pd.read_csv(L + "urine_sensitivity_vqc.csv").set_index(["endpoint", "horizon"])
u2 = pd.read_csv(L + "urine_sensitivity_more.csv").set_index(["endpoint", "horizon"])
u3 = pd.read_csv(L + "urine_sensitivity_rest.csv").set_index(["endpoint", "horizon"])
rows = []
for ep in ("creatinine-only", "creatinine+urine"):
    for h in (12, 24, 48):
        k = (ep, h)
        rows.append(f"{ep} & {h}\\,h & {int(u1.loc[k, 'n']):,} & {100 * u1.loc[k, 'prevalence']:.1f}\\% & "
                    f"{u1.loc[k, 'lr']:.3f} & {u2.loc[k, 'rf']:.3f} & {u2.loc[k, 'xgb']:.3f} & {u2.loc[k, 'mlp']:.3f} & "
                    f"{u3.loc[k, 'tabt']:.3f} & {u1.loc[k, 'vqc']:.3f} & {u3.loc[k, 'qsvm']:.3f} \\\\")
table("tab:s3", "AUROC under the creatinine-only (primary) and urine-inclusive endpoints for all seven tabular model "
      "families, delta block, one fixed untuned protocol per model on identical folds. Absolute values therefore "
      "differ from the tuned values in Table 1.",
      "llrrccccccc", "Endpoint & Horizon & Points & Prev. & LR & RF & XGB & MLP & Tab.\\ transf. & VQC & Kernel SVM", rows)

# S4: identical curvature features for both arms
cq = pd.read_csv(L + "curvature_quantum.csv")
rows = [f"{int(r.horizon)}\\,h & {r.model} & {'delta' if r.block == 'delta' else 'delta + 4 curvature'} & "
        f"{int(r.n_features)} & {r.auroc:.3f} [{r.lo:.3f}, {r.hi:.3f}] \\\\" for _, r in cq.iterrows()]
table("tab:s4", "Both arms given identical curvature features (Section 5.4): logistic regression and the VQC "
      "(one qubit per feature), fixed untuned protocol, 95\\% patient-level bootstrap intervals.",
      "lllcc", "Horizon & Model & Feature block & Features & AUROC [95\\% CI]", rows)

# S5: tabular transformer
tt = pd.read_csv(L + "tab_transformer.csv")
rows = [f"{int(r.horizon)}\\,h & {r.block} & {r.auroc:.3f} [{r.auroc_lo:.3f}, {r.auroc_hi:.3f}] & {r.auprc:.3f} \\\\" for _, r in tt.iterrows()]
table("tab:s5", "Transformer on the tabular feature blocks (feature-token design, 2{,}545 parameters): the classical "
      "attention model on exactly the VQC's and the kernel SVM's inputs.",
      "llcc", "Horizon & Block & AUROC [95\\% CI] & AUPRC", rows)

# S6: matched pairs over five seeds at 24 h (if the run has finished)
INCLUDE_SEED_PAIRS = False  # run continues in background; not yet reported
if INCLUDE_SEED_PAIRS and os.path.exists(L + "seed_pairs_24h.csv"):
    sp = pd.read_csv(L + "seed_pairs_24h.csv")
    if sp.seed.nunique() >= 5:
        order = list(dict.fromkeys(sp.pair))
        g = sp.groupby("pair")["diff"]
        rows = [f"{p} & {g.mean()[p]:+.3f} $\\pm$ {g.std(ddof=1)[p]:.3f} & {g.min()[p]:+.3f} to {g.max()[p]:+.3f} & "
                f"{int((sp[sp.pair == p]['diff'] > 0).sum())}/{sp[sp.pair == p].seed.nunique()} \\\\" for p in order]
        table("tab:s6", "Every matched pair of Table 2(a) at 24\\,h, refitted with five initialisation seeds under one "
              "fixed, untuned configuration per model (bandwidth or scale 1.0; VQC two layers; kernel SVM on at most 700 "
              "rows per fold). Values are AUROC differences, quantum minus classical; absolute levels differ from the "
              "tuned single-seed values in Table 2(a). (abl.) marks ablations of the quantum model.",
              "lccc", "Pair & Mean $\\pm$ SD & Range & Positive seeds", rows)

# S7: sources for Table 5
pp = pd.read_csv(L + "published_predictors.csv")
def _s7_row(r):
    study = r.study.replace("Tomasev", "Toma\\v{s}ev")
    if isinstance(r.reported, str):
        rep = r.reported.replace('%', '\\%').replace('>=', '$\\geq$')
    else:  # eICU row: supports the Limitations, not Table 4
        study += " (Limitations)"
        rep = f"{r.cohort} (not in Table 4)"
    return f"{study} & \\texttt{{{r.doi}}} & {rep} \\\\"
rows = [_s7_row(r) for _, r in pp.iterrows()]
table("tab:s7", "Sources for the published figures in Table 4 and, in the last row, for the full-release size quoted in the Limitations (Section 7); each checked against the abstract of the cited paper.",
      "p{3.0cm}p{4.4cm}p{7.2cm}", "Study & DOI & Reported", rows)

env = pd.read_csv(L + "complexity_env.csv", index_col=0).value
T.append("\\paragraph{Computational environment} All experiments ran on "
         f"{env['system'].replace('_', '-')}, {env['cpu_count']} CPU cores ({env['machine']}), no GPU; Python "
         f"{env['python']}, PyTorch {env['torch']}, PennyLane {env['pennylane']} (\\texttt{{default.qubit}}, analytic "
         "expectations). Timings in Table 2(b) are medians of three runs on one fold of "
         f"{env['n_train']} training and {env['n_test']} test points. Every table here and in the manuscript is "
         "regenerated from the logs in the anonymous repository; raw PhysioNet data are not redistributed.\n")
open("paper/supplement_body.tex", "w").write("\n".join(T))
print("wrote paper/supplement_body.tex with", sum(1 for t in T if "\\begin{table}" in t), "tables")
