"""Generate LaTeX tables and \newcommand macros directly from result CSVs.

Every number that appears in the manuscript is emitted here, so the text cannot
drift from what the experiments actually produced.
"""
import os, sys, json
import numpy as np, pandas as pd

OUT = "paper"
NICE = {"LogisticRegression": "Logistic Regression", "RandomForest": "Random Forest",
        "QLSTM": "Minimal-QLSTM", "MLP": "MLP",
        "XGBoost": "XGBoost", "VQC": "VQC", "QSVM": "Quantum-kernel SVM",
        "LSTM": "LSTM", "TRE": "TRE (proposed)", "TRE_noent": "TRE, no interleaving"}
BNICE = {"static": "static", "delta": "delta", "combined": "combined",
         "trajectory": "trajectory"}


def esc(s):
    return str(s).replace("_", r"\_")


DIGITS = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
          "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}


def macro(name, val):
    # LaTeX control sequences may only contain letters, so digits in generated
    # macro names (e.g. horizon 12) are spelled out.
    name = "".join(DIGITS.get(c, c) for c in str(name))
    return f"\\newcommand{{\\{name}}}{{{val}}}\n"


def pfmt(p):
    """Bootstrap/DeLong p-values of exactly 0 are resolution limits, not zeros."""
    p = float(p)
    if p < 0.001:
        return r"{<}0.001"
    return f"{{=}}{p:.3f}"


def main():
    macros = ""
    # ---- data card macros
    dc = json.load(open("results/logs/data_card.json"))
    e, m = dc["eicu"], dc["mimic"]
    macros += macro("NStays", f"{e['n_stays_with_ge2_cr']:,}")
    macros += macro("NPatients", f"{e['n_patients']:,}")
    macros += macro("NHospitals", e["n_hospitals"])
    macros += macro("NPoints", f"{e['n_prediction_points']:,}")
    macros += macro("StayAKIRate", f"{100*e['stay_level_aki_rate']:.1f}\\%")
    macros += macro("StayAKIRateUO", f"{100*e['stay_level_aki_rate_uo_sensitivity']:.1f}\\%")
    macros += macro("ExtStays", m["n_stays_with_ge2_cr"])
    macros += macro("ExtPatients", m["n_patients"])
    macros += macro("ExtPoints", m["n_prediction_points"])
    for h in (12, 24, 48):
        macros += macro(f"NH{h}", f"{e[f'h{h}_n']:,}")
        macros += macro(f"PosH{h}", e[f"h{h}_pos"])
        macros += macro(f"RateH{h}", f"{100*e[f'h{h}_pos_rate']:.1f}\\%")

    allr = pd.read_csv("results/logs/all_results.csv")

    # ---- main results table: models x horizons, AUROC [CI] with AUPRC beneath
    # Families with both encodings occupy one row each, shown as static->delta, so
    # the delta-vs-static contrast is checkable cell by cell without doubling the
    # table's height. Single-encoding models keep their own row.
    paired = ["LogisticRegression", "RandomForest", "XGBoost", "MLP", "VQC", "QSVM"]
    order = [(m, "static|delta") for m in paired] + [
             ("LogisticRegression", "combined"),
             ("LSTM", "trajectory"), ("QLSTM", "trajectory"),
             ("TRE_noent", "trajectory"), ("TRE", "trajectory")]
    piv = {(r["model"], r["block"], r["horizon"]): r for _, r in allr.iterrows()}
    lines = [r"\begin{table}[!htbp]", r"\centering", r"\scriptsize", r"\renewcommand{\arraystretch}{1.00}",
             r"\setlength{\tabcolsep}{3.2pt}",
             r"\caption{Discrimination on the eICU cohort: AUROC with 95\% "
             r"patient-level bootstrap intervals, AUPRC in parentheses, from pooled "
             r"out-of-fold predictions of patient-grouped 5-fold cross-validation. "
             r"For families evaluated on both, the cell reads "
             r"static$\to$delta with the interval and AUPRC for delta. "
             r"\emph{static} = levels; \emph{delta} = KDIGO-aligned rates of change; "
             r"\emph{traj.} = raw trajectory. "
             r"Prevalence \RateHOneTwo{}, \RateHTwoFour{}, \RateHFourEight{}.}",
             r"\label{tab:main}",
             r"\begin{tabular}{llccc}", r"\hline",
             r"Model & Enc. & 12\,h & 24\,h & 48\,h \\", r"\hline"]
    for mdl, blk in order:
        cells = []
        for h in (12, 24, 48):
            if blk == "static|delta":
                st, dl = piv.get((mdl, "static", h)), piv.get((mdl, "delta", h))
                if st is None or dl is None:
                    cells.append("--")
                else:
                    cells.append(f"{st['auroc']:.3f}$\\to${dl['auroc']:.3f} "
                                 f"[{dl['auroc_lo']:.2f},{dl['auroc_hi']:.2f}] "
                                 f"({dl['auprc']:.2f})")
                continue
            r = piv.get((mdl, blk, h))
            if r is None:
                cells.append("--")
            else:
                cells.append(f"{r['auroc']:.3f} [{r['auroc_lo']:.2f},{r['auroc_hi']:.2f}] "
                             f"({r['auprc']:.2f})")
        bl = {"trajectory": "traj.", "static|delta": "stat.$\\to$delta"}.get(blk, blk)
        lines.append(f"{NICE.get(mdl, mdl)} & {bl} & " + " & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    open(f"{OUT}/tab_main.tex", "w").write("\n".join(lines))

    # ---- interaction table: pairs x horizons (compact)
    if os.path.exists("results/logs/interaction_tests.csv"):
        it = pd.read_csv("results/logs/interaction_tests.csv")
        L = [r"\begin{table}[!htbp]", r"\centering", r"\scriptsize", r"\renewcommand{\arraystretch}{1.00}",
             r"\setlength{\tabcolsep}{3.2pt}",
             r"\caption{The encoding-by-model interaction $I$ by horizon: the AUROC gain "
             r"the quantum model takes from delta encoding minus the classical "
             r"model's gain from the same features, with a 95\% patient-level "
             r"bootstrap interval. $I$ measures how differently the two model "
             r"families exploit the encoding; evidence for a quantum-component "
             r"contribution comes from $I$ together with the circuit ablations. "
             r"$^{*}$ $p<0.05$; "
             r"$^{\dagger}$ survives Bonferroni correction ($p<\BonfThresh{}$).}",
             r"\label{tab:interaction}",
             r"\begin{tabular}{llccc}", r"\hline",
             r"Quantum & vs.\ classical & 12\,h & 24\,h & 48\,h \\", r"\hline"]
        nb = 0.05 / max(len(it), 1)
        for qm in ("VQC", "QSVM"):
            for cm in ("LogisticRegression", "XGBoost", "RandomForest"):
                cells = []
                for h in (12, 24, 48):
                    r = it[(it.horizon == h) & (it.quantum == qm) & (it.classical == cm)]
                    if not len(r):
                        cells.append("--"); continue
                    r = r.iloc[0]
                    mark = (r"$^{\dagger}$" if r["p"] < nb
                            else r"$^{*}$" if r["p"] < 0.05 else "")
                    cells.append(f"{r['interaction']:+.3f} [{r['lo']:+.2f},{r['hi']:+.2f}]{mark}")
                L.append(f"{NICE.get(qm,qm)} & {NICE.get(cm,cm)} & " + " & ".join(cells) + r" \\")
        L += [r"\hline", r"\end{tabular}", r"\end{table}"]
        open(f"{OUT}/tab_interaction.tex", "w").write("\n".join(L))
        # headline macros
        best = it.loc[it.interaction.abs().idxmax()]
        macros += macro("MaxInteraction", f"{best['interaction']:+.3f}")
        macros += macro("NInterPos", int((it.interaction > 0).sum()))
        macros += macro("NInterTotal", len(it))
        macros += macro("NInterSig", int((it.p < 0.05).sum()))

    # ---- extra macros used in the Results prose
    if os.path.exists("results/logs/interaction_tests.csv"):
        it2 = pd.read_csv("results/logs/interaction_tests.csv")
        v = it2[it2.quantum == "VQC"]; k = it2[it2.quantum == "QSVM"]
        macros += macro("VQCInterPos", f"{int((v.interaction>0).sum())}/{len(v)}")
        macros += macro("VQCInterMean", f"{v.interaction.mean():+.3f}")
        macros += macro("QSVMInterPos", f"{int((k.interaction>0).sum())}/{len(k)}")
        macros += macro("QSVMInterMean", f"{k.interaction.mean():+.3f}")
        macros += macro("BonfThresh", f"{0.05/len(it2):.4f}")
    # patient-clustered inference replaces DeLong for all reported comparisons
    if os.path.exists("results/logs/clustered_inference.csv"):
        ci = pd.read_csv("results/logs/clustered_inference.csv")
        def _ci(lbl, h, field):
            r = ci[(ci.comparison == lbl) & (ci.horizon == h)]
            return None if not len(r) else r.iloc[0][field]
        for h in (12, 24, 48):
            for lbl, tag in (("TRE interleaving vs no interleaving", "TREc"),
                             ("QLSTM quantum gate vs classical gate", "GATEc")):
                d = _ci(lbl, h, "diff"); pv = _ci(lbl, h, "p_clustered")
                lo = _ci(lbl, h, "ci_lo"); hi = _ci(lbl, h, "ci_hi")
                if d is None:
                    continue
                macros += macro(f"{tag}diffH{h}", f"{d:+.3f}")
                macros += macro(f"{tag}pH{h}", pfmt(pv))
                macros += macro(f"{tag}ciH{h}", f"[{lo:+.3f},{hi:+.3f}]")
        macros += macro("NClusteredFlips",
                        int(((ci.p_clustered < 0.05) != (ci.p_delong < 0.05)).sum()))
        macros += macro("NClusteredTests", len(ci))

    # MLP control interaction
    if os.path.exists("results/logs/interaction_mlp.csv"):
        im = pd.read_csv("results/logs/interaction_mlp.csv")
        r = im[im.horizon == 24].iloc[0]
        macros += macro("MLPInteraction", f"{r.interaction:+.3f}")
        macros += macro("MLPInteractionCI", f"[{r.lo:+.3f},{r.hi:+.3f}]")
        macros += macro("MLPInteractionP", pfmt(r.p))
        macros += macro("MLPDeltaGain", f"{r.delta_classical:+.3f}")

    # TRE ablation numbers
    if os.path.exists("results/logs/tre_ablation.csv"):
        ta = pd.read_csv("results/logs/tre_ablation.csv")
        for _, r in ta.iterrows():
            h = int(r["horizon"])
            macros += macro(f"TREdiffH{h}", f"{r['diff']:+.3f}")
            macros += macro(f"TREciH{h}", f"[{r['lo']:+.3f},{r['hi']:+.3f}]")
            macros += macro(f"TREpH{h}", pfmt(r["p"]))
    # ---- external validation table
    if os.path.exists("results/logs/external_validation.csv"):
        ev = pd.read_csv("results/logs/external_validation.csv")
        L = [r"\begin{table}[!htbp]", r"\centering", r"\scriptsize", r"\renewcommand{\arraystretch}{1.00}",
             r"\setlength{\tabcolsep}{3.2pt}",
             r"\caption{External validation on MIMIC-IV (trained on the full eICU cohort, "
             r"no refitting). Four best configurations per horizon.}", r"\label{tab:external}",
             r"\begin{tabular}{llcc}", r"\hline",
             r"Horizon & Model (encoding) & AUROC & AUPRC \\", r"\hline"]
        ev = (ev.sort_values(["horizon", "auroc"], ascending=[True, False])
                .groupby("horizon", as_index=False).head(3))
        for _, r in ev.iterrows():
            L.append(f"{int(r['horizon'])}\\,h & {NICE.get(r['model'],r['model'])} "
                     f"({BNICE.get(r['block'],r['block'])}) & "
                     f"{r['auroc']:.3f} [{r['auroc_lo']:.3f},{r['auroc_hi']:.3f}] & "
                     f"{r['auprc']:.3f} \\\\")
        L += [r"\hline", r"\end{tabular}", r"\end{table}"]
        open(f"{OUT}/tab_external.tex", "w").write("\n".join(L))

    # ---- ablation table
    if os.path.exists("results/logs/ablations.csv"):
        ab = pd.read_csv("results/logs/ablations.csv")
        L = [r"\begin{table}[!htbp]", r"\centering", r"\footnotesize",
             r"\caption{Ablations on the 24\,h horizon (VQC, delta encoding unless stated).}",
             r"\label{tab:ablation}", r"\begin{tabular}{llcc}", r"\hline",
             r"Ablation & Setting & AUROC & AUPRC \\", r"\hline"]
        for name, grp in ab.groupby("ablation", sort=False):
            for _, r in grp.iterrows():
                v = r["value"]
                try:
                    v = f"{float(v):g}"
                except (TypeError, ValueError):
                    v = esc(v)
                L.append(f"{esc(name)} & {v} & {r['auroc']:.3f} & {r['auprc']:.3f} \\\\")
            L.append(r"\hline")
        L += [r"\end{tabular}", r"\end{table}"]
        open(f"{OUT}/tab_ablation.tex", "w").write("\n".join(L))

    # ---- taxonomy tables, separated by estimand
    if os.path.exists("results/logs/artifact_taxonomy.json"):
        gj = json.load(open("results/logs/artifact_taxonomy.json"))
        A = pd.DataFrame(gj["absolute"]); B = pd.DataFrame(gj["gap"])
        ref = gj["largest_controlled_effect"]
        L = [r"\begin{table}[!htbp]", r"\centering", r"\scriptsize", r"\renewcommand{\arraystretch}{1.00}",
             r"\setlength{\tabcolsep}{3.2pt}",
             r"\caption{Audited evaluation choices, each varied in isolation on the "
             r"same cohort. Panel (b) reports the shift in the quantum--classical gap "
             r"$G=\mathrm{AUROC}(Q)-\mathrm{AUROC}(C)$, the quantity an advantage "
             r"claim asserts; panel (a) reports inflation of absolute discrimination, "
             r"which distorts any reported figure but affects both arms. The largest "
             r"quantum-component effect significant under clustered inference is $"
             + f"{ref:+.3f}" + r"$.}",
             r"\label{tab:taxonomy}",
             r"\begin{tabular}{lcl}", r"\hline",
             r"\multicolumn{3}{l}{\textit{(a) inflation of absolute discrimination}} \\",
             r"Evaluation choice & $\Delta$AUROC & Measured as \\", r"\hline"]
        for _, r in A.iterrows():
            L.append(f"{esc(r['choice'])} & {r['magnitude']:+.3f} & {esc(r['detail'])} \\\\")
        L += [r"\hline",
              r"\multicolumn{3}{l}{\textit{(b) shift in the quantum--classical gap $G$}} \\",
              r"Evaluation choice & $\Delta G$ & Measured as \\", r"\hline"]
        for _, r in B.iterrows():
            mark = r"$^{\ddagger}$" if r["magnitude"] > ref else ""
            L.append(f"{esc(r['choice'])}{mark} & {r['magnitude']:+.3f} & "
                     f"{esc(r['detail'])} \\\\")
        L += [r"\hline",
              r"\multicolumn{3}{l}{\footnotesize $^{\ddagger}$ exceeds the largest "
              r"quantum-component effect established under control.}",
              r"\end{tabular}", r"\end{table}"]
        open(f"{OUT}/tab_taxonomy.tex", "w").write("\n".join(L))
        macros += macro("NGapChoices", len(B))
        macros += macro("NGapExceed", int((B.magnitude > ref).sum()))
        macros += macro("MaxGapShift", f"{B.magnitude.max():+.3f}")
        macros += macro("MinGapShift", f"{B.magnitude.min():+.3f}")
        macros += macro("AbsInflation", f"{A.magnitude.max():+.3f}")
        macros += macro("ControlledEffect", f"{ref:+.3f}")
        macros += macro("GapRatio", f"{B.magnitude.max()/ref:.1f}")

    # ---- capacity sweep of the classical control
    if os.path.exists("results/logs/capacity_scaling.csv"):
        cs = pd.read_csv("results/logs/capacity_scaling.csv")
        macros += macro("CapNTotal", len(cs))
        macros += macro("CapMinInter", f"{cs.interaction.min():+.3f}")
        macros += macro("CapMaxInter", f"{cs.interaction.max():+.3f}")
        macros += macro("CapMinParams", f"{int(cs.mlp_params.min()):,}")
        macros += macro("CapMaxParams", f"{int(cs.mlp_params.max()):,}")

    open(f"{OUT}/results_macros.tex", "w").write(
        "% auto-generated by src/make_tables.py -- do not edit\n" + macros)
    print(f"wrote {OUT}/results_macros.tex and tables")
    print(macros)


if __name__ == "__main__":
    main()
