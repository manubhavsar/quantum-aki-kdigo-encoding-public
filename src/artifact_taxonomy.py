"""Audited evaluation choices, separated by which quantity they move.

A magnitude is only interpretable against a stated estimand. Two are used, and
they are reported separately because conflating them overstates the case:

  (A) absolute discrimination -- the AUROC a paper would report for one model;
  (B) the quantum-classical gap  G = AUROC(quantum) - AUROC(classical)
      evaluated on identical samples, which is what an advantage claim asserts.

Label leakage is the clearest illustration of why the distinction matters: it
inflates absolute AUROC substantially for BOTH model families, so it distorts any
reported performance figure, yet it moves G comparatively little because both arms
benefit. Choices that act asymmetrically on the two arms move G directly.
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))


def main():
    absolute, gap = [], []

    # ---------- (A) choices that inflate absolute discrimination
    lg = pd.read_csv("results/logs/leakage_gap.csv")
    inc, leak = lg.iloc[0], lg.iloc[1]
    absolute.append({"choice": "Label leakage (post-onset windows retained)",
                     "magnitude": float(leak.vqc - inc.vqc),
                     "detail": f"VQC {inc.vqc:.3f}$\\to${leak.vqc:.3f}, LR {inc.logreg:.3f}$\\to${leak.logreg:.3f}",
                     "source": "leakage_gap.csv"})

    # ---------- (B) choices that shift the quantum-classical gap G
    qc = pd.read_csv("results/logs/qlstm_comparison.csv")
    ga = pd.read_csv("results/logs/gate_ablation.csv")
    unm = float(qc[(qc.horizon == 12) & (qc.model_b == "LSTM|trajectory")]["diff"].iloc[0])
    mat = float(ga[ga.horizon == 12]["diff"].iloc[0])
    gap.append({"choice": "Unmatched classical architecture", "magnitude": unm - mat,
                "detail": f"{unm:+.3f} vs generic LSTM, {mat:+.3f} vs width-matched", "source": "gate_ablation.csv"})

    ab = pd.read_csv("results/logs/ablations.csv")
    bw = ab[ab.ablation == "bandwidth"]
    gap.append({"choice": "Unreported encoding bandwidth",
                "magnitude": float(bw.auroc.max() - bw.auroc.min()),
                "detail": f"quantum arm {bw.auroc.min():.3f}$\\to${bw.auroc.max():.3f}; absent classically",
                "source": "ablations.csv"})

    lc = pd.read_csv("results/logs/learning_curve_stats.csv")
    gap.append({"choice": "Cohort scale (demo-scale evaluation only)",
                "magnitude": float(lc[lc.frac == 0.10]["mean"].iloc[0]
                                   - lc[lc.frac == 1.00]["mean"].iloc[0]),
                "detail": f"G {lc[lc.frac==1.0]['mean'].iloc[0]:+.3f} at full data, {lc[lc.frac==0.1]['mean'].iloc[0]:+.3f} at one tenth",
                "source": "learning_curve_stats.csv"})

    ho = pd.read_csv("results/logs/horizon_artifact.csv")
    gap.append({"choice": "Horizon selection (best of three)",
                "magnitude": float(ho.selection_gain.max()),
                "detail": f"best-horizon G exceeds the mean by {ho.selection_gain.max():+.3f}",
                "source": "horizon_artifact.csv"})

    it = pd.read_csv("results/logs/interaction_tests.csv")
    r = it[(it.horizon == 24) & (it.quantum == "VQC")
           & (it.classical == "LogisticRegression")].iloc[0]
    gap.append({"choice": "Crediting the model for its features",
                "magnitude": float(r.delta_quantum - r.interaction),
                "detail": f"raw gain {r.delta_quantum:+.3f} vs interaction {r.interaction:+.3f}", "source": "interaction_tests.csv"})

    ta = pd.read_csv("results/logs/tuning_asymmetry.csv")
    gap.append({"choice": "Untuned classical baseline", "magnitude": float(ta.inflation.max()),
                "detail": f"up to {ta.inflation.max():+.3f} depending on the baseline",
                "source": "tuning_asymmetry.csv"})

    ss = pd.read_csv("results/logs/seed_sensitivity.csv")
    gap.append({"choice": "Seed selection (best of eight)",
                "magnitude": float(ss.gap.max() - ss.gap.mean()),
                "detail": f"best seed {ss.gap.max():+.3f} against mean {ss.gap.mean():+.3f}",
                "source": "seed_sensitivity.csv"})

    gap.append({"choice": "Label leakage (post-onset windows retained)",
                "magnitude": float(leak.gap_G - inc.gap_G),
                "detail": f"G {inc.gap_G:+.3f}$\\to${leak.gap_G:+.3f}; both arms gain",
                "source": "leakage_gap.csv"})

    A = pd.DataFrame(absolute).sort_values("magnitude", ascending=False)
    B = pd.DataFrame(gap).sort_values("magnitude", ascending=False)
    A.to_csv("results/logs/taxonomy_absolute.csv", index=False)
    B.to_csv("results/logs/taxonomy_gap.csv", index=False)

    ci = pd.read_csv("results/logs/clustered_inference.csv")
    controlled = ci[ci.comparison.str.contains("interleaving|quantum gate")
                    & (ci.p_clustered < 0.05)]
    largest = float(controlled["diff"].max())

    print("=== (A) inflation of absolute discrimination ===")
    for _, r in A.iterrows():
        print(f"  {r.choice:46s} {r.magnitude:+.3f}   [{r.detail}]")
    print("\n=== (B) shift in the quantum-classical gap G ===")
    for _, r in B.iterrows():
        print(f"  {r.choice:46s} {r.magnitude:+.3f}   [{r.detail}]")
    print(f"\nlargest quantum-component effect significant under clustered "
          f"inference: {largest:+.3f}")
    print(f"gap-shifting choices exceeding it: "
          f"{int((B.magnitude > largest).sum())}/{len(B)}")
    json.dump({"absolute": A.to_dict("records"), "gap": B.to_dict("records"),
               "largest_controlled_effect": largest},
              open("results/logs/artifact_taxonomy.json", "w"), indent=2)


if __name__ == "__main__":
    main()
