"""Persist the standalone diagnostic measurements quoted in the manuscript.

These numbers (urine-charting density, kernel concentration, external AUROC
range) are cited in the text, so they must live in a saved log like every other
reported value rather than only in an interactive session.
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))

EICU = "data/raw/eicu_demo/eicu-collaborative-research-database-demo-2.0.1"


def urine_charting():
    io = pd.read_csv(f"{EICU}/intakeOutput.csv.gz",
                     usecols=["patientunitstayid", "intakeoutputoffset",
                              "celllabel", "cellvaluenumeric"])
    ur = io[io.celllabel.str.contains("urine", case=False, na=False)
            & io.cellvaluenumeric.between(0, 5000)]
    ur = ur[ur.intakeoutputoffset >= 0].copy()
    gaps = ur.sort_values("intakeoutputoffset").groupby(
        "patientunitstayid").intakeoutputoffset.diff().dropna()
    ur["bin"] = (ur.intakeoutputoffset // 360).astype(int)
    per = ur.groupby(["patientunitstayid", "bin"]).size()
    w = pd.read_csv(f"{EICU}/patient.csv.gz",
                    usecols=["patientunitstayid", "admissionweight"])
    w = w[w.admissionweight.between(20, 300)]
    s = (ur.groupby(["patientunitstayid", "bin"]).cellvaluenumeric.sum()
           .reset_index().merge(w, on="patientunitstayid"))
    s["rate"] = s.cellvaluenumeric / s.admissionweight / 6
    s2 = s.merge(per.rename("n"), on=["patientunitstayid", "bin"])
    return {"median_gap_min": float(gaps.median()),
            "frac_bins_single_entry": float((per == 1).mean()),
            "pct_bins_single_entry": round(100 * float((per == 1).mean()), 1),
            "pct_oliguric_all_bins": round(100 * float((s.rate < 0.5).mean()), 1),
            "pct_oliguric_bins_ge2_entries":
                round(100 * float((s2[s2.n >= 2].rate < 0.5).mean()), 1)}


def kernel_concentration():
    from quantum import quantum_kernel_matrix
    rng = np.random.default_rng(0)
    rows = []
    for nq in (4, 6, 8, 10, 12):
        X = rng.normal(size=(50, nq))
        K = quantum_kernel_matrix(X, X, nq, scale=1.0)
        off = K[~np.eye(50, dtype=bool)]
        rows.append({"n_qubits": nq, "offdiag_mean": float(off.mean()),
                     "offdiag_std": float(off.std())})
    X = rng.normal(size=(60, 8))
    bw = []
    for g in (0.15, 0.3, 0.6, 1.0, 1.5, float(np.pi)):
        K = quantum_kernel_matrix(X, X, 8, scale=g)
        off = K[~np.eye(60, dtype=bool)]
        bw.append({"bandwidth": g, "offdiag_mean": float(off.mean()),
                   "offdiag_std": float(off.std())})
    return rows, bw


def main():
    out = {"urine_charting": urine_charting()}
    qs, bw = kernel_concentration()
    pd.DataFrame(qs).to_csv("results/logs/kernel_concentration_qubits.csv", index=False)
    pd.DataFrame(bw).to_csv("results/logs/kernel_concentration_bandwidth.csv", index=False)
    out["kernel_concentration_qubits"] = qs
    ev = pd.read_csv("results/logs/external_validation.csv")
    out["external_auroc_range"] = {"min": float(ev.auroc.min()),
                                   "max": float(ev.auroc.max()),
                                   "min_rounded_2dp": round(float(ev.auroc.min()), 2),
                                   "max_rounded_2dp": round(float(ev.auroc.max()), 2)}
    json.dump(out, open("results/logs/diagnostics.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
