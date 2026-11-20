# Separating temporal representation from quantum circuit contributions in AKI prediction

Code, experiment logs and pooled out-of-fold predictions for the paper *Separating Temporal
Representation from Quantum Circuit Contributions in Acute Kidney Injury Prediction*.
Every number reported in the paper is regenerated from the logs in `results/logs/`.

Acute kidney injury is defined by a **rate of change**, not a level: the KDIGO
guideline calls it AKI when serum creatinine rises ≥0.3 mg/dL within 48 hours, or
reaches 1.5× a 7-day baseline. Every existing quantum machine-learning study of
kidney disease targets the *chronic* condition from a static snapshot.

That makes AKI an unusually clean testbed for a question applied QML rarely gets to
ask cleanly: **once clinically aligned temporal representation is controlled for,
does quantum circuit structure contribute anything measurable — and is whatever
remains larger than the ordinary benchmarking choices that surround it?**

## Three hypotheses

- **H1** — clinically aligned temporal representation improves AKI prediction
  across model families.
- **H2** — after that representation is controlled for, a quantum circuit component
  provides an additional measurable contribution.
- **H3** — that contribution is smaller than several ordinary benchmarking choices.

Rate-of-change features improve *any* model, so "quantum model + delta features
beats classical model + static features" would prove nothing. The design is built
around removing that confound.

## The design

- **A 2×2 factorial** over {classical, quantum} × {static, delta}, where the
  **interaction** is the result. Both feature blocks derive from the same two lab
  series and differ *only* in level vs. rate-of-change.
- **Three controls that isolate the quantum component.** Removing the interleaved
  entanglers from the Trajectory Re-uploading Encoding (TRE) leaves per-qubit angles
  **additive in the timepoints** (the output is `g(Σⱼ fⱼ(cⱼ))` for a fixed readout `g`,
  not `Σⱼ fⱼ(cⱼ)`), removing every entangling operation between timepoints; the
  resulting exchange symmetry is verified numerically in `src/verify_tre_collapse.py`. The Minimal-QLSTM's quantum
  gate is swapped for a classical one with **more** parameters. An MLP control is
  matched on inputs, optimiser, folds and training protocol, with slightly more
  trainable parameters than the VQC (61 vs 49).
- **Patient-clustered inference.** Prediction points repeat within patients, so
  intervals resample patients and comparisons use a paired permutation test
  exchanging per-patient predictions. DeLong is anti-conservative here and is
  reported only alongside.
- **Incident AKI only.** Prediction points stop once a patient meets KDIGO criteria.

## What we found

**H1 holds.** The delta block raises AUROC in **18/18** family-horizon cells across
six model families.

**H2 holds, at a modest size.** All 9/9 variational-classifier interaction estimates
are positive; the largest, **+0.104** at 24 h, survives Bonferroni correction over
18 tests. Both quantum components beat their own controls at 24 h — TRE interleaving
**+0.052**, quantum gate **+0.042** — and the VQC-vs-MLP interaction is **+0.117**.
A fidelity quantum kernel evaluated identically is positive in only 5/9, so the two
quantum paradigms do not support the same conclusion.

**The falsification test bites.** Give the classical arm explicit trajectory
curvature — second differences, acceleration, direction changes, dispersion — and it
reaches **0.721 at 24 h, above every quantum configuration measured.** The structure
at issue is largely classically computable, so the contribution reflects
representational capacity within the tested model family rather than an irreducibly
quantum representation of AKI trajectories.

**H3 holds.** Of 8 audited evaluation choices, **2 shift the quantum–classical gap
by more than the largest contribution established under control** (+0.065), the
larger by a factor of 1.7.

## The evaluation audit

Each choice is varied in isolation on one cohort, separated by *which estimand it
distorts* — a distinction that turns out to matter:

| choice | shifts absolute AUROC | shifts the gap *G* |
|---|---|---|
| post-onset windows retained (leakage) | **+0.193** | +0.023 |
| unmatched classical architecture | — | **+0.109** |
| unreported encoding bandwidth | — | +0.101 |

Leakage is the largest effect on any headline number and nearly the smallest on the
quantity an advantage claim actually asserts, because both arms benefit. **An audit
must state which quantity it protects.**

A sliding-window AKI pipeline that does not exclude windows after onset is no longer
doing early prediction — it is asking "is this patient currently in AKI":

| labelling | prevalence | windows after onset | AUROC |
|---|---|---|---|
| incident only (used here) | 16.5% | 0% | **0.681** |
| prevalent windows included | 49.2% | 39.2% | **0.880** |

A **+0.199** swing on this reference model; the audit table above reports **+0.193**
because it measures the same choice on the audited VQC arm (0.629→0.822) rather than
on this one. Replicated on MIMIC-IV (+0.160). Reported AKI AUROCs above the
published 0.75–0.90 range are worth checking against this.

## Cohorts

Both are open-access PhysioNet demo releases, no credentialing required — the whole
study is reproducible by anyone.

| | role | scale |
|---|---|---|
| [eICU-CRD Demo v2.0.1](https://physionet.org/content/eicu-crd-demo/2.0.1/) | primary | 1,609 patients, 2,058 stays, 4,955 prediction points |
| [MIMIC-IV Demo v2.2](https://physionet.org/content/mimic-iv-demo/2.2/) | transportability check | independent hospital system |

The eICU files carry 186 distinct `hospitalid` values, of which **182 contribute at
least one eligible stay**. These do not correspond one-to-one with the 20 source
hospitals named in the release note, so they are reported as *contributing site
identifiers*, not hospitals.

Download and unpack to `data/raw/eicu_demo/` and `data/raw/mimic_iv_demo/`.
Raw data is not redistributed here.

## Models

| | static features | delta features | trajectory |
|---|---|---|---|
| **classical** | LR · RF · XGBoost · MLP | LR · RF · XGBoost · MLP | LSTM |
| **quantum** | VQC · quantum-kernel SVM | VQC · quantum-kernel SVM | TRE · Minimal-QLSTM |

All share patient-grouped 5-fold CV with hyperparameters tuned inside each training
fold, and pooled out-of-fold predictions so every model is scored on identical
samples. Circuits are simulated with PennyLane; **no quantum hardware was used**, so
results are an upper bound for these circuits on current devices.

## Running it

```bash
pip install -r requirements.txt
```

```bash
python3 src/data_pipeline.py && python3 src/trajectories.py
```

Then the model runs — quantum steps dominate the runtime, everything classical
finishes in minutes:

```bash
python3 src/classical.py; python3 src/run_quantum.py; python3 src/run_temporal.py; python3 src/run_qlstm.py; python3 src/run_gate_ablation.py
```

Controls, ablations and the audit:

```bash
python3 src/mlp_control.py; python3 src/capacity_scaling.py; python3 src/curvature_baseline.py; python3 src/order_sensitivity.py; python3 src/verify_tre_collapse.py; python3 src/ablations.py; python3 src/learning_curve.py; python3 src/noise_robustness.py; python3 src/leakage_probe.py; python3 src/artifact_taxonomy.py; python3 src/external_validation.py
```

Analysis, tables and figures:

```bash
python3 src/clustered_inference.py && python3 src/analysis.py && python3 src/figures.py
```

## Repository layout

```
src/              pipeline, models, evaluation, verification tooling
results/logs/     every experiment log and citation-verification record
results/preds/    pooled out-of-fold predictions (enable paired re-tests)
results/figures/  figures used in the manuscript
```

## Reproducibility guarantees

Enforced by code, not convention:

- **No hand-typed results.** `src/make_tables.py` emits every number in the paper as
  a LaTeX macro read from the result CSVs; `src/audit_numbers.py` then checks every
  decimal in the prose against a saved log.
- **No unverified citations.** The bibliography is *generated* from resolver
  records; a reference that did not resolve against Crossref / Semantic Scholar /
  arXiv cannot appear. `src/final_citation_check.py` re-resolves them all live.
- **No patient leakage.** Splits are grouped on patient identity and asserted in
  `src/splits.py:check_no_leakage`.
- **Simulator only.** No quantum hardware was used.
