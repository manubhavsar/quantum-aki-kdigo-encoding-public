"""Rich feature extraction for the 'best achievable' arm.

The matched 8-feature blocks used for the 2x2 factorial deliberately stay small,
so that the static-versus-delta contrast is controlled and the width matches a
tractable qubit count. This module adds a much wider feature set to establish
what absolute discrimination is achievable on this cohort, so the controlled
comparison is not mistaken for a performance ceiling.

Leakage control. Every feature is computed strictly from history in [0, t].
APACHE APS variables are deliberately EXCLUDED: they summarise the worst value
over the first 24 h of the stay, which would leak future information at a 12 h
prediction point, and the APACHE `dialysis` flag would leak the outcome itself.
Only time-stamped labs and vitals are used.
"""
import os, sys
import numpy as np, pandas as pd

EICU = "data/raw/eicu_demo/eicu-collaborative-research-database-demo-2.0.1"
H = 60.0

LABS = ["creatinine", "BUN", "bicarbonate", "sodium", "potassium", "chloride",
        "calcium", "glucose", "WBC x 1000", "Hgb", "Hct", "platelets x 1000",
        "albumin", "anion gap"]
LAB_RANGE = {"creatinine": (0.1, 20), "BUN": (1, 200), "bicarbonate": (5, 50),
             "sodium": (100, 180), "potassium": (1.5, 9), "chloride": (70, 140),
             "calcium": (4, 15), "glucose": (20, 1000), "WBC x 1000": (0.1, 100),
             "Hgb": (3, 22), "Hct": (10, 65), "platelets x 1000": (5, 1000),
             "albumin": (0.5, 6), "anion gap": (1, 40)}
VITALS_P = ["heartrate", "respiration", "sao2"]
VITALS_A = ["noninvasivemean", "noninvasivesystolic"]
VIT_RANGE = {"heartrate": (20, 250), "respiration": (4, 70), "sao2": (50, 100),
             "noninvasivemean": (30, 160), "noninvasivesystolic": (50, 250)}


def _series_map(df, key, tcol, vcol):
    """stay -> (sorted times, values)."""
    out = {}
    for k, g in df.groupby(key):
        t = g[tcol].values.astype(float)
        v = g[vcol].values.astype(float)
        m = np.isfinite(t) & np.isfinite(v)
        t, v = t[m], v[m]
        o = np.argsort(t)
        out[k] = (t[o], v[o])
    return out


def _agg(smap, sid, t, prefix, feats):
    """Last / mean / min / max over [0,t] plus delta and per-hour rate."""
    ts, vs = smap.get(sid, (None, None))
    if ts is None or len(ts) == 0:
        return
    hi = np.searchsorted(ts, t, side="right")
    lo = np.searchsorted(ts, 0.0, side="left")
    if hi - lo < 1:
        return
    tw, vw = ts[lo:hi], vs[lo:hi]
    feats[f"{prefix}_last"] = vw[-1]
    feats[f"{prefix}_mean"] = vw.mean()
    feats[f"{prefix}_min"] = vw.min()
    feats[f"{prefix}_max"] = vw.max()
    if len(vw) >= 2:
        feats[f"{prefix}_delta"] = vw[-1] - vw[0]
        dt = (tw[-1] - tw[0]) / H
        feats[f"{prefix}_rate"] = (vw[-1] - vw[0]) / dt if dt > 0 else 0.0
    else:
        feats[f"{prefix}_delta"] = 0.0
        feats[f"{prefix}_rate"] = 0.0


def build(samples_csv="data/processed/eicu_samples.csv",
          out_csv="data/processed/eicu_rich.csv"):
    df = pd.read_csv(samples_csv)
    stays = set(df.stay)

    lab = pd.read_csv(f"{EICU}/lab.csv.gz",
                      usecols=["patientunitstayid", "labresultoffset", "labname", "labresult"])
    lab = lab[lab.patientunitstayid.isin(stays)]
    lab_maps = {}
    for name in LABS:
        lo, hi = LAB_RANGE[name]
        sub = lab[(lab.labname == name) & lab.labresult.between(lo, hi)]
        lab_maps[name] = _series_map(sub, "patientunitstayid", "labresultoffset", "labresult")

    vp = pd.read_csv(f"{EICU}/vitalPeriodic.csv.gz",
                     usecols=["patientunitstayid", "observationoffset"] + VITALS_P)
    vp = vp[vp.patientunitstayid.isin(stays)]
    va = pd.read_csv(f"{EICU}/vitalAperiodic.csv.gz",
                     usecols=["patientunitstayid", "observationoffset"] + VITALS_A)
    va = va[va.patientunitstayid.isin(stays)]
    vit_maps = {}
    for name in VITALS_P + VITALS_A:
        src = vp if name in VITALS_P else va
        lo, hi = VIT_RANGE[name]
        sub = src[src[name].between(lo, hi)][["patientunitstayid", "observationoffset", name]]
        vit_maps[name] = _series_map(sub, "patientunitstayid", "observationoffset", name)

    rows = []
    for r in df.itertuples(index=False):
        t = r.t_hours * H
        f = {"stay": r.stay, "t_hours": r.t_hours}
        for name in LABS:
            _agg(lab_maps[name], r.stay, t, name.replace(" ", "_").replace(".", ""), f)
        for name in VITALS_P + VITALS_A:
            _agg(vit_maps[name], r.stay, t, name, f)
        # a clinically meaningful ratio, computed only where both are present
        if "BUN_last" in f and "creatinine_last" in f and f["creatinine_last"] > 0:
            f["bun_cr_ratio"] = f["BUN_last"] / f["creatinine_last"]
        rows.append(f)

    rich = pd.DataFrame(rows)
    rich.to_csv(out_csv, index=False)
    cols = [c for c in rich.columns if c not in ("stay", "t_hours")]
    print(f"rich features: {rich.shape[0]} rows x {len(cols)} features")
    miss = rich[cols].isna().mean().sort_values(ascending=False)
    print(f"median missingness {miss.median()*100:.1f}%, "
          f"features >40% missing: {(miss > 0.4).sum()}")
    print(miss.head(6).round(3).to_string())
    return rich


if __name__ == "__main__":
    build()
