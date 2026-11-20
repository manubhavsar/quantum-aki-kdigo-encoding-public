"""Cohort construction, KDIGO labelling and feature extraction for AKI prediction.

Primary cohort  : eICU Collaborative Research Database Demo v2.0.1 (multi-centre)
External cohort : MIMIC-IV Clinical Database Demo v2.2 (independent hospital system)

Design
------
Rather than a single observation window anchored at admission (which leaves most
stays with <2 creatinine values and therefore a degenerate rate-of-change
feature), prediction points are placed every PRED_EVERY_H hours along each stay,
following the continuous-prediction setup standard in the AKI literature. At
each point t the model sees only history in [0, t] and predicts incident KDIGO
stage>=1 AKI in (t, t+H].

Leakage control: splits are made on *patient* identifier, so all prediction
points and all unit stays belonging to one patient fall in a single split.
"""
import os, json, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from kdigo import aki_onset, kdigo_stage, creatinine_aki_onset, urine_aki_onset

H = 60.0
PRED_EVERY_H = 12                       # spacing of prediction points (hours)
MAX_T_H = 96                            # last prediction point (hours)
HORIZONS = [12, 24, 48]                 # prediction horizons (hours)
MIN_CR_IN_HISTORY = 2                   # needed for a rate-of-change feature

EICU = "data/raw/eicu_demo/eicu-collaborative-research-database-demo-2.0.1"
MIMIC = "data/raw/mimic_iv_demo/mimic-iv-clinical-database-demo-2.2"

DEMOG = ["age", "sex_male", "weight_kg"]
STATIC_TV = ["cr_last", "cr_mean", "cr_max", "bun_last", "bun_mean"]
DELTA_TV = ["cr_delta_48h", "cr_ratio_7d", "cr_rate_per_h", "bun_delta", "bun_rate_per_h"]
FEATURES_STATIC = DEMOG + STATIC_TV
FEATURES_DELTA = DEMOG + DELTA_TV
FEATURES_ALL = DEMOG + STATIC_TV + DELTA_TV


# ----------------------------------------------------------------- eICU loading
def load_eicu():
    pt = pd.read_csv(f"{EICU}/patient.csv.gz")
    lab = pd.read_csv(f"{EICU}/lab.csv.gz",
                      usecols=["patientunitstayid", "labresultoffset", "labname", "labresult"])
    io = pd.read_csv(f"{EICU}/intakeOutput.csv.gz",
                     usecols=["patientunitstayid", "intakeoutputoffset",
                              "celllabel", "cellvaluenumeric"])

    age = pt.age.astype(str).str.strip()
    pt["age_num"] = pd.to_numeric(age.replace({"> 89": "90", "": np.nan}), errors="coerce")
    pt["sex_male"] = (pt.gender == "Male").astype(float)
    w = pt.admissionweight.copy()
    pt["weight_kg"] = w.where((w > 20) & (w < 300))       # drop implausible weights

    cr = lab[(lab.labname == "creatinine") & lab.labresult.between(0.1, 20)]
    bun = lab[(lab.labname == "BUN") & lab.labresult.between(1, 200)]
    ur = io[io.celllabel.str.contains("urine", case=False, na=False)
            & io.cellvaluenumeric.between(0, 5000)]

    stays = pt[["patientunitstayid", "uniquepid", "hospitalid", "age_num", "sex_male",
                "weight_kg", "unitdischargeoffset", "unittype",
                "unitdischargestatus"]].rename(columns={"age_num": "age"})
    return stays, cr, bun, ur


# ------------------------------------------------------------ MIMIC-IV loading
def load_mimic():
    icu = pd.read_csv(f"{MIMIC}/icu/icustays.csv.gz", parse_dates=["intime", "outtime"])
    pats = pd.read_csv(f"{MIMIC}/hosp/patients.csv.gz")
    lab = pd.read_csv(f"{MIMIC}/hosp/labevents.csv.gz",
                      usecols=["subject_id", "hadm_id", "itemid", "charttime", "valuenum"],
                      parse_dates=["charttime"])
    oe = pd.read_csv(f"{MIMIC}/icu/outputevents.csv.gz",
                     usecols=["stay_id", "charttime", "itemid", "value"],
                     parse_dates=["charttime"])
    ditems = pd.read_csv(f"{MIMIC}/icu/d_items.csv.gz")
    ce = pd.read_csv(f"{MIMIC}/icu/chartevents.csv.gz",
                     usecols=["stay_id", "charttime", "itemid", "valuenum"],
                     parse_dates=["charttime"])

    icu = icu.merge(pats[["subject_id", "gender", "anchor_age"]], on="subject_id", how="left")
    icu["sex_male"] = (icu.gender == "M").astype(float)
    icu["age"] = icu.anchor_age.astype(float)
    icu["unitdischargeoffset"] = (icu.outtime - icu.intime).dt.total_seconds() / 60.0

    # admission weight from chartevents (kg item ids)
    wt_ids = ditems[ditems.label.str.contains("admission weight|daily weight",
                                              case=False, na=False)].itemid.tolist()
    wt = ce[ce.itemid.isin(wt_ids) & ce.valuenum.between(20, 300)]
    wt = wt.sort_values("charttime").groupby("stay_id").valuenum.first()
    icu["weight_kg"] = icu.stay_id.map(wt)

    lab = lab.merge(icu[["subject_id", "hadm_id", "stay_id", "intime"]],
                    on=["subject_id", "hadm_id"], how="inner")
    lab["offset"] = (lab.charttime - lab.intime).dt.total_seconds() / 60.0
    cr = lab[(lab.itemid == 50912) & lab.valuenum.between(0.1, 20)]
    bun = lab[(lab.itemid == 51006) & lab.valuenum.between(1, 200)]

    urine_ids = ditems[ditems.label.str.contains("urine|foley", case=False, na=False)
                       & (ditems.linksto == "outputevents")].itemid.tolist()
    ur = oe[oe.itemid.isin(urine_ids) & oe.value.between(0, 5000)]
    ur = ur.merge(icu[["stay_id", "intime"]], on="stay_id", how="inner")
    ur["offset"] = (ur.charttime - ur.intime).dt.total_seconds() / 60.0

    stays = icu[["stay_id", "subject_id", "age", "sex_male", "weight_kg",
                 "unitdischargeoffset"]].rename(
        columns={"stay_id": "patientunitstayid", "subject_id": "uniquepid"})
    stays["hospitalid"] = -1
    cr = cr.rename(columns={"stay_id": "patientunitstayid", "offset": "labresultoffset",
                            "valuenum": "labresult"})
    bun = bun.rename(columns={"stay_id": "patientunitstayid", "offset": "labresultoffset",
                              "valuenum": "labresult"})
    ur = ur.rename(columns={"stay_id": "patientunitstayid", "offset": "intakeoutputoffset",
                            "value": "cellvaluenumeric"})
    return stays, cr, bun, ur


# ------------------------------------------------------------- feature builder
def _series(df, sid, tcol, vcol, cache):
    if sid not in cache:
        cache[sid] = (np.array([]), np.array([]))
    return cache[sid]


def build_cohort(stays, cr, bun, ur, cohort_name):
    crg = {k: (g[0].values, g[1].values) for k, g in
           cr.groupby("patientunitstayid")[["labresultoffset", "labresult"]]
           .apply(lambda d: (d.labresultoffset, d.labresult)).items()}
    bung = {k: (g[0].values, g[1].values) for k, g in
            bun.groupby("patientunitstayid")[["labresultoffset", "labresult"]]
            .apply(lambda d: (d.labresultoffset, d.labresult)).items()}
    urg = {k: (g[0].values, g[1].values) for k, g in
           ur.groupby("patientunitstayid")[["intakeoutputoffset", "cellvaluenumeric"]]
           .apply(lambda d: (d.intakeoutputoffset, d.cellvaluenumeric)).items()}

    rows = []
    stay_meta = []
    for r in stays.itertuples(index=False):
        sid = r.patientunitstayid
        ct, cv = crg.get(sid, (np.array([]), np.array([])))
        bt, bv = bung.get(sid, (np.array([]), np.array([])))
        ut, uv = urg.get(sid, (np.array([]), np.array([])))
        ct, cv = np.asarray(ct, float), np.asarray(cv, float)
        m = np.isfinite(ct) & np.isfinite(cv)
        ct, cv = ct[m], cv[m]
        o = np.argsort(ct); ct, cv = ct[o], cv[o]
        if len(ct) < MIN_CR_IN_HISTORY:
            continue

        # Primary label: creatinine-only KDIGO. Urine charting in these cohorts is
        # too sparse to support the 6 h oliguria criterion reliably (51% of 6 h bins
        # contain a single charted entry), so the urine-inclusive definition is kept
        # only as a sensitivity analysis.
        onset, crit = creatinine_aki_onset(ct, cv)
        onset_uo, crit_uo = aki_onset(ct, cv, ut, uv, r.weight_kg, use_urine=True)
        stay_meta.append({"stay": sid, "patient": r.uniquepid, "hospital": r.hospitalid,
                          "aki": onset is not None, "onset_min": onset, "criterion": crit,
                          "aki_uo": onset_uo is not None, "onset_uo_min": onset_uo,
                          "criterion_uo": crit_uo,
                          "peak_stage": kdigo_stage(ct, cv), "los_min": r.unitdischargeoffset,
                          "n_cr": len(ct)})

        los = r.unitdischargeoffset if np.isfinite(r.unitdischargeoffset) else np.inf
        for th in range(PRED_EVERY_H, MAX_T_H + 1, PRED_EVERY_H):
            t = th * H
            if t > los:
                break
            hist = ct <= t
            if hist.sum() < MIN_CR_IN_HISTORY:
                continue
            if onset is not None and onset <= t:
                break                          # AKI already present: stop this stay
            f = _features(ct[hist], cv[hist], bt, bv, ut, uv, t, r)
            lab_row = {"cohort": cohort_name, "stay": sid, "patient": r.uniquepid,
                       "hospital": r.hospitalid, "t_hours": th, **f}
            for Hh in HORIZONS:
                end = t + Hh * H
                lab_row[f"y_{Hh}h"] = _label(onset, t, end, los)
                lab_row[f"y_{Hh}h_uo"] = _label(onset_uo, t, end, los)
            rows.append(lab_row)
    return pd.DataFrame(rows), pd.DataFrame(stay_meta)


def _label(onset, t, end, los):
    """1 if onset falls in (t, end]; 0 if the patient is event-free and observed
    through end; NaN if censored by discharge before the horizon closes."""
    if onset is not None and t < onset <= end:
        return 1
    if onset is not None and onset > end:
        return 0
    if onset is None and los >= end:
        return 0
    return np.nan


def _win(t_arr, v_arr, lo, hi):
    t_arr = np.asarray(t_arr, float); v_arr = np.asarray(v_arr, float)
    m = np.isfinite(t_arr) & np.isfinite(v_arr) & (t_arr >= lo) & (t_arr <= hi)
    return t_arr[m], v_arr[m]


def _uo_rate(ut, uv, lo, hi, weight):
    t, v = _win(ut, uv, lo, hi)
    if len(t) == 0 or weight is None or not np.isfinite(weight) or weight <= 0:
        return np.nan
    hours = max((hi - lo) / H, 1e-6)
    return v.sum() / weight / hours


def _features(ct, cv, bt, bv, ut, uv, t, r):
    """Static and delta feature blocks computed from history in [0, t] only."""
    f = {"age": r.age, "sex_male": r.sex_male, "weight_kg": r.weight_kg}

    # --- static block: level-based summaries (mirrors static CKD-QML encodings)
    f["cr_last"] = cv[-1]
    f["cr_mean"] = cv.mean()
    f["cr_max"] = cv.max()
    bt_, bv_ = _win(bt, bv, -np.inf, t)
    bo = np.argsort(bt_)
    bt_, bv_ = bt_[bo], bv_[bo]
    f["bun_last"] = bv_[-1] if len(bv_) else np.nan
    f["bun_mean"] = bv_.mean() if len(bv_) else np.nan

    # --- delta block: KDIGO-aligned rate-of-change quantities
    m48 = ct >= t - 48 * H
    f["cr_delta_48h"] = cv[-1] - cv[m48].min() if m48.any() else 0.0   # KDIGO C1 quantity
    m7 = ct >= t - 7 * 24 * H
    base7 = cv[m7].min() if m7.any() else np.nan
    f["cr_ratio_7d"] = cv[-1] / base7 if base7 and base7 > 0 else np.nan  # KDIGO C2 quantity
    dt_h = (ct[-1] - ct[0]) / H
    f["cr_rate_per_h"] = (cv[-1] - cv[0]) / dt_h if dt_h > 0 else 0.0
    if len(bv_) >= 2:
        f["bun_delta"] = bv_[-1] - bv_[0]
        bdt = (bt_[-1] - bt_[0]) / H
        f["bun_rate_per_h"] = (bv_[-1] - bv_[0]) / bdt if bdt > 0 else 0.0
    else:
        f["bun_delta"] = 0.0
        f["bun_rate_per_h"] = 0.0
    # retained for the cohort description / sensitivity analysis only
    f["uo_rate_mean_aux"] = _uo_rate(ut, uv, 0, t, r.weight_kg)
    return f


def main():
    os.makedirs("data/processed", exist_ok=True)
    out = {}
    print("=== eICU (primary) ===")
    s, c, b, u = load_eicu()
    df_e, meta_e = build_cohort(s, c, b, u, "eicu")
    df_e.to_csv("data/processed/eicu_samples.csv", index=False)
    meta_e.to_csv("data/processed/eicu_stays.csv", index=False)
    out["eicu"] = _summary(df_e, meta_e)
    print(json.dumps(out["eicu"], indent=2, default=str))

    print("\n=== MIMIC-IV (external validation) ===")
    s, c, b, u = load_mimic()
    df_m, meta_m = build_cohort(s, c, b, u, "mimic")
    df_m.to_csv("data/processed/mimic_samples.csv", index=False)
    meta_m.to_csv("data/processed/mimic_stays.csv", index=False)
    out["mimic"] = _summary(df_m, meta_m)
    print(json.dumps(out["mimic"], indent=2, default=str))

    os.makedirs("results/logs", exist_ok=True)
    with open("results/logs/data_card.json", "w") as f:
        json.dump(out, f, indent=2, default=str)


def _summary(df, meta):
    s = {"n_stays_with_ge2_cr": int(len(meta)),
         "n_patients": int(meta.patient.nunique()) if len(meta) else 0,
         "n_hospitals": int(meta.hospital.nunique()) if len(meta) else 0,
         "stay_level_aki_rate": float(meta.aki.mean()) if len(meta) else 0.0,
         "criterion_counts": meta.criterion.value_counts().to_dict() if len(meta) else {},
         "peak_stage_counts": meta.peak_stage.value_counts().sort_index().to_dict() if len(meta) else {},
         "n_prediction_points": int(len(df))}
    if len(meta):
        s["stay_level_aki_rate_uo_sensitivity"] = float(meta.aki_uo.mean())
        s["criterion_counts_uo_sensitivity"] = meta.criterion_uo.value_counts().to_dict()
    for Hh in HORIZONS:
        for suf in ("", "_uo"):
            col = f"y_{Hh}h{suf}"
            if col in df:
                v = df[col].dropna()
                s[f"h{Hh}{suf}_n"] = int(len(v))
                s[f"h{Hh}{suf}_pos"] = int(v.sum())
                s[f"h{Hh}{suf}_pos_rate"] = round(float(v.mean()), 4) if len(v) else 0.0
    return s


if __name__ == "__main__":
    main()
