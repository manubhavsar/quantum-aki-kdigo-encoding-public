"""KDIGO 2012 acute kidney injury labelling.

Implements the three KDIGO stage->=1 criteria explicitly, because the
rate-of-change nature of this definition is what motivates the temporal quantum
encoding studied in this work:

  (C1) serum creatinine rise >= 0.3 mg/dL within a 48 h window
  (C2) serum creatinine >= 1.5 x baseline, where baseline is the minimum value
       in the preceding 7 days
  (C3) urine output < 0.5 mL/kg/h sustained for >= 6 h

Times are in minutes relative to ICU (unit) admission throughout.
"""
import numpy as np
import pandas as pd

H = 60.0                      # minutes per hour
W48, W7D = 48 * H, 7 * 24 * H
DELTA_ABS = 0.3               # mg/dL, C1
RATIO = 1.5                   # x baseline, C2
UO_THRESH = 0.5               # mL/kg/h, C3
UO_HOURS = 6.0                # sustained duration, C3


def creatinine_aki_onset(times, values):
    """Earliest AKI onset time from creatinine criteria C1/C2.

    times: array of minutes, values: array of mg/dL, both sorted by time.
    Returns (onset_time_minutes, criterion) or (None, None).
    """
    t = np.asarray(times, dtype=float)
    v = np.asarray(values, dtype=float)
    if len(t) == 0:
        return None, None
    order = np.argsort(t)
    t, v = t[order], v[order]

    for i in range(len(t)):
        # C1: absolute rise vs the minimum of the trailing 48 h
        m48 = (t >= t[i] - W48) & (t <= t[i])
        if m48.any() and (v[i] - v[m48].min()) >= DELTA_ABS:
            return float(t[i]), "C1_delta_0.3_48h"
        # C2: relative rise vs the minimum of the trailing 7 d
        m7 = (t >= t[i] - W7D) & (t <= t[i])
        if m7.any():
            base = v[m7].min()
            if base > 0 and v[i] >= RATIO * base:
                return float(t[i]), "C2_ratio_1.5_7d"
    return None, None


def urine_aki_onset(uo_times, uo_volumes, weight_kg):
    """Earliest AKI onset from the urine-output criterion C3.

    Charted urine volumes are irregular, so volumes are accumulated into
    consecutive non-overlapping 6 h bins anchored at unit admission and the
    mean rate over each bin is compared with 0.5 mL/kg/h. A bin is only
    evaluated when it actually contains charted output, so absent charting is
    treated as missing rather than as oliguria.
    """
    if weight_kg is None or not np.isfinite(weight_kg) or weight_kg <= 0:
        return None, None
    t = np.asarray(uo_times, dtype=float)
    v = np.asarray(uo_volumes, dtype=float)
    keep = np.isfinite(t) & np.isfinite(v) & (t >= 0) & (v >= 0)
    t, v = t[keep], v[keep]
    if len(t) == 0:
        return None, None

    span = UO_HOURS * H
    end = t.max()
    edge = 0.0
    while edge + span <= end + 1e-9:
        m = (t >= edge) & (t < edge + span)
        if m.any():
            rate = v[m].sum() / weight_kg / UO_HOURS      # mL/kg/h
            if rate < UO_THRESH:
                return float(edge + span), "C3_uo_lt_0.5_6h"
        edge += span
    return None, None


def aki_onset(cr_times, cr_values, uo_times, uo_volumes, weight_kg,
              use_urine=True):
    """Combined KDIGO onset: earliest of the creatinine and urine criteria."""
    c_t, c_c = creatinine_aki_onset(cr_times, cr_values)
    if not use_urine:
        return c_t, c_c
    u_t, u_c = urine_aki_onset(uo_times, uo_volumes, weight_kg)
    cands = [(x, c) for x, c in ((c_t, c_c), (u_t, u_c)) if x is not None]
    if not cands:
        return None, None
    return min(cands, key=lambda z: z[0])


def kdigo_stage(times, values):
    """Peak KDIGO stage from creatinine, for cohort description only."""
    t = np.asarray(times, float); v = np.asarray(values, float)
    if len(t) == 0:
        return 0
    o = np.argsort(t); t, v = t[o], v[o]
    stage = 0
    for i in range(len(t)):
        m7 = (t >= t[i] - W7D) & (t <= t[i])
        m48 = (t >= t[i] - W48) & (t <= t[i])
        base = v[m7].min() if m7.any() else np.nan
        s = 0
        if v[i] >= 4.0:
            s = 3
        elif np.isfinite(base) and base > 0:
            r = v[i] / base
            if r >= 3.0:
                s = 3
            elif r >= 2.0:
                s = 2
            elif r >= 1.5:
                s = 1
        if s == 0 and m48.any() and (v[i] - v[m48].min()) >= DELTA_ABS:
            s = 1
        stage = max(stage, s)
    return stage
