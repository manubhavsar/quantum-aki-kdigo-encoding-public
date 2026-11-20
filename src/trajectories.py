"""Extract short creatinine/BUN trajectories for each prediction point.

Needed by two models:
  * the LSTM baseline, which consumes the raw sequence rather than engineered
    summary features;
  * the trajectory re-uploading quantum encoding (TRE), which interleaves one
    rotation per timepoint with entangling layers.

For each prediction point at time t we take the last K creatinine measurements in
[0, t], most-recent-last, together with their age relative to t in hours.
Sequences shorter than K are left-padded and a mask is returned.
"""
import numpy as np, pandas as pd, os, sys

K = 4
EICU = "data/raw/eicu_demo/eicu-collaborative-research-database-demo-2.0.1"
MIMIC = "data/raw/mimic_iv_demo/mimic-iv-clinical-database-demo-2.2"
H = 60.0


def _cr_map_eicu():
    lab = pd.read_csv(f"{EICU}/lab.csv.gz",
                      usecols=["patientunitstayid", "labresultoffset", "labname", "labresult"])
    cr = lab[(lab.labname == "creatinine") & lab.labresult.between(0.1, 20)]
    return {k: (g.labresultoffset.values, g.labresult.values)
            for k, g in cr.groupby("patientunitstayid")}


def _cr_map_mimic():
    import data_pipeline as DP
    stays, cr, bun, ur = DP.load_mimic()
    return {k: (g.labresultoffset.values, g.labresult.values)
            for k, g in cr.groupby("patientunitstayid")}


def build(samples_csv, cohort, out_prefix):
    df = pd.read_csv(samples_csv)
    cmap = _cr_map_eicu() if cohort == "eicu" else _cr_map_mimic()
    seq = np.zeros((len(df), K, 2), dtype=float)   # (value, hours_before_t)
    mask = np.zeros((len(df), K), dtype=float)
    for i, r in enumerate(df.itertuples(index=False)):
        t = r.t_hours * H
        ts, vs = cmap.get(r.stay, (np.array([]), np.array([])))
        m = (ts <= t) & np.isfinite(vs)
        ts_, vs_ = ts[m], vs[m]
        o = np.argsort(ts_); ts_, vs_ = ts_[o], vs_[o]
        take = min(K, len(ts_))
        if take == 0:
            continue
        seq[i, K - take:, 0] = vs_[-take:]
        seq[i, K - take:, 1] = (t - ts_[-take:]) / H
        mask[i, K - take:] = 1.0
    np.save(f"{out_prefix}_seq.npy", seq)
    np.save(f"{out_prefix}_mask.npy", mask)
    print(f"{cohort}: seq {seq.shape}, mean valid steps "
          f"{mask.sum(1).mean():.2f}/{K}, full-length rows "
          f"{(mask.sum(1) == K).mean():.1%}")
    return seq, mask


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(__file__))
    os.makedirs("data/processed", exist_ok=True)
    build("data/processed/eicu_samples.csv", "eicu", "data/processed/eicu")
    build("data/processed/mimic_samples.csv", "mimic", "data/processed/mimic")
