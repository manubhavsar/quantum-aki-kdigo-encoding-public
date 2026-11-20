"""Patient-level splitting and feature matrix assembly.

All prediction points and unit stays belonging to one patient fall in a single
split, so no patient contributes to both training and evaluation.
"""
import numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit

DEMOG = ["age", "sex_male", "weight_kg"]
STATIC_TV = ["cr_last", "cr_mean", "cr_max", "bun_last", "bun_mean"]
DELTA_TV = ["cr_delta_48h", "cr_ratio_7d", "cr_rate_per_h", "bun_delta", "bun_rate_per_h"]
BLOCKS = {"static": DEMOG + STATIC_TV,
          "delta": DEMOG + DELTA_TV,
          "combined": DEMOG + STATIC_TV + DELTA_TV}

SEED = 20260826


def load(horizon, path="data/processed/eicu_samples.csv"):
    df = pd.read_csv(path)
    y = f"y_{horizon}h"
    df = df.dropna(subset=[y]).reset_index(drop=True)
    df[y] = df[y].astype(int)
    return df


def split(df, horizon, test_size=0.2, val_size=0.2, seed=SEED):
    """Group split by patient: train / val / test."""
    y = f"y_{horizon}h"
    g = df.patient.values
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr_idx, te_idx = next(gss.split(df, df[y], groups=g))
    tr = df.iloc[tr_idx].reset_index(drop=True)
    te = df.iloc[te_idx].reset_index(drop=True)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    tr2_idx, va_idx = next(gss2.split(tr, tr[y], groups=tr.patient.values))
    return (tr.iloc[tr2_idx].reset_index(drop=True),
            tr.iloc[va_idx].reset_index(drop=True), te)


def matrices(tr, va, te, block, horizon, scale=True):
    """Median-impute (train-fit) and standardise (train-fit). Returns arrays + names."""
    cols = BLOCKS[block]
    y = f"y_{horizon}h"
    med = tr[cols].median()
    Xtr, Xva, Xte = (d[cols].fillna(med).values.astype(float) for d in (tr, va, te))
    if scale:
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd[sd == 0] = 1.0
        Xtr, Xva, Xte = ((X - mu) / sd for X in (Xtr, Xva, Xte))
    return (Xtr, tr[y].values, Xva, va[y].values, Xte, te[y].values, cols)


def check_no_leakage(tr, va, te):
    a, b, c = set(tr.patient), set(va.patient), set(te.patient)
    assert not (a & b) and not (a & c) and not (b & c), "patient overlap between splits!"
    return True
