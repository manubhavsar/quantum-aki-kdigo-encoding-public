"""Cost of the tabular transformer on the same fold and protocol as complexity.py
(median of three runs), appended to results/logs/complexity.csv."""
import os, sys, time, warnings
import numpy as np, pandas as pd, torch
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
import splits as S
import tab_transformer as tabt

H, SEED = 24, 20260826
df = S.load(H); y = df[f"y_{H}h"].values.astype(int)
tr, te = next(GroupKFold(n_splits=5).split(df, y, df.patient.values))
_, i_va = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
               .split(df.iloc[tr], y[tr], df.iloc[tr].patient.values))
va = tr[i_va]
cols = S.BLOCKS["delta"]; med = df.iloc[tr][cols].median()
X = df[cols].fillna(med).values.astype(float); mu, sd = X[tr].mean(0), X[tr].std(0); sd[sd == 0] = 1
X = (X - mu) / sd
fits, preds = [], []
for _ in range(3):
    t0 = time.perf_counter(); m = tabt.fit(X[tr], y[tr], X[va], y[va]); fits.append(time.perf_counter() - t0)
    t0 = time.perf_counter()
    with torch.no_grad():
        torch.sigmoid(m(torch.tensor(X[te], dtype=torch.float32)))
    preds.append(time.perf_counter() - t0)
row = {"model": "Transformer (tabular)", "kind": "classical",
       "params": sum(p.numel() for p in m.parameters()),
       "fit_seconds": round(float(np.median(fits)), 2),
       "predict_seconds": round(float(np.median(preds)), 3), "note": ""}
c = pd.read_csv("results/logs/complexity.csv")
c = pd.concat([c[c.model != row["model"]], pd.DataFrame([row])], ignore_index=True)
c.to_csv("results/logs/complexity.csv", index=False); print(row)
