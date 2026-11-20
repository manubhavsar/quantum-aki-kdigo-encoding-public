"""Urine-inclusive endpoint for the remaining tabular models (reviewer point R1.5).

Extends urine_sensitivity_vqc.py to random forest, XGBoost and the matched
perceptron on the delta block, under the same fixed, untuned protocol and folds.
"""
import os, sys, warnings
import numpy as np, pandas as pd, torch
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

import splits as S
from urine_sensitivity import build
from mlp_control import fit_mlp, pick_hidden, vqc_param_count

SEED = 20260826


def evaluate(df, y, cols):
    g = df.patient.values
    oof = {k: np.full(len(df), np.nan) for k in ("rf", "xgb", "mlp")}
    for tr, te in GroupKFold(n_splits=5).split(df, y, g):
        med = df.iloc[tr][cols].median()
        X = df[cols].fillna(med).fillna(0.0).values.astype(float)
        mu, sd = X[tr].mean(0), X[tr].std(0); sd[sd == 0] = 1
        X = (X - mu) / sd
        oof["rf"][te] = RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                            random_state=SEED, n_jobs=-1).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        oof["xgb"][te] = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                            verbosity=0, random_state=SEED).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        tr_df = df.iloc[tr]
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
        _, i_va = next(gss.split(tr_df, y[tr], tr_df.patient.values))
        b = tr[i_va]
        m = fit_mlp(X[tr], y[tr], X[b], y[b], pick_hidden(len(cols), vqc_param_count(len(cols))))
        with torch.no_grad():
            oof["mlp"][te] = torch.sigmoid(m(torch.tensor(X[te], dtype=torch.float64))).squeeze(-1).numpy()
    return {k: roc_auc_score(y, v) for k, v in oof.items()}


def main():
    rows = []
    for h in (12, 24, 48):
        base = S.load(h).reset_index(drop=True)
        uo = build(h).dropna(subset=["patient"]).reset_index(drop=True)
        for name, df, y in (("creatinine-only", base, base[f"y_{h}h"].values.astype(int)),
                            ("creatinine+urine", uo, uo._y.values.astype(int))):
            r = evaluate(df, y, S.BLOCKS["delta"])
            rows.append({"horizon": h, "endpoint": name, **r})
            print(f"h={h:2d} {name:16s} RF={r['rf']:.3f} XGB={r['xgb']:.3f} MLP={r['mlp']:.3f}", flush=True)
            pd.DataFrame(rows).to_csv("results/logs/urine_sensitivity_more.csv", index=False)


if __name__ == "__main__":
    main()
