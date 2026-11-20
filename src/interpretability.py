"""Quantum feature attribution by the parameter-shift rule, compared with SHAP.

For a trained VQC the encoded angle of feature i enters through a single R_Y
rotation, so the exact derivative of the expectation value with respect to that
angle is given by the parameter-shift rule

    d<Z>/d theta_i = [ f(theta_i + pi/2) - f(theta_i - pi/2) ] / 2,

evaluated per sample and averaged in absolute value over the dataset. This is an
exact analytic gradient, not a finite-difference approximation
(Schuld et al., 2019).
"""
import os, sys, json
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(__file__))
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

import splits as S
from quantum import VQC

SEED = 20260826
HOR = 24
BLOCK = "delta"


def parameter_shift_importance(model, X):
    """Mean |d<Z>/d theta_i| over samples, via the exact parameter-shift rule."""
    Xa = np.tanh(np.asarray(X, float)) * model.scale     # encoded angles
    n_feat = Xa.shape[1]
    imp = np.zeros(n_feat)
    with torch.no_grad():
        for i in range(n_feat):
            plus, minus = Xa.copy(), Xa.copy()
            plus[:, i] += np.pi / 2
            minus[:, i] -= np.pi / 2
            fp = model.circuit(torch.tensor(plus, dtype=torch.float64), model.w)
            fm = model.circuit(torch.tensor(minus, dtype=torch.float64), model.w)
            imp[i] = float(torch.mean(torch.abs((fp - fm) / 2.0)))
    return imp


def main():
    df = S.load(HOR)
    cols = S.BLOCKS[BLOCK]
    y = df[f"y_{HOR}h"].values.astype(int)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    tr, te = next(gss.split(df, y, df.patient.values))
    med = df.iloc[tr][cols].median()
    X = df[cols].fillna(med).values.astype(float)
    mu, sd = X[tr].mean(0), X[tr].std(0); sd[sd == 0] = 1
    X = (X - mu) / sd

    q = VQC(len(cols), n_layers=3, epochs=60, lr=0.1, scale=0.6, seed=SEED).fit(X[tr], y[tr])
    qi = parameter_shift_importance(q, X[te])
    qi_n = qi / qi.sum()

    xgb = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                        random_state=SEED, n_jobs=-1).fit(X[tr], y[tr])
    try:
        import shap
        sv = shap.TreeExplainer(xgb).shap_values(X[te])
        si = np.abs(sv).mean(0)
    except Exception as e:
        print("SHAP unavailable, falling back to gain importance:", str(e)[:60])
        si = xgb.feature_importances_
    si_n = si / si.sum()

    out = pd.DataFrame({"feature": cols, "param_shift": qi_n, "shap": si_n})
    out["rank_q"] = out.param_shift.rank(ascending=False).astype(int)
    out["rank_c"] = out.shap.rank(ascending=False).astype(int)
    out = out.sort_values("param_shift", ascending=False)
    out.to_csv("results/logs/interpretability.csv", index=False)
    print(out.to_string(index=False))
    rho = out.param_shift.corr(out.shap, method="spearman")
    print(f"\nSpearman rank correlation (parameter-shift vs SHAP): {rho:.3f}")
    json.dump({"spearman": float(rho)}, open("results/logs/interpretability_summary.json", "w"))


if __name__ == "__main__":
    main()
