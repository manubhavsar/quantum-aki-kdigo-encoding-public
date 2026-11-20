"""Robustness to input missingness (hypothesis H2).

Models are trained on clean data and evaluated on progressively corrupted test
folds, which is the deployment-relevant direction: a model fitted on curated
retrospective data meets messier data in service. Missingness is simulated by
zeroing standardised feature entries, i.e. replacing them with the training-set
mean -- exactly what a mean-imputing pipeline does with an absent value.

The question is not which model is best at zero noise (that is the main result)
but whose curve is flattest: we report degradation relative to each model's own
clean score, so models with different baselines remain comparable.
"""
import os, sys, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

import splits as S
from quantum import VQC

SEED = 20260826
HOR = 24
BLOCK = "delta"
RATES = [0.0, 0.10, 0.20, 0.30, 0.40]
SEEDS = [0, 1, 2, 3, 4]


def corrupt(X, rate, rng):
    if rate <= 0:
        return X
    out = X.copy()
    out[rng.random(X.shape) < rate] = 0.0     # standardised zero == train mean
    return out


def main():
    df = S.load(HOR)
    cols = S.BLOCKS[BLOCK]
    y = df[f"y_{HOR}h"].values.astype(int)
    groups = df.patient.values
    rows = []

    # one clean training pass per fold; the fitted models are then probed at
    # every noise level, so noise affects evaluation only
    folds = list(GroupKFold(n_splits=5).split(df, y, groups))
    fitted = []
    for tr_i, te_i in folds:
        tr, te = df.iloc[tr_i], df.iloc[te_i]
        med = tr[cols].median()
        Xtr = tr[cols].fillna(med).values.astype(float)
        Xte = te[cols].fillna(med).values.astype(float)
        mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
        ytr = y[tr_i]
        pw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
        models = {
            "LogisticRegression": LogisticRegression(
                max_iter=3000, C=1.0, class_weight="balanced",
                random_state=SEED).fit(Xtr, ytr),
            "RandomForest": RandomForestClassifier(
                n_estimators=300, max_depth=5, min_samples_leaf=5,
                class_weight="balanced", random_state=SEED, n_jobs=-1).fit(Xtr, ytr),
            "XGBoost": XGBClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, scale_pos_weight=pw, eval_metric="logloss",
                random_state=SEED, n_jobs=-1).fit(Xtr, ytr),
            "VQC": VQC(len(cols), n_layers=2, epochs=40, lr=0.1, scale=1.0,
                       seed=SEED).fit(Xtr, ytr),
        }
        fitted.append((models, Xte, te_i))
        print(f"  fold fitted ({len(tr_i)} train)", flush=True)

    for rate in RATES:
        for s in SEEDS:
            rng = np.random.default_rng(SEED + s)
            oof = {k: np.full(len(df), np.nan) for k in
                   ("LogisticRegression", "RandomForest", "XGBoost", "VQC")}
            for models, Xte, te_i in fitted:
                Xc = corrupt(Xte, rate, rng)
                for name, m in models.items():
                    p = (m.predict_proba(Xc) if name == "VQC"
                         else m.predict_proba(Xc)[:, 1])
                    oof[name][te_i] = p
            for name, p in oof.items():
                ok = np.isfinite(p)
                rows.append({"rate": rate, "seed": s, "model": name,
                             "auroc": roc_auc_score(y[ok], p[ok])})
            if rate == 0.0:
                break                     # deterministic at zero noise
        pd.DataFrame(rows).to_csv("results/logs/noise_robustness.csv", index=False)
        d = pd.DataFrame(rows); d = d[d.rate == rate].groupby("model").auroc.mean()
        print(f"rate={rate:.2f}  " + "  ".join(f"{k}={v:.3f}" for k, v in d.items()),
              flush=True)

    d = pd.DataFrame(rows)
    piv = d.groupby(["rate", "model"]).auroc.mean().unstack()
    print("\n=== mean AUROC by missingness rate ===")
    print(piv.round(3).to_string())
    rel = piv - piv.loc[0.0]
    print("\n=== degradation from each model's own clean score ===")
    print(rel.round(3).to_string())
    rel.to_csv("results/logs/noise_degradation.csv")
    print("\nretention at 40% missingness (higher = more robust):")
    print((piv.loc[0.40] / piv.loc[0.0]).round(3).to_string())


if __name__ == "__main__":
    main()
