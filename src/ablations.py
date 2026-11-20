"""Ablations: qubit count, circuit depth, encoding bandwidth, and the TRE
entangling-interleave ablation.

Run on the 24 h horizon (middle horizon, adequate positive count). Qubit count is
varied by PCA-projecting the feature block to k dimensions, so the encoding width
changes while the information available to the model is controlled.
"""
import os, sys, time, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from sklearn.model_selection import GroupKFold
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score

import splits as S
from quantum import VQC

SEED = 20260826
HOR = 24


def prep(tr_df, te_df, cols, k=None):
    med = tr_df[cols].median()
    Xtr = tr_df[cols].fillna(med).values.astype(float)
    Xte = te_df[cols].fillna(med).values.astype(float)
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    if k is not None and k < Xtr.shape[1]:
        pca = PCA(n_components=k, random_state=SEED).fit(Xtr)
        Xtr, Xte = pca.transform(Xtr), pca.transform(Xte)
        s = Xtr.std(0); s[s == 0] = 1
        Xtr, Xte = Xtr / s, Xte / s
    return Xtr, Xte


def cv_vqc(df, block, k=None, n_layers=3, bw=0.6, epochs=40):
    cols = S.BLOCKS[block]
    y = df[f"y_{HOR}h"].values.astype(int)
    oof = np.full(len(df), np.nan)
    for tr, te in GroupKFold(n_splits=5).split(df, y, df.patient.values):
        Xtr, Xte = prep(df.iloc[tr], df.iloc[te], cols, k)
        m = VQC(Xtr.shape[1], n_layers=n_layers, epochs=epochs, lr=0.1,
                scale=bw, seed=SEED).fit(Xtr, y[tr])
        oof[te] = m.predict_proba(Xte)
    return roc_auc_score(y, oof), average_precision_score(y, oof)


def main():
    df = S.load(HOR)
    rows = []

    print("--- qubit count (PCA-projected delta block) ---", flush=True)
    for k in [2, 3, 4, 6, 8]:
        t = time.time(); a, p = cv_vqc(df, "delta", k=k)
        rows.append({"ablation": "n_qubits", "value": k, "block": "delta",
                     "auroc": a, "auprc": p})
        print(f"  qubits={k}: AUROC={a:.3f} AUPRC={p:.3f} ({time.time()-t:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv("results/logs/ablations.csv", index=False)

    print("--- circuit depth (delta block, 8 qubits) ---", flush=True)
    for L in [1, 2, 3, 4, 5]:
        t = time.time(); a, p = cv_vqc(df, "delta", n_layers=L)
        rows.append({"ablation": "n_layers", "value": L, "block": "delta",
                     "auroc": a, "auprc": p})
        print(f"  layers={L}: AUROC={a:.3f} AUPRC={p:.3f} ({time.time()-t:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv("results/logs/ablations.csv", index=False)

    print("--- encoding bandwidth ---", flush=True)
    for bw in [0.15, 0.3, 0.6, 1.0, 1.5, float(np.pi)]:
        t = time.time(); a, p = cv_vqc(df, "delta", bw=bw)
        rows.append({"ablation": "bandwidth", "value": bw, "block": "delta",
                     "auroc": a, "auprc": p})
        print(f"  bw={bw:.2f}: AUROC={a:.3f} AUPRC={p:.3f} ({time.time()-t:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv("results/logs/ablations.csv", index=False)

    print("--- encoding block (8 qubits, 3 layers) ---", flush=True)
    for blk in ["static", "delta", "combined"]:
        t = time.time(); a, p = cv_vqc(df, blk, k=8 if blk == "combined" else None)
        rows.append({"ablation": "block", "value": blk, "block": blk,
                     "auroc": a, "auprc": p})
        print(f"  block={blk}: AUROC={a:.3f} AUPRC={p:.3f} ({time.time()-t:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv("results/logs/ablations.csv", index=False)

    print("\nsaved results/logs/ablations.csv")


if __name__ == "__main__":
    main()
