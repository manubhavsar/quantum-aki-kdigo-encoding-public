"""Matched quantum-classical pairs repeated over five seeds at 24 h (optional
reviewer request: extend seed repetition beyond the TRE ablation).

Each model is refitted with five initialisation seeds on the same patient-grouped
folds. To keep five seeds of every model tractable on a simulator, every model uses
one fixed, untuned configuration (bandwidth or scale 1.0; VQC two layers; kernel SVM
C = 1 on at most 700 training rows per fold). Absolute AUROCs therefore differ from
the tuned values in the main tables; the quantity reported is each pair's AUROC
difference, mean and SD over seeds. Logistic regression has no random state and is
fitted once.
"""
import os, sys, time, warnings
import numpy as np, pandas as pd, torch
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC

import splits as S
from quantum import VQC, TRE, quantum_kernel_matrix
from mlp_control import fit_mlp, pick_hidden, vqc_param_count
import tab_transformer as tabt
from lstm_baseline import _fit as fit_lstm
import transformer_baseline as tb
from qlstm import MinimalQLSTM, ClassicalGateLSTM, fit_generic, predict as qpredict
from run_qlstm import prep_static, prep_seq
from run_temporal import _static_mat, _seq_norm, N_QUBITS

H, SEEDS, SPLIT_SEED = 24, [0, 1, 2, 3, 4], 20260826
PAIRS = [("VQC vs MLP control", "VQC", "MLP"), ("VQC vs logistic regression", "VQC", "LR"),
         ("VQC vs tabular transformer", "VQC", "TabT"),
         ("Kernel SVM vs logistic regression", "QSVM", "LR"),
         ("Kernel SVM vs tabular transformer", "QSVM", "TabT"),
         ("Minimal-QLSTM vs classical gate (abl.)", "QLSTM", "Gate"),
         ("Minimal-QLSTM vs transformer", "QLSTM", "SeqT"),
         ("TRE vs additive control (abl.)", "TRE", "TREadd"), ("TRE vs LSTM", "TRE", "LSTM"),
         ("TRE vs transformer", "TRE", "SeqT")]


def main():
    df = S.load(H)
    y = df[f"y_{H}h"].values.astype(int); g = df.patient.values
    cols = S.BLOCKS["delta"]
    full = pd.read_csv("data/processed/eicu_samples.csv")
    sel = full.dropna(subset=[f"y_{H}h"]).index.values
    seq, mask = np.load("data/processed/eicu_seq.npy")[sel], np.load("data/processed/eicu_mask.npy")[sel]
    folds = list(GroupKFold(n_splits=5).split(df, y, g))
    T32 = lambda a: torch.tensor(a, dtype=torch.float32)
    rows = []
    for s in SEEDS:
        t0 = time.time()
        oof = {k: np.full(len(df), np.nan) for k in
               ("VQC", "MLP", "LR", "TabT", "QSVM", "QLSTM", "Gate", "TRE", "TREadd", "LSTM", "SeqT")}
        rng = np.random.default_rng(s)
        for tr, te in folds:
            med = df.iloc[tr][cols].median()
            X = df[cols].fillna(med).values.astype(float)
            mu, sd = X[tr].mean(0), X[tr].std(0); sd[sd == 0] = 1; X = (X - mu) / sd
            tr_df = df.iloc[tr]
            a_i, b_i = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SPLIT_SEED)
                            .split(tr_df, y[tr], tr_df.patient.values))
            a, b = tr[a_i], tr[b_i]
            oof["VQC"][te] = VQC(len(cols), n_layers=2, epochs=60, lr=0.1, scale=1.0,
                                 seed=s).fit(X[tr], y[tr], X[b], y[b]).predict_proba(X[te])
            m = fit_mlp(X[tr], y[tr], X[b], y[b], pick_hidden(len(cols), vqc_param_count(len(cols))), seed=s)
            with torch.no_grad():
                oof["MLP"][te] = torch.sigmoid(m(torch.tensor(X[te], dtype=torch.float64))).squeeze(-1).numpy()
            oof["LR"][te] = LogisticRegression(max_iter=2000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
            m = tabt.fit(X[a], y[a], X[b], y[b], seed=s)
            with torch.no_grad():
                oof["TabT"][te] = torch.sigmoid(m(T32(X[te]))).numpy()
            pos, neg = np.where(y[tr] == 1)[0], np.where(y[tr] == 0)[0]
            sub = np.sort(np.concatenate([pos, rng.choice(neg, size=min(max(700 - len(pos), 50), len(neg)), replace=False)]))
            Xs, ys = X[tr][sub], y[tr][sub]
            svc = SVC(kernel="precomputed", C=1.0).fit(quantum_kernel_matrix(Xs, Xs, len(cols), scale=1.0), ys)
            oof["QSVM"][te] = svc.decision_function(quantum_kernel_matrix(X[te], Xs, len(cols), scale=1.0))
            Stq, Sqq = prep_static(df, tr), prep_seq(seq, mask, tr)
            for k, cls in (("QLSTM", MinimalQLSTM), ("Gate", ClassicalGateLSTM)):
                m = fit_generic(cls, Sqq[tr], Stq[tr], mask[tr], y[tr], Sqq[b], Stq[b], mask[b], y[b],
                                epochs=40, scale=1.0, seed=s)
                oof[k][te] = qpredict(m, Sqq[te], Stq[te], mask[te])
            St, Sq = _static_mat(df, tr), _seq_norm(seq, mask, tr)
            for k, ent in (("TRE", True), ("TREadd", False)):
                oof[k][te] = TRE(n_qubits=N_QUBITS, K=seq.shape[1], epochs=40, scale=1.0,
                                 entangle_between=ent, seed=s).fit(
                    Sq[tr], St[tr], y[tr], Sq[b], St[b], y[b]).predict_proba(Sq[te], St[te])
            DEM = ["age", "sex_male", "weight_kg"]
            dmed = df.iloc[tr][DEM].median()
            D = df[DEM].fillna(dmed).values.astype(float)
            dmu, dsd = D[tr].mean(0), D[tr].std(0); dsd[dsd == 0] = 1; D = (D - dmu) / dsd
            q = seq.copy(); qm = q[tr].reshape(-1, 2).mean(0); qs = q[tr].reshape(-1, 2).std(0); qs[qs == 0] = 1
            q = ((q - qm) / qs) * mask[..., None]
            for k, fitfn in (("LSTM", fit_lstm), ("SeqT", tb._fit)):
                m = fitfn(q[a], D[a], mask[a], y[a], q[b], D[b], mask[b], y[b], seed=s)
                with torch.no_grad():
                    oof[k][te] = torch.sigmoid(m(T32(q[te]), T32(D[te]), T32(mask[te]))).numpy()
        au = {k: roc_auc_score(y, v) for k, v in oof.items()}
        for label, qk, ck in PAIRS:
            rows.append({"seed": s, "pair": label, "quantum": au[qk], "classical": au[ck],
                         "diff": au[qk] - au[ck]})
        pd.DataFrame(rows).to_csv("results/logs/seed_pairs_24h.csv", index=False)
        print(f"seed {s} done in {time.time()-t0:.0f}s: " +
              ", ".join(f"{k}={v:.3f}" for k, v in au.items()), flush=True)
    d = pd.DataFrame(rows)
    print(d.groupby("pair")["diff"].agg(["mean", "std", "min", "max"]).round(3))


if __name__ == "__main__":
    main()
