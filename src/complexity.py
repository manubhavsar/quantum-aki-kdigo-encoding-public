"""Computational cost of each model family (reviewer request R2.2), median of three runs.

For one representative fold at the 24 h horizon this reports, per model:
  * trainable parameters, counted from the fitted model's own tensors
  * wall-clock fit time
  * wall-clock time to score the held-out fold
The quantum rows are *simulation* costs. Statevector simulation of an n-qubit
circuit needs O(2^n) memory and O(2^n) time per gate, whereas a classical network
costs time linear in its parameter count per example; a circuit that is cheap on
hardware can therefore be expensive here and vice versa. Hardware and library
versions are written alongside the results.
"""
import os, sys, time, platform, warnings
import numpy as np, pandas as pd, torch
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import splits as S

H = 24
SEED = 20260826
rows = []


def record(name, kind, params, fit_s, pred_s, note=""):
    rows.append({"model": name, "kind": kind, "params": params,
                 "fit_seconds": round(fit_s, 2), "predict_seconds": round(pred_s, 3),
                 "note": note})
    print(f"{name:24s} {kind:9s} params={params!s:>9} fit={fit_s:8.2f}s "
          f"predict={pred_s:7.3f}s {note}", flush=True)
    pd.DataFrame(rows).to_csv("results/logs/complexity.csv", index=False)


REPEATS = 3


def timed(fn):
    """Median wall-clock over REPEATS runs; the last fitted object is returned."""
    times, out = [], None
    for _ in range(REPEATS):
        t0 = time.perf_counter(); out = fn(); times.append(time.perf_counter() - t0)
    return out, float(np.median(times))


def nparams(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def main():
    df = S.load(H)
    y = df[f"y_{H}h"].values.astype(int)
    g = df.patient.values
    tr, te = next(GroupKFold(n_splits=5).split(df, y, g))
    tr_df = df.iloc[tr]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    _, i_va = next(gss.split(tr_df, y[tr], tr_df.patient.values))
    va = tr[i_va]

    cols = S.BLOCKS["delta"]
    med = df.iloc[tr][cols].median()
    X = df[cols].fillna(med).values.astype(float)
    mu, sd = X[tr].mean(0), X[tr].std(0); sd[sd == 0] = 1
    X = (X - mu) / sd

    # ---- tabular classical models
    m, f = timed(lambda: LogisticRegression(max_iter=2000).fit(X[tr], y[tr]))
    _, p = timed(lambda: m.predict_proba(X[te]))
    record("Logistic Regression", "classical", X.shape[1] + 1, f, p)

    from xgboost import XGBClassifier
    m, f = timed(lambda: XGBClassifier(n_estimators=200, max_depth=3, verbosity=0,
                                       random_state=SEED, n_jobs=1).fit(X[tr], y[tr]))
    _, p = timed(lambda: m.predict_proba(X[te]))
    record("XGBoost", "classical", "200 trees", f, p)

    from mlp_control import mlp_params, fit_mlp, pick_hidden, vqc_param_count
    hid = pick_hidden(len(cols), vqc_param_count(len(cols)))
    m, f = timed(lambda: fit_mlp(X[tr], y[tr], X[va], y[va], hid))
    Xt = torch.tensor(X[te], dtype=torch.float64)
    def _mlp_pred():
        with torch.no_grad():
            return torch.sigmoid(m(Xt))
    _, p = timed(_mlp_pred)
    record("MLP control", "classical", nparams(m), f, p)

    # ---- sequence inputs
    seq_all = np.load("data/processed/eicu_seq.npy")
    mask_all = np.load("data/processed/eicu_mask.npy")
    full = pd.read_csv("data/processed/eicu_samples.csv")
    sel = full.dropna(subset=[f"y_{H}h"]).index.values
    seq, mask = seq_all[sel], mask_all[sel]
    DEM = ["age", "sex_male", "weight_kg"]
    smed = df.iloc[tr][DEM].median()
    St = df[DEM].fillna(smed).values.astype(float)
    smu, ssd = St[tr].mean(0), St[tr].std(0); ssd[ssd == 0] = 1
    St = (St - smu) / ssd
    sq = seq.copy()
    qmu = sq[tr].reshape(-1, 2).mean(0); qsd = sq[tr].reshape(-1, 2).std(0); qsd[qsd == 0] = 1
    sq = ((sq - qmu) / qsd) * mask[..., None]
    T32 = lambda a: torch.tensor(a, dtype=torch.float32)

    from lstm_baseline import _fit as fit_lstm
    import transformer_baseline as tb
    from gru_baseline import GRUNet

    def torch_seq(name, fit_fn):
        m, f = timed(lambda: fit_fn(sq[tr], St[tr], mask[tr], y[tr],
                                    sq[va], St[va], mask[va], y[va]))
        def _pred():
            with torch.no_grad():
                return torch.sigmoid(m(T32(sq[te]), T32(St[te]), T32(mask[te])))
        _, p = timed(_pred)
        record(name, "classical", nparams(m), f, p)

    torch_seq("LSTM", fit_lstm)
    torch_seq("Transformer", tb._fit)
    original = tb.TransformerNet; tb.TransformerNet = GRUNet
    try:
        torch_seq("GRU", tb._fit)
    finally:
        tb.TransformerNet = original

    from qlstm import MinimalQLSTM, ClassicalGateLSTM, fit_generic, predict
    from run_qlstm import prep_static, prep_seq
    Stq, Sqq = prep_static(df, tr), prep_seq(seq, mask, tr)
    for name, cls, kind in (("Classical-gate LSTM", ClassicalGateLSTM, "classical"),
                            ("Minimal-QLSTM (6 qubits)", MinimalQLSTM, "quantum")):
        m, f = timed(lambda: fit_generic(cls, Sqq[tr], Stq[tr], mask[tr], y[tr],
                                         Sqq[va], Stq[va], mask[va], y[va],
                                         epochs=40, scale=1.0))
        _, p = timed(lambda: predict(m, Sqq[te], Stq[te], mask[te]))
        record(name, kind, nparams(m), f, p)

    # ---- quantum models on the tabular block
    from quantum import VQC, TRE, quantum_kernel_matrix
    nq = len(cols)
    for layers in (2, 3):
        m, f = timed(lambda: VQC(nq, n_layers=layers, epochs=60, lr=0.1, scale=1.0,
                                 seed=SEED).fit(X[tr], y[tr], X[va], y[va]))
        _, p = timed(lambda: m.predict_proba(X[te]))
        record(f"VQC ({nq} qubits, {layers} layers)", "quantum",
               int(m.w.numel() + m.b.numel()), f, p)

    def _qsvm_fit():
        K = quantum_kernel_matrix(X[tr], X[tr], nq, scale=1.0)
        return SVC(kernel="precomputed", C=1.0, probability=False).fit(K, y[tr])
    m, f = timed(_qsvm_fit)
    _, p = timed(lambda: m.decision_function(
        quantum_kernel_matrix(X[te], X[tr], nq, scale=1.0)))
    record(f"Kernel SVM ({nq} qubits)", "quantum", f"{len(tr)} support rows", f, p,
           note="fit = one n x n kernel matrix")

    NQ = 6
    StQ = np.hstack([St, np.zeros((len(St), NQ - St.shape[1]))])
    m, f = timed(lambda: TRE(n_qubits=NQ, K=seq.shape[1], epochs=40, scale=1.0,
                             seed=SEED).fit(sq[tr], StQ[tr], y[tr], sq[va], StQ[va], y[va]))
    _, p = timed(lambda: m.predict_proba(sq[te], StQ[te]))
    tre_params = int(sum(t.numel() for t in (m.a, m.b, m.W, m.Ws, m.bias)))
    record(f"TRE ({NQ} qubits)", "quantum", tre_params, f, p)

    import pennylane as qml
    env = {"machine": platform.machine(), "processor": platform.processor(),
           "system": platform.platform(), "cpu_count": os.cpu_count(),
           "python": platform.python_version(), "torch": torch.__version__,
           "pennylane": qml.__version__, "n_train": int(len(tr)), "n_test": int(len(te))}
    pd.Series(env).to_csv("results/logs/complexity_env.csv", header=["value"])
    print("\n", env)


if __name__ == "__main__":
    main()
