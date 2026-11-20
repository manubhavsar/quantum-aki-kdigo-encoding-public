"""Quantum models: variational quantum classifier and quantum-kernel SVM.

Two encodings are compared under an otherwise identical circuit, so that any
difference is attributable to the encoded quantity rather than to architecture:

  static : angle encoding of level-based features (cr_last, cr_mean, ...), which
           mirrors the static encodings used in existing CKD-QML work.
  delta  : angle encoding of KDIGO-aligned rate-of-change features
           (cr_delta_48h, cr_ratio_7d, ...).

Motivation for the delta encoding. Sequential RY rotations on one qubit compose
additively, RY(a) RY(-b) = RY(a - b), so a rotation angle carrying a difference
of two timepoints is the natural single-qubit representation of the quantity
KDIGO actually thresholds. Encoding levels instead forces the variational layers
to synthesise that difference indirectly.
"""
import os, sys, time
import numpy as np
import torch
import pennylane as qml

sys.path.insert(0, os.path.dirname(__file__))
SEED = 20260826
torch.manual_seed(SEED)


# --------------------------------------------------------------------- circuit
def make_vqc(n_qubits, n_layers, reupload=1):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(x, weights):
        for r in range(reupload):
            qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights[r], wires=range(n_qubits))
        return qml.expval(qml.PauliZ(0))

    shape = (reupload, n_layers, n_qubits, 3)
    return circuit, shape


class VQC:
    """Variational quantum classifier trained with weighted BCE."""

    def __init__(self, n_qubits, n_layers=3, reupload=1, epochs=60, lr=0.05,
                 batch=64, scale=np.pi, seed=SEED, verbose=False):
        self.n_qubits, self.n_layers, self.reupload = n_qubits, n_layers, reupload
        self.epochs, self.lr, self.batch = epochs, lr, batch
        self.scale, self.seed, self.verbose = scale, seed, verbose

    def _prep(self, X):
        """Squash standardised features into a bounded rotation range."""
        return torch.tensor(np.tanh(np.asarray(X, float)) * self.scale, dtype=torch.float64)

    def fit(self, X, y, Xva=None, yva=None):
        g = torch.Generator().manual_seed(self.seed)
        self.circuit, shape = make_vqc(self.n_qubits, self.n_layers, self.reupload)
        self.w = (0.1 * torch.randn(shape, generator=g, dtype=torch.float64)).requires_grad_(True)
        self.b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        Xt, yt = self._prep(X), torch.tensor(np.asarray(y), dtype=torch.float64)
        pos = float(yt.sum()); neg = float(len(yt) - pos)
        pw = torch.tensor(neg / max(pos, 1.0), dtype=torch.float64)
        opt = torch.optim.Adam([self.w, self.b], lr=self.lr)
        n = len(Xt)
        best, best_state, patience = np.inf, None, 0
        for ep in range(self.epochs):
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, self.batch):
                idx = perm[i:i + self.batch]
                opt.zero_grad()
                logits = self.circuit(Xt[idx], self.w) + self.b
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, yt[idx], pos_weight=pw)
                loss.backward()
                opt.step()
            if Xva is not None:
                with torch.no_grad():
                    lv = self.circuit(self._prep(Xva), self.w) + self.b
                    vl = torch.nn.functional.binary_cross_entropy_with_logits(
                        lv, torch.tensor(np.asarray(yva), dtype=torch.float64),
                        pos_weight=pw).item()
                if vl < best - 1e-4:
                    best, best_state, patience = vl, (self.w.detach().clone(),
                                                      self.b.detach().clone()), 0
                else:
                    patience += 1
                    if patience >= 10:
                        break
        if best_state is not None:
            with torch.no_grad():
                self.w.copy_(best_state[0]); self.b.copy_(best_state[1])
        return self

    def predict_proba(self, X):
        with torch.no_grad():
            logits = self.circuit(self._prep(X), self.w) + self.b
            return torch.sigmoid(logits).numpy()


# ---------------------------------------------------------------- quantum kernel
def quantum_kernel_matrix(A, B, n_qubits, scale=np.pi, chunk=20000, symmetric=None):
    """Fidelity kernel K(a,b) = |<phi(a)|phi(b)>|^2 via the adjoint embedding test.

    All (i, j) pairs are evaluated in a few large batched circuit calls rather
    than one call per row, which dominates runtime on a statevector simulator.
    When A is B the upper triangle is computed and mirrored, halving the work.
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def kern(a, b):
        qml.AngleEmbedding(a, wires=range(n_qubits), rotation="Y")
        qml.adjoint(qml.AngleEmbedding)(b, wires=range(n_qubits), rotation="Y")
        return qml.probs(wires=range(n_qubits))

    At = torch.tensor(np.tanh(np.asarray(A, float)) * scale, dtype=torch.float64)
    Bt = torch.tensor(np.tanh(np.asarray(B, float)) * scale, dtype=torch.float64)
    n, m = len(At), len(Bt)
    if symmetric is None:
        symmetric = (n == m) and np.array_equal(np.asarray(A), np.asarray(B))

    if symmetric:
        iu = np.triu_indices(n, k=1)
        I, J = iu[0], iu[1]
    else:
        I, J = np.meshgrid(np.arange(n), np.arange(m), indexing="ij")
        I, J = I.ravel(), J.ravel()

    vals = np.empty(len(I), dtype=float)
    for s0 in range(0, len(I), chunk):
        sl = slice(s0, s0 + chunk)
        with torch.no_grad():
            vals[sl] = kern(At[I[sl]], Bt[J[sl]])[:, 0].numpy()

    if symmetric:
        K = np.eye(n)
        K[I, J] = vals
        K[J, I] = vals
        return K
    return vals.reshape(n, m)


# ------------------------------------------- trajectory re-uploading encoding
class TRE:
    """Trajectory Re-uploading Encoding (proposed).

    One rotation block per measurement timepoint, with alternating sign,
    interleaved with entangling layers:

        for j = 0..K-1:
            RY_q( s_j * ( a_q * cr_j + b_q * dt_j ) )   for every qubit q
            StronglyEntanglingLayers(W_j)

    where s_j = (-1)^(K-1-j) alternates.

    Why this is not merely "delta features in a circuit". Remove the entangling
    layers and the per-qubit rotations commute, so the net rotation on qubit q
    collapses to

        sum_j s_j (a_q cr_j + b_q dt_j)
          = a_q (cr_K - cr_{K-1} + cr_{K-2} - ...) + ...,

    i.e. exactly a (signed) difference of the trajectory -- the delta encoding.
    With entangling layers between the rotation blocks the blocks no longer
    commute, so the circuit can represent trajectory structure beyond a single
    difference (curvature, non-monotonicity). The `entangle_between=False`
    ablation therefore recovers delta encoding as a special case, which makes
    the contribution of the interleaving directly measurable.
    """

    def __init__(self, n_qubits=6, K=4, n_layers=1, epochs=60, lr=0.1, batch=64,
                 scale=1.0, entangle_between=True, seed=SEED):
        self.n_qubits, self.K, self.n_layers = n_qubits, K, n_layers
        self.epochs, self.lr, self.batch = epochs, lr, batch
        self.scale, self.entangle_between, self.seed = scale, entangle_between, seed

    def _build(self):
        nq, K, nl = self.n_qubits, self.K, self.n_layers
        dev = qml.device("default.qubit", wires=nq)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(seq, static, a, b, W, Ws):
            # static context encoded once
            qml.AngleEmbedding(static, wires=range(nq), rotation="Y")
            qml.StronglyEntanglingLayers(Ws, wires=range(nq))
            for j in range(K):
                s = (-1.0) ** (K - 1 - j)
                cr = seq[..., j, 0].unsqueeze(-1)
                dt = seq[..., j, 1].unsqueeze(-1)
                ang = torch.tanh(s * (cr * a + dt * b)) * self.scale
                qml.AngleEmbedding(ang, wires=range(nq), rotation="Y")
                if self.entangle_between:
                    qml.StronglyEntanglingLayers(W[j], wires=range(nq))
            if not self.entangle_between:
                qml.StronglyEntanglingLayers(W[0], wires=range(nq))
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit
        return (nl, nq, 3)

    def fit(self, seq, static, y, seq_va=None, st_va=None, y_va=None):
        g = torch.Generator().manual_seed(self.seed)
        wshape = self._build()
        D = lambda x: torch.tensor(np.asarray(x, float), dtype=torch.float64)
        self.a = (0.5 * torch.randn(self.n_qubits, generator=g, dtype=torch.float64)).requires_grad_(True)
        self.b = (0.1 * torch.randn(self.n_qubits, generator=g, dtype=torch.float64)).requires_grad_(True)
        self.W = (0.1 * torch.randn((self.K,) + wshape, generator=g, dtype=torch.float64)).requires_grad_(True)
        self.Ws = (0.1 * torch.randn(wshape, generator=g, dtype=torch.float64)).requires_grad_(True)
        self.bias = torch.zeros(1, dtype=torch.float64, requires_grad=True)

        Sq, St, yt = D(seq), D(np.tanh(np.asarray(static, float))), D(y)
        pos = float(yt.sum()); pw = torch.tensor(
            (len(yt) - pos) / max(pos, 1.0), dtype=torch.float64)
        params = [self.a, self.b, self.W, self.Ws, self.bias]
        opt = torch.optim.Adam(params, lr=self.lr)
        n = len(Sq)
        best, best_state, patience = np.inf, None, 0
        for ep in range(self.epochs):
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, self.batch):
                idx = perm[i:i + self.batch]
                opt.zero_grad()
                out = self.circuit(Sq[idx], St[idx], self.a, self.b, self.W, self.Ws) + self.bias
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    out, yt[idx], pos_weight=pw)
                loss.backward(); opt.step()
            if seq_va is not None:
                with torch.no_grad():
                    ov = self.circuit(D(seq_va), D(np.tanh(np.asarray(st_va, float))),
                                      self.a, self.b, self.W, self.Ws) + self.bias
                    vl = torch.nn.functional.binary_cross_entropy_with_logits(
                        ov, D(y_va), pos_weight=pw).item()
                if vl < best - 1e-4:
                    best, patience = vl, 0
                    best_state = [p.detach().clone() for p in params]
                else:
                    patience += 1
                    if patience >= 10:
                        break
        if best_state:
            with torch.no_grad():
                for p, s in zip(params, best_state):
                    p.copy_(s)
        return self

    def predict_proba(self, seq, static):
        D = lambda x: torch.tensor(np.asarray(x, float), dtype=torch.float64)
        with torch.no_grad():
            out = self.circuit(D(seq), D(np.tanh(np.asarray(static, float))),
                               self.a, self.b, self.W, self.Ws) + self.bias
            return torch.sigmoid(out).numpy()
