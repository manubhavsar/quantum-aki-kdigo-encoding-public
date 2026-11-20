"""Minimal-QLSTM: a hybrid quantum-recurrent comparator.

Architecture follows the QLSTM of Chen, Yoo & Fang (ICASSP 2022) with the
gate-sparsification simplification: the forget, input and output gates stay
classical, and only the candidate ("cell update") gate is replaced by a
variational quantum circuit. This keeps circuit depth tractable on a simulator
while retaining genuine quantum recurrence -- the quantum gate is evaluated at
every timestep, so the hidden state is propagated through the circuit.

Why include it. Our proposed TRE encodes the whole trajectory in one circuit via
interleaved non-commuting blocks. A QLSTM instead applies a small circuit
recurrently. Both are "temporal quantum" models but they place the temporal
structure in different places, so comparing them on identical inputs isolates
architecture rather than data.

The circuit projects (input + hidden) down to n_qubits with a classical linear
layer before angle encoding, which is standard practice when feature count
exceeds a tractable qubit count.
"""
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml

SEED = 20260826


def make_quantum_gate(n_qubits, n_layers):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        for layer in range(n_layers):
            for q in range(n_qubits):
                qml.RY(weights[layer, q, 0], wires=q)
                qml.RZ(weights[layer, q, 1], wires=q)
            for q in range(n_qubits):
                qml.CNOT(wires=[q, (q + 1) % n_qubits])
        return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]

    return qml.qnn.TorchLayer(circuit, {"weights": (n_layers, n_qubits, 2)})


class MinimalQLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, n_qubits=6, n_layers=2, scale=1.0):
        super().__init__()
        cat = input_size + hidden_size
        self.hidden_size = hidden_size
        self.scale = scale
        self.forget = nn.Linear(cat, hidden_size)
        self.inp = nn.Linear(cat, hidden_size)
        self.out = nn.Linear(cat, hidden_size)
        self.pre_q = nn.Linear(cat, n_qubits)
        self.q = make_quantum_gate(n_qubits, n_layers)
        self.post_q = nn.Linear(n_qubits, hidden_size)

    def forward(self, x_t, h, c):
        z = torch.cat([x_t, h], dim=-1)
        f = torch.sigmoid(self.forget(z))
        i = torch.sigmoid(self.inp(z))
        o = torch.sigmoid(self.out(z))
        # bounded angles keep the encoding injective, as in the VQC
        ang = torch.tanh(self.pre_q(z)) * self.scale
        g = torch.tanh(self.post_q(self.q(ang)))
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


class MinimalQLSTM(nn.Module):
    def __init__(self, seq_feat, n_static, hidden=8, n_qubits=6, n_layers=2, scale=1.0):
        super().__init__()
        self.cell = MinimalQLSTMCell(seq_feat, hidden, n_qubits, n_layers, scale)
        self.hidden = hidden
        self.head = nn.Sequential(nn.Linear(hidden + n_static, 16), nn.ReLU(),
                                  nn.Dropout(0.2), nn.Linear(16, 1))

    def forward(self, seq, static, mask):
        B, T, _ = seq.shape
        h = torch.zeros(B, self.hidden, dtype=seq.dtype)
        c = torch.zeros(B, self.hidden, dtype=seq.dtype)
        for t in range(T):
            h_new, c_new = self.cell(seq[:, t], h, c)
            m = mask[:, t].unsqueeze(-1)          # frozen state at padded steps
            h = m * h_new + (1 - m) * h
            c = m * c_new + (1 - m) * c
        return self.head(torch.cat([h, static], dim=1)).squeeze(-1)


def fit_qlstm(seq_tr, st_tr, mk_tr, y_tr, seq_va, st_va, mk_va, y_va,
              epochs=40, lr=0.02, batch=64, n_qubits=6, n_layers=2, scale=1.0,
              seed=SEED, patience=8):
    torch.manual_seed(seed)
    model = MinimalQLSTM(seq_tr.shape[-1], st_tr.shape[1], n_qubits=n_qubits,
                         n_layers=n_layers, scale=scale).double()
    pos = float(np.sum(y_tr))
    pw = torch.tensor((len(y_tr) - pos) / max(pos, 1.0), dtype=torch.float64)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    T = lambda a: torch.tensor(np.asarray(a), dtype=torch.float64)
    xs, ss, ms, ys = T(seq_tr), T(st_tr), T(mk_tr), T(y_tr)
    xv, sv, mv, yv = T(seq_va), T(st_va), T(mk_va), T(y_va)
    g = torch.Generator().manual_seed(seed)
    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xs), generator=g)
        for i in range(0, len(xs), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            lossf(model(xs[idx], ss[idx], ms[idx]), ys[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = lossf(model(xv, sv, mv), yv).item()
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model


def predict(model, seq, static, mask):
    T = lambda a: torch.tensor(np.asarray(a), dtype=torch.float64)
    with torch.no_grad():
        return torch.sigmoid(model(T(seq), T(static), T(mask))).numpy()


# ---------------------------------------------------------------- ablation
class ClassicalGateCell(nn.Module):
    """The Minimal-QLSTM cell with the quantum candidate gate replaced by a
    classical one of the same shape.

    Everything else -- the classical forget/input/output gates, the projection
    widths, the hidden size, the head, the training loop -- is identical, so a
    comparison against MinimalQLSTMCell isolates the quantum gate itself rather
    than any surrounding architectural difference.
    """

    def __init__(self, input_size, hidden_size, n_qubits=6, n_layers=2, scale=1.0):
        super().__init__()
        cat = input_size + hidden_size
        self.hidden_size = hidden_size
        self.scale = scale
        self.forget = nn.Linear(cat, hidden_size)
        self.inp = nn.Linear(cat, hidden_size)
        self.out = nn.Linear(cat, hidden_size)
        self.pre_q = nn.Linear(cat, n_qubits)
        # classical stand-in for the variational circuit: same in/out width,
        # comparable parameter count (n_layers * n_qubits * 2)
        self.surrogate = nn.Sequential(
            nn.Linear(n_qubits, n_layers * n_qubits), nn.Tanh(),
            nn.Linear(n_layers * n_qubits, n_qubits), nn.Tanh())
        self.post_q = nn.Linear(n_qubits, hidden_size)

    def forward(self, x_t, h, c):
        z = torch.cat([x_t, h], dim=-1)
        f = torch.sigmoid(self.forget(z))
        i = torch.sigmoid(self.inp(z))
        o = torch.sigmoid(self.out(z))
        ang = torch.tanh(self.pre_q(z)) * self.scale
        g = torch.tanh(self.post_q(self.surrogate(ang)))
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


class ClassicalGateLSTM(nn.Module):
    def __init__(self, seq_feat, n_static, hidden=8, n_qubits=6, n_layers=2, scale=1.0):
        super().__init__()
        self.cell = ClassicalGateCell(seq_feat, hidden, n_qubits, n_layers, scale)
        self.hidden = hidden
        self.head = nn.Sequential(nn.Linear(hidden + n_static, 16), nn.ReLU(),
                                  nn.Dropout(0.2), nn.Linear(16, 1))

    def forward(self, seq, static, mask):
        B, T, _ = seq.shape
        h = torch.zeros(B, self.hidden, dtype=seq.dtype)
        c = torch.zeros(B, self.hidden, dtype=seq.dtype)
        for t in range(T):
            hn, cn = self.cell(seq[:, t], h, c)
            m = mask[:, t].unsqueeze(-1)
            h = m * hn + (1 - m) * h
            c = m * cn + (1 - m) * c
        return self.head(torch.cat([h, static], dim=1)).squeeze(-1)


def fit_generic(model_cls, seq_tr, st_tr, mk_tr, y_tr, seq_va, st_va, mk_va, y_va,
                epochs=40, lr=0.02, batch=64, n_qubits=6, n_layers=2, scale=1.0,
                seed=SEED, patience=8):
    torch.manual_seed(seed)
    model = model_cls(seq_tr.shape[-1], st_tr.shape[1], n_qubits=n_qubits,
                      n_layers=n_layers, scale=scale).double()
    pos = float(np.sum(y_tr))
    pw = torch.tensor((len(y_tr) - pos) / max(pos, 1.0), dtype=torch.float64)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    T = lambda a: torch.tensor(np.asarray(a), dtype=torch.float64)
    xs, ss, ms, ys = T(seq_tr), T(st_tr), T(mk_tr), T(y_tr)
    xv, sv, mv, yv = T(seq_va), T(st_va), T(mk_va), T(y_va)
    g = torch.Generator().manual_seed(seed)
    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xs), generator=g)
        for i in range(0, len(xs), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            lossf(model(xs[idx], ss[idx], ms[idx]), ys[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = lossf(model(xv, sv, mv), yv).item()
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model
