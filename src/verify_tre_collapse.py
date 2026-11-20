"""Exact characterisation of what the no-interleaving TRE ablation computes.

Without interleaved entanglers all rotation blocks act on the same qubits and the
RY rotations commute, so the net rotation on qubit q is the SUM

    theta_q = sum_j tanh( s_j ( a_q c_j + b_q dt_j ) ) * gamma ,   s_j = (-1)^{K-1-j}.

A sum is invariant to reordering its terms. Timepoints sharing a sign (with K = 4,
positions 0 and 2, and positions 1 and 3) can therefore be exchanged without
changing theta_q at all, so the ablated circuit cannot distinguish such
trajectories. Interleaving entanglers breaks the commutation and removes that
invariance. This is an exact algebraic property, verified below without any
numerical fitting.
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from quantum import TRE

SEED, K = 20260826, 4


def main():
    rng = np.random.default_rng(SEED)
    n = 300
    seq = rng.normal(size=(n, K, 2))
    static = rng.normal(size=(n, 6))
    y = ((seq[:, -1, 0] - seq[:, -2, 0]) > 0).astype(int)

    # exchange timepoints within each sign class: (0,2) and (1,3)
    swapped = seq.copy()
    swapped[:, [0, 2], :] = seq[:, [2, 0], :]
    swapped[:, [1, 3], :] = seq[:, [3, 1], :]

    print("Exchanging timepoints within the same sign class (0<->2, 1<->3):")
    res = {}
    for ent in (False, True):
        m = TRE(n_qubits=6, K=K, epochs=10, entangle_between=ent, seed=SEED).fit(
            seq, static, y, seq, static, y)
        d = np.abs(m.predict_proba(seq, static) - m.predict_proba(swapped, static)).max()
        res[ent] = d
        lab = "without entanglers" if not ent else "with entanglers   "
        print(f"  {lab}: max |prediction change| = {d:.3e}")

    print()
    if res[False] < 1e-9 and res[True] > 1e-3:
        print("CONFIRMED. The ablated circuit is exactly invariant to these exchanges:")
        print("the trajectory enters only through a sum of per-timepoint terms, so no")
        print("interaction between timepoints can be represented. The interleaved")
        print("circuit is not invariant and can represent such interactions.")
    else:
        print(f"NOT CONFIRMED (ablated {res[False]:.2e}, interleaved {res[True]:.2e})")


if __name__ == "__main__":
    main()
