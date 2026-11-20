"""Feature-set sensitivity: does the narrow matched feature set understate what
this cohort supports?

Compares the best narrow-block model against the best rich-feature model at each
horizon on identical out-of-fold samples, with DeLong's test. Result is saved so
the manuscript can cite it like any other measured quantity.
"""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from metrics import delong_test

def main():
    narrow = pd.read_csv("results/logs/all_results.csv")
    rich = pd.read_csv("results/logs/rich_classical_results.csv")
    rows = []
    for h in (12, 24, 48):
        n = narrow[narrow.horizon == h].nlargest(1, "auroc").iloc[0]
        r = rich[rich.horizon == h].nlargest(1, "auroc").iloc[0]
        y = np.load(f"results/preds/_y_{h}h.npy")
        pn = np.load(f"results/preds/{n['model']}_{n['block']}_{h}h.npy")
        pr = np.load(f"results/preds/{r['model']}_{r['block']}_{h}h.npy")
        a, b, p = delong_test(y, pn, pr)
        rows.append({"horizon": h,
                     "narrow_model": f"{n['model']}|{n['block']}", "narrow_auroc": a,
                     "rich_model": f"{r['model']}|{r['block']}", "rich_auroc": b,
                     "rich_n_features": int(r["n_features"]),
                     "diff": b - a, "delong_p": p})
        print(f"h={h}: narrow {a:.3f} vs rich {b:.3f} ({int(r['n_features'])} feats) "
              f"diff={b-a:+.3f} p={p:.3f}")
    d = pd.DataFrame(rows)
    d.to_csv("results/logs/feature_sensitivity.csv", index=False)
    print(f"\nmax |diff| = {d['diff'].abs().max():.3f}; "
          f"min p = {d.delong_p.min():.3f}; none significant at 0.05: "
          f"{(d.delong_p >= 0.05).all()}")

if __name__ == "__main__":
    main()
