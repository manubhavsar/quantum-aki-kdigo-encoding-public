"""Publication figures. Every figure is generated from saved prediction files,
so nothing in the manuscript is drawn by hand."""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score
from sklearn.calibration import calibration_curve
sys.path.insert(0, os.path.dirname(__file__))

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": 300, "savefig.dpi": 300, "axes.grid": True,
    "grid.alpha": 0.3, "grid.linewidth": 0.4, "axes.axisbelow": True,
    "font.family": "serif", "savefig.bbox": "tight",
})
FIG = "results/figures"
P = "results/preds"
HOR = [12, 24, 48]


def lp(m, b, h):
    f = f"{P}/{m}_{b}_{h}h.npy"
    return np.load(f) if os.path.exists(f) else None


def ly(h):
    return np.load(f"{P}/_y_{h}h.npy")


def fig_interaction():
    """2x2: AUROC by model family and encoding, per horizon."""
    fam = {"Logistic Reg.": "LogisticRegression", "Random Forest": "RandomForest",
           "XGBoost": "XGBoost", "VQC": "VQC", "QSVM": "QSVM"}
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.85), sharey=True)
    for ax, h in zip(axes, HOR):
        y = ly(h)
        names, st, dl = [], [], []
        for lab, m in fam.items():
            a, b = lp(m, "static", h), lp(m, "delta", h)
            if a is None or b is None:
                continue
            names.append(lab)
            st.append(roc_auc_score(y, a)); dl.append(roc_auc_score(y, b))
        x = np.arange(len(names)); w = 0.38
        ax.bar(x - w/2, st, w, label="static encoding", color="#b0b0b0", edgecolor="k", linewidth=0.4)
        ax.bar(x + w/2, dl, w, label="delta encoding (proposed)", color="#2b6cb0",
               edgecolor="k", linewidth=0.4)
        ax.axhline(0.5, color="k", ls=":", lw=0.6)
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=35, ha="right")
        ax.set_title(f"{h} h horizon"); ax.set_ylim(0.45, 0.80)
        if h == HOR[0]:
            ax.set_ylabel("AUROC")
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig2_interaction.pdf"); plt.close(fig)
    print("fig2_interaction.pdf")


def fig_roc_pr():
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.4))
    show = [("LogisticRegression", "delta", "LR (delta)"),
            ("LogisticRegression", "static", "LR (static)"),
            ("VQC", "delta", "VQC (delta)"), ("VQC", "static", "VQC (static)"),
            ("QSVM", "delta", "QSVM (delta)"), ("TRE", "trajectory", "TRE (proposed)"),
            ("LSTM", "trajectory", "LSTM")]
    for j, h in enumerate(HOR):
        y = ly(h)
        for m, b, lab in show:
            p = lp(m, b, h)
            if p is None:
                continue
            fpr, tpr, _ = roc_curve(y, p)
            axes[0, j].plot(fpr, tpr, lw=1.0, label=f"{lab} ({roc_auc_score(y,p):.2f})")
            pr, rc, _ = precision_recall_curve(y, p)
            axes[1, j].plot(rc, pr, lw=1.0)
        axes[0, j].plot([0, 1], [0, 1], "k:", lw=0.6)
        axes[0, j].set_title(f"{h} h horizon"); axes[0, j].set_xlabel("1 - specificity")
        axes[1, j].axhline(y.mean(), color="k", ls=":", lw=0.6)
        axes[1, j].set_xlabel("recall")
        if j == 0:
            axes[0, j].set_ylabel("sensitivity"); axes[1, j].set_ylabel("precision")
    axes[0, 0].legend(frameon=False, fontsize=5.5, loc="lower right")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig3_roc_pr.pdf"); plt.close(fig)
    print("fig3_roc_pr.pdf")


def fig_concentration():
    """Kernel concentration vs bandwidth and qubit count (Thanasilp et al. 2024)."""
    from quantum import quantum_kernel_matrix
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.3))
    X = rng.normal(size=(60, 8))
    bws = [0.15, 0.3, 0.6, 1.0, 1.5, np.pi]
    means, stds = [], []
    for bw in bws:
        K = quantum_kernel_matrix(X, X, 8, scale=bw)
        off = K[~np.eye(60, dtype=bool)]
        means.append(off.mean()); stds.append(off.std())
    axes[0].errorbar(bws, means, yerr=stds, marker="o", ms=3, lw=1, capsize=2, color="#2b6cb0")
    axes[0].set_xlabel("encoding bandwidth"); axes[0].set_ylabel("off-diagonal kernel value")
    axes[0].set_title("(a) bandwidth vs kernel spread"); axes[0].set_xscale("log")
    qs = [4, 6, 8, 10, 12]
    m2, s2 = [], []
    for nq in qs:
        Xq = rng.normal(size=(50, nq))
        K = quantum_kernel_matrix(Xq, Xq, nq, scale=1.0)
        off = K[~np.eye(50, dtype=bool)]
        m2.append(off.mean()); s2.append(off.std())
    axes[1].errorbar(qs, m2, yerr=s2, marker="s", ms=3, lw=1, capsize=2, color="#c05621")
    axes[1].set_xlabel("number of qubits"); axes[1].set_ylabel("off-diagonal kernel value")
    axes[1].set_title("(b) exponential concentration"); axes[1].set_yscale("log")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig4_concentration.pdf"); plt.close(fig)
    print("fig4_concentration.pdf")


def fig_calibration():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))
    for ax, h in zip(axes, HOR):
        y = ly(h)
        for m, b, lab in [("LogisticRegression", "delta", "LR (delta)"),
                          ("VQC", "delta", "VQC (delta)"),
                          ("XGBoost", "delta", "XGBoost (delta)")]:
            p = lp(m, b, h)
            if p is None:
                continue
            p = (p - p.min()) / (p.max() - p.min() + 1e-9)
            try:
                fo, mp = calibration_curve(y, p, n_bins=6, strategy="quantile")
                ax.plot(mp, fo, marker="o", ms=3, lw=1, label=lab)
            except Exception:
                pass
        ax.plot([0, 1], [0, 1], "k:", lw=0.6)
        ax.set_title(f"{h} h"); ax.set_xlabel("predicted probability")
        if h == HOR[0]:
            ax.set_ylabel("observed frequency"); ax.legend(frameon=False, fontsize=6)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig5_calibration.pdf"); plt.close(fig)
    print("fig5_calibration.pdf")


def fig_ablation():
    ab = pd.read_csv("results/logs/ablations.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2))
    specs = [("n_qubits", "number of qubits", "(a) encoding width"),
             ("n_layers", "circuit depth (layers)", "(b) variational depth"),
             ("bandwidth", r"encoding bandwidth $\gamma$", "(c) encoding bandwidth")]
    for ax, (key, xlab, title) in zip(axes, specs):
        s = ab[ab.ablation == key].copy()
        s["value"] = s.value.astype(float)
        s = s.sort_values("value")
        ax.plot(s.value, s.auroc, marker="o", ms=3.5, lw=1.1, color="#2b6cb0")
        ax.set_xlabel(xlab); ax.set_title(title)
        if key == "bandwidth":
            ax.set_xscale("log")
            ax.axvline(np.pi, color="#c05621", ls="--", lw=0.8)
            ax.text(np.pi, ax.get_ylim()[0], r" $\pi$", color="#c05621",
                    fontsize=6, va="bottom")
    axes[0].set_ylabel("AUROC (24 h)")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig6_ablation.pdf"); plt.close(fig)
    print("fig6_ablation.pdf")


def fig_ablation_combined():
    """Ablations plus the measured kernel concentration, as one 1x4 float.

    Panel geometry is set so that tick labels remain legible at the printed
    width: log axes use explicit sparse ticks rather than the automatic locator,
    which otherwise crowds the 1.7 in panels.
    """
    from matplotlib.ticker import FixedLocator, LogFormatterSciNotation
    ab = pd.read_csv("results/logs/ablations.csv")
    fig, axes = plt.subplots(1, 4, figsize=(7.1, 1.70))
    specs = [("n_qubits", "qubits", "(a) encoding width"),
             ("n_layers", "layers", "(b) variational depth"),
             ("bandwidth", r"bandwidth $\gamma$", "(c) encoding bandwidth")]
    for ax, (key, xlab, title) in zip(axes, specs):
        d = ab[ab.ablation == key].copy()
        d["value"] = d.value.astype(float)
        d = d.sort_values("value")
        ax.plot(d.value, d.auroc, marker="o", ms=3.2, lw=1.1, color="#2b6cb0")
        ax.set_xlabel(xlab, fontsize=7)
        ax.set_title(title, fontsize=7.5)
        ax.tick_params(labelsize=6)
        if key == "bandwidth":
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(FixedLocator([0.2, 0.5, 1.0, 3.0]))
            ax.xaxis.set_minor_locator(FixedLocator([]))
            ax.set_xticklabels(["0.2", "0.5", "1.0", "3.0"])
            ax.axvline(np.pi, color="#c05621", ls="--", lw=0.8)
        else:
            ax.xaxis.set_major_locator(FixedLocator(sorted(d.value.unique())))
    axes[0].set_ylabel("AUROC (24 h)", fontsize=7)

    kc = pd.read_csv("results/logs/kernel_concentration_qubits.csv")
    axes[3].errorbar(kc.n_qubits, kc.offdiag_mean, yerr=kc.offdiag_std,
                     marker="s", ms=3.2, lw=1.1, capsize=2, color="#c05621")
    axes[3].set_yscale("log")
    axes[3].yaxis.set_major_locator(FixedLocator([0.1, 0.3, 1.0]))
    axes[3].yaxis.set_minor_locator(FixedLocator([]))
    axes[3].set_yticklabels(["0.1", "0.3", "1.0"])
    axes[3].xaxis.set_major_locator(FixedLocator(list(kc.n_qubits)))
    axes[3].set_xlabel("qubits", fontsize=7)
    axes[3].set_ylabel("kernel off-diagonal", fontsize=7)
    axes[3].set_title("(d) concentration", fontsize=7.5)
    axes[3].tick_params(labelsize=6)
    fig.tight_layout(pad=0.4, w_pad=1.1)
    fig.savefig(f"{FIG}/fig_ablation_combined.pdf")
    plt.close(fig)
    print("fig_ablation_combined.pdf")



SHORT_LABEL = {
    "Unmatched classical architecture": "architecture matching",
    "Unreported encoding bandwidth": "encoding bandwidth",
    "Cohort scale (demo-scale only)": "cohort scale",
    "Cohort scale (demo-scale evaluation only)": "cohort scale",
    "Horizon selection (best of three)": "horizon selection",
    "Crediting the model for its features": "feature crediting",
    "Feature confounding (no interaction test)": "feature crediting",
    "Label leakage (prevalent windows kept)": "label leakage",
    "Label leakage (post-onset windows retained)": "label leakage",
    "Tuning asymmetry (untuned classical baseline)": "classical tuning",
    "Untuned classical baseline": "classical tuning",
    "Seed selection (best-of-N reporting)": "seed selection",
    "Seed selection (best of eight)": "seed selection",
}


def _short(name):
    return SHORT_LABEL.get(name, name.split(" (")[0])


def fig_taxonomy():
    """Audited evaluation choices, separated by the quantity each one moves."""
    import json
    t = json.load(open("results/logs/artifact_taxonomy.json"))
    A = pd.DataFrame(t["absolute"]).sort_values("magnitude")
    B = pd.DataFrame(t["gap"]).sort_values("magnitude")
    ref = t["largest_controlled_effect"]

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 1.70),
                             gridspec_kw={"width_ratios": [1, 2.3]})

    ax = axes[0]
    ax.barh(np.arange(len(A)), A.magnitude, color="#7b341e", edgecolor="k",
            linewidth=0.4, height=0.55)
    for i, m in enumerate(A.magnitude):
        ax.text(m + 0.006, i, f"{m:+.3f}", va="center", fontsize=6.5)
    ax.set_yticks(np.arange(len(A)))
    ax.set_yticklabels([_short(a) for a in A.choice], fontsize=6.6)
    ax.set_xlim(0, max(A.magnitude) * 1.45)
    ax.set_ylim(-0.8, len(A) - 0.2)
    ax.set_xlabel("shift in absolute AUROC", fontsize=7)
    ax.set_title("(a) inflates reported performance", fontsize=7.5)
    ax.tick_params(labelsize=6)

    ax = axes[1]
    colors = ["#c05621" if m > ref else "#a0aec0" for m in B.magnitude]
    ax.barh(np.arange(len(B)), B.magnitude, color=colors, edgecolor="k",
            linewidth=0.4, height=0.62)
    for i, m in enumerate(B.magnitude):
        ax.text(m + 0.002, i, f"{m:+.3f}", va="center", fontsize=6.5)
    ax.axvline(ref, color="#2b6cb0", ls="--", lw=1.2)
    ax.text(ref + 0.002, -1.25,
            f"largest quantum-component\neffect under control ({ref:+.3f})",
            color="#2b6cb0", fontsize=6.2, va="bottom")
    ax.set_yticks(np.arange(len(B)))
    ax.set_yticklabels([_short(b) for b in B.choice], fontsize=6.6)
    ax.set_xlim(0, max(B.magnitude) * 1.30)
    ax.set_ylim(-1.8, len(B) - 0.4)
    ax.set_xlabel("shift in the quantum--classical gap $G$", fontsize=7)
    ax.set_title("(b) shifts the quantum--classical comparison", fontsize=7.5)
    ax.tick_params(labelsize=6)
    ax.grid(axis="x", alpha=0.3, lw=0.4)

    fig.tight_layout(pad=0.4, w_pad=1.5)
    fig.savefig(f"{FIG}/fig1_taxonomy.pdf")
    plt.close(fig)
    print("fig1_taxonomy.pdf")


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "main"):
        fig_taxonomy()
        fig_ablation_combined()
    if which == "extra":
        fig_interaction(); fig_roc_pr(); fig_calibration(); fig_concentration()
