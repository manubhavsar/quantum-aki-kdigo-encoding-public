"""Forest plot of the matched quantum-classical pairs (Table 3a), one panel per horizon.

Reads results/logs/paired_comparisons.csv. Ablation pairs (the quantum model against
its own ablated control) and head-to-head pairs (against an independent classical
model) differ in colour and marker shape, so identity never rests on colour alone.
Filled markers have unadjusted clustered p < 0.05.
"""
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d3"
HEAD, ABL = "#2a78d6", "#eb6834"          # validated pair (light surface)
ORDER = [
    ("TRE vs additive control", "TRE vs additive control (abl.)", True),
    ("Minimal-QLSTM vs classical-gate LSTM", "QLSTM vs classical gate (abl.)", True),
    ("TRE vs transformer", "TRE vs transformer", False),
    ("TRE vs LSTM", "TRE vs LSTM", False),
    ("Minimal-QLSTM vs transformer", "QLSTM vs transformer", False),
    ("VQC vs MLP control (delta)", "VQC vs MLP control", False),
    ("VQC vs logistic regression (delta)", "VQC vs logistic regr.", False),
    ("VQC vs tabular transformer (delta)", "VQC vs tabular transf.", False),
    ("Kernel SVM vs logistic regression (delta)", "Kernel SVM vs logistic regr.", False),
    ("Kernel SVM vs tabular transformer (delta)", "Kernel SVM vs tabular transf.", False),
]

d = pd.read_csv("results/logs/paired_comparisons.csv")
plt.rcParams.update({"font.family": "serif", "font.size": 7, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK})
fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.3), sharey=True,
                         gridspec_kw={"wspace": 0.06})
ys = list(range(len(ORDER)))[::-1]
for ax, h in zip(axes, (12, 24, 48)):
    ax.axvline(0, color=MUTED, lw=0.8, zorder=1)
    for y, (key, label, abl) in zip(ys, ORDER):
        r = d[(d.pair == key) & (d.horizon == h)].iloc[0]
        col, mk = (ABL, "s") if abl else (HEAD, "o")
        ax.plot([r.lo, r.hi], [y, y], color=col, lw=1.4, solid_capstyle="round", zorder=2)
        ax.plot(r["diff"], y, marker=mk, ms=4.6, mec=col, mew=1.2,
                mfc=col if r.p < 0.05 else "white", zorder=3)
    ax.set_title(f"{h} h", fontsize=7.5, color=INK, pad=3)
    ax.set_xlim(-0.11, 0.14)
    ax.set_xticks([-0.1, -0.05, 0, 0.05, 0.1])
    ax.set_xticklabels(["-0.10", "-0.05", "0", "+0.05", "+0.10"])
    ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=2, width=0.6)
axes[0].set_yticks(ys)
axes[0].set_yticklabels([lab for _, lab, _ in ORDER])
axes[1].set_xlabel("AUROC difference, quantum minus classical (95% patient-level CI)", labelpad=2)
from matplotlib.lines import Line2D
handles = [Line2D([], [], color=ABL, marker="s", mfc=ABL, ms=4.5, lw=1.4, label="ablation of the quantum model"),
           Line2D([], [], color=HEAD, marker="o", mfc=HEAD, ms=4.5, lw=1.4, label="independent classical model"),
           Line2D([], [], color=MUTED, marker="o", mfc="white", ms=4.5, lw=0, label="hollow: p ≥ 0.05")]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=6.6,
           bbox_to_anchor=(0.6, 0.0))
fig.subplots_adjust(left=0.228, right=0.99, top=0.89, bottom=0.27)
for out in ("results/figures/fig_forest.pdf", "paper/figures/fig_forest.pdf"):
    fig.savefig(out)
fig.savefig("results/figures/fig_forest.png", dpi=200)
print("saved fig_forest")
