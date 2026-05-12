"""
generate_plots.py
==================
Roda o simulador, o otimizador bayesiano e a análise de sensibilidade
e salva os gráficos em docs/screenshots/ para uso no README.

Uso:
    pip install -r requirements.txt
    python generate_plots.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── Estilo geral ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0f1117",
    "axes.facecolor":    "#1a1d27",
    "axes.edgecolor":    "#2e3250",
    "axes.labelcolor":   "#c9d1d9",
    "xtick.color":       "#8b949e",
    "ytick.color":       "#8b949e",
    "text.color":        "#c9d1d9",
    "grid.color":        "#2e3250",
    "grid.linewidth":    0.8,
    "font.family":       "DejaVu Sans",
    "figure.dpi":        150,
})

BLUE   = "#378ADD"
GREEN  = "#1D9E75"
AMBER  = "#E6A817"
RED    = "#D4537E"
PURPLE = "#7C5CBF"
CYAN   = "#17B8C8"

os.makedirs("docs/screenshots", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. SIMULATION — Recovery vs each parameter
# ─────────────────────────────────────────────────────────────────────────────
print("Generating plot 1: Parameter sweep...")

from optimizer.reservoir_simulator import simulate, parameter_bounds

bounds = parameter_bounds()
base = dict(porosity=0.20, permeability=500, water_saturation=0.30,
            net_pay=80, pressure=3500)

params_info = [
    ("porosity",         "Porosity (fraction)",    BLUE),
    ("permeability",     "Permeability (mD)",       GREEN),
    ("water_saturation", "Water Saturation (frac)", RED),
    ("net_pay",          "Net Pay (m)",             AMBER),
    ("pressure",         "Pressure (psi)",          PURPLE),
]

fig, axes = plt.subplots(1, 5, figsize=(18, 4))
fig.suptitle("Oil Recovery vs Reservoir Parameters — Sensitivity Sweep",
             fontsize=13, fontweight="bold", color="white", y=1.02)

for ax, (param, label, color) in zip(axes, params_info):
    lo, hi = bounds[param]
    xs = np.linspace(lo, hi, 60)
    ys = []
    for x in xs:
        p = {**base, param: x}
        ys.append(simulate(**p))
    ax.plot(xs, ys, color=color, linewidth=2.5)
    ax.fill_between(xs, ys, alpha=0.12, color=color)
    ax.set_xlabel(label, fontsize=9)
    ax.set_ylabel("Recovery (BPD)" if ax == axes[0] else "", fontsize=9)
    ax.grid(True, alpha=0.4)
    ax.set_title(param.replace("_", " ").title(), fontsize=9, color=color)

plt.tight_layout()
plt.savefig("docs/screenshots/1_parameter_sweep.png", bbox_inches="tight",
            facecolor="#0f1117")
plt.close()
print("  ✓ Saved: docs/screenshots/1_parameter_sweep.png")

# ─────────────────────────────────────────────────────────────────────────────
# 2. BAYESIAN OPTIMIZATION — Convergence curve
# ─────────────────────────────────────────────────────────────────────────────
print("Generating plot 2: Bayesian optimization convergence...")

from optimizer.bayesian.optimizer import run_bayesian_optimization

result = run_bayesian_optimization(n_iterations=40, n_initial_points=10)
history = result["convergence_history"]
n_init  = 10

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Bayesian Optimization — Gaussian Process + Expected Improvement",
             fontsize=13, fontweight="bold", color="white")

# Left: convergence curve
iters = list(range(1, len(history) + 1))
ax1.axvspan(1, n_init, alpha=0.08, color=AMBER, label="Random exploration")
ax1.axvspan(n_init, len(history), alpha=0.08, color=BLUE, label="BO exploitation")
ax1.plot(iters, history, color=BLUE, linewidth=2.5, zorder=3)
ax1.scatter(iters, history, color=BLUE, s=20, zorder=4, alpha=0.7)
ax1.axhline(result["best_recovery"], color=GREEN, linestyle="--",
            linewidth=1.5, label=f"Best: {result['best_recovery']:.0f} BPD")
ax1.set_xlabel("Iteration", fontsize=10)
ax1.set_ylabel("Best Recovery (BPD)", fontsize=10)
ax1.set_title("Convergence History", fontsize=10)
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.4)

# Right: best parameters bar chart
bp = result["best_params"]
labels_r = ["Porosity\n(×100%)", "Perm\n(÷100 mD)", "1-Sw\n(oil sat)", "Net Pay\n(÷10 m)", "Pres\n(÷1000 psi)"]
values_r  = [
    bp["porosity"] * 100,
    bp["permeability"] / 100,
    (1 - bp["water_saturation"]) * 100,
    bp["net_pay"] / 10,
    bp["pressure"] / 1000,
]
colors_r = [BLUE, GREEN, RED, AMBER, PURPLE]
bars = ax2.bar(labels_r, values_r, color=colors_r, alpha=0.85, edgecolor="none", width=0.55)
for bar, val in zip(bars, values_r):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f"{val:.1f}", ha="center", va="bottom", fontsize=8, color="white")
ax2.set_title(f"Optimal Parameters\n(Best recovery: {result['best_recovery']:.0f} BPD · "
              f"+{result['improvement_pct']:.0f}% vs initial)", fontsize=9)
ax2.set_ylabel("Normalised Value", fontsize=9)
ax2.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("docs/screenshots/2_bayesian_optimization.png", bbox_inches="tight",
            facecolor="#0f1117")
plt.close()
print("  ✓ Saved: docs/screenshots/2_bayesian_optimization.png")

# ─────────────────────────────────────────────────────────────────────────────
# 3. SENSITIVITY ANALYSIS — Sobol indices
# ─────────────────────────────────────────────────────────────────────────────
print("Generating plot 3: Sobol sensitivity analysis...")

from optimizer.sensitivity.sobol import run_sensitivity_analysis

sens = run_sensitivity_analysis(n_samples=256)
params_s = sens["parameters"]
s1 = sens["first_order"]
st = sens["total_order"]

x = np.arange(len(params_s))
w = 0.35

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Sobol Global Sensitivity Analysis — Which Parameters Drive Recovery?",
             fontsize=13, fontweight="bold", color="white")

# Left: grouped bar chart
bars1 = ax1.bar(x - w/2, s1, w, label="First-order (S1)", color=BLUE,   alpha=0.85, edgecolor="none")
bars2 = ax1.bar(x + w/2, st, w, label="Total-order (ST)",  color=GREEN,  alpha=0.85, edgecolor="none")
ax1.set_xticks(x)
ax1.set_xticklabels([p.replace(" (", "\n(") for p in params_s], fontsize=8)
ax1.set_ylabel("Sensitivity Index", fontsize=10)
ax1.set_title("Sobol Sensitivity Indices", fontsize=10)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.4, axis="y")
ax1.set_ylim(0, max(st) * 1.3)
for bar, val in zip(bars2, st):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f"{val:.2f}", ha="center", va="bottom", fontsize=7.5, color="white")

# Right: horizontal ranked bar
sorted_idx = np.argsort(st)[::-1]
sorted_params = [params_s[i].replace(" (mD)", "").replace(" (psi)", "").replace(" (m)", "") for i in sorted_idx]
sorted_st = [st[i] for i in sorted_idx]
bar_colors = [GREEN, BLUE, AMBER, PURPLE, RED]
bars3 = ax2.barh(sorted_params, sorted_st, color=bar_colors, alpha=0.85, edgecolor="none", height=0.55)
for bar, val in zip(bars3, sorted_st):
    ax2.text(val + 0.005, bar.get_y() + bar.get_height()/2,
             f"{val:.3f}", va="center", fontsize=9, color="white")
ax2.set_xlabel("Total-order Sobol Index (ST)", fontsize=10)
ax2.set_title(f"Parameter Ranking\nMost influential: {sens['most_influential']}", fontsize=10)
ax2.grid(True, alpha=0.4, axis="x")
ax2.invert_yaxis()

plt.tight_layout()
plt.savefig("docs/screenshots/3_sensitivity_analysis.png", bbox_inches="tight",
            facecolor="#0f1117")
plt.close()
print("  ✓ Saved: docs/screenshots/3_sensitivity_analysis.png")

# ─────────────────────────────────────────────────────────────────────────────
# 4. DASHBOARD — Combined summary
# ─────────────────────────────────────────────────────────────────────────────
print("Generating plot 4: Dashboard summary...")

fig = plt.figure(figsize=(16, 9))
gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

fig.patch.set_facecolor("#0f1117")
fig.suptitle("🛢️  Reservoir Optimizer — Dashboard Summary",
             fontsize=15, fontweight="bold", color="white", y=0.98)

# ── KPI boxes ────────────────────────────────────────────────────────────────
kpis = [
    ("Best Recovery",    f"{result['best_recovery']:.0f} BPD", GREEN),
    ("Improvement",      f"+{result['improvement_pct']:.0f}%",  BLUE),
    ("Most Influential", sens["most_influential"].split()[0],   AMBER),
]
for i, (label, value, color) in enumerate(kpis):
    ax = fig.add_subplot(gs[0, i])
    ax.set_facecolor("#1a1d27")
    ax.text(0.5, 0.62, value, transform=ax.transAxes, fontsize=22,
            fontweight="bold", color=color, ha="center", va="center")
    ax.text(0.5, 0.25, label, transform=ax.transAxes, fontsize=10,
            color="#8b949e", ha="center", va="center")
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(2)
    ax.set_xticks([]); ax.set_yticks([])

# ── Convergence (bottom left) ─────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
ax4.set_facecolor("#1a1d27")
ax4.plot(range(1, len(history)+1), history, color=BLUE, linewidth=2)
ax4.fill_between(range(1, len(history)+1), history, alpha=0.15, color=BLUE)
ax4.axhline(result["best_recovery"], color=GREEN, linestyle="--", linewidth=1.2)
ax4.set_title("BO Convergence", fontsize=9, color="white")
ax4.set_xlabel("Iteration", fontsize=8)
ax4.set_ylabel("Recovery (BPD)", fontsize=8)
ax4.grid(True, alpha=0.3)

# ── Sensitivity (bottom center) ───────────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
ax5.set_facecolor("#1a1d27")
short = [p.split()[0] for p in params_s]
ax5.bar(short, st, color=[BLUE, GREEN, RED, AMBER, PURPLE], alpha=0.85, edgecolor="none")
ax5.set_title("Sobol ST Indices", fontsize=9, color="white")
ax5.set_ylabel("Total-order Index", fontsize=8)
ax5.grid(True, alpha=0.3, axis="y")

# ── Parameter sweep — permeability (bottom right) ────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
ax6.set_facecolor("#1a1d27")
xs2 = np.linspace(1, 3000, 80)
ys2 = [simulate(**{**base, "permeability": x}) for x in xs2]
ax6.plot(xs2, ys2, color=GREEN, linewidth=2)
ax6.fill_between(xs2, ys2, alpha=0.12, color=GREEN)
ax6.axvline(result["best_params"]["permeability"], color=AMBER,
            linestyle="--", linewidth=1.2, label="Optimal")
ax6.set_title("Recovery vs Permeability", fontsize=9, color="white")
ax6.set_xlabel("Permeability (mD)", fontsize=8)
ax6.set_ylabel("Recovery (BPD)", fontsize=8)
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3)

plt.savefig("docs/screenshots/4_dashboard.png", bbox_inches="tight",
            facecolor="#0f1117")
plt.close()
print("  ✓ Saved: docs/screenshots/4_dashboard.png")

print("\n✅ All plots saved to docs/screenshots/")
print("   Add them to your README with:")
print("   ![Parameter Sweep](docs/screenshots/1_parameter_sweep.png)")
print("   ![Bayesian Opt](docs/screenshots/2_bayesian_optimization.png)")
print("   ![Sensitivity](docs/screenshots/3_sensitivity_analysis.png)")
print("   ![Dashboard](docs/screenshots/4_dashboard.png)")
