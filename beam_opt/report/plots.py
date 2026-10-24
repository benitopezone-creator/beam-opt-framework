"""
beam_opt.report.plots
=====================
Matplotlib result plots.
"""

from __future__ import annotations

import numpy as np

from ..optimization.problem import OptimizationProblem

_COLORS = ["#2563EB", "#DC2626", "#16A34A", "#D97706"]
_MARKERS = ["o", "s", "^", "D"]


def plot_results(
    problem: OptimizationProblem,
    x_opt: np.ndarray,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """
    Two figures: section parameters p(s) and von Mises stress σ_VM(s).

    Parameters
    ----------
    problem : OptimizationProblem
        The solved problem.
    x_opt : ndarray
        Optimised design vector.
    save_path : str | None
        Path prefix for PNG output (e.g. ``"results/run1"``): saves
        ``<prefix>_sections.png`` and ``<prefix>_stress.png``.
    show : bool
        Call ``plt.show()`` at the end.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib import ticker
    except ImportError:
        print("  [WARNING] matplotlib not available – plots skipped.")
        return

    law = problem.law_from_x(x_opt)
    s_mid = problem.model.beam.mesh.s_elem
    params = law.eval(s_mid)
    sigma_vals = problem.evaluate(x_opt).sigma
    sigma_amm = problem.sigma_adm

    # ── Figure 1: section parameters ─────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    lows, highs = [], []
    for k, (name, values) in enumerate(params.items()):
        ax1.plot(s_mid, values, color=_COLORS[k % 4], linewidth=1.8,
                 marker=_MARKERS[k % 4], markersize=2.5, label=f"{name} [mm]")  # fmt: skip
        lo, hi = problem.bounds[name]
        lows.append(lo)
        highs.append(hi)
        style = {"color": "gray", "linewidth": 0.8, "linestyle": "--"}
        ax1.axhline(lo, label=f"limits {name}", **style)
        ax1.axhline(hi, **style)
    ax1.set_xlabel("Curvilinear abscissa s [mm]", fontsize=11)
    ax1.set_ylabel("Section dimension [mm]", fontsize=11)
    ax1.set_title("Section dimension distribution along the beam", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.set_xlim(s_mid[0], s_mid[-1])
    ax1.set_ylim(min(lows) - 0.3, max(highs) + 0.3)
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax1.grid(True, which="major", linestyle="-", alpha=0.35)
    ax1.grid(True, which="minor", linestyle=":", alpha=0.20)
    fig1.tight_layout()

    if save_path:
        path = f"{save_path}_sections.png"
        fig1.savefig(path, dpi=150)
        print(f"  Section plot saved: {path}")

    # ── Figure 2: σ_VM(s) ────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    util = sigma_vals / sigma_amm
    colors = np.where(util > 1.0, "#DC2626", np.where(util > 0.8, "#D97706", "#16A34A"))

    ax2.scatter(s_mid, sigma_vals, c=colors, s=18, zorder=3)
    ax2.plot(s_mid, sigma_vals, color="#6B7280", linewidth=1.2, zorder=2, label="σ_VM [MPa]")
    ax2.axhline(sigma_amm, color="#DC2626", linewidth=1.4, linestyle="--",
                label=f"σ_adm = {sigma_amm:.1f} MPa")  # fmt: skip
    ax2.axhline(0.8 * sigma_amm, color="#D97706", linewidth=0.9, linestyle=":",
                label=f"80% σ_adm = {0.8 * sigma_amm:.1f} MPa")  # fmt: skip
    ax2.set_xlabel("Curvilinear abscissa s [mm]", fontsize=11)
    ax2.set_ylabel("Equivalent stress σ_VM [MPa]", fontsize=11)
    ax2.set_title("Von Mises equivalent stress along the beam", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.set_xlim(s_mid[0], s_mid[-1])
    ax2.set_ylim(bottom=0)
    ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax2.grid(True, which="major", linestyle="-", alpha=0.35)
    ax2.grid(True, which="minor", linestyle=":", alpha=0.20)
    fig2.tight_layout()

    if save_path:
        path = f"{save_path}_stress.png"
        fig2.savefig(path, dpi=150)
        print(f"  Stress plot saved:  {path}")

    if show:
        plt.show()
