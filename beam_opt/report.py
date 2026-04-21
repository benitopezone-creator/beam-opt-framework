"""
beam_opt.report
===============
Textual optimisation report and matplotlib result plots.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


# -----------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------

def plot_results(
    optimizer,
    x_opt: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """
    Generate two matplotlib figures:
      1. Section distributions b(s) and h(s) along the spline.
      2. Von Mises equivalent stress σ_VM(s).

    Parameters
    ----------
    optimizer : BeamOptimizer
    x_opt     : optimised design vector
    save_path : path prefix for PNG output (e.g. ``"results/v05"``).
                Saves ``<prefix>_sections.png`` and ``<prefix>_stress.png``.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("  [WARNING] matplotlib not available – plots skipped.")
        return

    law = optimizer._law_from_x(x_opt)
    s_mid = optimizer.model.s_elem
    b_vals, h_vals = law.eval(s_mid)
    sigma_vals = optimizer._cache_sigma
    sigma_amm = optimizer.sigma_s / optimizer.fs

    b_lo, b_hi = optimizer.b_bounds

    # ── Figure 1: b(s) and h(s) ──────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(s_mid, b_vals, color="#2563EB", linewidth=1.8,
             marker="o", markersize=2.5, label="b [mm]")
    ax1.plot(s_mid, h_vals, color="#DC2626", linewidth=1.8,
             marker="s", markersize=2.5, label="h [mm]")
    ax1.axhline(b_lo, color="gray", linewidth=0.8, linestyle="--", label="limits b,h")
    ax1.axhline(b_hi, color="gray", linewidth=0.8, linestyle="--")
    ax1.set_xlabel("Curvilinear abscissa s [mm]", fontsize=11)
    ax1.set_ylabel("Section dimension [mm]", fontsize=11)
    ax1.set_title("Section dimension distribution along the spline", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.set_xlim(s_mid[0], s_mid[-1])
    ax1.set_ylim(b_lo - 0.3, b_hi + 0.3)
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
    colors = np.where(util > 1.0, "#DC2626",
              np.where(util > 0.8, "#D97706", "#16A34A"))

    ax2.scatter(s_mid, sigma_vals, c=colors, s=18, zorder=3)
    ax2.plot(s_mid, sigma_vals, color="#6B7280", linewidth=1.2, zorder=2,
             label="σ_VM [MPa]")
    ax2.axhline(sigma_amm, color="#DC2626", linewidth=1.4, linestyle="--",
                label=f"σ_adm = {sigma_amm:.1f} MPa")
    ax2.axhline(0.8 * sigma_amm, color="#D97706", linewidth=0.9, linestyle=":",
                label=f"80% σ_adm = {0.8 * sigma_amm:.1f} MPa")
    ax2.set_xlabel("Curvilinear abscissa s [mm]", fontsize=11)
    ax2.set_ylabel("Equivalent stress σ_VM [MPa]", fontsize=11)
    ax2.set_title("Von Mises equivalent stress along the spline", fontsize=12)
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

    plt.show()


# -----------------------------------------------------------------------
# Textual report
# -----------------------------------------------------------------------

def print_report(
    result,
    optimizer,
    show_plots: bool = True,
    save_plots_path: Optional[str] = None,
) -> None:
    """
    Print a full optimisation report and (optionally) show/save plots.

    Parameters
    ----------
    result          : ``scipy.optimize.OptimizeResult``
    optimizer       : ``BeamOptimizer`` instance
    show_plots      : if ``True``, call ``plot_results()``
    save_plots_path : path prefix for plot PNG files (optional)
    """
    SEP  = "=" * 80
    SEP2 = "-" * 80

    sigma_amm = optimizer.sigma_s / optimizer.fs

    print()
    print(SEP)
    print(" " * 20 + "BEAM-OPT FRAMEWORK – OPTIMISATION REPORT")
    print(SEP)

    # ── Convergence ──
    print("\n[CONVERGENCE]")
    status = "✓ CONVERGED" if result.success else "✗ NOT CONVERGED"
    print(f"  {status}")
    if not result.success:
        print(f"  Reason : {result.message}")
    print(f"  FEM evaluations : {result.nfev}")

    # ── Reconstruct solution ──
    x_opt = result.x
    optimizer._solve_if_needed(x_opt)

    # ── Mass ──
    print("\n[STRUCTURAL MASS]")
    m_opt   = result.fun
    m_fixed = optimizer.mass_fixed * 1e6
    m_total = m_opt + m_fixed
    print(f"  Optimisable elements : {m_opt:8.3f} g")
    print(f"  Fixed elements       : {m_fixed:8.3f} g")
    print(f"  {'─'*35}")
    print(f"  TOTAL                : {m_total:8.3f} g")

    # ── Stress ──
    sigma_max = float(optimizer._cache_sigma.max())
    print("\n[STRESS VERIFICATION]")
    print(f"  σ_max    : {sigma_max:7.2f} MPa")
    print(f"  σ_adm    : {sigma_amm:7.2f} MPa  "
          f"(σ_s={optimizer.sigma_s} / FS={optimizer.fs})")
    print(f"  Utilisation : {sigma_max / sigma_amm * 100:6.1f} %")
    print(f"  {'✓ OK' if sigma_max <= sigma_amm else '✗ VIOLATED'}")

    # ── Reaction ──
    f_react = float(optimizer._cache_react)
    print("\n[REACTION FORCE VERIFICATION]")
    print(f"  F_react  : {f_react:7.4f} N")
    print(f"  F_max    : {optimizer.f_react_max:7.4f} N")
    print(f"  Utilisation : {f_react / optimizer.f_react_max * 100:6.1f} %")
    print(f"  {'✓ OK' if f_react <= optimizer.f_react_max else '✗ VIOLATED'}")

    # ── Element table ──
    law = optimizer._law_from_x(x_opt)
    b_vals, h_vals = law.eval(optimizer.model.s_elem)
    sigma_vals = optimizer._cache_sigma
    s_mids = optimizer.model.s_elem

    print()
    print(SEP2)
    print("  ELEMENT-WISE SECTION DISTRIBUTION AND STRESS")
    print(SEP2)
    print(
        f"  {'Elem':>4}  {'s [mm]':>8}  {'b [mm]':>7}  {'h [mm]':>7}  "
        f"{'A [mm²]':>8}  {'σ_VM [MPa]':>11}  {'util. [%]':>9}  Status"
    )
    print("  " + "-" * 72)

    for i, (s, b, h, sv) in enumerate(zip(s_mids, b_vals, h_vals, sigma_vals)):
        A  = b * h
        ut = sv / sigma_amm * 100.0
        ok = "✓" if sv <= sigma_amm else "✗"
        print(
            f"  {i+1:>4}  {s:8.1f}  {b:7.3f}  {h:7.3f}  "
            f"{A:8.3f}  {sv:11.2f}  {ut:9.1f}  {ok}"
        )

    print("  " + "-" * 72)

    # ── Summary statistics ──
    print()
    print("  [SECTION STATISTICS]")
    print(f"    b  — min: {b_vals.min():.3f}  max: {b_vals.max():.3f}  "
          f"mean: {b_vals.mean():.3f}  idx_max: {b_vals.argmax()+1}")
    print(f"    h  — min: {h_vals.min():.3f}  max: {h_vals.max():.3f}  "
          f"mean: {h_vals.mean():.3f}  idx_max: {h_vals.argmax()+1}")
    print(f"    σ  — min: {sigma_vals.min():.2f}  max: {sigma_vals.max():.2f}  "
          f"idx_max: {sigma_vals.argmax()+1}")
    idx_c = int(sigma_vals.argmax())
    print(
        f"\n    Critical element: #{idx_c+1}  "
        f"(s={s_mids[idx_c]:.1f} mm, "
        f"b={b_vals[idx_c]:.3f} mm, "
        f"h={h_vals[idx_c]:.3f} mm, "
        f"σ={sigma_vals[idx_c]:.2f} MPa)"
    )

    print()
    print(SEP)

    # ── Plots ──
    if show_plots:
        plot_results(optimizer, x_opt, save_path=save_plots_path)
