"""
Example 06 — low-cycle fatigue of a snap clip: grip force against life.

The C-shaped clip of example 02 is opened and released at every use
(zero-to-peak imposed displacement, R = 0) and must still grip the part it
holds.  The opening is large, so the clip yields locally: the design
criterion is no longer "stress below yield" but **the number of openings
before a crack starts**, from the strain-life curve of the material (local
strain approach, Neuber's rule, Smith-Watson-Topper mean-stress correction).

Under an imposed displacement a stiffer clip grips more and lives less:
the two goals conflict.  The example traces the trade-off, life against
grip force, for the optimised clip and for the clip of constant thickness,
by the epsilon-constraint method (maximum life for each minimum grip).

Run it (built-in solver, no ANSYS needed)::

    python examples/06_lcf_clip/lcf_clip.py --save-plots results/lcf_clip

Units: mm, N, ton, s, MPa.  The strain-life constants are typical of a
medium-carbon steel and are illustrative only.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np

from beam_opt import (
    FatigueLife,
    FatigueModel,
    NativeSolver,
    OptimizationProblem,
    StrainLifeCurve,
    epsilon_constraint,
)

# The geometry, supports and material of example 02
_spec = importlib.util.spec_from_file_location(
    "c_spring", Path(__file__).parents[1] / "02_c_spring" / "c_spring.py"
)
clip = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(clip)

OPENING = 36.0  # imposed opening at every use [mm]
T_BOUNDS = (0.3, 6.0)  # radial thickness [mm]
GRIPS = [10.0, 20.0, 40.0, 60.0, 90.0, 120.0]  # required grip forces of the sweep [N]
CURVE = StrainLifeCurve(E=2.0e5, sigma_f=930.0, b=-0.095, eps_f=0.26, c=-0.47, K=1258.0, n=0.208)
CYCLE = FatigueModel(CURVE, load_ratio=0.0, mean_stress="swt")


def make_problem(solver=None, n_elem: int = 60, n_ctrl: int = 8) -> OptimizationProblem:
    """Maximum fatigue life with a minimum grip force; local yielding accepted."""
    return OptimizationProblem(
        solver if solver is not None else NativeSolver(),
        clip.build_model(n_elem, opening=OPENING),
        n_ctrl=n_ctrl,
        bounds={clip.THICKNESS: T_BOUNDS},
        safety_factor=None,  # no static stress limit: fatigue governs
        objective=FatigueLife(),
        fatigue=CYCLE,
        reaction_nodes=["end"],
        f_react_min=GRIPS[0],
        interpolator="pchip",
    )


def constant_clip(problem: OptimizationProblem, grip: float) -> tuple[float, float]:
    """Constant thickness [mm] that grips with exactly *grip* [N], and its life [cycles]."""
    lo, hi = T_BOUNDS
    for _ in range(60):
        t = 0.5 * (lo + hi)
        lo, hi = (
            (lo, t) if problem.evaluate(np.full(problem.n_vars, t)).reaction > grip else (t, hi)
        )
    t = 0.5 * (lo + hi)
    return t, float(problem.lives(problem.evaluate(np.full(problem.n_vars, t))).min())


def trade_off(problem: OptimizationProblem, grips=GRIPS, max_evals: int = 3000) -> list[dict]:
    """Optimised and constant-thickness clips for every required grip force."""
    front = epsilon_constraint(
        problem, "f_react_min", grips, x0=np.full(problem.n_vars, 1.0), max_evals=max_evals
    )
    rows = []
    for grip, point in zip(grips, front):
        ev = problem.evaluate(point.x)
        t_const, life_const = constant_clip(problem, grip)
        cycle = CYCLE.local_cycle(ev.sigma.max())
        rows.append(
            {
                "grip": grip,
                "x": point.x,
                "feasible": point.feasible,
                "life": float(problem.lives(ev).min()),
                "force": float(ev.reaction),
                "eps_a": float(cycle["eps_a"]),
                "plastic": float(cycle["eps_a"] - cycle["sigma_a"] / CURVE.E),
                "mass": point.mass,
                "t_const": t_const,
                "life_const": life_const,
                "mass_const": problem.mass_g(np.full(problem.n_vars, t_const)),
            }
        )
    return rows


def print_trade_off(rows: list[dict]) -> None:
    print(f"\n[LOW-CYCLE FATIGUE OF THE CLIP]  opening {OPENING:g} mm, R = {CYCLE.load_ratio:g}")
    print(f"  {'grip [N]':>8} {'optimised life':>15} {'ε_a [%]':>8} {'ε_p,a [%]':>9} {'mass [g]':>9}"
          f" {'constant life':>14} {'t [mm]':>7} {'gain':>6}")  # fmt: skip
    for r in rows:
        print(
            f"  {r['grip']:8.0f} {r['life']:15.3g} {100 * r['eps_a']:8.3f} {100 * r['plastic']:9.3f}"
            f" {r['mass']:9.2f} {r['life_const']:14.3g} {r['t_const']:7.2f}"
            f" {r['life'] / r['life_const']:5.1f}×"
        )


def plot_trade_off(problem, rows, save_path: str | None = None, show: bool = True) -> None:
    """Life against grip force, and the thickness of three optimised clips."""
    import matplotlib.pyplot as plt

    grips = [r["grip"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax1.semilogy(grips, [r["life"] for r in rows], "o-", color="#2563EB", lw=1.8,
                 label="optimised thickness")  # fmt: skip
    ax1.semilogy(grips, [r["life_const"] for r in rows], "s--", color="k", lw=1.2,
                 label="constant thickness")  # fmt: skip
    ax1.axhspan(1.0, 1e5, color="#D97706", alpha=0.08, label="low-cycle range (< 10⁵)")
    ax1.set_ylim(min(r["life_const"] for r in rows) / 3.0, max(r["life"] for r in rows) * 3.0)
    ax1.set_xlabel("Required grip force [N]")
    ax1.set_ylabel("Openings to crack initiation [cycles]")
    ax1.set_title(f"Grip against life (opening {OPENING:g} mm, R = 0)")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend()

    mesh = problem.model.beam.mesh
    s = np.linspace(0.0, mesh.s[-1], 300)
    colors = ["#93C5FD", "#2563EB", "#1E3A8A"]
    for color, r in zip(colors, [rows[1], rows[3], rows[-1]]):
        law = problem.law_from_x(r["x"])
        ax2.plot(s, law.eval(s)[clip.THICKNESS], color=color, lw=1.8,
                 label=f"{r['grip']:g} N: {r['life']:.2g} cycles")  # fmt: skip
        ax2.axhline(r["t_const"], color=color, ls="--", lw=1.0)
    ax2.set_xlabel("Curvilinear abscissa s from the back [mm]")
    ax2.set_ylabel("Radial thickness t [mm]")
    ax2.set_title("Optimised thickness (dashed: constant, same grip)")
    ax2.set_ylim(bottom=0.0)
    ax2.grid(alpha=0.35)
    ax2.legend(fontsize=9)
    fig.tight_layout()
    if save_path:
        fig.savefig(f"{save_path}_trade_off.png", dpi=150)
        print(f"  Trade-off plot saved: {save_path}_trade_off.png")
    if show:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n-elem", type=int, default=60)
    parser.add_argument("--n-ctrl", type=int, default=8)
    parser.add_argument("--save-plots", default=None, help="path prefix for the PNG output")
    parser.add_argument("--no-plots", action="store_true", help="do not open plot windows")
    args = parser.parse_args()

    problem = make_problem(n_elem=args.n_elem, n_ctrl=args.n_ctrl)
    rows = trade_off(problem)
    print_trade_off(rows)
    if args.save_plots or not args.no_plots:
        plot_trade_off(problem, rows, args.save_plots, show=not args.no_plots)


if __name__ == "__main__":
    main()
