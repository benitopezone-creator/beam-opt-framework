"""
Example 05 — stiffness against mass for the helical tube beam.

The helix of example 03 is now designed for **stiffness**: minimum
compliance (the work of the loads, C = Σ F·u) under a mass budget, with the
von Mises stress constraint kept.  Sweeping the budget traces the Pareto
front between mass and stiffness (epsilon-constraint method); every point
is compared with the tube of constant radius of the same mass.

Run it (built-in solver, no ANSYS needed)::

    python examples/05_stiffness_pareto/stiffness_pareto.py --save-plots results/stiffness

Units: mm, N, ton, s, MPa.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np

from beam_opt import (
    Compliance,
    Mass,
    NativeSolver,
    OptimizationProblem,
    epsilon_constraint,
    optimize,
)

_spec = importlib.util.spec_from_file_location(
    "helix", Path(__file__).parents[1] / "03_helix" / "helix.py"
)
helix = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(helix)

BUDGET_FACTORS = [1.0, 1.15, 1.35, 1.6, 2.0, 2.4]  # mass budgets / lightest admissible mass


def make_problem(objective=None, solver=None, n_elem: int = 60, n_ctrl: int = 8, mass_max=None):
    """Helix with the stress constraint of example 03 and no deflection limit."""
    return OptimizationProblem(
        solver if solver is not None else NativeSolver(),
        helix.build_model(n_elem),
        n_ctrl=n_ctrl,
        bounds={"r": helix.R_BOUNDS},
        safety_factor=helix.SAFETY_FACTOR,
        objective=objective,
        mass_max=mass_max,
        interpolator="pchip",
    )


def lightest(n_elem: int = 60, n_ctrl: int = 8):
    """Minimum-mass helix under the stress constraint only: the start of the front."""
    problem = make_problem(Mass(), n_elem=n_elem, n_ctrl=n_ctrl)
    return problem, optimize(problem, "slsqp", x0=problem.upper)


def constant_tube(problem: OptimizationProblem, mass: float) -> dict:
    """Tube of constant radius with the given mass [g]: radius, compliance, stress check."""
    lo, hi = helix.R_BOUNDS
    for _ in range(60):
        r = 0.5 * (lo + hi)
        lo, hi = (r, hi) if problem.mass_g(np.full(problem.n_vars, r)) < mass else (lo, r)
    x = np.full(problem.n_vars, 0.5 * (lo + hi))
    ev = problem.evaluate(x)
    return {
        "r": float(x[0]),
        "compliance": Compliance().value(problem, ev),
        "stress_ok": bool(ev.sigma.max() <= problem.sigma_adm * (1 + 1e-4)),
    }


def trade_off(n_elem: int = 60, n_ctrl: int = 8, factors=BUDGET_FACTORS) -> tuple:
    """Pareto front mass–compliance and the constant-radius tubes of equal mass."""
    _, light = lightest(n_elem, n_ctrl)
    problem = make_problem(Compliance(), n_elem=n_elem, n_ctrl=n_ctrl)
    m_min = problem.mass_g(light.x)
    problem.mass_max = m_min
    levels = [f * m_min for f in factors]
    front = epsilon_constraint(problem, "mass_max", levels, x0=light.x)
    rows = []
    for point in front:
        tube = constant_tube(problem, point.mass)
        rows.append(
            {
                "budget": point.level,
                "mass": point.mass,
                "compliance": point.objective,
                "feasible": point.feasible,
                "x": point.x,
                "tube": tube,
                "gain": 1.0 - point.objective / tube["compliance"],
            }
        )
    return problem, rows


def print_trade_off(rows: list[dict]) -> None:
    print("\n[STIFFNESS OF THE HELICAL TUBE]  (compliance C = work of the loads)")
    print(f"  {'mass [g]':>9} {'C opt [N·mm]':>13} {'C const [N·mm]':>15} {'r const':>8}"
          f" {'const stress':>13} {'stiffer by':>11}")  # fmt: skip
    for r in rows:
        ok = "ok" if r["tube"]["stress_ok"] else "VIOLATED"
        print(
            f"  {r['mass']:9.1f} {r['compliance']:13.1f} {r['tube']['compliance']:15.1f}"
            f" {r['tube']['r']:8.2f} {ok:>13} {r['gain']:10.1%}"
        )


def plot_trade_off(problem, rows, save_path: str | None = None, show: bool = True) -> None:
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    mass = [r["mass"] for r in rows]
    ax1.plot(mass, [r["compliance"] for r in rows], "o-", color="#2563EB", lw=1.8,
             label="optimised radius (Pareto front)")  # fmt: skip
    tube_c = [r["tube"]["compliance"] for r in rows]
    ax1.plot(mass, tube_c, "--", color="k", lw=1.2, label="constant radius")
    bad = [i for i, r in enumerate(rows) if not r["tube"]["stress_ok"]]
    ax1.plot([mass[i] for i in bad], [tube_c[i] for i in bad], "x", color="#DC2626", ms=8,
             label="constant radius: stress violated")  # fmt: skip
    ax1.set_yscale("log")
    ax1.set_xlabel("Mass of the beam [g]")
    ax1.set_ylabel("Compliance C = Σ F·u [N·mm]  (lower = stiffer)")
    ax1.set_title("Stiffness against mass")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=9)

    mesh = problem.model.beam.mesh
    s = np.linspace(0.0, mesh.s[-1], 400)
    for color, r in zip(["#93C5FD", "#2563EB", "#1E3A8A"], [rows[0], rows[2], rows[-1]]):
        ax2.plot(s, problem.law_from_x(r["x"]).eval(s)["r"], color=color, lw=1.8,
                 label=f"{r['mass']:.0f} g")  # fmt: skip
    ax2.axvline(mesh.s[mesh.anchors["prop"]], color="#6B7280", ls=":", label="strut")
    ax2.set_xlabel("Curvilinear abscissa s from the clamp [mm]")
    ax2.set_ylabel("Outer radius r [mm]")
    ax2.set_ylim(0.0, helix.R_BOUNDS[1] * 1.08)
    ax2.set_title("Stiffest tube for each mass")
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

    problem, rows = trade_off(args.n_elem, args.n_ctrl)
    print_trade_off(rows)
    if args.save_plots or not args.no_plots:
        plot_trade_off(problem, rows, args.save_plots, show=not args.no_plots)


if __name__ == "__main__":
    main()
