"""
Example 01 — minimum-mass cantilever with a tip load.

A straight cantilever of constant width *b* carries a transverse tip load *P*;
the height *h(s)* is optimised for minimum mass under the stress constraint
σ ≤ σ_adm.  The problem has a closed-form solution (beam of uniform
strength), so it is the reference check of the whole framework:

    h*(s) = sqrt(6 P (L − s) / (b σ_adm))        parabolic profile
    m*    = 2/3 · ρ b h*(0) L                    2/3 of the prismatic beam

(with the technological lower bound ``h ≥ h_min`` active near the tip).

Run it with a licensed MAPDL::

    python examples/01_cantilever/cantilever.py --save-plots results/cantilever
    python examples/01_cantilever/cantilever.py --method slsqp

Units: mm, N, ton, s, MPa.
"""

from __future__ import annotations

import argparse

import numpy as np

from beam_opt import (
    BeamPath,
    Clamp,
    Material,
    NodalForce,
    OptimizableBeam,
    OptimizationProblem,
    StructuralModel,
)
from beam_opt.geometry import discretize
from beam_opt.optimization import METHODS, optimize
from beam_opt.optimization.problem import TON_TO_G
from beam_opt.sections import RectSection
from beam_opt.solver import SolverSettings

# ── Problem data (invented values) ────────────────────────────────────────
LENGTH = 200.0  # beam length [mm]
WIDTH = 5.0  # constant section width b [mm]
LOAD = 20.0  # tip load, along −Z [N]
H_BOUNDS = (1.0, 8.0)  # technological limits on the height h [mm]
SAFETY_FACTOR = 1.5  # [-]
MATERIAL = Material(E=2.0e5, nu=0.3, rho=7.85e-9, sigma_y=355.0)  # structural steel
SIGMA_ADM = MATERIAL.sigma_y / SAFETY_FACTOR  # admissible stress [MPa]


# ── Model ─────────────────────────────────────────────────────────────────


def fixed_width_section(params) -> RectSection:
    """Rectangle with the constant width ``WIDTH``: only *h* is a design parameter."""
    return RectSection(WIDTH, params["h"])


def build_model(n_elem: int = 50) -> StructuralModel:
    """
    Cantilever along global X, clamped at ``"start"``, loaded at ``"end"``.

    The load acts along −Z: with the default BEAM188 orientation the section
    height *h* (local z) is the dimension in the bending plane.
    """
    path = BeamPath([[0.0, 0.0, 0.0], [LENGTH, 0.0, 0.0]], kind="polyline")
    mesh = discretize(path, n_elem)
    return StructuralModel(
        beam=OptimizableBeam(mesh, section_factory=fixed_width_section),
        material=MATERIAL,
        bcs=[Clamp("start"), NodalForce("end", "FZ", -LOAD)],
    )


def make_problem(
    mapdl, n_elem: int = 50, n_ctrl: int = 8, history_file: str | None = None
) -> OptimizationProblem:
    """Minimum-mass problem on an injected MAPDL session."""
    return OptimizationProblem(
        mapdl,
        build_model(n_elem),
        n_ctrl=n_ctrl,
        bounds={"h": H_BOUNDS},
        safety_factor=SAFETY_FACTOR,
        f_react_max=5.0 * LOAD,  # never active: the reaction equals the load
        reaction_nodes=["start"],
        settings=SolverSettings(jobname="cantilever", nlgeom=False),  # linear, as the theory
        interpolator="pchip",  # no overshoot between control points
        history_file=history_file,
    )


def run(problem: OptimizationProblem, method: str = "cobyla", max_evals: int = 2000, seed=None):
    """Optimise from the upper bound of *h* (a feasible prismatic beam)."""
    return optimize(problem, method, x0=problem.upper, max_evals=max_evals, seed=seed)


# ── Closed-form reference ─────────────────────────────────────────────────


def analytic_height(s) -> np.ndarray:
    """
    Height of the beam of uniform strength at abscissa *s* [mm].

    Bending only (the shear stress is negligible for this slender beam),
    limited by ``H_BOUNDS``.
    """
    s = np.asarray(s, dtype=float)
    h = np.sqrt(6.0 * LOAD * np.clip(LENGTH - s, 0.0, None) / (WIDTH * SIGMA_ADM))
    return np.clip(h, *H_BOUNDS)


def analytic_mass(n: int = 200_001) -> float:
    """Mass of the beam of uniform strength [g] (trapezoidal quadrature of ρ b h*(s))."""
    s = np.linspace(0.0, LENGTH, n)
    h = analytic_height(s)
    integral = np.sum(0.5 * (h[1:] + h[:-1]) * np.diff(s))
    return float(MATERIAL.rho * WIDTH * integral * TON_TO_G)


def prismatic_mass() -> float:
    """Mass of the lightest admissible constant-height beam, h = h*(0) [g]."""
    return float(MATERIAL.rho * WIDTH * analytic_height(0.0) * LENGTH * TON_TO_G)


def comparison(problem: OptimizationProblem, x_opt: np.ndarray) -> dict[str, float]:
    """
    Optimised design against the closed-form solution.

    Returns
    -------
    dict[str, float]
        ``mass`` and ``mass_ref`` [g], ``mass_error`` [-], ``mass_prismatic``
        [g], ``saving`` with respect to the prismatic beam [-],
        ``h_error_max`` [mm] at the element mid-points and ``utilisation``
        (σ_max / σ_adm) [-].
    """
    ev = problem.evaluate(x_opt)
    s_mid = problem.model.beam.mesh.s_elem
    h = problem.law_from_x(x_opt).eval(s_mid)["h"]
    mass = ev.mass * TON_TO_G
    ref = analytic_mass()
    return {
        "mass": mass,
        "mass_ref": ref,
        "mass_error": mass / ref - 1.0,
        "mass_prismatic": prismatic_mass(),
        "saving": 1.0 - mass / prismatic_mass(),
        "h_error_max": float(np.max(np.abs(h - analytic_height(s_mid)))),
        "utilisation": float(ev.sigma.max() / problem.sigma_adm),
    }


def print_comparison(c: dict[str, float]) -> None:
    """Print the table returned by ``comparison``."""
    print("\n[COMPARISON WITH THE BEAM OF UNIFORM STRENGTH]")
    print(f"  Optimised mass        : {c['mass']:8.3f} g")
    print(f"  Closed-form mass      : {c['mass_ref']:8.3f} g   ({c['mass_error']:+.2%})")
    print(f"  Prismatic beam        : {c['mass_prismatic']:8.3f} g   (saving {c['saving']:.1%})")
    print(f"  Max |h − h*|          : {c['h_error_max']:8.3f} mm")
    print(f"  Stress utilisation    : {c['utilisation']:8.1%}")


def plot_comparison(problem, x_opt, save_path: str | None = None, show: bool = True) -> None:
    """Optimised h(s) and stress utilisation against the closed-form profile."""
    import matplotlib.pyplot as plt

    mesh = problem.model.beam.mesh
    law = problem.law_from_x(x_opt)
    s = np.linspace(0.0, LENGTH, 400)
    sigma = problem.evaluate(x_opt).sigma

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax1.plot(s, analytic_height(s), "k--", linewidth=1.4, label="uniform strength (closed form)")
    ax1.plot(s, law.eval(s)["h"], color="#2563EB", linewidth=1.8, label="optimised h(s)")
    ax1.plot(problem.s_ctrl, law.ctrl["h"], "o", color="#2563EB", label="control points")
    ax1.set_ylabel("Section height h [mm]")
    ax1.set_ylim(bottom=0.0)
    ax1.legend()
    ax1.grid(alpha=0.35)

    ax2.plot(mesh.s_elem, sigma / problem.sigma_adm, color="#DC2626", linewidth=1.6)
    ax2.axhline(1.0, color="gray", linestyle="--", linewidth=0.9)
    ax2.set_xlabel("Curvilinear abscissa s [mm]")
    ax2.set_ylabel("σ_VM / σ_adm [-]")
    ax2.set_ylim(0.0, 1.1)
    ax2.grid(alpha=0.35)
    fig.suptitle("Minimum-mass cantilever with a tip load")
    fig.tight_layout()

    if save_path:
        fig.savefig(f"{save_path}_comparison.png", dpi=150)
        print(f"  Comparison plot saved: {save_path}_comparison.png")
    if show:
        plt.show()


# ── Entry point ───────────────────────────────────────────────────────────


def main() -> None:
    from beam_opt import MapdlSession
    from beam_opt.report import print_report

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n-elem", type=int, default=50, help="number of beam elements")
    parser.add_argument("--n-ctrl", type=int, default=8, help="section-law control points")
    parser.add_argument("--nproc", type=int, default=2, help="MAPDL processors")
    parser.add_argument("--method", default="cobyla", choices=sorted(METHODS), help="algorithm")
    parser.add_argument("--max-evals", type=int, default=2000, help="budget of FEM solutions")
    parser.add_argument("--seed", type=int, default=None, help="seed of the evolutionary method")
    parser.add_argument("--history", default=None, help="CSV log of every FEM evaluation")
    parser.add_argument("--run-location", default="mapdl_tmp", help="MAPDL working directory")
    parser.add_argument("--save-plots", default=None, help="path prefix for the PNG output")
    parser.add_argument("--no-plots", action="store_true", help="do not open plot windows")
    args = parser.parse_args()

    with MapdlSession(args.run_location, nproc=args.nproc, jobname="cantilever") as mapdl:
        problem = make_problem(mapdl, args.n_elem, args.n_ctrl, args.history)
        result = run(problem, args.method, args.max_evals, args.seed)
        print_report(result, problem)
        print_comparison(comparison(problem, result.x))
        if args.save_plots or not args.no_plots:
            plot_comparison(problem, result.x, args.save_plots, show=not args.no_plots)


if __name__ == "__main__":
    main()
