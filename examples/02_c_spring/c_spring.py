"""
Example 02 — C-shaped spring clip opened by an imposed displacement.

A strongly curved planar beam: a circular clip that is spread open by a given
amount when it is fitted, and must then grip with at least a minimum force
without yielding.  The load is a *displacement*, so the usual intuition is
reversed: a thicker clip is stiffer and therefore more stressed, a thinner
one grips less.  The radial thickness t(s) is optimised for minimum mass
under

    σ_VM ≤ σ_adm      F_grip ≥ F_min      F_grip ≤ F_max

Only half of the clip is modelled, with a symmetry plane at the back.

Run it (built-in solver, no ANSYS needed)::

    python examples/02_c_spring/c_spring.py --save-plots results/c_spring
    python examples/02_c_spring/c_spring.py --export-apdl results/c_spring.inp

Units: mm, N, ton, s, MPa.
"""

from __future__ import annotations

import argparse

import numpy as np

from beam_opt import (
    BeamPath,
    ImposedDisplacement,
    Material,
    NativeSolver,
    OptimizableBeam,
    OptimizationProblem,
    RectSection,
    StructuralModel,
    Symmetry,
    optimize,
)
from beam_opt.apdl import write_apdl
from beam_opt.geometry import discretize
from beam_opt.optimization import METHODS

# ── Problem data (invented values) ────────────────────────────────────────
RADIUS = 40.0  # radius of the clip centre line [mm]
TIP_ANGLE = 40.0  # angular position of the free tip; the back is at 180° [deg]
WIDTH = 10.0  # constant width, out of the plane of the clip [mm]
OPENING = 12.0  # total opening imposed between the two tips [mm]
F_MIN, F_MAX = 20.0, 60.0  # admissible grip force [N]
T_BOUNDS = (0.6, 4.0)  # technological limits on the radial thickness [mm]
SAFETY_FACTOR = 1.5
MATERIAL = Material(E=2.0e5, nu=0.3, rho=7.85e-9, sigma_y=390.0, name="steel")

# With the default BEAM188 orientation the local y axis of every element lies
# in the plane of the clip: the radial thickness is the section dimension "b"
# and the out-of-plane width is "h".
THICKNESS, CONSTANT = "b", {"h": WIDTH}


def build_model(n_elem: int = 60) -> StructuralModel:
    """Upper half of the clip, from the back (symmetry plane y = 0) to the tip."""
    theta = np.radians(np.linspace(180.0, TIP_ANGLE, 41))
    points = np.column_stack([RADIUS * np.cos(theta), RADIUS * np.sin(theta), np.zeros_like(theta)])
    mesh = discretize(BeamPath(points), n_elem)
    return StructuralModel(
        beam=OptimizableBeam(mesh, section_factory=RectSection.with_fixed(**CONSTANT)),
        material=MATERIAL,
        bcs=[
            Symmetry("start", normal="Y"),
            ImposedDisplacement("start", "UX"),  # no rigid translation along x
            ImposedDisplacement(["start", "end"], "UZ"),  # the clip stays in its plane
            ImposedDisplacement("end", "UY", OPENING / 2.0),  # half of the opening
        ],
    )


def make_problem(solver=None, n_elem: int = 60, n_ctrl: int = 8, history_file=None):
    """Minimum-mass problem; *solver* is a MAPDL session or (default) ``NativeSolver()``."""
    return OptimizationProblem(
        solver if solver is not None else NativeSolver(),
        build_model(n_elem),
        n_ctrl=n_ctrl,
        bounds={THICKNESS: T_BOUNDS},
        safety_factor=SAFETY_FACTOR,
        f_react_max=F_MAX,
        f_react_min=F_MIN,
        reaction_nodes=["end"],  # the force needed to hold the tip open
        interpolator="pchip",
        history_file=history_file,
    )


def run(problem: OptimizationProblem, method: str = "slsqp", max_evals: int = 3000, seed=None):
    """Optimise from the constant-thickness clip that grips with the minimum force."""
    x0 = np.full(problem.n_vars, uniform_design(problem)[0])
    return optimize(problem, method, x0=x0, max_evals=max_evals, seed=seed)


def uniform_design(problem: OptimizationProblem, n: int = 120) -> tuple[float, float, float]:
    """
    Thinnest clip of constant thickness that grips with at least ``F_MIN``.

    Returns
    -------
    thickness : float
        [mm]
    mass : float
        [g]
    utilisation : float
        σ_max / σ_adm [-]: above 1, no constant thickness is admissible.
    """
    for t in np.linspace(*T_BOUNDS, n):
        x = np.full(problem.n_vars, t)
        ev = problem.evaluate(x)
        if ev.reaction >= F_MIN:
            return float(t), problem.objective(x), float(ev.sigma.max() / problem.sigma_adm)
    raise RuntimeError("no constant thickness reaches the minimum grip force")


def summary(problem: OptimizationProblem, x_opt: np.ndarray) -> dict[str, float]:
    """Key figures of the optimised clip against the constant-thickness one."""
    ev = problem.evaluate(x_opt)
    t_uniform, m_uniform, u_uniform = uniform_design(problem)
    mass = problem.objective(x_opt)
    return {
        "mass": mass,
        "mass_uniform": m_uniform,
        "t_uniform": t_uniform,
        "utilisation_uniform": u_uniform,
        "saving": 1.0 - mass / m_uniform,
        "force": ev.reaction,
        "utilisation": float(ev.sigma.max() / problem.sigma_adm),
        "t_min": float(np.min(x_opt)),
        "t_max": float(np.max(x_opt)),
    }


def print_summary(s: dict[str, float]) -> None:
    print("\n[C-SPRING CLIP]")
    print(f"  Optimised mass (half clip) : {s['mass']:7.3f} g")
    verdict = "admissible" if s["utilisation_uniform"] <= 1.0 else "NOT admissible"
    print(
        f"  Constant thickness {s['t_uniform']:.2f} mm : {s['mass_uniform']:7.3f} g   "
        f"(stress at {s['utilisation_uniform']:.0%}: {verdict})"
    )
    print(f"  Mass saving                : {s['saving']:7.1%}")
    print(f"  Grip force                 : {s['force']:7.2f} N   (required {F_MIN:g} to {F_MAX:g})")
    print(f"  Stress utilisation         : {s['utilisation']:7.1%}")
    print(f"  Thickness range            : {s['t_min']:.2f} to {s['t_max']:.2f} mm")


def plot_design(problem, x_opt, save_path: str | None = None, show: bool = True) -> None:
    """The clip drawn with its (exaggerated) thickness, coloured by stress utilisation."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    mesh = problem.model.beam.mesh
    law = problem.law_from_x(x_opt)
    util = problem.evaluate(x_opt).sigma / problem.sigma_adm
    t_node = law.eval(mesh.s)[THICKNESS]
    radial = mesh.coords[:, :2] / np.linalg.norm(mesh.coords[:, :2], axis=1)[:, None]
    scale = 3.0  # thickness exaggeration in the drawing
    inner = mesh.coords[:, :2] - 0.5 * scale * t_node[:, None] * radial
    outer = mesh.coords[:, :2] + 0.5 * scale * t_node[:, None] * radial

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), width_ratios=[1.0, 1.25])
    for mirror in (1.0, -1.0):
        quads = [
            np.array([inner[i], inner[i + 1], outer[i + 1], outer[i]]) * [1.0, mirror]
            for i in range(mesh.n_elem)
        ]
        patches = PolyCollection(
            quads, array=util, cmap="viridis", clim=(0.0, 1.0), edgecolor="none"
        )
        ax1.add_collection(patches)
    ax1.annotate(
        "", xy=(mesh.coords[-1, 0], mesh.coords[-1, 1] + 14), xytext=tuple(mesh.coords[-1, :2]),
        arrowprops={"arrowstyle": "->", "color": "#DC2626", "lw": 1.6},
    )  # fmt: skip
    ax1.annotate(
        "", xy=(mesh.coords[-1, 0], -mesh.coords[-1, 1] - 14),
        xytext=(mesh.coords[-1, 0], -mesh.coords[-1, 1]),
        arrowprops={"arrowstyle": "->", "color": "#DC2626", "lw": 1.6},
    )  # fmt: skip
    ax1.set_aspect("equal")
    ax1.set_xlim(-RADIUS - 12, RADIUS + 12)
    ax1.set_ylim(-RADIUS - 18, RADIUS + 18)
    ax1.set_xlabel("x [mm]")
    ax1.set_ylabel("y [mm]")
    ax1.set_title(f"Optimised clip (thickness ×{scale:g}), imposed opening")
    fig.colorbar(patches, ax=ax1, label="σ_VM / σ_adm [-]", shrink=0.85)

    s = np.linspace(0.0, mesh.s[-1], 300)
    ax2.plot(s, law.eval(s)[THICKNESS], color="#2563EB", linewidth=1.8, label="optimised t(s)")
    ax2.plot(problem.s_ctrl, law.ctrl[THICKNESS], "o", color="#2563EB", label="control points")
    ax2.axhline(uniform_design(problem)[0], color="k", linestyle="--", linewidth=1.2,
                label="constant thickness, same grip")  # fmt: skip
    ax2.set_xlabel("Curvilinear abscissa s from the back [mm]")
    ax2.set_ylabel("Radial thickness t [mm]")
    ax2.set_ylim(bottom=0.0)
    ax2.grid(alpha=0.35)
    ax2.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(f"{save_path}_design.png", dpi=150)
        print(f"  Design plot saved: {save_path}_design.png")
    if show:
        plt.show()


def main() -> None:
    from beam_opt.report import print_report

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n-elem", type=int, default=60, help="number of beam elements")
    parser.add_argument("--n-ctrl", type=int, default=8, help="section-law control points")
    parser.add_argument("--method", default="slsqp", choices=sorted(METHODS), help="algorithm")
    parser.add_argument("--max-evals", type=int, default=3000, help="budget of FEM solutions")
    parser.add_argument("--solver", default="native", choices=["native", "mapdl"])
    parser.add_argument("--save-plots", default=None, help="path prefix for the PNG output")
    parser.add_argument("--no-plots", action="store_true", help="do not open plot windows")
    parser.add_argument("--export-apdl", default=None, help="write the optimum as an APDL script")
    args = parser.parse_args()

    def solve_and_report(session) -> None:
        problem = make_problem(session, args.n_elem, args.n_ctrl)
        result = run(problem, args.method, args.max_evals)
        print_report(result, problem)
        print_summary(summary(problem, result.x))
        if args.export_apdl:
            write_apdl(
                problem.model, problem.law_from_x(result.x), args.export_apdl, "C-spring clip"
            )
            print(f"  APDL script written: {args.export_apdl}")
        if args.save_plots or not args.no_plots:
            plot_design(problem, result.x, args.save_plots, show=not args.no_plots)

    if args.solver == "native":
        solve_and_report(NativeSolver())
    else:
        from beam_opt import MapdlSession

        with MapdlSession(jobname="c_spring") as mapdl:
            solve_and_report(mapdl)


if __name__ == "__main__":
    main()
