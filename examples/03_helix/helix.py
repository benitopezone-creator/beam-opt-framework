"""
Example 03 — helical tube beam in space, propped by a strut.

A beam that follows a helix: every element works in bending about two axes,
in torsion and in shear at the same time, which is where a curved 3-D beam
differs most from a straight one.  The beam is a thin tube clamped at its
base, carries a load at the free end and is propped half-way by a fixed
strut, so the structure is statically indeterminate: changing the sections
changes how the load is shared.

The outer radius r(s) of the tube is optimised for minimum mass under

    σ_VM ≤ σ_adm                (bending + torsion + shear)
    |UZ| at the free end ≤ limit (stiffness)

Run it (built-in solver, no ANSYS needed)::

    python examples/03_helix/helix.py --save-plots results/helix
    python examples/03_helix/helix.py --method differential-evolution --max-evals 6000

Units: mm, N, ton, s, MPa.
"""

from __future__ import annotations

import argparse

import numpy as np

from beam_opt import (
    BeamPath,
    Clamp,
    FixedBeam,
    Material,
    NativeSolver,
    NodalForce,
    OptimizableBeam,
    OptimizationProblem,
    StructuralModel,
    TubeSection,
    optimize,
)
from beam_opt.apdl import write_apdl
from beam_opt.geometry import discretize
from beam_opt.optimization import METHODS

# ── Problem data (invented values) ────────────────────────────────────────
HELIX_RADIUS = 150.0  # radius of the helix [mm]
PITCH = 120.0  # rise per turn [mm]
TURNS = 1.25  # number of turns
WALL = 1.5  # constant wall thickness of the tube [mm]
R_BOUNDS = (4.0, 16.0)  # technological limits on the outer radius [mm]
LOAD = (0.0, 60.0, -250.0)  # force at the free end (FX, FY, FZ) [N]
DEFLECTION_LIMIT = 35.0  # admissible vertical deflection of the free end [mm]
STRUT_TURN = 0.75  # the strut meets the helix after this many turns
STRUT_RADIUS, STRUT_WALL = 8.0, 1.5  # strut tube [mm]
SAFETY_FACTOR = 1.5
MATERIAL = Material(E=7.0e4, nu=0.33, rho=2.70e-9, sigma_y=240.0, name="aluminium alloy")


def helix_point(turn: float) -> np.ndarray:
    """Point of the centre line after *turn* turns [mm]."""
    angle = 2.0 * np.pi * turn
    return np.array([HELIX_RADIUS * np.cos(angle), HELIX_RADIUS * np.sin(angle), PITCH * turn])


def build_model(n_elem: int = 80) -> StructuralModel:
    """Helix clamped at the base, loaded at the free end, propped by a strut to the ground."""
    turns = np.linspace(0.0, TURNS, 61)
    path = BeamPath(np.array([helix_point(t) for t in turns]))
    k_strut = int(np.argmin(np.abs(turns - STRUT_TURN)))  # a definition point of the path
    mesh = discretize(path, n_elem, anchors={"prop": path.s_knots[k_strut]})
    foot = helix_point(turns[k_strut]) * [1.0, 1.0, 0.0]  # vertically below, on the ground
    return StructuralModel(
        beam=OptimizableBeam(mesh, section_factory=TubeSection.with_fixed(t=WALL)),
        material=MATERIAL,
        points={"foot": foot},
        fixed=[FixedBeam("prop", "foot", TubeSection(STRUT_RADIUS, STRUT_WALL))],
        bcs=[
            Clamp(["start", "foot"]),
            NodalForce("end", "FX", LOAD[0]),
            NodalForce("end", "FY", LOAD[1]),
            NodalForce("end", "FZ", LOAD[2]),
        ],
    )


def make_problem(solver=None, n_elem: int = 80, n_ctrl: int = 10, history_file=None):
    """Minimum-mass problem; *solver* is a MAPDL session or (default) ``NativeSolver()``."""
    return OptimizationProblem(
        solver if solver is not None else NativeSolver(),
        build_model(n_elem),
        n_ctrl=n_ctrl,
        bounds={"r": R_BOUNDS},
        safety_factor=SAFETY_FACTOR,
        f_react_max=10.0 * float(np.linalg.norm(LOAD)),  # not a design requirement here
        reaction_nodes=["start"],
        displacement_limits=[("end", "UZ", DEFLECTION_LIMIT)],
        interpolator="pchip",
        history_file=history_file,
    )


def run(problem: OptimizationProblem, method: str = "slsqp", max_evals: int = 4000, seed=None):
    """Optimise from the heaviest admissible tube (upper bound everywhere)."""
    return optimize(problem, method, x0=problem.upper, max_evals=max_evals, seed=seed)


def uniform_design(problem: OptimizationProblem, n: int = 97) -> tuple[float, float]:
    """Lightest admissible tube of constant radius: (radius [mm], mass [g])."""
    for r in np.linspace(*R_BOUNDS, n):
        x = np.full(problem.n_vars, r)
        if problem.max_violation(problem.evaluate(x)) <= 1e-9:
            return float(r), problem.objective(x)
    raise RuntimeError("no constant radius satisfies the constraints")


def summary(problem: OptimizationProblem, x_opt: np.ndarray) -> dict[str, float]:
    """Key figures of the optimised helix against the constant-radius one."""
    ev = problem.evaluate(x_opt)
    r_uniform, m_uniform = uniform_design(problem)
    mass = problem.objective(x_opt)
    return {
        "mass": mass,
        "mass_uniform": m_uniform,
        "r_uniform": r_uniform,
        "saving": 1.0 - mass / m_uniform,
        "utilisation": float(ev.sigma.max() / problem.sigma_adm),
        "deflection": float(abs(ev.displacements[0])),
        "r_min": float(np.min(x_opt)),
        "r_max": float(np.max(x_opt)),
    }


def print_summary(s: dict[str, float]) -> None:
    print("\n[HELICAL TUBE BEAM]")
    print(f"  Optimised mass          : {s['mass']:8.2f} g")
    print(
        f"  Constant radius {s['r_uniform']:5.2f} mm: {s['mass_uniform']:8.2f} g   (saving {s['saving']:.1%})"
    )
    print(f"  Stress utilisation      : {s['utilisation']:8.1%}")
    print(f"  Free-end deflection     : {s['deflection']:8.2f} mm   (limit {DEFLECTION_LIMIT:g})")
    print(f"  Radius range            : {s['r_min']:.2f} to {s['r_max']:.2f} mm")


def plot_design(problem, x_opt, save_path: str | None = None, show: bool = True) -> None:
    """3-D view of the helix (line width ∝ radius, colour = utilisation) and r(s)."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    model = problem.model
    mesh = model.beam.mesh
    law = problem.law_from_x(x_opt)
    util = problem.evaluate(x_opt).sigma / problem.sigma_adm
    r_elem = law.eval(mesh.s_elem)["r"]

    fig = plt.figure(figsize=(11.5, 5.0))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    segments = np.stack([mesh.coords[:-1], mesh.coords[1:]], axis=1)
    lines = Line3DCollection(segments, cmap="viridis", linewidths=0.9 * r_elem)
    lines.set_array(util)
    lines.set_clim(0.0, 1.0)
    ax.add_collection3d(lines)
    coords = model.named_coords()
    strut = np.array([coords["prop"], coords["foot"]])
    ax.plot(*strut.T, color="#6B7280", linewidth=4.0, label="fixed strut")
    ax.scatter(
        *np.array([coords["start"], coords["foot"]]).T, color="k", marker="s", s=45, label="clamps"
    )
    tip = coords["end"]
    arrow = np.array(LOAD) / np.linalg.norm(LOAD) * 70.0
    ax.quiver(*tip, *arrow, color="#DC2626", linewidth=2.0, label="load")
    lim = HELIX_RADIUS * 1.15
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-20.0, PITCH * TURNS + 20.0)
    ax.set_box_aspect((2 * lim, 2 * lim, PITCH * TURNS + 40.0))
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("z [mm]")
    ax.view_init(elev=24, azim=-58)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Optimised helix (line width ∝ tube radius)")
    fig.colorbar(lines, ax=ax, label="σ_VM / σ_adm [-]", shrink=0.7, pad=0.08)

    ax2 = fig.add_subplot(1, 2, 2)
    s = np.linspace(0.0, mesh.s[-1], 400)
    ax2.plot(s, law.eval(s)["r"], color="#2563EB", linewidth=1.8, label="optimised r(s)")
    ax2.plot(problem.s_ctrl, law.ctrl["r"], "o", color="#2563EB", label="control points")
    ax2.axhline(uniform_design(problem)[0], color="k", linestyle="--", linewidth=1.2,
                label="best constant radius")  # fmt: skip
    ax2.axvline(mesh.s[mesh.anchors["prop"]], color="#6B7280", linestyle=":", label="strut")
    ax2.set_xlabel("Curvilinear abscissa s from the clamp [mm]")
    ax2.set_ylabel("Outer radius r [mm]")
    ax2.set_ylim(0.0, R_BOUNDS[1] * 1.08)
    ax2.grid(alpha=0.35)
    ax2.legend(fontsize=9)
    fig.tight_layout()

    if save_path:
        fig.savefig(f"{save_path}_design.png", dpi=150)
        print(f"  Design plot saved: {save_path}_design.png")
    if show:
        plt.show()


def main() -> None:
    from beam_opt.report import print_report

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n-elem", type=int, default=80, help="number of beam elements")
    parser.add_argument("--n-ctrl", type=int, default=10, help="section-law control points")
    parser.add_argument("--method", default="slsqp", choices=sorted(METHODS), help="algorithm")
    parser.add_argument("--max-evals", type=int, default=4000, help="budget of FEM solutions")
    parser.add_argument("--seed", type=int, default=None, help="seed of the evolutionary method")
    parser.add_argument("--solver", default="native", choices=["native", "mapdl"])
    parser.add_argument("--save-plots", default=None, help="path prefix for the PNG output")
    parser.add_argument("--no-plots", action="store_true", help="do not open plot windows")
    parser.add_argument("--export-apdl", default=None, help="write the optimum as an APDL script")
    args = parser.parse_args()

    def solve_and_report(session) -> None:
        problem = make_problem(session, args.n_elem, args.n_ctrl)
        result = run(problem, args.method, args.max_evals, args.seed)
        print_report(result, problem)
        print_summary(summary(problem, result.x))
        if args.export_apdl:
            write_apdl(
                problem.model, problem.law_from_x(result.x), args.export_apdl, "Helical beam"
            )
            print(f"  APDL script written: {args.export_apdl}")
        if args.save_plots or not args.no_plots:
            plot_design(problem, result.x, args.save_plots, show=not args.no_plots)

    if args.solver == "native":
        solve_and_report(NativeSolver())
    else:
        from beam_opt import MapdlSession

        with MapdlSession(jobname="helix") as mapdl:
            solve_and_report(mapdl)


if __name__ == "__main__":
    main()
