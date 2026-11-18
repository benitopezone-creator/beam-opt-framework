"""
Example 07 — curved bracket with two independent load cases.

A quarter-circle bracket, clamped at one end, carries at the other end

* a frequent load in its own plane (service), and
* a rarer load out of its plane (lateral knocks), which also twists it.

The rectangular section is free in both dimensions: the in-plane width b
resists the service load, the out-of-plane height h resists the knocks.
The bracket is designed for the longest fatigue life at a given mass, with
damage summed over the two cases by Miner's rule.

The point of the example: with independent load cases, **the optimal shape
depends on how often each case occurs**.  The design that ignores the rare
case, or the one tuned for another mix, loses life.

Run it (built-in solver, no ANSYS needed)::

    python examples/07_bracket_load_cases/bracket.py --save-plots results/bracket

Units: mm, N, ton, s, MPa.  Strain-life constants as in example 06
(illustrative).
"""

from __future__ import annotations

import argparse

import numpy as np

from beam_opt import (
    BeamPath,
    Clamp,
    FatigueLife,
    FatigueModel,
    LoadCase,
    Material,
    NativeSolver,
    NodalForce,
    OptimizableBeam,
    OptimizationProblem,
    StrainLifeCurve,
    StructuralModel,
    optimize,
)
from beam_opt.geometry import discretize

RADIUS = 150.0  # radius of the centre line [mm]
P_SERVICE = 300.0  # in-plane load at the free end [N]
P_KNOCK = 40.0  # out-of-plane load at the free end [N]
SERVICE_CYCLES = 1000.0  # service cycles per day (zero to peak)
KNOCKS_PER_DAY = [0.0, 10.0, 100.0, 1000.0]  # mixes studied (fully reversed knocks)
BOUNDS = {"b": (2.0, 20.0), "h": (2.0, 20.0)}  # [mm]
MASS_BUDGET = 60.0  # [g]
MATERIAL = Material(E=2.0e5, nu=0.3, rho=7.85e-9, sigma_y=355.0, name="steel")
CURVE = StrainLifeCurve(E=2.0e5, sigma_f=930.0, b=-0.095, eps_f=0.26, c=-0.47, K=1258.0, n=0.208)


def build_model(n_elem: int = 50) -> StructuralModel:
    """Quarter circle in the XY plane, clamped at (R, 0, 0), free at (0, R, 0)."""
    t = np.linspace(0.0, np.pi / 2.0, 31)
    points = np.column_stack([RADIUS * np.cos(t), RADIUS * np.sin(t), np.zeros_like(t)])
    return StructuralModel(
        beam=OptimizableBeam(discretize(BeamPath(points), n_elem)),  # b in plane, h out of plane
        material=MATERIAL,
        bcs=[Clamp("start")],
    )


def load_cases(knocks_per_day: float) -> list[LoadCase]:
    """Service (zero to peak) and knocks (fully reversed, as 0.5-cycle pairs)."""
    from beam_opt import LoadBlock, LoadSpectrum

    service = LoadSpectrum((LoadBlock(1.0, 0.0, SERVICE_CYCLES),), name="day")
    knocks = LoadSpectrum((LoadBlock(1.0, -1.0, knocks_per_day),), name="day")
    return [
        LoadCase("service", [NodalForce("end", "FX", -P_SERVICE)], spectrum=service),
        LoadCase("knock", [NodalForce("end", "FZ", P_KNOCK)], spectrum=knocks),
    ]


def make_problem(knocks_per_day: float, n_elem: int = 50, n_ctrl: int = 6, solver=None):
    """Longest life in days at the mass budget."""
    return OptimizationProblem(
        solver if solver is not None else NativeSolver(),
        build_model(n_elem),
        n_ctrl=n_ctrl,
        bounds=BOUNDS,
        safety_factor=None,
        objective=FatigueLife(),
        fatigue=FatigueModel(CURVE, mean_stress="swt"),
        load_cases=load_cases(knocks_per_day),
        mass_max=MASS_BUDGET,
        interpolator="pchip",
    )


def life_days(problem: OptimizationProblem, x: np.ndarray) -> float:
    return float(problem.lives(problem.evaluate(x)).min())


def study(n_elem: int = 50, n_ctrl: int = 6) -> dict:
    """Optimum for every mix, and every optimum evaluated under every mix."""
    problems = {k: make_problem(k, n_elem, n_ctrl) for k in KNOCKS_PER_DAY}
    x0 = np.full(problems[KNOCKS_PER_DAY[0]].n_vars, 6.0)
    designs = {k: optimize(p, "slsqp", x0=x0).x for k, p in problems.items()}
    table = {
        (designed, used): life_days(problems[used], designs[designed])
        for designed in KNOCKS_PER_DAY
        for used in KNOCKS_PER_DAY
    }
    return {"problems": problems, "designs": designs, "table": table}


def print_study(s: dict) -> None:
    mixes = KNOCKS_PER_DAY
    print(f"\n[CURVED BRACKET, TWO LOAD CASES]  {SERVICE_CYCLES:g} service cycles per day,"
          f" mass {MASS_BUDGET:g} g")  # fmt: skip
    print("  life in days of the design optimised for a mix (rows) used under a mix (columns)")
    print("  knocks/day " + "".join(f"{m:>12g}" for m in mixes))
    for d in mixes:
        row = "".join(f"{s['table'][(d, u)]:12.3g}" for u in mixes)
        print(f"  {d:>10g} {row}")
    print("  section at the clamp of each optimum:")
    for d in mixes:
        law = s["problems"][d].law_from_x(s["designs"][d]).eval([0.0])
        b, h = float(law["b"][0]), float(law["h"][0])
        print(f"    {d:>6g} knocks/day: b = {b:5.2f} mm, h = {h:5.2f} mm, h/b = {h / b:4.2f}")
    for u in mixes[1:]:
        loss = s["table"][(u, u)] / s["table"][(mixes[0], u)]
        print(f"  with {u:g} knocks/day, the design that ignores them lasts {loss:.3g} times less")


def plot_study(s: dict, save_path: str | None = None, show: bool = True) -> None:
    import matplotlib.pyplot as plt

    mixes = KNOCKS_PER_DAY
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    any_problem = s["problems"][mixes[0]]
    mesh = any_problem.model.beam.mesh
    sv = np.linspace(0.0, mesh.s[-1], 300)
    colors = ["#1E3A8A", "#2563EB", "#60A5FA", "#BFDBFE"]
    for color, m in zip(colors, mixes):
        law = s["problems"][m].law_from_x(s["designs"][m]).eval(sv)
        ax1.plot(sv, law["h"] / law["b"], color=color, lw=1.8, label=f"{m:g} knocks/day")
    ax1.set_xlabel("Curvilinear abscissa s from the clamp [mm]")
    ax1.set_ylabel("Section aspect h / b [-]  (h out of plane)")
    ax1.set_title("Optimal section depends on the load mix")
    ax1.grid(alpha=0.35)
    ax1.legend()

    x = np.arange(len(mixes))
    tuned = [s["table"][(m, m)] for m in mixes]
    ignore = [s["table"][(mixes[0], m)] for m in mixes]
    ax2.bar(x - 0.2, tuned, 0.4, color="#2563EB", label="designed for this mix")
    ax2.bar(x + 0.2, ignore, 0.4, color="#9CA3AF", label="designed ignoring the knocks")
    ax2.set_yscale("log")
    ax2.set_xticks(x, [f"{m:g}" for m in mixes])
    ax2.set_xlabel("Knocks per day in service")
    ax2.set_ylabel("Fatigue life [days]")
    ax2.set_title(f"Same mass ({MASS_BUDGET:g} g), different designs")
    ax2.grid(True, axis="y", which="both", alpha=0.3)
    ax2.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(f"{save_path}_load_mix.png", dpi=150)
        print(f"  Plot saved: {save_path}_load_mix.png")
    if show:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n-elem", type=int, default=50)
    parser.add_argument("--n-ctrl", type=int, default=6)
    parser.add_argument("--save-plots", default=None, help="path prefix for the PNG output")
    parser.add_argument("--no-plots", action="store_true", help="do not open plot windows")
    args = parser.parse_args()
    s = study(args.n_elem, args.n_ctrl)
    print_study(s)
    if args.save_plots or not args.no_plots:
        plot_study(s, args.save_plots, show=not args.no_plots)


if __name__ == "__main__":
    main()
