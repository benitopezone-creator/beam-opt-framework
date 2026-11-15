"""
Example 04 — from an APDL script to an optimised APDL script.

``arch.inp`` is an ordinary APDL input file: a clamped semicircular arch with
a vertical and an out-of-plane load at the crown, written with parameters and
``*DO`` loops.  This example

1. reads the script (no ANSYS needed) and rebuilds the model;
2. checks the design of the script against the constraints;
3. optimises width and height of the rectangular section along the arch;
4. writes the optimum back as an APDL script, ready to be run in ANSYS.

Run it::

    python examples/04_apdl_arch/apdl_arch.py --output results/arch_optimised.inp

Units: mm, N, ton, s, MPa.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from beam_opt import NativeSolver, optimize
from beam_opt.apdl import ApdlImport, read_apdl, write_apdl
from beam_opt.optimization import METHODS

SCRIPT = Path(__file__).with_name("arch.inp")
SIGMA_Y = 355.0  # yield stress [MPa]: APDL scripts do not carry it
SAFETY_FACTOR = 1.5
BOUNDS = {"b": (6.0, 30.0), "h": (6.0, 30.0)}  # technological limits [mm]
CROWN_LIMIT = 1.0  # admissible vertical deflection of the crown [mm]


def load(parameters: dict[str, float] | None = None) -> ApdlImport:
    """Read ``arch.inp``; *parameters* are made available to the script."""
    return read_apdl(str(SCRIPT), sigma_y=SIGMA_Y, parameters=parameters)


def make_problem(imported: ApdlImport, solver=None, n_ctrl: int = 7):
    """Minimum-mass problem on the imported model."""
    mesh = imported.model.beam.mesh
    crown = next(n for n, i in mesh.anchors.items() if i == mesh.n_elem // 2 and n.startswith("n"))
    return imported.problem(
        solver if solver is not None else NativeSolver(),
        n_ctrl=n_ctrl,
        bounds=BOUNDS,
        safety_factor=SAFETY_FACTOR,
        f_react_max=1.0e5,  # not a design requirement here
        displacement_limits=[(crown, "UY", CROWN_LIMIT)],
        interpolator="pchip",
    )


def run(problem, imported: ApdlImport, method: str = "slsqp", max_evals: int = 4000):
    """Optimise starting from the sections written in the script."""
    return optimize(problem, method, x0=imported.x0(problem), max_evals=max_evals)


def summary(problem, imported: ApdlImport, x_opt: np.ndarray) -> dict[str, float]:
    """The script design against the optimised one."""
    x_script = imported.x0(problem)
    ev0, ev1 = problem.evaluate(x_script), problem.evaluate(x_opt)
    return {
        "mass_script": problem.objective(x_script),
        "mass": problem.objective(x_opt),
        "saving": 1.0 - problem.objective(x_opt) / problem.objective(x_script),
        "utilisation_script": float(ev0.sigma.max() / problem.sigma_adm),
        "utilisation": float(ev1.sigma.max() / problem.sigma_adm),
        "deflection_script": float(abs(ev0.displacements[0])),
        "deflection": float(abs(ev1.displacements[0])),
    }


def print_summary(s: dict[str, float]) -> None:
    print("\n[ARCH FROM AN APDL SCRIPT]")
    print(f"  {'':24}{'script':>10}{'optimised':>12}")
    print(
        f"  {'Mass [g]':24}{s['mass_script']:10.1f}{s['mass']:12.1f}   (saving {s['saving']:.1%})"
    )
    print(f"  {'Stress utilisation':24}{s['utilisation_script']:10.1%}{s['utilisation']:12.1%}")
    print(
        f"  {'Crown deflection [mm]':24}{s['deflection_script']:10.2f}{s['deflection']:12.2f}"
        f"   (limit {CROWN_LIMIT:g})"
    )


def main() -> None:
    from beam_opt.report import print_report

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n-ctrl", type=int, default=7, help="section-law control points")
    parser.add_argument("--method", default="slsqp", choices=sorted(METHODS), help="algorithm")
    parser.add_argument("--output", default=None, help="APDL script of the optimum to write")
    args = parser.parse_args()

    imported = load()
    print(f"Read {SCRIPT.name}: {imported.model.beam.mesh.n_elem} elements to optimise,")
    print(f"  section parameters {imported.param_names}, {len(imported.notes)} commands ignored")
    problem = make_problem(imported, n_ctrl=args.n_ctrl)
    result = run(problem, imported, args.method)
    print_report(result, problem)
    print_summary(summary(problem, imported, result.x))
    if args.output:
        write_apdl(problem.model, problem.law_from_x(result.x), args.output, "Optimised arch")
        print(f"  Optimised APDL script written: {args.output}")


if __name__ == "__main__":
    main()
