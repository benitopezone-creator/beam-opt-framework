"""Displacement limits and minimum reaction (built-in solver)."""

from __future__ import annotations

import numpy as np
import pytest

from beam_opt.geometry import BeamPath, discretize
from beam_opt.materials import Material
from beam_opt.model import Clamp, ImposedDisplacement, NodalForce, OptimizableBeam, StructuralModel
from beam_opt.native import NativeSolver
from beam_opt.optimization import OptimizationProblem, optimize
from beam_opt.sections import RectSection

STEEL = Material(E=2.0e5, nu=0.3, rho=7.85e-9, sigma_y=355.0)
L, WIDTH = 200.0, 5.0  # [mm]


def _problem(bcs, **kwargs) -> OptimizationProblem:
    path = BeamPath([[0.0, 0.0, 0.0], [L, 0.0, 0.0]], kind="polyline")
    model = StructuralModel(
        OptimizableBeam(discretize(path, 40), section_factory=RectSection.with_fixed(b=WIDTH)),
        STEEL,
        bcs=bcs,
    )
    options = {
        "n_ctrl": 6,
        "bounds": {"h": (1.0, 12.0)},
        "safety_factor": 1.5,
        "f_react_max": 1e3,
        "reaction_nodes": ["start"],
        "interpolator": "pchip",
    }
    return OptimizationProblem(NativeSolver(), model, **(options | kwargs))


def test_displacement_limit_is_active_when_it_governs():
    load = [Clamp("start"), NodalForce("end", "FZ", -20.0)]
    free = _problem(load)
    light = optimize(free, "slsqp", x0=free.upper)

    limit = 4.0  # [mm], well below the deflection of the stress-only optimum
    stiff_problem = _problem(load, displacement_limits=[("end", "UZ", limit)])
    stiff = optimize(stiff_problem, "slsqp", x0=stiff_problem.upper)
    ev = stiff_problem.evaluate(stiff.x)

    assert stiff.success and stiff.max_violation <= 1e-4
    assert abs(ev.displacements[0]) == pytest.approx(limit, rel=1e-3)
    assert stiff.fun > light.fun  # stiffness costs mass
    assert ev.sigma.max() < stiff_problem.sigma_adm  # the stress no longer governs everywhere


def test_minimum_reaction_under_an_imposed_displacement():
    # Leaf spring deflected by 5 mm: it must push back with at least 8 N
    # without yielding.  Thinner is softer, thicker is more stressed.
    f_min = 8.0
    problem = _problem([Clamp("start"), ImposedDisplacement("end", "UZ", 5.0)], f_react_min=f_min)
    result = optimize(problem, "slsqp", x0=np.full(problem.n_vars, 4.0))
    ev = problem.evaluate(result.x)
    assert result.success and result.max_violation <= 1e-4
    assert ev.reaction == pytest.approx(f_min, rel=1e-3)
    assert ev.sigma.max() <= problem.sigma_adm * (1 + 1e-4)
    assert problem.constraint_reaction_min(result.x) >= -1e-4


def test_without_a_minimum_the_softest_spring_wins():
    problem = _problem([Clamp("start"), ImposedDisplacement("end", "UZ", 5.0)])
    result = optimize(problem, "slsqp", x0=np.full(problem.n_vars, 4.0))
    np.testing.assert_allclose(result.x, problem.lower, atol=1e-6)


def test_invalid_displacement_limit_is_rejected():
    for bad in [("nowhere", "UZ", 1.0), ("end", "UW", 1.0), ("end", "UZ", 0.0)]:
        with pytest.raises(ValueError, match="invalid displacement limit"):
            _problem([Clamp("start")], displacement_limits=[bad])


def test_failed_evaluation_violates_the_displacement_limit():
    problem = _problem([NodalForce("end", "FZ", -1.0)], displacement_limits=[("end", "UZ", 2.0)])
    ev = problem.evaluate(problem.upper)  # no supports: singular
    assert ev.failed
    assert problem.max_violation(ev) > 1.0
