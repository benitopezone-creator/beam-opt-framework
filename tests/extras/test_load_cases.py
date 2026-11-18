"""Independent load cases: superposition, envelopes, combined objectives, fatigue mix."""

from __future__ import annotations

import numpy as np
import pytest
from test_fatigue import STEEL

from beam_opt.extras import Compliance, FatigueLife, FatigueModel, LoadCase
from beam_opt.geometry import BeamPath, discretize
from beam_opt.materials import Material
from beam_opt.model import Clamp, ImposedDisplacement, NodalForce, OptimizableBeam, StructuralModel
from beam_opt.native import NativeSolver
from beam_opt.optimization import Mass, OptimizationProblem, optimize

L = 200.0  # [mm]
MAT = Material(E=2.0e5, nu=0.3, rho=7.85e-9, sigma_y=355.0)
PV, PL = 60.0, 30.0  # vertical and lateral tip loads [N]
VERTICAL = LoadCase("vertical", [NodalForce("end", "FZ", -PV)])
LATERAL = LoadCase("lateral", [NodalForce("end", "FY", PL)])


def model(bcs=None) -> StructuralModel:
    path = BeamPath([[0.0, 0.0, 0.0], [L, 0.0, 0.0]], kind="polyline")
    return StructuralModel(
        OptimizableBeam(discretize(path, 40)),  # rectangle: b along y (lateral), h along z
        MAT,
        bcs=bcs or [Clamp("start"), NodalForce("end", "FZ", -PV)],
    )


def make(cases=None, objective=None, **kwargs) -> OptimizationProblem:
    options = {
        "n_ctrl": 6,
        "bounds": {"b": (1.0, 15.0), "h": (1.0, 15.0)},
        "safety_factor": 1.5,
        "reaction_nodes": ["start"],
        "interpolator": "pchip",
    } | kwargs
    return OptimizationProblem(
        NativeSolver(shear_deformation=False), model(), objective=objective, load_cases=cases,
        **options,
    )  # fmt: skip


X = np.concatenate([np.full(6, 4.0), np.full(6, 6.0)])  # b = 4, h = 6 mm


# ── Basics ───────────────────────────────────────────────────────────────


def test_one_case_with_the_model_loads_changes_nothing():
    plain = make().evaluate(X)
    single = make([VERTICAL]).evaluate(X)
    np.testing.assert_allclose(single.sigma, plain.sigma, rtol=1e-12)
    assert single.reaction == pytest.approx(plain.reaction)
    assert [c.name for c in single.cases] == ["vertical"]


def test_superposition_of_linear_load_cases():
    both = LoadCase("both", [*VERTICAL.loads, *LATERAL.loads])
    problem = make([VERTICAL, LATERAL, both], objective=Compliance())
    v, lat, vl = problem.evaluate(X).cases
    for dof in ("UZ", "UY"):
        key = ("end", dof)
        assert vl.watched[key] == pytest.approx(v.watched[key] + lat.watched[key], abs=1e-12)
    # closed forms of the tip deflections
    b, h = 4.0, 6.0
    assert -v.watched[("end", "UZ")] == pytest.approx(PV * L**3 / (3 * MAT.E * b * h**3 / 12))
    assert lat.watched[("end", "UY")] == pytest.approx(PL * L**3 / (3 * MAT.E * h * b**3 / 12))


def test_stress_envelope_and_constraints_cover_every_case():
    problem = make([VERTICAL, LATERAL])
    ev = problem.evaluate(X)
    np.testing.assert_allclose(ev.sigma, np.maximum(ev.cases[0].sigma, ev.cases[1].sigma))
    assert ev.reaction == pytest.approx(max(PV, PL))
    n_elem = problem.model.beam.mesh.n_elem
    assert problem.constraint_vector(X).size == 2 * n_elem


def test_compliance_is_the_weighted_sum_of_the_cases():
    cases = [
        LoadCase("vertical", VERTICAL.loads, weight=2.0),
        LoadCase("lateral", LATERAL.loads, weight=0.5),
    ]
    problem = make(cases, objective=Compliance())
    ev = problem.evaluate(X)
    c_v, c_l = Compliance().case_values(problem, ev)
    assert problem.objective(X) == pytest.approx(2.0 * c_v + 0.5 * c_l)
    b, h = 4.0, 6.0
    assert c_v == pytest.approx(PV**2 * L**3 / (3 * MAT.E * b * h**3 / 12))


def test_case_imposed_displacement_replaces_the_model_one():
    base = model([Clamp("start"), ImposedDisplacement("end", "UZ", 1.0)])
    case = LoadCase("pushed", [ImposedDisplacement("end", "UZ", 2.5)])
    bcs = case.apply_to(base).bcs
    imposed = [bc for bc in bcs if isinstance(bc, ImposedDisplacement)]
    assert [(bc.nodes, bc.dof, bc.value) for bc in imposed] == [(["end"], "UZ", 2.5)]


def test_invalid_load_cases_are_rejected():
    with pytest.raises(TypeError, match="supports belong to the model"):
        LoadCase("bad", [Clamp("end")])
    with pytest.raises(ValueError, match="unique"):
        make([VERTICAL, VERTICAL])
    with pytest.raises(ValueError, match="non-negative"):
        LoadCase("neg", VERTICAL.loads, cycles=-1.0)


# ── Fatigue: damage summed over the cases ────────────────────────────────


def test_fatigue_damage_is_summed_over_the_cases():
    cases = [
        LoadCase("vertical", VERTICAL.loads, cycles=1000.0),
        LoadCase("lateral", LATERAL.loads, cycles=50.0),
    ]
    fatigue = FatigueModel(STEEL, load_ratio=-1.0)
    problem = make(cases, fatigue=fatigue, safety_factor=None)
    ev = problem.evaluate(X)
    damage = 1000.0 * fatigue.damage(ev.cases[0].sigma) + 50.0 * fatigue.damage(ev.cases[1].sigma)
    with np.errstate(divide="ignore"):
        expected = np.where(damage > 0.0, 1.0 / damage, 0.5e12)  # no damage: run-out
    np.testing.assert_allclose(problem.lives(ev), expected, rtol=1e-12)
    assert problem.life_unit == "repetitions"


# ── Optimisation ─────────────────────────────────────────────────────────


def test_minimum_mass_design_must_resist_every_case():
    vertical_only = make([VERTICAL])
    x_v = optimize(vertical_only, "slsqp", x0=vertical_only.upper).x
    both = make([VERTICAL, LATERAL])
    result = optimize(both, "slsqp", x0=both.upper)
    assert result.success and result.max_violation <= 1e-4
    assert both.max_violation(both.evaluate(x_v)) > 0.05  # the lateral case breaks it
    assert result.fun > vertical_only.objective(x_v)


def root_aspect(problem, x) -> float:
    """h / b at the clamp."""
    p = problem.law_from_x(x).eval([0.0])
    return float(p["h"][0] / p["b"][0])


def test_the_mix_of_cycles_changes_the_optimal_section():
    # Same loads, different numbers of cycles per repetition: the more lateral
    # cycles, the wider the section must be.
    aspects = []
    for lateral_cycles in (1.0, 30.0, 1000.0):
        cases = [
            LoadCase("vertical", VERTICAL.loads, cycles=1000.0),
            LoadCase("lateral", LATERAL.loads, cycles=lateral_cycles),
        ]
        problem = make(
            cases,
            objective=FatigueLife(),
            fatigue=FatigueModel(STEEL, load_ratio=0.0),
            safety_factor=None,
            mass_max=45.0,
        )
        result = optimize(problem, "slsqp", x0=np.concatenate([np.full(6, 6.0), np.full(6, 6.0)]))
        assert result.success and result.max_violation <= 1e-4
        aspects.append(root_aspect(problem, result.x))
    assert aspects[0] > aspects[1] > aspects[2]
    assert aspects[0] > 3.0 and aspects[2] < 2.0  # from tall and narrow to almost square


def test_mass_objective_with_cases_and_report_runs():
    import contextlib
    import io

    from beam_opt.report import print_report

    problem = make([VERTICAL, LATERAL], objective=Mass())
    result = optimize(problem, "slsqp", x0=problem.upper)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        print_report(result, problem)
    assert "[LOAD CASES]" in out.getvalue() and "lateral" in out.getvalue()
