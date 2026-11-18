"""Extra examples: stiffness of the helix, fatigue of the clip, bracket with load cases."""

from __future__ import annotations

import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pytest

from beam_opt import optimize

EXAMPLES = Path(__file__).parents[2] / "examples" / "extras"


def _load(folder: str, name: str):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / folder / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stiff = _load("stiffness_pareto", "stiffness_pareto")
lcf = _load("lcf_clip", "lcf_clip")
bracket = _load("bracket_load_cases", "bracket")


@pytest.fixture(scope="module")
def stiffness_front():
    return stiff.trade_off(n_elem=40, n_ctrl=6, factors=[1.0, 1.4, 2.0])


def test_stiffness_front_is_monotone_and_uses_the_budget(stiffness_front):
    problem, rows = stiffness_front
    assert all(r["feasible"] for r in rows)
    assert all(a["compliance"] > b["compliance"] for a, b in itertools.pairwise(rows))
    for r in rows:
        assert r["mass"] == pytest.approx(r["budget"], rel=2e-3)
        assert problem.evaluate(r["x"]).sigma.max() <= problem.sigma_adm * (1 + 1e-3)


def test_optimised_helix_is_stiffer_than_the_constant_tube_of_equal_mass(stiffness_front):
    _, rows = stiffness_front
    assert all(r["gain"] > 0.1 for r in rows)
    assert not rows[0]["tube"]["stress_ok"]  # at the lightest mass, no constant tube is admissible


def test_first_point_of_the_front_is_the_lightest_admissible_helix(stiffness_front):
    problem, rows = stiffness_front
    _, light = stiff.lightest(n_elem=40, n_ctrl=6)
    assert rows[0]["mass"] == pytest.approx(problem.mass_g(light.x), rel=2e-3)


@pytest.fixture(scope="module")
def fatigue_front():
    problem = lcf.make_problem(n_elem=40, n_ctrl=6)
    return problem, lcf.trade_off(problem, grips=[20.0, 90.0])


def test_clip_trades_grip_for_life(fatigue_front):
    _, rows = fatigue_front
    low, high = rows
    assert all(r["feasible"] for r in rows)
    assert low["life"] > 5.0 * high["life"]  # a stronger grip shortens the life
    for r in rows:
        assert r["force"] == pytest.approx(r["grip"], rel=2e-3)  # the grip limit is active


def test_variable_thickness_multiplies_the_life(fatigue_front):
    problem, rows = fatigue_front
    for r in rows:
        assert r["life"] > 3.0 * r["life_const"]
    high = rows[-1]
    assert high["life"] < 1e5 and high["plastic"] > 0.0  # low-cycle fatigue with local yielding
    lives = problem.lives(problem.evaluate(high["x"]))
    body = np.log10(lives[: int(0.7 * len(lives))])
    assert np.ptp(body) < 0.3  # the optimum spreads the damage along the clip


def test_clip_service_life_under_a_daily_spectrum():
    problem = lcf.make_problem(n_elem=40, n_ctrl=6)
    result = lcf.service(problem, grip=60.0)
    days = result["days"]
    assert result["spectrum"].cycles == pytest.approx(lcf.USES_PER_DAY)
    assert days["full openings"] > 3.0 * days["constant thickness"]
    # One load pattern scaled: minimising the worst damage is minimising the worst
    # stress, whatever the life curve.  Started from the optimum for full openings,
    # the optimisation for the spectrum must find nothing better.
    assert days["daily spectrum"] == pytest.approx(days["full openings"], rel=1e-2)
    np.testing.assert_allclose(result["x_tuned"], result["x_full"], rtol=1e-2)
    assert sum(share for _, share in result["shares"]) == pytest.approx(1.0)


def test_bracket_optimum_depends_on_the_load_mix():
    rare, frequent = 0.0, 100.0
    problems = {k: bracket.make_problem(k, n_elem=30, n_ctrl=5) for k in (rare, frequent)}
    x0 = np.full(problems[rare].n_vars, 6.0)
    designs = {k: optimize(p, "slsqp", x0=x0).x for k, p in problems.items()}
    for k, p in problems.items():
        assert p.max_violation(p.evaluate(designs[k])) <= 1e-4
        assert p.mass_g(designs[k]) == pytest.approx(bracket.MASS_BUDGET, rel=2e-3)

    def aspect(k):  # mean h / b along the bracket
        law = problems[k].law_from_x(designs[k]).eval(problems[k].model.beam.mesh.s_elem)
        return float(np.mean(law["h"] / law["b"]))

    assert aspect(frequent) > 1.3 * aspect(rare)  # more knocks: deeper out of plane
    # each design is the better one for its own mix
    for own, other in ((rare, frequent), (frequent, rare)):
        mine = bracket.life_days(problems[own], designs[own])
        theirs = bracket.life_days(problems[own], designs[other])
        assert mine > theirs
    ignoring = bracket.life_days(problems[frequent], designs[rare])
    assert bracket.life_days(problems[frequent], designs[frequent]) > 10.0 * ignoring
