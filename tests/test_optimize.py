"""
Phase 4: algorithm selection, scaling, failure handling, history and restart.

Every optimisation runs on the cantilever of example 01 with the closed-form
MAPDL stand-in, so the expected optimum is known.
"""

from __future__ import annotations

import numpy as np
import pytest
from fakes import StaticCantileverMapdl
from test_example_cantilever import ex

from beam_opt.exceptions import SolutionNotConvergedError
from beam_opt.optimization import METHODS, best_design, optimize, read_history


def _problem(tmp_path, n_elem=50, n_ctrl=8, mapdl_cls=StaticCantileverMapdl, **kwargs):
    tmp_path.mkdir(parents=True, exist_ok=True)
    problem = ex.make_problem(mapdl_cls(str(tmp_path)), n_elem, n_ctrl)
    for name, value in kwargs.items():
        setattr(problem, name, value)
    return problem


# ── Algorithm selection ──────────────────────────────────────────────────


def test_methods_registry():
    assert set(METHODS) == {"cobyla", "slsqp", "nelder-mead", "differential-evolution"}


def test_unknown_method_and_wrong_x0_are_rejected(tmp_path):
    problem = _problem(tmp_path)
    with pytest.raises(ValueError, match="unknown method"):
        optimize(problem, "genetic")
    with pytest.raises(ValueError, match="expected 8"):
        optimize(problem, "cobyla", x0=np.ones(3))


@pytest.mark.parametrize(
    ("method", "n_elem", "n_ctrl"),
    [
        ("cobyla", 50, 8),
        ("slsqp", 50, 8),
        ("slsqp", 100, 16),
        ("cobyla", 100, 16),
    ],
)
def test_local_methods_reach_the_closed_form_optimum(tmp_path, method, n_elem, n_ctrl):
    # (100, 16) from the upper bounds is the case where COBYLA used to stop far
    # above the optimum while reporting success: its first simplex left the
    # admissible box, where the clipped sections give a flat response.
    problem = _problem(tmp_path, n_elem, n_ctrl)
    result = optimize(problem, method, x0=problem.upper)
    c = ex.comparison(problem, result.x)
    assert result.success
    assert result.max_violation <= 1e-4
    assert 0.0 <= c["mass_error"] < 0.03
    assert c["utilisation"] > 0.99


@pytest.mark.slow
def test_differential_evolution_reaches_the_optimum_without_a_good_start(tmp_path):
    problem = _problem(tmp_path, n_elem=20, n_ctrl=4)
    result = optimize(problem, "differential-evolution", max_evals=4000, seed=1)
    reference = optimize(problem, "slsqp", x0=problem.upper)
    assert result.max_violation <= 1e-4
    assert result.fun == pytest.approx(reference.fun, rel=0.02)


def test_differential_evolution_is_reproducible_with_a_seed(tmp_path):
    runs = [
        optimize(
            _problem(tmp_path / str(k), 10, 3), "differential-evolution", max_evals=600, seed=7
        )
        for k in range(2)
    ]
    np.testing.assert_array_equal(runs[0].x, runs[1].x)


def test_nelder_mead_returns_a_feasible_lighter_design(tmp_path):
    problem = _problem(tmp_path, n_elem=20, n_ctrl=3)
    result = optimize(problem, "nelder-mead", x0=problem.upper, max_evals=1500)
    assert result.max_violation <= 1e-4
    assert result.fun < 0.8 * problem.objective(problem.upper)


def test_result_is_in_physical_units_and_counts_fem_solutions(tmp_path):
    problem = _problem(tmp_path)
    result = optimize(problem, "slsqp", x0=problem.upper)
    assert np.all(result.x >= problem.lower) and np.all(result.x <= problem.upper)
    assert result.fun == pytest.approx(problem.objective(result.x))
    assert result.nfev == len(problem.history) == problem.mapdl.n_solves
    assert result.method == "slsqp"


def test_infeasible_problem_is_reported_not_hidden(tmp_path):
    problem = _problem(tmp_path)
    problem.sigma_adm = 1.0  # [MPa] cannot be met within the bounds on h
    result = optimize(problem, "cobyla", x0=problem.upper, max_evals=200)
    assert not result.success
    assert "no feasible design" in result.message
    assert result.max_violation > 1.0


# ── Evaluation cache and objective ───────────────────────────────────────


def test_objective_needs_no_fem_solution(tmp_path):
    problem = _problem(tmp_path)
    problem.objective(problem.upper)
    assert problem.mapdl.n_solves == 0


def test_cache_keeps_more_than_the_last_design(tmp_path):
    problem = _problem(tmp_path)
    a, b = problem.upper, 0.9 * problem.upper
    for x in (a, b, a, b):
        problem.evaluate(x)
    assert problem.mapdl.n_solves == 2


def test_field_constraint_only_for_overshooting_interpolators(tmp_path):
    problem = _problem(tmp_path)  # pchip
    n_elem = problem.model.beam.mesh.n_elem
    assert problem.constraint_vector(problem.upper).size == n_elem + 1
    problem.interpolator = "natural_cubic"
    assert problem.constraint_vector(problem.upper).size == 3 * n_elem + 1


# ── FEM failures ─────────────────────────────────────────────────────────


class FailsWhenThin(StaticCantileverMapdl):
    """The solution "does not converge" when the thinnest section is below 1.5 mm."""

    def solve(self, *args) -> None:
        super().solve(*args)
        thin = min(h for _, h in self._secs.values()) < 1.5
        self.scalars[("ACTIVE", "SOLU", "CNVG")] = 0.0 if thin else 1.0


def test_failed_evaluation_is_flagged_and_penalised(tmp_path):
    problem = _problem(tmp_path, mapdl_cls=FailsWhenThin)
    ev = problem.evaluate(problem.lower)
    assert ev.failed
    assert problem.max_violation(ev) > 1.0
    assert not problem.evaluate(problem.upper).failed


def test_failed_evaluation_can_raise(tmp_path):
    problem = _problem(tmp_path, mapdl_cls=FailsWhenThin, on_failure="raise")
    with pytest.raises(SolutionNotConvergedError):
        problem.evaluate(problem.lower)


def test_optimisation_survives_fem_failures(tmp_path):
    problem = _problem(tmp_path, mapdl_cls=FailsWhenThin)
    result = optimize(problem, "cobyla", x0=problem.upper)
    assert any(ev.failed for ev in problem.history)
    assert result.max_violation <= 1e-4
    assert not problem.evaluate(result.x).failed
    assert min(problem.law_from_x(result.x).eval(problem.model.beam.mesh.s_elem)["h"]) >= 1.5


# ── History and restart ──────────────────────────────────────────────────


def test_history_file_has_one_row_per_fem_solution(tmp_path):
    log = tmp_path / "runs" / "history.csv"
    problem = ex.make_problem(StaticCantileverMapdl(str(tmp_path)), history_file=str(log))
    result = optimize(problem, "slsqp", x0=problem.upper)
    h = read_history(str(log))
    assert len(h["eval"]) == result.nfev
    assert list(h["names"]) == [f"h_{k}" for k in range(8)]
    np.testing.assert_allclose(h["x"][0], problem.upper)
    np.testing.assert_allclose(best_design(str(log)), result.x)


def test_restart_from_history_continues_the_log(tmp_path):
    log = str(tmp_path / "history.csv")
    first = ex.make_problem(StaticCantileverMapdl(str(tmp_path)), history_file=log)
    stopped = optimize(first, "cobyla", x0=first.upper, max_evals=40)  # interrupted early

    second = ex.make_problem(StaticCantileverMapdl(str(tmp_path)), history_file=log)
    x0 = best_design(log)
    resumed = optimize(second, "slsqp", x0=x0)
    assert resumed.fun < stopped.fun
    assert 0.0 <= ex.comparison(second, resumed.x)["mass_error"] < 0.03
    assert len(read_history(log)["eval"]) == stopped.nfev + resumed.nfev


def test_history_of_another_problem_is_rejected(tmp_path):
    log = str(tmp_path / "history.csv")
    ex.make_problem(StaticCantileverMapdl(str(tmp_path)), n_ctrl=8, history_file=log)
    with pytest.raises(ValueError, match="columns do not match"):
        ex.make_problem(StaticCantileverMapdl(str(tmp_path)), n_ctrl=6, history_file=log)


def test_best_design_needs_a_successful_evaluation(tmp_path):
    log = str(tmp_path / "history.csv")
    problem = ex.make_problem(FailsWhenThin(str(tmp_path)), history_file=log)
    problem.evaluate(problem.lower)
    with pytest.raises(ValueError, match="no successful FEM evaluation"):
        best_design(log)
