"""
Example 01 (cantilever of uniform strength) without ANSYS.

MAPDL is replaced by ``StaticCantileverMapdl``, which returns the closed-form
internal forces of the statically determinate beam: these tests check the
optimisation loop (section law, stress evaluator, constraints, COBYLA) against
the analytical optimum, not MAPDL.  The MAPDL counterpart is at the bottom.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from fakes import StaticCantileverMapdl

_PATH = Path(__file__).parents[1] / "examples" / "01_cantilever" / "cantilever.py"
_spec = importlib.util.spec_from_file_location("example_cantilever", _PATH)
ex = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ex)


def test_closed_form_reference():
    h_root = np.sqrt(6.0 * ex.LOAD * ex.LENGTH / (ex.WIDTH * ex.SIGMA_ADM))
    assert ex.analytic_height(0.0) == pytest.approx(h_root)
    assert ex.analytic_height(ex.LENGTH) == ex.H_BOUNDS[0]  # lower bound active at the tip
    # Without the lower bound the mass is exactly 2/3 of the prismatic beam;
    # the bound adds material near the tip only.
    assert 2.0 / 3.0 < ex.analytic_mass() / ex.prismatic_mass() < 0.68


def test_prismatic_beam_stress_is_m_over_w(tmp_path):
    problem = ex.make_problem(StaticCantileverMapdl(str(tmp_path)), n_elem=40)
    h = 6.0  # [mm]
    sigma = problem.evaluate(np.full(problem.n_vars, h)).sigma
    s_i = problem.model.beam.mesh.s[:-1]  # the root-side node governs each element
    expected = 6.0 * ex.LOAD * (ex.LENGTH - s_i) / (ex.WIDTH * h**2)
    np.testing.assert_allclose(sigma[:-1], expected[:-1], rtol=1e-6)
    assert problem.evaluate(np.full(problem.n_vars, h)).reaction == pytest.approx(ex.LOAD)


@pytest.fixture(scope="module")
def optimum(tmp_path_factory):
    mapdl = StaticCantileverMapdl(str(tmp_path_factory.mktemp("cantilever")))
    problem = ex.make_problem(mapdl, n_elem=50, n_ctrl=8)
    result = ex.optimize(problem)
    return problem, result


def test_optimum_is_feasible(optimum):
    problem, result = optimum
    c = ex.comparison(problem, result.x)
    assert c["utilisation"] <= 1.0 + 1e-3
    lo, hi = ex.H_BOUNDS
    assert np.all(result.x >= lo - 1e-3) and np.all(result.x <= hi + 1e-3)


def test_optimum_matches_the_beam_of_uniform_strength(optimum):
    problem, result = optimum
    c = ex.comparison(problem, result.x)
    # Slightly heavier than the closed form: 8 control points cannot follow
    # the vertical tangent of sqrt(L − s) at the tip, and each element is
    # sized on its root-side node.
    assert 0.0 <= c["mass_error"] < 0.05
    assert c["saving"] > 0.28  # closed form: 1/3 less than the prismatic beam
    assert c["utilisation"] > 0.99  # the stress constraint is active


@pytest.mark.mapdl
def test_cantilever_example_with_mapdl(mapdl_session):
    problem = ex.make_problem(mapdl_session, n_elem=50, n_ctrl=8)
    result = ex.optimize(problem)
    c = ex.comparison(problem, result.x)
    assert c["utilisation"] <= 1.0 + 1e-3
    assert 0.0 <= c["mass_error"] < 0.05
