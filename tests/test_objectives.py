"""
Objectives other than the mass, on a tip-loaded cantilever of constant width.

For this statically determinate beam the closed forms are known: at a given
mass, minimum compliance, minimum tip deflection, minimum peak stress and
maximum fatigue life all lead to the same profile h(s) = c·sqrt(L − s), the
beam of uniform strength.
"""

from __future__ import annotations

import contextlib
import io
import itertools

import numpy as np
import pytest
from test_fatigue import STEEL

from beam_opt.fatigue import FatigueModel
from beam_opt.geometry import BeamPath, discretize
from beam_opt.materials import Material
from beam_opt.model import Clamp, NodalForce, OptimizableBeam, StructuralModel
from beam_opt.native import NativeSolver
from beam_opt.optimization import (
    Compliance,
    Displacement,
    FatigueLife,
    Mass,
    OptimizationProblem,
    PeakStress,
    WeightedSum,
    epsilon_constraint,
    ks_max,
    optimize,
    read_history,
)
from beam_opt.report import print_report
from beam_opt.sections import RectSection

L, B, P = 200.0, 5.0, 20.0  # [mm], [mm], [N]
MAT = Material(E=2.0e5, nu=0.3, rho=7.85e-9, sigma_y=355.0)
H0 = 6.0  # root height of the reference sqrt profile [mm]
C = H0 / np.sqrt(L)  # h = C·sqrt(L − s)
MASS = MAT.rho * B * C * (2.0 / 3.0) * L**1.5 * 1e6  # mass of that profile [g]


def make(objective=None, n_elem=60, n_ctrl=10, **kwargs) -> OptimizationProblem:
    path = BeamPath([[0.0, 0.0, 0.0], [L, 0.0, 0.0]], kind="polyline")
    model = StructuralModel(
        OptimizableBeam(discretize(path, n_elem), section_factory=RectSection.with_fixed(b=B)),
        MAT,
        bcs=[Clamp("start"), NodalForce("end", "FZ", -P)],
    )
    options = {"safety_factor": None, "interpolator": "pchip", "mass_max": MASS} | kwargs
    return OptimizationProblem(
        NativeSolver(shear_deformation=False),
        model,
        n_ctrl=n_ctrl,
        bounds={"h": (0.5, 12.0)},
        objective=objective,
        **options,
    )


def discrete_uniform_stress(problem) -> float:
    """
    Stress of the discrete beam of uniform strength that has the mass budget [MPa].

    Every element has a constant section and is governed by its root-side
    node: h_e = sqrt(6 P (L − s_I) / (b σ)), not below the lower bound.  The
    continuous closed form 6 P L / (b h0²) is about 3 % lower with 60
    elements.
    """
    mesh = problem.model.beam.mesh
    arm = L - mesh.s[:-1]
    lo_h = problem.bounds["h"][0]

    def mass(sigma):
        h = np.maximum(np.sqrt(6.0 * P * arm / (B * sigma)), lo_h)
        return MAT.rho * B * np.sum(h * mesh.lengths) * 1e6

    lo, hi = 1.0, 1e4
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        lo, hi = (mid, hi) if mass(mid) > MASS else (lo, mid)
    return float(np.sqrt(lo * hi))


def profile_error(problem, x) -> float:
    """Largest relative deviation of h(s) from the sqrt profile, away from the tip."""
    s = problem.s_ctrl[:-2]
    h = problem.law_from_x(x).eval(s)["h"]
    return float(np.max(np.abs(h / (C * np.sqrt(L - s)) - 1.0)))


# ── Compliance and displacement ──────────────────────────────────────────


def test_compliance_of_a_prismatic_beam_is_p_delta():
    problem = make(Compliance())
    h = 4.0
    inertia = B * h**3 / 12.0
    expected = P**2 * L**3 / (3.0 * MAT.E * inertia)  # P · δ
    assert problem.objective(np.full(problem.n_vars, h)) == pytest.approx(expected, rel=1e-8)


@pytest.fixture(scope="module")
def stiffest():
    problem = make(Compliance())
    return problem, optimize(problem, "slsqp", x0=np.full(problem.n_vars, 4.0))


def test_minimum_compliance_at_given_mass_is_the_sqrt_profile(stiffest):
    problem, result = stiffest
    c_exact = 12.0 * P**2 / (MAT.E * B * C**3) * (2.0 / 3.0) * L**1.5
    assert result.success and result.max_violation <= 1e-4
    assert problem.mass_g(result.x) == pytest.approx(MASS, rel=1e-3)  # budget active
    assert result.fun == pytest.approx(c_exact, rel=0.02)
    assert profile_error(problem, result.x) < 0.03


def test_stiffest_beam_beats_the_prismatic_beam_of_equal_mass(stiffest):
    problem, result = stiffest
    h_uniform = MASS / 1e6 / (MAT.rho * B * L)
    prismatic = problem.objective(np.full(problem.n_vars, h_uniform))
    # closed form: C_prismatic = 27/2 P² L^1.5 / (E b c³), C_optimum = 8 (same factor)
    assert prismatic / result.fun == pytest.approx(27.0 / 16.0, rel=0.02)


def test_tip_deflection_objective_matches_compliance(stiffest):
    _, stiff = stiffest
    problem = make(Displacement("end", "UZ"))
    result = optimize(problem, "slsqp", x0=np.full(problem.n_vars, 4.0))
    assert result.fun == pytest.approx(stiff.fun / P, rel=1e-3)
    np.testing.assert_allclose(result.x, stiff.x, rtol=0.02, atol=0.05)


def test_without_a_mass_budget_the_stiffest_beam_is_the_heaviest():
    problem = make(Compliance(), mass_max=None)
    result = optimize(problem, "slsqp", x0=np.full(problem.n_vars, 4.0))
    np.testing.assert_allclose(result.x, problem.upper, atol=1e-6)


# ── Peak stress ──────────────────────────────────────────────────────────


def test_ks_envelope_bounds_the_maximum():
    v = np.array([0.2, 0.9, 1.0, 0.95])
    for rho in (10.0, 50.0, 200.0):
        k = ks_max(v, rho)
        assert 1.0 <= k <= 1.0 + np.log(len(v)) / rho


def test_minimum_peak_stress_at_given_mass():
    problem = make(PeakStress(), safety_factor=1.0)
    result = optimize(problem, "slsqp", x0=np.full(problem.n_vars, 4.0))
    sigma = problem.evaluate(result.x).sigma
    assert sigma.max() == pytest.approx(discrete_uniform_stress(problem), rel=0.01)
    assert sigma.max() == pytest.approx(6.0 * P * L / (B * H0**2), rel=0.04)  # continuous
    body = sigma[: int(0.8 * len(sigma))]
    assert body.std() / body.mean() < 0.005  # uniform strength
    assert profile_error(problem, result.x) < 0.04


# ── Fatigue life ─────────────────────────────────────────────────────────

ZERO_TO_PEAK = FatigueModel(STEEL, load_ratio=0.0)


def test_maximum_fatigue_life_at_given_mass():
    # Longest life = lowest peak stress: the uniform-strength beam again.  This
    # optimum lies in the high-cycle range, where 1 % of stress is 0.1 decades
    # of life, so the check is on the stress, the life follows.
    problem = make(FatigueLife(), fatigue=ZERO_TO_PEAK)
    result = optimize(problem, "slsqp", x0=np.full(problem.n_vars, 4.0))
    ev = problem.evaluate(result.x)
    lives = problem.lives(ev)
    exact_stress = discrete_uniform_stress(problem)
    assert result.success and result.max_violation <= 1e-4
    assert ev.sigma.max() == pytest.approx(exact_stress, rel=0.01)
    assert np.log10(lives.min()) == pytest.approx(
        np.log10(ZERO_TO_PEAK.life(exact_stress)), abs=0.1
    )
    body = np.log10(lives[: int(0.8 * len(lives))])
    assert np.ptp(body) < 0.02  # every element has the same life
    assert result.fun >= -np.log10(lives.min())  # the KS envelope is conservative
    assert result.fun == pytest.approx(-np.log10(lives.min()), abs=0.1)


def test_minimum_mass_for_a_required_fatigue_life():
    # The uniform-strength beam whose stress gives exactly the required life
    n_req = 2.0e4
    lo, hi = 1.0, 2000.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if ZERO_TO_PEAK.life(mid) > n_req else (lo, mid)
    s_req = 0.5 * (lo + hi)
    h_root = np.sqrt(6.0 * P * L / (B * s_req))
    mass_ref = MAT.rho * B * (2.0 / 3.0) * h_root * L * 1e6

    problem = make(Mass(), mass_max=None, fatigue=ZERO_TO_PEAK, life_min=n_req, n_ctrl=8)
    result = optimize(problem, "slsqp", x0=np.full(problem.n_vars, 6.0))
    lives = problem.lives(problem.evaluate(result.x))
    assert result.success and result.max_violation <= 1e-4
    assert lives.min() == pytest.approx(n_req, rel=0.01)
    assert 0.0 <= result.fun / mass_ref - 1.0 < 0.03
    assert s_req > MAT.sigma_y  # local yielding at the root: a low-cycle fatigue design


def test_life_constraint_needs_a_fatigue_model():
    with pytest.raises(ValueError, match="life_min needs"):
        make(life_min=1e4)
    with pytest.raises(ValueError, match="FatigueLife needs"):
        make(FatigueLife())


# ── Weighted sum, Pareto front ───────────────────────────────────────────


def test_weighted_sum_uses_the_references():
    problem = make(WeightedSum([(Mass(), 2.0), (Compliance(), 1.0)], references=[10.0, 50.0]))
    x = np.full(problem.n_vars, 4.0)
    ev = problem.evaluate(x)
    expected = 2.0 * ev.mass * 1e6 / 10.0 + Compliance().value(problem, ev) / 50.0
    assert problem.objective(x) == pytest.approx(expected)


def test_weighted_sum_finds_a_compromise():
    lightest = optimize(make(Mass(), mass_max=None, safety_factor=1.5), "slsqp")
    problem = make(WeightedSum([(Mass(), 1.0), (Compliance(), 1.0)]), mass_max=None)
    mixed = optimize(problem, "slsqp", x0=np.full(problem.n_vars, 4.0))
    m = problem.mass_g(mixed.x)
    assert problem.mass_g(problem.lower) < m < problem.mass_g(problem.upper)
    assert np.isfinite(lightest.fun)


def test_epsilon_constraint_traces_a_monotone_front():
    problem = make(Compliance(), n_ctrl=6)
    levels = [0.6 * MASS, 0.8 * MASS, MASS, 1.3 * MASS]
    front = epsilon_constraint(problem, "mass_max", levels, x0=np.full(problem.n_vars, 3.0))
    assert problem.mass_max == pytest.approx(MASS)  # restored
    assert all(p.feasible for p in front)
    compliance = [p.objective for p in front]
    assert all(a > b for a, b in itertools.pairwise(compliance))
    for p in front:
        assert p.mass == pytest.approx(p.level, rel=2e-3)  # the budget is always used
    # C ∝ 1/m³ for geometrically similar sqrt profiles
    ratio = compliance[0] / compliance[-1]
    assert ratio == pytest.approx((1.3 / 0.6) ** 3, rel=0.05)


def test_epsilon_constraint_rejects_unknown_limits():
    with pytest.raises(ValueError, match="attribute must be"):
        epsilon_constraint(make(Compliance()), "n_ctrl", [1])


# ── Plumbing ─────────────────────────────────────────────────────────────


def test_failed_analysis_gets_a_bad_objective_and_is_never_returned():
    problem = make(Compliance())
    problem.model.bcs[:] = [NodalForce("end", "FZ", -P)]  # no supports: singular
    x = np.full(problem.n_vars, 4.0)
    assert problem.objective(x) == pytest.approx(problem.failure_penalty)
    assert problem.max_violation(problem.evaluate(x)) == problem.failure_penalty
    with pytest.raises(RuntimeError, match="every FEM evaluation"):
        optimize(problem, "cobyla", x0=x, max_evals=20)


def test_history_records_the_objective(tmp_path):
    log = str(tmp_path / "h.csv")
    problem = make(Compliance(), n_ctrl=5, history_file=log)
    result = optimize(problem, "slsqp", x0=np.full(problem.n_vars, 4.0))
    h = read_history(log)
    k = int(np.argmin(np.where(h["max_violation"] <= 1e-4, h["objective"], np.inf)))
    assert h["objective"][k] == pytest.approx(result.fun, rel=1e-9)
    np.testing.assert_allclose(h["x"][k], result.x)


def test_report_shows_objective_fatigue_and_budget():
    problem = make(FatigueLife(), fatigue=ZERO_TO_PEAK, n_ctrl=5)
    result = optimize(problem, "slsqp", x0=np.full(problem.n_vars, 4.0))
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        print_report(result, problem)
    text = out.getvalue()
    for piece in (
        "[OBJECTIVE]",
        "fatigue life:",
        "[FATIGUE LIFE]",
        "Budget",
        "no stress constraint",
    ):
        assert piece in text
