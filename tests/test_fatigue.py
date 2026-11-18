"""Low-cycle fatigue model (local strain approach) against closed-form relations."""

from __future__ import annotations

import numpy as np
import pytest

from beam_opt.fatigue import FatigueModel, StrainLifeCurve

E = 2.0e5  # [MPa]
# Steel with typical strain-life constants (values of the order of an SAE 1045)
STEEL = StrainLifeCurve(E=E, sigma_f=930.0, b=-0.095, eps_f=0.26, c=-0.47, K=1258.0, n=0.208)
# Elastic only: negligible ductility term and plastic strain
ELASTIC = StrainLifeCurve(E=E, sigma_f=930.0, b=-0.095, eps_f=1e-12, c=-0.6, K=1e9, n=0.2)


def basquin_life(stress_amplitude):
    """σ_a = σ'_f (2N)^b  →  N."""
    return 0.5 * (np.asarray(stress_amplitude) / ELASTIC.sigma_f) ** (1.0 / ELASTIC.b)


@pytest.mark.parametrize("mean_stress", ["none", "morrow", "swt"])
def test_elastic_fully_reversed_life_is_basquin(mean_stress):
    s = np.array([150.0, 250.0, 400.0])
    life = FatigueModel(ELASTIC, load_ratio=-1.0, mean_stress=mean_stress).life(s)
    np.testing.assert_allclose(life, basquin_life(s), rtol=1e-6)


def test_neuber_rule_on_the_cyclic_curve():
    s = np.array([200.0, 450.0, 700.0])
    sigma, eps = STEEL.neuber(s)
    np.testing.assert_allclose(sigma * eps, s**2 / E, rtol=1e-9)
    np.testing.assert_allclose(eps, STEEL.strain(sigma), rtol=1e-12)
    assert np.all(sigma < s) and np.all(eps > s / E)  # yielding: less stress, more strain
    ds = 2.0 * s
    dsig, deps = STEEL.neuber_range(ds)
    np.testing.assert_allclose(dsig * deps, ds**2 / E, rtol=1e-9)


def test_manson_coffin_round_trip_in_the_plastic_range():
    # Pick a life, build the local cycle that gives it, recover the elastic
    # stress by Neuber, and ask the model for the life again.
    for cycles in (2e2, 1e3, 1e4, 1e5):
        eps_a = STEEL.strain_amplitude(cycles)
        sigma_a = _invert(STEEL.strain, eps_a)  # Masing: amplitude on the cyclic curve
        elastic = np.sqrt(E * sigma_a * eps_a)
        life = FatigueModel(STEEL, load_ratio=-1.0, mean_stress="none").life(elastic)
        assert life == pytest.approx(cycles, rel=1e-6)
        if cycles <= 1e3:
            assert eps_a - sigma_a / E > sigma_a / E  # mostly plastic strain: LCF


def _invert(fun, target, lo=0.0, hi=5000.0):
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if fun(mid) > target else (mid, hi)
    return 0.5 * (lo + hi)


def test_mean_stress_shortens_the_life_at_equal_amplitude():
    s_a = 300.0
    reversed_ = FatigueModel(STEEL, load_ratio=-1.0).life(s_a)
    for method in ("swt", "morrow"):
        zero_to_peak = FatigueModel(STEEL, load_ratio=0.0, mean_stress=method).life(2.0 * s_a)
        assert zero_to_peak < reversed_
    # and at equal peak the smaller range of R = 0 lasts longer
    assert FatigueModel(STEEL, load_ratio=0.0).life(s_a) > reversed_


def test_stress_concentration_scales_the_elastic_stress():
    m1 = FatigueModel(STEEL, load_ratio=0.0, kt=1.0)
    m2 = FatigueModel(STEEL, load_ratio=0.0, kt=2.2)
    assert m2.life(200.0) == pytest.approx(m1.life(440.0), rel=1e-9)


def test_life_decreases_with_stress_and_is_capped_at_runout():
    s = np.linspace(150.0, 900.0, 30)  # above the run-out stress
    life = FatigueModel(STEEL, load_ratio=0.0).life(s)
    assert np.all(np.diff(life) < 0.0)
    assert FatigueModel(STEEL).life(1.0) == pytest.approx(0.5e12)


@pytest.mark.parametrize(
    "kwargs",
    [{"load_ratio": 1.0}, {"load_ratio": -1.5}, {"kt": 0.5}, {"mean_stress": "goodman"}],
)
def test_invalid_models_are_rejected(kwargs):
    with pytest.raises(ValueError):
        FatigueModel(STEEL, **kwargs)


def test_invalid_curves_are_rejected():
    with pytest.raises(ValueError, match="negative"):
        StrainLifeCurve(E=E, sigma_f=900.0, b=0.1, eps_f=0.3, c=-0.5, K=1200.0, n=0.2)
    with pytest.raises(ValueError, match="positive"):
        StrainLifeCurve(E=E, sigma_f=-900.0, b=-0.1, eps_f=0.3, c=-0.5, K=1200.0, n=0.2)
