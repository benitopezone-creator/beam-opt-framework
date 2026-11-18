"""Variable-amplitude fatigue: rainflow counting and Miner's rule."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
from test_fatigue import ELASTIC, STEEL
from test_objectives import make

from beam_opt.extras import FatigueLife, FatigueModel, LoadBlock, LoadSpectrum, rainflow
from beam_opt.optimization import Mass, optimize

# ── Rainflow counting ────────────────────────────────────────────────────


def _ranges(cycles) -> dict[float, float]:
    out = Counter()
    for hi, lo, n in cycles:
        out[hi - lo] += n
    return dict(sorted(out.items()))


def test_rainflow_reproduces_the_astm_e1049_example():
    # ASTM E1049-85, Fig. 6: rainflow counting of the history below
    history = [-2, 1, -3, 5, -1, 3, -4, 4, -2]
    assert _ranges(rainflow(history)) == {3: 0.5, 4: 1.5, 6: 0.5, 8: 1.0, 9: 0.5}


def test_rainflow_ignores_points_that_are_not_reversals():
    dense = [0.0, 0.5, 1.0, 1.0, 0.4, -0.2, -1.0, -0.3, 2.0, 2.0, 1.0, 0.0]
    sparse = [0.0, 1.0, -1.0, 2.0, 0.0]
    assert rainflow(dense) == rainflow(sparse)


def test_repeated_block_gives_full_cycles():
    history = [0.0] + [1.0, 0.0] * 50
    counts = _ranges(rainflow(history))
    assert counts == {1.0: 50.0}  # 99 half cycles of the residue would be wrong


def test_spectrum_from_history_merges_equal_cycles():
    rng = np.random.default_rng(3)
    peaks = rng.choice([0.6, 0.8, 1.0], size=200)
    history = np.ravel(np.column_stack([np.zeros_like(peaks), peaks]))
    spectrum = LoadSpectrum.from_history(np.append(history, 0.0), name="day")
    assert {(b.max_factor, b.min_factor) for b in spectrum.blocks} == {
        (0.6, 0.0), (0.8, 0.0), (1.0, 0.0)
    }  # fmt: skip
    assert spectrum.cycles == pytest.approx(200.0)
    for level in (0.6, 0.8, 1.0):
        block = next(b for b in spectrum.blocks if b.max_factor == level)
        assert block.cycles == pytest.approx(np.sum(peaks == level), abs=0.5)


# ── Miner's rule ─────────────────────────────────────────────────────────


def test_one_block_spectrum_is_constant_amplitude():
    s = np.array([200.0, 400.0, 650.0])
    constant = FatigueModel(STEEL, load_ratio=0.0).life(s)
    spectrum = LoadSpectrum((LoadBlock(1.0, 0.0, 1.0),))
    np.testing.assert_allclose(FatigueModel(STEEL, spectrum=spectrum).life(s), constant, rtol=1e-12)
    reversed_ = LoadSpectrum((LoadBlock(1.0, -1.0, 1.0),))
    np.testing.assert_allclose(
        FatigueModel(STEEL, spectrum=reversed_).life(s),
        FatigueModel(STEEL, load_ratio=-1.0).life(s),
        rtol=1e-12,
    )


def test_damage_is_additive():
    s = 500.0
    blocks = (LoadBlock(1.0, 0.0, 30.0), LoadBlock(0.7, 0.2, 400.0), LoadBlock(0.4, -0.4, 2e3))
    model = FatigueModel(STEEL, spectrum=LoadSpectrum(blocks))
    single = FatigueModel(STEEL)
    expected = sum(b.cycles / single.cycle_life(b.peak * s, b.range * s) for b in blocks)
    assert model.damage(s) == pytest.approx(expected, rel=1e-12)
    assert model.life(s) == pytest.approx(1.0 / expected, rel=1e-12)


def test_elastic_spectrum_matches_basquin_in_closed_form():
    # N(S) = ½ (S / σ'_f)^(1/b)  →  D = Σ n_k / N(λ_k S)
    s, b, sf = 300.0, ELASTIC.b, ELASTIC.sigma_f
    levels = [(1.0, 10.0), (0.8, 100.0), (0.5, 1000.0)]
    spectrum = LoadSpectrum(tuple(LoadBlock(lam, -lam, n) for lam, n in levels))
    model = FatigueModel(ELASTIC, mean_stress="none", spectrum=spectrum)
    damage = sum(n * 2.0 * (lam * s / sf) ** (-1.0 / b) for lam, n in levels)
    assert model.life(s) == pytest.approx(1.0 / damage, rel=1e-6)


def test_lives_are_counted_in_repetitions():
    spectrum = LoadSpectrum((LoadBlock(1.0, 0.0, 50.0),), name="day")
    model = FatigueModel(STEEL, spectrum=spectrum)
    assert model.unit == "days" and FatigueModel(STEEL).unit == "cycles"
    assert model.life(450.0) == pytest.approx(FatigueModel(STEEL, load_ratio=0.0).life(450.0) / 50)


def test_small_cycles_below_the_run_out_add_no_damage():
    spectrum = LoadSpectrum((LoadBlock(1e-6, 0.0, 1e6),))
    assert FatigueModel(STEEL, spectrum=spectrum).life(100.0) == pytest.approx(0.5e12)


@pytest.mark.parametrize(
    ("args", "message"),
    [((0.0, 1.0, 1.0), "min_factor"), ((1.0, 0.0, -1.0), "non-negative")],
)
def test_invalid_blocks_are_rejected(args, message):
    with pytest.raises(ValueError, match=message):
        LoadBlock(*args)


def test_empty_spectrum_is_rejected():
    with pytest.raises(ValueError, match="at least one block"):
        LoadSpectrum(())


# ── Optimisation with a spectrum ─────────────────────────────────────────

SERVICE = LoadSpectrum(
    (LoadBlock(1.0, 0.0, 5.0), LoadBlock(0.6, 0.0, 200.0), LoadBlock(0.3, -0.3, 2000.0)),
    name="mission",
)
FATIGUE = FatigueModel(STEEL, spectrum=SERVICE)


def test_longest_service_life_is_still_the_beam_of_uniform_strength():
    problem = make(FatigueLife(), fatigue=FATIGUE, n_ctrl=8)
    result = optimize(problem, "slsqp", x0=np.full(problem.n_vars, 4.0))
    lives = problem.lives(problem.evaluate(result.x))
    body = np.log10(lives[: int(0.8 * len(lives))])
    assert result.success and result.max_violation <= 1e-4
    assert np.ptp(body) < 0.05  # uniform damage, within about 1 % of life
    assert result.fun == pytest.approx(-np.log10(lives.min()), abs=1e-9)


def test_minimum_mass_for_a_service_life_in_missions():
    required = 2.0e3  # missions
    problem = make(Mass(), mass_max=None, fatigue=FATIGUE, life_min=required, n_ctrl=8)
    result = optimize(problem, "slsqp", x0=np.full(problem.n_vars, 6.0))
    lives = problem.lives(problem.evaluate(result.x))
    assert result.success and result.max_violation <= 1e-4
    assert lives.min() == pytest.approx(required, rel=0.01)
    # the full cycles dominate the damage, but not alone: without them the life is longer
    ev = problem.evaluate(result.x)
    k = int(np.argmin(lives))
    alone = FatigueModel(STEEL, spectrum=LoadSpectrum(SERVICE.blocks[:1])).life(ev.sigma[k])
    assert alone > lives.min()
