"""
beam_opt.fatigue
================
Low-cycle fatigue life by the local strain approach.

The structural analysis is linear: it gives the *elastic* stress at the
peak of the load cycle.  The elastic-plastic response at the critical point
is recovered with Neuber's rule on the cyclic stress-strain curve, the cycle
with Masing's hypothesis, and the life from the strain-life curve:

    cyclic curve       ε = σ/E + (σ/K')^(1/n')                       (Ramberg-Osgood)
    Neuber, peak       σ_max ε_max = S_max² / E
    Neuber, range      Δσ Δε = ΔS² / E,   Δε = Δσ/E + 2 (Δσ / 2K')^(1/n')   (Masing)
    strain-life        ε_a = σ'_f/E (2N)^b + ε'_f (2N)^c               (Manson-Coffin-Basquin)

with the mean stress accounted for by Smith-Watson-Topper

    σ_max ε_a = σ'_f²/E (2N)^(2b) + σ'_f ε'_f (2N)^(b+c)

or by Morrow, ``ε_a = (σ'_f − σ_m)/E (2N)^b + ε'_f (2N)^c``.

Variable amplitude
------------------
A ``LoadSpectrum`` lists blocks of cycles between ``λ_min·P`` and
``λ_max·P``; it can be obtained from a load history by rainflow counting
(``LoadSpectrum.from_history``).  Damage is summed with Miner's rule,
``D = Σ n_k / N_k``, and the life is the number of repetitions of the
spectrum, ``1 / D``.

Assumptions
-----------
* Proportional loading: the stress at every point scales with the load, so
  one linear analysis at the load ``P`` describes every cycle.  This holds
  for loads and for imposed displacements alike, as long as one load
  pattern is scaled (several independent load cases are not covered).
* The equivalent stress is the von Mises stress taken as positive (tensile):
  conservative for points in compression when the mean stress matters.
* Miner's rule is linear: load-sequence effects and the memory of the
  hysteresis loops across blocks are neglected.

No material is predefined: the strain-life constants come from the
application (handbooks, tests).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

#: Lives are computed up to 2N = 10^_LOG2N_MAX reversals ("run-out").
_LOG2N_MAX = 12
_BISECTION_STEPS = 80


def _bisect(fun, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Root of an increasing function between *lo* and *hi*, element-wise."""
    lo, hi = np.array(lo, dtype=float), np.array(hi, dtype=float)
    for _ in range(_BISECTION_STEPS):
        mid = 0.5 * (lo + hi)
        above = fun(mid) > 0.0
        hi = np.where(above, mid, hi)
        lo = np.where(above, lo, mid)
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class StrainLifeCurve:
    """
    Cyclic stress-strain and strain-life properties of a material.

    Parameters
    ----------
    E : float
        Young's modulus [MPa].
    sigma_f : float
        Fatigue strength coefficient σ'_f [MPa].
    b : float
        Fatigue strength exponent [-] (negative, typically −0.05 to −0.12).
    eps_f : float
        Fatigue ductility coefficient ε'_f [-].
    c : float
        Fatigue ductility exponent [-] (negative, typically −0.4 to −0.7).
    K : float
        Cyclic strength coefficient K' [MPa].
    n : float
        Cyclic strain-hardening exponent n' [-].
    """

    E: float
    sigma_f: float
    b: float
    eps_f: float
    c: float
    K: float
    n: float

    def __post_init__(self) -> None:
        if min(self.E, self.sigma_f, self.eps_f, self.K, self.n) <= 0.0:
            raise ValueError("E, sigma_f, eps_f, K and n must be positive")
        if self.b >= 0.0 or self.c >= 0.0:
            raise ValueError("the exponents b and c must be negative")

    # -- cyclic curve ---------------------------------------------------

    def strain(self, sigma: ArrayLike) -> np.ndarray:
        """Strain on the cyclic curve at stress *sigma* [MPa] (Ramberg-Osgood) [-]."""
        s = np.asarray(sigma, dtype=float)
        return s / self.E + (s / self.K) ** (1.0 / self.n)

    def strain_range(self, stress_range: ArrayLike) -> np.ndarray:
        """Strain range for a stress range on the hysteresis loop (Masing) [-]."""
        ds = np.asarray(stress_range, dtype=float)
        return ds / self.E + 2.0 * (ds / (2.0 * self.K)) ** (1.0 / self.n)

    def neuber(self, elastic_stress: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """
        Local stress and strain from the elastic stress by Neuber's rule.

        Solves ``σ ε(σ) = S² / E`` on the cyclic curve.

        Returns
        -------
        sigma, eps : ndarray
            Local stress [MPa] and strain [-].
        """
        s = np.abs(np.asarray(elastic_stress, dtype=float))
        target = s**2 / self.E
        sigma = _bisect(lambda x: x * self.strain(x) - target, np.zeros_like(s), s)
        return sigma, self.strain(sigma)

    def neuber_range(self, elastic_range: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Local stress and strain ranges from the elastic stress range (Neuber + Masing)."""
        ds = np.abs(np.asarray(elastic_range, dtype=float))
        target = ds**2 / self.E
        dsig = _bisect(lambda x: x * self.strain_range(x) - target, np.zeros_like(ds), ds)
        return dsig, self.strain_range(dsig)

    # -- strain-life ------------------------------------------------------

    def strain_amplitude(self, cycles: ArrayLike, mean_stress: ArrayLike = 0.0) -> np.ndarray:
        """Strain amplitude that gives *cycles* to failure (Morrow form) [-]."""
        rev = 2.0 * np.asarray(cycles, dtype=float)
        sf = self.sigma_f - np.asarray(mean_stress, dtype=float)
        return sf / self.E * rev**self.b + self.eps_f * rev**self.c

    def swt_parameter(self, cycles: ArrayLike) -> np.ndarray:
        """Smith-Watson-Topper parameter σ_max ε_a that gives *cycles* to failure [MPa]."""
        rev = 2.0 * np.asarray(cycles, dtype=float)
        return self.sigma_f**2 / self.E * rev ** (
            2.0 * self.b
        ) + self.sigma_f * self.eps_f * rev ** (self.b + self.c)


@dataclass(frozen=True)
class LoadBlock:
    """
    ``cycles`` load cycles between ``min_factor·P`` and ``max_factor·P``.

    ``P`` is the load of the structural analysis; the factors may be
    negative (load reversal).

    Parameters
    ----------
    max_factor, min_factor : float
        Extreme load factors of the cycle [-], ``min_factor <= max_factor``.
    cycles : float
        Number of cycles per repetition of the spectrum [-] (may be 0.5:
        a half cycle from rainflow counting).
    """

    max_factor: float
    min_factor: float
    cycles: float

    def __post_init__(self) -> None:
        if self.min_factor > self.max_factor:
            raise ValueError("min_factor must not exceed max_factor")
        if self.cycles < 0.0:
            raise ValueError("cycles must be non-negative")

    @property
    def peak(self) -> float:
        """Largest absolute load factor of the cycle [-]."""
        return max(abs(self.max_factor), abs(self.min_factor))

    @property
    def range(self) -> float:
        """Load-factor range of the cycle [-]."""
        return self.max_factor - self.min_factor


def rainflow(history: ArrayLike) -> list[tuple[float, float, float]]:
    """
    Rainflow cycle counting of a load history (ASTM E1049-85, section 5.4.4).

    Parameters
    ----------
    history : array-like
        Load (factor) values in time order [-].

    Returns
    -------
    list[tuple[float, float, float]]
        ``(max, min, count)`` of every counted range, with ``count`` 1.0 for a
        full cycle and 0.5 for a half cycle.
    """
    values = np.asarray(history, dtype=float).ravel()
    # keep only the reversals (peaks and valleys)
    reversals: list[float] = []
    for v in values:
        if reversals and v == reversals[-1]:
            continue
        if len(reversals) >= 2 and (reversals[-1] - reversals[-2]) * (v - reversals[-1]) > 0.0:
            reversals[-1] = v  # same direction: extend the excursion
        else:
            reversals.append(float(v))

    counted: list[tuple[float, float, float]] = []
    stack: list[float] = []
    for point in reversals:
        stack.append(point)
        while len(stack) >= 3:
            x = abs(stack[-1] - stack[-2])
            y = abs(stack[-2] - stack[-3])
            if x < y:
                break
            a, b = stack[-3], stack[-2]
            if len(stack) == 3:  # Y contains the starting point: half cycle
                counted.append((max(a, b), min(a, b), 0.5))
                stack.pop(0)
            else:
                counted.append((max(a, b), min(a, b), 1.0))
                del stack[-3:-1]
    for a, b in itertools.pairwise(stack):
        counted.append((max(a, b), min(a, b), 0.5))
    return counted


@dataclass(frozen=True)
class LoadSpectrum:
    """
    Variable-amplitude loading: blocks of cycles, repeated.

    Damage is accumulated with Miner's linear rule, ``D = Σ n_k / N_k`` per
    repetition, so the life is a number of *repetitions* of the spectrum
    (missions, days, kilometres...).  Load-sequence effects are neglected.

    Parameters
    ----------
    blocks : tuple[LoadBlock, ...]
        Cycles of one repetition.
    name : str
        What one repetition is (e.g. ``"day"``), for reports.
    """

    blocks: tuple[LoadBlock, ...]
    name: str = "repetition"

    def __post_init__(self) -> None:
        if not self.blocks:
            raise ValueError("a spectrum needs at least one block")

    @classmethod
    def from_history(cls, history: ArrayLike, name: str = "repetition", decimals: int = 6):
        """
        Spectrum of a load-factor history by rainflow counting.

        Equal cycles (after rounding the factors to *decimals*) are merged
        into one block.
        """
        merged: dict[tuple[float, float], float] = {}
        for hi, lo, count in rainflow(history):
            if hi == lo:
                continue
            key = (round(hi, decimals), round(lo, decimals))
            merged[key] = merged.get(key, 0.0) + count
        blocks = tuple(LoadBlock(hi, lo, n) for (hi, lo), n in sorted(merged.items()))
        return cls(blocks, name)

    @property
    def cycles(self) -> float:
        """Number of cycles per repetition [-]."""
        return float(sum(b.cycles for b in self.blocks))


@dataclass(frozen=True)
class FatigueModel:
    """
    Fatigue of the optimisable beam under constant or variable amplitude.

    Parameters
    ----------
    curve : StrainLifeCurve
        Material properties.
    load_ratio : float
        Constant amplitude: ``R = P_min / P_max`` of the load cycle,
        ``−1 ≤ R < 1``.  The analysed load is the peak ``P_max``: ``R = −1``
        is fully reversed, ``R = 0`` is zero-to-peak (e.g. a clip opened and
        released).  Ignored when a *spectrum* is given.
    kt : float
        Elastic stress concentration factor applied to the beam stress [-]
        (fillets, holes, welds not represented by the beam model).
    mean_stress : str
        ``"swt"`` (Smith-Watson-Topper, default), ``"morrow"`` or ``"none"``.
    spectrum : LoadSpectrum | None
        Variable-amplitude loading (Miner's rule).  Lives are then counted in
        repetitions of the spectrum.
    """

    curve: StrainLifeCurve
    load_ratio: float = -1.0
    kt: float = 1.0
    mean_stress: str = "swt"
    spectrum: LoadSpectrum | None = None

    def __post_init__(self) -> None:
        if not -1.0 <= self.load_ratio < 1.0:
            raise ValueError("load_ratio must satisfy -1 <= R < 1")
        if self.kt < 1.0:
            raise ValueError("kt must be >= 1")
        if self.mean_stress not in ("swt", "morrow", "none"):
            raise ValueError("mean_stress must be 'swt', 'morrow' or 'none'")

    @property
    def unit(self) -> str:
        """Unit of the lives: ``"cycles"`` or the repetitions of the spectrum."""
        return "cycles" if self.spectrum is None else f"{self.spectrum.name}s"

    def local_cycle(
        self, peak_stress: ArrayLike, stress_range: ArrayLike | None = None
    ) -> dict[str, np.ndarray]:
        """
        Local elastic-plastic cycle from the elastic stresses [MPa].

        Parameters
        ----------
        peak_stress : array-like
            Largest elastic stress of the cycle [MPa].
        stress_range : array-like | None
            Elastic stress range [MPa]; defaults to ``(1 − R)·peak_stress``.

        Returns
        -------
        dict[str, ndarray]
            ``sigma_max`` [MPa], ``sigma_a`` [MPa], ``sigma_m`` [MPa] and
            ``eps_a`` [-].
        """
        c = self.curve
        s_max = self.kt * np.abs(np.asarray(peak_stress, dtype=float))
        if stress_range is None:
            s_range = (1.0 - self.load_ratio) * s_max
        else:
            s_range = self.kt * np.abs(np.asarray(stress_range, dtype=float))
        sigma_max, _ = c.neuber(s_max)
        dsig, deps = c.neuber_range(s_range)
        sigma_a = 0.5 * dsig
        return {
            "sigma_max": sigma_max,
            "sigma_a": sigma_a,
            "sigma_m": sigma_max - sigma_a,
            "eps_a": 0.5 * deps,
        }

    def cycle_life(self, peak_stress: ArrayLike, stress_range: ArrayLike | None = None):
        """
        Cycles to crack initiation of one cycle type (see ``local_cycle``).

        Returns
        -------
        ndarray
            Cycles [-], capped at the run-out ``0.5·10^12``.
        """
        cyc = self.local_cycle(peak_stress, stress_range)
        c = self.curve
        lo = np.zeros_like(cyc["eps_a"])  # log10(2N)
        hi = np.full_like(lo, float(_LOG2N_MAX))

        if self.mean_stress == "swt":
            demand = np.maximum(cyc["sigma_max"], 0.0) * cyc["eps_a"]

            def excess(log_rev):
                return demand - c.swt_parameter(0.5 * 10.0**log_rev)

        else:
            mean = cyc["sigma_m"] if self.mean_stress == "morrow" else 0.0

            def excess(log_rev):
                return cyc["eps_a"] - c.strain_amplitude(0.5 * 10.0**log_rev, mean)

        # capacity decreases with the life: excess(log 2N) increases
        log_rev = _bisect(excess, lo, hi)
        return 0.5 * 10.0**log_rev

    def damage(self, stress: ArrayLike) -> np.ndarray:
        """
        Miner damage of one repetition (spectrum) or of one cycle (constant amplitude).

        Parameters
        ----------
        stress : array-like
            Elastic stress under the analysed load ``P`` [MPa].
        """
        s = np.abs(np.asarray(stress, dtype=float))
        if self.spectrum is None:
            return self._cycle_damage(self.cycle_life(s), 1.0)
        blocks = [b for b in self.spectrum.blocks if b.cycles > 0.0 and b.range > 0.0]
        if not blocks:
            return np.zeros_like(s)
        # every block at once: arrays of shape (n_blocks, *s.shape)
        peak = np.array([b.peak for b in blocks]).reshape((-1,) + (1,) * s.ndim)
        rng = np.array([b.range for b in blocks]).reshape(peak.shape)
        cycles = np.array([b.cycles for b in blocks]).reshape(peak.shape)
        lives = self.cycle_life(peak * s, rng * s)
        return np.sum(self._cycle_damage(lives, cycles), axis=0)

    @staticmethod
    def _cycle_damage(lives: np.ndarray, cycles: float) -> np.ndarray:
        """``cycles / N``, zero for cycles at the run-out (below the fatigue limit)."""
        runout = lives >= 0.5 * 10.0**_LOG2N_MAX * (1.0 - 1e-9)
        return np.where(runout, 0.0, cycles / lives)

    def life(self, stress: ArrayLike) -> np.ndarray:
        """
        Fatigue life for the elastic stress *stress* under the analysed load [MPa].

        Constant amplitude: cycles to crack initiation (capped at the run-out
        ``0.5·10^12``).  Spectrum: repetitions of the spectrum, ``1 / D``.
        """
        if self.spectrum is None:
            return self.cycle_life(stress)
        damage = self.damage(stress)
        cap = 0.5 * 10.0**_LOG2N_MAX
        return np.where(damage > 0.0, 1.0 / np.maximum(damage, 1.0 / cap), cap)
