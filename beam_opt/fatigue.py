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

Assumptions
-----------
* Proportional loading between ``R·P`` and ``P``: the stress at every point
  scales with the load, so one linear analysis at the peak load ``P``
  describes the whole cycle.  This holds for loads and for imposed
  displacements alike.
* The equivalent stress is the von Mises stress taken as positive (tensile):
  conservative for points in compression when the mean stress matters.
* One cycle type (constant amplitude).  Miner's rule for load spectra is a
  natural extension.

No material is predefined: the strain-life constants come from the
application (handbooks, tests).
"""

from __future__ import annotations

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
class FatigueModel:
    """
    Constant-amplitude fatigue of the optimisable beam.

    Parameters
    ----------
    curve : StrainLifeCurve
        Material properties.
    load_ratio : float
        ``R = P_min / P_max`` of the load cycle, ``−1 ≤ R < 1``.  The analysed
        load is the peak ``P_max``: ``R = −1`` is fully reversed, ``R = 0``
        is zero-to-peak (e.g. a clip opened and released).
    kt : float
        Elastic stress concentration factor applied to the beam stress [-]
        (fillets, holes, welds not represented by the beam model).
    mean_stress : str
        ``"swt"`` (Smith-Watson-Topper, default), ``"morrow"`` or ``"none"``.
    """

    curve: StrainLifeCurve
    load_ratio: float = -1.0
    kt: float = 1.0
    mean_stress: str = "swt"

    def __post_init__(self) -> None:
        if not -1.0 <= self.load_ratio < 1.0:
            raise ValueError("load_ratio must satisfy -1 <= R < 1")
        if self.kt < 1.0:
            raise ValueError("kt must be >= 1")
        if self.mean_stress not in ("swt", "morrow", "none"):
            raise ValueError("mean_stress must be 'swt', 'morrow' or 'none'")

    def local_cycle(self, peak_stress: ArrayLike) -> dict[str, np.ndarray]:
        """
        Local elastic-plastic cycle at the peak elastic stress *peak_stress* [MPa].

        Returns
        -------
        dict[str, ndarray]
            ``sigma_max`` [MPa], ``sigma_a`` [MPa], ``sigma_m`` [MPa] and
            ``eps_a`` [-].
        """
        c = self.curve
        s_max = self.kt * np.abs(np.asarray(peak_stress, dtype=float))
        sigma_max, _ = c.neuber(s_max)
        dsig, deps = c.neuber_range((1.0 - self.load_ratio) * s_max)
        sigma_a = 0.5 * dsig
        return {
            "sigma_max": sigma_max,
            "sigma_a": sigma_a,
            "sigma_m": sigma_max - sigma_a,
            "eps_a": 0.5 * deps,
        }

    def life(self, peak_stress: ArrayLike) -> np.ndarray:
        """
        Cycles to crack initiation for the peak elastic stress *peak_stress* [MPa].

        Returns
        -------
        ndarray
            Cycles [-], capped at the run-out ``0.5·10^12``.
        """
        cyc = self.local_cycle(peak_stress)
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
