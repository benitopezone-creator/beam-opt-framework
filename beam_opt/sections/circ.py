"""
beam_opt.sections.circ
======================
Circular cross-sections: solid (``CSOLID``) and tube (``CTUBE``).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike

from .base import Section, StressPoints

#: Check points on the outer circle.  The bending maximum falls at most half
#: a step away from a point: the stress is underestimated by less than
#: 1 − cos(π/36) = 0.4 %.
N_ANGLES = 36


class TubeSection(Section):
    """
    Circular tube of outer radius *r* and wall thickness *t*.

    Parameters
    ----------
    r : float or array-like
        Outer radius [mm].
    t : float or array-like
        Wall thickness [mm], ``0 < t <= r`` (``t = r`` is the solid circle).
    """

    param_names = ("r", "t")

    def __init__(self, r: ArrayLike, t: ArrayLike) -> None:
        self.r = np.asarray(r, dtype=float)
        self.t = np.asarray(t, dtype=float)

    @classmethod
    def from_params(cls, params: Mapping[str, ArrayLike]) -> TubeSection:
        return cls(params["r"], params["t"])

    @property
    def r_inner(self) -> np.ndarray:
        """Inner radius [mm]."""
        return np.maximum(self.r - self.t, 0.0)

    @property
    def area(self) -> np.ndarray:
        return np.pi * (self.r**2 - self.r_inner**2)

    @property
    def iy(self) -> np.ndarray:
        return np.pi / 4.0 * (self.r**4 - self.r_inner**4)

    @property
    def iz(self) -> np.ndarray:
        return self.iy

    @property
    def torsion_constant(self) -> np.ndarray:
        return 2.0 * self.iy  # polar second moment: exact for circular sections

    @property
    def torsion_modulus(self) -> np.ndarray:
        return self.torsion_constant / self.r

    @property
    def shear_factor(self) -> np.ndarray:
        """
        τ_max · A / V at the neutral axis [-] (``V·Q / (I·width)``).

        4/3 for the solid circle, 2 for a thin tube.
        """
        ro, ri = self.r, self.r_inner
        return 4.0 / 3.0 * (ro**2 + ro * ri + ri**2) / (ro**2 + ri**2)

    def stress_points(self) -> StressPoints:
        """
        ``N_ANGLES`` points on the outer circle, at y = r cos θ, z = r sin θ.

        Torsion: ``T·r / J``, tangential.  Shear: the Jourawski stress of the
        chord through the point, which on the boundary is tangential too and
        varies as the sine of the angle from the load direction.
        """
        r = np.broadcast_arrays(self.r, self.t)[0]
        tors = 1.0 / self.torsion_modulus  # [1/mm³]
        shear = self.shear_factor / self.area  # [1/mm²]
        rows = []
        for theta in np.linspace(0.0, 2.0 * np.pi, N_ANGLES, endpoint=False):
            c, s = np.cos(theta), np.sin(theta)
            # tangent (−s, c); Vy shear = shear·s·(s, −c), Vz shear = shear·c·(−s, c)
            rows.append(
                (
                    r * c,
                    r * s,
                    -s * tors,
                    c * tors,
                    s * s * shear,
                    -s * c * shear,
                    -c * s * shear,
                    c * c * shear,
                )
            )
        return StressPoints.from_rows(rows, r.shape)

    def define(self, mapdl, secnum: int, name: str) -> None:
        mapdl.sectype(secnum, "BEAM", "CTUBE", name)
        mapdl.secdata(float(self.r_inner), float(self.r))
        mapdl.secoffset("CENT")


class CircSection(TubeSection):
    """
    Solid circle of radius *r*.

    Parameters
    ----------
    r : float or array-like
        Radius [mm].
    """

    param_names = ("r",)

    def __init__(self, r: ArrayLike) -> None:
        super().__init__(r, r)

    @classmethod
    def from_params(cls, params: Mapping[str, ArrayLike]) -> CircSection:
        return cls(params["r"])

    def define(self, mapdl, secnum: int, name: str) -> None:
        mapdl.sectype(secnum, "BEAM", "CSOLID", name)
        mapdl.secdata(float(self.r))
        mapdl.secoffset("CENT")
