"""
beam_opt.sections.rect
======================
Solid rectangular cross-section (MAPDL ``SECTYPE,,BEAM,RECT``).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike

from .base import Section, StressPoints


class RectSection(Section):
    """
    Solid rectangle *b × h*.

    MAPDL convention (``SECDATA,B,H``): *b* along the local y axis, *h* along
    the local z axis.

    Parameters
    ----------
    b : float or array-like
        Width [mm].
    h : float or array-like
        Height [mm].
    """

    param_names = ("b", "h")

    def __init__(self, b: ArrayLike, h: ArrayLike) -> None:
        self.b = np.asarray(b, dtype=float)
        self.h = np.asarray(h, dtype=float)

    @classmethod
    def from_params(cls, params: Mapping[str, ArrayLike]) -> RectSection:
        return cls(params["b"], params["h"])

    @property
    def area(self) -> np.ndarray:
        return self.b * self.h

    @property
    def iy(self) -> np.ndarray:
        return self.b * self.h**3 / 12.0

    @property
    def iz(self) -> np.ndarray:
        return self.h * self.b**3 / 12.0

    def _sides(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a = np.maximum(self.b, self.h)  # long side
        t = np.minimum(self.b, self.h)  # short side
        return a, t, t / a

    @property
    def torsion_constant(self) -> np.ndarray:
        # Roark's formulas for stress and strain, Table 10.7 case 4
        a, t, beta = self._sides()
        return a * t**3 * (1.0 / 3.0 - 0.21 * beta * (1.0 - beta**4 / 12.0))

    @property
    def torsion_modulus(self) -> np.ndarray:
        # τ_max at the mid-side of the long sides (Roark, Table 10.7 case 4)
        a, t, beta = self._sides()
        k = 1.0 + 0.6095 * beta + 0.8865 * beta**2 - 1.8023 * beta**3 + 0.9100 * beta**4
        return a * t**2 / (3.0 * k)

    def stress_points(self) -> StressPoints:
        """
        Four corners, four mid-sides and the centroid.

        * Corners: extreme bending fibres; shear and torsion stresses vanish.
        * Mid-sides: Jourawski maximum ``1.5 V / A`` of the shear force
          parallel to the side, plus the torsion stress, which has the same
          direction.  ``|T| / Wt`` is exact on the long sides and
          conservative on the short ones.
        * Centroid: both Jourawski maxima, no bending and no torsion.
        """
        b, h = np.broadcast_arrays(self.b, self.h)
        jour = 1.5 / (b * h)  # [1/mm²]
        tors = 1.0 / self.torsion_modulus  # [1/mm³]
        rows = [(sy * b / 2, sz * h / 2, 0, 0, 0, 0, 0, 0) for sy in (1, -1) for sz in (1, -1)]
        # (y, z, ty_t, tz_t, ty_vy, tz_vy, ty_vz, tz_vz)
        rows += [(s * b / 2, 0, 0, s * tors, 0, 0, 0, jour) for s in (1, -1)]
        rows += [(0, s * h / 2, -s * tors, 0, jour, 0, 0, 0) for s in (1, -1)]
        rows.append((0, 0, 0, 0, jour, 0, 0, jour))
        return StressPoints.from_rows(rows, b.shape)

    def define(self, mapdl, secnum: int, name: str) -> None:
        mapdl.sectype(secnum, "BEAM", "RECT", name)
        mapdl.secdata(float(self.b), float(self.h))
        mapdl.secoffset("CENT")
