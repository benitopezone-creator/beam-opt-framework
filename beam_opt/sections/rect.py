"""
beam_opt.sections.rect
======================
Solid rectangular cross-section (MAPDL ``SECTYPE,,BEAM,RECT``).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike

from .base import Section


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

    def define(self, mapdl, secnum: int, name: str) -> None:
        mapdl.sectype(secnum, "BEAM", "RECT", name)
        mapdl.secdata(float(self.b), float(self.h))
        mapdl.secoffset("CENT")
