"""
beam_opt.sections.box
=====================
Rectangular hollow section with uniform wall (MAPDL ``SECTYPE,,BEAM,HREC``).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike

from .base import Section, StressPoints


class BoxSection(Section):
    """
    Rectangular tube *b × h* (outer dimensions) with wall thickness *t*.

    *b* along the local y axis, *h* along the local z axis.  Torsion and
    shear stresses follow the thin-walled theory (Bredt, shear flow), which
    is accurate for ``t`` small with respect to *b* and *h*.

    Parameters
    ----------
    b, h : float or array-like
        Outer width and height [mm].
    t : float or array-like
        Wall thickness [mm], ``t < min(b, h) / 2``.
    """

    param_names = ("b", "h", "t")

    def __init__(self, b: ArrayLike, h: ArrayLike, t: ArrayLike) -> None:
        self.b = np.asarray(b, dtype=float)
        self.h = np.asarray(h, dtype=float)
        self.t = np.asarray(t, dtype=float)

    @classmethod
    def from_params(cls, params: Mapping[str, ArrayLike]) -> BoxSection:
        return cls(params["b"], params["h"], params["t"])

    @property
    def _inner(self) -> tuple[np.ndarray, np.ndarray]:
        return self.b - 2.0 * self.t, self.h - 2.0 * self.t

    @property
    def area(self) -> np.ndarray:
        bi, hi = self._inner
        return self.b * self.h - bi * hi

    @property
    def iy(self) -> np.ndarray:
        bi, hi = self._inner
        return (self.b * self.h**3 - bi * hi**3) / 12.0

    @property
    def iz(self) -> np.ndarray:
        bi, hi = self._inner
        return (self.h * self.b**3 - hi * bi**3) / 12.0

    @property
    def _enclosed_area(self) -> np.ndarray:
        return (self.b - self.t) * (self.h - self.t)  # by the wall mid-line [mm²]

    @property
    def torsion_constant(self) -> np.ndarray:
        # Bredt: J = 4 Am² t / (mid-line perimeter)
        perimeter = 2.0 * (self.b + self.h - 2.0 * self.t)
        return 4.0 * self._enclosed_area**2 * self.t / perimeter

    @property
    def torsion_modulus(self) -> np.ndarray:
        return 2.0 * self._enclosed_area * self.t  # Bredt: τ = T / (2 Am t)

    @property
    def shear_correction(self) -> tuple[np.ndarray, np.ndarray]:
        # Thin-walled: the shear force is carried by the two walls parallel to it
        return 2.0 * self.b * self.t / self.area, 2.0 * self.h * self.t / self.area

    def stress_points(self) -> StressPoints:
        """
        Mid-sides and corners, each corner seen from both walls that meet there.

        * Mid-sides of the walls parallel to z (webs for Vz): shear-flow
          maximum ``Vz·Q / (Iy·2t)`` plus the Bredt torsion stress.  The
          same, with y and z swapped, for the walls parallel to y.
        * Corners: extreme bending fibres, with the torsion stress and the
          shear flow that the adjacent wall carries there.
        """
        b, h, t = np.broadcast_arrays(self.b, self.h, self.t)
        bi, hi = b - 2.0 * t, h - 2.0 * t
        tors = 1.0 / self.torsion_modulus  # [1/mm³]
        # Shear flow per unit force, divided by the two walls that carry it [1/mm²]
        web_mid = (b * h**2 - bi * hi**2) / 8.0 / (self.iy * 2.0 * t)
        web_end = (b * t * (h - t) / 2.0) / (self.iy * 2.0 * t)
        flange_mid = (h * b**2 - hi * bi**2) / 8.0 / (self.iz * 2.0 * t)
        flange_end = (h * t * (b - t) / 2.0) / (self.iz * 2.0 * t)

        # (y, z, ty_t, tz_t, ty_vy, tz_vy, ty_vz, tz_vz)
        rows = []
        for s in (1, -1):
            rows.append((s * b / 2, 0, 0, s * tors, 0, 0, 0, web_mid))
            rows.append((0, s * h / 2, -s * tors, 0, flange_mid, 0, 0, 0))
        for sy in (1, -1):
            for sz in (1, -1):
                y, z = sy * b / 2, sz * h / 2
                rows.append((y, z, 0, sy * tors, 0, 0, 0, web_end))  # wall parallel to z
                rows.append((y, z, -sz * tors, 0, flange_end, 0, 0, 0))  # wall parallel to y
        return StressPoints.from_rows(rows, b.shape)

    @classmethod
    def from_secdata(cls, *data: float) -> BoxSection:
        if len(set(data[2:6])) != 1:
            raise NotImplementedError("BoxSection needs four equal wall thicknesses")
        return cls(data[0], data[1], data[2])

    def mapdl_section(self) -> tuple[str, tuple[float, ...]]:
        t = float(self.t)
        return "HREC", (float(self.b), float(self.h), t, t, t, t)
