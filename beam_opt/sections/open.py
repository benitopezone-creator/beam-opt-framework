"""
beam_opt.sections.open
======================
Doubly symmetric I section (MAPDL ``SECTYPE,,BEAM,I``) and solid ellipse.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike

from .base import Section, StressPoints

#: Check points on the ellipse boundary (see ``beam_opt.sections.circ.N_ANGLES``).
N_ANGLES = 36


class ISection(Section):
    """
    I (double-T) section with equal flanges.

    Flanges parallel to the local y axis, web along the local z axis.
    Torsion and shear stresses follow the thin-walled theory for open
    sections (``J = Σ l t³ / 3``, ``τ = T t / J``, shear flow ``V Q / (I t)``).

    Parameters
    ----------
    b : float or array-like
        Flange width [mm].
    h : float or array-like
        Overall depth [mm].
    tw : float or array-like
        Web thickness [mm].
    tf : float or array-like
        Flange thickness [mm], ``tf < h / 2``.
    """

    param_names = ("b", "h", "tw", "tf")

    def __init__(self, b: ArrayLike, h: ArrayLike, tw: ArrayLike, tf: ArrayLike) -> None:
        self.b = np.asarray(b, dtype=float)
        self.h = np.asarray(h, dtype=float)
        self.tw = np.asarray(tw, dtype=float)
        self.tf = np.asarray(tf, dtype=float)

    @classmethod
    def from_params(cls, params: Mapping[str, ArrayLike]) -> ISection:
        return cls(params["b"], params["h"], params["tw"], params["tf"])

    @classmethod
    def from_secdata(cls, *data: float) -> ISection:
        w1, w2, w3, t1, t2, t3 = data[:6]
        if w1 != w2 or t1 != t2:
            raise NotImplementedError("ISection needs equal flanges")
        return cls(w1, w3, t3, t1)

    @property
    def _web_height(self) -> np.ndarray:
        return self.h - 2.0 * self.tf

    @property
    def area(self) -> np.ndarray:
        return 2.0 * self.b * self.tf + self._web_height * self.tw

    @property
    def iy(self) -> np.ndarray:
        return (self.b * self.h**3 - (self.b - self.tw) * self._web_height**3) / 12.0

    @property
    def iz(self) -> np.ndarray:
        return (2.0 * self.tf * self.b**3 + self._web_height * self.tw**3) / 12.0

    @property
    def torsion_constant(self) -> np.ndarray:
        return (2.0 * self.b * self.tf**3 + (self.h - self.tf) * self.tw**3) / 3.0

    @property
    def torsion_modulus(self) -> np.ndarray:
        return self.torsion_constant / np.maximum(self.tw, self.tf)

    @property
    def shear_correction(self) -> tuple[np.ndarray, np.ndarray]:
        # Vy is carried by the flanges, Vz by the web
        return 5.0 / 6.0 * 2.0 * self.b * self.tf / self.area, self.h * self.tw / self.area

    def stress_points(self) -> StressPoints:
        """
        Flange tips, flange centres, web-flange junctions and web centre.

        * Flange tips: extreme fibres for both bending moments, no shear.
        * Flange centres (outer surface): torsion ``T·tf/J``, the parabolic
          shear of ``Vy`` in the flange and the shear flow that ``Vz`` sends
          from each half-flange to the web (both directions are listed).
        * Junctions and web centre: shear flow of ``Vz`` in the web plus the
          torsion stress ``T·tw/J`` on either web face.
        """
        b, h, tw, tf = np.broadcast_arrays(self.b, self.h, self.tw, self.tf)
        j = self.torsion_constant
        iy = self.iy
        q_flange = b * tf * (h - tf) / 2.0  # first moment of one flange [mm³]
        q_half = q_flange + tw * (h / 2.0 - tf) ** 2 / 2.0
        vy_flange = 1.5 / (2.0 * b * tf)
        vz_half_flange = (b / 2.0) * (h - tf) / 2.0 / iy  # τ_xy at the web side of a half flange

        # (y, z, ty_t, tz_t, ty_vy, tz_vy, ty_vz, tz_vz)
        rows = [(sy * b / 2, sz * h / 2, 0, 0, 0, 0, 0, 0) for sy in (1, -1) for sz in (1, -1)]
        for sz in (1, -1):
            for side in (1, -1):
                rows.append(
                    (0, sz * h / 2, -sz * tf / j, 0, vy_flange, 0, side * vz_half_flange, 0)
                )
                rows.append((0, sz * (h / 2 - tf), 0, side * tw / j, 0, 0, 0, q_flange / (iy * tw)))
        for side in (1, -1):
            rows.append((0, 0, 0, side * tw / j, 0, 0, 0, q_half / (iy * tw)))
        return StressPoints.from_rows(rows, b.shape)

    def mapdl_section(self) -> tuple[str, tuple[float, ...]]:
        b, h, tw, tf = (float(v) for v in (self.b, self.h, self.tw, self.tf))
        return "I", (b, b, h, tf, tf, tw)


class EllipseSection(Section):
    """
    Solid ellipse with semi-axes *a* (along local y) and *b* (along local z).

    Exact Saint-Venant torsion.  MAPDL has no elliptical library section: it
    is written as an arbitrary section (``ASEC``).

    Parameters
    ----------
    a, b : float or array-like
        Semi-axes [mm].
    """

    param_names = ("a", "b")

    def __init__(self, a: ArrayLike, b: ArrayLike) -> None:
        self.a = np.asarray(a, dtype=float)
        self.b = np.asarray(b, dtype=float)

    @classmethod
    def from_params(cls, params: Mapping[str, ArrayLike]) -> EllipseSection:
        return cls(params["a"], params["b"])

    @property
    def area(self) -> np.ndarray:
        return np.pi * self.a * self.b

    @property
    def iy(self) -> np.ndarray:
        return np.pi * self.a * self.b**3 / 4.0

    @property
    def iz(self) -> np.ndarray:
        return np.pi * self.a**3 * self.b / 4.0

    @property
    def torsion_constant(self) -> np.ndarray:
        return np.pi * self.a**3 * self.b**3 / (self.a**2 + self.b**2)

    @property
    def torsion_modulus(self) -> np.ndarray:
        # maximum at the ends of the minor axis: τ = 2 T / (π · major · minor²)
        return np.pi * np.maximum(self.a, self.b) * np.minimum(self.a, self.b) ** 2 / 2.0

    @property
    def shear_correction(self) -> tuple[float, float]:
        return 0.9, 0.9

    def stress_points(self) -> StressPoints:
        """
        ``N_ANGLES`` points on the boundary, at y = a cos θ, z = b sin θ.

        Torsion from the exact stress function; shear as the Jourawski stress
        of the chord through the point, directed along the boundary.
        """
        a, b = np.broadcast_arrays(self.a, self.b)
        k_t = 2.0 / (np.pi * a * b)  # [1/mm²]
        k_v = 4.0 / (3.0 * np.pi * a * b)  # 4/(3A) [1/mm²]
        rows = []
        for theta in np.linspace(0.0, 2.0 * np.pi, N_ANGLES, endpoint=False):
            c, s = np.cos(theta), np.sin(theta)
            rows.append(
                (
                    a * c,
                    b * s,
                    -k_t * s / b,
                    k_t * c / a,
                    k_v * s * s,
                    -k_v * (b / a) * s * c,
                    -k_v * (a / b) * s * c,
                    k_v * c * c,
                )
            )
        return StressPoints.from_rows(rows, a.shape)

    def mapdl_section(self) -> tuple[str, tuple[float, ...]]:
        return self.asec()
