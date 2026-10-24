"""
beam_opt.geometry.curve
=======================
3-D beam centre line parameterised by the curvilinear abscissa *s*.

Two shapes through the ordered definition points:

* ``"spline"`` — parametric cubic spline (not-a-knot), reparameterised by
  arc length: *s* is the true length measured along the curve, computed by
  Gauss–Legendre quadrature;
* ``"polyline"`` — straight segments between the points.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import CubicHermiteSpline, CubicSpline

KINDS = ("spline", "polyline")

_GAUSS_X, _GAUSS_W = np.polynomial.legendre.leggauss(5)
_SUBDIV = 32  # arc-length table intervals per spline segment


class BeamPath:
    """
    Beam centre line through ordered 3-D points.

    Parameters
    ----------
    points : array-like, shape (N, 3)
        Ordered definition points [mm], N >= 2, consecutive points distinct.
    kind : str
        ``"spline"`` (default) or ``"polyline"``.  With two points the
        spline is the straight segment.

    Attributes
    ----------
    s_knots : ndarray, shape (N,)
        Arc length at each definition point [mm]; use it to place anchors on
        definition points, e.g. ``{"support_1": path.s_knots[22]}``.
    """

    def __init__(self, points: ArrayLike, kind: str = "spline") -> None:
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2:
            raise ValueError("points must have shape (N, 3) with N >= 2")
        if kind not in KINDS:
            raise ValueError(f"unknown kind {kind!r}: expected one of {KINDS}")
        chords = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        if np.any(chords <= 0.0):
            raise ValueError("consecutive points must be distinct")

        self.points = pts
        self.kind = kind
        self._chords = chords
        u_knots = np.concatenate(([0.0], np.cumsum(chords)))  # chord-length parameter

        if kind == "polyline" or len(pts) == 2:
            self._spline = None
            self.s_knots = u_knots
            return

        self._spline = CubicSpline(u_knots, pts, axis=0)
        self._dspline = self._spline.derivative()

        # Arc-length table s(u) at _SUBDIV points per segment, Gauss quadrature
        # of the speed |C'(u)| on each table interval.
        segments = [np.linspace(u0, u1, _SUBDIV, endpoint=False) for u0, u1 in pairwise(u_knots)]
        u_tab = np.concatenate([*segments, u_knots[-1:]])
        a, b = u_tab[:-1], u_tab[1:]
        half = 0.5 * (b - a)
        uq = 0.5 * (a + b)[:, None] + half[:, None] * _GAUSS_X[None, :]
        speed_q = np.linalg.norm(self._dspline(uq), axis=-1)
        s_tab = np.concatenate(([0.0], np.cumsum(half * (speed_q @ _GAUSS_W))))

        speed_tab = np.linalg.norm(self._dspline(u_tab), axis=-1)
        if np.any(speed_tab <= 0.0):
            raise ValueError("the spline through the points has a cusp (zero speed)")
        # Inverse map u(s), with du/ds = 1/|C'(u)| at the table points
        self._u_of_s = CubicHermiteSpline(s_tab, u_tab, 1.0 / speed_tab)
        self.s_knots = s_tab[::_SUBDIV]

    @property
    def length(self) -> float:
        """Total length of the centre line [mm]."""
        return float(self.s_knots[-1])

    def point_at(self, s: ArrayLike) -> np.ndarray:
        """
        Coordinates at curvilinear abscissa *s*.

        Parameters
        ----------
        s : array-like
            Curvilinear abscissa in ``[0, length]`` [mm].

        Returns
        -------
        ndarray, shape (..., 3)
            Coordinates [mm].
        """
        s = np.asarray(s, dtype=float)
        if self._spline is None:
            return np.stack(
                [np.interp(s, self.s_knots, self.points[:, k]) for k in range(3)], axis=-1
            )
        return self._spline(self._u_of_s(s))

    def tangent_at(self, s: ArrayLike) -> np.ndarray:
        """
        Unit tangent at curvilinear abscissa *s* (direction of increasing *s*).

        On a polyline corner the tangent of the following segment is returned.

        Returns
        -------
        ndarray, shape (..., 3)
            Unit vectors [-].
        """
        s = np.asarray(s, dtype=float)
        if self._spline is None:
            seg = np.clip(
                np.searchsorted(self.s_knots, s, side="right") - 1, 0, len(self._chords) - 1
            )
            return np.diff(self.points, axis=0)[seg] / self._chords[seg, None]
        d = self._dspline(self._u_of_s(s))
        return d / np.linalg.norm(d, axis=-1, keepdims=True)
