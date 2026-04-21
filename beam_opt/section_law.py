"""
beam_opt.section_law
====================
Generic spline-based section distribution along a 1-D curvilinear abscissa.

Keeping this module separate from the optimizer makes it straightforward to
swap in, e.g., a piecewise-linear law or a Bézier parameterisation without
touching the optimisation or FEM code.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import CubicSpline

from .config import B_MIN, B_MAX, H_MIN, H_MAX


class SectionLaw:
    """
    Cubic-spline interpolation of rectangular cross-section dimensions
    ``b(s)`` and ``h(s)`` along a curvilinear abscissa *s*.

    Parameters
    ----------
    s_ctrl : array-like, shape (n_ctrl,)
        Curvilinear abscissa of the spline control points [mm].
    b_ctrl : array-like, shape (n_ctrl,)
        Width values at the control points [mm].
    h_ctrl : array-like, shape (n_ctrl,)
        Height values at the control points [mm].
    b_bounds : tuple[float, float]
        (b_min, b_max) used for clipping.  Defaults to the global config values.
    h_bounds : tuple[float, float]
        (h_min, h_max) used for clipping.  Defaults to the global config values.
    """

    def __init__(
        self,
        s_ctrl: ArrayLike,
        b_ctrl: ArrayLike,
        h_ctrl: ArrayLike,
        b_bounds: tuple[float, float] = (B_MIN, B_MAX),
        h_bounds: tuple[float, float] = (H_MIN, H_MAX),
    ) -> None:
        self.s_ctrl = np.asarray(s_ctrl, dtype=float)
        self.b_ctrl = np.asarray(b_ctrl, dtype=float)
        self.h_ctrl = np.asarray(h_ctrl, dtype=float)
        self.b_bounds = b_bounds
        self.h_bounds = h_bounds

        self._spline_b = CubicSpline(self.s_ctrl, self.b_ctrl, bc_type="natural")
        self._spline_h = CubicSpline(self.s_ctrl, self.h_ctrl, bc_type="natural")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def eval(self, s: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """
        Return ``(b, h)`` arrays evaluated at curvilinear positions *s*,
        clipped to the admissible bounds.

        Parameters
        ----------
        s : array-like
            Curvilinear abscissa values [mm].

        Returns
        -------
        b, h : ndarray
            Cross-section dimensions [mm], guaranteed within ``b_bounds``
            and ``h_bounds``.
        """
        s = np.asarray(s, dtype=float)
        b = np.clip(self._spline_b(s), *self.b_bounds)
        h = np.clip(self._spline_h(s), *self.h_bounds)
        return b, h
