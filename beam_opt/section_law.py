"""
beam_opt.section_law
====================
Distribution of named section parameters p(s) along the curvilinear abscissa.

Keeping this module separate from the optimizer makes it straightforward to
swap the interpolation scheme without touching the optimisation or FEM code.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import CubicSpline, PchipInterpolator

#: Available interpolation schemes.  ``"pchip"`` and ``"linear"`` never
#: overshoot the control values; ``"natural_cubic"`` is C2 but can.
INTERPOLATORS: dict[str, Callable[[np.ndarray, np.ndarray], Callable]] = {
    "natural_cubic": lambda s, v: CubicSpline(s, v, bc_type="natural"),
    "pchip": PchipInterpolator,
    "linear": lambda s, v: lambda x: np.interp(x, s, v),
}

#: Interpolators whose values always stay between the control values.
OVERSHOOT_FREE = frozenset({"pchip", "linear"})


class SectionLaw:
    """
    Interpolated section parameters along *s*.

    Parameters
    ----------
    s_ctrl : array-like, shape (n_ctrl,)
        Curvilinear abscissa of the control points [mm], increasing.
    ctrl : mapping[str, array-like]
        ``{parameter name: values at the control points}``, e.g.
        ``{"b": [...], "h": [...]}`` [mm].  The order of the mapping is the
        order of the parameters in the design vector.
    bounds : mapping[str, tuple[float, float]] | None
        Admissible ``(min, max)`` per parameter, used for clipping.  Missing
        parameters are not clipped.
    interpolator : str
        Key of ``INTERPOLATORS``.
    """

    def __init__(
        self,
        s_ctrl: ArrayLike,
        ctrl: Mapping[str, ArrayLike],
        bounds: Mapping[str, tuple[float, float]] | None = None,
        interpolator: str = "natural_cubic",
    ) -> None:
        if interpolator not in INTERPOLATORS:
            raise ValueError(f"unknown interpolator {interpolator!r}: {sorted(INTERPOLATORS)}")
        self.s_ctrl = np.asarray(s_ctrl, dtype=float)
        self.ctrl = {name: np.asarray(v, dtype=float) for name, v in ctrl.items()}
        self.bounds = dict(bounds or {})
        self.interpolator = interpolator
        make = INTERPOLATORS[interpolator]
        self._interp = {name: make(self.s_ctrl, v) for name, v in self.ctrl.items()}

    @property
    def names(self) -> tuple[str, ...]:
        """Parameter names, in design-vector order."""
        return tuple(self.ctrl)

    def eval(self, s: ArrayLike, clip: bool = True) -> dict[str, np.ndarray]:
        """
        Evaluate every parameter at curvilinear positions *s*.

        Parameters
        ----------
        s : array-like
            Curvilinear abscissa values [mm].
        clip : bool
            If ``False``, return the raw interpolated values.  A cubic spline
            can overshoot the bounds between control points even when all
            control values are admissible; constraints must see the raw
            values, otherwise the overshoot is hidden by the clipping.

        Returns
        -------
        dict[str, ndarray]
            Parameter values (same units as ``ctrl``); within ``bounds`` when
            ``clip=True``.
        """
        s = np.asarray(s, dtype=float)
        out = {}
        for name, f in self._interp.items():
            v = np.asarray(f(s), dtype=float)
            if clip and name in self.bounds:
                v = np.clip(v, *self.bounds[name])
            out[name] = v
        return out
