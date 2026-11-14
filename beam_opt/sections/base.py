"""
beam_opt.sections.base
======================
Abstract cross-section interface.

A ``Section`` may hold scalar dimensions (one section, e.g. to write it to
MAPDL) or arrays (the same shape evaluated at many abscissae, for vectorised
post-processing).

Stress recovery is described by the section itself through its *check
points* (``Section.stress_points``): for each point, the position (y, z) and
the shear stress per unit torque and per unit shear force.  The generic
evaluator (``beam_opt.post.SectionStressEvaluator``) needs nothing else, so
adding a shape means adding a ``Section`` subclass.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass
class StressPoints:
    """
    Check points of a cross-section.

    Every array has shape ``(n_points, *section_shape)``.  With the internal
    forces N, My, Mz, T, Vy, Vz (BEAM188 local axes) the stresses at a point
    are

        σ_x  = N/A + My·z/Iy − Mz·y/Iz
        τ_xy = ty_t·T + ty_vy·Vy + ty_vz·Vz
        τ_xz = tz_t·T + tz_vy·Vy + tz_vz·Vz

    The shear stress is a vector in the section plane: the signs of the
    coefficients must be mutually consistent, so that torsion and shear add
    on one side of the section and subtract on the other.  Shapes are
    expected to list the points of both sides.

    Attributes
    ----------
    y, z : ndarray
        Position in the section local axes [mm].
    ty_t, tz_t : ndarray
        Shear stress per unit torque [1/mm³].
    ty_vy, tz_vy : ndarray
        Shear stress per unit shear force Vy [1/mm²].
    ty_vz, tz_vz : ndarray
        Shear stress per unit shear force Vz [1/mm²].
    """

    y: np.ndarray
    z: np.ndarray
    ty_t: np.ndarray
    tz_t: np.ndarray
    ty_vy: np.ndarray
    tz_vy: np.ndarray
    ty_vz: np.ndarray
    tz_vz: np.ndarray

    @classmethod
    def from_rows(cls, rows: list[tuple], shape: tuple[int, ...]) -> StressPoints:
        """
        Build from one tuple per point, in the order of the attributes.

        Each entry may be a scalar or an array broadcastable to *shape*.
        """
        cols = [
            np.stack([np.broadcast_to(np.asarray(v, float), shape) for v in c]) for c in zip(*rows)
        ]
        return cls(*cols)


class Section(ABC):
    """Beam cross-section with the properties needed by the framework."""

    #: Names of the section-law parameters this shape is built from.
    param_names: tuple[str, ...] = ()

    @classmethod
    @abstractmethod
    def from_params(cls, params: Mapping[str, ArrayLike]) -> Section:
        """Build the section from named parameters (e.g. ``{"b": ..., "h": ...}``)."""

    @property
    @abstractmethod
    def area(self) -> np.ndarray:
        """Cross-sectional area A [mm²]."""

    @property
    @abstractmethod
    def iy(self) -> np.ndarray:
        """Second moment of area about the local y axis [mm⁴]."""

    @property
    @abstractmethod
    def iz(self) -> np.ndarray:
        """Second moment of area about the local z axis [mm⁴]."""

    @property
    @abstractmethod
    def torsion_constant(self) -> np.ndarray:
        """Saint-Venant torsion constant J [mm⁴]."""

    @property
    @abstractmethod
    def torsion_modulus(self) -> np.ndarray:
        """Torsional resistance modulus Wt, such that τ_T,max = |T| / Wt [mm³]."""

    @property
    def shear_correction(self) -> tuple[np.ndarray | float, np.ndarray | float]:
        """
        Shear correction factors (κ_y, κ_z) for shear along local y and z [-].

        Used by the built-in solver for the shear deformation only (the
        stresses do not depend on it).  Default: 5/6, the solid rectangle.
        """
        return 5.0 / 6.0, 5.0 / 6.0

    @abstractmethod
    def stress_points(self) -> StressPoints:
        """Check points where the equivalent stress is evaluated."""

    @abstractmethod
    def define(self, mapdl, secnum: int, name: str) -> None:
        """
        Write the section to MAPDL (``SECTYPE``/``SECDATA``/...) as number *secnum*.

        Only valid for a section holding scalar dimensions.
        """
