"""
beam_opt.sections.base
======================
Abstract cross-section interface.

A ``Section`` may hold scalar dimensions (one section, e.g. to write it to
MAPDL) or arrays (the same shape evaluated at many abscissae, for vectorised
post-processing).  Stress-check points per section come with Phase 3.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike


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

    @abstractmethod
    def define(self, mapdl, secnum: int, name: str) -> None:
        """
        Write the section to MAPDL (``SECTYPE``/``SECDATA``/...) as number *secnum*.

        Only valid for a section holding scalar dimensions.
        """
