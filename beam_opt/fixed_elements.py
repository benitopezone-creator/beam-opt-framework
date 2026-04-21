"""
beam_opt.fixed_elements
=======================
Fixed structural elements that are included in the FEM model but whose
cross-sections are NOT optimised.

Design notes
------------
* Sections are **always** redefined inside ``create()`` so that the objects
  remain valid after ``mapdl.clear()`` – which resets all section definitions.
* ``estimate_mass()`` is intentionally geometry-only (no MAPDL call) so that
  the fixed-element mass can be computed once before the optimisation loop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike


# -----------------------------------------------------------------------
# Default section IDs for fixed elements
# (must not collide with the optimisable-section numbering 1…n_elem)
# -----------------------------------------------------------------------
FIXED_BEAM_SECNUM: int = 301
FIXED_SHELL_SECNUM: int = 1000


# -----------------------------------------------------------------------
# Abstract base
# -----------------------------------------------------------------------
class FixedElement(ABC):
    """Abstract base class for a non-optimisable structural element."""

    def __init__(self, elem_id: int) -> None:
        self.elem_id = int(elem_id)

    @abstractmethod
    def create(self, mapdl) -> None:
        """Create the element (and its section) inside an active MAPDL /PREP7 session."""

    @abstractmethod
    def estimate_mass(self, nodes_dict: dict, rho: float) -> float:
        """
        Return a geometry-based mass estimate [model mass unit].

        Parameters
        ----------
        nodes_dict : dict[int, array-like]
            ``{node_id: [x, y, z]}`` mapping for the full model.
        rho : float
            Material density [ton/mm³].
        """


# -----------------------------------------------------------------------
# BEAM188 (rectangular section, fixed geometry)
# -----------------------------------------------------------------------
class FixedBeam(FixedElement):
    """
    BEAM188 element with a fixed rectangular cross-section.

    Parameters
    ----------
    elem_id : int
        Logical element ID.
    node_i, node_j : int
        End-node IDs.
    b, h : float
        Section width and height [mm].
    secnum : int
        MAPDL section number (must not overlap optimisable sections).
    """

    def __init__(
        self,
        elem_id: int,
        node_i: int,
        node_j: int,
        b: float,
        h: float,
        secnum: int = FIXED_BEAM_SECNUM,
    ) -> None:
        super().__init__(elem_id)
        self.node_i = int(node_i)
        self.node_j = int(node_j)
        self.b = float(b)
        self.h = float(h)
        self.secnum = int(secnum)

    def create(self, mapdl) -> None:
        # Section is redefined unconditionally to survive mapdl.clear()
        mapdl.sectype(self.secnum, "BEAM", "RECT", f"FIXED_SEC{self.secnum}")
        mapdl.secdata(self.b, self.h)
        mapdl.secoffset("CENT")

        mapdl.type(1)   # ET 1 = BEAM188
        mapdl.mat(1)
        mapdl.secnum(self.secnum)
        mapdl.e(self.node_i, self.node_j)

    def estimate_mass(self, nodes_dict: dict, rho: float) -> float:
        p1 = np.asarray(nodes_dict[self.node_i], dtype=float)
        p2 = np.asarray(nodes_dict[self.node_j], dtype=float)
        length = float(np.linalg.norm(p2 - p1))
        return rho * length * self.b * self.h


# -----------------------------------------------------------------------
# SHELL181 (constant thickness, fixed geometry)
# -----------------------------------------------------------------------
class FixedShell(FixedElement):
    """
    SHELL181 element with a fixed shell thickness.

    Parameters
    ----------
    elem_id : int
        Logical element ID.
    nodes : list[int]
        Four corner node IDs (quadrilateral connectivity).
    thickness : float
        Shell thickness [mm].
    secnum : int
        MAPDL section number.
    """

    def __init__(
        self,
        elem_id: int,
        nodes: list[int],
        thickness: float,
        secnum: int = FIXED_SHELL_SECNUM,
    ) -> None:
        super().__init__(elem_id)
        self.nodes = list(nodes)
        self.thickness = float(thickness)
        self.secnum = int(secnum)

    def create(self, mapdl) -> None:
        mapdl.sectype(self.secnum, "SHELL", "", f"FIXED_SHELL{self.secnum}")
        mapdl.secdata(self.thickness)

        mapdl.type(2)   # ET 2 = SHELL181
        mapdl.mat(1)
        mapdl.secnum(self.secnum)
        mapdl.e(*self.nodes)

    def estimate_mass(self, nodes_dict: dict, rho: float) -> float:
        coords = np.array([nodes_dict[n] for n in self.nodes], dtype=float)
        x, y = coords[:, 0], coords[:, 1]
        # Shoelace formula for planar quadrilateral area
        area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        return rho * area * self.thickness
