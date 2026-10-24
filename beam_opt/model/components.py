"""
beam_opt.model.components
=========================
Structural components and the model that groups them.

All nodes are referred to by **name**: an anchor of the optimisable beam
mesh (``"start"``, ``"end"``, ``"tip"``, ...) or a named extra point of the
model.  MAPDL numbers are assigned by ``ModelBuilder``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike

from ..geometry import BeamMesh
from ..materials import Material
from ..sections import RectSection, Section


@dataclass
class OptimizableBeam:
    """
    Chain of BEAM188 elements whose section follows the section law.

    Parameters
    ----------
    mesh : BeamMesh
        Discretised centre line.
    section_factory : callable
        ``params -> Section``, where ``params`` is the dict returned by
        ``SectionLaw.eval``.  Defaults to ``RectSection.from_params``.
    """

    mesh: BeamMesh
    section_factory: Callable[[Mapping[str, ArrayLike]], Section] = RectSection.from_params

    def sections_at(self, law, s: ArrayLike) -> Section:
        """Section (vectorised) at abscissae *s* [mm] for section law *law*."""
        return self.section_factory(law.eval(s))

    def mass(self, law, rho: float) -> float:
        """Mass with mid-element sections [ton]; *rho* in [ton/mm³]."""
        area = self.sections_at(law, self.mesh.s_elem).area
        return float(rho * np.sum(area * self.mesh.lengths))


class FixedElement(ABC):
    """A non-optimisable element connecting named nodes."""

    #: MAPDL element type name (``ET``) used by this component.
    element_type: str = ""

    def __init__(self, nodes: list[str]) -> None:
        self.nodes = list(nodes)

    @abstractmethod
    def create(self, mapdl, node_ids: list[int], secnum: int, etype: int, mat: int) -> None:
        """Write the element and its section in an active ``/PREP7`` session."""

    @abstractmethod
    def estimate_mass(self, coords: Mapping[str, np.ndarray], rho: float) -> float:
        """
        Geometry-based mass [ton].

        Parameters
        ----------
        coords : mapping[str, ndarray]
            Coordinates [mm] of every named node of the model.
        rho : float
            Density [ton/mm³].
        """


class FixedBeam(FixedElement):
    """
    BEAM188 element with a constant section.

    Parameters
    ----------
    node_a, node_b : str
        End-node names.
    section : Section
        Scalar cross-section.
    """

    element_type = "BEAM188"

    def __init__(self, node_a: str, node_b: str, section: Section) -> None:
        super().__init__([node_a, node_b])
        self.section = section

    def create(self, mapdl, node_ids: list[int], secnum: int, etype: int, mat: int) -> None:
        self.section.define(mapdl, secnum, f"FIXED_SEC{secnum}")
        mapdl.type(etype)
        mapdl.mat(mat)
        mapdl.secnum(secnum)
        mapdl.e(*node_ids)

    def estimate_mass(self, coords: Mapping[str, np.ndarray], rho: float) -> float:
        p1, p2 = (np.asarray(coords[n], dtype=float) for n in self.nodes)
        return float(rho * np.linalg.norm(p2 - p1) * self.section.area)


class FixedShell(FixedElement):
    """
    SHELL181 quadrilateral with constant thickness.

    Parameters
    ----------
    nodes : list[str]
        Four corner-node names in connectivity order (repeat the third for a
        triangle).
    thickness : float
        Shell thickness [mm].
    """

    element_type = "SHELL181"

    def __init__(self, nodes: list[str], thickness: float) -> None:
        if len(nodes) != 4:
            raise ValueError("FixedShell needs 4 node names")
        super().__init__(nodes)
        self.thickness = float(thickness)

    def create(self, mapdl, node_ids: list[int], secnum: int, etype: int, mat: int) -> None:
        mapdl.sectype(secnum, "SHELL", "", f"FIXED_SHELL{secnum}")
        mapdl.secdata(self.thickness)
        mapdl.type(etype)
        mapdl.mat(mat)
        mapdl.secnum(secnum)
        mapdl.e(*node_ids)

    def estimate_mass(self, coords: Mapping[str, np.ndarray], rho: float) -> float:
        quad = np.array([coords[n] for n in self.nodes], dtype=float)
        return rho * quad_area_3d(quad) * self.thickness


def quad_area_3d(coords: ArrayLike) -> float:
    """
    Area of a quadrilateral in 3-D space, split into triangles (1-2-3) and (1-3-4).

    Exact for planar quadrilaterals in any orientation; for slightly warped
    ones it is the area of the two-triangle surface.  A degenerate quad with
    node 4 = node 3 (triangle) is handled naturally.

    Parameters
    ----------
    coords : array-like, shape (4, 3)
        Corner coordinates in connectivity order [mm].

    Returns
    -------
    float
        Area [mm²].
    """
    p1, p2, p3, p4 = np.asarray(coords, dtype=float)
    a1 = 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))
    a2 = 0.5 * np.linalg.norm(np.cross(p3 - p1, p4 - p1))
    return float(a1 + a2)


@dataclass
class StructuralModel:
    """
    Complete structure: optimisable beam, fixed parts, named points, BCs.

    Parameters
    ----------
    beam : OptimizableBeam
        The beam whose section is optimised.
    material : Material
        Material of every element.
    points : dict[str, array-like]
        Named extra nodes (supports, shell corners, ...) [mm].  Names must
        not clash with the beam anchors.
    fixed : list[FixedElement]
        Non-optimisable elements.
    bcs : list
        Boundary conditions (``beam_opt.model.bcs``), applied in order.
    """

    beam: OptimizableBeam
    material: Material
    points: dict[str, ArrayLike] = field(default_factory=dict)
    fixed: list[FixedElement] = field(default_factory=list)
    bcs: list = field(default_factory=list)

    def __post_init__(self) -> None:
        self.points = {k: np.asarray(v, dtype=float) for k, v in self.points.items()}
        clash = set(self.points) & set(self.beam.mesh.anchors)
        if clash:
            raise ValueError(f"point names clash with beam anchors: {sorted(clash)}")
        known = self.node_names()
        refs = [n for fe in self.fixed for n in fe.nodes]
        refs += [n for bc in self.bcs for n in bc.node_names()]
        unknown = sorted(set(refs) - known)
        if unknown:
            raise ValueError(f"unknown node names: {unknown}")

    def node_names(self) -> set[str]:
        """All names a component or BC may refer to."""
        return set(self.beam.mesh.anchors) | set(self.points)

    def named_coords(self) -> dict[str, np.ndarray]:
        """Coordinates [mm] of every named node."""
        mesh = self.beam.mesh
        out = {name: mesh.coords[i] for name, i in mesh.anchors.items()}
        out.update(self.points)
        return out

    def fixed_mass(self) -> float:
        """Mass of the fixed elements [ton]."""
        coords = self.named_coords()
        return float(sum(fe.estimate_mass(coords, self.material.rho) for fe in self.fixed))
