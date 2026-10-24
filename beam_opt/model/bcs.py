"""
beam_opt.model.bcs
==================
Boundary conditions and loads, referring to nodes by name.

Each object lists the node names it needs (``node_names``) and writes itself
to MAPDL (``apply``) given the name -> MAPDL number map built by
``ModelBuilder``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping


class BoundaryCondition(ABC):
    """Constraint or load applied to named nodes."""

    @abstractmethod
    def node_names(self) -> list[str]:
        """Names of the nodes this condition refers to."""

    @abstractmethod
    def apply(self, mapdl, node_ids: Mapping[str, int]) -> None:
        """Write the condition in an active ``/PREP7`` session."""


def _as_list(nodes: str | list[str]) -> list[str]:
    return [nodes] if isinstance(nodes, str) else list(nodes)


class Clamp(BoundaryCondition):
    """All degrees of freedom fixed (``D,node,ALL,0``)."""

    def __init__(self, nodes: str | list[str]) -> None:
        self.nodes = _as_list(nodes)

    def node_names(self) -> list[str]:
        return self.nodes

    def apply(self, mapdl, node_ids: Mapping[str, int]) -> None:
        for n in self.nodes:
            mapdl.d(node_ids[n], "ALL", 0)


class Symmetry(BoundaryCondition):
    """
    Symmetry about the plane normal to a global axis (``DSYM,SYMM,<normal>``).

    Parameters
    ----------
    nodes : str | list[str]
        Nodes lying on the symmetry plane.
    normal : str
        Global axis normal to the plane: ``"X"``, ``"Y"`` or ``"Z"``.
    """

    def __init__(self, nodes: str | list[str], normal: str = "Z") -> None:
        self.nodes = _as_list(nodes)
        self.normal = normal.upper()

    def node_names(self) -> list[str]:
        return self.nodes

    def apply(self, mapdl, node_ids: Mapping[str, int]) -> None:
        for n in self.nodes:
            mapdl.nsel("S", "NODE", "", node_ids[n])
            mapdl.dsym("SYMM", self.normal, 0)
        mapdl.allsel("ALL")


class ImposedDisplacement(BoundaryCondition):
    """
    Prescribed value of one degree of freedom (``D,node,<dof>,<value>``).

    Parameters
    ----------
    nodes : str | list[str]
        Constrained nodes.
    dof : str
        MAPDL DOF label (``"UX"``, ``"UY"``, ``"ROTZ"``, ...).
    value : float
        Prescribed value [mm] or [rad].
    """

    def __init__(self, nodes: str | list[str], dof: str, value: float = 0.0) -> None:
        self.nodes = _as_list(nodes)
        self.dof = dof.upper()
        self.value = float(value)

    def node_names(self) -> list[str]:
        return self.nodes

    def apply(self, mapdl, node_ids: Mapping[str, int]) -> None:
        for n in self.nodes:
            mapdl.d(node_ids[n], self.dof, self.value)


class RigidRegion(BoundaryCondition):
    """
    Rigid link from a master node to slave nodes (``CERIG``).

    Parameters
    ----------
    master : str
        Master node.
    slaves : list[str]
        Slave nodes (the master, if listed, is skipped).
    dofs : str
        ``CERIG`` DOF label.
    """

    def __init__(self, master: str, slaves: list[str], dofs: str = "ALL") -> None:
        self.master = master
        self.slaves = list(slaves)
        self.dofs = dofs

    def node_names(self) -> list[str]:
        return [self.master, *self.slaves]

    def apply(self, mapdl, node_ids: Mapping[str, int]) -> None:
        for s in self.slaves:
            if s != self.master:
                mapdl.cerig(node_ids[self.master], node_ids[s], self.dofs)


class NodalForce(BoundaryCondition):
    """
    Concentrated force or moment (``F,node,<direction>,<value>``).

    Parameters
    ----------
    node : str
        Loaded node.
    direction : str
        ``"FX"``, ``"FY"``, ``"FZ"``, ``"MX"``, ``"MY"`` or ``"MZ"``.
    value : float
        Magnitude [N] or [N·mm].
    """

    def __init__(self, node: str, direction: str, value: float) -> None:
        self.node = node
        self.direction = direction.upper()
        self.value = float(value)

    def node_names(self) -> list[str]:
        return [self.node]

    def apply(self, mapdl, node_ids: Mapping[str, int]) -> None:
        mapdl.f(node_ids[self.node], self.direction, self.value)
