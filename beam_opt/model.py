"""
beam_opt.model
==============
Geometry-agnostic FEM model builder.

``BeamModel`` owns:
* the spline-node coordinates (main beam path);
* the dictionary of additional nodes (supports, shell patches, …);
* the list of fixed elements;
* the master node on which displacements are imposed;
* the boundary conditions (symmetry, CERIG, imposed displacements).

All boundary-condition node IDs are **derived from the data** passed in at
construction time – there are no hard-coded node numbers anywhere in this
module.
"""

from __future__ import annotations

import os
import logging
from typing import Optional

import numpy as np

from .config import E, NU, RHO, SolverSettings, DEFAULT_SOLVER
from .mapdl_manager import MapdlManager
from .section_law import SectionLaw
from .fixed_elements import FixedElement

logger = logging.getLogger(__name__)


class BeamModel:
    """
    Parameterised FEM model for a spline beam with optional fixed sub-structures.

    Parameters
    ----------
    nodes_coords : ndarray, shape (N, 3)
        3-D coordinates of the N spline nodes.  Nodes are numbered 1 … N in
        MAPDL.
    fixed_elements : list[FixedElement]
        Fixed BEAM/SHELL elements (non-optimisable).
    additional_nodes : dict[int, array-like]
        ``{mapdl_node_id: [x, y, z]}`` for nodes outside the spline (supports,
        shell patches, …).  Node IDs must not collide with 1 … N.
    master_node : int
        MAPDL node ID on which ``UX_IMP`` and ``UZ_IMP`` are imposed.
    symmetry_nodes : list[int] | None
        MAPDL node IDs on the symmetry plane.  A ``DSYM,SYMM,Z`` constraint is
        applied to each.  If *None*, they are inferred as ``additional_nodes``
        keys with IDs in the range [51, 999].
    uy_fixed_nodes : list[int] | None
        MAPDL node IDs on which UY = 0 is enforced additionally.  If *None*,
        inferred as the first and last element of ``symmetry_nodes``.
    shell_patch_min_id : int
        Minimum node ID considered part of the shell patch (default 1001).
        Used to identify which additional nodes belong to CERIG constraints.
    ux_imp : float
        Imposed displacement in X [mm].
    uz_imp : float
        Imposed displacement in Z [mm].
    settings : SolverSettings
        MAPDL launch / solver settings.
    """

    def __init__(
        self,
        nodes_coords: np.ndarray,
        fixed_elements: list[FixedElement],
        additional_nodes: dict[int, list],
        master_node: int,
        *,
        symmetry_nodes: Optional[list[int]] = None,
        uy_fixed_nodes: Optional[list[int]] = None,
        shell_patch_min_id: int = 1001,
        ux_imp: float = 30.0,
        uz_imp: float = 90.0,
        settings: SolverSettings = DEFAULT_SOLVER,
    ) -> None:
        self.nodes_coords = np.asarray(nodes_coords, dtype=float)
        self.fixed_elements = fixed_elements
        self.additional_nodes = additional_nodes
        self.master_node = int(master_node)
        self.shell_patch_min_id = shell_patch_min_id
        self.ux_imp = float(ux_imp)
        self.uz_imp = float(uz_imp)
        self.settings = settings

        self.n_nodes = len(self.nodes_coords)
        self.n_elem = self.n_nodes - 1   # optimisable beam elements

        # Curvilinear abscissa along the spline
        self.s, self.s_elem, self.L = self._compute_curvilinear()

        # Full node dictionary: spline nodes (1-based) + additional
        self.nodes_dict: dict[int, np.ndarray] = {}
        for i, coords in enumerate(self.nodes_coords, 1):
            self.nodes_dict[i] = np.asarray(coords, dtype=float)
        for nid, coords in additional_nodes.items():
            self.nodes_dict[int(nid)] = np.asarray(coords, dtype=float)

        # Derive BC node sets from data if not provided explicitly
        self._symmetry_nodes = self._resolve_symmetry_nodes(symmetry_nodes)
        self._uy_fixed_nodes = self._resolve_uy_fixed_nodes(uy_fixed_nodes)

        # CERIG slave set: all shell-patch nodes + first/last spline nodes
        self._cerig_slave_nodes = self._resolve_cerig_slaves()

        # Current section law (set by build())
        self.law: Optional[SectionLaw] = None

        self.mapdl = MapdlManager.get(settings=settings)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, law: SectionLaw) -> None:
        """
        (Re)build the full MAPDL model for a given section law.

        This method is designed to be called at the start of every FEM
        evaluation: it calls ``mapdl.clear()`` and rebuilds everything from
        scratch to avoid state contamination between iterations.
        """
        self.law = law
        m = self.mapdl
        m.clear()
        m.filname(self.settings.jobname)
        m.prep7()

        # Element types
        m.et(1, "BEAM188")
        m.et(2, "SHELL181")

        # Material (isotropic elastic)
        m.mp("EX",   1, E)
        m.mp("PRXY", 1, NU)
        m.mp("DENS", 1, RHO)

        # ---- Nodes ----
        for nid, coords in self.nodes_dict.items():
            m.n(nid, *coords)

        # ---- Optimisable sections (1 … n_elem) ----
        b_vals, h_vals = law.eval(self.s_elem)
        for i, (b, h) in enumerate(zip(b_vals, h_vals), 1):
            m.sectype(i, "BEAM", "RECT", f"OPT{i}")
            m.secdata(float(b), float(h))
            m.secoffset("CENT")

        # ---- Optimisable beam elements ----
        m.type(1)
        m.mat(1)
        for i in range(self.n_elem):
            m.secnum(i + 1)
            m.e(i + 1, i + 2)

        # ---- Fixed elements ----
        for fe in self.fixed_elements:
            fe.create(m)

        # ---- Boundary conditions ----
        self._apply_bcs(m)

        m.allsel("ALL")
        m.finish()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_curvilinear(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        L = np.linalg.norm(np.diff(self.nodes_coords, axis=0), axis=1)
        s = np.concatenate(([0.0], np.cumsum(L)))
        s_elem = 0.5 * (s[:-1] + s[1:])
        return s, s_elem, L

    def _resolve_symmetry_nodes(self, explicit: Optional[list[int]]) -> list[int]:
        if explicit is not None:
            return list(explicit)
        # Infer: additional nodes with IDs in [51, shell_patch_min_id)
        return sorted(
            nid for nid in self.additional_nodes
            if 51 <= nid < self.shell_patch_min_id
        )

    def _resolve_uy_fixed_nodes(self, explicit: Optional[list[int]]) -> list[int]:
        if explicit is not None:
            return list(explicit)
        sym = self._symmetry_nodes
        if len(sym) >= 2:
            return [sym[0], sym[-1]]
        return sym

    def _resolve_cerig_slaves(self) -> list[int]:
        # Shell-patch nodes
        patch_nodes = [nid for nid in self.additional_nodes if nid >= self.shell_patch_min_id]
        # First and last spline nodes (also shared with the shell)
        edge_spline = [1, self.n_nodes]
        return patch_nodes + edge_spline

    def _apply_bcs(self, m) -> None:
        # Symmetry on Z = 0 plane for support nodes
        for nid in self._symmetry_nodes:
            m.nsel("S", "NODE", "", nid)
            m.dsym("SYMM", "Z", 0)
        m.allsel("ALL")

        # Additional UY = 0 constraints
        for nid in self._uy_fixed_nodes:
            m.d(nid, "UY", 0)

        # Rigid region: all shell/edge nodes slave to master
        for slave in self._cerig_slave_nodes:
            if slave != self.master_node:
                m.cerig(self.master_node, slave, "ALL")

        # Imposed displacements on master node
        m.d(self.master_node, "UX", self.ux_imp)
        m.d(self.master_node, "UZ", self.uz_imp)

        m.allsel("ALL")
