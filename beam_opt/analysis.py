"""
beam_opt.analysis
=================
FEM solve and post-processing orchestrator.

``BeamAnalysis`` wraps the MAPDL ``/SOLU`` and ``/POST1`` phases and
delegates the stress-field computation to a pluggable ``AbstractStressEvaluator``.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from .mapdl_manager import MapdlManager
from .model import BeamModel
from .stress import AbstractStressEvaluator, RectBeamStressEvaluator

logger = logging.getLogger(__name__)


class BeamAnalysis:
    """
    Solver and post-processor for a ``BeamModel``.

    Parameters
    ----------
    model : BeamModel
        The FEM model.
    stress_evaluator : AbstractStressEvaluator | None
        Callable that computes the per-element equivalent stress.
        Defaults to ``RectBeamStressEvaluator()`` (BEAM188 rectangular).
    """

    def __init__(
        self,
        model: BeamModel,
        stress_evaluator: AbstractStressEvaluator | None = None,
    ) -> None:
        self.model = model
        self.mapdl = MapdlManager.get(settings=model.settings)
        self._stress_evaluator = stress_evaluator or RectBeamStressEvaluator()

    # ------------------------------------------------------------------
    # Solver
    # ------------------------------------------------------------------

    def solve(self) -> None:
        """
        Run a nonlinear static analysis and verify that a result file was written.

        Raises
        ------
        RuntimeError
            If MAPDL does not produce an RST file (solution failed).
        """
        m = self.mapdl
        s = self.model.settings

        rst_path = os.path.join(m.directory, f"{s.jobname}.rst")

        # Remove stale RST so we can detect a fresh successful solve
        if os.path.exists(rst_path):
            try:
                os.remove(rst_path)
            except PermissionError:
                logger.warning("Could not remove stale RST file: %s", rst_path)

        m.run("/SOLU")
        m.antype("STATIC")
        m.nlgeom("ON" if s.nlgeom else "OFF")
        m.nsubst(s.nsubst_target, s.nsubst_max, s.nsubst_min)
        m.autots("ON")
        m.outres("ALL", "ALL")
        m.solve()
        m.finish()

        if not os.path.exists(rst_path):
            raise RuntimeError(
                f"FEM solution failed – RST not found at {rst_path}"
            )

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def stress_field(self) -> np.ndarray:
        """
        Compute the per-element equivalent stress using the plugged-in evaluator.

        Returns
        -------
        vm_per_elem : ndarray, shape (n_elem,)
        """
        m = self.mapdl
        m.post1()
        m.set("LAST")

        result = self._stress_evaluator(m, self.model, self.model.law)

        m.allsel("ALL")
        m.finish()
        return result

    def max_reaction(self) -> float:
        """
        Compute the resultant reaction force on the master-node shell patch.

        The shell-patch nodes (ID ≥ ``shell_patch_min_id``) plus the two
        edge spline nodes (1 and ``n_nodes``) are summed via FSUM.

        Returns
        -------
        float
            Resultant reaction modulus [N].
        """
        m = self.mapdl
        mdl = self.model

        m.post1()
        m.set("LAST")

        # Build the reaction node set
        patch_nodes = [nid for nid in mdl.additional_nodes if nid >= mdl.shell_patch_min_id]
        reaction_nodes = patch_nodes + [1, mdl.n_nodes]

        m.nsel("S", "NODE", "", reaction_nodes[0])
        for nid in reaction_nodes[1:]:
            m.nsel("A", "NODE", "", nid)

        m.fsum()

        fx = m.get_value("FSUM", 0, "ITEM", "FX")
        fy = m.get_value("FSUM", 0, "ITEM", "FY")
        fz = m.get_value("FSUM", 0, "ITEM", "FZ")

        m.allsel("ALL")
        m.finish()

        return float(np.sqrt(fx**2 + fy**2 + fz**2))
