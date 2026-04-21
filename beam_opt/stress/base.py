"""
beam_opt.stress.base
====================
Abstract base class for stress-field evaluators.

All evaluators must be callable objects so that they can be swapped without
changing the ``BeamAnalysis`` interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class AbstractStressEvaluator(ABC):
    """
    Interface for per-element equivalent-stress evaluators.

    Implementors receive the live MAPDL session (already positioned in
    ``/POST1`` with the last result set loaded), the ``BeamModel`` instance,
    and the current ``SectionLaw``.  They must return one scalar per
    optimisable element.
    """

    @abstractmethod
    def __call__(self, mapdl, model, law) -> np.ndarray:
        """
        Compute the element-wise equivalent stress field.

        Parameters
        ----------
        mapdl : ansys.mapdl.core.Mapdl
            Active MAPDL session, already in /POST1 with LAST set loaded.
        model : BeamModel
            The FEM model (provides ``n_elem``, ``s``, ``s_elem``, …).
        law : SectionLaw
            Current cross-section distribution.

        Returns
        -------
        vm_per_elem : ndarray, shape (n_elem,)
            Conservative von Mises (or equivalent) stress for each
            optimisable beam element [MPa].
        """
