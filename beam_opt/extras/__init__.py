"""
beam_opt.extras
===============
Extensions beyond the core of the framework.

The core of beam-opt-framework is the minimum-mass sizing of curved beams
under stress, stiffness and reaction constraints.  This package adds tools
developed on top of it, kept separate from the core:

* objectives other than the mass: ``Compliance`` (stiffness),
  ``Displacement``, ``PeakStress``, ``FatigueLife``, ``WeightedSum``;
* low-cycle fatigue by the local strain approach: ``StrainLifeCurve``,
  ``FatigueModel``; variable amplitude with Miner's rule: ``LoadBlock``,
  ``LoadSpectrum``, ``rainflow``;
* independent load cases: ``LoadCase``;
* trade-off curves: ``epsilon_constraint``.

They plug into the core through generic hooks of ``OptimizationProblem``
(``objective``, ``mass_max``, ``fatigue``, ``life_min``, ``load_cases``).
See ``docs/extras.md`` for the theory and ``examples/extras`` for examples.
"""

from .fatigue import FatigueModel, LoadBlock, LoadSpectrum, StrainLifeCurve, rainflow
from .loadcase import LoadCase
from .objectives import Compliance, Displacement, FatigueLife, PeakStress, WeightedSum, ks_max
from .pareto import ParetoPoint, epsilon_constraint

__all__ = [
    "Compliance",
    "Displacement",
    "FatigueLife",
    "FatigueModel",
    "LoadBlock",
    "LoadCase",
    "LoadSpectrum",
    "ParetoPoint",
    "PeakStress",
    "StrainLifeCurve",
    "WeightedSum",
    "epsilon_constraint",
    "ks_max",
    "rainflow",
]
