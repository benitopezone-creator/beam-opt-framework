"""
beam_opt.post
=============
Result extraction and stress evaluation.

To add a cross-section shape with its own stress recovery, subclass
``StressEvaluator`` and pass an instance to ``OptimizationProblem``.
"""

from .extract import beam_smisc, last_results, reaction_resultant
from .stress import RectBeamStressEvaluator, StressEvaluator

__all__ = [
    "RectBeamStressEvaluator",
    "StressEvaluator",
    "beam_smisc",
    "last_results",
    "reaction_resultant",
]
