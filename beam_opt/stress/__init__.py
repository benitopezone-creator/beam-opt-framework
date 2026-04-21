"""
beam_opt.stress
===============
Pluggable stress-field evaluators.

To add a new cross-section shape:
1. Subclass ``AbstractStressEvaluator`` in a new module.
2. Implement ``__call__(mapdl, model, law) -> np.ndarray``.
3. Pass an instance to ``BeamOptimizer(…, stress_evaluator=MyEvaluator())``.
"""

from .base import AbstractStressEvaluator
from .rect_beam import RectBeamStressEvaluator

__all__ = ["AbstractStressEvaluator", "RectBeamStressEvaluator"]
