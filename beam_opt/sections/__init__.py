"""
beam_opt.sections
=================
Beam cross-section shapes.

To add a shape, subclass ``Section``: give it a ``from_params`` constructor
taking the section-law parameters by name and its ``stress_points``.
"""

from .base import Section, StressPoints
from .box import BoxSection
from .circ import CircSection, TubeSection
from .rect import RectSection

__all__ = ["BoxSection", "CircSection", "RectSection", "Section", "StressPoints", "TubeSection"]
