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
from .custom import ScaledSection, SectionShape, polygon_shape
from .open import EllipseSection, ISection
from .rect import RectSection

#: MAPDL ``SECTYPE,,BEAM,<subtype>`` -> section class (used by the APDL import).
MAPDL_SUBTYPES: dict[str, type[Section]] = {
    "RECT": RectSection,
    "CSOLID": CircSection,
    "CTUBE": TubeSection,
    "HREC": BoxSection,
    "I": ISection,
}

__all__ = [
    "MAPDL_SUBTYPES",
    "BoxSection",
    "CircSection",
    "EllipseSection",
    "ISection",
    "RectSection",
    "ScaledSection",
    "Section",
    "SectionShape",
    "StressPoints",
    "TubeSection",
    "polygon_shape",
]
