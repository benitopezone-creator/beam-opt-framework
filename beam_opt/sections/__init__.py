"""
beam_opt.sections
=================
Beam cross-section shapes.

To add a shape, subclass ``Section`` and give it a ``from_params``
constructor taking the section-law parameters by name.
"""

from .base import Section
from .rect import RectSection

__all__ = ["RectSection", "Section"]
