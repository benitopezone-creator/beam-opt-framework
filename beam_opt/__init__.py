"""
beam_opt – Dimensional optimisation framework for complex beam structures.
"""

from .mapdl_manager import MapdlManager
from .section_law import SectionLaw
from .fixed_elements import FixedBeam, FixedShell, FIXED_BEAM_SECNUM, FIXED_SHELL_SECNUM
from .model import BeamModel
from .analysis import BeamAnalysis
from .optimization import BeamOptimizer

__all__ = [
    "MapdlManager",
    "SectionLaw",
    "FixedBeam",
    "FixedShell",
    "FIXED_BEAM_SECNUM",
    "FIXED_SHELL_SECNUM",
    "BeamModel",
    "BeamAnalysis",
    "BeamOptimizer",
]
