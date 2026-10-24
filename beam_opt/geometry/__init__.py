"""
beam_opt.geometry
=================
Beam centre-line description and its discretisation.
"""

from .curve import BeamPath
from .mesh import BeamMesh, discretize

__all__ = ["BeamMesh", "BeamPath", "discretize"]
