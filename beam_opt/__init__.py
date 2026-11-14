"""
beam_opt – Dimensional optimisation framework for complex beam structures.
"""

from .exceptions import SolutionNotConvergedError
from .geometry import BeamMesh, BeamPath
from .materials import Material
from .model import (
    Clamp,
    FixedBeam,
    FixedShell,
    ImposedDisplacement,
    ModelBuilder,
    NodalForce,
    OptimizableBeam,
    RigidRegion,
    StructuralModel,
    Symmetry,
)
from .optimization import CobylaDriver, OptimizationProblem, optimize
from .post import RectBeamStressEvaluator, SectionStressEvaluator, StressEvaluator
from .section_law import SectionLaw
from .sections import BoxSection, CircSection, RectSection, Section, TubeSection
from .session import MapdlSession
from .solver import SolverSettings, solve

__all__ = [
    "BeamMesh",
    "BeamPath",
    "BoxSection",
    "CircSection",
    "Clamp",
    "CobylaDriver",
    "FixedBeam",
    "FixedShell",
    "ImposedDisplacement",
    "MapdlSession",
    "Material",
    "ModelBuilder",
    "NodalForce",
    "OptimizableBeam",
    "OptimizationProblem",
    "RectBeamStressEvaluator",
    "RectSection",
    "RigidRegion",
    "Section",
    "SectionLaw",
    "SectionStressEvaluator",
    "SolutionNotConvergedError",
    "SolverSettings",
    "StressEvaluator",
    "StructuralModel",
    "Symmetry",
    "TubeSection",
    "optimize",
    "solve",
]
