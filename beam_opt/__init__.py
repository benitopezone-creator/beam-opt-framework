"""
beam_opt – Dimensional optimisation framework for complex beam structures.

Extensions beyond the core (other objectives, fatigue, load cases, trade-off
curves) are in ``beam_opt.extras``.
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
from .native import NativeSolver
from .optimization import CobylaDriver, Mass, OptimizationProblem, optimize
from .post import RectBeamStressEvaluator, SectionStressEvaluator, StressEvaluator
from .section_law import SectionLaw
from .sections import (
    BoxSection,
    CircSection,
    EllipseSection,
    ISection,
    RectSection,
    Section,
    SectionShape,
    TubeSection,
    polygon_shape,
)
from .session import MapdlSession
from .solver import SolverSettings, solve

__all__ = [
    "BeamMesh",
    "BeamPath",
    "BoxSection",
    "CircSection",
    "Clamp",
    "CobylaDriver",
    "EllipseSection",
    "FixedBeam",
    "FixedShell",
    "ISection",
    "ImposedDisplacement",
    "MapdlSession",
    "Mass",
    "Material",
    "ModelBuilder",
    "NativeSolver",
    "NodalForce",
    "OptimizableBeam",
    "OptimizationProblem",
    "RectBeamStressEvaluator",
    "RectSection",
    "RigidRegion",
    "Section",
    "SectionLaw",
    "SectionShape",
    "SectionStressEvaluator",
    "SolutionNotConvergedError",
    "SolverSettings",
    "StressEvaluator",
    "StructuralModel",
    "Symmetry",
    "TubeSection",
    "optimize",
    "polygon_shape",
    "solve",
]
