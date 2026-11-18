"""
beam_opt – Dimensional optimisation framework for complex beam structures.
"""

from .exceptions import SolutionNotConvergedError
from .fatigue import FatigueModel, StrainLifeCurve
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
from .optimization import (
    CobylaDriver,
    Compliance,
    Displacement,
    FatigueLife,
    Mass,
    OptimizationProblem,
    PeakStress,
    WeightedSum,
    epsilon_constraint,
    optimize,
)
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
    "Compliance",
    "Displacement",
    "EllipseSection",
    "FatigueLife",
    "FatigueModel",
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
    "PeakStress",
    "RectBeamStressEvaluator",
    "RectSection",
    "RigidRegion",
    "Section",
    "SectionLaw",
    "SectionShape",
    "SectionStressEvaluator",
    "SolutionNotConvergedError",
    "SolverSettings",
    "StrainLifeCurve",
    "StressEvaluator",
    "StructuralModel",
    "Symmetry",
    "TubeSection",
    "WeightedSum",
    "epsilon_constraint",
    "optimize",
    "polygon_shape",
    "solve",
]
