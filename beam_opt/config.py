"""
beam_opt.config
===============
Global physical constants and solver parameters.

All values are intentionally kept here (not scattered across modules) so that
switching material or tightening solver tolerances requires a single-file edit.

Geometry – node coordinates, fixed elements, master node – lives in the
application-specific configuration modules under examples/.
"""

from dataclasses import dataclass, field


# -----------------------------------------------------------------------
# Material: Ti6Al4V (additive manufacturing grade)
# -----------------------------------------------------------------------
E: float = 104_800.0   # Young's modulus [MPa]
NU: float = 0.31       # Poisson's ratio
RHO: float = 4.428e-9  # Density [ton/mm³]  (MAPDL uses SI-mm: ton, mm, N, MPa)

# Yield stress and default safety factor
SIGMA_S: float = 820.0  # [MPa]
FS: float = 1.5


# -----------------------------------------------------------------------
# Section bounds (additive manufacturing constraints)
# -----------------------------------------------------------------------
B_MIN: float = 1.2   # [mm]
B_MAX: float = 3.0   # [mm]
H_MIN: float = 1.2   # [mm]
H_MAX: float = 3.0   # [mm]


# -----------------------------------------------------------------------
# Opening-force constraint
# -----------------------------------------------------------------------
F_REACT_MAX: float = 5.0  # [N]


# -----------------------------------------------------------------------
# Imposed displacements
# -----------------------------------------------------------------------
UX_IMP: float = 30.0   # [mm]
UZ_IMP: float = 90.0   # [mm]  positive for V05, negative for V07


# -----------------------------------------------------------------------
# MAPDL solver settings
# -----------------------------------------------------------------------
JOBNAME: str = "beam_opt"

@dataclass
class SolverSettings:
    """Centralised MAPDL solver knobs – override per-run via dataclass replace."""
    nproc: int = 4
    jobname: str = JOBNAME
    nsubst_target: int = 10
    nsubst_max: int = 100
    nsubst_min: int = 5
    nlgeom: bool = True          # large-displacement analysis

DEFAULT_SOLVER = SolverSettings()
