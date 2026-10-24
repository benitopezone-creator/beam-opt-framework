"""
Synthetic "curved frame" scenario shared by the regression and mesh tests.

A strongly curved 3-D arch of 10 elements with three fixed struts to
supports on the Z = 0 symmetry plane, and a rigid plate driven by imposed
displacements.  All values are made up; fake BEAM188 results are used, so
only the build sequence and the post-processing arithmetic are exercised:
the model is never solved.
"""

from __future__ import annotations

import numpy as np

N_NODES = 11  # definition points -> 10 optimisable elements

# Generic aluminium alloy [MPa], [-], [ton/mm³], [MPa]
MATERIAL = {"E": 70_000.0, "nu": 0.33, "rho": 2.70e-9, "sigma_y": 250.0}
BOUNDS = (2.0, 6.0)  # section dimension bounds [mm]
SAFETY_FACTOR = 2.0
F_REACT_MAX = 40.0  # [N]
UX, UZ = 12.0, -25.0  # imposed plate displacements [mm]
STRUT = (3.0, 3.0)  # fixed strut section b x h [mm]
PLATE_T = 2.0  # plate thickness [mm]
JOINTS = {"joint_a": 1, "joint_b": 5, "joint_c": 9}  # definition-point indices

# Design vector [b_ctrl(4), h_ctrl(4)] [mm]: 6.8 and 1.6 exercise the
# control-point clipping, the 6.0/2.0 jumps exercise spline overshoot.
X_DESIGN = [2.5, 6.8, 4.5, 2.0, 4.0, 1.6, 5.8, 6.0]

# Fake per-element SMISC results (10 elements)
_k = np.arange(1, N_NODES)
SMISC = {
    sid: (100.0 if sid >= 31 else 10.0 if sid in (5, 6, 18, 19) else 50.0)
    * np.sin(0.37 * sid + 0.71 * _k)
    for sid in [4, 5, 6, 17, 18, 19, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
}

FSUM = {"FX": 1.2, "FY": -0.5, "FZ": 2.0}  # [N]


def scenario_points() -> dict[str, np.ndarray]:
    """Coordinates [mm] of the arch definition points and of the named extra points."""
    t = np.linspace(0.0, np.pi, N_NODES)
    arch = np.column_stack([80.0 * np.cos(t), 25.0 * np.sin(t), 70.0 * np.sin(t)])
    return {
        "arch": arch,
        "support_a": np.array([40.0, -20.0, 0.0]),
        "support_b": np.array([0.0, -20.0, 0.0]),
        "support_c": np.array([-40.0, -20.0, 0.0]),
        # inclined, non-XY-planar plate
        "plate_0": np.array([-8.0, 50.0, -6.0]),
        "plate_1": np.array([8.0, 50.0, -6.0]),
        "plate_2": np.array([8.0, 56.0, 6.0]),
        "plate_3": np.array([-8.0, 56.0, 6.0]),
    }
