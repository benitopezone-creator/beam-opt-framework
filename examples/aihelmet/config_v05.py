"""
AI Helmet – geometry configuration V05.

This module supplies the four objects consumed by ``run_optimization.py``:
  - NODES_COORDS      : (50, 3) float array of spline-node coordinates
  - ADDITIONAL_NODES  : {node_id: [x, y, z]} for support + shell-patch nodes
  - MASTER_NODE       : ID of the node on which displacements are imposed
  - FIXED_ELEMENTS    : list of FixedBeam / FixedShell instances
  - UZ_IMP            : imposed displacement in Z (sign convention for V05)

All boundary-condition node sets (symmetry nodes, UY-fixed nodes) are
derived automatically by ``BeamModel`` from the ``ADDITIONAL_NODES`` keys;
nothing is hard-coded inside the framework.
"""

import numpy as np
from beam_opt.fixed_elements import (
    FixedBeam, FixedShell,
    FIXED_BEAM_SECNUM, FIXED_SHELL_SECNUM,
)

CONFIG_NAME = "v05"
UZ_IMP = 90.0   # mm (positive for V05)

# -----------------------------------------------------------------------
# Main spline nodes (50 nodes → 49 BEAM188 optimisable elements)
# -----------------------------------------------------------------------
NODES_COORDS = np.array([
    [0.0, 0.0, 0.0],
    [13.70672288, 4.92931219, 4.86945466],
    [19.40312888, 18.34290564, 7.47073467],
    [9.14977404, 28.96329272, 5.38478884],
    [-0.98532357, 40.3302473, 5.20578545],
    [-3.6159161, 55.41814283, 4.75497323],
    [-2.62032333, 70.53975996, 1.81034742],
    [-0.16783008, 84.66169247, -3.91273441],
    [3.6976795, 97.06898193, -12.23450609],
    [9.33157964, 106.99890914, -22.60238118],
    [19.26300891, 113.89579774, -31.96390328],
    [33.82743602, 116.09463311, -32.45918512],
    [44.69466892, 112.79211113, -22.31673952],
    [49.01722862, 106.34007663, -9.07524006],
    [52.80641682, 97.2510942, 2.63597777],
    [63.9874838, 88.77556201, 8.31241931],
    [79.14937354, 86.92898807, 7.65585089],
    [93.27418329, 89.191264, 2.04865438],
    [105.00442722, 92.01628962, -7.56696449],
    [116.77216342, 89.93069212, -17.19843448],
    [127.83703611, 81.99859835, -24.30787751],
    [136.61886965, 69.93328259, -28.04340628],
    [142.06307267, 55.53716717, -28.8441039],
    [144.51751238, 40.28499846, -28.79046051],
    [145.10604995, 24.83748521, -28.81001772],
    [143.82726794, 9.43927524, -28.80460644],
    [140.41510895, -5.63277028, -28.8043755],
    [134.98095339, -20.09729946, -28.80132954],
    [127.57195021, -33.65155124, -28.44741237],
    [118.1194813, -45.80898317, -27.34352604],
    [106.25705227, -55.40445405, -25.13911783],
    [93.20490665, -59.26493569, -18.89744859],
    [87.9502248, -51.17012953, -7.6512685],
    [94.34282723, -39.26704377, -0.72201427],
    [104.98233319, -28.25200069, 0.97325944],
    [115.23187576, -16.83259355, -0.65897403],
    [122.9135495, -4.02555047, -4.53827684],
    [127.53889113, 10.46889578, -6.99820158],
    [128.56208312, 25.83020243, -7.31022469],
    [123.73032167, 39.9060908, -3.95119345],
    [113.10865514, 48.50000304, 2.83451899],
    [99.87170852, 50.7672619, 10.34097846],
    [86.02122428, 48.24283363, 16.5819272],
    [73.17856607, 40.67551568, 20.19040198],
    [63.09456444, 29.04032941, 21.08473958],
    [53.56651644, 16.92056996, 20.23182813],
    [41.98010003, 7.44749819, 16.66105969],
    [29.07894519, 1.37274191, 10.73838055],
    [15.21786238, -2.97766581, 5.44976651],
    [0.0, -6.56000786, 0.0],
])

# -----------------------------------------------------------------------
# Symmetry-plane support nodes (IDs 51-53)
# -----------------------------------------------------------------------
_STRUCTURAL_NODES = {
    51: [142.06307267, 55.53716717, -51.5141039],
    52: [145.10604995, 24.83748521, -51.4800177],
    53: [140.41510895, -5.63277028, -51.4746044],
}

# -----------------------------------------------------------------------
# Shell-patch nodes for the ear-piece transducer (IDs 1001-1014)
# -----------------------------------------------------------------------
_H_TRAS = 26.0    # mm – housing width  (x direction)
_L_TRAS = 18.6    # mm – housing height (y direction)
_H_SHELL = _H_TRAS / 3.0
_L_SHELL = _L_TRAS / 3.0


def _make_shell_nodes(nodes_coords: np.ndarray) -> dict:
    x1,  y1,  z1  = nodes_coords[0]
    x50, y50, z50 = nodes_coords[49]
    return {
        1001: [x1,               y1 + _L_SHELL,   z1],
        1002: [x1 - _H_SHELL,    y1 + _L_SHELL,   z1],
        1003: [x1 - 2*_H_SHELL,  y1 + _L_SHELL,   z1],
        1004: [x1 - 3*_H_SHELL,  y1 + _L_SHELL,   z1],
        1005: [x1 - 3*_H_SHELL,  y1,               z1],
        1006: [x50 - 3*_H_SHELL, y50,              z50],
        1007: [x50 - 3*_H_SHELL, y50 - _L_SHELL,  z50],
        1008: [x50 - 2*_H_SHELL, y50 - _L_SHELL,  z50],
        1009: [x50 - _H_SHELL,   y50 - _L_SHELL,  z50],
        1010: [x50,              y50 - _L_SHELL,  z50],
        1011: [x1 - _H_SHELL,    y1,               z1],
        1012: [x1 - 2*_H_SHELL,  y1,               z1],
        1013: [x50 - _H_SHELL,   y50,             z50],
        1014: [x50 - 2*_H_SHELL, y50,             z50],
    }


_SHELL_NODES = _make_shell_nodes(NODES_COORDS)
ADDITIONAL_NODES = {**_STRUCTURAL_NODES, **_SHELL_NODES}

MASTER_NODE = 1011

# -----------------------------------------------------------------------
# Fixed elements
# -----------------------------------------------------------------------
_SHELL_T = 7.5  # mm – shell thickness

FIXED_ELEMENTS = [
    # BEAM188 links to symmetry plane
    FixedBeam(50, 23, 51, 8.0, 3.0, FIXED_BEAM_SECNUM),
    FixedBeam(51, 25, 52, 8.0, 3.0, FIXED_BEAM_SECNUM),
    FixedBeam(52, 27, 53, 8.0, 3.0, FIXED_BEAM_SECNUM),

    # SHELL181 3×3 grid of the ear-piece housing
    FixedShell(53, [1001, 1,    1011, 1002], _SHELL_T, FIXED_SHELL_SECNUM),
    FixedShell(54, [1,    50,   1013, 1011], _SHELL_T, FIXED_SHELL_SECNUM),
    FixedShell(55, [50,   1010, 1009, 1013], _SHELL_T, FIXED_SHELL_SECNUM),
    FixedShell(56, [1002, 1011, 1012, 1003], _SHELL_T, FIXED_SHELL_SECNUM),
    FixedShell(57, [1011, 1013, 1014, 1012], _SHELL_T, FIXED_SHELL_SECNUM),
    FixedShell(58, [1013, 1009, 1008, 1014], _SHELL_T, FIXED_SHELL_SECNUM),
    FixedShell(59, [1003, 1012, 1005, 1004], _SHELL_T, FIXED_SHELL_SECNUM),
    FixedShell(60, [1012, 1014, 1006, 1005], _SHELL_T, FIXED_SHELL_SECNUM),
    FixedShell(61, [1014, 1008, 1007, 1006], _SHELL_T, FIXED_SHELL_SECNUM),
]
