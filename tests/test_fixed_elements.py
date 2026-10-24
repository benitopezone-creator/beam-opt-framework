"""Unit tests for fixed (non-optimisable) elements."""

from __future__ import annotations

import numpy as np
import pytest

from beam_opt.model import FixedShell, quad_area_3d


def _rotation_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


# 4 mm x 2.5 mm rectangle in the XY plane [mm]
RECT = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [4.0, 2.5, 0.0], [0.0, 2.5, 0.0]])


@pytest.mark.parametrize("angle_deg", [0.0, 30.0, 60.0, 90.0])
def test_area_is_invariant_under_rotation(angle_deg):
    # The old XY-shoelace formula gave 10*cos(angle): 5 at 60°, 0 at 90°.
    coords = RECT @ _rotation_x(np.radians(angle_deg)).T + [1.0, -2.0, 3.0]
    assert quad_area_3d(coords) == pytest.approx(10.0)


def test_degenerate_quad_is_a_triangle():
    tri = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 0.0, 4.0], [0.0, 0.0, 4.0]])
    assert quad_area_3d(tri) == pytest.approx(6.0)


def test_estimate_mass_of_vertical_shell():
    # Shell in the XZ plane: zero area with the old projection.
    coords = {"a": [0, 0, 0], "b": [4, 0, 0], "c": [4, 0, 2.5], "d": [0, 0, 2.5]}
    shell = FixedShell(["a", "b", "c", "d"], thickness=0.8)
    rho = 7.85e-9  # [ton/mm³]
    assert shell.estimate_mass(coords, rho) == pytest.approx(rho * 10.0 * 0.8)
