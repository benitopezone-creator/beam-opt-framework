"""Tests for the section-field constraint and the COBYLA driver set-up."""

from __future__ import annotations

import numpy as np
import pytest
from fakes import FakeMapdl

from beam_opt.geometry import BeamMesh
from beam_opt.materials import Material
from beam_opt.model import Clamp, OptimizableBeam, StructuralModel
from beam_opt.optimization import CobylaDriver, OptimizationProblem

BOUNDS = (2.0, 6.0)  # [mm]
# All control values admissible, but the natural cubic spline overshoots 6.0 mm
# between the two central control points.
OVERSHOOT = np.array([2.0, 6.0, 6.0, 2.0])  # [mm]


def _problem(interpolator: str = "natural_cubic") -> OptimizationProblem:
    pts = np.column_stack([np.linspace(0.0, 30.0, 61), np.zeros(61), np.zeros(61)])
    model = StructuralModel(
        OptimizableBeam(BeamMesh.from_points(pts)),
        Material(E=2.0e5, nu=0.3, rho=7.85e-9, sigma_y=355.0),
        bcs=[Clamp("start")],
    )
    return OptimizationProblem(
        FakeMapdl(),  # no FEM run is triggered by these tests
        model,
        n_ctrl=4,
        bounds={"b": BOUNDS, "h": BOUNDS},
        safety_factor=2.0,
        f_react_max=40.0,
        reaction_nodes=["start"],
        interpolator=interpolator,
    )


def test_section_field_constraint_detects_overshoot():
    x = np.concatenate([OVERSHOOT, OVERSHOOT])
    assert _problem().constraint_section_field(x).min() < 0.0


def test_pchip_law_does_not_overshoot():
    x = np.concatenate([OVERSHOOT, OVERSHOOT])
    assert _problem("pchip").constraint_section_field(x).min() >= -1e-12


def test_section_field_constraint_satisfied_for_constant_law():
    x = np.full(8, 4.0)
    assert _problem().constraint_section_field(x).min() >= 0.0


def test_unknown_reaction_node_is_rejected():
    with pytest.raises(ValueError, match="unknown reaction node names"):
        OptimizationProblem(
            FakeMapdl(),
            _problem().model,
            n_ctrl=4,
            bounds={"b": BOUNDS},
            safety_factor=1.0,
            f_react_max=1.0,
            reaction_nodes=["nowhere"],
        )


def test_driver_default_x0_and_size_check():
    driver = CobylaDriver(_problem())
    np.testing.assert_allclose(driver.default_x0(), np.full(8, 4.0))
    with pytest.raises(ValueError, match="expected 8"):
        driver.run(x0=np.ones(3))
