"""
Phase 2 acceptance: one configuration, any discretisation.

The curved-frame scenario is written the way an application configuration
should be after Phase 2: a spline path through the definition points and
anchors placed on definition points (``path.s_knots[i]``) instead of node IDs.
"""

from __future__ import annotations

import numpy as np
import pytest
import scenario as sc
from fakes import FakeMapdl, canonical_model
from test_phase1_regression import MATERIAL, frame_parts, make_problem

from beam_opt.geometry import BeamPath, discretize
from beam_opt.model import ModelBuilder, OptimizableBeam, StructuralModel


def curved_frame(n_elem: int) -> StructuralModel:
    pts = sc.scenario_points()
    path = BeamPath(pts.pop("arch"))
    anchors = {name: path.s_knots[i] for name, i in sc.JOINTS.items()}
    fixed, bcs = frame_parts()
    return StructuralModel(
        OptimizableBeam(discretize(path, n_elem, anchors)), MATERIAL, pts, fixed, bcs
    )


def _fixed_part(n_elem: int):
    model = curved_frame(n_elem)
    problem = make_problem(model, FakeMapdl())
    fake = problem.mapdl
    ModelBuilder(fake).build(model, problem.law_from_x(np.full(8, 4.0)))
    canon = canonical_model(fake.calls)
    strut = list(sc.STRUT)
    fixed = [e for e in canon["elements"] if e[0] == "SHELL181" or e[2][2] == strut]
    bcs = [c for c in canon["constraints"] if c[0] != "FSUM"]
    return fixed, bcs


@pytest.mark.parametrize("n_elem", [49, 196])
def test_joints_are_the_definition_points(n_elem):
    mesh = curved_frame(n_elem).beam.mesh
    arch = sc.scenario_points()["arch"]
    assert mesh.n_elem == n_elem
    for name, i in sc.JOINTS.items():
        np.testing.assert_allclose(mesh.coords[mesh.anchors[name]], arch[i], atol=1e-9)


def test_fixed_parts_and_bcs_do_not_depend_on_n_elem():
    fixed_49, bcs_49 = _fixed_part(49)
    fixed_196, bcs_196 = _fixed_part(196)
    assert len(fixed_49) == len(fixed_196) == 4
    for e49, e196 in zip(fixed_49, fixed_196, strict=True):
        assert e49[:3] == e196[:3]
        np.testing.assert_allclose(e49[3], e196[3], atol=1e-9)
    assert len(bcs_49) == len(bcs_196)
    for c49, c196 in zip(bcs_49, bcs_196, strict=True):
        assert c49[0] == c196[0]
        np.testing.assert_allclose(c49[1], c196[1], atol=1e-9)


def test_design_variables_do_not_depend_on_n_elem():
    p49 = make_problem(curved_frame(49), FakeMapdl())
    p196 = make_problem(curved_frame(196), FakeMapdl())
    assert p49.n_vars == p196.n_vars == 8
    np.testing.assert_allclose(p49.s_ctrl, p196.s_ctrl, rtol=1e-12)


def test_beam_mass_converges_with_n_elem():
    # Smooth law well inside the bounds: no clipping kinks, so the mid-point
    # mass converges at second order (a clipped law drops to ~first order).
    x = np.array([3.0, 4.5, 5.0, 3.5, 4.0, 3.0, 4.5, 5.0])  # [mm]
    masses = []
    for n in (25, 50, 100, 200, 400):
        p = make_problem(curved_frame(n), FakeMapdl())
        masses.append(p.model.beam.mass(p.law_from_x(x), MATERIAL.rho))
    diffs = np.abs(np.diff(masses))
    assert np.all(diffs[:-1] / diffs[1:] > 3.0)
    assert diffs[-1] / masses[-1] < 1e-5
