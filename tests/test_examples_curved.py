"""Examples 02 (C-spring), 03 (helix) and 04 (arch from APDL) with the built-in solver."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from beam_opt import NativeSolver, optimize
from beam_opt.apdl import read_apdl, write_apdl

EXAMPLES = Path(__file__).parents[1] / "examples"


def _load(folder: str, name: str):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / folder / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


clip = _load("02_c_spring", "c_spring")
helix = _load("03_helix", "helix")
arch = _load("04_apdl_arch", "apdl_arch")


# ── 02: C-spring ─────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def clip_optimum():
    problem = clip.make_problem()
    return problem, clip.run(problem)


def test_clip_geometry_is_half_a_circular_clip():
    mesh = clip.build_model(60).beam.mesh
    arc = clip.RADIUS * np.radians(180.0 - clip.TIP_ANGLE)
    assert mesh.s[-1] == pytest.approx(arc, rel=1e-4)
    np.testing.assert_allclose(np.linalg.norm(mesh.coords, axis=1), clip.RADIUS, rtol=1e-4)
    np.testing.assert_allclose(mesh.coords[0], [-clip.RADIUS, 0.0, 0.0], atol=1e-9)


def test_grip_force_of_a_constant_clip_matches_castigliano():
    # Half clip, force F along y at the tip: M(θ) = F·R·(cos θt − cos θ)
    problem = clip.make_problem(n_elem=120)
    t = 1.5
    force = problem.evaluate(np.full(problem.n_vars, t)).reaction
    tt = np.radians(clip.TIP_ANGLE)
    integral = (
        (np.pi - tt) * np.cos(tt) ** 2
        + 2.0 * np.cos(tt) * np.sin(tt)
        + 0.5 * (np.pi - tt)
        - 0.25 * np.sin(2.0 * tt)
    )  # ∫ (cos θt − cos θ)² dθ from θt to π
    inertia = clip.WIDTH * t**3 / 12.0
    compliance = clip.RADIUS**3 * integral / (clip.MATERIAL.E * inertia)
    assert force == pytest.approx(0.5 * clip.OPENING / compliance, rel=5e-3)


def test_clip_optimum_satisfies_what_no_constant_thickness_can(clip_optimum):
    problem, result = clip_optimum
    s = clip.summary(problem, result.x)
    assert result.success and result.max_violation <= 1e-4
    assert s["utilisation_uniform"] > 1.0  # the constant clip that grips enough yields
    assert s["utilisation"] <= 1.0 + 1e-3
    assert s["force"] == pytest.approx(clip.F_MIN, rel=1e-3)
    assert s["saving"] > 0.10
    assert result.x[0] > result.x[-1]  # thick at the back, thin at the tip


# ── 03: helix ────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def helix_optimum():
    problem = helix.make_problem(n_elem=60, n_ctrl=8)
    return problem, helix.run(problem)


def test_helix_geometry():
    model = helix.build_model(80)
    mesh = model.beam.mesh
    turn_length = np.hypot(2.0 * np.pi * helix.HELIX_RADIUS, helix.PITCH)
    assert mesh.s[-1] == pytest.approx(helix.TURNS * turn_length, rel=1e-4)
    prop = mesh.coords[mesh.anchors["prop"]]
    np.testing.assert_allclose(prop, helix.helix_point(helix.STRUT_TURN), atol=1e-6)
    np.testing.assert_allclose(model.points["foot"][:2], prop[:2])


def test_helix_works_in_bending_torsion_and_shear_together():
    problem = helix.make_problem(n_elem=60, n_ctrl=8)
    problem.evaluate(problem.upper)
    f = problem.mapdl.last.forces
    assert np.abs(f[4]).max() > 0.2 * np.hypot(f[2], f[3]).max()  # torque vs bending
    assert np.abs(f[2]).max() > 0 and np.abs(f[3]).max() > 0


def test_helix_optimum_uses_the_strut(helix_optimum):
    problem, result = helix_optimum
    s = helix.summary(problem, result.x)
    assert result.max_violation <= 1e-4
    assert s["saving"] > 0.25
    assert s["deflection"] <= helix.DEFLECTION_LIMIT * (1 + 1e-4)
    assert s["utilisation"] <= 1.0 + 1e-3
    # Between the clamp and the strut the tube is much lighter than beyond the strut
    mesh = problem.model.beam.mesh
    r = problem.law_from_x(result.x).eval(mesh.s_elem)["r"]
    before = mesh.s_elem < mesh.s[mesh.anchors["prop"]]
    assert r[before].mean() < 0.6 * r[~before].mean()


# ── 04: arch read from an APDL script ────────────────────────────────────


def test_arch_script_is_read_as_written():
    imp = arch.load()
    mesh = imp.model.beam.mesh
    assert mesh.n_elem == 48
    np.testing.assert_allclose(np.linalg.norm(mesh.coords, axis=1), 300.0)
    np.testing.assert_allclose(imp.initial["b"], 12.0)
    np.testing.assert_allclose(imp.initial["h"], 20.0)
    loads = {(bc.node, bc.direction): bc.value for bc in imp.model.bcs if hasattr(bc, "direction")}
    assert loads == {("n25", "FY"): -900.0, ("n25", "FZ"): 250.0}


def test_arch_script_parameters_can_drive_the_mesh():
    assert arch.load({"unused": 1.0}).model.beam.mesh.n_elem == 48


def test_arch_is_optimised_and_written_back(tmp_path):
    imp = arch.load()
    problem = arch.make_problem(imp, n_ctrl=5)
    result = arch.run(problem, imp)
    s = arch.summary(problem, imp, result.x)
    assert result.max_violation <= 1e-4
    assert s["saving"] > 0.4
    assert s["utilisation"] <= 1.0 + 2e-3
    assert s["deflection"] <= arch.CROWN_LIMIT * (1 + 1e-3)
    assert s["deflection"] > s["deflection_script"]  # lighter, hence more flexible

    out = tmp_path / "arch_optimised.inp"
    write_apdl(problem.model, problem.law_from_x(result.x), str(out))
    again = read_apdl(str(out), sigma_y=arch.SIGMA_Y)
    mass = again.model.beam.mass(again.initial_law(), again.model.material.rho) * 1e6
    assert mass == pytest.approx(result.fun, rel=1e-9)


def test_budget_of_fem_solutions_is_a_hard_limit():
    imp = arch.load()
    problem = arch.make_problem(imp, n_ctrl=5)
    result = optimize(problem, "slsqp", x0=imp.x0(problem), max_evals=60)
    assert result.nfev <= 61
    assert not result.success and "budget" in result.message
    assert isinstance(problem.mapdl, NativeSolver)
