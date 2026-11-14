"""
Built-in beam solver (``beam_opt.native``) against closed-form solutions.

Units: mm, N, MPa.  Unless stated, Euler-Bernoulli elements are used so the
classical formulas hold exactly; one test checks the Timoshenko shear term.
"""

from __future__ import annotations

import numpy as np
import pytest
import scenario as sc
from fakes import StaticCantileverMapdl
from test_mesh_independence import curved_frame

from beam_opt.exceptions import SolutionNotConvergedError
from beam_opt.geometry import BeamMesh, BeamPath, discretize
from beam_opt.materials import Material
from beam_opt.model import (
    Clamp,
    FixedBeam,
    FixedShell,
    ImposedDisplacement,
    NodalForce,
    OptimizableBeam,
    RigidRegion,
    StructuralModel,
    Symmetry,
)
from beam_opt.native import NativeSolver, element_axes
from beam_opt.optimization import OptimizationProblem, optimize
from beam_opt.post import RectBeamStressEvaluator
from beam_opt.section_law import SectionLaw
from beam_opt.sections import CircSection, RectSection

E, NU = 2.0e5, 0.3
G = E / (2.0 * (1.0 + NU))
STEEL = Material(E=E, nu=NU, rho=7.85e-9, sigma_y=355.0)
L, P = 300.0, 10.0  # [mm], [N]
B, H = 4.0, 6.0  # [mm]
RECT = RectSection(B, H)
IY, IZ, AREA, J = (float(v) for v in (RECT.iy, RECT.iz, RECT.area, RECT.torsion_constant))
RECT_LAW = SectionLaw([0.0, L], {"b": [B, B], "h": [H, H]}, interpolator="linear")


def straight(n_elem: int, bcs, anchors=None, points=None, fixed=None) -> StructuralModel:
    path = BeamPath([[0.0, 0.0, 0.0], [L, 0.0, 0.0]], kind="polyline")
    return StructuralModel(
        OptimizableBeam(discretize(path, n_elem, anchors)), STEEL, points or {}, fixed or [], bcs
    )


def solve(model, law=RECT_LAW, reaction=("start",), shear=False):
    return NativeSolver(shear_deformation=shear).analyze(model, law, list(reaction))


def disp(res, node: str) -> np.ndarray:
    return res.displacements[res.node_index[node]]


# ── Straight beams ───────────────────────────────────────────────────────


@pytest.mark.parametrize("n_elem", [1, 7])
def test_cantilever_tip_load_is_exact_for_any_mesh(n_elem):
    res = solve(straight(n_elem, [Clamp("start"), NodalForce("end", "FZ", -P)]))
    assert disp(res, "end")[2] == pytest.approx(-P * L**3 / (3.0 * E * IY), rel=1e-10)
    assert disp(res, "end")[4] == pytest.approx(P * L**2 / (2.0 * E * IY), rel=1e-10)  # ROTY
    assert abs(res.forces[2][0]) == pytest.approx(P * L, rel=1e-10)  # MY at the root
    assert abs(res.forces[5][0]) == pytest.approx(P, rel=1e-10)  # SFZ
    assert res.forces[15][-1] == pytest.approx(0.0, abs=1e-8)  # MY at the tip
    assert res.reaction == pytest.approx(P, rel=1e-10)


def test_bending_about_the_other_axis_uses_iz():
    res = solve(straight(3, [Clamp("start"), NodalForce("end", "FY", P)]))
    assert disp(res, "end")[1] == pytest.approx(P * L**3 / (3.0 * E * IZ), rel=1e-10)
    assert abs(res.forces[3][0]) == pytest.approx(P * L, rel=1e-10)  # MZ
    assert abs(res.forces[6][0]) == pytest.approx(P, rel=1e-10)  # SFY


def test_axial_force_and_torque():
    res = solve(
        straight(4, [Clamp("start"), NodalForce("end", "FX", P), NodalForce("end", "MX", 50.0)])
    )
    assert disp(res, "end")[0] == pytest.approx(P * L / (E * AREA), rel=1e-10)
    assert disp(res, "end")[3] == pytest.approx(50.0 * L / (G * J), rel=1e-10)
    np.testing.assert_allclose(res.forces[1], P)  # FX: tension is positive
    np.testing.assert_allclose(np.abs(res.forces[4]), 50.0)  # TQ


def test_timoshenko_adds_the_shear_deflection():
    res = solve(straight(5, [Clamp("start"), NodalForce("end", "FZ", -P)]), shear=True)
    expected = P * L**3 / (3.0 * E * IY) + P * L / (5.0 / 6.0 * G * AREA)
    assert -disp(res, "end")[2] == pytest.approx(expected, rel=1e-10)


def test_internal_moment_follows_the_stress_resultant_convention():
    # Tip load along −Z puts the +z fibres of the root in tension: MY > 0,
    # so that σ = MY·z/Iy (BEAM188: SBzT = MY·z_max/Iyy).
    res = solve(straight(2, [Clamp("start"), NodalForce("end", "FZ", -P)]))
    assert res.forces[2][0] == pytest.approx(P * L)
    # Tip load along +Y puts the −y fibres of the root in tension: σ = −MZ·y/Iz, MZ > 0.
    res = solve(straight(2, [Clamp("start"), NodalForce("end", "FY", P)]))
    assert res.forces[3][0] == pytest.approx(P * L)


# ── Statically indeterminate beams ───────────────────────────────────────


def test_fixed_fixed_beam_with_mid_span_load():
    model = straight(
        6, [Clamp(["start", "end"]), NodalForce("mid", "FZ", -P)], anchors={"mid": L / 2}
    )
    res = solve(model, reaction=("start", "end"))
    assert -disp(res, "mid")[2] == pytest.approx(P * L**3 / (192.0 * E * IY), rel=1e-10)
    assert abs(res.forces[2][0]) == pytest.approx(P * L / 8.0, rel=1e-10)  # end moment
    assert res.reaction == pytest.approx(P, rel=1e-10)


def test_propped_cantilever_reaction():
    bcs = [Clamp("start"), ImposedDisplacement("end", "UZ"), NodalForce("mid", "FZ", -P)]
    res = solve(straight(8, bcs, anchors={"mid": L / 2}), reaction=("end",))
    assert res.reaction == pytest.approx(5.0 * P / 16.0, rel=1e-10)


def test_imposed_displacement():
    delta = 2.5  # [mm]
    res = solve(straight(4, [Clamp("start"), ImposedDisplacement("end", "UZ", delta)]))
    assert disp(res, "end")[2] == pytest.approx(delta)
    assert res.reaction == pytest.approx(3.0 * E * IY * delta / L**3, rel=1e-10)


def test_symmetry_plane_reproduces_the_full_simply_supported_beam():
    # Half of a simply supported beam of span 2L with a central load 2P.
    bcs = [
        ImposedDisplacement("start", "UZ"),
        ImposedDisplacement("start", "UY"),
        ImposedDisplacement("start", "ROTX"),
        Symmetry("end", normal="X"),
        NodalForce("end", "FZ", -P),
    ]
    res = solve(straight(5, bcs))
    span, load = 2.0 * L, 2.0 * P
    assert -disp(res, "end")[2] == pytest.approx(load * span**3 / (48.0 * E * IY), rel=1e-10)


def test_redundant_constraints_are_accepted():
    bcs = [Clamp("start"), ImposedDisplacement("start", "UZ"), NodalForce("end", "FZ", -P)]
    res = solve(straight(3, bcs))
    assert -disp(res, "end")[2] == pytest.approx(P * L**3 / (3.0 * E * IY), rel=1e-10)


def test_inconsistent_constraints_are_rejected():
    bcs = [Clamp("start"), ImposedDisplacement("start", "UZ", 1.0)]
    with pytest.raises(ValueError, match="inconsistent"):
        solve(straight(3, bcs))


# ── Rigid regions and fixed elements ─────────────────────────────────────


def test_rigid_arm_transfers_force_and_moment():
    arm = 40.0  # [mm]
    model = straight(
        4,
        [Clamp("start"), RigidRegion("end", ["handle"]), NodalForce("handle", "FZ", -P)],
        points={"handle": [L + arm, 0.0, 0.0]},
    )
    res = solve(model)
    expected = P * L**3 / (3.0 * E * IY) + P * arm * L**2 / (2.0 * E * IY)
    assert -disp(res, "end")[2] == pytest.approx(expected, rel=1e-10)
    rot_y = disp(res, "end")[4]
    assert disp(res, "handle")[2] == pytest.approx(disp(res, "end")[2] - rot_y * arm, rel=1e-10)


def test_fixed_beam_prop_shares_the_load():
    # A vertical strut of the same section under the tip: axial spring k = EA/h
    # (its small bending restraint on the tip rotation is neglected: 1e-3).
    height = 50.0
    model = straight(
        4,
        [Clamp(["start", "ground"]), NodalForce("end", "FZ", -P)],
        points={"ground": [L, 0.0, -height]},
        fixed=[FixedBeam("end", "ground", RECT)],
    )
    res = solve(model)
    k_beam = 3.0 * E * IY / L**3
    k_strut = E * AREA / height
    assert -disp(res, "end")[2] == pytest.approx(P / (k_beam + k_strut), rel=1e-3)


def test_deformable_shell_is_not_supported():
    pts = {f"p{i}": [L + 10.0 * (i % 2), 10.0 * (i // 2), 5.0] for i in range(4)}
    model = straight(2, [Clamp("start")], points=pts, fixed=[FixedShell(list(pts), 1.0)])
    with pytest.raises(NotImplementedError, match="FixedShell"):
        solve(model)


def test_unsupported_structure_is_a_failed_solution():
    with pytest.raises(SolutionNotConvergedError, match="singular"):
        solve(straight(3, [NodalForce("end", "FZ", -P)]))


# ── Curved beams ─────────────────────────────────────────────────────────

RADIUS, ROD = 200.0, 2.0  # arc radius and rod radius [mm]
ROD_SEC = CircSection(ROD)
ROD_I, ROD_J = float(ROD_SEC.iy), float(ROD_SEC.torsion_constant)


def quarter_arc(n_elem: int, load: NodalForce) -> StructuralModel:
    """Quarter circle in the XY plane, clamped at (R, 0, 0), free at (0, R, 0)."""
    t = np.linspace(0.0, np.pi / 2.0, n_elem + 1)
    pts = np.column_stack([RADIUS * np.cos(t), RADIUS * np.sin(t), np.zeros_like(t)])
    mesh = BeamMesh.from_points(pts)
    return StructuralModel(
        OptimizableBeam(mesh, section_factory=CircSection.from_params),
        STEEL,
        bcs=[Clamp("start"), load],
    )


def arc_law(length: float) -> SectionLaw:
    return SectionLaw([0.0, length], {"r": [ROD, ROD]}, interpolator="linear")


@pytest.mark.parametrize(
    ("direction", "dof", "factor"),
    [("FY", 1, np.pi / 4.0), ("FX", 0, 3.0 * np.pi / 4.0 - 2.0)],
)
def test_quarter_circle_in_plane_bending(direction, dof, factor):
    model = quarter_arc(200, NodalForce("end", direction, P))
    res = solve(model, arc_law(model.beam.mesh.s[-1]))
    # Castigliano, bending only (axial strain is ~r²/4R² = 2.5e-5 of it)
    assert disp(res, "end")[dof] == pytest.approx(factor * P * RADIUS**3 / (E * ROD_I), rel=1e-3)


def test_quarter_circle_out_of_plane_bending_and_torsion():
    model = quarter_arc(200, NodalForce("end", "FZ", P))
    res = solve(model, arc_law(model.beam.mesh.s[-1]))
    expected = P * RADIUS**3 * (np.pi / (4.0 * E * ROD_I) + (3.0 * np.pi - 8.0) / (4.0 * G * ROD_J))
    assert disp(res, "end")[2] == pytest.approx(expected, rel=1e-3)
    # Root: bending moment P·R and torque P·R
    assert abs(res.forces[4][0]) == pytest.approx(P * RADIUS, rel=1e-2)
    assert np.hypot(res.forces[2][0], res.forces[3][0]) == pytest.approx(P * RADIUS, rel=1e-2)


def test_arc_converges_with_the_mesh():
    exact = np.pi / 4.0 * P * RADIUS**3 / (E * ROD_I)
    errors = []
    for n in (10, 20, 40):
        model = quarter_arc(n, NodalForce("end", "FY", P))
        res = solve(model, arc_law(model.beam.mesh.s[-1]))
        errors.append(abs(disp(res, "end")[1] / exact - 1.0))
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 2e-3


def test_element_axes_follow_the_beam188_default():
    p_i = np.zeros((3, 3))
    p_j = np.array([[2.0, 0.0, 0.0], [0.0, 0.0, 5.0], [1.0, 1.0, 1.0]])
    length, axes = element_axes(p_i, p_j)
    np.testing.assert_allclose(length, [2.0, 5.0, np.sqrt(3.0)])
    np.testing.assert_allclose(axes[0], np.eye(3), atol=1e-12)  # along X: local = global
    np.testing.assert_allclose(axes[1][1], [0.0, 1.0, 0.0], atol=1e-12)  # vertical: y = global Y
    assert axes[2][1][2] == pytest.approx(0.0)  # general: local y parallel to the XY plane
    for a in axes:
        np.testing.assert_allclose(a @ a.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(a) == pytest.approx(1.0)


# ── Inside the optimisation problem ──────────────────────────────────────


def _cantilever_problem(mapdl, tmp_path=None):
    return OptimizationProblem(
        mapdl,
        straight(30, [Clamp("start"), NodalForce("end", "FZ", -P)]),
        n_ctrl=5,
        bounds={"b": (2.0, 8.0), "h": (2.0, 10.0)},
        safety_factor=1.5,
        f_react_max=5.0 * P,
        reaction_nodes=["start"],
        interpolator="pchip",
    )


def test_same_stresses_as_the_closed_form_stand_in(tmp_path):
    x = np.array([4.0, 3.5, 5.0, 3.0, 2.5, 8.0, 6.0, 7.0, 4.0, 3.0])  # [mm]
    native = _cantilever_problem(NativeSolver()).evaluate(x)
    reference = _cantilever_problem(StaticCantileverMapdl(str(tmp_path))).evaluate(x)
    np.testing.assert_allclose(native.sigma, reference.sigma, rtol=1e-6)
    assert native.reaction == pytest.approx(reference.reaction)


def test_optimisation_runs_without_ansys():
    problem = _cantilever_problem(NativeSolver())
    result = optimize(problem, "slsqp", x0=problem.upper)
    assert result.success
    assert result.max_violation <= 1e-4
    assert result.fun < 0.5 * problem.objective(problem.upper)
    assert result.nfev == problem.mapdl.n_solves


def test_legacy_evaluator_is_rejected_with_the_native_solver():
    with pytest.raises(TypeError, match="from_forces"):
        OptimizationProblem(
            NativeSolver(),
            straight(3, [Clamp("start")]),
            n_ctrl=2,
            bounds={"b": (2.0, 8.0), "h": (2.0, 10.0)},
            safety_factor=1.5,
            f_react_max=1.0,
            reaction_nodes=["start"],
            evaluator=RectBeamStressEvaluator(),
        )


def test_curved_frame_with_rigid_plate_and_imposed_displacements():
    # Strongly curved arch, three fixed struts on a symmetry plane, a plate
    # (rigid region) driven by imposed displacements: statically indeterminate.
    model = curved_frame(60)
    solver = NativeSolver()
    law = SectionLaw(
        [0.0, model.beam.mesh.s[-1]], {"b": [4.0, 4.0], "h": [4.0, 4.0]}, interpolator="linear"
    )
    res = solver.analyze(model, law, ["plate_0", "plate_1", "plate_2", "plate_3", "start", "end"])

    plate = disp(res, "plate_0")
    assert plate[0] == pytest.approx(sc.UX) and plate[2] == pytest.approx(sc.UZ)
    np.testing.assert_allclose(
        disp(res, "start")[3:], plate[3:], atol=1e-12
    )  # rigid: same rotation
    for support in ("support_a", "support_b", "support_c"):
        assert disp(res, support)[2] == pytest.approx(0.0, abs=1e-9)  # on the symmetry plane
    assert np.isfinite(res.reaction) and res.reaction > 0.0
    # Element equilibrium: the axial force is constant along each unloaded element
    np.testing.assert_allclose(res.forces[1], res.forces[14], rtol=1e-9, atol=1e-9)
