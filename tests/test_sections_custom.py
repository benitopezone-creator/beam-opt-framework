"""
I section, ellipse and user-defined shapes (polygon outline, known properties),
including sections with a product of inertia in the built-in solver.
"""

from __future__ import annotations

import numpy as np
import pytest
from fakes import FakeMapdl
from test_section_stress import FX, MY, MZ, SFZ, SQRT3, TQ

from beam_opt.geometry import BeamMesh, BeamPath, discretize
from beam_opt.materials import Material
from beam_opt.model import Clamp, NodalForce, OptimizableBeam, StructuralModel
from beam_opt.native import NativeSolver
from beam_opt.optimization import OptimizationProblem, optimize
from beam_opt.post import SectionStressEvaluator
from beam_opt.section_law import SectionLaw
from beam_opt.sections import (
    MAPDL_SUBTYPES,
    CircSection,
    EllipseSection,
    ISection,
    RectSection,
    SectionShape,
    polygon_shape,
)

STEEL = Material(E=2.0e5, nu=0.3, rho=7.85e-9, sigma_y=355.0)
IPE = ISection(100.0, 200.0, 5.6, 8.5)  # IPE 200 without fillets [mm]


def _vm(factory, params: dict[str, float], smisc: dict[int, float]) -> float:
    """Element stress of a one-element beam with a constant section [MPa]."""
    law = SectionLaw([0.0, 1.0], {k: [v, v] for k, v in params.items()}, None, "linear")
    beam = OptimizableBeam(BeamMesh.from_points([[0, 0, 0], [1, 0, 0]]), section_factory=factory)
    fake = FakeMapdl(smisc={k: [v] for k, v in smisc.items()})
    return float(SectionStressEvaluator()(fake, beam, law)[0])


IPE_PARAMS = {"b": 100.0, "h": 200.0, "tw": 5.6, "tf": 8.5}


# ── I section ────────────────────────────────────────────────────────────


def test_i_section_properties():
    assert IPE.area == pytest.approx(2.0 * 100.0 * 8.5 + 183.0 * 5.6)
    assert IPE.iy == pytest.approx((100.0 * 200.0**3 - 94.4 * 183.0**3) / 12.0)
    assert IPE.iz == pytest.approx((2.0 * 8.5 * 100.0**3 + 183.0 * 5.6**3) / 12.0)
    assert IPE.torsion_constant == pytest.approx((2 * 100.0 * 8.5**3 + 191.5 * 5.6**3) / 3.0)
    # Tabulated IPE 200: A = 2848 mm², Iy = 1943e4 mm⁴, Iz = 142.4e4 mm⁴.  The
    # tables include the root fillets, which this idealised shape does not have.
    assert IPE.area == pytest.approx(2848.0, rel=0.06)
    assert IPE.iy == pytest.approx(1943e4, rel=0.06)
    assert IPE.iz == pytest.approx(142.4e4, rel=0.06)


def test_i_section_stresses():
    iy, iz, j = float(IPE.iy), float(IPE.iz), float(IPE.torsion_constant)
    assert _vm(ISection.from_params, IPE_PARAMS, {MY: 1e6}) == pytest.approx(1e6 * 100.0 / iy)
    assert _vm(ISection.from_params, IPE_PARAMS, {MZ: 1e5}) == pytest.approx(1e5 * 50.0 / iz)
    assert _vm(ISection.from_params, IPE_PARAMS, {TQ: 1e4}) == pytest.approx(SQRT3 * 1e4 * 8.5 / j)
    q_half = 100.0 * 8.5 * 95.75 + 5.6 * 91.5**2 / 2.0  # first moment of half the section
    web = SQRT3 * 1e4 * q_half / (iy * 5.6)
    assert _vm(ISection.from_params, IPE_PARAMS, {SFZ: 1e4}) == pytest.approx(web)
    # The web carries nearly all of Vz: average V / (h·tw) within 15 %
    assert web / SQRT3 == pytest.approx(1e4 / (200.0 * 5.6), rel=0.15)


def test_fixed_parameters_are_not_design_variables():
    factory = ISection.with_fixed(tw=5.6, tf=8.5)
    section = factory({"b": np.array([100.0, 120.0]), "h": np.array([200.0, 200.0])})
    assert section.area[0] == pytest.approx(float(IPE.area))
    assert section.area[1] > section.area[0]
    with pytest.raises(ValueError, match="no parameters"):
        ISection.with_fixed(radius=1.0)


@pytest.mark.parametrize(
    ("subtype", "data"),
    [
        ("RECT", (4.0, 6.0)),
        ("CSOLID", (3.0,)),
        ("CTUBE", (9.0, 10.0)),
        ("HREC", (20.0, 30.0, 1.0, 1.0, 1.0, 1.0)),
        ("I", (100.0, 100.0, 200.0, 8.5, 8.5, 5.6)),
    ],
)
def test_secdata_round_trip(subtype, data):
    section = MAPDL_SUBTYPES[subtype].from_secdata(*data)
    assert section.mapdl_section() == (subtype, pytest.approx(data))


# ── Ellipse ──────────────────────────────────────────────────────────────


def test_ellipse_reduces_to_the_circle():
    rng = np.random.default_rng(1)
    for _ in range(20):
        forces = dict(zip(range(1, 7), rng.uniform(-1, 1, 6) * [100, 500, 500, 500, 50, 50]))
        ellipse = _vm(EllipseSection.from_params, {"a": 4.0, "b": 4.0}, forces)
        circle = _vm(CircSection.from_params, {"r": 4.0}, forces)
        assert ellipse == pytest.approx(circle, rel=1e-9)


def test_ellipse_torsion_peaks_at_the_minor_axis():
    a, b, tq = 6.0, 3.0, 900.0
    sec = EllipseSection(a, b)
    assert sec.torsion_constant == pytest.approx(np.pi * a**3 * b**3 / (a**2 + b**2))
    expected = SQRT3 * 2.0 * tq / (np.pi * a * b**2)
    assert _vm(EllipseSection.from_params, {"a": a, "b": b}, {TQ: tq}) == pytest.approx(expected)


def test_shape_without_a_mapdl_library_equivalent_is_written_as_asec():
    fake = FakeMapdl()
    sec = EllipseSection(6.0, 3.0)
    sec.define(fake, 3, "ELL")
    assert fake.calls[0] == ("sectype", (3, "BEAM", "ASEC", "ELL"))
    data = fake.calls[1][1]
    assert data[:6] == pytest.approx(
        (float(sec.area), float(sec.iy), 0.0, float(sec.iz), 0.0, float(sec.torsion_constant))
    )
    assert data[-2:] == pytest.approx((6.0, 12.0))  # TKz, TKy: overall dimensions


# ── Polygon outline ──────────────────────────────────────────────────────

B, H = 4.0, 6.0
RECT_SHAPE = polygon_shape([(0, 0), (B, 0), (B, H), (0, H)])


def test_polygon_rectangle_properties():
    rect = RectSection(B, H)
    assert RECT_SHAPE.area == pytest.approx(B * H)
    assert RECT_SHAPE.iy == pytest.approx(float(rect.iy))
    assert RECT_SHAPE.iz == pytest.approx(float(rect.iz))
    assert RECT_SHAPE.iyz == pytest.approx(0.0, abs=1e-9)
    assert RECT_SHAPE.torsion_constant == pytest.approx(float(rect.torsion_constant), rel=5e-3)


@pytest.mark.parametrize(
    ("forces", "tol"),
    [
        ({FX: 100.0, MY: 400.0, MZ: -300.0}, 1e-9),  # bending: exact at the vertices
        ({SFZ: 60.0}, 1e-2),  # Jourawski
        ({TQ: 800.0}, 2.5e-2),  # numerical Saint-Venant torsion
        ({TQ: 800.0, SFZ: 60.0, MY: 300.0}, 2.5e-2),
    ],
)
def test_polygon_rectangle_stresses_match_the_rectangle(forces, tol):
    polygon = _vm(RECT_SHAPE, {"scale": 1.0}, forces)
    exact = _vm(RectSection.from_params, {"b": B, "h": H}, forces)
    assert polygon == pytest.approx(exact, rel=tol)


def test_polygon_torsion_of_an_equilateral_triangle():
    a = 10.0
    shape = polygon_shape([(0, 0), (a, 0), (a / 2, a * np.sqrt(3) / 2)])
    assert shape.torsion_constant == pytest.approx(np.sqrt(3.0) * a**4 / 80.0, rel=5e-3)
    tau = np.hypot(shape.points.ty_t, shape.points.tz_t).max()
    assert tau == pytest.approx(20.0 / a**3, rel=2.5e-2)  # τ_max = 20 T / a³


def test_polygon_does_not_depend_on_origin_or_direction():
    pts = np.array([(0, 0), (6, 0), (4, 3), (2, 3)], dtype=float)  # trapezoid
    ref = polygon_shape(pts, resolution=40)
    moved = polygon_shape(pts[::-1] + [50.0, -20.0], resolution=40)
    for name in ("area", "iy", "iz", "iyz", "torsion_constant"):
        assert getattr(moved, name) == pytest.approx(getattr(ref, name), rel=1e-6, abs=1e-9)
    assert ref.points.y.mean() == pytest.approx(moved.points.y.mean(), abs=1e-9)


def test_scaling_laws():
    s = np.array([1.0, 2.0])
    section = RECT_SHAPE({"scale": s})
    np.testing.assert_allclose(section.area, B * H * s**2)
    np.testing.assert_allclose(section.iy, RECT_SHAPE.iy * s**4)
    np.testing.assert_allclose(section.torsion_modulus[1], 8.0 * section.torsion_modulus[0])
    pts = section.stress_points()
    assert pts.y.shape == (len(RECT_SHAPE.points.y), 2)
    # same torque, double size: one eighth of the stress
    assert _vm(RECT_SHAPE, {"scale": 2.0}, {TQ: 800.0}) == pytest.approx(
        _vm(RECT_SHAPE, {"scale": 1.0}, {TQ: 800.0}) / 8.0
    )


def test_shape_from_properties_bounds_the_stress():
    rect = RectSection(B, H)
    shape = SectionShape.from_properties(
        float(rect.area),
        float(rect.iy),
        float(rect.iz),
        float(rect.torsion_constant),
        y_max=B / 2,
        z_max=H / 2,
        torsion_modulus=float(rect.torsion_modulus),
    )
    for forces in ({MY: 400.0, MZ: -300.0}, {TQ: 800.0}, {SFZ: 60.0, TQ: 500.0}):
        exact = _vm(RectSection.from_params, {"b": B, "h": H}, forces)
        assert _vm(shape, {"scale": 1.0}, forces) == pytest.approx(exact)


# ── Product of inertia: rotated rectangle ────────────────────────────────

BETA = np.radians(30.0)
L, P = 300.0, 10.0


def rotated_rectangle():
    c, s = np.cos(BETA), np.sin(BETA)
    corners = np.array([(-B / 2, -H / 2), (B / 2, -H / 2), (B / 2, H / 2), (-B / 2, H / 2)])
    return polygon_shape(corners @ np.array([[c, s], [-s, c]]))


def test_rotated_rectangle_has_a_product_of_inertia():
    shape = rotated_rectangle()
    rect = RectSection(B, H)
    assert shape.iy + shape.iz == pytest.approx(float(rect.iy + rect.iz))
    assert abs(shape.iyz) == pytest.approx(
        abs(float(rect.iy - rect.iz)) / 2.0 * np.sin(2.0 * BETA), rel=1e-9
    )


def test_native_solver_bends_an_unsymmetric_section_sideways():
    shape = rotated_rectangle()
    path = BeamPath([[0.0, 0.0, 0.0], [L, 0.0, 0.0]], kind="polyline")
    model = StructuralModel(
        OptimizableBeam(discretize(path, 4), section_factory=shape),
        STEEL,
        bcs=[Clamp("start"), NodalForce("end", "FZ", -P)],
    )
    law = SectionLaw([0.0, L], {"scale": [1.0, 1.0]}, interpolator="linear")
    res = NativeSolver(shear_deformation=False).analyze(model, law, ["start"])

    # Closed form: resolve the load along the principal axes of the rectangle
    rect = RectSection(B, H)
    e_b = np.array([np.cos(BETA), np.sin(BETA)])  # direction of side b in (y, z)
    e_h = np.array([-np.sin(BETA), np.cos(BETA)])
    load = np.array([0.0, -P])
    k = L**3 / (3.0 * STEEL.E)
    expected = (load @ e_b) * k / float(rect.iz) * e_b + (load @ e_h) * k / float(rect.iy) * e_h
    tip = res.displacements[res.node_index["end"]]
    np.testing.assert_allclose(tip[1:3], expected, rtol=1e-9)
    assert abs(tip[1]) > 0.1 * abs(tip[2])  # a clear sideways deflection

    # Root stress: the two principal bending stresses add at one corner
    sigma = SectionStressEvaluator().from_forces(res.forces, model.beam, law)
    m_root = P * L
    expected_sigma = m_root * abs(e_h[1]) * (H / 2) / float(rect.iy) + m_root * abs(e_b[1]) * (
        B / 2
    ) / float(rect.iz)
    assert sigma[0] == pytest.approx(expected_sigma, rel=1e-6)


def test_optimisation_with_a_custom_shape():
    shape = polygon_shape([(0, 0), (6, 0), (4, 3), (2, 3)], resolution=40)  # trapezoid [mm]
    path = BeamPath([[0.0, 0.0, 0.0], [200.0, 0.0, 0.0]], kind="polyline")
    model = StructuralModel(
        OptimizableBeam(discretize(path, 40), section_factory=shape),
        STEEL,
        bcs=[Clamp("start"), NodalForce("end", "FZ", -20.0)],
    )
    problem = OptimizationProblem(
        NativeSolver(),
        model,
        n_ctrl=6,
        bounds={"scale": (0.3, 3.0)},
        safety_factor=1.5,
        f_react_max=100.0,
        reaction_nodes=["start"],
        interpolator="pchip",
    )
    result = optimize(problem, "slsqp", x0=problem.upper)
    assert result.success and result.max_violation <= 1e-4
    # Uniform strength: σ ∝ M / s³ constant  →  s ∝ (L − x)^(1/3), tapering to the tip
    assert np.all(np.diff(result.x) < 0.0)
    assert result.x[3] / result.x[0] == pytest.approx((1.0 - 0.6) ** (1.0 / 3.0), rel=0.03)
    assert problem.evaluate(result.x).sigma.max() == pytest.approx(problem.sigma_adm, rel=1e-3)
