"""
Phase 3: section shapes, their check points and the generic stress evaluator.

Internal forces are fed through ``FakeMapdl`` (SMISC 1–6 at node I, 14–19 at
node J: FX, MY, MZ, TQ, SFZ, SFY) and the result is compared with textbook
formulas.
"""

from __future__ import annotations

import numpy as np
import pytest
from fakes import FakeMapdl, StaticCantileverMapdl

from beam_opt.geometry import BeamMesh, BeamPath, discretize
from beam_opt.materials import Material
from beam_opt.model import Clamp, NodalForce, OptimizableBeam, StructuralModel
from beam_opt.optimization import OptimizationProblem, optimize
from beam_opt.post import RectBeamStressEvaluator, SectionStressEvaluator
from beam_opt.section_law import SectionLaw
from beam_opt.sections import BoxSection, CircSection, RectSection, TubeSection

FX, MY, MZ, TQ, SFZ, SFY = 1, 2, 3, 4, 5, 6  # SMISC items at node I (+13 at node J)
SQRT3 = np.sqrt(3.0)

RECT = RectSection(2.0, 3.0)
CIRC = CircSection(4.0)
TUBE = TubeSection(10.0, 0.5)
BOX = BoxSection(20.0, 30.0, 1.0)
SHAPES = [RECT, CIRC, TUBE, BOX]


def _vm(section, smisc: dict[int, float], evaluator=None) -> float:
    """Element stress of a one-element beam with constant *section* [MPa]."""
    names = type(section).param_names
    law = SectionLaw(
        [0.0, 1.0], {n: [float(getattr(section, n))] * 2 for n in names}, None, "linear"
    )
    beam = OptimizableBeam(
        BeamMesh.from_points([[0, 0, 0], [1, 0, 0]]), section_factory=type(section).from_params
    )
    fake = FakeMapdl(smisc={k: [v] for k, v in smisc.items()})
    return float((evaluator or SectionStressEvaluator())(fake, beam, law)[0])


# ── Section properties ───────────────────────────────────────────────────


def test_circle_properties():
    r = 4.0
    assert CIRC.area == pytest.approx(np.pi * r**2)
    assert CIRC.iy == pytest.approx(np.pi * r**4 / 4.0)
    assert CIRC.torsion_constant == pytest.approx(np.pi * r**4 / 2.0)
    assert CIRC.torsion_modulus == pytest.approx(np.pi * r**3 / 2.0)
    assert CIRC.shear_factor == pytest.approx(4.0 / 3.0)


def test_thin_tube_tends_to_the_thin_wall_formulas():
    r, t = 10.0, 0.01
    tube = TubeSection(r, t)
    rm = r - t / 2.0
    assert tube.area == pytest.approx(2.0 * np.pi * rm * t, rel=1e-6)
    assert tube.iy == pytest.approx(np.pi * rm**3 * t, rel=1e-3)
    assert tube.shear_factor == pytest.approx(2.0, rel=1e-3)


def test_box_properties():
    b, h, t = 20.0, 30.0, 1.0
    assert BOX.area == pytest.approx(b * h - 18.0 * 28.0)
    assert BOX.iy == pytest.approx((b * h**3 - 18.0 * 28.0**3) / 12.0)
    assert BOX.iz == pytest.approx((h * b**3 - 28.0 * 18.0**3) / 12.0)
    am = 19.0 * 29.0
    assert BOX.torsion_constant == pytest.approx(4.0 * am**2 * t / (2.0 * (19.0 + 29.0)))
    assert BOX.torsion_modulus == pytest.approx(2.0 * am * t)


@pytest.mark.parametrize(
    ("section", "expected"),
    [
        (CIRC, [("sectype", (7, "BEAM", "CSOLID", "S")), ("secdata", (4.0,))]),
        (TUBE, [("sectype", (7, "BEAM", "CTUBE", "S")), ("secdata", (9.5, 10.0))]),
        (
            BOX,
            [("sectype", (7, "BEAM", "HREC", "S")), ("secdata", (20.0, 30.0, 1.0, 1.0, 1.0, 1.0))],
        ),
    ],
)
def test_define_writes_the_mapdl_section(section, expected):
    fake = FakeMapdl()
    section.define(fake, 7, "S")
    assert fake.calls[:2] == expected


@pytest.mark.parametrize("cls", [RectSection, CircSection, TubeSection, BoxSection])
def test_stress_points_are_vectorised(cls):
    values = {"b": [20.0, 22.0, 24.0], "h": [30.0, 30.0, 30.0], "r": [8.0, 9.0, 10.0], "t": 1.0}
    section = cls.from_params({n: np.asarray(values[n]) for n in cls.param_names})
    pts = section.stress_points()
    assert pts.y.shape == pts.tz_vz.shape == (len(pts.y), 3)


# ── Generic evaluator against textbook formulas ──────────────────────────


@pytest.mark.parametrize("section", SHAPES)
def test_axial_force(section):
    assert _vm(section, {FX: 100.0}) == pytest.approx(100.0 / float(section.area))


@pytest.mark.parametrize("section", [RECT, BOX])
def test_biaxial_bending_adds_at_a_corner(section):
    n, my, mz = 50.0, 400.0, -300.0  # [N], [N·mm]
    expected = (
        n / section.area
        + my * (section.h / 2) / section.iy
        + abs(mz) * (section.b / 2) / section.iz
    )
    assert _vm(section, {FX: n, MY: my, MZ: mz}) == pytest.approx(float(expected))


@pytest.mark.parametrize("section", [CIRC, TUBE])
def test_circular_bending_uses_the_resultant_moment(section):
    my, mz = 400.0, -300.0  # resultant 500 N·mm, not aligned with a check point
    exact = 500.0 * float(section.r) / float(section.iy)
    assert _vm(section, {MY: my, MZ: mz}) == pytest.approx(exact, rel=4e-3)
    assert _vm(section, {MY: my, MZ: mz}) <= exact * (1 + 1e-12)


@pytest.mark.parametrize("section", SHAPES)
def test_pure_torsion(section):
    assert _vm(section, {TQ: 800.0}) == pytest.approx(
        SQRT3 * 800.0 / float(section.torsion_modulus)
    )


@pytest.mark.parametrize(
    ("section", "factor"), [(RECT, 1.5), (CIRC, 4.0 / 3.0), (TUBE, float(TUBE.shear_factor))]
)
@pytest.mark.parametrize("item", [SFZ, SFY])
def test_pure_shear(section, factor, item):
    assert _vm(section, {item: 60.0}) == pytest.approx(SQRT3 * factor * 60.0 / float(section.area))


def test_box_shear_is_carried_by_the_two_webs():
    vz = 60.0
    q = (20.0 * 30.0**2 - 18.0 * 28.0**2) / 8.0  # first moment of half the section [mm³]
    assert _vm(BOX, {SFZ: vz}) == pytest.approx(SQRT3 * vz * q / (float(BOX.iy) * 2.0))


@pytest.mark.parametrize("section", SHAPES)
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_torsion_and_shear_add_linearly_on_one_side(section, sign):
    tq, vz = 800.0, 60.0
    separate = _vm(section, {TQ: tq}) + _vm(section, {SFZ: vz})
    assert _vm(section, {TQ: sign * tq, SFZ: vz}) == pytest.approx(separate)


@pytest.mark.parametrize("section", SHAPES)
def test_node_j_forces_are_used(section):
    assert _vm(section, {MY + 13: 500.0}) == pytest.approx(_vm(section, {MY: 500.0}))
    assert _vm(section, {TQ + 13: 500.0}) == pytest.approx(_vm(section, {TQ: 500.0}))


# ── Rectangle: generic evaluator vs. the legacy one ──────────────────────


def _legacy(smisc: dict[int, float]) -> float:
    """Legacy evaluator fed with the BEAM188 stresses of the same internal forces."""
    b, h, a = float(RECT.b), float(RECT.h), float(RECT.area)
    full = dict(smisc)
    for first, stress0 in ((1, 31), (14, 36)):
        n, my, mz = (smisc.get(first + k, 0.0) for k in range(3))
        sy = -mz * (b / 2) / float(RECT.iz)  # SByT
        sz = my * (h / 2) / float(RECT.iy)  # SBzT
        full |= {
            stress0: n / a,
            stress0 + 1: sy,
            stress0 + 2: -sy,
            stress0 + 3: sz,
            stress0 + 4: -sz,
        }
    return _vm(RECT, full, RectBeamStressEvaluator())


def test_rectangle_coincides_with_legacy_without_torsion():
    rng = np.random.default_rng(0)
    for _ in range(50):
        n, my, mz, vz, vy = rng.uniform(-1.0, 1.0, 5) * [100.0, 500.0, 500.0, 50.0, 50.0]
        forces = {FX: n, MY: my, MZ: mz, SFZ: vz, SFY: vy}
        forces |= {k + 13: v for k, v in forces.items()}  # same forces at node J
        assert _vm(RECT, forces) == pytest.approx(_legacy(forces), rel=1e-12)


def test_rectangle_differences_from_legacy_with_torsion():
    torsion = {TQ: 800.0, TQ + 13: 800.0}
    # Pure torsion: identical (maximum at the mid-side of the long side).
    assert _vm(RECT, torsion) == pytest.approx(_legacy(torsion))
    # Bending about one axis + torsion: identical (same mid-side fibre).
    bending = torsion | {MY: 2000.0, MY + 13: 2000.0}
    assert _vm(RECT, bending) == pytest.approx(_legacy(bending))
    # Biaxial bending + torsion: the legacy evaluator put the torsion stress
    # at the corners, where it vanishes, and was therefore higher.
    biaxial = bending | {MZ: 1500.0, MZ + 13: 1500.0}
    assert _vm(RECT, biaxial) < _legacy(biaxial)
    # Shear + torsion: parallel at the mid-side, they add linearly; the legacy
    # evaluator combined them as orthogonal and was therefore lower.
    shear = torsion | {SFZ: 300.0, SFZ + 13: 300.0}
    assert _vm(RECT, shear) > _legacy(shear)


# ── A full optimisation with a non-rectangular section ───────────────────

LENGTH, LOAD, R_BOUNDS = 200.0, 20.0, (1.0, 6.0)  # [mm], [N], [mm]
STEEL = Material(E=2.0e5, nu=0.3, rho=7.85e-9, sigma_y=355.0)


def test_circular_cantilever_reaches_the_beam_of_uniform_strength(tmp_path):
    # Closed form: σ = 4 P (L − s) / (π r³) = σ_adm  →  r*(s) ∝ (L − s)^(1/3)
    path = BeamPath([[0.0, 0.0, 0.0], [LENGTH, 0.0, 0.0]], kind="polyline")
    model = StructuralModel(
        OptimizableBeam(discretize(path, 60), section_factory=CircSection.from_params),
        STEEL,
        bcs=[Clamp("start"), NodalForce("end", "FZ", -LOAD)],
    )
    problem = OptimizationProblem(
        StaticCantileverMapdl(str(tmp_path)),
        model,
        n_ctrl=8,
        bounds={"r": R_BOUNDS},
        safety_factor=1.5,
        f_react_max=5.0 * LOAD,
        reaction_nodes=["start"],
        interpolator="pchip",
    )
    result = optimize(problem, "slsqp", x0=problem.upper)

    s = np.linspace(0.0, LENGTH, 200_001)
    r_star = np.clip(np.cbrt(4.0 * LOAD * (LENGTH - s) / (np.pi * problem.sigma_adm)), *R_BOUNDS)
    area = np.pi * r_star**2
    mass_ref = STEEL.rho * np.sum(0.5 * (area[1:] + area[:-1]) * np.diff(s)) * 1e6  # [g]

    assert result.success
    assert result.max_violation <= 1e-4
    assert result.x[0] == pytest.approx(r_star[0], rel=0.02)
    assert 0.0 <= result.fun / mass_ref - 1.0 < 0.04
