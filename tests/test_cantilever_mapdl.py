"""
Integration test: BEAM188 cantilever with a tip load vs. Euler-Bernoulli.

Reference: tip deflection δ = P L³ / (3 E I).  The beam is slender
(L/h = 50) so the Timoshenko shear contribution of BEAM188 is ~1e-3 of δ,
well inside the 1 % tolerance together with the discretisation error of
10 linear elements.
"""

from __future__ import annotations

import numpy as np
import pytest

from beam_opt.geometry import BeamMesh, BeamPath, discretize
from beam_opt.materials import Material
from beam_opt.model import Clamp, ModelBuilder, NodalForce, OptimizableBeam, StructuralModel
from beam_opt.section_law import SectionLaw
from beam_opt.solver import SolverSettings, check_convergence, solve

L = 100.0  # beam length [mm]
B = 2.0  # section width [mm]
H = 2.0  # section height [mm]
E = 2.0e5  # Young's modulus [MPa]
NU = 0.3  # Poisson's ratio [-]
P = 1.0  # tip load [N]
N_ELEM = 10
DELTA_REF = P * L**3 / (3.0 * E * (B * H**3 / 12.0))  # Euler-Bernoulli tip deflection [mm]


@pytest.mark.mapdl
def test_cantilever_tip_deflection_matches_euler_bernoulli(mapdl_session):
    mapdl = mapdl_session
    mapdl.clear()
    mapdl.prep7()
    mapdl.et(1, "BEAM188")
    mapdl.mp("EX", 1, E)
    mapdl.mp("PRXY", 1, NU)
    mapdl.sectype(1, "BEAM", "RECT")
    mapdl.secdata(B, H)

    for i, x in enumerate(np.linspace(0.0, L, N_ELEM + 1), 1):
        mapdl.n(i, x, 0.0, 0.0)
    for i in range(1, N_ELEM + 1):
        mapdl.e(i, i + 1)
    tip = N_ELEM + 1

    mapdl.d(1, "ALL", 0.0)
    mapdl.f(tip, "FY", -P)
    mapdl.finish()

    mapdl.run("/SOLU")
    mapdl.antype("STATIC")
    mapdl.nlgeom("OFF")
    mapdl.solve()
    check_convergence(mapdl)

    mapdl.post1()
    mapdl.set("LAST")
    uy_tip = mapdl.get_value("NODE", tip, "U", "Y")
    mapdl.finish()

    assert -uy_tip == pytest.approx(DELTA_REF, rel=1e-2)


@pytest.mark.mapdl
@pytest.mark.parametrize("n_elem", [7, 25, 101])
def test_mid_span_load_on_an_anchor_with_odd_n_elem(mapdl_session, n_elem):
    # Odd n_elem: only the anchor guarantees a node at L/2 for the load.
    path = BeamPath([[0.0, 0.0, 0.0], [L / 3, 0.0, 0.0], [L, 0.0, 0.0]])
    mesh = discretize(path, n_elem, {"mid": L / 2})
    model = StructuralModel(
        OptimizableBeam(mesh),
        Material(E=E, nu=NU, rho=7.85e-9, sigma_y=355.0),
        bcs=[Clamp("start"), NodalForce("mid", "FY", -P)],
    )
    law = SectionLaw([0.0, L], {"b": [B, B], "h": [H, H]}, interpolator="linear")
    settings = SolverSettings(jobname="midload", nlgeom=False)

    mapdl = mapdl_session
    ids = ModelBuilder(mapdl, jobname=settings.jobname).build(model, law)
    solve(mapdl, settings)
    mapdl.post1()
    mapdl.set("LAST")
    uy_tip = mapdl.get_value("NODE", ids["end"], "U", "Y")
    mapdl.finish()

    a = L / 2
    delta = P * a**2 * (3.0 * L - a) / (6.0 * E * (B * H**3 / 12.0))  # [mm]
    assert -uy_tip == pytest.approx(delta, rel=1e-2)


@pytest.mark.mapdl
def test_cantilever_built_with_the_framework(mapdl_session):
    pts = np.column_stack([np.linspace(0.0, L, N_ELEM + 1), np.zeros((2, N_ELEM + 1)).T])
    model = StructuralModel(
        OptimizableBeam(BeamMesh.from_points(pts, anchors={"root": 0, "tip": -1})),
        Material(E=E, nu=NU, rho=7.85e-9, sigma_y=355.0),
        bcs=[Clamp("root"), NodalForce("tip", "FY", -P)],
    )
    law = SectionLaw([0.0, L], {"b": [B, B], "h": [H, H]}, interpolator="linear")
    settings = SolverSettings(jobname="cantilever", nlgeom=False)

    mapdl = mapdl_session
    ids = ModelBuilder(mapdl, jobname=settings.jobname).build(model, law)
    solve(mapdl, settings)

    mapdl.post1()
    mapdl.set("LAST")
    uy_tip = mapdl.get_value("NODE", ids["tip"], "U", "Y")
    mapdl.finish()

    assert -uy_tip == pytest.approx(DELTA_REF, rel=1e-2)
