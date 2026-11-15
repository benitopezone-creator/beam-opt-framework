"""APDL import and export (no ANSYS involved: the scripts are interpreted as text)."""

from __future__ import annotations

import numpy as np
import pytest
from test_mesh_independence import curved_frame

from beam_opt.apdl import ApdlError, read_apdl, write_apdl
from beam_opt.model import Clamp, FixedBeam, FixedShell, ImposedDisplacement, NodalForce
from beam_opt.native import NativeSolver
from beam_opt.optimization import optimize
from beam_opt.section_law import SectionLaw
from beam_opt.sections import RectSection, ScaledSection

CANTILEVER = """
! Tapered cantilever written with parameters and a loop
/PREP7
L = 200            ! length [mm]
NE = 10
*AFUN,DEG
B0 = 2*SIN(30)*5   ! = 5 mm
ET,1,BEAM188
MP,EX,1,2.0E5 $ MP,PRXY,1,0.3 $ MP,DENS,1,7.85E-9
*DO,I,1,NE+1
  N,I,(I-1)*L/NE,0,0
*ENDDO
*DO,I,1,NE
  SECTYPE,I,BEAM,RECT
  SECDATA,B0,8-0.4*(I-1)
  SECNUM,I
  E,I,I+1
*ENDDO
ESEL,S,SEC,,1,NE
CM,OPT_BEAM,ELEM
ALLSEL
D,1,ALL,0
F,NE+1,FZ,-20
FINISH
/SOLU
ANTYPE,STATIC
NLGEOM,ON
SOLVE
"""


def test_parametric_script_is_interpreted():
    imp = read_apdl(CANTILEVER, sigma_y=355.0)
    mesh = imp.model.beam.mesh
    assert mesh.n_elem == 10 and mesh.s[-1] == pytest.approx(200.0)
    assert imp.param_names == ("b", "h")
    np.testing.assert_allclose(imp.initial["b"], 5.0)
    np.testing.assert_allclose(imp.initial["h"], 8.0 - 0.4 * np.arange(10))
    assert imp.elements == list(range(1, 11))
    assert imp.model.material.E == 2.0e5 and imp.model.material.rho == 7.85e-9
    kinds = [type(bc) for bc in imp.model.bcs]
    assert kinds == [Clamp, NodalForce]
    assert imp.model.bcs[0].nodes == ["n1"] and imp.model.bcs[1].node == "n11"
    assert any("NLGEOM" in note for note in imp.notes)


def test_imported_model_solves_like_the_script_describes():
    imp = read_apdl(CANTILEVER, sigma_y=355.0)
    res = NativeSolver().analyze(imp.model, imp.initial_law(), ["n1"])
    assert res.reaction == pytest.approx(20.0)
    assert abs(res.forces[2][0]) == pytest.approx(20.0 * 200.0)  # root moment
    # element 1: 5 x 8 mm  →  σ = 6 M / (b h²)
    problem = imp.problem(
        NativeSolver(),
        n_ctrl=4,
        bounds={"b": (2, 8), "h": (2, 12)},
        safety_factor=1.5,
        f_react_max=1e3,
    )
    assert problem.reaction_nodes == ["n1"]
    x0 = imp.x0(problem)
    np.testing.assert_allclose(x0[:4], 5.0)
    assert x0[4] > x0[7]  # the taper of the script is kept


def test_optimisation_of_an_imported_script():
    imp = read_apdl(CANTILEVER, sigma_y=355.0)
    problem = imp.problem(
        NativeSolver(),
        n_ctrl=5,
        bounds={"b": (4.0, 6.0), "h": (1.0, 10.0)},
        safety_factor=1.5,
        f_react_max=1e3,
        interpolator="pchip",
    )
    start = problem.objective(imp.x0(problem))
    result = optimize(problem, "slsqp", x0=imp.x0(problem))
    assert result.success and result.max_violation <= 1e-4
    assert result.fun < 0.8 * start


def test_parameters_can_be_overridden_from_python():
    text = "ET,1,188\nMP,EX,1,E0\nMP,DENS,1,1E-9\nN,1 $ N,2,LEN\nSECTYPE,1,BEAM,CSOLID\nSECDATA,3\n"
    text += "E,1,2\nCM,OPT_BEAM,ELEM\nD,1,ALL\n"
    imp = read_apdl(text, sigma_y=200.0, parameters={"E0": 7.0e4, "len": 50.0})
    assert imp.model.material.E == 7.0e4
    assert imp.model.beam.mesh.s[-1] == pytest.approx(50.0)
    assert imp.param_names == ("r",)


@pytest.mark.parametrize(
    ("script", "message"),
    [
        ("ET,1,188\nN,1\nN,2,1\nE,1,2\n", "no element component"),
        ("ET,1,188\nK,1,0,0,0\n", "line 2.*unsupported command K"),
        ("ET,1,188\nN,1\nN,2,1\nN,3,0,1\nE,1,2,3\nCM,OPT_BEAM,ELEM\n", "orientation nodes"),
        ("X = 2*UNDEFINED\n", "line 1.*undefined parameter"),
        ("*IF,1,EQ,1,THEN\n", r"line 1.*unsupported command \*IF"),
        (
            "ET,1,188\nN,1\nN,2,1\nN,3,2\nN,4,1,1\nE,1,2\nE,2,3\nE,2,4\nCM,OPT_BEAM,ELEM\n",
            "one open chain",
        ),
    ],
)
def test_unsupported_scripts_are_rejected_with_the_line(script, message):
    with pytest.raises(ApdlError, match=message):
        read_apdl(script, sigma_y=355.0)


def test_asec_section_and_shells_are_imported():
    text = """
    ET,1,BEAM188 $ ET,2,SHELL181
    MP,EX,1,7E4 $ MP,NUXY,1,0.33 $ MP,DENS,1,2.7E-9
    N,1,0,0,0 $ N,2,50,0,0 $ N,3,100,0,0 $ N,4,100,0,-40
    N,11,100,-5,0 $ N,12,110,-5,0 $ N,13,110,5,0 $ N,14,100,5,0
    SECTYPE,1,BEAM,RECT $ SECDATA,4,6
    SECTYPE,2,BEAM,ASEC $ SECDATA,24,72,0,32,0,75.1,0,0,0,0,6,4
    SECTYPE,3,SHELL $ SECDATA,1.5
    SECNUM,1 $ E,1,2 $ E,2,3
    CM,OPT_BEAM,ELEM
    SECNUM,2 $ E,3,4
    TYPE,2 $ SECNUM,3 $ E,11,12,13,14
    D,1,ALL $ D,4,UZ,0 $ D,4,UX,1.5
    CERIG,3,11,ALL $ CERIG,3,12,ALL $ CERIG,3,13,ALL $ CERIG,3,14,ALL
    """
    imp = read_apdl(text, sigma_y=250.0)
    strut, plate = imp.model.fixed
    assert isinstance(strut, FixedBeam) and isinstance(strut.section, ScaledSection)
    assert strut.section.area == pytest.approx(24.0)
    assert isinstance(plate, FixedShell) and plate.thickness == 1.5
    imposed = [bc for bc in imp.model.bcs if isinstance(bc, ImposedDisplacement)]
    assert {(bc.dof, bc.value) for bc in imposed} == {("UZ", 0.0), ("UX", 1.5)}
    res = NativeSolver().analyze(imp.model, imp.initial_law(), ["n1"])  # plate is rigid: accepted
    assert res.displacements[res.node_index["n4"]][0] == pytest.approx(1.5)


# ── Export ───────────────────────────────────────────────────────────────


def test_written_script_contains_the_model(tmp_path):
    imp = read_apdl(CANTILEVER, sigma_y=355.0)
    path = tmp_path / "out" / "design.inp"
    text = write_apdl(imp.model, imp.initial_law(), str(path), title="tapered")
    assert path.read_text(encoding="utf-8") == text
    lines = text.splitlines()
    assert "/TITLE,tapered" in lines and "ET,1,BEAM188" in lines
    assert "SECTYPE,1,BEAM,RECT,OPT1" in lines and "SECDATA,5,8" in lines
    assert "CM,OPT_BEAM,ELEM" in lines and "SOLVE" in lines
    assert "F,11,FZ,-20" in lines and "D,1,ALL,0" in lines


def test_round_trip_of_a_curved_frame_with_struts_shell_and_symmetry():
    # Export the framework model, read the script back, solve both.
    model = curved_frame(40)
    s_end = model.beam.mesh.s[-1]
    law = SectionLaw(
        [0.0, s_end / 2, s_end], {"b": [3.0, 5.0, 4.0], "h": [5.0, 3.5, 4.5]}, interpolator="linear"
    )
    imp = read_apdl(write_apdl(model, law), sigma_y=model.material.sigma_y)

    assert imp.model.beam.mesh.n_elem == 40
    assert len(imp.model.fixed) == len(model.fixed)
    np.testing.assert_allclose(imp.initial["b"], law.eval(model.beam.mesh.s_elem)["b"], rtol=1e-10)

    nodes = ["plate_0", "plate_1", "plate_2", "plate_3", "start", "end"]
    ref = NativeSolver().analyze(model, law, nodes)
    ids = {name: f"n{i + 1}" for name, i in ref.node_index.items()}
    back = NativeSolver().analyze(imp.model, imp.initial_law(), [ids[n] for n in nodes])
    assert back.reaction == pytest.approx(ref.reaction, rel=1e-8)
    for item in (1, 2, 3, 4, 5, 6):
        np.testing.assert_allclose(back.forces[item], ref.forces[item], rtol=1e-7, atol=1e-7)


def test_exported_optimum_keeps_its_stresses(tmp_path):
    imp = read_apdl(CANTILEVER, sigma_y=355.0)
    problem = imp.problem(
        NativeSolver(), n_ctrl=5, bounds={"b": (4.0, 6.0), "h": (1.0, 10.0)},
        safety_factor=1.5, f_react_max=1e3, interpolator="pchip",
    )  # fmt: skip
    result = optimize(problem, "slsqp", x0=imp.x0(problem))
    text = write_apdl(imp.model, problem.law_from_x(result.x))
    again = read_apdl(text, sigma_y=355.0)
    check = again.problem(
        NativeSolver(), n_ctrl=5, bounds={"b": (4.0, 6.0), "h": (1.0, 10.0)},
        safety_factor=1.5, f_react_max=1e3, interpolator="linear",
    )  # fmt: skip
    res = NativeSolver().analyze(again.model, again.initial_law(), ["n1"])
    sigma = check.evaluator.from_forces(res.forces, again.model.beam, again.initial_law())
    np.testing.assert_allclose(sigma, problem.evaluate(result.x).sigma, rtol=1e-8)
    assert isinstance(again.section_factory({"b": 1.0, "h": 1.0}), RectSection)
