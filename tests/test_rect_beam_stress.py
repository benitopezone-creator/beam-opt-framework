"""Unit tests for result extraction and the rectangular BEAM188 stress evaluator."""

from __future__ import annotations

import numpy as np
import pytest
from fakes import FakeMapdl

from beam_opt.geometry import BeamMesh
from beam_opt.model import OptimizableBeam
from beam_opt.post import RectBeamStressEvaluator, beam_smisc, reaction_resultant
from beam_opt.section_law import SectionLaw
from beam_opt.sections import RectSection

B, H = 2.0, 3.0  # [mm]
_SEC = RectSection(B, H)
A, WT = float(_SEC.area), float(_SEC.torsion_modulus)

BEAM = OptimizableBeam(BeamMesh.from_points([[0, 0, 0], [1, 0, 0]]))
LAW = SectionLaw([0.0, 1.0], {"b": [B, B], "h": [H, H]}, interpolator="linear")


def _evaluate(smisc: dict[int, list[float]]) -> float:
    vm = RectBeamStressEvaluator()(FakeMapdl(smisc=smisc), BEAM, LAW)
    return float(vm[0])


def test_sfz_is_combined_with_y_side_fibres():
    # SMISC 5 = SFZ: its Jourawski maximum lies on the ±Y faces (SByT).
    sfz, sbyt = 10.0, 50.0  # [N], [MPa]
    vm = _evaluate({5: [sfz], 32: [sbyt]})
    assert vm == pytest.approx(np.sqrt(sbyt**2 + 3.0 * (1.5 * sfz / A) ** 2))


def test_sfy_is_combined_with_z_side_fibres():
    # SMISC 6 = SFY: its Jourawski maximum lies on the ±Z faces (SBzT).
    sfy, sbzt = 10.0, 50.0  # [N], [MPa]
    vm = _evaluate({6: [sfy], 34: [sbzt]})
    assert vm == pytest.approx(np.sqrt(sbzt**2 + 3.0 * (1.5 * sfy / A) ** 2))


def test_node_j_uses_its_own_torque():
    # Torque only at node J (SMISC 17): must not be ignored.
    tq_j = 100.0  # [N·mm]
    vm = _evaluate({17: [tq_j]})
    assert vm == pytest.approx(np.sqrt(3.0) * tq_j / WT)


@pytest.mark.parametrize("sid", [18, 19])
def test_node_j_uses_its_own_shear(sid):
    v_j = 10.0  # [N]
    vm = _evaluate({sid: [v_j]})
    assert vm == pytest.approx(np.sqrt(3.0) * 1.5 * v_j / A)


def test_beam_smisc_is_vectorised():
    fake = FakeMapdl(smisc={4: [1.0, 2.0, 3.0, 99.0]})  # element 4 is not optimisable
    out = beam_smisc(fake, n_elem=3, smisc_ids=[4])
    np.testing.assert_allclose(out[4], [1.0, 2.0, 3.0])
    assert not any(name == "get_value" for name, _ in fake.calls)


def test_beam_smisc_rejects_short_arrays():
    with pytest.raises(RuntimeError, match="expected 3"):
        beam_smisc(FakeMapdl(smisc={4: [1.0]}), n_elem=3, smisc_ids=[4])


def test_reaction_resultant():
    fsum = {"FX": 3.0, "FY": 0.0, "FZ": 4.0}  # [N]
    fake = FakeMapdl(scalars={("FSUM", "ITEM", k): v for k, v in fsum.items()})
    assert reaction_resultant(fake, [7, 9]) == pytest.approx(5.0)
    assert fake.calls[:2] == [("nsel", ("S", "NODE", "", 7)), ("nsel", ("A", "NODE", "", 9))]


@pytest.mark.mapdl
def test_vget_extraction_matches_get_value_loop(mapdl_session):
    mapdl = mapdl_session
    mapdl.clear()
    mapdl.prep7()
    mapdl.et(1, "BEAM188")
    mapdl.mp("EX", 1, 2.0e5)
    mapdl.mp("PRXY", 1, 0.3)
    for i in range(1, 6):
        mapdl.sectype(i, "BEAM", "RECT")
        mapdl.secdata(2.0, 2.0 + 0.2 * i)
    for i in range(1, 7):
        mapdl.n(i, 20.0 * (i - 1), 0.0, 0.0)
    for i in range(1, 6):
        mapdl.secnum(i)
        mapdl.e(i, i + 1)
    mapdl.d(1, "ALL", 0)
    mapdl.f(6, "FY", -1.0)
    mapdl.f(6, "MX", 5.0)
    mapdl.finish()
    mapdl.run("/SOLU")
    mapdl.antype("STATIC")
    mapdl.solve()
    mapdl.finish()

    mapdl.post1()
    mapdl.set("LAST")
    out = beam_smisc(mapdl, 5, [4, 6, 32])
    for sid, values in out.items():
        loop = [mapdl.get_value("ELEM", e, "ETAB", f"SM{sid}") for e in range(1, 6)]
        np.testing.assert_allclose(values, loop)
    mapdl.finish()
