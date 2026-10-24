"""Unit tests for beam_opt.model (components, BCs, builder) on a FakeMapdl."""

from __future__ import annotations

import numpy as np
import pytest
from fakes import FakeMapdl

from beam_opt.geometry import BeamMesh
from beam_opt.materials import Material
from beam_opt.model import (
    Clamp,
    FixedBeam,
    FixedShell,
    ImposedDisplacement,
    ModelBuilder,
    NodalForce,
    OptimizableBeam,
    RigidRegion,
    StructuralModel,
    Symmetry,
)
from beam_opt.section_law import SectionLaw
from beam_opt.sections import RectSection

STEEL = Material(E=2.0e5, nu=0.3, rho=7.85e-9, sigma_y=355.0)


def _cantilever(n_elem: int = 2, length: float = 100.0) -> StructuralModel:
    pts = np.column_stack([np.linspace(0, length, n_elem + 1), np.zeros((2, n_elem + 1)).T])
    beam = OptimizableBeam(BeamMesh.from_points(pts, anchors={"root": 0, "tip": -1}))
    return StructuralModel(beam, STEEL, bcs=[Clamp("root"), NodalForce("tip", "FY", -1.0)])


def _constant_law(length: float = 100.0, b: float = 2.0, h: float = 3.0) -> SectionLaw:
    return SectionLaw([0.0, length], {"b": [b, b], "h": [h, h]}, interpolator="linear")


def test_build_sequence_minimal_cantilever():
    fake = FakeMapdl()
    ids = ModelBuilder(fake, jobname="job").build(_cantilever(), _constant_law())
    assert ids == {"start": 1, "end": 3, "root": 1, "tip": 3}
    assert fake.calls == [
        ("clear", ()),
        ("filname", ("job",)),
        ("prep7", ()),
        ("et", (1, "BEAM188")),
        ("et", (2, "SHELL181")),
        ("mp", ("EX", 1, 2.0e5)),
        ("mp", ("PRXY", 1, 0.3)),
        ("mp", ("DENS", 1, 7.85e-9)),
        ("n", (1, 0.0, 0.0, 0.0)),
        ("n", (2, 50.0, 0.0, 0.0)),
        ("n", (3, 100.0, 0.0, 0.0)),
        ("sectype", (1, "BEAM", "RECT", "OPT1")),
        ("secdata", (2.0, 3.0)),
        ("secoffset", ("CENT",)),
        ("sectype", (2, "BEAM", "RECT", "OPT2")),
        ("secdata", (2.0, 3.0)),
        ("secoffset", ("CENT",)),
        ("type", (1,)),
        ("mat", (1,)),
        ("secnum", (1,)),
        ("e", (1, 2)),
        ("secnum", (2,)),
        ("e", (2, 3)),
        ("d", (1, "ALL", 0)),
        ("f", (3, "FY", -1.0)),
        ("allsel", ("ALL",)),
        ("finish", ()),
    ]


@pytest.mark.parametrize("n_elem", [10, 50, 400])
def test_same_configuration_runs_with_any_discretisation(n_elem):
    fake = FakeMapdl()
    ModelBuilder(fake).build(_cantilever(n_elem), _constant_law())
    assert sum(name == "e" for name, _ in fake.calls) == n_elem


def test_fixed_sections_never_collide_with_optimisable_ones():
    model = _cantilever(n_elem=400)
    model.points["support"] = np.array([0.0, -10.0, 0.0])
    model.fixed.append(FixedBeam("root", "support", RectSection(1.0, 1.0)))
    fake = FakeMapdl()
    ModelBuilder(fake).build(model, _constant_law())
    sectypes = [args[0] for name, args in fake.calls if name == "sectype"]
    assert sectypes == list(range(1, 402))


def test_bcs_write_expected_commands():
    ids = {"a": 1, "b": 2, "c": 3}
    fake = FakeMapdl()
    Symmetry(["a", "b"], normal="z").apply(fake, ids)
    ImposedDisplacement(["a", "c"], "uy").apply(fake, ids)
    RigidRegion("a", ["a", "b", "c"]).apply(fake, ids)
    assert fake.calls == [
        ("nsel", ("S", "NODE", "", 1)),
        ("dsym", ("SYMM", "Z", 0)),
        ("nsel", ("S", "NODE", "", 2)),
        ("dsym", ("SYMM", "Z", 0)),
        ("allsel", ("ALL",)),
        ("d", (1, "UY", 0.0)),
        ("d", (3, "UY", 0.0)),
        ("cerig", (1, 2, "ALL")),
        ("cerig", (1, 3, "ALL")),
    ]


def test_unknown_node_name_is_rejected_early():
    beam = OptimizableBeam(BeamMesh.from_points([[0, 0, 0], [1, 0, 0]]))
    with pytest.raises(ValueError, match="unknown node names: \\['tipp'\\]"):
        StructuralModel(beam, STEEL, bcs=[Clamp("tipp")])


def test_point_name_cannot_shadow_an_anchor():
    beam = OptimizableBeam(BeamMesh.from_points([[0, 0, 0], [1, 0, 0]]))
    with pytest.raises(ValueError, match="clash"):
        StructuralModel(beam, STEEL, points={"end": [2, 0, 0]})


def test_beam_and_fixed_mass():
    model = _cantilever()
    model.points |= {"p": np.array([0.0, 0.0, 10.0])}
    model.fixed += [
        FixedBeam("root", "p", RectSection(1.0, 2.0)),
        FixedShell(["root", "tip", "tip", "p"], thickness=0.5),  # triangle
    ]
    rho = STEEL.rho
    assert model.beam.mass(_constant_law(), rho) == pytest.approx(rho * 100.0 * 6.0)
    expected_fixed = rho * (10.0 * 2.0 + 0.5 * 100.0 * 10.0 * 0.5)
    assert model.fixed_mass() == pytest.approx(expected_fixed)
