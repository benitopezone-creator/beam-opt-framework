"""
Phase 1 regression: the refactored API reproduces the pre-refactor numerics.

``data/phase1_reference.json`` was produced by running the 0.1 package
(``BeamModel`` / ``BeamOptimizer``, before the Phase 1 refactor) on the
synthetic curved-frame scenario of ``data/scenario.py``.  The same case is
rebuilt here with the name-based API; the boundary conditions that the 0.1
model builder hard-wired are listed explicitly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import scenario as sc
from fakes import FakeMapdl, canonical_model

from beam_opt.geometry import BeamMesh
from beam_opt.materials import Material
from beam_opt.model import (
    FixedBeam,
    FixedShell,
    ImposedDisplacement,
    OptimizableBeam,
    RigidRegion,
    StructuralModel,
    Symmetry,
)
from beam_opt.optimization import OptimizationProblem
from beam_opt.sections import RectSection

DATA = Path(__file__).parent / "data"
REFERENCE = json.loads((DATA / "phase1_reference.json").read_text(encoding="utf-8"))

MATERIAL = Material(**sc.MATERIAL)
PLATE = [f"plate_{i}" for i in range(4)]
SUPPORTS = ["support_a", "support_b", "support_c"]


class LegacyTorsionRect(RectSection):
    """
    ``RectSection`` with the pre-0.2 torsion formulas.

    The reference stresses were computed with them; the corrected formulas
    are tested separately in ``test_sections.py``.
    """

    @property
    def torsion_constant(self) -> np.ndarray:
        t_min, t_max = np.minimum(self.b, self.h), np.maximum(self.b, self.h)
        beta = t_min / t_max
        return (t_min * t_max**3 / 3.0) * (1.0 - 0.63 * beta + 0.052 * beta**5)

    @property
    def torsion_modulus(self) -> np.ndarray:
        return self.torsion_constant / (np.minimum(self.b, self.h) / 2.0)


def frame_parts():
    """Fixed elements and BCs of the curved frame (names only)."""
    fixed = [
        FixedBeam("joint_a", "support_a", RectSection(*sc.STRUT)),
        FixedBeam("joint_b", "support_b", RectSection(*sc.STRUT)),
        FixedBeam("joint_c", "support_c", RectSection(*sc.STRUT)),
        FixedShell(PLATE, thickness=sc.PLATE_T),
    ]
    bcs = [
        Symmetry(SUPPORTS, normal="Z"),
        ImposedDisplacement(["support_a", "support_c"], "UY", 0.0),
        RigidRegion("plate_0", [*PLATE, "start", "end"]),
        ImposedDisplacement("plate_0", "UX", sc.UX),
        ImposedDisplacement("plate_0", "UZ", sc.UZ),
    ]
    return fixed, bcs


def curved_frame_model(section_factory=RectSection.from_params) -> StructuralModel:
    """One element per segment between definition points (as in 0.1)."""
    pts = sc.scenario_points()
    mesh = BeamMesh.from_points(pts.pop("arch"), anchors=sc.JOINTS)
    fixed, bcs = frame_parts()
    return StructuralModel(
        OptimizableBeam(mesh, section_factory=section_factory), MATERIAL, pts, fixed, bcs
    )


def make_problem(model: StructuralModel, mapdl) -> OptimizationProblem:
    return OptimizationProblem(
        mapdl,
        model,
        n_ctrl=4,
        bounds={"b": sc.BOUNDS, "h": sc.BOUNDS},
        safety_factor=sc.SAFETY_FACTOR,
        f_react_max=sc.F_REACT_MAX,
        reaction_nodes=[*PLATE, "start", "end"],
    )


def fake_mapdl(directory: str) -> FakeMapdl:
    scalars = {("FSUM", "ITEM", k): v for k, v in sc.FSUM.items()}
    scalars |= {("ACTIVE", "SOLU", "CNVG"): 1.0, ("ACTIVE", "SET", "TIME"): 1.0}
    return FakeMapdl(smisc=sc.SMISC, scalars=scalars, directory=directory)


@pytest.fixture(scope="module")
def evaluated(tmp_path_factory):
    fake = fake_mapdl(str(tmp_path_factory.mktemp("w")))
    problem = make_problem(curved_frame_model(LegacyTorsionRect.from_params), fake)
    x = np.asarray(sc.X_DESIGN)
    problem.evaluate(x)
    return problem, x, canonical_model(fake.calls)


def test_model_written_to_mapdl_is_unchanged(evaluated):
    _, _, model = evaluated
    assert model == REFERENCE["model"]


@pytest.mark.parametrize(
    "name",
    ["constraint_bounds", "constraint_section_field", "constraint_stress", "constraint_reaction"],
)
def test_constraints_are_unchanged(evaluated, name):
    problem, x, _ = evaluated
    np.testing.assert_allclose(getattr(problem, name)(x), REFERENCE[name], rtol=1e-12, atol=1e-14)


def test_masses_and_stress_are_unchanged(evaluated):
    problem, x, _ = evaluated
    assert problem.mass_fixed == pytest.approx(REFERENCE["mass_fixed"], rel=1e-12)
    assert problem.objective(x) == pytest.approx(REFERENCE["obj_mass"], rel=1e-12)
    np.testing.assert_allclose(problem.evaluate(x).sigma, REFERENCE["sigma"], rtol=1e-12)


def test_evaluation_is_cached(evaluated):
    problem, x, _ = evaluated
    n = len(problem.history)
    problem.objective(x)
    problem.constraint_stress(x)
    assert len(problem.history) == n
