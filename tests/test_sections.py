"""Unit tests for cross-sections and the section law."""

from __future__ import annotations

import numpy as np
import pytest
from fakes import FakeMapdl

from beam_opt.section_law import SectionLaw
from beam_opt.sections import RectSection

S_CTRL = np.array([0.0, 10.0, 20.0, 30.0])  # [mm]
OVERSHOOT = np.array([2.0, 6.0, 6.0, 2.0])  # [mm]
S_FINE = np.linspace(0.0, 30.0, 61)  # [mm]


def test_rect_basic_properties():
    sec = RectSection(2.0, 3.0)
    assert sec.area == pytest.approx(6.0)
    assert sec.iy == pytest.approx(2.0 * 27.0 / 12.0)
    assert sec.iz == pytest.approx(3.0 * 8.0 / 12.0)


@pytest.mark.parametrize(
    ("ratio", "alpha", "beta"),
    # Timoshenko & Goodier: τ_max = T / (α a t²), J = β a t³ (a long side, t short side)
    [(1.0, 0.208, 0.141), (2.0, 0.246, 0.229), (10.0, 0.312, 0.312)],
)
@pytest.mark.parametrize("swap", [False, True])
def test_rect_torsion_matches_tabulated_values(ratio, alpha, beta, swap):
    t = 1.5  # [mm]
    a = ratio * t
    sec = RectSection(t, a) if swap else RectSection(a, t)
    assert sec.torsion_modulus == pytest.approx(alpha * a * t**2, rel=5e-3)
    assert sec.torsion_constant == pytest.approx(beta * a * t**3, rel=5e-3)


def test_rect_is_vectorised():
    sec = RectSection.from_params({"b": [1.0, 2.0], "h": [3.0, 4.0]})
    np.testing.assert_allclose(sec.area, [3.0, 8.0])


def test_rect_define_writes_mapdl_section():
    fake = FakeMapdl()
    RectSection(2.0, 3.0).define(fake, 7, "S7")
    assert fake.calls == [
        ("sectype", (7, "BEAM", "RECT", "S7")),
        ("secdata", (2.0, 3.0)),
        ("secoffset", ("CENT",)),
    ]


def test_natural_cubic_overshoots_and_clip_bounds_it():
    law = SectionLaw(S_CTRL, {"b": OVERSHOOT}, bounds={"b": (2.0, 6.0)})
    assert law.eval(S_FINE, clip=False)["b"].max() > 6.0
    assert law.eval(S_FINE)["b"].max() <= 6.0


@pytest.mark.parametrize("interpolator", ["pchip", "linear"])
def test_monotone_interpolators_do_not_overshoot(interpolator):
    law = SectionLaw(S_CTRL, {"b": OVERSHOOT}, interpolator=interpolator)
    b = law.eval(S_FINE, clip=False)["b"]
    assert b.max() == pytest.approx(6.0)
    np.testing.assert_allclose(law.eval(S_CTRL)["b"], OVERSHOOT)


def test_names_keep_design_vector_order():
    law = SectionLaw(S_CTRL, {"h": OVERSHOOT, "b": OVERSHOOT})
    assert law.names == ("h", "b")


def test_unknown_interpolator():
    with pytest.raises(ValueError, match="unknown interpolator"):
        SectionLaw(S_CTRL, {"b": OVERSHOOT}, interpolator="akima")
