"""
Phase 2: arc-length beam path and user-imposed discretisation with anchors.

Written before the implementation (see ROADMAP, Phase 2).
"""

from __future__ import annotations

import numpy as np
import pytest

from beam_opt.geometry import BeamPath, discretize

R = 50.0  # circle radius [mm]


def semicircle(n_points: int = 13) -> np.ndarray:
    """Definition points on a semicircle of radius R in a tilted plane [mm]."""
    t = np.linspace(0.0, np.pi, n_points)
    pts = np.column_stack([R * np.cos(t), R * np.sin(t), np.zeros_like(t)])
    c, s = np.cos(0.4), np.sin(0.4)  # rotate about X: a genuinely 3-D curve
    return pts @ np.array([[1, 0, 0], [0, c, -s], [0, s, c]]).T


# ---------------------------------------------------------------------
# BeamPath
# ---------------------------------------------------------------------


def test_spline_length_of_a_known_arc():
    assert BeamPath(semicircle()).length == pytest.approx(np.pi * R, rel=2e-5)


def test_polyline_length_is_the_chord_sum():
    pts = semicircle()
    chords = np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
    assert BeamPath(pts, kind="polyline").length == pytest.approx(chords)


def test_spline_passes_through_definition_points():
    pts = semicircle()
    path = BeamPath(pts)
    assert np.all(np.diff(path.s_knots) > 0)
    assert path.s_knots[0] == 0.0
    assert path.s_knots[-1] == pytest.approx(path.length)
    np.testing.assert_allclose(path.point_at(path.s_knots), pts, atol=1e-9)


@pytest.mark.parametrize("kind", ["spline", "polyline"])
def test_parameter_is_arc_length(kind):
    path = BeamPath(semicircle(), kind=kind)
    s = np.linspace(0.0, path.length, 200)
    # unit tangent everywhere ...
    np.testing.assert_allclose(np.linalg.norm(path.tangent_at(s), axis=-1), 1.0, atol=1e-6)
    # ... and small chords equal to the increment of s
    ds = 1e-3
    chord = np.linalg.norm(path.point_at(s[:-1] + ds) - path.point_at(s[:-1]), axis=-1)
    np.testing.assert_allclose(chord, ds, rtol=1e-4)


def test_unknown_kind():
    with pytest.raises(ValueError, match="kind"):
        BeamPath(semicircle(), kind="bezier")


# ---------------------------------------------------------------------
# discretize
# ---------------------------------------------------------------------


@pytest.mark.parametrize("n_elem", [7, 49, 196])
def test_requested_number_of_elements_and_monotone_s(n_elem):
    path = BeamPath(semicircle())
    mesh = discretize(path, n_elem)
    assert mesh.n_elem == n_elem
    assert mesh.s[0] == 0.0
    assert mesh.s[-1] == pytest.approx(path.length)
    assert np.all(np.diff(mesh.s) > 0)


@pytest.mark.parametrize("n_elem", [10, 49, 196])
def test_anchors_fall_exactly_on_nodes(n_elem):
    path = BeamPath(semicircle())
    anchors = {"support_1": 0.237 * path.length, "knot_5": path.s_knots[5], "tip": path.length}
    mesh = discretize(path, n_elem, anchors)
    for name, s in anchors.items():
        i = mesh.anchors[name]
        assert mesh.s[i] == pytest.approx(s, abs=1e-12)
        np.testing.assert_allclose(mesh.coords[i], path.point_at(s), atol=1e-12)
    assert mesh.anchors["tip"] == mesh.anchors["end"] == n_elem


def test_total_length_converges_with_n_elem():
    path = BeamPath(semicircle())
    errors = [path.length - discretize(path, n).lengths.sum() for n in (10, 20, 40, 80)]
    assert all(e > 0 for e in errors)  # chords are shorter than the arc
    ratios = np.array(errors[:-1]) / np.array(errors[1:])
    np.testing.assert_allclose(ratios, 4.0, rtol=0.05)  # second-order convergence


def test_elements_are_nearly_uniform_between_anchors():
    path = BeamPath(semicircle())
    mesh = discretize(path, 100, {"a": 0.31 * path.length, "b": 0.77 * path.length})
    ds = np.diff(mesh.s)
    assert ds.max() / ds.min() < 1.1


def test_coincident_anchors_share_a_node():
    path = BeamPath(semicircle())
    mesh = discretize(path, 20, {"a": 30.0, "b": 30.0})
    assert mesh.anchors["a"] == mesh.anchors["b"]


def test_polyline_corners_are_kept_as_anchors():
    corners = np.array([[0.0, 0, 0], [10.0, 0, 0], [10.0, 7.0, 0]])
    path = BeamPath(corners, kind="polyline")
    mesh = discretize(path, 17, {"corner": path.s_knots[1]})
    np.testing.assert_allclose(mesh.coords[mesh.anchors["corner"]], corners[1])
    assert mesh.lengths.sum() == pytest.approx(17.0)


@pytest.mark.parametrize(
    ("n_elem", "anchors", "match"),
    [
        (0, {}, "n_elem"),
        (2, {"a": 10.0, "b": 50.0}, "at least 3"),
        (10, {"a": -1.0}, "outside"),
        (10, {"end": 5.0}, "reserved"),
    ],
)
def test_invalid_requests(n_elem, anchors, match):
    with pytest.raises(ValueError, match=match):
        discretize(BeamPath(semicircle()), n_elem, anchors)
