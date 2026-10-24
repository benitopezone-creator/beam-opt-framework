"""Unit tests for beam_opt.geometry."""

from __future__ import annotations

import numpy as np
import pytest

from beam_opt.geometry import BeamMesh, BeamPath

SQUARE = [[0, 0, 0], [3, 0, 0], [3, 4, 0], [3, 4, 12]]  # segment lengths 3, 4, 12 [mm]


def test_polyline_abscissa_and_point_at():
    path = BeamPath(SQUARE, kind="polyline")
    np.testing.assert_allclose(path.s_knots, [0, 3, 7, 19])
    assert path.length == pytest.approx(19.0)
    np.testing.assert_allclose(path.point_at(5.0), [3, 2, 0])


def test_path_rejects_duplicate_points():
    with pytest.raises(ValueError):
        BeamPath([[0, 0, 0], [0, 0, 0]])


def test_mesh_from_points():
    mesh = BeamMesh.from_points(SQUARE, anchors={"tip": -1, "corner": 2})
    assert (mesh.n_nodes, mesh.n_elem) == (4, 3)
    np.testing.assert_allclose(mesh.lengths, [3, 4, 12])
    np.testing.assert_allclose(mesh.s_elem, [1.5, 5.0, 13.0])
    assert mesh.anchors == {"start": 0, "end": 3, "tip": 3, "corner": 2}


def test_mesh_rejects_out_of_range_anchor():
    with pytest.raises(ValueError, match="anchor 'x'"):
        BeamMesh.from_points(SQUARE, anchors={"x": 4})
