"""
beam_opt.geometry.mesh
======================
Discretised beam: node coordinates, element connectivity and named anchors.

Nodes are referred to by *anchor* names (e.g. ``"tip"``), never by MAPDL
numbers: the numbering is decided by ``ModelBuilder``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike

from .curve import BeamPath

RESERVED_ANCHORS = ("start", "end")


@dataclass
class BeamMesh:
    """
    Chain of two-node beam elements.

    Parameters
    ----------
    coords : ndarray, shape (n_elem + 1, 3)
        Node coordinates in chain order [mm].
    anchors : dict[str, int]
        Named nodes as 0-based indices into ``coords``.  ``"start"`` and
        ``"end"`` are always defined.
    s : ndarray | None, shape (n_elem + 1,)
        Curvilinear abscissa of the nodes along the centre line [mm].
        Defaults to the cumulative chord length.

    Attributes
    ----------
    lengths : ndarray, shape (n_elem,)
        Element (chord) lengths [mm].
    s_elem : ndarray, shape (n_elem,)
        Abscissa of the element mid-points [mm].
    """

    coords: np.ndarray
    anchors: dict[str, int] = field(default_factory=dict)
    s: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.coords = np.asarray(self.coords, dtype=float)
        n = len(self.coords)
        resolved = {"start": 0, "end": n - 1}
        for name, idx in self.anchors.items():
            i = int(idx) + n if int(idx) < 0 else int(idx)
            if not 0 <= i < n:
                raise ValueError(f"anchor {name!r}: index {idx} out of range")
            resolved[name] = i
        self.anchors = resolved
        self.lengths = np.linalg.norm(np.diff(self.coords, axis=0), axis=1)
        if self.s is None:
            self.s = np.concatenate(([0.0], np.cumsum(self.lengths)))
        else:
            self.s = np.asarray(self.s, dtype=float)
            if self.s.shape != (n,) or np.any(np.diff(self.s) <= 0.0):
                raise ValueError("s must have one strictly increasing value per node")
        self.s_elem = 0.5 * (self.s[:-1] + self.s[1:])

    @classmethod
    def from_points(cls, points: ArrayLike, anchors: Mapping[str, int] | None = None) -> BeamMesh:
        """
        One element per segment between consecutive definition points.

        Parameters
        ----------
        points : array-like, shape (N, 3)
            Ordered definition points [mm].
        anchors : mapping[str, int] | None
            ``{name: point index}``; negative indices count from the end.
        """
        path = BeamPath(points, kind="polyline")
        return cls(path.points, dict(anchors or {}))

    @property
    def n_nodes(self) -> int:
        """Number of nodes."""
        return len(self.coords)

    @property
    def n_elem(self) -> int:
        """Number of elements."""
        return len(self.coords) - 1


def discretize(
    path: BeamPath,
    n_elem: int,
    anchors: Mapping[str, float] | None = None,
) -> BeamMesh:
    """
    Resample *path* into *n_elem* elements with a node exactly at every anchor.

    The ends and the anchors are hard points that split the centre line into
    intervals; each interval receives a number of equal elements (in *s*) so
    that the largest element is as small as possible.  The section law and
    the BCs only depend on *s* and on anchor names, so the same configuration
    runs with any ``n_elem``.

    Parameters
    ----------
    path : BeamPath
        Centre line.
    n_elem : int
        Number of elements; at least the number of intervals between hard points.
    anchors : mapping[str, float] | None
        ``{name: s}`` with *s* in ``[0, path.length]`` [mm]; anchors closer
        than ``1e-9 * length`` share one node.  ``"start"`` and ``"end"`` are
        reserved (always defined).

    Returns
    -------
    BeamMesh
        Mesh with ``s`` equal to the arc length along *path*.

    Raises
    ------
    ValueError
        For ``n_elem < 1``, reserved or out-of-range anchors, or too few
        elements for the anchors.
    """
    anchors = dict(anchors or {})
    n_elem = int(n_elem)
    if n_elem < 1:
        raise ValueError("n_elem must be >= 1")
    reserved = sorted(set(anchors) & set(RESERVED_ANCHORS))
    if reserved:
        raise ValueError(f"anchor names {reserved} are reserved")

    length = path.length
    tol = 1e-9 * length
    for name, s in anchors.items():
        if not -tol <= s <= length + tol:
            raise ValueError(f"anchor {name!r} at s={s:g} mm is outside [0, {length:g}]")

    # Hard points: ends + anchors, merging values closer than tol
    hard = [0.0]
    for s in sorted(min(max(float(v), 0.0), length) for v in anchors.values()):
        if s - hard[-1] > tol:
            hard.append(s)
    if length - hard[-1] > tol:
        hard.append(length)
    else:
        hard[-1] = length
    hard = np.array(hard)
    spans = np.diff(hard)
    if n_elem < len(spans):
        raise ValueError(f"n_elem={n_elem}: the anchors need at least {len(spans)} elements")

    # Greedy allocation minimising the largest element size
    counts = np.ones(len(spans), dtype=int)
    for _ in range(n_elem - len(spans)):
        counts[np.argmax(spans / counts)] += 1

    s_nodes = np.concatenate(
        [np.linspace(a, b, c, endpoint=False) for a, b, c in zip(hard[:-1], hard[1:], counts)]
        + [hard[-1:]]
    )
    first_node = np.concatenate(([0], np.cumsum(counts)))  # node index of each hard point
    idx = {name: int(first_node[np.argmin(np.abs(hard - s))]) for name, s in anchors.items()}
    return BeamMesh(path.point_at(s_nodes), anchors=idx, s=s_nodes)
