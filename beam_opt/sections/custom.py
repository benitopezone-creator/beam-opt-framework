"""
beam_opt.sections.custom
========================
User-defined cross-sections.

A ``SectionShape`` is a fixed shape that the optimiser scales uniformly: the
only design parameter is ``scale``.  Two ways to obtain one:

* ``polygon_shape(vertices)`` — any solid shape given by its outline (a
  trapezoid, a profile with a notch, ...).  Area and inertias come from the
  outline; the Saint-Venant torsion problem is solved once, numerically.
* ``SectionShape.from_properties(...)`` — a section known only through its
  properties, e.g. an ``ASEC`` section of an APDL script.

>>> shape = polygon_shape([(0, 0), (6, 0), (4, 3), (2, 3)])       # trapezoid [mm]
>>> beam = OptimizableBeam(mesh, section_factory=shape)
>>> problem = OptimizationProblem(..., bounds={"scale": (0.5, 3.0)})

In MAPDL the section is written as an arbitrary section (``ASEC``).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial import Delaunay

from .base import Section, StressPoints


class SectionShape:
    """
    A cross-section shape of unit scale.

    Callable as a section factory: ``shape({"scale": s})`` returns the shape
    scaled by *s* (lengths × s, areas × s², inertias × s⁴).

    Parameters
    ----------
    area : float
        Area at unit scale [mm²].
    iy, iz, iyz : float
        Centroidal second moments and product of inertia at unit scale [mm⁴].
    torsion_constant : float
        Saint-Venant torsion constant at unit scale [mm⁴].
    points : StressPoints
        Check points at unit scale (arrays of shape ``(n_points,)``),
        positions relative to the centroid.
    shear_correction : tuple[float, float]
        Shear correction factors (κ_y, κ_z) [-].
    name : str
        Label, for reports only.
    """

    def __init__(
        self,
        area: float,
        iy: float,
        iz: float,
        iyz: float,
        torsion_constant: float,
        points: StressPoints,
        shear_correction: tuple[float, float] = (5.0 / 6.0, 5.0 / 6.0),
        name: str = "custom",
    ) -> None:
        self.area = float(area)
        self.iy = float(iy)
        self.iz = float(iz)
        self.iyz = float(iyz)
        self.torsion_constant = float(torsion_constant)
        self.points = points
        self.shear_correction = shear_correction
        self.name = name

    def __call__(self, params: Mapping[str, ArrayLike] | ArrayLike) -> ScaledSection:
        scale = params["scale"] if isinstance(params, Mapping) else params
        return ScaledSection(self, scale)

    #: Explicit alias of ``__call__``.
    section = __call__

    @classmethod
    def from_properties(
        cls,
        area: float,
        iy: float,
        iz: float,
        torsion_constant: float,
        y_max: float,
        z_max: float,
        torsion_modulus: float | None = None,
        iyz: float = 0.0,
        name: str = "properties",
    ) -> SectionShape:
        """
        Shape known only through its properties (e.g. a MAPDL ``ASEC`` section).

        The stresses are bounded conservatively on the box ``±y_max`` ×
        ``±z_max`` around the centroid: full bending at the four corners;
        at the mid-sides, the torsion stress ``T / Wt`` added to the shear
        stress ``1.5 V / A``.

        Parameters
        ----------
        area, iy, iz, torsion_constant, iyz
            Section properties [mm²], [mm⁴].
        y_max, z_max : float
            Largest distances of the outline from the centroid [mm].
        torsion_modulus : float | None
            ``Wt`` such that ``τ_max = T / Wt`` [mm³].  Defaults to
            ``J / max(y_max, z_max)``, exact for a circle only.
        """
        wt = torsion_modulus or torsion_constant / max(y_max, z_max)
        tors, shear = 1.0 / wt, 1.5 / area
        # (y, z, ty_t, tz_t, ty_vy, tz_vy, ty_vz, tz_vz)
        rows = [(sy * y_max, sz * z_max, 0, 0, 0, 0, 0, 0) for sy in (1, -1) for sz in (1, -1)]
        rows += [(s * y_max, 0, 0, s * tors, 0, 0, 0, shear) for s in (1, -1)]
        rows += [(0, s * z_max, -s * tors, 0, shear, 0, 0, 0) for s in (1, -1)]
        points = StressPoints.from_rows(rows, ())
        return cls(area, iy, iz, iyz, torsion_constant, points, name=name)


class ScaledSection(Section):
    """
    A ``SectionShape`` scaled uniformly.

    Parameters
    ----------
    shape : SectionShape
        Shape at unit scale.
    scale : float or array-like
        Scale factor [-].
    """

    param_names = ("scale",)

    def __init__(self, shape: SectionShape, scale: ArrayLike) -> None:
        self.shape = shape
        self.scale = np.asarray(scale, dtype=float)

    @classmethod
    def from_params(cls, params: Mapping[str, ArrayLike]) -> ScaledSection:
        raise TypeError("use the SectionShape itself as section factory: shape(params)")

    @property
    def area(self) -> np.ndarray:
        return self.shape.area * self.scale**2

    @property
    def iy(self) -> np.ndarray:
        return self.shape.iy * self.scale**4

    @property
    def iz(self) -> np.ndarray:
        return self.shape.iz * self.scale**4

    @property
    def iyz(self) -> np.ndarray:
        return self.shape.iyz * self.scale**4

    @property
    def torsion_constant(self) -> np.ndarray:
        return self.shape.torsion_constant * self.scale**4

    @property
    def torsion_modulus(self) -> np.ndarray:
        p = self.shape.points
        tau = np.max(np.hypot(p.ty_t, p.tz_t))  # per unit torque, unit scale
        return self.scale**3 / tau

    @property
    def shear_correction(self) -> tuple[float, float]:
        return self.shape.shear_correction

    def stress_points(self) -> StressPoints:
        """Check points of the shape: positions × s, torsion ÷ s³, shear ÷ s²."""
        p, s = self.shape.points, self.scale
        outer = np.multiply.outer
        return StressPoints(
            outer(p.y, s),
            outer(p.z, s),
            outer(p.ty_t, s**-3),
            outer(p.tz_t, s**-3),
            outer(p.ty_vy, s**-2),
            outer(p.tz_vy, s**-2),
            outer(p.ty_vz, s**-2),
            outer(p.tz_vz, s**-2),
        )

    def mapdl_section(self) -> tuple[str, tuple[float, ...]]:
        return self.asec()


# ---------------------------------------------------------------------------
# Polygonal outline
# ---------------------------------------------------------------------------


def _polygon_properties(v: np.ndarray) -> tuple[float, np.ndarray, float, float, float]:
    """Signed area, centroid and second moments (about the origin) of a polygon."""
    y0, z0 = v[:, 0], v[:, 1]
    y1, z1 = np.roll(y0, -1), np.roll(z0, -1)
    cross = y0 * z1 - y1 * z0
    area = 0.5 * cross.sum()
    centroid = np.array([((y0 + y1) * cross).sum(), ((z0 + z1) * cross).sum()]) / (6.0 * area)
    i_y = ((z0**2 + z0 * z1 + z1**2) * cross).sum() / 12.0  # ∫ z² dA
    i_z = ((y0**2 + y0 * y1 + y1**2) * cross).sum() / 12.0  # ∫ y² dA
    i_yz = ((y0 * z1 + 2.0 * y0 * z0 + 2.0 * y1 * z1 + y1 * z0) * cross).sum() / 24.0
    return area, centroid, i_y, i_z, i_yz


def _inside(points: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Even-odd test: which *points* lie inside polygon *v*."""
    y, z = points[:, 0][:, None], points[:, 1][:, None]
    y0, z0 = v[:, 0][None, :], v[:, 1][None, :]
    y1, z1 = np.roll(v[:, 0], -1)[None, :], np.roll(v[:, 1], -1)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        crosses = ((z0 > z) != (z1 > z)) & (y < (y1 - y0) * (z - z0) / (z1 - z0) + y0)
    return crosses.sum(axis=1) % 2 == 1


def _edge_distance(points: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Distance of *points* from the outline of polygon *v*."""
    a, b = v[None, :, :], np.roll(v, -1, axis=0)[None, :, :]
    p = points[:, None, :]
    ab = b - a
    t = np.clip(((p - a) * ab).sum(-1) / (ab * ab).sum(-1), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[..., None] * ab), axis=-1).min(axis=1)


def _chord_moment(v: np.ndarray, levels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Chord width and first moment of the area beyond each level.

    For a polygon *v* with columns (u, w): at each ``w = level``, the total
    chord length along u, and ``Q = ∫ width·w dw`` from the level to the top.
    """
    u0, w0 = v[:, 0], v[:, 1]
    u1, w1 = np.roll(u0, -1), np.roll(w0, -1)
    width = np.zeros_like(levels)
    for k, lev in enumerate(levels):
        hit = (w0 > lev) != (w1 > lev)
        u = np.sort(u0[hit] + (lev - w0[hit]) * (u1[hit] - u0[hit]) / (w1[hit] - w0[hit]))
        width[k] = (u[1::2] - u[0::2]).sum()
    f = width * levels
    steps = 0.5 * (f[1:] + f[:-1]) * np.diff(levels)
    moment = np.concatenate([np.cumsum(steps[::-1])[::-1], [0.0]])
    return width, moment


def polygon_shape(
    vertices: ArrayLike, resolution: int = 120, name: str = "polygon"
) -> SectionShape:
    """
    Solid cross-section of arbitrary polygonal outline.

    Parameters
    ----------
    vertices : array-like, shape (n, 2)
        Outline as (y, z) points in the section local axes [mm], in either
        direction, not self-intersecting.  The position of the origin does
        not matter: the beam axis passes through the centroid.
    resolution : int
        Number of mesh divisions along the largest dimension for the torsion
        problem.  With the default, J is within 0.2 % and the torsion stress
        is underestimated by about 1 % (first-order accurate: the error
        halves when the resolution doubles).
    name : str
        Label, for reports only.

    Returns
    -------
    SectionShape
        Shape at unit scale (the given dimensions), to be used as section
        factory with the design parameter ``scale``.

    Notes
    -----
    * Torsion: Prandtl stress function ``∇²φ = −2`` with ``φ = 0`` on the
      outline, linear triangular finite elements; ``J = 2 ∫ φ dA`` and
      ``τ = (T/J) |∇φ|``.  Valid for solid (simply connected) sections; the
      theoretical stress peak at a re-entrant corner is not resolved.
    * Shear: Jourawski formula ``V·Q / (I·chord)`` in the y and z directions
      (approximate for shapes without symmetry axes).
    * Check points: the vertices (extreme bending fibres) and the mid-points
      of the outline mesh.
    """
    v = np.asarray(vertices, dtype=float)
    if v.ndim != 2 or v.shape[1] != 2 or len(v) < 3:
        raise ValueError("vertices must be an (n, 2) array with n >= 3")
    area, centroid, i_y, i_z, i_yz = _polygon_properties(v)
    if area < 0.0:  # make it counter-clockwise
        v = v[::-1]
        area, centroid, i_y, i_z, i_yz = _polygon_properties(v)
    v = v - centroid
    area, _, i_y, i_z, i_yz = _polygon_properties(v)

    size = float(np.ptp(v, axis=0).max())
    h = size / int(resolution)

    # Mesh nodes: the outline, subdivided, then an interior grid
    boundary = []
    for a, b in zip(v, np.roll(v, -1, axis=0)):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / h)))
        boundary.append(a + (b - a) * np.linspace(0.0, 1.0, n, endpoint=False)[:, None])
    boundary = np.vstack(boundary)
    nb = len(boundary)
    gy, gz = np.meshgrid(
        np.arange(v[:, 0].min(), v[:, 0].max() + h, h),
        np.arange(v[:, 1].min(), v[:, 1].max() + h, h),
    )
    grid = np.column_stack([gy.ravel(), gz.ravel()])
    grid = grid[_inside(grid, v)]
    grid = grid[_edge_distance(grid, v) > 0.5 * h]
    nodes = np.vstack([boundary, grid])

    tri = Delaunay(nodes).simplices
    tri = tri[_inside(nodes[tri].mean(axis=1), v)]
    p0, p1, p2 = nodes[tri[:, 0]], nodes[tri[:, 1]], nodes[tri[:, 2]]
    area2 = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p2[:, 0] - p0[:, 0]) * (
        p1[:, 1] - p0[:, 1]
    )
    keep = np.abs(area2) > 1e-12 * h * h
    tri, p0, p1, p2, area2 = tri[keep], p0[keep], p1[keep], p2[keep], area2[keep]
    tri_area = 0.5 * np.abs(area2)

    # Linear triangles: gradients of the three shape functions, shape (n_tri, 3, 2)
    grad = (
        np.stack(
            [
                np.column_stack([p1[:, 1] - p2[:, 1], p2[:, 0] - p1[:, 0]]),
                np.column_stack([p2[:, 1] - p0[:, 1], p0[:, 0] - p2[:, 0]]),
                np.column_stack([p0[:, 1] - p1[:, 1], p1[:, 0] - p0[:, 0]]),
            ],
            axis=1,
        )
        / area2[:, None, None]
    )
    ke = tri_area[:, None, None] * np.einsum("eik,ejk->eij", grad, grad)
    rows = np.repeat(tri, 3, axis=1).ravel()
    cols = np.tile(tri, (1, 3)).ravel()
    n_nodes = len(nodes)
    stiff = coo_matrix((ke.ravel(), (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
    load = np.bincount(tri.ravel(), weights=np.repeat(2.0 * tri_area / 3.0, 3), minlength=n_nodes)

    phi = np.zeros(n_nodes)
    free = np.arange(nb, n_nodes)
    phi[free] = spsolve(stiff[free][:, free].tocsc(), load[free])
    torsion_constant = float(2.0 * (tri_area * phi[tri].mean(axis=1)).sum())

    # Shear stress per unit torque in each triangle: (∂φ/∂z, −∂φ/∂y) / J
    dphi = np.einsum("ei,eik->ek", phi[tri], grad)
    tau_y, tau_z = dphi[:, 1] / torsion_constant, -dphi[:, 0] / torsion_constant

    # Outline segments -> the triangle that owns each of them
    owner = {}
    for e, t in enumerate(tri):
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            if a < nb and b < nb:
                owner[(min(a, b), max(a, b))] = e
    mids, ty_t, tz_t = [], [], []
    for i in range(nb):
        j = (i + 1) % nb
        e = owner.get((min(i, j), max(i, j)))
        if e is not None:
            mids.append(0.5 * (boundary[i] + boundary[j]))
            ty_t.append(tau_y[e])
            tz_t.append(tau_z[e])
    mids = np.array(mids)

    # Jourawski shear per unit force, interpolated at the outline points
    lev_z = np.linspace(v[:, 1].min(), v[:, 1].max(), 801)[1:-1]
    wid, mom = _chord_moment(v, lev_z)
    tz_vz = np.interp(mids[:, 1], lev_z, mom / (i_y * wid))
    lev_y = np.linspace(v[:, 0].min(), v[:, 0].max(), 801)[1:-1]
    wid, mom = _chord_moment(v[:, ::-1], lev_y)
    ty_vy = np.interp(mids[:, 0], lev_y, mom / (i_z * wid))

    zeros_v, zeros_m = np.zeros(len(v)), np.zeros(len(mids))
    points = StressPoints(
        y=np.concatenate([v[:, 0], mids[:, 0]]),
        z=np.concatenate([v[:, 1], mids[:, 1]]),
        ty_t=np.concatenate([zeros_v, ty_t]),
        tz_t=np.concatenate([zeros_v, tz_t]),
        ty_vy=np.concatenate([zeros_v, ty_vy]),
        tz_vy=np.concatenate([zeros_v, zeros_m]),
        ty_vz=np.concatenate([zeros_v, zeros_m]),
        tz_vz=np.concatenate([zeros_v, tz_vz]),
    )
    return SectionShape(area, i_y, i_z, i_yz, torsion_constant, points, name=name)
