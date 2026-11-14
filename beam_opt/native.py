"""
beam_opt.native
===============
Built-in linear beam solver: the framework without ANSYS.

``NativeSolver`` takes the place of the MAPDL session:

>>> problem = OptimizationProblem(NativeSolver(), model, ...)
>>> result = optimize(problem, "slsqp")

It analyses the same ``StructuralModel`` with two-node, six-degrees-of-freedom
3-D beam elements (Timoshenko, exact for nodal loads) and returns the element
internal forces in the BEAM188 convention, so the stress evaluation is shared
with the MAPDL path.

Scope and limits
----------------
* Linear static analysis only: no large displacements (``NLGEOM``).
  ``SolverSettings`` is ignored.
* Element axes as BEAM188 without an orientation node: local x from node I
  to node J, local y parallel to the global XY plane (global Y for an
  element parallel to global Z).
* Boundary conditions: ``Clamp``, ``Symmetry``, ``ImposedDisplacement``,
  ``NodalForce`` and ``RigidRegion`` with all degrees of freedom.
* ``FixedBeam`` elements are supported.  A ``FixedShell`` is accepted only
  when all its nodes belong to one rigid region (it then carries no strain
  and adds no stiffness); deformable shells need MAPDL.
* Sections are constant per element (the section law at the element
  mid-point), as in the MAPDL model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve

from .exceptions import SolutionNotConvergedError
from .model.bcs import Clamp, ImposedDisplacement, NodalForce, RigidRegion, Symmetry
from .model.builder import ModelBuilder
from .model.components import FixedBeam, FixedShell, StructuralModel

_DOF = {"UX": 0, "UY": 1, "UZ": 2, "ROTX": 3, "ROTY": 4, "ROTZ": 5}
_LOAD = {"FX": 0, "FY": 1, "FZ": 2, "MX": 3, "MY": 4, "MZ": 5}
# DSYM,SYMM: out-of-plane translation and in-plane rotations are fixed
_SYMMETRY = {"X": (0, 4, 5), "Y": (1, 3, 5), "Z": (2, 3, 4)}
# BEAM188 SMISC items at node I (FX, MY, MZ, TQ, SFZ, SFY) -> local force component
_SMISC_COMPONENT = (0, 4, 5, 3, 2, 1)


@dataclass
class NativeResult:
    """
    Result of one analysis.

    Attributes
    ----------
    displacements : ndarray, shape (n_nodes, 6)
        ``UX, UY, UZ`` [mm] and ``ROTX, ROTY, ROTZ`` [rad] of every node, in
        the numbering of ``node_index``.
    node_index : dict[str, int]
        Row of every named node in ``displacements``; the nodes of the
        optimisable beam occupy the first rows, in chain order.
    forces : dict[int, ndarray]
        Internal forces of the optimisable elements keyed by BEAM188 SMISC
        item (1–6 at node I, 14–19 at node J: FX, MY, MZ, TQ, SFZ, SFY)
        [N], [N·mm].
    reaction : float
        Resultant force on the reaction nodes [N].
    """

    displacements: np.ndarray
    node_index: dict[str, int]
    forces: dict[int, np.ndarray]
    reaction: float


def element_axes(p_i: np.ndarray, p_j: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Lengths and local axes of beam elements (BEAM188 default orientation).

    Parameters
    ----------
    p_i, p_j : ndarray, shape (m, 3)
        End-node coordinates [mm].

    Returns
    -------
    length : ndarray, shape (m,)
        Element lengths [mm].
    axes : ndarray, shape (m, 3, 3)
        Rows: local x, y, z unit vectors in global components.
    """
    d = p_j - p_i
    length = np.linalg.norm(d, axis=1)
    x = d / length[:, None]
    y = np.cross([0.0, 0.0, 1.0], x)
    norm = np.linalg.norm(y, axis=1)
    vertical = norm < 1e-4  # parallel to global Z
    y[vertical] = [0.0, 1.0, 0.0]
    norm[vertical] = 1.0
    y /= norm[:, None]
    z = np.cross(x, y)
    return length, np.stack([x, y, z], axis=1)


def local_stiffness(ea, gj, eiy, eiz, length, phi_y, phi_z) -> np.ndarray:
    """
    12×12 local stiffness matrices of Timoshenko beam elements.

    Degrees of freedom: ``u, v, w, θx, θy, θz`` at node I, then at node J.

    Parameters
    ----------
    ea, gj, eiy, eiz : ndarray, shape (m,)
        Axial, torsional and bending stiffnesses [N], [N·mm²].
    length : ndarray, shape (m,)
        Element lengths [mm].
    phi_y, phi_z : ndarray, shape (m,)
        Shear-flexibility parameters ``12 EI / (κ G A L²)`` for bending in
        the local x-y plane (uses ``eiz``) and x-z plane (uses ``eiy``) [-];
        zero gives the Euler-Bernoulli element.

    Returns
    -------
    ndarray, shape (m, 12, 12)
    """
    m = len(length)
    k = np.zeros((m, 12, 12))
    k[:, 0, 0] = k[:, 6, 6] = ea / length
    k[:, 0, 6] = -ea / length
    k[:, 3, 3] = k[:, 9, 9] = gj / length
    k[:, 3, 9] = -gj / length

    # (translation I, rotation I, translation J, rotation J, EI, Φ, sign of the coupling)
    for t1, r1, t2, r2, ei, phi, sgn in (
        (1, 5, 7, 11, eiz, phi_y, 1.0),
        (2, 4, 8, 10, eiy, phi_z, -1.0),
    ):
        a = 12.0 * ei / (length**3 * (1.0 + phi))
        b = sgn * 6.0 * ei / (length**2 * (1.0 + phi))
        c = (4.0 + phi) * ei / (length * (1.0 + phi))
        d = (2.0 - phi) * ei / (length * (1.0 + phi))
        k[:, t1, t1] = k[:, t2, t2] = a
        k[:, t1, t2] = -a
        k[:, t1, r1] = k[:, t1, r2] = b
        k[:, r1, t2] = k[:, t2, r2] = -b
        k[:, r1, r1] = k[:, r2, r2] = c
        k[:, r1, r2] = d

    upper = np.triu(k, 1)
    return k + np.transpose(upper, (0, 2, 1)) - np.tril(k, -1)


class NativeSolver:
    """
    Linear 3-D beam solver used in place of a MAPDL session.

    Parameters
    ----------
    shear_deformation : bool
        ``True`` (default): Timoshenko elements, like BEAM188, with the
        shear correction factors of the ``Section``.  ``False``:
        Euler-Bernoulli elements.

    Attributes
    ----------
    last : NativeResult | None
        Result of the most recent analysis (e.g. to read displacements).
    n_solves : int
        Number of analyses run.
    """

    def __init__(self, shear_deformation: bool = True) -> None:
        self.shear_deformation = bool(shear_deformation)
        self.last: NativeResult | None = None
        self.n_solves = 0
        self._prepared: tuple[StructuralModel, dict] | None = None

    # ------------------------------------------------------------------
    # Model data that do not depend on the section law (computed once)
    # ------------------------------------------------------------------

    def _prepare(self, model: StructuralModel) -> dict:
        if self._prepared is not None and self._prepared[0] is model:
            return self._prepared[1]

        mesh = model.beam.mesh
        index = {name: num - 1 for name, num in ModelBuilder.node_ids(model).items()}
        coords = np.vstack([mesh.coords, *[p[None, :] for p in model.points.values()]])
        n_dof = 6 * len(coords)

        rigid_groups = [{bc.master, *bc.slaves} for bc in model.bcs if isinstance(bc, RigidRegion)]
        fixed_beams = []
        for fe in model.fixed:
            if isinstance(fe, FixedBeam):
                fixed_beams.append(fe)
            elif isinstance(fe, FixedShell) and any(set(fe.nodes) <= g for g in rigid_groups):
                continue  # rigid: no strain, no stiffness
            else:
                raise NotImplementedError(
                    f"NativeSolver cannot model {type(fe).__name__} on nodes {fe.nodes}: "
                    "only FixedBeam, or a FixedShell whose nodes all belong to one RigidRegion"
                )

        # Linear constraints C u = d and load vector
        rows, rhs = [], []
        load = np.zeros(n_dof)

        def fix(node: str, dof: int, value: float = 0.0) -> None:
            row = np.zeros(n_dof)
            row[6 * index[node] + dof] = 1.0
            rows.append(row)
            rhs.append(value)

        for bc in model.bcs:
            if isinstance(bc, Clamp):
                for node in bc.nodes:
                    for dof in range(6):
                        fix(node, dof)
            elif isinstance(bc, Symmetry):
                for node in bc.nodes:
                    for dof in _SYMMETRY[bc.normal]:
                        fix(node, dof)
            elif isinstance(bc, ImposedDisplacement):
                if bc.dof not in _DOF:
                    raise NotImplementedError(f"NativeSolver: unsupported DOF label {bc.dof!r}")
                for node in bc.nodes:
                    fix(node, _DOF[bc.dof], bc.value)
            elif isinstance(bc, NodalForce):
                load[6 * index[bc.node] + _LOAD[bc.direction]] += bc.value
            elif isinstance(bc, RigidRegion):
                if str(bc.dofs).upper() != "ALL":
                    raise NotImplementedError("NativeSolver: RigidRegion supports dofs='ALL' only")
                m0 = 6 * index[bc.master]
                for slave in bc.slaves:
                    if slave == bc.master:
                        continue
                    s0 = 6 * index[slave]
                    rx, ry, rz = coords[index[slave]] - coords[index[bc.master]]
                    # u_s = u_m + θ_m × r,  θ_s = θ_m
                    cross = [(4, rz, 5, -ry), (5, rx, 3, -rz), (3, ry, 4, -rx)]
                    for k, (d1, c1, d2, c2) in enumerate(cross):
                        row = np.zeros(n_dof)
                        row[s0 + k], row[m0 + k] = 1.0, -1.0
                        row[m0 + d1], row[m0 + d2] = -c1, -c2
                        rows.append(row)
                        rhs.append(0.0)
                    for k in (3, 4, 5):
                        row = np.zeros(n_dof)
                        row[s0 + k], row[m0 + k] = 1.0, -1.0
                        rows.append(row)
                        rhs.append(0.0)
            else:
                raise NotImplementedError(
                    f"NativeSolver does not support the boundary condition {type(bc).__name__}"
                )

        # Null-space form of the constraints: u = u_p + Z y  (redundant
        # constraints, e.g. a symmetry plus an imposed displacement on the
        # same DOF, are handled by the rank-revealing decomposition).
        if rows:
            c_mat, d_vec = np.array(rows), np.array(rhs)
            u_svd, s_svd, vt = np.linalg.svd(c_mat, full_matrices=True)
            rank = int(np.sum(s_svd > 1e-10 * s_svd[0]))
            u_p = vt[:rank].T @ ((u_svd[:, :rank].T @ d_vec) / s_svd[:rank])
            if not np.allclose(c_mat @ u_p, d_vec, atol=1e-8 * max(1.0, np.abs(d_vec).max())):
                raise ValueError("NativeSolver: the boundary conditions are inconsistent")
            null = vt[rank:].T
        else:
            u_p, null = np.zeros(n_dof), np.eye(n_dof)

        # Element connectivity: optimisable chain first, then fixed beams
        n_elem = mesh.n_elem
        conn = [(i, i + 1) for i in range(n_elem)]
        conn += [(index[fe.nodes[0]], index[fe.nodes[1]]) for fe in fixed_beams]
        conn = np.array(conn)
        length, axes = element_axes(coords[conn[:, 0]], coords[conn[:, 1]])
        rot = np.zeros((len(conn), 12, 12))
        for k in range(4):
            rot[:, 3 * k : 3 * k + 3, 3 * k : 3 * k + 3] = axes
        dofs = np.concatenate(
            [6 * conn[:, :1] + np.arange(6), 6 * conn[:, 1:] + np.arange(6)], axis=1
        )

        data = {
            "index": index,
            "n_dof": n_dof,
            "load": load,
            "u_p": u_p,
            "null": null,
            "length": length,
            "rot": rot,
            "dofs": dofs,
            "fixed_sections": [fe.section for fe in fixed_beams],
        }
        self._prepared = (model, data)
        return data

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def analyze(self, model: StructuralModel, law, reaction_nodes: list[str]) -> NativeResult:
        """
        Solve *model* with the sections given by *law*.

        Parameters
        ----------
        model : StructuralModel
            Structure to analyse.
        law : SectionLaw
            Section law of the optimisable beam.
        reaction_nodes : list[str]
            Nodes whose element forces are summed into ``reaction``.

        Returns
        -------
        NativeResult

        Raises
        ------
        SolutionNotConvergedError
            If the stiffness matrix is singular (insufficient supports or a
            vanishing section).
        NotImplementedError
            For components or boundary conditions outside the solver scope.
        """
        data = self._prepare(model)
        mesh, mat = model.beam.mesh, model.material
        n_elem = mesh.n_elem
        shear_mod = mat.E / (2.0 * (1.0 + mat.nu))

        def props(section, size: int) -> np.ndarray:
            ky, kz = section.shear_correction
            values = (section.area, section.iy, section.iz, section.torsion_constant, ky, kz)
            return np.array([np.broadcast_to(np.asarray(v, float), (size,)) for v in values])

        parts = [props(model.beam.sections_at(law, mesh.s_elem), n_elem)]
        parts += [props(sec, 1) for sec in data["fixed_sections"]]
        area, iy, iz, tors, ky, kz = np.concatenate(parts, axis=1)

        length = data["length"]
        if self.shear_deformation:
            phi_y = 12.0 * mat.E * iz / (ky * shear_mod * area * length**2)
            phi_z = 12.0 * mat.E * iy / (kz * shear_mod * area * length**2)
        else:
            phi_y = phi_z = np.zeros_like(length)
        k_loc = local_stiffness(
            mat.E * area, shear_mod * tors, mat.E * iy, mat.E * iz, length, phi_y, phi_z
        )
        rot, dofs = data["rot"], data["dofs"]
        k_glob = np.transpose(rot, (0, 2, 1)) @ k_loc @ rot

        n_dof = data["n_dof"]
        stiff = np.zeros((n_dof, n_dof))
        np.add.at(stiff, (dofs[:, :, None], dofs[:, None, :]), k_glob)

        null, u_p = data["null"], data["u_p"]
        reduced = null.T @ stiff @ null
        residual = null.T @ (data["load"] - stiff @ u_p)
        try:
            with np.errstate(all="raise"):
                y = cho_solve(cho_factor(reduced), residual)
        except (LinAlgError, FloatingPointError, ValueError) as err:
            raise SolutionNotConvergedError(
                f"native solver: singular stiffness matrix ({err}); check the supports"
            ) from err
        u = u_p + null @ y
        self.n_solves += 1

        # Internal forces of the optimisable elements (stress resultants on
        # the +x face): minus the nodal force at I, plus the nodal force at J.
        u_loc = np.einsum("eij,ej->ei", rot[:n_elem], u[dofs[:n_elem]])
        f_loc = np.einsum("eij,ej->ei", k_loc[:n_elem], u_loc)
        forces = {}
        for k, comp in enumerate(_SMISC_COMPONENT):
            forces[1 + k] = -f_loc[:, comp]
            forces[14 + k] = f_loc[:, 6 + comp]

        nodal = (stiff @ u).reshape(-1, 6)
        rows = [data["index"][name] for name in reaction_nodes]
        reaction = float(np.linalg.norm(nodal[rows, :3].sum(axis=0)))

        self.last = NativeResult(u.reshape(-1, 6), dict(data["index"]), forces, reaction)
        return self.last
