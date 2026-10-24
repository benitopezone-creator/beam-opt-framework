"""
beam_opt.model.builder
======================
Writes a ``StructuralModel`` into an injected MAPDL session.

Numbering (decided here, never by configurations):

* beam nodes ``1 .. n_elem + 1`` in chain order, named points after them;
* beam elements and sections ``1 .. n_elem``;
* fixed-element sections ``n_elem + 1, n_elem + 2, ...`` (no collision for
  any ``n_elem``).
"""

from __future__ import annotations

from .components import StructuralModel

#: Element types defined by every build, with their ``ET`` numbers.
ELEMENT_TYPES: dict[str, int] = {"BEAM188": 1, "SHELL181": 2}
MATERIAL_ID = 1


class ModelBuilder:
    """
    (Re)builds the full model at every FEM evaluation.

    Parameters
    ----------
    mapdl
        MAPDL session (or a test double), owned by the caller.
    jobname : str | None
        If given, re-issued with ``/FILNAME`` after ``/CLEAR``.
    """

    def __init__(self, mapdl, jobname: str | None = None) -> None:
        self.mapdl = mapdl
        self.jobname = jobname

    @staticmethod
    def node_ids(model: StructuralModel) -> dict[str, int]:
        """Name -> MAPDL node number for every named node of *model*."""
        ids = {name: i + 1 for name, i in model.beam.mesh.anchors.items()}
        first = model.beam.mesh.n_nodes + 1
        ids.update({name: first + k for k, name in enumerate(model.points)})
        return ids

    def build(self, model: StructuralModel, law) -> dict[str, int]:
        """
        Clear the session and write nodes, sections, elements and BCs.

        Parameters
        ----------
        model : StructuralModel
            Structure to write.
        law : SectionLaw
            Section law of the optimisable beam.

        Returns
        -------
        dict[str, int]
            Name -> MAPDL node number.
        """
        m = self.mapdl
        mesh = model.beam.mesh
        ids = self.node_ids(model)

        m.clear()
        if self.jobname:
            m.filname(self.jobname)
        m.prep7()

        for name, num in ELEMENT_TYPES.items():
            m.et(num, name)

        mat = model.material
        m.mp("EX", MATERIAL_ID, mat.E)
        m.mp("PRXY", MATERIAL_ID, mat.nu)
        m.mp("DENS", MATERIAL_ID, mat.rho)

        for i, xyz in enumerate(mesh.coords, 1):
            m.n(i, *xyz)
        for name, xyz in model.points.items():
            m.n(ids[name], *xyz)

        # Optimisable sections and elements (1 .. n_elem)
        params = law.eval(mesh.s_elem)
        for i in range(mesh.n_elem):
            section = model.beam.section_factory({k: v[i] for k, v in params.items()})
            section.define(m, i + 1, f"OPT{i + 1}")

        m.type(ELEMENT_TYPES["BEAM188"])
        m.mat(MATERIAL_ID)
        for i in range(mesh.n_elem):
            m.secnum(i + 1)
            m.e(i + 1, i + 2)

        for k, fe in enumerate(model.fixed):
            fe.create(
                m,
                [ids[n] for n in fe.nodes],
                secnum=mesh.n_elem + 1 + k,
                etype=ELEMENT_TYPES[fe.element_type],
                mat=MATERIAL_ID,
            )

        for bc in model.bcs:
            bc.apply(m, ids)

        m.allsel("ALL")
        m.finish()
        return ids
