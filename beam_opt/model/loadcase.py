"""
beam_opt.model.loadcase
=======================
Independent load cases.

A ``LoadCase`` lists the loads of one situation (nodal forces and imposed
displacements).  The supports of the ``StructuralModel`` are common to every
case; its own nodal forces are replaced by those of the case, and an imposed
displacement of the case replaces the model's on the same node and DOF.

>>> cases = [
...     LoadCase("vertical", [NodalForce("tip", "FZ", -200.0)], cycles=1000),
...     LoadCase("lateral", [NodalForce("tip", "FY", 80.0)], cycles=50),
... ]
>>> problem = OptimizationProblem(..., load_cases=cases)

Every constraint must hold in every case; the objectives combine the cases
(see ``beam_opt.optimization.objectives``), and fatigue damage is summed
over the cases with Miner's rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .bcs import BoundaryCondition, ImposedDisplacement, NodalForce
from .components import StructuralModel


@dataclass
class LoadCase:
    """
    One load situation of the structure.

    Parameters
    ----------
    name : str
        Label of the case.
    loads : list[BoundaryCondition]
        ``NodalForce`` and ``ImposedDisplacement`` objects of the case.
    weight : float
        Weight of the case in combined objectives (e.g. the compliance) [-].
    spectrum : LoadSpectrum | None
        Cycles of this case per repetition, as load factors of its loads.
        ``None``: one cycle of the ``FatigueModel`` (its ``load_ratio``) per
        repetition — or *cycles* of them.
    cycles : float
        Number of constant-amplitude cycles per repetition when *spectrum*
        is ``None`` [-].
    """

    name: str
    loads: list[BoundaryCondition] = field(default_factory=list)
    weight: float = 1.0
    spectrum: object | None = None
    cycles: float = 1.0

    def __post_init__(self) -> None:
        for bc in self.loads:
            if not isinstance(bc, NodalForce | ImposedDisplacement):
                raise TypeError(
                    f"load case {self.name!r}: only NodalForce and ImposedDisplacement are loads, "
                    f"got {type(bc).__name__} (supports belong to the model)"
                )
        if self.weight < 0.0 or self.cycles < 0.0:
            raise ValueError("weight and cycles must be non-negative")

    def apply_to(self, model: StructuralModel) -> StructuralModel:
        """The model with the loads of this case."""
        imposed = {
            (node, bc.dof)
            for bc in self.loads
            if isinstance(bc, ImposedDisplacement)
            for node in bc.nodes
        }
        bcs = []
        for bc in model.bcs:
            if isinstance(bc, NodalForce):
                continue
            if isinstance(bc, ImposedDisplacement) and any(
                (n, bc.dof) in imposed for n in bc.nodes
            ):
                kept = [n for n in bc.nodes if (n, bc.dof) not in imposed]
                if kept:
                    bcs.append(ImposedDisplacement(kept, bc.dof, bc.value))
                continue
            bcs.append(bc)
        return StructuralModel(
            beam=model.beam,
            material=model.material,
            points=dict(model.points),
            fixed=list(model.fixed),
            bcs=bcs + list(self.loads),
        )
