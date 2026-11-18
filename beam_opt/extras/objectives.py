"""
beam_opt.extras.objectives
==========================
Objectives other than the mass.

>>> from beam_opt.extras import Compliance, FatigueLife
>>> OptimizationProblem(..., objective=Compliance(), mass_max=150.0)  # stiffest for 150 g
>>> OptimizationProblem(..., objective=FatigueLife(), fatigue=FatigueModel(curve, load_ratio=0))

``PeakStress``, ``FatigueLife`` and, with several load cases,
``Displacement`` are min-max objectives: SLSQP and COBYLA solve them in the
bound form ``min τ  s.t.  value_i(x) ≤ τ`` (see ``beam_opt.optimization``).
"""

from __future__ import annotations

import numpy as np

from ..model.bcs import NodalForce
from ..optimization.objectives import Objective

_FORCE_TO_DOF = {"FX": "UX", "FY": "UY", "FZ": "UZ", "MX": "ROTX", "MY": "ROTY", "MZ": "ROTZ"}


def ks_max(values: np.ndarray, rho: float) -> float:
    """
    Kreisselmeier-Steinhauser envelope: a smooth upper bound of ``max(values)``.

    ``max ≤ KS ≤ max + ln(n) / ρ``; larger ``ρ`` is closer to the maximum
    and less smooth.
    """
    v = np.asarray(values, dtype=float)
    top = float(np.max(v))
    return top + float(np.log(np.sum(np.exp(rho * (v - top))))) / rho


class Compliance(Objective):
    """
    Compliance ``C = Σ F_i u_i`` of the nodal loads [N·mm]: minimum compliance
    is maximum stiffness.

    Only nodal forces and moments do work; imposed displacements are
    supports (zero work).  With load cases, the weighted sum
    ``Σ_c w_c C_c``.
    """

    name, unit = "compliance", "N·mm"

    @staticmethod
    def _loads(model) -> list[NodalForce]:
        return [bc for bc in model.bcs if isinstance(bc, NodalForce)]

    def check(self, problem) -> None:
        if not any(self._loads(m) for m in problem.case_models):
            raise ValueError("Compliance needs at least one NodalForce")

    def watch(self, problem) -> list[tuple[str, str]]:
        return [
            (bc.node, _FORCE_TO_DOF[bc.direction])
            for m in problem.case_models
            for bc in self._loads(m)
        ]

    def case_values(self, problem, ev) -> list[float]:
        """Compliance of every load case [N·mm]."""
        return [
            float(sum(bc.value * res.watched[(bc.node, _FORCE_TO_DOF[bc.direction])]
                      for bc in self._loads(m)))
            for m, res in zip(problem.case_models, ev.cases)
        ]  # fmt: skip

    def value(self, problem, ev) -> float:
        weights = [c.weight for c in problem.load_cases] or [1.0]
        return float(sum(w * c for w, c in zip(weights, self.case_values(problem, ev))))


class Displacement(Objective):
    """
    Absolute value of one displacement or rotation [mm] or [rad].

    Parameters
    ----------
    node : str
        Node name.
    dof : str
        ``"UX"``, ``"UY"``, ``"UZ"``, ``"ROTX"``, ``"ROTY"`` or ``"ROTZ"``.
    """

    name = "displacement"

    def __init__(self, node: str, dof: str) -> None:
        self.node, self.dof = str(node), str(dof).upper()
        self.unit = "rad" if self.dof.startswith("ROT") else "mm"

    def check(self, problem) -> None:
        if self.node not in problem.model.node_names() or self.dof not in _FORCE_TO_DOF.values():
            raise ValueError(f"invalid displacement objective {(self.node, self.dof)}")

    def watch(self, problem) -> list[tuple[str, str]]:
        return [(self.node, self.dof)]

    def values(self, problem, ev) -> np.ndarray:
        """Magnitude of the displacement in every load case."""
        return np.array([abs(float(res.watched[(self.node, self.dof)])) for res in ev.cases])

    def is_minmax(self, problem) -> bool:
        """A worst case only with several load cases."""
        return len(problem.load_cases) > 1

    def value(self, problem, ev) -> float:
        return float(np.max(self.values(problem, ev)))

    def describe(self, problem, ev) -> str:
        return f"|{self.dof}| at {self.node!r}: {self.value(problem, ev):.6g} {self.unit}"


class PeakStress(Objective):
    """
    Peak equivalent stress over the beam, as a fraction of σ_adm [-].

    A min-max objective: the stress of every element is bounded by the
    optimisation variable τ (see the module docstring).
    """

    name, unit = "peak stress / σ_adm", "-"

    def values(self, problem, ev) -> np.ndarray:
        return np.concatenate([res.sigma for res in ev.cases]) / problem.sigma_adm

    def value(self, problem, ev) -> float:
        return float(np.max(self.values(problem, ev)))

    def describe(self, problem, ev) -> str:
        return f"peak stress: {float(np.max(ev.sigma)):.4g} MPa ({np.max(ev.sigma) / problem.sigma_adm:.1%} of σ_adm)"


class FatigueLife(Objective):
    """
    Fatigue life of the beam, maximised: the value is ``−log10(N_min)`` [-].

    ``N_min`` is the life of the critical element, from the problem's
    ``FatigueModel``.  A min-max objective on ``log10(1/N)`` per element
    (see the module docstring).
    """

    name, unit = "fatigue life", "cycles"

    def check(self, problem) -> None:
        if problem.fatigue is None:
            raise ValueError("FatigueLife needs OptimizationProblem(fatigue=FatigueModel(...))")

    def values(self, problem, ev) -> np.ndarray:
        return -np.log10(problem.lives(ev))

    def value(self, problem, ev) -> float:
        return float(np.max(self.values(problem, ev)))

    def describe(self, problem, ev) -> str:
        lives = problem.lives(ev)
        k = int(np.argmin(lives))
        return f"fatigue life: {lives[k]:.4g} {problem.life_unit} (element {k + 1})"


class WeightedSum(Objective):
    """
    Weighted sum of objectives, each divided by a reference value [-].

    Parameters
    ----------
    terms : list[tuple[Objective, float]]
        ``(objective, weight)`` pairs.
    references : list[float] | None
        Values that make the terms comparable (e.g. each objective at the
        starting design).  Default: the value at the upper bounds,
        computed at the first evaluation.

    Notes
    -----
    A weighted sum finds only the convex part of a Pareto front: use
    ``beam_opt.optimization.epsilon_constraint`` to trace the whole front.
    """

    name, unit = "weighted sum", "-"

    def __init__(self, terms: list[tuple[Objective, float]], references=None) -> None:
        self.terms = [(obj, float(w)) for obj, w in terms]
        self.references = None if references is None else [float(r) for r in references]
        self.needs_fem = any(obj.needs_fem for obj, _ in self.terms)

    def check(self, problem) -> None:
        for obj, _ in self.terms:
            obj.check(problem)

    def watch(self, problem) -> list[tuple[str, str]]:
        return [w for obj, _ in self.terms for w in obj.watch(problem)]

    def value(self, problem, ev) -> float:
        if self.references is None:
            self.references = [1.0] * len(self.terms)  # placeholder while evaluating
            ref = problem.evaluate(problem.upper)
            self.references = [abs(obj.value(problem, ref)) or 1.0 for obj, _ in self.terms]
        return float(
            sum(w * obj.value(problem, ev) / r for (obj, w), r in zip(self.terms, self.references))
        )

    def describe(self, problem, ev) -> str:
        parts = "; ".join(obj.describe(problem, ev) for obj, _ in self.terms)
        return f"weighted sum {self.value(problem, ev):.6g} ({parts})"
