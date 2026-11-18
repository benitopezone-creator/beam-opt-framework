"""
beam_opt.optimization.objectives
================================
What the optimisation minimises: the ``Objective`` interface and the mass.

The framework minimises the mass by default (``Mass``).  The objective is
replaceable: other objectives (stiffness, peak stress, fatigue life...) are
provided as extras in ``beam_opt.extras``, and a user objective only has to
subclass ``Objective``.

Every objective is a value to *minimise*, computed from an ``Evaluation``.
When the objective is not the mass, a mass budget (``mass_max``) is
usually needed, otherwise the optimum is simply the heaviest design.

Min-max objectives (the worst of several values, e.g. over the elements)
also give the individual values (``values``).  The gradient-based and
simplex-based algorithms then solve the equivalent *bound formulation*

    min τ   subject to   value_e(x) ≤ τ   for every element e

whose functions are smooth, instead of the non-smooth maximum.

To add an objective, subclass ``Objective`` and implement ``value`` (and
``values`` for a min-max objective); list in ``watch`` the nodal
displacements it needs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Objective(ABC):
    """A scalar to minimise."""

    #: Short label for reports.
    name: str = "objective"
    #: Unit of ``value`` for reports.
    unit: str = ""
    #: ``False`` if ``value`` needs no FEM solution (e.g. the mass).
    needs_fem: bool = True

    def watch(self, problem) -> list[tuple[str, str]]:
        """Nodal displacements ``(node, dof)`` the objective needs."""
        return []

    def values(self, problem, ev) -> np.ndarray | None:
        """
        Per-element values of a min-max objective (``value`` is their maximum).

        ``None`` for an ordinary scalar objective.
        """
        return None

    def check(self, problem) -> None:
        """Raise ``ValueError`` if the objective does not apply to *problem*."""

    @abstractmethod
    def value(self, problem, ev) -> float:
        """Value of the objective for evaluation *ev* of *problem*."""

    def describe(self, problem, ev) -> str:
        """One line for the report."""
        return f"{self.name}: {self.value(problem, ev):.6g} {self.unit}".rstrip()


class Mass(Objective):
    """Mass of the optimisable beam [g]."""

    name, unit, needs_fem = "mass", "g", False

    def value(self, problem, ev) -> float:
        from .problem import TON_TO_G

        return ev.mass * TON_TO_G
