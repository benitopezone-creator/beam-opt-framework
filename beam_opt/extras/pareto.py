"""
beam_opt.extras.pareto
======================
Trade-off curves between two goals by the epsilon-constraint method.

One goal is the objective of the problem, the other is turned into a
constraint whose limit is swept:

>>> problem = OptimizationProblem(..., objective=Compliance(), mass_max=100.0)
>>> front = epsilon_constraint(problem, "mass_max", [60, 80, 100, 140, 200])
>>> [(p.level, p.objective) for p in front if p.feasible]

Unlike a weighted sum, the method also finds the non-convex parts of the
front.  Each run starts from the optimum of the previous one, so sweeping
the limit from the tightest to the loosest (or the reverse) is cheap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import OptimizeResult

from ..optimization.driver import optimize
from ..optimization.problem import OptimizationProblem


@dataclass
class ParetoPoint:
    """
    One optimum of the sweep.

    Attributes
    ----------
    level : float
        Value of the swept limit.
    x : ndarray
        Optimal design [mm].
    objective : float
        Objective at the optimum.
    mass : float
        Mass of the optimisable beam at the optimum [g].
    feasible : bool
        Whether the optimum satisfies every constraint.
    result : OptimizeResult
        Full result of ``optimize``.
    """

    level: float
    x: np.ndarray
    objective: float
    mass: float
    feasible: bool
    result: OptimizeResult


def epsilon_constraint(
    problem: OptimizationProblem,
    attribute: str,
    levels,
    method: str = "slsqp",
    x0: np.ndarray | None = None,
    feas_tol: float = 1e-4,
    **kwargs,
) -> list[ParetoPoint]:
    """
    Optimise *problem* once per value of a constraint limit.

    Parameters
    ----------
    problem : OptimizationProblem
        Problem whose objective is the first goal.
    attribute : str
        Constraint limit to sweep, an attribute of the problem:
        ``"mass_max"`` [g], ``"life_min"`` [cycles], ``"f_react_min"`` or
        ``"f_react_max"`` [N].
    levels : iterable of float
        Values of the limit, in the order of the sweep.
    method : str
        Algorithm (see ``beam_opt.optimization.METHODS``).
    x0 : ndarray | None
        Starting design of the first run; each next run starts from the
        previous optimum.
    feas_tol : float
        Largest constraint violation of a feasible optimum [-].
    **kwargs
        Passed to ``optimize`` (e.g. ``max_evals``).

    Returns
    -------
    list[ParetoPoint]
        One point per level.  The limit is restored afterwards.
    """
    allowed = ("mass_max", "life_min", "f_react_min", "f_react_max")
    if attribute not in allowed:
        raise ValueError(f"attribute must be one of {allowed}, got {attribute!r}")
    if attribute == "life_min" and problem.fatigue is None:
        raise ValueError("sweeping life_min needs a FatigueModel")
    if attribute.startswith("f_react") and not problem.reaction_nodes:
        raise ValueError("sweeping a reaction limit needs reaction_nodes")

    original = getattr(problem, attribute)
    points: list[ParetoPoint] = []
    start = x0
    try:
        for level in levels:
            setattr(problem, attribute, float(level))
            result = optimize(problem, method, x0=start, feas_tol=feas_tol, **kwargs)
            points.append(
                ParetoPoint(
                    level=float(level),
                    x=result.x.copy(),
                    objective=float(result.fun),
                    mass=problem.mass_g(result.x),
                    feasible=result.max_violation <= feas_tol,
                    result=result,
                )
            )
            start = result.x
    finally:
        setattr(problem, attribute, original)
    return points
