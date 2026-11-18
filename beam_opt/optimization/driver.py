"""
beam_opt.optimization.driver
============================
Solves an ``OptimizationProblem`` with a selectable algorithm.

>>> result = optimize(problem, method="cobyla")
>>> result = optimize(problem, method="differential-evolution", max_evals=5000, seed=1)

Available methods (``METHODS``):

``"cobyla"``
    Derivative-free, linear approximations on a simplex.  Good default for
    a few tens of variables and noisy FEM responses.
``"slsqp"``
    Gradient-based (sequential quadratic programming), gradients by finite
    differences: ``n_vars + 1`` FEM solutions per iteration.  Fast and
    accurate on smooth problems, converges to the nearest local optimum.
``"nelder-mead"``
    Simplex search; the constraints enter through an exact penalty.  Robust
    but slow, only for few variables.
``"differential-evolution"``
    Evolutionary, population-based global search (the same family as the
    genetic algorithm: mutation, crossover, selection).  Does not need a
    good starting point and can escape local optima, at the price of
    thousands of FEM solutions.

All methods work on variables scaled to [0, 1] and on the mass divided by
the mass at the upper bounds, so the same tolerances suit every problem.
The design returned is the lightest *feasible* one among those evaluated.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
from scipy.optimize import (
    Bounds,
    NonlinearConstraint,
    OptimizeResult,
    differential_evolution,
    minimize,
)

from .problem import OptimizationProblem

logger = logging.getLogger(__name__)


class _BudgetReached(Exception):
    """The budget of FEM solutions is exhausted: stop the algorithm."""


class _Scaled:
    """The problem seen by the algorithms: z in [0, 1], objective relative to its magnitude."""

    def __init__(self, problem: OptimizationProblem, max_evals: int) -> None:
        self.problem = problem
        self.last_solve = len(problem.history) + int(max_evals)
        self.lo = problem.lower
        self.span = problem.upper - problem.lower
        self.f_ref = problem.objective_scale()
        self.n = problem.n_vars
        self.bounds = Bounds(np.zeros(self.n), np.ones(self.n))

    def x(self, z: np.ndarray) -> np.ndarray:
        return self.lo + np.clip(np.asarray(z, dtype=float), 0.0, 1.0) * self.span

    def z(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=float) - self.lo) / self.span

    def _check_budget(self) -> None:
        if len(self.problem.history) >= self.last_solve:
            raise _BudgetReached

    def objective(self, z: np.ndarray) -> float:
        if self.problem.objective_fn.needs_fem:
            self._check_budget()
        return self.problem.objective(self.x(z)) / self.f_ref

    def constraints(self, z: np.ndarray) -> np.ndarray:
        self._check_budget()
        return self.problem.constraint_vector(self.x(z))


def _inside(z0: np.ndarray, step: float) -> np.ndarray:
    """
    Move a starting point away from the upper bounds by the first step size.

    COBYLA and Nelder-Mead build their first simplex at ``z0 + step`` along
    every variable.  From a point on the upper bounds those vertices would
    fall outside the admissible box, where the response is flat (the
    sections are clipped): the simplex degenerates and the search stops at a
    poor design while reporting success.
    """
    return np.clip(z0, 0.0, max(0.0, 1.0 - step))


def _cobyla(sp: _Scaled, z0, max_evals, tol, seed, options) -> OptimizeResult:
    opts = {"maxiter": max_evals, "rhobeg": 0.1, "disp": False} | options
    return minimize(
        sp.objective,
        _inside(z0, opts["rhobeg"]),
        method="COBYLA",
        bounds=sp.bounds,
        constraints=[{"type": "ineq", "fun": sp.constraints}],
        tol=1e-5 if tol is None else tol,
        options=opts,
    )


def _slsqp(sp: _Scaled, z0, max_evals, tol, seed, options) -> OptimizeResult:
    # One iteration costs about n + 1 FEM solutions (forward differences).
    opts = {"maxiter": max(1, max_evals // (sp.n + 1)), "eps": 1e-4, "disp": False} | options
    return minimize(
        sp.objective,
        z0,
        method="SLSQP",
        bounds=sp.bounds,
        constraints=[{"type": "ineq", "fun": sp.constraints}],
        tol=1e-5 if tol is None else tol,
        options=opts,
    )


def _nelder_mead(sp: _Scaled, z0, max_evals, tol, seed, options) -> OptimizeResult:
    options = dict(options)
    weight = options.pop("penalty", 100.0)
    step = options.pop("initial_step", 0.1)  # first simplex size, fraction of the bounds range
    z0 = _inside(z0, step)
    simplex = np.vstack([z0, z0 + step * np.eye(sp.n)])

    def penalised(z: np.ndarray) -> float:
        violation = max(0.0, -float(np.min(sp.constraints(z))))
        return sp.objective(z) + weight * violation

    tol = 1e-6 if tol is None else tol
    opts = {"maxfev": max_evals, "xatol": tol, "fatol": tol, "adaptive": True}
    opts |= {"initial_simplex": simplex} | options
    return minimize(penalised, z0, method="Nelder-Mead", bounds=sp.bounds, options=opts)


def _differential_evolution(sp: _Scaled, z0, max_evals, tol, seed, options) -> OptimizeResult:
    options = dict(options)
    popsize = options.pop("popsize", 10)  # individuals = popsize * n_vars
    generations = max(1, max_evals // (popsize * sp.n) - 1)
    return differential_evolution(
        sp.objective,
        bounds=sp.bounds,
        constraints=NonlinearConstraint(sp.constraints, 0.0, np.inf),
        x0=z0,
        maxiter=generations,
        popsize=popsize,
        tol=1e-3 if tol is None else tol,
        seed=seed,
        polish=False,  # a gradient polish would cost many more FEM solutions: use "slsqp" after
        **options,
    )


#: Method name -> runner.  Register a new algorithm by adding an entry with
#: the signature ``(scaled_problem, z0, max_evals, tol, seed, options)``.
METHODS: dict[str, Callable[..., OptimizeResult]] = {
    "cobyla": _cobyla,
    "slsqp": _slsqp,
    "nelder-mead": _nelder_mead,
    "differential-evolution": _differential_evolution,
}


def default_x0(problem: OptimizationProblem) -> np.ndarray:
    """Mid-range value of every design variable [mm]."""
    return 0.5 * (problem.lower + problem.upper)


def optimize(
    problem: OptimizationProblem,
    method: str = "cobyla",
    x0: np.ndarray | None = None,
    max_evals: int = 2000,
    tol: float | None = None,
    feas_tol: float = 1e-4,
    seed: int | None = None,
    **options,
) -> OptimizeResult:
    """
    Minimise the objective of *problem* (the mass by default) with the chosen algorithm.

    Parameters
    ----------
    problem : OptimizationProblem
        Problem to solve.
    method : str
        Key of ``METHODS``: ``"cobyla"``, ``"slsqp"``, ``"nelder-mead"`` or
        ``"differential-evolution"`` (see the module docstring).
    x0 : ndarray | None
        Starting design [mm] (length ``problem.n_vars``); for the
        evolutionary method it is one individual of the first generation.
        Defaults to ``default_x0(problem)``.
    max_evals : int
        Budget of FEM solutions: the search stops when it is exhausted and
        the best design found so far is returned (``success=False``).
    tol : float | None
        Convergence tolerance of the method, on scaled quantities [-];
        ``None`` picks a suitable default.
    feas_tol : float
        Largest constraint violation accepted in the returned design [-].
    seed : int | None
        Random seed (evolutionary method only), for reproducible runs.
    **options
        Method-specific options, e.g. ``rhobeg`` (COBYLA, fraction of the
        bounds range), ``eps`` (SLSQP finite-difference step, scaled),
        ``penalty`` and ``initial_step`` (Nelder-Mead), ``popsize``, ``mutation``,
        ``recombination`` (differential evolution).

    Returns
    -------
    OptimizeResult
        ``x`` [mm] and ``fun`` (objective value, e.g. mass [g]) of the best
        feasible design evaluated,
        ``max_violation`` [-], ``nfev`` (FEM solutions of this run),
        ``success``, ``message``, ``method`` and ``raw`` (the result of the
        underlying scipy routine, in scaled variables).  ``success`` is
        ``False`` when no evaluated design is feasible; ``x`` is then the
        least violating one.

    Raises
    ------
    ValueError
        For an unknown method or a starting point of the wrong size.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}: choose among {sorted(METHODS)}")
    x0 = default_x0(problem) if x0 is None else np.asarray(x0, dtype=float)
    if x0.size != problem.n_vars:
        raise ValueError(f"x0 has {x0.size} entries, expected {problem.n_vars}")

    start = len(problem.history)
    sp = _Scaled(problem, max_evals)
    logger.info("Starting %s  n_vars=%d  max_evals=%d", method, problem.n_vars, max_evals)
    z0 = np.clip(sp.z(x0), 0.0, 1.0)
    try:
        raw = METHODS[method](sp, z0, int(max_evals), tol, seed, options)
    except _BudgetReached:
        last = problem.history[-1].x if len(problem.history) > start else x0
        raw = OptimizeResult(
            x=sp.z(last), success=False, message=f"budget of {max_evals} FEM solutions reached"
        )

    # Best feasible design among those evaluated in this run (the point
    # returned by the algorithm is one of them, or is added here).
    candidates = [ev for ev in problem.history[start:] if not ev.failed]
    final = problem.evaluate(sp.x(raw.x))
    if not final.failed:
        candidates.append(final)
    if not candidates:
        raise RuntimeError("every FEM evaluation of the run failed")
    violation = np.array([problem.max_violation(ev) for ev in candidates])
    value = np.array([problem.objective_of(ev) for ev in candidates])
    feasible = violation <= feas_tol
    if np.any(feasible):
        best = int(np.flatnonzero(feasible)[np.argmin(value[feasible])])
        success, message = bool(raw.success), str(raw.message)
    else:
        best = int(np.argmin(violation))
        success = False
        message = f"no feasible design found (smallest violation {violation[best]:.2e})"

    return OptimizeResult(
        x=candidates[best].x.copy(),
        fun=float(value[best]),
        max_violation=float(violation[best]),
        nfev=len(problem.history) - start,
        success=success,
        message=message,
        method=method,
        raw=raw,
    )


class CobylaDriver:
    """
    COBYLA driver, kept for compatibility: ``optimize(problem, "cobyla")``.

    Parameters
    ----------
    problem : OptimizationProblem
        Problem to solve.
    """

    def __init__(self, problem: OptimizationProblem) -> None:
        self.problem = problem

    def default_x0(self) -> np.ndarray:
        """Mid-range value of every control point [mm]."""
        return default_x0(self.problem)

    def run(
        self,
        x0: np.ndarray | None = None,
        maxiter: int = 3000,
        tol: float = 1e-5,
        rhobeg: float = 0.1,
        disp: bool = False,
    ) -> OptimizeResult:
        """
        Launch the optimisation (see ``optimize``).

        ``rhobeg`` is the initial step as a fraction of the bounds range [-].
        """
        return optimize(
            self.problem, "cobyla", x0=x0, max_evals=maxiter, tol=tol, rhobeg=rhobeg, disp=disp
        )
