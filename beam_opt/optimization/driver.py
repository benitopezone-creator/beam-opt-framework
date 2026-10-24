"""
beam_opt.optimization.driver
============================
Runs an ``OptimizationProblem`` with ``scipy.optimize.minimize``.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import OptimizeResult, minimize

from .problem import OptimizationProblem

logger = logging.getLogger(__name__)


class CobylaDriver:
    """
    COBYLA driver (derivative-free, inequality constraints).

    Parameters
    ----------
    problem : OptimizationProblem
        Problem to solve.
    """

    def __init__(self, problem: OptimizationProblem) -> None:
        self.problem = problem

    def default_x0(self) -> np.ndarray:
        """Mid-range value of every control point."""
        p = self.problem
        return np.concatenate([np.full(p.n_ctrl, 0.5 * (lo + hi)) for lo, hi in p.bounds.values()])

    def run(
        self,
        x0: np.ndarray | None = None,
        maxiter: int = 3000,
        tol: float = 1e-4,
        rhobeg: float = 0.3,
        disp: bool = True,
    ) -> OptimizeResult:
        """
        Launch the optimisation.

        Parameters
        ----------
        x0 : ndarray | None
            Starting point (length ``problem.n_vars``).  Defaults to
            ``default_x0()``.
        maxiter, tol, rhobeg, disp
            Passed to ``scipy.optimize.minimize``; ``rhobeg`` in design-vector
            units [mm].

        Returns
        -------
        OptimizeResult
        """
        p = self.problem
        x0 = self.default_x0() if x0 is None else np.asarray(x0, dtype=float)
        if x0.size != p.n_vars:
            raise ValueError(f"x0 has {x0.size} entries, expected {p.n_vars}")

        logger.info(
            "Starting COBYLA optimisation  n_ctrl=%d  n_vars=%d  maxiter=%d",
            p.n_ctrl,
            p.n_vars,
            maxiter,
        )
        return minimize(
            p.objective,
            x0,
            method="COBYLA",
            constraints=p.constraints(),
            options={"maxiter": maxiter, "tol": tol, "rhobeg": rhobeg, "disp": disp},
        )
