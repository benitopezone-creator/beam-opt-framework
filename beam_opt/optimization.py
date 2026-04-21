"""
beam_opt.optimization
=====================
COBYLA-based dimensional optimizer for spline beam structures.

``BeamOptimizer`` wraps ``BeamModel`` + ``BeamAnalysis`` and exposes the
objective function, constraint functions, and a one-call ``run()`` method.

The design variables are:
    x = [b_ctrl_0, …, b_ctrl_{n-1}, h_ctrl_0, …, h_ctrl_{n-1}]
i.e. ``2 * n_ctrl`` scalar values.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy.optimize import minimize, OptimizeResult

from .config import B_MIN, B_MAX, H_MIN, H_MAX, RHO, SIGMA_S, FS, F_REACT_MAX
from .model import BeamModel
from .analysis import BeamAnalysis
from .section_law import SectionLaw
from .stress import AbstractStressEvaluator

logger = logging.getLogger(__name__)


class BeamOptimizer:
    """
    Dimensional optimizer for a spline beam structure.

    Parameters
    ----------
    model : BeamModel
    n_ctrl : int
        Number of spline control points for the section law.
    sigma_s : float
        Material yield stress [MPa].
    fs : float
        Safety factor.
    f_react_max : float
        Maximum admissible resultant reaction [N].
    b_bounds, h_bounds : tuple[float, float]
        Section dimension bounds [mm].
    stress_evaluator : AbstractStressEvaluator | None
        Custom stress evaluator; defaults to ``RectBeamStressEvaluator``.
    """

    def __init__(
        self,
        model: BeamModel,
        n_ctrl: int = 8,
        sigma_s: float = SIGMA_S,
        fs: float = FS,
        f_react_max: float = F_REACT_MAX,
        b_bounds: tuple[float, float] = (B_MIN, B_MAX),
        h_bounds: tuple[float, float] = (H_MIN, H_MAX),
        stress_evaluator: Optional[AbstractStressEvaluator] = None,
    ) -> None:
        self.model = model
        self.n_ctrl = int(n_ctrl)
        self.sigma_s = float(sigma_s)
        self.fs = float(fs)
        self.f_react_max = float(f_react_max)
        self.b_bounds = b_bounds
        self.h_bounds = h_bounds

        self.analysis = BeamAnalysis(model, stress_evaluator)
        self.s_ctrl = np.linspace(0.0, model.s[-1], n_ctrl)

        # Optimisation cache (avoids redundant FEM solves)
        self._cache_x: Optional[np.ndarray] = None
        self._cache_sigma: Optional[np.ndarray] = None
        self._cache_react: Optional[float] = None
        self._cache_mopt: Optional[float] = None

        # Fixed-element mass (constant throughout the optimisation)
        self.mass_fixed: float = self._compute_fixed_mass()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        x0: Optional[np.ndarray] = None,
        maxiter: int = 3000,
        tol: float = 1e-4,
        rhobeg: float = 0.3,
        disp: bool = True,
    ) -> OptimizeResult:
        """
        Launch the COBYLA optimisation.

        Parameters
        ----------
        x0 : ndarray | None
            Starting point (length ``2 * n_ctrl``).  Defaults to all-1.5 mm.
        maxiter, tol, rhobeg, disp
            Passed directly to ``scipy.optimize.minimize``.

        Returns
        -------
        OptimizeResult
        """
        if x0 is None:
            x0 = np.ones(2 * self.n_ctrl) * 1.5

        constraints = [
            {"type": "ineq", "fun": self.constraint_section_field},
            {"type": "ineq", "fun": self.constraint_bounds},
            {"type": "ineq", "fun": self.constraint_stress},
            {"type": "ineq", "fun": self.constraint_reaction},
        ]

        logger.info(
            "Starting COBYLA optimisation  n_ctrl=%d  n_vars=%d  maxiter=%d",
            self.n_ctrl, len(x0), maxiter,
        )

        result = minimize(
            self.obj_mass,
            x0,
            method="COBYLA",
            constraints=constraints,
            options={"maxiter": maxiter, "tol": tol, "rhobeg": rhobeg, "disp": disp},
        )

        return result

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------

    def obj_mass(self, x: np.ndarray) -> float:
        """Objective: mass of optimisable elements [g]."""
        self._solve_if_needed(x)
        return self._cache_mopt * 1e6  # ton → g

    def total_mass(self, x: np.ndarray) -> float:
        """Total mass of the full structure [g]."""
        return self.obj_mass(x) + self.mass_fixed * 1e6

    # ------------------------------------------------------------------
    # Constraints  (all return values >= 0 when satisfied)
    # ------------------------------------------------------------------

    def constraint_bounds(self, x: np.ndarray) -> np.ndarray:
        """Box bounds on the spline control-point values."""
        b_lo, b_hi = self.b_bounds
        h_lo, h_hi = self.h_bounds
        g = []
        for i in range(self.n_ctrl):
            g.extend([x[i] - b_lo, b_hi - x[i]])
        for i in range(self.n_ctrl):
            g.extend([x[self.n_ctrl + i] - h_lo, h_hi - x[self.n_ctrl + i]])
        return np.array(g)

    def constraint_section_field(self, x: np.ndarray) -> np.ndarray:
        """Box bounds on b(s) and h(s) at element mid-points."""
        law = self._law_from_x(x)
        b, h = law.eval(self.model.s_elem)
        b_lo, b_hi = self.b_bounds
        h_lo, h_hi = self.h_bounds
        span_b = b_hi - b_lo
        span_h = h_hi - h_lo
        t = []
        t.extend((b - b_lo) / span_b)
        t.extend((b_hi - b) / span_b)
        t.extend((h - h_lo) / span_h)
        t.extend((h_hi - h) / span_h)
        return np.array(t)

    def constraint_stress(self, x: np.ndarray) -> np.ndarray:
        """Per-element stress constraint: σ_VM ≤ σ_s / FS."""
        self._solve_if_needed(x)
        sigma_amm = self.sigma_s / self.fs
        return (sigma_amm - self._cache_sigma) / sigma_amm

    def constraint_reaction(self, x: np.ndarray) -> float:
        """Reaction force constraint: F_react ≤ F_react_max."""
        self._solve_if_needed(x)
        return 1.0 - self._cache_react / self.f_react_max

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _law_from_x(self, x: np.ndarray) -> SectionLaw:
        x = np.asarray(x, dtype=np.float64).copy()
        b_lo, b_hi = self.b_bounds
        h_lo, h_hi = self.h_bounds
        x[: self.n_ctrl] = np.clip(x[: self.n_ctrl], b_lo, b_hi)
        x[self.n_ctrl :] = np.clip(x[self.n_ctrl :], h_lo, h_hi)
        return SectionLaw(
            self.s_ctrl,
            x[: self.n_ctrl],
            x[self.n_ctrl :],
            b_bounds=self.b_bounds,
            h_bounds=self.h_bounds,
        )

    def _solve_if_needed(self, x: np.ndarray) -> None:
        """Run FEM only if the design vector has changed since the last call."""
        x = np.asarray(x, dtype=np.float64)
        if self._cache_x is not None and np.allclose(x, self._cache_x, rtol=1e-6, atol=1e-8):
            return

        law = self._law_from_x(x)
        self.model.build(law)
        self.analysis.solve()

        self._cache_sigma = self.analysis.stress_field()
        self._cache_react = self.analysis.max_reaction()

        b, h = law.eval(self.model.s_elem)
        self._cache_mopt = float(RHO * np.sum(b * h * self.model.L))
        self._cache_x = x.copy()

    def _compute_fixed_mass(self) -> float:
        total = 0.0
        for elem in self.model.fixed_elements:
            total += elem.estimate_mass(self.model.nodes_dict, RHO)
        return total
