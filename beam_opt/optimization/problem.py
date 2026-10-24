"""
beam_opt.optimization.problem
=============================
Objective and constraints of the minimum-mass sizing problem.

The design vector concatenates the control-point values of each section
parameter, in the order of ``bounds``:

    x = [p1_ctrl_0, …, p1_ctrl_{n-1}, p2_ctrl_0, …]

e.g. ``[b_0 … b_{n-1}, h_0 … h_{n-1}]`` for a rectangular section.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from ..model import ModelBuilder, StructuralModel
from ..post import RectBeamStressEvaluator, StressEvaluator, last_results, reaction_resultant
from ..section_law import SectionLaw
from ..solver import SolverSettings, solve

logger = logging.getLogger(__name__)

TON_TO_G = 1e6


@dataclass
class Evaluation:
    """
    Results of one FEM evaluation.

    Attributes
    ----------
    x : ndarray
        Design vector.
    mass : float
        Mass of the optimisable beam [ton].
    sigma : ndarray, shape (n_elem,)
        Equivalent stress per optimisable element [MPa].
    reaction : float
        Resultant reaction on ``reaction_nodes`` [N].
    """

    x: np.ndarray
    mass: float
    sigma: np.ndarray
    reaction: float


class OptimizationProblem:
    """
    Minimum mass subject to stress, reaction and technological bounds.

    Parameters
    ----------
    mapdl
        MAPDL session (or a test double), owned by the caller.
    model : StructuralModel
        Structure to optimise.
    n_ctrl : int
        Number of control points of the section law, equally spaced in *s*.
    bounds : mapping[str, tuple[float, float]]
        Admissible ``(min, max)`` per section parameter [mm]; its order is the
        order of the parameters in the design vector.
    safety_factor : float
        Admissible stress = ``material.sigma_y / safety_factor`` [-].
    f_react_max : float
        Maximum admissible resultant reaction [N].
    reaction_nodes : list[str]
        Names of the nodes whose reactions are summed.
    evaluator : StressEvaluator | None
        Defaults to ``RectBeamStressEvaluator()``.
    settings : SolverSettings | None
        Defaults to ``SolverSettings()``.
    interpolator : str
        Section-law interpolator (see ``beam_opt.section_law.INTERPOLATORS``).
    """

    def __init__(
        self,
        mapdl,
        model: StructuralModel,
        *,
        n_ctrl: int,
        bounds: Mapping[str, tuple[float, float]],
        safety_factor: float,
        f_react_max: float,
        reaction_nodes: list[str],
        evaluator: StressEvaluator | None = None,
        settings: SolverSettings | None = None,
        interpolator: str = "natural_cubic",
    ) -> None:
        unknown = sorted(set(reaction_nodes) - model.node_names())
        if unknown:
            raise ValueError(f"unknown reaction node names: {unknown}")
        self.mapdl = mapdl
        self.model = model
        self.n_ctrl = int(n_ctrl)
        self.bounds = {k: (float(lo), float(hi)) for k, (lo, hi) in bounds.items()}
        self.sigma_adm = model.material.sigma_y / float(safety_factor)
        self.f_react_max = float(f_react_max)
        self.reaction_nodes = list(reaction_nodes)
        self.evaluator = evaluator or RectBeamStressEvaluator()
        self.settings = settings or SolverSettings()
        self.interpolator = interpolator

        self.builder = ModelBuilder(mapdl, jobname=self.settings.jobname)
        self.s_ctrl = np.linspace(0.0, model.beam.mesh.s[-1], self.n_ctrl)
        self.mass_fixed: float = model.fixed_mass()
        self.history: list[Evaluation] = []
        self._last: Evaluation | None = None

    # ------------------------------------------------------------------
    # Design vector
    # ------------------------------------------------------------------

    @property
    def n_vars(self) -> int:
        """Length of the design vector."""
        return self.n_ctrl * len(self.bounds)

    def _blocks(self, x: np.ndarray) -> dict[str, np.ndarray]:
        x = np.asarray(x, dtype=np.float64)
        n = self.n_ctrl
        return {name: x[k * n : (k + 1) * n] for k, name in enumerate(self.bounds)}

    def law_from_x(self, x: np.ndarray) -> SectionLaw:
        """Section law for design vector *x* (control values clipped to ``bounds``)."""
        ctrl = {name: np.clip(v, *self.bounds[name]) for name, v in self._blocks(x).items()}
        return SectionLaw(self.s_ctrl, ctrl, bounds=self.bounds, interpolator=self.interpolator)

    # ------------------------------------------------------------------
    # FEM evaluation (cached)
    # ------------------------------------------------------------------

    def evaluate(self, x: np.ndarray) -> Evaluation:
        """Build, solve and post-process; reuse the last result if *x* is unchanged."""
        x = np.asarray(x, dtype=np.float64)
        last = self._last
        if last is not None and np.allclose(x, last.x, rtol=1e-6, atol=1e-8):
            return last

        law = self.law_from_x(x)
        m = self.mapdl
        ids = self.builder.build(self.model, law)
        solve(m, self.settings)

        with last_results(m):
            sigma = self.evaluator(m, self.model.beam, law)
        with last_results(m):
            reaction = reaction_resultant(m, [ids[n] for n in self.reaction_nodes])

        mass = self.model.beam.mass(law, self.model.material.rho)
        self._last = Evaluation(x.copy(), mass, sigma, reaction)
        self.history.append(self._last)
        return self._last

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------

    def objective(self, x: np.ndarray) -> float:
        """Mass of the optimisable beam [g]."""
        return self.evaluate(x).mass * TON_TO_G

    def total_mass(self, x: np.ndarray) -> float:
        """Mass of the full structure [g]."""
        return self.objective(x) + self.mass_fixed * TON_TO_G

    # ------------------------------------------------------------------
    # Constraints  (all return values >= 0 when satisfied)
    # ------------------------------------------------------------------

    def constraint_bounds(self, x: np.ndarray) -> np.ndarray:
        """Box bounds on the control-point values [mm]."""
        g = []
        for name, v in self._blocks(x).items():
            lo, hi = self.bounds[name]
            for xi in v:
                g.extend([xi - lo, hi - xi])
        return np.array(g)

    def constraint_section_field(self, x: np.ndarray) -> np.ndarray:
        """
        Box bounds on every section parameter at element mid-points [-].

        Evaluated on the *unclipped* law: with clipped values this
        constraint would always be satisfied, while the spline can still
        overshoot the bounds between admissible control points.
        """
        values = self.law_from_x(x).eval(self.model.beam.mesh.s_elem, clip=False)
        t = []
        for name, v in values.items():
            lo, hi = self.bounds[name]
            span = hi - lo
            t.extend((v - lo) / span)
            t.extend((hi - v) / span)
        return np.array(t)

    def constraint_stress(self, x: np.ndarray) -> np.ndarray:
        """Per-element stress constraint σ_VM ≤ σ_adm [-]."""
        return (self.sigma_adm - self.evaluate(x).sigma) / self.sigma_adm

    def constraint_reaction(self, x: np.ndarray) -> float:
        """Reaction constraint F_react ≤ F_react_max [-]."""
        return 1.0 - self.evaluate(x).reaction / self.f_react_max

    def constraints(self) -> list[dict]:
        """Constraints in ``scipy.optimize.minimize`` format."""
        return [
            {"type": "ineq", "fun": self.constraint_section_field},
            {"type": "ineq", "fun": self.constraint_bounds},
            {"type": "ineq", "fun": self.constraint_stress},
            {"type": "ineq", "fun": self.constraint_reaction},
        ]
