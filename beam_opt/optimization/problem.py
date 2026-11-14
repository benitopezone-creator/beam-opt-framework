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
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from ..exceptions import SolutionNotConvergedError
from ..model import ModelBuilder, StructuralModel
from ..native import NativeSolver
from ..post import SectionStressEvaluator, StressEvaluator, last_results, reaction_resultant
from ..section_law import OVERSHOOT_FREE, SectionLaw
from ..solver import SolverSettings, solve
from .history import HistoryWriter

logger = logging.getLogger(__name__)

TON_TO_G = 1e6

#: Number of FEM evaluations kept in the cache (a population-based method
#: asks for the constraints of a whole generation before the objective).
CACHE_SIZE = 1024


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
    failed : bool
        ``True`` if the FEM solution did not converge: ``sigma`` and
        ``reaction`` are then penalty values, not results.
    """

    x: np.ndarray
    mass: float
    sigma: np.ndarray
    reaction: float
    failed: bool = False


class OptimizationProblem:
    """
    Minimum mass subject to stress, reaction and technological bounds.

    Parameters
    ----------
    mapdl
        MAPDL session (or a test double), owned by the caller, or a
        ``beam_opt.native.NativeSolver`` to run without ANSYS.
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
        Defaults to ``SectionStressEvaluator()``, valid for every ``Section``.
    settings : SolverSettings | None
        Defaults to ``SolverSettings()``.
    interpolator : str
        Section-law interpolator (see ``beam_opt.section_law.INTERPOLATORS``).
    on_failure : str
        What to do when the FEM solution does not converge: ``"penalty"``
        (default) records the design as failed, with a clearly infeasible
        stress and reaction, so the optimiser moves away from it;
        ``"raise"`` propagates ``SolutionNotConvergedError``.
    failure_penalty : float
        Stress and reaction of a failed evaluation, as multiples of the
        admissible values [-]: clearly infeasible, but finite.
    history_file : str | None
        CSV file receiving one row per FEM evaluation (appended if it
        exists); see ``beam_opt.optimization.history``.
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
        on_failure: str = "penalty",
        failure_penalty: float = 10.0,
        history_file: str | None = None,
    ) -> None:
        unknown = sorted(set(reaction_nodes) - model.node_names())
        if unknown:
            raise ValueError(f"unknown reaction node names: {unknown}")
        if on_failure not in ("penalty", "raise"):
            raise ValueError(f"on_failure must be 'penalty' or 'raise', got {on_failure!r}")
        self.mapdl = mapdl
        self.native = isinstance(mapdl, NativeSolver)
        self.model = model
        self.n_ctrl = int(n_ctrl)
        self.bounds = {k: (float(lo), float(hi)) for k, (lo, hi) in bounds.items()}
        self.sigma_adm = model.material.sigma_y / float(safety_factor)
        self.f_react_max = float(f_react_max)
        self.reaction_nodes = list(reaction_nodes)
        self.evaluator = evaluator or SectionStressEvaluator()
        if self.native and not hasattr(self.evaluator, "from_forces"):
            raise TypeError(
                "with NativeSolver the stress evaluator must provide from_forces() "
                "(e.g. SectionStressEvaluator)"
            )
        self.settings = settings or SolverSettings()
        self.interpolator = interpolator
        self.on_failure = on_failure
        self.failure_penalty = float(failure_penalty)

        self.builder = ModelBuilder(mapdl, jobname=self.settings.jobname)
        self.s_ctrl = np.linspace(0.0, model.beam.mesh.s[-1], self.n_ctrl)
        self.mass_fixed: float = model.fixed_mass()
        self.history: list[Evaluation] = []
        self._cache: OrderedDict[bytes, Evaluation] = OrderedDict()
        self._writer = HistoryWriter(history_file, self.variable_names()) if history_file else None

    # ------------------------------------------------------------------
    # Design vector
    # ------------------------------------------------------------------

    @property
    def n_vars(self) -> int:
        """Length of the design vector."""
        return self.n_ctrl * len(self.bounds)

    @property
    def lower(self) -> np.ndarray:
        """Lower bound of every design variable [mm]."""
        return np.concatenate([np.full(self.n_ctrl, lo) for lo, _ in self.bounds.values()])

    @property
    def upper(self) -> np.ndarray:
        """Upper bound of every design variable [mm]."""
        return np.concatenate([np.full(self.n_ctrl, hi) for _, hi in self.bounds.values()])

    def variable_names(self) -> list[str]:
        """Names of the design variables: ``<parameter>_<control point>``."""
        return [f"{name}_{k}" for name in self.bounds for k in range(self.n_ctrl)]

    def _blocks(self, x: np.ndarray) -> dict[str, np.ndarray]:
        x = np.asarray(x, dtype=np.float64)
        n = self.n_ctrl
        return {name: x[k * n : (k + 1) * n] for k, name in enumerate(self.bounds)}

    def law_from_x(self, x: np.ndarray) -> SectionLaw:
        """
        Section law for design vector *x*.

        Keeping the control values admissible is the job of the optimiser
        bounds (``lower``/``upper``); they are still clipped here as a safety
        net, so that MAPDL never receives an inadmissible section.
        """
        ctrl = {name: np.clip(v, *self.bounds[name]) for name, v in self._blocks(x).items()}
        return SectionLaw(self.s_ctrl, ctrl, bounds=self.bounds, interpolator=self.interpolator)

    # ------------------------------------------------------------------
    # FEM evaluation (cached)
    # ------------------------------------------------------------------

    def evaluate(self, x: np.ndarray) -> Evaluation:
        """
        Build, solve and post-process; reuse a cached result if *x* was seen.

        Raises
        ------
        SolutionNotConvergedError
            If the FEM solution does not converge and ``on_failure="raise"``.
        """
        x = np.asarray(x, dtype=np.float64)
        key = np.round(x, 9).tobytes()
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached

        law = self.law_from_x(x)
        mass = self.model.beam.mass(law, self.model.material.rho)
        try:
            sigma, reaction = self._solve(law)
        except SolutionNotConvergedError as err:
            if self.on_failure == "raise":
                raise
            logger.warning("FEM evaluation %d failed: %s", len(self.history) + 1, err)
            ev = Evaluation(
                x.copy(),
                mass,
                np.full(self.model.beam.mesh.n_elem, self.failure_penalty * self.sigma_adm),
                self.failure_penalty * self.f_react_max,
                failed=True,
            )
        else:
            ev = Evaluation(x.copy(), mass, sigma, reaction)

        self.history.append(ev)
        self._cache[key] = ev
        if len(self._cache) > CACHE_SIZE:
            self._cache.popitem(last=False)
        if self._writer:
            self._writer.write(ev, self.max_violation(ev))
        return ev

    def _solve(self, law: SectionLaw) -> tuple[np.ndarray, float]:
        """Element stresses [MPa] and reaction [N] for section law *law*."""
        if self.native:
            res = self.mapdl.analyze(self.model, law, self.reaction_nodes)
            return self.evaluator.from_forces(res.forces, self.model.beam, law), res.reaction

        m = self.mapdl
        ids = self.builder.build(self.model, law)
        solve(m, self.settings)
        with last_results(m):
            sigma = self.evaluator(m, self.model.beam, law)
        with last_results(m):
            reaction = reaction_resultant(m, [ids[n] for n in self.reaction_nodes])
        return sigma, reaction

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------

    def objective(self, x: np.ndarray) -> float:
        """Mass of the optimisable beam [g] (geometric: no FEM solution needed)."""
        return self.model.beam.mass(self.law_from_x(x), self.model.material.rho) * TON_TO_G

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

    @property
    def needs_field_constraint(self) -> bool:
        """Whether the section law can leave the bounds between control points."""
        return self.interpolator not in OVERSHOOT_FREE

    def constraint_vector(self, x: np.ndarray) -> np.ndarray:
        """
        All non-bound constraints in one vector (>= 0 when satisfied) [-].

        Stress per element, reaction and, only for interpolators that can
        overshoot, the section field.  The box bounds on the design variables
        are not included: pass ``lower``/``upper`` to the optimiser.
        """
        parts = [self.constraint_stress(x), [self.constraint_reaction(x)]]
        if self.needs_field_constraint:
            parts.append(self.constraint_section_field(x))
        return np.concatenate(parts)

    def max_violation(self, ev: Evaluation) -> float:
        """
        Largest constraint violation of an evaluated design [-]; 0 if feasible.

        Includes the box bounds, normalised by their range.
        """
        span = self.upper - self.lower
        g = [
            (self.sigma_adm - ev.sigma) / self.sigma_adm,
            [1.0 - ev.reaction / self.f_react_max],
            (ev.x - self.lower) / span,
            (self.upper - ev.x) / span,
        ]
        if self.needs_field_constraint:
            g.append(self.constraint_section_field(ev.x))
        return float(max(0.0, -np.min(np.concatenate(g))))

    def constraints(self) -> list[dict]:
        """
        Non-bound constraints in ``scipy.optimize.minimize`` format.

        Prefer ``beam_opt.optimization.optimize``, which also scales the
        variables and passes the bounds.
        """
        return [{"type": "ineq", "fun": self.constraint_vector}]
