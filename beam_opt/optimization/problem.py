"""
beam_opt.optimization.problem
=============================
Objective and constraints of the sizing problem.

The objective is the mass by default; stiffness, a displacement, the peak
stress or the fatigue life can be optimised instead (see
``beam_opt.optimization.objectives``).

The design vector concatenates the control-point values of each section
parameter, in the order of ``bounds``:

    x = [p1_ctrl_0, …, p1_ctrl_{n-1}, p2_ctrl_0, …]

e.g. ``[b_0 … b_{n-1}, h_0 … h_{n-1}]`` for a rectangular section.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import SimpleNamespace

import numpy as np

from ..exceptions import SolutionNotConvergedError
from ..fatigue import FatigueModel
from ..model import LoadCase, ModelBuilder, StructuralModel
from ..native import NativeSolver
from ..post import SectionStressEvaluator, StressEvaluator, last_results, reaction_resultant
from ..section_law import OVERSHOOT_FREE, SectionLaw
from ..solver import SolverSettings, solve
from .history import HistoryWriter
from .objectives import Mass, Objective

logger = logging.getLogger(__name__)

TON_TO_G = 1e6

_DOFS = ("UX", "UY", "UZ", "ROTX", "ROTY", "ROTZ")

#: Number of FEM evaluations kept in the cache (a population-based method
#: asks for the constraints of a whole generation before the objective).
CACHE_SIZE = 1024


@dataclass
class CaseResult:
    """
    Results of one load case of an evaluation.

    Attributes
    ----------
    name : str
        Load case (``"default"`` without load cases).
    sigma : ndarray, shape (n_elem,)
        Equivalent stress per optimisable element [MPa].
    reaction : float
        Resultant reaction on ``reaction_nodes`` [N].
    displacements : ndarray
        Displacement of every entry of ``displacement_limits`` [mm] or [rad].
    watched : dict[tuple[str, str], float]
        Displacements needed by the objective.
    """

    name: str
    sigma: np.ndarray
    reaction: float
    displacements: np.ndarray
    watched: dict


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
        Equivalent stress per optimisable element [MPa]; with several load
        cases, the envelope (largest over the cases).
    reaction : float
        Resultant reaction on ``reaction_nodes`` [N]; with several load
        cases, the largest.
    failed : bool
        ``True`` if the FEM solution did not converge: ``sigma`` and
        ``reaction`` are then penalty values, not results.
    displacements : ndarray
        Displacement of every entry of ``displacement_limits`` [mm] or [rad];
        with several load cases, the one of largest magnitude.
    watched : dict[tuple[str, str], float]
        Displacements needed by the objective, keyed by ``(node, dof)``
        (first load case).
    derived : dict
        Quantities computed from the results on demand (e.g. fatigue lives).
    cases : list[CaseResult]
        Results of every load case.
    """

    x: np.ndarray
    mass: float
    sigma: np.ndarray
    reaction: float
    failed: bool = False
    displacements: np.ndarray = field(default_factory=lambda: np.zeros(0))
    watched: dict = field(default_factory=dict)
    derived: dict = field(default_factory=dict)
    cases: list = field(default_factory=list)


class OptimizationProblem:
    """
    Sizing problem: an objective subject to stress, stiffness, reaction,
    mass, fatigue and technological constraints.

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
    safety_factor : float | None
        Admissible stress = ``material.sigma_y / safety_factor`` [-].
        ``None`` removes the stress constraint (e.g. when local yielding is
        accepted and a fatigue-life constraint governs instead).
    f_react_max : float | None
        Maximum admissible resultant reaction [N]; ``None``: no limit.
    reaction_nodes : list[str] | None
        Names of the nodes whose reactions are summed.
    objective : Objective | None
        What to minimise; defaults to ``Mass()``.
    f_react_min : float | None
        Minimum resultant reaction [N], e.g. the clamping force that a
        spring must still give under an imposed displacement.
    displacement_limits : list[tuple[str, str, float]] | None
        Stiffness constraints ``(node name, DOF, limit)``: the absolute value
        of the displacement (``"UX"``, ``"UY"``, ``"UZ"`` [mm]) or rotation
        (``"ROTX"``, ``"ROTY"``, ``"ROTZ"`` [rad]) must not exceed the limit.
    mass_max : float | None
        Mass budget of the optimisable beam [g]; needed when the objective
        is not the mass.
    fatigue : FatigueModel | None
        Load cycle and material for fatigue lives (objective ``FatigueLife``
        and constraint ``life_min``).
    life_min : float | None
        Required fatigue life of every element [cycles, or repetitions of the
        spectrum / of the load cases].
    load_cases : list[LoadCase] | None
        Independent load situations (see ``beam_opt.model.LoadCase``): every
        design is analysed once per case, every constraint must hold in every
        case, fatigue damage is summed over the cases.  ``None``: the loads
        of the model are the only case.
    evaluator : StressEvaluator | None
        Defaults to ``SectionStressEvaluator()``, valid for every ``Section``.
    settings : SolverSettings | None
        Defaults to ``SolverSettings()``.
    interpolator : str
        Section-law interpolator (see ``beam_opt.section_law.INTERPOLATORS``).
    on_failure : str
        What to do when the FEM solution does not converge: ``"penalty"``
        (default) records the design as failed, so the optimiser moves away
        from it and it is never returned as optimum; ``"raise"`` propagates
        ``SolutionNotConvergedError``.
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
        safety_factor: float | None,
        f_react_max: float | None = None,
        reaction_nodes: list[str] | None = None,
        objective: Objective | None = None,
        f_react_min: float | None = None,
        displacement_limits: list[tuple[str, str, float]] | None = None,
        mass_max: float | None = None,
        fatigue: FatigueModel | None = None,
        life_min: float | None = None,
        load_cases: list[LoadCase] | None = None,
        evaluator: StressEvaluator | None = None,
        settings: SolverSettings | None = None,
        interpolator: str = "natural_cubic",
        on_failure: str = "penalty",
        failure_penalty: float = 10.0,
        history_file: str | None = None,
    ) -> None:
        reaction_nodes = list(reaction_nodes or [])
        unknown = sorted(set(reaction_nodes) - model.node_names())
        if unknown:
            raise ValueError(f"unknown reaction node names: {unknown}")
        if (f_react_max is not None or f_react_min is not None) and not reaction_nodes:
            raise ValueError("reaction limits need reaction_nodes")
        if on_failure not in ("penalty", "raise"):
            raise ValueError(f"on_failure must be 'penalty' or 'raise', got {on_failure!r}")
        if life_min is not None and fatigue is None:
            raise ValueError("life_min needs fatigue=FatigueModel(...)")
        self.mapdl = mapdl
        self.native = isinstance(mapdl, NativeSolver)
        self.model = model
        self.n_ctrl = int(n_ctrl)
        self.bounds = {k: (float(lo), float(hi)) for k, (lo, hi) in bounds.items()}
        self.stress_limited = safety_factor is not None
        self.sigma_adm = model.material.sigma_y / float(safety_factor or 1.0)
        self.f_react_max = None if f_react_max is None else float(f_react_max)
        self.reaction_nodes = reaction_nodes
        self.f_react_min = None if f_react_min is None else float(f_react_min)
        self.displacement_limits = [
            (str(node), str(dof).upper(), float(lim))
            for node, dof, lim in displacement_limits or []
        ]
        for node, dof, lim in self.displacement_limits:
            if node not in model.node_names() or dof not in _DOFS or lim <= 0.0:
                raise ValueError(f"invalid displacement limit {(node, dof, lim)}")
        self.mass_max = None if mass_max is None else float(mass_max)
        self.fatigue = fatigue
        self.life_min = None if life_min is None else float(life_min)
        self.evaluator = evaluator or SectionStressEvaluator()
        if self.native and not hasattr(self.evaluator, "from_forces"):
            raise TypeError(
                "with NativeSolver the stress evaluator must provide from_forces() "
                "(e.g. SectionStressEvaluator)"
            )
        self.load_cases = list(load_cases or [])
        names = [c.name for c in self.load_cases]
        if len(set(names)) != len(names):
            raise ValueError(f"load case names must be unique: {names}")
        self.case_models = (
            [c.apply_to(model) for c in self.load_cases] if self.load_cases else [model]
        )
        self.objective_fn = objective or Mass()
        self.objective_fn.check(self)
        self.settings = settings or SolverSettings()
        self.interpolator = interpolator
        self.on_failure = on_failure
        self.failure_penalty = float(failure_penalty)

        self.builder = ModelBuilder(mapdl, jobname=self.settings.jobname)
        self.s_ctrl = np.linspace(0.0, model.beam.mesh.s[-1], self.n_ctrl)
        self.mass_fixed: float = model.fixed_mass()
        self.history: list[Evaluation] = []
        self._cache: OrderedDict[bytes, Evaluation] = OrderedDict()
        self._failure_value: float | None = None
        self._watch = list(dict.fromkeys(self.objective_fn.watch(self)))
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
            cases = [
                self._solve(law, m, name) for m, name in zip(self.case_models, self.case_names)
            ]
        except SolutionNotConvergedError as err:
            if self.on_failure == "raise":
                raise
            logger.warning("FEM evaluation %d failed: %s", len(self.history) + 1, err)
            cases = [
                CaseResult(
                    name,
                    np.full(self.model.beam.mesh.n_elem, self.failure_penalty * self.sigma_adm),
                    self.failure_penalty * (self.f_react_max or 1.0),
                    self.failure_penalty * self._disp_limits,
                    dict.fromkeys(self._watch, np.nan),
                )
                for name in self.case_names
            ]
            failed = True
        else:
            failed = False
        ev = self._envelope(x.copy(), mass, cases, failed)

        self.history.append(ev)
        self._cache[key] = ev
        if len(self._cache) > CACHE_SIZE:
            self._cache.popitem(last=False)
        if self._writer:
            value = np.nan if ev.failed else self.objective_fn.value(self, ev)
            self._writer.write(ev, self.max_violation(ev), value)
        return ev

    @property
    def case_names(self) -> list[str]:
        """Names of the load cases (``["default"]`` without load cases)."""
        return [c.name for c in self.load_cases] or ["default"]

    @staticmethod
    def _envelope(x, mass, cases: list[CaseResult], failed: bool) -> Evaluation:
        sigma = np.max([c.sigma for c in cases], axis=0)
        reaction = float(max(c.reaction for c in cases))
        if cases[0].displacements.size:
            stack = np.array([c.displacements for c in cases])
            worst = np.argmax(np.abs(stack), axis=0)
            disp = stack[worst, np.arange(stack.shape[1])]
        else:
            disp = cases[0].displacements
        return Evaluation(
            x, mass, sigma, reaction, failed=failed, displacements=disp,
            watched=cases[0].watched, cases=cases,
        )  # fmt: skip

    @property
    def _disp_limits(self) -> np.ndarray:
        return np.array([lim for _, _, lim in self.displacement_limits])

    def _solve(self, law: SectionLaw, model: StructuralModel, name: str) -> CaseResult:
        """Element stresses [MPa], reaction [N], limited and watched displacements of one case."""
        limits = [(node, dof) for node, dof, _ in self.displacement_limits]
        wanted = list(dict.fromkeys(limits + self._watch))
        if self.native:
            res = self.mapdl.analyze(model, law, self.reaction_nodes)
            sigma = self.evaluator.from_forces(res.forces, model.beam, law)
            values = {
                (n, d): float(res.displacements[res.node_index[n], _DOFS.index(d)])
                for n, d in wanted
            }
            reaction = res.reaction
        else:
            m = self.mapdl
            ids = self.builder.build(model, law)
            solve(m, self.settings)
            with last_results(m):
                sigma = self.evaluator(m, model.beam, law)
            with last_results(m):
                reaction = (
                    reaction_resultant(m, [ids[n] for n in self.reaction_nodes])
                    if self.reaction_nodes
                    else 0.0
                )
                values = {
                    (n, d): float(
                        m.get_value("NODE", ids[n], "ROT" if d.startswith("ROT") else "U", d[-1])
                    )
                    for n, d in wanted
                }
        disp = np.array([values[w] for w in limits])
        return CaseResult(name, sigma, float(reaction), disp, {w: values[w] for w in self._watch})

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------

    def mass_g(self, x: np.ndarray) -> float:
        """Mass of the optimisable beam [g] (geometric: no FEM solution needed)."""
        return self.model.beam.mass(self.law_from_x(x), self.model.material.rho) * TON_TO_G

    def objective(self, x: np.ndarray) -> float:
        """
        Value of the objective at *x* (in the unit of the objective).

        For a failed FEM evaluation, a value clearly worse than the design at
        the upper bounds.
        """
        if not self.objective_fn.needs_fem:
            law = self.law_from_x(x)
            proxy = SimpleNamespace(x=x, mass=self.model.beam.mass(law, self.model.material.rho))
            return self.objective_fn.value(self, proxy)
        return self.objective_of(self.evaluate(x))

    def objective_of(self, ev: Evaluation) -> float:
        """Value of the objective for an evaluated design."""
        if ev.failed:
            return self._failed_objective()
        return self.objective_fn.value(self, ev)

    def objective_vector(self, x: np.ndarray) -> np.ndarray:
        """
        Per-element values of a min-max objective at *x* (see ``Objective.values``).

        For a failed FEM evaluation, every value is the failure value.
        """
        ev = self.evaluate(x)
        values = np.asarray(self.objective_fn.values(self, ev), dtype=float)
        if ev.failed:
            return np.full(values.shape, self._failed_objective())
        return values

    @property
    def is_minmax(self) -> bool:
        """Whether the objective is the worst of several values (elements, load cases)."""
        own = getattr(self.objective_fn, "is_minmax", None)
        if callable(own):
            return bool(own(self))
        return type(self.objective_fn).values is not Objective.values

    def _failed_objective(self) -> float:
        if self._failure_value is None:
            self._failure_value = 1e30  # guards the evaluation below
            ref = self.evaluate(self.upper)
            base = 1.0 if ref.failed else abs(self.objective_fn.value(self, ref))
            self._failure_value = self.failure_penalty * max(base, 1.0)
        return self._failure_value

    def objective_scale(self) -> float:
        """Magnitude of the objective, used to scale it for the algorithms."""
        value = abs(self.objective(self.upper))
        return value if np.isfinite(value) and value > 1e-12 else 1.0

    def total_mass(self, x: np.ndarray) -> float:
        """Mass of the full structure [g]."""
        return self.mass_g(x) + self.mass_fixed * TON_TO_G

    @property
    def life_unit(self) -> str:
        """Unit of the fatigue lives: cycles, or repetitions of the spectrum / of the cases."""
        if not self.load_cases:
            return self.fatigue.unit
        for case in self.load_cases:
            if case.spectrum is not None:
                return f"{case.spectrum.name}s"
        return self.fatigue.unit if self.fatigue.spectrum is not None else "repetitions"

    def lives(self, ev: Evaluation) -> np.ndarray:
        """
        Fatigue life of every optimisable element (needs ``fatigue``), in ``life_unit``.

        With load cases, Miner damage is summed over the cases: the life is
        the number of repetitions of the whole set of cases.  Cycles formed
        by the transitions between different cases are not counted.
        """
        if self.fatigue is None:
            raise ValueError("no FatigueModel: pass OptimizationProblem(fatigue=...)")
        if "lives" not in ev.derived:
            if not self.load_cases:
                ev.derived["lives"] = self.fatigue.life(ev.sigma)
            else:
                damage = np.zeros(self.model.beam.mesh.n_elem)
                for case, res in zip(self.load_cases, ev.cases):
                    if case.spectrum is not None:
                        model = replace(self.fatigue, spectrum=case.spectrum)
                        damage = damage + model.damage(res.sigma)
                    else:
                        damage = damage + case.cycles * self.fatigue.damage(res.sigma)
                cap = 0.5e12
                ev.derived["lives"] = np.where(
                    damage > 0.0, 1.0 / np.maximum(damage, 1.0 / cap), cap
                )
        return ev.derived["lives"]

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
        """
        Per-element stress constraint σ_VM ≤ σ_adm [-] (empty without a safety factor).

        With load cases, on the envelope of the stresses.
        """
        if not self.stress_limited:
            return np.zeros(0)
        return (self.sigma_adm - self.evaluate(x).sigma) / self.sigma_adm

    def constraint_reaction(self, x: np.ndarray) -> float:
        """Reaction constraint F_react ≤ F_react_max [-] (1 if no maximum is set)."""
        if self.f_react_max is None:
            return 1.0
        return 1.0 - self.evaluate(x).reaction / self.f_react_max

    def constraint_reaction_min(self, x: np.ndarray) -> float:
        """Reaction constraint F_react ≥ F_react_min [-] (1 if no minimum is set)."""
        if self.f_react_min is None:
            return 1.0
        return self.evaluate(x).reaction / self.f_react_min - 1.0

    def constraint_displacement(self, x: np.ndarray) -> np.ndarray:
        """Stiffness constraints |u| ≤ limit, one per entry of ``displacement_limits`` [-]."""
        return 1.0 - np.abs(self.evaluate(x).displacements) / self._disp_limits

    def constraint_mass(self, x: np.ndarray) -> float:
        """Mass budget m ≤ mass_max [-] (1 if no budget is set)."""
        if self.mass_max is None:
            return 1.0
        return 1.0 - self.mass_g(x) / self.mass_max

    def constraint_life(self, x: np.ndarray) -> np.ndarray:
        """Fatigue constraint N ≥ life_min per element, in decades of life [-]."""
        if self.life_min is None:
            return np.zeros(0)
        return np.log10(self.lives(self.evaluate(x))) - np.log10(self.life_min)

    def _response_constraints(self, ev: Evaluation) -> list:
        """Every constraint that depends on the FEM results of *ev* (in every load case)."""
        g = []
        for case in ev.cases:
            if self.stress_limited:
                g.append((self.sigma_adm - case.sigma) / self.sigma_adm)
            if self.f_react_max is not None:
                g.append([1.0 - case.reaction / self.f_react_max])
            if self.f_react_min is not None:
                g.append([case.reaction / self.f_react_min - 1.0])
            if self.displacement_limits:
                g.append(1.0 - np.abs(case.displacements) / self._disp_limits)
        if self.life_min is not None:
            g.append(np.log10(self.lives(ev)) - np.log10(self.life_min))
        return g

    @property
    def needs_field_constraint(self) -> bool:
        """Whether the section law can leave the bounds between control points."""
        return self.interpolator not in OVERSHOOT_FREE

    def constraint_vector(self, x: np.ndarray) -> np.ndarray:
        """
        All non-bound constraints in one vector (>= 0 when satisfied) [-].

        Stress per element, reaction limits, displacement limits, mass
        budget, fatigue life and, only for interpolators that can overshoot,
        the section field.  The box bounds on the design variables are not
        included: pass ``lower``/``upper`` to the optimiser.
        """
        parts = self._response_constraints(self.evaluate(x))
        if self.mass_max is not None:
            parts.append([self.constraint_mass(x)])
        if self.needs_field_constraint:
            parts.append(self.constraint_section_field(x))
        if not parts:  # bounds only: a constant, always satisfied constraint
            return np.ones(1)
        return np.concatenate(parts)

    def max_violation(self, ev: Evaluation) -> float:
        """
        Largest constraint violation of an evaluated design [-]; 0 if feasible.

        Includes the box bounds, normalised by their range.  A failed FEM
        evaluation is always infeasible.
        """
        if ev.failed:
            return self.failure_penalty
        span = self.upper - self.lower
        g = [
            *self._response_constraints(ev),
            [self.constraint_mass(ev.x)],
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
