# Changelog

## [0.1.0] – 2026-04

### Added
- `beam_opt` installable package extracted from the monolithic `ott_aihelmet.py`.
- `beam_opt/stress/` sub-package with `AbstractStressEvaluator` interface and
  `RectBeamStressEvaluator` implementation (BEAM188 rectangular cross-section,
  von Mises with Saint-Venant torsion + Jourawski shear).
- `BeamModel`: geometry-agnostic FEM builder; all BC node sets are derived
  from the passed-in data – no hard-coded node numbers.
- `BeamAnalysis`: solver + post-processing orchestrator, delegates stress
  computation to the pluggable evaluator.
- `BeamOptimizer`: COBYLA wrapper with `run()` convenience method.
- `MapdlManager`: MAPDL singleton with `atexit` shutdown hook (fixes batch
  process not closing on optimisation end or crash).
- `SolverSettings` dataclass centralises all MAPDL knobs.
- `SectionLaw`: generic CubicSpline section-distribution class, bounds
  configurable per-instance (not global constants).
- `examples/aihelmet/config_v05.py`: V05 geometry extracted from the original
  `helmet_config.py`; shell-patch node generation moved to a local helper
  function with no side-effects.
- `examples/aihelmet/run_optimization.py`: CLI entry point replacing the
  `__main__` block; supports `--config`, `--n-ctrl`, `--nproc`,
  `--run-location`, `--save-plots`, `--no-plots`, `--maxiter`.
- `pyproject.toml` for `pip install -e .` workflow.

### Changed
- `stress_field()` extracted from `HelmetAnalysis` into its own module;
  swapping cross-section shape no longer requires touching the analysis or
  optimisation code.
- `RHO`, `E`, `NU`, `SIGMA_S`, `FS`, `F_REACT_MAX`, `UX_IMP`, `UZ_IMP`
  consolidated in `beam_opt/config.py`; `UZ_IMP` sign is now per-config
  (V05 = +90, V07 = −90).
- Fixed elements use `ABC` / `abstractmethod` instead of bare `raise
  NotImplementedError`.

### Fixed
- MAPDL batch process left running after optimisation: resolved via `atexit`
  in `MapdlManager` and explicit `MapdlManager.close()` at end of `main()`.
- Symmetry nodes `[51, 52, 53]` were hard-coded in `HelmetModel.build()`;
  now inferred automatically from `additional_nodes` key range.
- `max_reaction()` referenced node `50` and `1` by literal integers; now
  uses `model.n_nodes` and the shell-patch ID threshold.
