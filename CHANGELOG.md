# Changelog

## [Unreleased] – Built-in solver (the framework without ANSYS)

### Added
- `beam_opt.native.NativeSolver`: linear 3-D beam solver (two-node Timoshenko elements,
  BEAM188 default orientation and internal-force convention) used in place of the MAPDL
  session: `OptimizationProblem(NativeSolver(), model, ...)`. Supports `FixedBeam`,
  `Clamp`, `Symmetry`, `ImposedDisplacement`, `NodalForce`, `RigidRegion`, redundant
  constraints, and shells that belong to a rigid region.
- `SectionStressEvaluator.from_forces()`: stress evaluation from internal forces, shared
  by both solvers. `Section.shear_correction`.
- `tests/test_native.py`: closed-form checks (cantilever, fixed-fixed and propped beams,
  imposed displacement, symmetry, rigid arm, quarter-circle arcs in and out of plane) and
  the curved frame with rigid plate.
- Example 01 runs with the built-in solver by default (`--solver native | mapdl`);
  `docs/img/cantilever_comparison.png` is its actual output.

### Limits
- Linear analysis only; deformable shells need MAPDL. Not yet compared with MAPDL results.

## [Unreleased] – Phase 3 (stresses for generic sections)

### Added
- `sections.StressPoints` and `Section.stress_points()`: each shape describes its check
  points (position, shear stress per unit torque and per unit shear force).
- `post.SectionStressEvaluator`: von Mises stress for any `Section` from the BEAM188
  internal forces (SMISC 1–6 and 14–19); it is the new default of `OptimizationProblem`.
- `sections.CircSection` (`CSOLID`), `sections.TubeSection` (`CTUBE`) and
  `sections.BoxSection` (`HREC`, thin-walled torsion and shear).
- Tests against textbook formulas for every shape, comparison with the legacy evaluator
  and a full optimisation of a circular cantilever against its closed-form optimum.

### Changed
- `Section` subclasses must implement `stress_points()`.
- Rectangle, with respect to `RectBeamStressEvaluator` (kept, no longer the default): no
  torsion stress at the corners (lower stress under biaxial bending with torsion); torsion
  and shear stresses added linearly at the mid-sides (higher stress under shear with
  torsion); no mid-length candidate. Identical results without torsion.

### Fixed
- COBYLA and Nelder-Mead started on the upper bounds built their first simplex outside the
  admissible box, where the clipped sections give a flat response, and could stop at a
  poor design while reporting success. The first simplex is now kept inside the bounds.
  This was the actual cause of the COBYLA stall listed under Phase 4.

### Not done
- Generic shape read from the MAPDL section mesher and the comparison with the MAPDL
  section stresses: both need a MAPDL installation.

## [Unreleased] – Phase 4 (robust optimisation, selectable algorithm)

### Added
- `optimization.optimize(problem, method=...)`: one entry point for every algorithm, chosen
  by name from the `METHODS` registry — `"cobyla"`, `"slsqp"` (finite-difference
  gradients), `"nelder-mead"` (penalised constraints) and `"differential-evolution"`
  (evolutionary global search). Uniform `max_evals` budget, `seed` for reproducible runs.
- Variables scaled to [0, 1] and mass scaled by the mass at the upper bounds; box bounds
  passed to the algorithm as `bounds`.
- The result is the lightest feasible design among those evaluated; `success=False` with an
  explicit message when no design is feasible.
- `OptimizationProblem(on_failure="penalty" | "raise", failure_penalty=...)`: a
  non-converged FEM solution becomes a flagged, penalised `Evaluation(failed=True)`.
- `OptimizationProblem(history_file=...)` and `optimization.history` (`read_history`,
  `best_design`): CSV log of every FEM evaluation and restart from it.
- `OptimizationProblem.lower`, `upper`, `variable_names()`, `constraint_vector()`,
  `max_violation()`.
- `slow` pytest marker.

### Changed
- `OptimizationProblem.constraints()` no longer contains the box bounds, and contains the
  section-field constraint only for interpolators that can overshoot (`natural_cubic`).
- `OptimizationProblem.objective()` is geometric and does not trigger a FEM solution.
- The evaluation cache keeps the last 1024 designs instead of the last one.
- `CobylaDriver` is a thin wrapper of `optimize`; its `rhobeg` is now a fraction of the
  bounds range (default 0.1) and the default FEM failure policy is `"penalty"`.

### Fixed
- COBYLA stopped far from the optimum while reporting success with many control points
  and elements (cantilever, `n_ctrl=16`, `n_elem ≥ 100`: +80 % mass). Bounds, scaling and
  the removal of redundant constraints made the case pass, but the cause was the starting
  point on the upper bounds: see the fix under Phase 3.

## [Unreleased] – Phase 6, first part (cantilever example)

### Added
- `examples/01_cantilever/`: minimum-mass cantilever with a tip load and constant width,
  compared with the closed-form beam of uniform strength (`h*(s) ∝ sqrt(L − s)`, 2/3 of
  the prismatic mass); command-line entry point, report and comparison plot.
- `tests/fakes.py`: `StaticCantileverMapdl`, a closed-form stand-in for MAPDL on a
  statically determinate cantilever, so a full optimisation loop runs without ANSYS.
- `tests/test_example_cantilever.py`: the example reaches the closed-form optimum
  (mass within 5 %, stress constraint active); a `mapdl`-marked test repeats the
  comparison with the real solver (written, not executed yet).
- `pyproject.toml`: `authors`.

## [Unreleased] – Phase 2 (user-imposed discretisation)

### Added
- `BeamPath(points, kind="spline" | "polyline")`: parametric cubic spline through the
  definition points, reparameterised by true arc length (Gauss–Legendre quadrature);
  `s_knots`, `point_at(s)`, `tangent_at(s)`.
- `geometry.discretize(path, n_elem, anchors)`: resampling in *s* with a node exactly at
  every named anchor and element sizes balanced between hard points.
- `BeamMesh(coords, anchors, s=...)`: explicit nodal arc length.
- Tests written before the implementation (`tests/test_discretize.py`: known arc length,
  anchors on nodes, monotone *s*, second-order convergence of the length), mesh
  independence of a synthetic curved frame (`tests/test_mesh_independence.py`) and a
  MAPDL test with a load on an anchor and odd `n_elem`.
- `docs/anchors-e-discretizzazione.md`: anchors, conversion from node IDs, mesh studies.

### Changed
- `BeamPath` defaults to `kind="spline"`; `BeamPath.s` is replaced by `s_knots` and
  `lengths` is removed (use `BeamMesh.lengths`).
- With `discretize`, `mesh.s[-1]` is the arc length of the path, so the section-law
  control points (`OptimizationProblem.s_ctrl`) and the number of design variables do not
  depend on `n_elem`.
- `BeamMesh.from_points` (one element per segment) keeps the 0.1 behaviour: the Phase 1
  regression reference is unchanged.

## [Unreleased] – Phase 1 (modularisation and testability)

Breaking: the 0.1 API (`BeamModel`, `BeamAnalysis`, `BeamOptimizer`, `MapdlManager`,
`beam_opt.config`) is replaced; see `docs/phase1-plan.md` for the old → new map.

### Added
- `materials.Material`, `geometry.BeamPath`, `geometry.BeamMesh` (named anchors,
  `"start"`/`"end"` always defined).
- `sections.Section` interface and `sections.RectSection`.
- `model.components` (`OptimizableBeam`, `FixedBeam`, `FixedShell`, `StructuralModel`),
  `model.bcs` (`Clamp`, `Symmetry`, `ImposedDisplacement`, `RigidRegion`, `NodalForce`)
  and `model.builder.ModelBuilder`; every node is referred to by name.
- `solver.solve` / `solver.SolverSettings`, `post.extract` (`beam_smisc`,
  `reaction_resultant`, `last_results`), `post.stress.StressEvaluator`.
- `optimization.OptimizationProblem` (cached evaluations, history) and `CobylaDriver`.
- `session.MapdlSession` context manager, the only place that launches MAPDL.
- `SectionLaw` interpolators `"pchip"` and `"linear"` (no overshoot).
- Tests: `tests/fakes.py` (`FakeMapdl`, `canonical_model`), golden regression against the
  0.1 code (`tests/test_phase1_regression.py`), exact build sequence of a minimal model,
  architecture checks of CLAUDE.md rules 3/4/6, MAPDL cantilever tests vs PL³/3EI.
- `.gitattributes` (LF line endings); pytest `pythonpath` for test helpers.

### Changed
- Boundary conditions formerly hard-wired in `BeamModel._apply_bcs` are explicit BC
  objects in the configuration.
- Fixed-element section numbers are allocated after `n_elem` (no collision with the
  former 301/1000 when `n_elem` > 300).
- SMISC items are extracted with `*VGET` (`mapdl.get_array`) instead of one
  `get_value` per element.
- `SectionLaw` takes named parameters (`{"b": ..., "h": ...}`) and optional bounds; no
  package-level default bounds.
- `CobylaDriver.run()` default starting point is the mid-range of the bounds (was 1.5 mm).
- `print_report` no longer draws plots; call `report.plot_results` explicitly.
- Builds verified unchanged against the 0.1 code on a synthetic curved frame
  (model written to MAPDL, masses, all constraints, stresses: relative difference < 1e-12).

### Fixed
- Rectangular torsion: J cubed the long side and τ_T used t/2 instead of t, so the
  torsional shear stress was underestimated (−26 % for a square, ≈ 12× for an aspect ratio of 2.5).
  Now Roark's formulas, checked against tabulated coefficients. **Stresses with
  significant torsion increase.**
- All `FixedBeam`s shared section 301 by default, so in MAPDL every fixed beam took the
  last defined section; each fixed element now has its own section number.

### Removed
- `beam_opt/config.py`, `mapdl_manager.py`, `model.py`, `fixed_elements.py`,
  `analysis.py`, `optimization.py`, `report.py`, `stress/` (replaced as listed above).

## [Unreleased] – Phase 0 (safety fixes and packaging)

### Added
- `tests/` pytest suite with a `FakeMapdl` test double (`tests/conftest.py`);
  `mapdl` marker registered in `pyproject.toml` for tests that need ANSYS.
- `beam_opt.exceptions.SolutionNotConvergedError` (subclass of `RuntimeError`).
- `beam_opt.analysis.check_convergence()`: checks the SOLU convergence flag and
  that the last results set reaches time 1.0.
- `beam_opt.fixed_elements.quad_area_3d()`.
- `SectionLaw.eval(s, clip=False)` to obtain the raw spline values.
- `LICENSE` (MIT).

### Changed
- `BeamAnalysis.solve()` raises `SolutionNotConvergedError` when the RST is
  missing or the solution stops before the end of the load step.
- `constraint_section_field` is evaluated on the unclipped spline, so it detects
  overshoot between admissible control points (it was always satisfied before).
  Strategy for replacing the clipping in Phase 4 is documented in `ROADMAP.md`.
- README quick start reflects the actual repository content (no `examples/` yet)
  and states the supported Python range (3.10–3.14, limited by `ansys-mapdl-core`).

### Fixed
- `pyproject.toml`: build backend `setuptools.backends.legacy:build` does not
  exist; now `setuptools.build_meta`, so `pip install -e ".[dev]"` works.
- `RectBeamStressEvaluator`: SMISC 5/6 labels were swapped (5 = SFZ, 6 = SFY);
  node-J torque and shear now read SMISC 17–19 instead of reusing node-I values.
  Results at node I are unchanged; results at node J change where TQ, SFY or SFZ
  vary along the element.
- `FixedShell.estimate_mass`: the area was the XY projection (shoelace formula);
  now the 3-D area of the two triangles, independent of the shell orientation.

## [0.1.0] – 2026-04

First version: a beam-optimisation prototype extracted from a single application
script into the installable `beam_opt` package (model builder, COBYLA optimiser, BEAM188
rectangular-section stress evaluator, MAPDL session manager, report). Its application
examples were not part of the public repository. Superseded by the 0.2 API above.
