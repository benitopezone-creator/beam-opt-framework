# Fase 1 — Piano di modularizzazione

Documento di lavoro: mappa "vecchio file → nuovo modulo" e interfacce pubbliche.
L'autore ha autorizzato a procedere senza approvazione preventiva; il piano resta qui come
riferimento per la revisione. **Stato: implementato.**

## Verifica di invarianza numerica (senza ANSYS)

Prima del refactor è stato generato `tests/data/phase1_reference.json` facendo girare il
codice *vecchio* (`BeamModel` + `BeamOptimizer`) su un telaio curvo sintetico
(`tests/data/scenario.py`, valori inventati) con un `FakeMapdl`. Il riferimento contiene:

- il modello scritto in MAPDL in forma **canonica** (nodi per coordinate, sezioni per dati,
  vincoli per coordinate dei nodi: indipendente dalla numerazione, `tests/fakes.py`);
- massa fissa, obiettivo, valori di tutti i vincoli e tensioni σ_VM per elemento
  (con risultati SMISC fittizi) per un vettore di progetto fissato.

`tests/test_phase1_regression.py` ricostruisce lo stesso caso con la nuova API e confronta.

## Mappa vecchio → nuovo

| Vecchio | Nuovo | Note |
|---|---|---|
| `config.py` (costanti di materiale) | `materials.py` → `Material` | nessun materiale predefinito nel package |
| `config.py` (limiti, carichi, spostamenti imposti) | configurazione dell'esempio | parametri espliciti delle classi |
| `config.SolverSettings` | `solver.SolverSettings` | |
| `mapdl_manager.MapdlManager` (singleton) | `session.MapdlSession` (context manager) | unico punto di lancio |
| `model.BeamModel.__init__` (nodi, s, L) | `geometry.curve.BeamPath`, `geometry.mesh.BeamMesh` | |
| `model.BeamModel.build` | `model.builder.ModelBuilder.build` | sessione iniettata |
| `model.BeamModel._apply_bcs` (vincoli cablati) | `model.bcs`: `Symmetry`, `ImposedDisplacement`, `RigidRegion` | elencati nella config |
| `fixed_elements.FixedBeam/FixedShell` | `model.components.FixedBeam/FixedShell` | nodi per **nome**, non per ID |
| `fixed_elements.FIXED_*_SECNUM` | numerazione automatica nel builder | niente collisioni con n_elem > 300 |
| `section_law.SectionLaw` (b, h) | `section_law.SectionLaw` (parametri con nome, interpolatore selezionabile) | |
| — | `sections.base.Section`, `sections.rect.RectSection` | A, Iy, Iz, J, Wt, comandi SECTYPE |
| `analysis.BeamAnalysis.solve`, `check_convergence` | `solver.solve`, `solver.check_convergence` | |
| `analysis.BeamAnalysis.max_reaction` | `post.extract.reaction_resultant` | nodi per nome |
| `stress.rect_beam` (get_value per elemento) | `post.extract.beam_smisc` (`*VGET`) + `post.stress.RectBeamStressEvaluator` | regola 6 |
| `optimization.BeamOptimizer` | `optimization.problem.OptimizationProblem` + `optimization.driver.CobylaDriver` | |
| `report.py` | `report/text.py`, `report/plots.py` | |
| `exceptions.py` | invariato | |

## Interfacce principali

```python
Material(E, nu, rho, sigma_y)                          # [MPa], [-], [ton/mm³], [MPa]

BeamPath(points)                                        # .s, .length, .point_at(s)
BeamMesh.from_points(points, anchors={"tip": -1})       # .coords, .s, .s_elem, .lengths,
                                                        # .anchors {"start", "end", ...}

SectionLaw(s_ctrl, {"b": [...], "h": [...]},
           bounds={"b": (lo, hi), "h": (lo, hi)},
           interpolator="natural_cubic")                # .eval(s, clip=True) -> dict
Section (ABC): area, iy, iz, j, torsion_modulus, define(mapdl, secnum, name)
RectSection(b, h); RectSection.from_params({"b":..., "h":...})

OptimizableBeam(mesh, section_factory=RectSection.from_params)
FixedBeam("node_a", "node_b", section)                  # nomi di anchor o di punti
FixedShell(["p0", "p1", "p2", "p3"], thickness)
Clamp(node) | Symmetry(nodes, normal="Z") | ImposedDisplacement(nodes, dof, value)
RigidRegion(master, slaves) | NodalForce(node, direction, value)
StructuralModel(beam, material, points={...}, fixed=[...], bcs=[...])

ModelBuilder(mapdl).build(model, law) -> NodeMap        # nome -> ID MAPDL
solve(mapdl, settings)                                  # solleva SolutionNotConvergedError
beam_smisc(mapdl, n_elem, ids) -> dict[int, ndarray]
reaction_resultant(mapdl, node_ids) -> float
RectBeamStressEvaluator()(mapdl, mesh, law) -> ndarray

OptimizationProblem(model, evaluator, n_ctrl, bounds, sigma_adm, f_react_max,
                    reaction_nodes)                     # .objective, .constraints
CobylaDriver(problem).run(x0, ...)
MapdlSession(run_location, nproc, jobname)              # with MapdlSession(...) as mapdl:
```

## Fuori scopo (fasi successive)

- ricampionamento della curva e anchor esatti con `n_elem` libero → Fase 2;
- punti di verifica generici delle sezioni ed evaluator unico → Fase 3;
- `bounds` scipy, normalizzazione, CSV/checkpoint → Fase 4.
