# ROADMAP — beam-opt-framework v0.2

Piano di lavoro a fasi. Ogni fase ha: obiettivo, criteri di accettazione, prompt pronto da
incollare in Claude Code. Eseguire nell'ordine: le fasi successive presuppongono le precedenti.

---

## Fase 0 — Messa in sicurezza (bug e packaging)

**Obiettivo.** Rendere la repo installabile e coerente prima di qualunque refactor.

Problemi da correggere:
- `pyproject.toml`: `build-backend = "setuptools.backends.legacy:build"` non esiste →
  `pip install -e .` fallisce. Usare `setuptools.build_meta`.
- README e CHANGELOG citano `examples/` e `run_optimization.py` che non sono nella repo;
  il quick start usa `--config` senza valore. Manca il file `LICENSE` (README dice MIT).
- `stress/rect_beam.py`: SMISC 5 e 6 hanno le etichette invertite (5 = SFZ, 6 = SFY);
  inoltre al nodo J si usano Mt e tagli del nodo I (usare SMISC 17, 18, 19).
- `analysis.solve()`: aggiungere il controllo di convergenza (tempo finale raggiunto = 1.0);
  in caso contrario sollevare `SolutionNotConvergedError`.
- `fixed_elements.FixedShell.estimate_mass`: area calcolata sulla proiezione XY (shoelace);
  sostituire con area 3D (somma di prodotti vettoriali dei due triangoli).
- `optimization.constraint_section_field` è sempre soddisfatto perché la legge fa clipping:
  il vincolo è inutile e il clipping crea zone piatte per COBYLA. Decidere una strategia
  (vedi Fase 4) e documentarla.

**Accettazione.** `pip install -e ".[dev]"` funziona; `pytest` gira (anche con 0 test MAPDL);
README coerente con i file presenti.

**Prompt.**
> Leggi CLAUDE.md e ROADMAP.md. Esegui la Fase 0. Per ogni punto: mostrami il problema nel
> codice, proponi la correzione, poi applicala. Crea una cartella tests/ con conftest.py,
> il marker `mapdl` registrato in pyproject e un primo test unitario per l'area 3D della shell.
> Alla fine riassumi le modifiche e aggiorna CHANGELOG.md.

---

## Fase 1 — Modularizzazione e testabilità

**Obiettivo.** Separare responsabilità e rendere tutto testabile senza ANSYS.

Struttura target:
```
beam_opt/
  materials.py          Material (E, nu, rho, sigma_y) — dataclass, nessun materiale predefinito
  geometry/
    curve.py            BeamPath: curva 3D da punti, ascissa s, ricampionamento
    mesh.py             discretize(path, n_elem, anchors) -> BeamMesh (nodi, connettività)
  sections/
    base.py             Section (ABC): A, Iy, Iz, J, punti di verifica, coeff. di taglio
    rect.py circ.py tube.py ...
  section_law.py        SectionLaw: p(s) per N parametri, interpolatore selezionabile
  model/
    components.py       OptimizableBeam, FixedBeam, FixedShell, ...
    bcs.py              Clamp, Symmetry, RigidRegion, ImposedDisplacement, NodalForce
    builder.py          ModelBuilder: scrive il modello in una sessione mapdl iniettata
  solver.py             solve(mapdl, settings) + controllo convergenza
  post/
    extract.py          estrazione vettoriale risultanti interne e reazioni
    stress.py           StressEvaluator generico (usa Section)
  optimization/
    problem.py          obiettivo + registro vincoli
    driver.py           COBYLA (e altri metodi scipy), cache, storico iterazioni
  report/               testo, grafici, export CSV/JSON
  session.py            MapdlSession (ex MapdlManager): context manager, unico punto di lancio
tests/
  fakes.py              FakeMapdl che registra i comandi
```

Le condizioni al contorno oggi cablate in `model.py` diventano oggetti `bcs` elencati nella
configurazione dell'esempio.

**Accettazione.** Nessun modulo del package importa costanti di un'applicazione; nessuno
crea una sessione MAPDL nel costruttore; un test con `FakeMapdl` verifica la sequenza di
comandi di build per un modello minimo.

**Prompt.**
> Fase 1 della ROADMAP. Prima entra in plan mode: leggi tutto beam_opt/, proponi la mappa
> "vecchio file → nuovo modulo" e le interfacce pubbliche principali (firme e docstring),
> senza scrivere codice. Dopo la mia approvazione, implementa a piccoli passi con un
> commit per modulo, mantenendo i risultati numerici invariati.

---

## Fase 2 — Discretizzazione imposta dall'utente

**Obiettivo.** `n_elem` è un input; geometria, anchor ed elementi fissi si adattano.

Specifica:
- `BeamPath` costruito dai punti di definizione della linea d'asse con
  interpolazione di forma (spline parametrica per lunghezza d'arco; opzione polilinea).
- `discretize(path, n_elem, anchors)` ricampiona per lunghezza d'arco e **inserisce nodi
  esatti negli anchor** (punti duri: attacchi dei supporti, estremità, carichi).
- Elementi fissi e BC si riferiscono agli anchor per nome, mai a un numero di nodo.
- Numerazione: nodi/elementi/sezioni ottimizzabili da 1..n; entità fisse con offset calcolato
  automaticamente (niente collisione con le sezioni 301/1000 attuali quando n_elem > 300).
- La legge di sezione dipende solo da s e dai punti di controllo: il numero di variabili di
  progetto **non** cambia con la mesh. Questo abilita lo studio di convergenza di mesh.

**Accettazione.** Test: la lunghezza totale converge al crescere di n_elem; gli anchor cadono
esattamente su nodi; la stessa configurazione di una struttura curva gira con `n_elem`
diversi (es. 49 e 196) senza modifiche.

**Prompt.**
> Fase 2. Implementa BeamPath e discretize con anchor. Scrivi prima i test (lunghezza di un
> arco di circonferenza noto, anchor su nodi, monotonia di s), poi l'implementazione.
> Converti le configurazioni basate su ID di nodo per usare anchor con nome.

*Stato:* implementata e verificata sul telaio curvo sintetico dei test con `n_elem` = 49 e
196 (`tests/test_mesh_independence.py`). I test MAPDL della fase sono scritti ma da eseguire.

---

## Fase 3 — Tensioni per sezioni generiche

**Obiettivo.** Un solo evaluator, sezioni intercambiabili.

Specifica:
- Input: risultanti ai nodi I e J (SMISC 1–6 e 14–19) + `Section` valutata in s_I e s_J.
- Ogni `Section` espone: A, Iy, Iz, J e una lista di **punti di verifica** (y, z) con, per
  ciascuno, coefficienti di tensione tangenziale da torsione e da taglio (τ = c_T·T +
  c_y·Vy + c_z·Vz). σ_x = N/A + My·z/Iy − Mz·y/Iz (verificare convenzioni di segno MAPDL).
- σ_VM = sqrt(σ_x² + 3τ²) per punto, massimo per elemento.
- Implementazioni: rettangolo (correggendo la torsione nulla agli spigoli), circolare piena,
  tubo, e un caso "generico" che legge A, I, J dal section mesher di MAPDL (SECTYPE/SLIST)
  per forme arbitrarie (es. sezioni con intaglio).
- Test di verifica: confronto con MAPDL stesso (`/ESHAPE` + tensioni di sezione) su una
  trave semplice per ciascuna forma, entro una tolleranza dichiarata.

**Accettazione.** Il rettangolo nuovo riproduce i risultati del vecchio evaluator dove i due
devono coincidere, e le differenze residue sono spiegate.

*Stato:* implementati `StressPoints`, `SectionStressEvaluator` (ora predefinito) e le sezioni
rettangolare, circolare piena, tubo e scatolare. La tensione tangenziale è trattata come
vettore (τ_xy, τ_xz) anziché come somma scalare. Il rettangolo coincide con il vecchio
evaluator in assenza di torsione; le differenze con torsione sono documentate nel CHANGELOG.
Non fatti, perché richiedono MAPDL: il caso "generico" dal section mesher e il confronto con
le tensioni di sezione di MAPDL (`/ESHAPE`); le convenzioni di segno delle risultanti sono
quindi da verificare sul solutore vero.

---

## Fase 4 — Ottimizzazione più robusta

- Sostituire il clipping con `bounds` di scipy (COBYLA li supporta nelle versioni recenti)
  o con riparametrizzazione esplicita; rimuovere vincoli ridondanti.
  *Strategia decisa in Fase 0:* i limiti sui **punti di controllo** passano ai `bounds`
  (sostituiscono `constraint_bounds` e il clip di x in `_law_from_x`); il vincolo di
  **campo** `constraint_section_field` resta, valutato sulla spline non clippata
  (`SectionLaw.eval(s, clip=False)`), perché i bounds sui punti di controllo non impediscono
  l'overshoot della spline cubica tra un punto e l'altro. Il clip in `eval` resta solo come
  rete di sicurezza per la geometria passata a MAPDL.
- Normalizzare variabili e obiettivo; registrare lo storico (x, massa, σ_max, F) su CSV per
  ogni valutazione FEM, con possibilità di ripartire da un checkpoint.
- Gestione dei fallimenti FEM: penalità o rifiuto esplicito, mai valori silenziosamente errati.
- Interfaccia per altri metodi (SLSQP con differenze finite, eventualmente surrogati) senza
  toccare il problema.

*Stato:* implementata. `optimize(problem, method=...)` con COBYLA, SLSQP, Nelder-Mead ed
evoluzione differenziale; variabili normalizzate e `bounds`; storico CSV e ripartenza;
fallimenti FEM penalizzati e segnalati. Il clip dei punti di controllo in `law_from_x` resta
come rete di sicurezza (il riferimento di regressione della Fase 1 lo presuppone). Verificata
sulla trave a sbalzo con il sostituto analitico di MAPDL; da eseguire con MAPDL vero. Non
ancora fatti: surrogati e un algoritmo genetico in senso stretto (codifica binaria/reale con
operatori dedicati), che richiederebbe una dipendenza come `pymoo`.

---

## Fase 5 — Interfaccia di importazione APDL

**Obiettivo.** Uno script APDL dell'utente → configurazione pronta per l'ottimizzazione.

Approccio consigliato (più robusto di un parser testuale, che dovrebbe gestire `*DO`,
parametri, `*IF`, macro):
1. eseguire lo script in MAPDL (`mapdl.input(file)`);
2. leggere la mesh risultante (coordinate nodi, connettività, tipo, sezione, materiale)
   e le BC tramite PyMAPDL;
3. identificare gli elementi ottimizzabili con una **convenzione**: un component
   `CM,OPT_BEAM,ELEM` definito nello script (eventualmente più component per più leggi);
4. ricostruire il percorso ordinato delle travi ottimizzabili (catena di nodi) e generare un
   modulo Python di configurazione (o YAML) leggibile e modificabile.

Nota: PyMAPDL include già un convertitore APDL → Python (`convert_script` / comando
`pymapdl convert`). Valutare se usarlo come opzione "traduzione letterale"; il valore aggiunto
di questo modulo è la parte 3–4 (cosa ottimizzare e come parametrizzarlo).

**Accettazione.** Esempio: script APDL della trave a sbalzo → config generata → stessa
soluzione dell'esempio scritto a mano.

*Stato:* implementata con un approccio diverso da quello consigliato sopra. Poiché ANSYS
non era disponibile, `beam_opt.apdl.read_apdl` interpreta lo script come testo (parametri,
espressioni, `*DO`, nodi, elementi, sezioni, selezioni, vincoli, carichi) invece di
eseguirlo in MAPDL; `write_apdl` scrive il progetto ottimizzato. Funziona senza licenza ed è
verificata con un giro completo esportazione → importazione. Restano fuori modellazione
solida, `*IF`, macro e sezioni a mesh utente: per questi casi la via "esegui in MAPDL e
leggi la mesh" descritta sopra è ancora da fare.

---

## Fase 7 — Solutore interno (framework senza ANSYS)

**Obiettivo.** Permettere a chiunque di provare il framework senza licenza e far girare
ottimizzazioni vere nei test.

*Stato:* implementato `beam_opt.native.NativeSolver` (travi di Timoshenko 3D, analisi
lineare), intercambiabile con la sessione MAPDL. Verificato su soluzioni analitiche.
Da fare: confronto diretto con MAPDL sullo stesso modello (spostamenti e risultanti),
gradienti analitici (metodo aggiunto) per `slsqp`, non linearità geometrica, rendere
`ansys-mapdl-core` una dipendenza opzionale.

---

## Fase 6 — Esempi e documentazione

- `examples/01_cantilever/`: trave a sbalzo, carico in punta. Verifica analitica (freccia
  PL³/3EI, σ = M/W) e ottimizzazione a massa minima con vincolo di tensione → confronto con la
  soluzione a "uniforme resistenza" (h(s) ∝ sqrt(L−s) a b costante). È il test di regressione
  ideale del framework.
- `examples/02_curved_frame/`: struttura 3D fortemente curva con sotto-strutture fisse
  (travi e shell), simmetria e spostamenti imposti; studio di convergenza di mesh.
- `docs/`: teoria (legge di sezione, recupero tensioni, ipotesi), guida utente, come aggiungere
  una sezione e un vincolo.
- GitHub Actions: lint + `pytest -m "not mapdl"` a ogni push.

*Stato:* fatti quattro esempi eseguibili con il solutore interno (01 trave a sbalzo, 02
molla a C a spostamento imposto, 03 elica con puntone, 04 arco da script APDL), la
documentazione di teoria (`docs/teoria.md`) e la guida APDL (`docs/apdl.md`). GitHub Actions
attivo (`.github/workflows/ci.yml`).

---

## Fase 8 — Sezioni e vincoli aggiuntivi

*Stato:* fatti il profilo a I, l'ellisse, le sezioni personalizzate (`polygon_shape`,
`SectionShape.from_properties`), il prodotto d'inerzia, i vincoli su spostamento e reazione
minima.

Idee per il seguito:
- più travi ottimizzabili nello stesso modello e più casi di carico;
- vincoli di instabilità e di frequenza propria;
- sezioni aperte non simmetriche (C, L, T) con centro di taglio distinto dal baricentro;
- gradienti analitici (metodo aggiunto) nel solutore interno;
- dimensioni scelte da un catalogo di profili commerciali;
- lettura delle sezioni a mesh utente di MAPDL (`.SECT`).
