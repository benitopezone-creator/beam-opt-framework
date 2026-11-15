# Script APDL: importazione ed esportazione

Il modulo `beam_opt.apdl` collega il framework a un flusso di lavoro ANSYS esistente,
senza richiedere ANSYS per leggere o scrivere i file.

## Importazione: `read_apdl`

```python
from beam_opt.apdl import read_apdl

imported = read_apdl("modello.inp", sigma_y=355.0)
imported.model  # StructuralModel pronto per l'ottimizzazione
imported.param_names  # parametri di sezione, es. ("b", "h")
imported.initial  # valori dello script, uno per elemento ottimizzabile
imported.notes  # comandi ignorati, con il numero di riga
```

Lo script viene **interpretato come testo**: parametri, espressioni e cicli vengono
valutati, e ne risultano nodi, elementi, sezioni, vincoli e carichi.

### Come indicare la trave da ottimizzare

Con un *component* di elementi di nome `OPT_BEAM`:

```
ESEL,S,SEC,,1          ! oppure ESEL,S,ELEM,,1,48  /  ESEL,S,TYPE,,1 ...
CM,OPT_BEAM,ELEM
ALLSEL
```

Gli elementi del component devono formare un'unica catena aperta. Ogni altra trave diventa
un elemento fisso (`FixedBeam`), ogni shell un `FixedShell`. Un nome diverso si passa con
`read_apdl(..., component="NOME")`.

### Comandi supportati

| Categoria | Comandi |
|---|---|
| Parametri | `NOME = espressione`, `*SET`, `*AFUN` (gradi o radianti) |
| Espressioni | `+ - * / **`, parentesi, `SIN COS TAN ASIN ACOS ATAN ATAN2 SQRT ABS EXP LOG LOG10 NINT MOD MIN MAX SIGN`, `PI` |
| Cicli | `*DO` / `*ENDDO`, anche annidati |
| Tipi e materiale | `ET` (BEAM188, SHELL181), `MP` (`EX`, `PRXY`/`NUXY`, `DENS`) |
| Sezioni | `SECTYPE`, `SECDATA`; sottotipi trave `RECT`, `CSOLID`, `CTUBE`, `HREC`, `I`, `ASEC`; shell |
| Mesh | `N`, `E`, `EN`, `TYPE`, `MAT`, `SECNUM` |
| Selezioni | `NSEL` (per numero o per coordinata), `ESEL` (per numero, sezione, tipo, materiale), `ALLSEL`, `CM` |
| Vincoli e carichi | `D` (anche `ALL` e intervalli), `F`, `DSYM,SYMM`, `CERIG` |

I comandi di soluzione e di post-processing (`SOLVE`, `ANTYPE`, `NLGEOM`, `OUTRES`, ...)
vengono ignorati ed elencati in `imported.notes`.

### Limiti

- **Niente modellazione solida** (`K`, `L`, `LMESH`, ...): servono nodi ed elementi. Se il
  modello nasce da keypoint e linee, si può generare la mesh in ANSYS e scrivere nodi ed
  elementi con `NWRITE`/`EWRITE` o `CDWRITE`, oppure riscrivere la generazione con `*DO`.
- **Niente `*IF`, macro e funzioni utente.** Un comando non supportato produce un errore
  `ApdlError` che indica la riga, non un modello sbagliato in silenzio.
- **Nodi di orientamento** delle travi non supportati: si usa l'orientamento predefinito di
  BEAM188.
- **Un solo materiale** per modello.
- Le sezioni utente a mesh (`SECTYPE,,BEAM,MESH` con file `.SECT`) non vengono lette: vedi
  sotto come definire una sezione personalizzata.

### Guidare uno script parametrico da Python

I parametri passati a `read_apdl` sono disponibili allo script come già definiti:

```python
imported = read_apdl("arco.inp", sigma_y=355.0, parameters={"RAGGIO": 250.0})
```

(purché lo script non li ridefinisca).

## Esportazione: `write_apdl`

```python
from beam_opt.apdl import write_apdl

write_apdl(problem.model, problem.law_from_x(result.x), "ottimo.inp", title="Progetto ottimo")
```

Il file contiene esattamente i comandi che il framework invia a MAPDL: nodi, una sezione
per elemento ottimizzabile, elementi fissi, vincoli, carichi, il component `OPT_BEAM` e un
blocco di soluzione statica lineare. Si può eseguire in ANSYS per la verifica finale,
oppure rileggere con `read_apdl` (il giro completo è verificato nei test).

## Sezioni personalizzate

In APDL una sezione non standard si definisce in due modi; il framework copre entrambi i
casi senza ANSYS.

### Sezione nota solo dalle proprietà (`ASEC`)

`SECTYPE,,BEAM,ASEC` con `SECDATA,A,Iyy,Iyz,Izz,Iw,J,CGy,CGz,SHy,SHz,TKz,TKy` viene
importata come `SectionShape.from_properties`. Poiché le proprietà non dicono dove si
trovano le fibre estreme, servono gli ingombri `TKz` e `TKy`: le tensioni sono maggiorate
in sicurezza sul rettangolo di ingombro.

### Sezione di forma qualsiasi (contorno poligonale)

Al posto di una sezione a mesh utente si descrive il contorno:

```python
from beam_opt import OptimizableBeam, polygon_shape

# profilo con intaglio [mm]
shape = polygon_shape([(0, 0), (20, 0), (20, 12), (12, 12), (12, 6), (8, 6), (8, 12), (0, 12)])
beam = OptimizableBeam(mesh, section_factory=shape)
problem = OptimizationProblem(..., bounds={"scale": (0.5, 2.0)})
```

Il framework calcola area, baricentro e inerzie (compreso il prodotto d'inerzia), risolve
numericamente il problema di torsione di Saint-Venant e genera i punti di verifica delle
tensioni. La variabile di progetto è il fattore di scala. In MAPDL la sezione viene scritta
come `ASEC`: MAPDL non calcola le tensioni per queste sezioni, ed è per questo che il
framework le ricava dalle sollecitazioni interne.
