# Anchor e discretizzazione (Fase 2)

Dalla Fase 2 il numero di elementi della trave ottimizzabile è un **input** (`n_elem`) e non
dipende più dal numero di punti con cui è definita la geometria.

## Concetti

- **`BeamPath(points, kind="spline")`**: la linea d'asse passa per i punti di definizione.
  Con `"spline"` è una spline cubica parametrica, con `"polyline"` una spezzata. L'ascissa
  *s* è la **lunghezza d'arco vera** [mm].
- **`path.s_knots[i]`**: ascissa del punto di definizione `i` (indice da 0).
- **Anchor**: punto della linea d'asse con un nome, definito dalla sua ascissa
  (`{"support_1": 37.5}`). `discretize` mette **sempre un nodo esattamente** su ogni anchor.
  `"start"` ed `"end"` esistono sempre e sono riservati.
- **`discretize(path, n_elem, anchors)`**: le estremità e gli anchor dividono la trave in
  tratti; ogni tratto riceve elementi di uguale lunghezza in *s*, distribuiti in modo che
  l'elemento più lungo sia il più corto possibile.

Elementi fissi e condizioni al contorno citano solo **nomi** (anchor o punti extra), quindi la
stessa configurazione gira con 10, 49, 196 o 400 elementi.

## Conversione di una configurazione basata su ID di nodo

Se una configurazione indica i nodi della trave con il loro numero (1..N, nodo *k* = punto di
definizione *k*), la conversione è:

| Prima (numeri di nodo) | Dopo (nomi) |
|---|---|
| nodo `1` | anchor `"start"` |
| nodo `N` | anchor `"end"` |
| nodo `k` | `{"nome": path.s_knots[k - 1]}` |
| `FixedBeam(id, k, 101, b, h)` | `FixedBeam("nome", "support_a", RectSection(b, h))` |
| nodi aggiuntivi `{101: [...], 201: [...]}` | `points={"support_a": [...], "plate_0": [...]}` |
| vincoli impliciti nel builder | oggetti `Symmetry`, `ImposedDisplacement`, `RigidRegion` |

Esempio (schema):

```python
path = BeamPath(definition_points)  # (N, 3) [mm]
anchors = {
    "joint_a": path.s_knots[1],
    "joint_b": path.s_knots[5],
    "joint_c": path.s_knots[9],
}
beam = OptimizableBeam(discretize(path, n_elem, anchors))
```

Un esempio completo e verificato (telaio curvo sintetico) è in
`tests/test_mesh_independence.py` (`curved_frame`).

## Studio di convergenza di mesh

Il vettore di progetto (valori ai punti di controllo della legge di sezione) non cambia con
`n_elem`, perché i punti di controllo stanno a `linspace(0, path.length, n_ctrl)`. Per uno
studio di convergenza basta quindi valutare lo stesso `x` con `n_elem` crescenti:

```python
for n in (49, 98, 196):
    problem = OptimizationProblem(mapdl, make_model(n), n_ctrl=8, ...)
    ev = problem.evaluate(x)
    print(n, ev.mass, ev.sigma.max(), ev.reaction)
```

Nota: la regola del punto medio usata per la massa converge al secondo ordine se la legge è
regolare; dove la legge è clippata ai limiti (spigoli) la convergenza può essere più lenta e
non monotona.
