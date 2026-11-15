# Esempio 04 — Da uno script APDL a uno script APDL ottimizzato

`arch.inp` è un normale file di input APDL: un arco semicircolare incastrato alle imposte,
con un carico verticale e uno fuori piano in chiave, scritto con parametri e cicli `*DO`.
L'esempio mostra il percorso completo, **senza ANSYS**:

1. lettura dello script e ricostruzione del modello (`read_apdl`);
2. verifica del progetto scritto nello script;
3. ottimizzazione di larghezza e altezza della sezione lungo l'arco;
4. scrittura dell'ottimo come nuovo script APDL (`write_apdl`), pronto per ANSYS.

## Lo script

| Grandezza | Valore |
|---|---|
| Raggio dell'arco | 300 mm |
| Elementi | 48 BEAM188, generati con `*DO` |
| Sezione iniziale | rettangolare 12 × 20 mm |
| Carichi in chiave | 900 N verticali, 250 N fuori piano |
| Trave da ottimizzare | component `OPT_BEAM` (`ESEL` + `CM,OPT_BEAM,ELEM`) |

La tensione di snervamento non è una proprietà APDL: si passa a `read_apdl(sigma_y=...)`.

## Risultato

| | Script | Ottimizzato |
|---|---|---|
| Massa | 1775 g | 706 g (**−60 %**) |
| Tensione / ammissibile | 49 % | 100 % |
| Freccia in chiave | 0,50 mm | 1,00 mm (limite 1 mm) |

## Esecuzione

```bash
python examples/04_apdl_arch/apdl_arch.py --output results/arch_optimised.inp
```

Lo script scritto contiene una sezione per elemento e lo stesso component `OPT_BEAM`:
si può eseguire in ANSYS per la verifica, oppure rileggere con `read_apdl`.

## Usarlo con il proprio script

```python
from beam_opt import NativeSolver, optimize
from beam_opt.apdl import read_apdl, write_apdl

imported = read_apdl("mio_modello.inp", sigma_y=355.0)
problem = imported.problem(
    NativeSolver(),
    n_ctrl=6,
    bounds={"b": (5, 30), "h": (5, 30)},
    safety_factor=1.5,
    f_react_max=1e4,
)
result = optimize(problem, "slsqp", x0=imported.x0(problem))
write_apdl(problem.model, problem.law_from_x(result.x), "mio_modello_ottimizzato.inp")
```

Comandi supportati e limiti: [docs/apdl.md](../../docs/apdl.md).
