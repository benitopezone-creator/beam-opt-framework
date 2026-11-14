# Esempio 01 — Trave a sbalzo a uniforme resistenza

Caso di verifica del framework: un problema di ottimizzazione dimensionale di cui si
conosce la soluzione esatta.

## Problema

Trave a sbalzo rettilinea, incastrata in `s = 0` e caricata in punta da una forza
trasversale `P`. La larghezza `b` della sezione rettangolare è costante; l'altezza `h(s)`
è la variabile di progetto, descritta da 8 punti di controllo lungo l'ascissa `s`.

| Grandezza | Valore |
|---|---|
| Lunghezza `L` | 200 mm |
| Larghezza `b` | 5 mm |
| Carico in punta `P` | 20 N |
| Limiti tecnologici su `h` | 1 – 8 mm |
| Materiale | acciaio, `σ_y` = 355 MPa, `FS` = 1,5 → `σ_adm` = 236,7 MPa |

Obiettivo: massa minima. Vincolo: `σ_VM ≤ σ_adm` in ogni elemento.

## Soluzione di riferimento

Con sola flessione, `σ = 6 P (L − s) / (b h²)`. La trave più leggera è quella in cui la
tensione vale `σ_adm` in ogni sezione (uniforme resistenza):

```
h*(s) = sqrt( 6 P (L − s) / (b σ_adm) )
```

cioè un profilo parabolico con `h*(0)` = 4,50 mm all'incastro. La sua massa è **2/3** di
quella della trave prismatica di altezza `h*(0)` (35,35 g). Vicino alla punta è attivo il
limite inferiore `h ≥ 1 mm`, che aggiunge una piccola quantità di materiale.

## Esecuzione

Richiede ANSYS MAPDL con licenza:

```bash
python examples/01_cantilever/cantilever.py --save-plots results/cantilever
```

Opzioni principali: `--n-elem` (numero di elementi, default 50), `--n-ctrl` (punti di
controllo, default 8), `--no-plots`. Lo script stampa il report del framework e il
confronto con la soluzione di riferimento, e salva `results/cantilever_comparison.png`.

## Cosa aspettarsi

La massa ottimizzata è di poco superiore a quella esatta, per due motivi:

- 8 punti di controllo non possono seguire la tangente verticale di `sqrt(L − s)` in punta;
- ogni elemento ha sezione costante ed è dimensionato sul suo nodo più sollecitato (quello
  lato incastro).

Entrambi gli scarti si riducono aumentando `--n-ctrl` e `--n-elem`.

## Verifica senza ANSYS

`tests/test_example_cantilever.py` esegue la stessa ottimizzazione sostituendo MAPDL con le
sollecitazioni esatte della trave isostatica (`StaticCantileverMapdl` in `tests/fakes.py`).
Verifica il ciclo di ottimizzazione del framework, non MAPDL; il test con il marker
`mapdl` nello stesso file ripete il confronto con il solutore vero.
