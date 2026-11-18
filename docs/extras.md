# Extras: oltre la massa

Il nucleo del framework è l'ottimizzazione **a massa minima** di travi curve con vincoli
di tensione, rigidezza e reazione (vedi [teoria.md](teoria.md)). Questa guida descrive
le estensioni sviluppate in seguito, raccolte nel sottopacchetto `beam_opt.extras` e
tenute separate dal nucleo:

| Estensione | Contenuto |
|---|---|
| Obiettivi | `Compliance` (rigidezza), `Displacement`, `PeakStress`, `FatigueLife`, `WeightedSum` |
| Fatica oligociclica | `StrainLifeCurve`, `FatigueModel` (Neuber, Masing, Manson-Coffin-Basquin, SWT/Morrow) |
| Ampiezza variabile | `LoadBlock`, `LoadSpectrum`, `rainflow` (regola di Miner, ASTM E1049) |
| Casi di carico | `LoadCase` |
| Fronti di Pareto | `epsilon_constraint` |

Si collegano al nucleo solo attraverso parametri generici di `OptimizationProblem`
(`objective`, `mass_max`, `fatigue`, `life_min`, `load_cases`): chi usa il framework per
la massa non ne vede nulla.

```python
from beam_opt import NativeSolver, OptimizationProblem, optimize
from beam_opt.extras import Compliance

problem = OptimizationProblem(NativeSolver(), model, ..., objective=Compliance(), mass_max=150.0)
result = optimize(problem, "slsqp")
```

Esempi in [examples/extras](../examples/extras):

| Esempio | Cosa mostra |
|---|---|
| [stiffness_pareto](../examples/extras/stiffness_pareto) | fronte massa-rigidezza dell'elica |
| [lcf_clip](../examples/extras/lcf_clip) | presa contro vita a fatica oligociclica della clip, vita in servizio con spettro giornaliero |
| [bracket_load_cases](../examples/extras/bracket_load_cases) | staffa curva con due casi di carico: la forma ottima dipende dal mix |

Gli extras sono verificati dai test in `tests/extras` (61 test), ma sono più recenti e
meno consolidati del nucleo.

## 1. Obiettivi diversi dalla massa

| Obiettivo | Definizione | Uso tipico |
|---|---|---|
| `Mass()` | `m = ρ Σ A_e L_e` | alleggerimento (predefinito) |
| `Compliance()` | `C = Σ F_i u_i`, lavoro dei carichi nodali | massima rigidezza a massa data |
| `Displacement(nodo, gdl)` | `|u|` in un nodo | rigidezza in un punto preciso |
| `PeakStress()` | `max_e σ_VM,e / σ_adm` | resistenza uniforme a massa data |
| `FatigueLife()` | `max_e log10(1/N_e)`, cioè la vita minima | durata a massa o rigidezza data |
| `WeightedSum([...])` | `Σ w_k f_k / f_k,rif` | compromesso tra più obiettivi |

### Formulazione a epigrafe per i problemi min-max

Tensione di picco e vita minima sono il **peggiore** dei valori per elemento: la funzione
`max_e g_e(x)` ha un gradiente discontinuo, perché l'elemento critico cambia mentre
l'ottimizzatore si muove. Con SLSQP e COBYLA il framework risolve quindi il problema
equivalente

```
min τ     t.c.   g_e(x) ≤ τ   per ogni elemento e
x, τ
```

in cui ogni funzione è liscia. Sulla clip dell'esempio extra `lcf_clip`, la forma diretta (anche
smussata con l'inviluppo KS) portava SLSQP a fermarsi su soluzioni diverse a seconda del
punto di partenza; con l'epigrafe tutti i punti di partenza convergono allo stesso ottimo,
con 5-10 volte meno analisi.

### Fronti di Pareto

Due obiettivi in conflitto si studiano con il metodo **epsilon-constraint**
(`epsilon_constraint`): si ottimizza il primo imponendo un limite al secondo, e si fa
variare il limite. A differenza della somma pesata trova anche i tratti non convessi del
fronte. Ogni ottimizzazione parte dall'ottimo precedente.

### Verifica sulla trave a sbalzo

Per la mensola isostatica a larghezza costante, a parità di massa, minima cedevolezza,
minima freccia, minima tensione di picco e massima vita a fatica portano tutte allo
stesso profilo `h(s) ∝ sqrt(L − s)`, la trave a uniforme resistenza. Rispetto alla trave
prismatica della stessa massa la cedevolezza si riduce di un fattore 27/16. I test lo
verificano: l'ottimizzatore arriva entro lo 0,3 % dalla soluzione discreta esatta.

## 2. Casi di carico indipendenti

Con `load_cases=[LoadCase(...), ...]` ogni progetto viene analizzato una volta per caso.
Ogni caso sostituisce i carichi del modello (forze nodali, spostamenti imposti) e ne
conserva i vincoli. Poi:

- **vincoli**: tensione, reazioni e spostamenti devono valere in ogni caso;
- **cedevolezza**: `Σ_c w_c C_c`, con i pesi dei casi;
- **tensione di picco e spostamento**: il peggiore tra tutti gli elementi e tutti i casi,
  con la stessa formulazione a epigrafe;
- **fatica**: ogni caso ha il proprio spettro (o un numero di cicli), e il danno di Miner
  si somma sui casi, `D = Σ_c D_c`. I cicli dovuti alla transizione tra casi diversi non
  sono contati.

Con un solo schema di carico la forma ottima non dipende dallo spettro (sezione 3); con
più casi che sollecitano la sezione in modi diversi sì, ed è quello che mostra l'esempio extra `bracket_load_cases`.

## 3. Fatica oligociclica: approccio locale in deformazione

L'analisi è lineare e fornisce la tensione *elastica* `S` al picco del ciclo di carico.
La risposta elasto-plastica nel punto critico si ricava così:

```
curva ciclica         ε = σ/E + (σ/K')^(1/n')                         (Ramberg-Osgood)
Neuber, picco         σ_max ε_max = (K_t S_max)² / E
Neuber, escursione    Δσ Δε = (K_t ΔS)² / E,   Δε = Δσ/E + 2 (Δσ/2K')^(1/n')   (Masing)
deformazione-vita     ε_a = σ'_f/E (2N)^b + ε'_f (2N)^c               (Manson-Coffin-Basquin)
```

con `ΔS = (1 − R) S_max`, `ε_a = Δε/2`, `σ_a = Δσ/2`, `σ_m = σ_max − σ_a`. La tensione
media si considera con Smith-Watson-Topper

```
σ_max ε_a = σ'_f²/E (2N)^(2b) + σ'_f ε'_f (2N)^(b+c)
```

oppure con Morrow, `ε_a = (σ'_f − σ_m)/E (2N)^b + ε'_f (2N)^c`. La vita si ottiene per
bisezione su `log10(2N)` tra 0 e 12; oltre `2N = 10¹²` si considera vita infinita.

### Ampiezza variabile: rainflow e regola di Miner

Uno spettro di carico (`LoadSpectrum`) è un elenco di blocchi: `n_k` cicli tra
`λ_min,k·P` e `λ_max,k·P`, dove `P` è il carico dell'analisi. Per ogni blocco l'elemento
vede `S_max = max(|λ_max|, |λ_min|)·S` e `ΔS = (λ_max − λ_min)·S`, da cui la vita `N_k`
con le relazioni precedenti. Il danno di una ripetizione dello spettro (una missione, un
giorno) si somma con la regola lineare di Miner

```
D = Σ_k n_k / N_k,        vita = 1 / D   ripetizioni
```

I cicli sotto il limite di fatica (vita oltre `0,5·10¹²` cicli) non danneggiano. Lo
spettro si può ricavare da una storia di carico nel tempo con il conteggio **rainflow**
(`rainflow`, `LoadSpectrum.from_history`), secondo la norma ASTM E1049-85, §5.4.4; i test
ne riproducono l'esempio.

Con un unico schema di carico scalato, il progetto a danno uniforme è ottimo per
qualunque spettro: lo spettro cambia la vita, non la forma ottima (esempio extra della clip).

Ipotesi:

- **Carico proporzionale**: ogni punto scala con il carico, quindi una sola analisi
  lineare descrive ogni ciclo. Vale sia per forze sia per spostamenti imposti, purché si
  scali un unico schema di carico (non più casi di carico indipendenti).
- **Regola di Miner lineare**: si trascurano gli effetti di sequenza e la memoria dei
  cicli di isteresi tra un blocco e l'altro.
- **Tensione equivalente di von Mises presa positiva**: conservativo nei punti compressi
  quando conta la tensione media.
- **Plasticità localizzata**: la regola di Neuber non redistribuisce le sollecitazioni
  nella struttura, ipotesi valida finché lo snervamento resta limitato a zone piccole.

I test verificano la regola di Neuber, la coincidenza con Basquin in campo elastico per
tutte e tre le correzioni, il giro completo sulla curva di Manson-Coffin in campo plastico
e l'effetto della tensione media.
