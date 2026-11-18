# Note di teoria

Formulazione, ipotesi e formule usate dal framework. Unità: mm, N, ton, s, MPa.

## 1. Il problema di ottimizzazione

Si cerca la distribuzione delle dimensioni della sezione lungo la trave che minimizza un
obiettivo rispettando i vincoli:

```
min   f(x)                              massa (predefinito), cedevolezza, spostamento,
x                                       tensione di picco, −log10 della vita a fatica
t.c.  σ_VM,e(x) ≤ σ_y / FS              per ogni elemento e (facoltativo)
      F_min ≤ F_reaz(x) ≤ F_max         reazione sui nodi scelti
      |u_k(x)| ≤ u_k,max                spostamenti o rotazioni scelti
      m(x) ≤ m_max                      budget di massa
      N_e(x) ≥ N_min                    vita a fatica di ogni elemento
      x_min ≤ x ≤ x_max                 limiti tecnologici
```

È un'ottimizzazione **dimensionale**: la linea d'asse non cambia, cambiano le sezioni.
Quando l'obiettivo non è la massa serve di norma un budget di massa, altrimenti l'ottimo
è semplicemente la trave più pesante.

## 2. Variabili di progetto: la legge di sezione

Ogni parametro della sezione (ad esempio `b` e `h`) è una funzione dell'ascissa curvilinea
`s`, definita dai suoi valori in `n_ctrl` punti di controllo equispaziati. Il vettore di
progetto raccoglie questi valori:

```
x = [p1(s_1) … p1(s_n), p2(s_1) … p2(s_n), …]
```

Tra i punti di controllo si interpola con uno schema a scelta:

| Interpolatore | Regolarità | Può sforare i limiti? |
|---|---|---|
| `pchip` | C¹, monotono | no |
| `linear` | C⁰ | no |
| `natural_cubic` | C² | sì: si aggiunge un vincolo sul campo |

Il numero di variabili **non dipende dalla mesh**: lo stesso progetto si può analizzare con
10 o 400 elementi, il che permette studi di convergenza. Ogni elemento ha sezione costante,
pari alla legge valutata nel suo punto medio.

## 3. Geometria e discretizzazione

La linea d'asse (`BeamPath`) è una spline cubica parametrica per i punti di definizione,
riparametrizzata per lunghezza d'arco. `discretize` la ricampiona in `n_elem` elementi e
inserisce un nodo esatto in ogni *anchor*, cioè nei punti a cui si attaccano vincoli,
carichi ed elementi fissi. Nessun riferimento è fatto per numero di nodo.

## 4. Analisi strutturale

### Solutore interno

Elementi di trave a due nodi con sei gradi di libertà per nodo, lineari, basati sulla
teoria di Timoshenko. La matrice di rigidezza locale è quella esatta per carichi nodali,
con parametro di deformabilità a taglio

```
Φ = 12 E I / (κ G A L²)
```

(`Φ = 0` restituisce l'elemento di Eulero-Bernoulli). Gli assi locali seguono la
convenzione predefinita di BEAM188: `x` dal nodo I al nodo J, `y` parallelo al piano
globale XY, `z = x × y`.

Per sezioni con prodotto d'inerzia non nullo l'elemento è formulato negli assi principali
d'inerzia, ruotati dell'angolo `α = ½ atan2(−2 I_yz, I_y − I_z)` attorno all'asse della
trave; le sollecitazioni vengono poi riportate negli assi della sezione.

I vincoli (incastri, simmetrie, spostamenti imposti, regioni rigide) sono equazioni lineari
`C u = d`. Si risolve nello spazio nullo di `C`: `u = u_p + Z y`, con `Z` base del nucleo
e `u_p` soluzione particolare, entrambe ottenute da una decomposizione ai valori singolari
che tollera vincoli ridondanti. Il sistema ridotto `ZᵀKZ y = Zᵀ(f − K u_p)` è simmetrico e
definito positivo; se non lo è, la struttura è labile e l'analisi viene segnalata come
fallita.

### ANSYS MAPDL

In alternativa lo stesso modello viene scritto in una sessione MAPDL (BEAM188 e SHELL181) e
risolto in statica, anche con grandi spostamenti. Ogni soluzione è verificata a convergenza
(tempo finale pari a 1): una soluzione parziale non arriva mai all'ottimizzatore.

## 5. Tensioni

Le tensioni si ricavano dalle sollecitazioni interne `N, M_y, M_z, T, V_y, V_z` ai due nodi
di ogni elemento, quale che sia il solutore. In ogni **punto di verifica** `(y, z)` della
sezione:

```
σ_x  = N/A + [(M_y I_z + M_z I_yz) z − (M_z I_y + M_y I_yz) y] / (I_y I_z − I_yz²)
τ_xy = a_T T + a_y V_y + a_z V_z
τ_xz = b_T T + b_y V_y + b_z V_z
σ_VM = sqrt(σ_x² + 3 (τ_xy² + τ_xz²))
```

Con `I_yz = 0` la prima formula si riduce a `N/A + M_y z/I_y − M_z y/I_z`. La tensione
tangenziale è trattata come vettore: torsione e taglio si sommano su un lato della sezione
e si sottraggono sull'altro, e ogni sezione elenca i punti di entrambi i lati. Il valore
dell'elemento è il massimo sui punti e sui due nodi.

I coefficienti dipendono dalla forma:

| Sezione | Torsione | Taglio |
|---|---|---|
| Rettangolo | formule di Roark (massimo a metà del lato lungo) | Jourawski, `1,5 V/A` |
| Cerchio, tubo | esatta, `T r / J` | `V Q / (I · corda)`: `4/3 V/A` pieno, `2 V/A` tubo sottile |
| Ellisse | esatta (funzione di Prandtl) | Jourawski sulla corda |
| Scatolare | Bredt, `T / (2 A_m t)` | flusso di taglio sulle due anime |
| Profilo a I | parete sottile aperta, `T t / J` con `J = Σ l t³/3` | flusso di taglio in anima e ali |
| Poligono qualsiasi | numerica (vedi sotto) | Jourawski sulle corde |

### Torsione di una sezione poligonale

Si risolve il problema di Saint-Venant con la funzione di tensione di Prandtl:

```
∇²φ = −2   nella sezione,      φ = 0   sul contorno
J = 2 ∫ φ dA,      τ = (T / J) |∇φ|
```

con elementi finiti triangolari lineari su una triangolazione di Delaunay del poligono. La
soluzione si calcola una sola volta alla scala unitaria; scalando la sezione di un fattore
`s`, le aree vanno come `s²`, le inerzie e `J` come `s⁴`, le tensioni di torsione come
`s⁻³`. Con la risoluzione predefinita `J` è entro lo 0,2 % e la tensione massima è
sottostimata di circa l'1 % su quadrato, triangolo equilatero ed ellisse. Vale per sezioni
piene; il picco teorico negli spigoli rientranti non è risolto.

## 6. Obiettivi diversi dalla massa

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

in cui ogni funzione è liscia. Sulla clip dell'esempio 06, la forma diretta (anche
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

## 7. Fatica oligociclica: approccio locale in deformazione

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
qualunque spettro: lo spettro cambia la vita, non la forma ottima (esempio 06).

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

## 8. Algoritmi

Tutti lavorano su variabili scalate in `[0, 1]` e sulla massa divisa per quella del
progetto più pesante. Viene sempre restituito il progetto ammissibile più leggero tra
quelli valutati.

| Metodo | Idea | Costo tipico |
|---|---|---|
| `slsqp` | programmazione quadratica sequenziale, gradienti per differenze finite | `n + 1` analisi per iterazione |
| `cobyla` | approssimazioni lineari su un simplesso, senza derivate | qualche decina di analisi per variabile |
| `nelder-mead` | simplesso, vincoli tramite penalità esatta | elevato; solo per poche variabili |
| `differential-evolution` | popolazione, mutazione, incrocio e selezione | migliaia di analisi |

Un'analisi non convergente diventa un progetto marcato come fallito, con tensioni e
reazioni pari a un multiplo dei valori ammissibili: l'algoritmo se ne allontana senza
interrompersi, e il progetto non può essere restituito come ottimo.

## 9. Ipotesi e limiti

- Materiale elastico lineare isotropo; verifica alla tensione di von Mises in campo
  elastico. Instabilità, fatica e concentrazioni di tensione non sono considerate.
- Teoria della trave: sezioni piccole rispetto al raggio di curvatura.
- Il solutore interno è lineare (piccoli spostamenti) e non modella shell deformabili.
- Il centro di taglio è assunto nel baricentro: le sezioni disponibili sono doppiamente
  simmetriche o piene, per le quali l'errore è nullo o piccolo.
- Le formule di taglio sono quelle di Jourawski, approssimate per forme prive di assi di
  simmetria.
