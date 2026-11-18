# Note di teoria

Formulazione, ipotesi e formule usate dal framework. Unità: mm, N, ton, s, MPa.

## 1. Il problema di ottimizzazione

Si cerca la distribuzione delle dimensioni della sezione lungo la trave che minimizza la
massa rispettando i vincoli:

```
min   m(x) = ρ Σ_e A_e L_e
x
t.c.  σ_VM,e(x) ≤ σ_y / FS              per ogni elemento e
      F_min ≤ F_reaz(x) ≤ F_max         reazione sui nodi scelti
      |u_k(x)| ≤ u_k,max                spostamenti o rotazioni scelti
      x_min ≤ x ≤ x_max                 limiti tecnologici
```

È un'ottimizzazione **dimensionale**: la linea d'asse non cambia, cambiano le sezioni.

L'obiettivo è intercambiabile (`OptimizationProblem(objective=...)`, sottoclasse di
`Objective`). Obiettivi diversi dalla massa, fatica, casi di carico multipli e fronti di
Pareto sono estensioni raccolte a parte in `beam_opt.extras`: la loro teoria è in
[extras.md](extras.md).

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

## 6. Algoritmi

Tutti lavorano su variabili scalate in `[0, 1]` e sulla massa divisa per quella del
progetto più pesante. Viene sempre restituito il progetto ammissibile migliore (con la massa, il più leggero) tra
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

## 7. Ipotesi e limiti

- Materiale elastico lineare isotropo; verifica alla tensione di von Mises in campo
  elastico. Instabilità, fatica e concentrazioni di tensione non sono considerate.
- Teoria della trave: sezioni piccole rispetto al raggio di curvatura.
- Il solutore interno è lineare (piccoli spostamenti) e non modella shell deformabili.
- Il centro di taglio è assunto nel baricentro: le sezioni disponibili sono doppiamente
  simmetriche o piene, per le quali l'errore è nullo o piccolo.
- Le formule di taglio sono quelle di Jourawski, approssimate per forme prive di assi di
  simmetria.
