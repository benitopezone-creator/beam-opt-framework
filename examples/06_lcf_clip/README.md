# Esempio 06 — Fatica oligociclica di una clip a scatto: presa contro vita

La clip a C dell'esempio 02 viene aperta e rilasciata a ogni utilizzo e deve continuare a
stringere il pezzo che trattiene. L'apertura è grande, quindi la clip **snerva
localmente**: il criterio di progetto non è più "tensione sotto lo snervamento", ma **il
numero di aperture prima che nasca una cricca**.

![Presa contro vita](../../docs/img/lcf_clip_trade_off.png)

## Problema

| Grandezza | Valore |
|---|---|
| Geometria e materiale | quelli dell'esempio 02 (raggio 40 mm, larghezza 10 mm) |
| Apertura a ogni utilizzo | 36 mm, ciclo da zero al massimo (R = 0) |
| Curva deformazione-vita | acciaio al carbonio medio: σ'_f = 930 MPa, b = −0,095, ε'_f = 0,26, c = −0,47, K' = 1258 MPa, n' = 0,208 (valori indicativi) |
| Correzione della tensione media | Smith-Watson-Topper |
| Variabile di progetto | spessore radiale `t(s)`, tra 0,3 e 6 mm |
| Obiettivo | massima vita a fatica dell'elemento più critico |
| Vincolo | forza di presa ≥ F_min |

Non c'è vincolo statico di tensione: lo snervamento locale è accettato, e a governare è
la vita.

## Perché è interessante

Con un carico in spostamento imposto, una clip più rigida stringe di più **e** si
deforma di più, quindi dura meno. Presa e vita sono obiettivi in conflitto: l'esempio
traccia il fronte di Pareto con il metodo epsilon-constraint, cioè la vita massima per
ogni presa minima richiesta.

| Presa [N] | Vita ottimizzata | Vita a spessore costante | Guadagno | ε_a [%] | ε plastica [%] |
|---|---|---|---|---|---|
| 10 | 1,35·10⁶ | 2,23·10⁵ | 6,1× | 0,10 | 0,011 |
| 20 | 3,18·10⁵ | 6,11·10⁴ | 5,2× | 0,14 | 0,025 |
| 40 | 8,6·10⁴ | 1,88·10⁴ | 4,6× | 0,19 | 0,054 |
| 60 | 4,23·10⁴ | 9,84·10³ | 4,3× | 0,22 | 0,080 |
| 90 | 2,15·10⁴ | 5,30·10³ | 4,1× | 0,27 | 0,115 |
| 120 | 1,35·10⁴ | 3,46·10³ | 3,9× | 0,31 | 0,148 |

A parità di forza di presa, lo spessore variabile moltiplica la vita **da 4 a 6 volte**.
L'ottimizzatore ispessisce il dorso, dove il momento è massimo, e assottiglia la punta,
così da distribuire lo stesso danno lungo tutta la clip. Per prese elevate si è in pieno
campo oligociclico (vite di 10³-10⁴ cicli, deformazione plastica comparabile con quella
elastica).

## Esecuzione

```bash
python examples/06_lcf_clip/lcf_clip.py --save-plots results/lcf_clip
```

Circa 30 secondi con il solutore interno.

## Note

- Le costanti deformazione-vita sono indicative: per un progetto reale servono quelle del
  materiale e del trattamento effettivi.
- Il modello è lineare: la plasticità entra solo nel punto critico, tramite la regola di
  Neuber, e non redistribuisce le sollecitazioni nella clip. È l'ipotesi standard
  dell'approccio locale in deformazione, valida finché lo snervamento resta localizzato.
- La teoria è in [docs/teoria.md](../../docs/teoria.md), sezioni 6 e 7.
