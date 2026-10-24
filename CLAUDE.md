# CLAUDE.md — beam-opt-framework

Istruzioni di progetto per Claude Code. Letto automaticamente a ogni sessione.

## Cos'è il progetto

Framework Python per l'**ottimizzazione dimensionale (size optimization) di strutture traviformi**
modellate in ANSYS MAPDL tramite PyMAPDL (BEAM188, con SHELL181 per sotto-strutture fisse).
Pensato per strutture traviformi complesse con forti curvature nello spazio, con
sotto-strutture fisse, spostamenti imposti e vincoli di simmetria.

**Repo pubblica e generica:** nessun caso applicativo specifico, nessun dato reale di
progetto. Esempi e test usano solo geometrie e valori inventati.

Variabili di progetto: valori ai punti di controllo di una legge di sezione p(s) lungo
l'ascissa curvilinea s. Obiettivo tipico: massa. Vincoli tipici: σ_VM per elemento ≤ σ_s/FS,
reazione vincolare ≤ F_max, limiti tecnologici sulle dimensioni.

## Lingua e stile

- Codice, docstring, nomi: **inglese**. Docstring stile NumPy.
- Conversazione con l'autore e documentazione utente (`docs/`, README degli esempi): **italiano**
  va bene; il README principale resta in inglese.
- Python ≥ 3.10, type hints ovunque, `from __future__ import annotations`.
- Lint/format: `ruff check .` e `ruff format .` (line-length 100).
- Unità: sistema mm–N–ton–s–MPa (coerente con MAPDL). Ogni grandezza fisica nei docstring
  riporta l'unità tra parentesi quadre.

## Comandi

```bash
pip install -e ".[dev]"          # installazione in sviluppo
pytest -m "not mapdl"            # test veloci, NON richiedono ANSYS
pytest -m mapdl                  # test di integrazione, richiedono MAPDL con licenza
ruff check . && ruff format --check .
```

## Regole architetturali (non negoziabili)

1. **Nessun ID numerico hard-coded** (nodi, elementi, sezioni) nel package `beam_opt/`.
   Le entità si riferiscono a punti della geometria tramite *anchor* con nome
   (es. `"tip"`, `"support_1"`) o tramite ascissa s; la numerazione la decide il mesher.
2. **Il livello di discretizzazione è un parametro** (`n_elem`). Lo stesso codice deve girare
   con 10, 50 o 400 elementi senza modifiche ai file di configurazione.
3. **Nessuna proprietà specifica di un'applicazione nel package** (materiale, spostamenti
   imposti, simmetrie, limiti b/h). Vivono nelle configurazioni sotto `examples/`.
4. **La sessione MAPDL si inietta**, non si crea nei costruttori. I moduli ricevono un oggetto
   `mapdl` (o un `FakeMapdl` nei test). Il lancio avviene solo negli entry point.
5. **Il calcolo delle tensioni dipende dalla sezione, non dal solver**: un evaluator generico
   usa le risultanti interne (N, My, Mz, T, Vy, Vz) e le proprietà/punti di verifica forniti
   dall'oggetto `Section`. Aggiungere una forma = aggiungere una classe `Section`.
6. Estrazione risultati **vettoriale** (`mapdl.get_array`, `*VGET`, mesh arrays), mai un
   `get_value` per elemento in un ciclo Python.
7. Ogni soluzione FEM va **verificata a convergenza** (tempo finale = 1.0); una soluzione
   non convergente solleva un'eccezione dedicata e non arriva mai all'ottimizzatore come
   risultato valido.

## Attenzione: conoscenza di dominio BEAM188

- Tabella SMISC di BEAM188 (verificare sempre su Table 188.2 della documentazione ANSYS):
  nodo I → 1=FX, 2=MY, 3=MZ, 4=TQ, **5=SFZ, 6=SFY**; nodo J → 14..19 nello stesso ordine;
  31..35 = SDIR, SByT, SByB, SBzT, SBzB al nodo I; 36..40 gli stessi al nodo J.
- Con NLGEOM,ON MAPDL può scrivere un .rst anche se la soluzione si ferma prima del carico
  completo: l'esistenza del file non è prova di convergenza.

## Come lavorare su questo repo

- Una fase della `ROADMAP.md` per volta, su un branch dedicato (`feat/<nome-fase>`).
- Prima di modificare codice: leggi i file coinvolti e proponi un piano; aspetta conferma
  per refactoring che toccano più di 3 moduli.
- Ogni modifica funzionale arriva con test. Un refactor puro non deve cambiare i risultati
  numerici: confronta con i test di regressione.
- Non cancellare codice dell'autore senza dirlo esplicitamente nel riepilogo.
- Commit piccoli, messaggi in inglese stile Conventional Commits (`feat:`, `fix:`, `refactor:`,
  `test:`, `docs:`). Aggiorna `CHANGELOG.md` a fine fase.
- Se un passo richiede ANSYS e MAPDL non è disponibile, scrivi il test con il marker `mapdl`
  e dichiara che non è stato eseguito.
