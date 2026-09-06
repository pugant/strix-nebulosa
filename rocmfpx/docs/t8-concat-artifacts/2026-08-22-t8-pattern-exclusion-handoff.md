# Handoff — T8 pattern-exclusion brainstorm (sessione dedicata a contesto pulito)

**Data:** 2026-08-22 · **Per:** la sessione che apre il nuovo ciclo T8 (pattern-exclusion)
**Stato:** decisione utente presa il 22/08 dopo gate fase A FAIL dello Stadio 2 concat.
**Come procedere:** usare `superpowers:brainstorming` PRIMA di qualsiasi design; questo
file è l'input di contesto, non il design.

## 1. Cosa è successo (sintesi one-page)

Lo Stadio 2 (concat MTP(k1)→DFlash) è stato **implementato e certificato per intero**
(11 commit sul ramo `t8-concat`, pushato su gitea; meccanismo verificato 41/41+20/20+14/14
+7/7; a k1=0 è bit-identico al T7 di produzione) — ma al **gate fase A** il
condizionamento DFlash-su-prefix-MTP ha dato:

- **Verdetto pooled (letterale spec §7): PASS 0.9815** — ma CONFOUND dimostrato: le
  request AG divergevano tra bracci per near-tie f32 (contenuti diversi).
- **Verdetto content-controlled (il dato vero): FAIL 0.831 < 0.90** (soglia relativa 0.90).

| Finestra | Contenuto | ratio acc_concat/acc_df7 |
|---|---|---|
| D1 | conteggio numerico | 0.654 ❌ |
| F1 | pattern alfabeto | 0.502 ❌ |
| F3 | log | 0.556 (n=4, sottile) ❌ |
| AG1 | **reasoning libero** | **1.133 ✅ indenne** |
| F4 | JSON | 1.250 (n=2, non significativo) |

Dettagli: `docs/research/2026-08-21-t8-fasea-conditioning.md` §9-10 (§6 superseded).

## 2. La causa meccanica (identificata, non ipotizzata)

Il blocco DFlash **condizionato** sull'head MTP, sui contenuti pattern, **clona il token
precedente del pattern** (off-by-one: propone la cifra/lettera del numero/elemento
precedente invece di avanzare). Caratteristiche misurate:

- Rifiuti **categorici**, non near-tie: p_dft mediana **0.000** su 43 stop (D1+F1).
- Head MTP accettata al **98%** — il problema è SOLO il blocco condizionato, non l'head.
- Il profilo NON è universale: pattern ad alta confidenza (D1) = muro netto alla
  **posizione 3** del segmento (pos 1-2 a parità); pattern a bassa confidenza (F1) =
  penalità dalla posizione 1 su tutto il segmento (~0.5×).
- Output greedy del target **invariato** a parità di contenuto (verify exact intatto):
  il danno è SOLO throughput (round morti), mai correttezza.
- tok/s D1 content-controlled: −27,7%.

**Il paradosso che motiva il nuovo ciclo:** i contenuti pattern (numeri, liste, log,
JSON) sono ESATTAMENTE il workload dove DFlash vince (det/agentic strutturati) — cioè il
target dello Stadio 2 — mentre sul reasoning libero (dove il concat NON servirebbe, la
policy non lo attiva lì) il condizionamento è sano o migliore (1.133).

## 3. Ipotesi sul perché (da validare/confutare nel brainstorm — NON fatti)

- Il lattice/selettore DFlash2 (block_size=8 dal GGUF, esteso a 8+k1 dal ramo t8-concat)
  potrebbe interpretare le posizioni head `n+1..n+k1` come parte del pattern atteso e
  "riempire" il blocco con la continuazione del pattern a partire da un'ancora
  disallineata di una posizione (clone = continuazione del pattern dall'offset sbagliato).
- Alternativa: l'head MTP sui pattern è un token "ovvio" (alta confidenza) e il blocco
  condizionato collassa sulla modalità copy ( comportamento noto dei drafter leggeri su
  pattern ripetitivi — cf. clone-rate fase 0).
- Dato utile per discriminare: sul braccio DF7 puro lo STESSO tratto pattern è draftato
  bene dallo stesso drafter → il peggioramento viene dall'INPUT al blocco (head reali al
  posto del placeholder noise), non dal drafter in sé.

## 4. Direzioni di design da esplorare nel brainstorm (nessuna è validata)

1. **Rilevatore pattern runtime → fallback DF7 puro**: segnale cheap disponibile nel
   round (ripetitività del draft, p_dft dei round precedenti, entropia) per disattivare
   l'head dove non paga. Rischio: rilevatore in ritardo di un round.
2. **k1 adattivo a memoria di round**: round morto a na<k1' → disattiva k1 per N round
   o per la request. Il dato bimodale (D1: 39 round pieni vs 43 morti a na=3) dice che
   il segnale ESISTE ed è netto.
3. **Condizionamento selettivo per posizione/contenuto**: passare l'head al lattice solo
   per alcune posizioni, o mascherare l'head dove il pattern è rilevato (il plumbing per
   per-token esiste già: le posizioni head sostituiscono i placeholder token-per-token).
4. **Indagare il disallineamento dell'ancora**: se il clone è un off-by-one del punto di
   attacco (ipotesi §3.1), un fix di allineamento nel plumbing del condizionamento
   potrebbe eliminare la causa senza esclusioni. DA VERIFICARE PER PRIMO: è il
   più economico se vero (errore di wiring, non di modello).

**Vincoli non negoziabili ereditati** (spec padre §1/§8): R1 lossless (verify exact
greedy), R2 mai regressione cache/RS non certificata, inference-time only (no
retraining), ncols ≤ 16, prosa/reasoning MAI in concat, gate numerico finale §1
(det ≥ 65 D1-primario, agentic ≥ 58, prosa ≥ −3%).

## 5. Cosa riusare (tutto già nel workspace)

- **Ramo `t8-concat` su gitea** (tip ae6bad9cf, patch durature `patches/t8-concat/
  0001-0011`): meccanismo concat COMPLETO e inerte a k1=0 — qualsiasi variante
  pattern-exclusion parte da qui (il flag k1, il plumbing head, il dispatch accept, le
  metriche sono pronti e certificati).
- **Strumentazione fasea** (`scripts/bench-t8-fasea.sh`): allineamento content-controlled
  token-per-token tra bracci + metrica §7 per-posizione con normalizzazione early-stop.
  RIUSARLA per il gate del nuovo ciclo (la lezione: MAI leggere pooled senza content-check).
- SPEC_VERIFY_LOG per-posizione; suite t8-concat-t1/t2 (60+20 check); causal
  ring-window script.
- **Attenzione infra**: llm-service container RIMOSSO (GPU libera); immagine T7
  produzione c0592a118fb9 NON più nello store (rebuildare dal commit se serve); double
  tag `vulkan-fork-dflash2-route` punta all'immagine t8-concat (NON produzione);
  immagini dangling accumulate (docker image prune opportuno).

## 6. Prima mossa suggerita per la sessione di brainstorm

Partire dall'ipotesi §3.1/§4.4 (off-by-one di wiring dell'ancora): è falsificabile in
poche ore con l'analisi per-posizione già disponibile (se il blocco condizionato clone-
risponde con shift di UNA posizione, si vede nei token proposti vs attesi). Se confermata
→ fix di allineamento + rigiro del gate fase A (costo basso, payload già tutto). Se
confutata → le direzioni §4.1-4.3 (esclusione/adattivo) col rilevatore cheap.

**Fonti primarie:** report fasea (sopra), memoria `project-t8-sinergia-dual-drafter.md`,
spec `docs/superpowers/specs/2026-08-21-t8-stadio2-concat-design.md` (§3 meccanica
condizionamento, §4 policy, §12 fuori-scope), piano fasea emendato
`docs/superpowers/plans/2026-08-21-t8-stadio2-faseA-gate.md` §10-10.1.
