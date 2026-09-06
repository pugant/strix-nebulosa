# Spec — T8 Stadio 2: concat MTP(k1)→DFlash nel round duale

**Data:** 2026-08-21 · **Stato:** bozza da review loop
**Contesto:** routing T7-f2 (MTP n6 ↔ DFlash2 n7 per-request) in produzione dal 20/08.
Fase 0 T8 chiusa oggi: Stadio 1 (ngram-overlay) **NO-GO** al gate clone-rate; sub-gate
ncols eseguito con **emendamento approvato dall'utente** (§2). Questa spec deriva dalla
`2026-08-20-t8-dual-drafter-synergy-design.md` §4, aggiornata con i fatti della fase 0
(report `docs/research/2026-08-21-t8-fase0-session-report.md`).

## 1. Obiettivo e gate (invariati dalla spec T8 §1/§4)

**Funzione obiettivo:** token accettati per forward del target (1 forward/round sempre;
costo draft → 0). Perimetro: TG end-to-end det/agentic.

**Gate di uscita (numerico, end-to-end):** det ≥ 65 tok/s (primario su D1 "conta",
baseline 57.4; reporting separato obbligatorio su D2, baseline ~41), agentic ≥ 58 (da
48), prosa ≥ −3% (mai peggiorata; prosa/reasoning NON toccati dal concat). T2/T3 del
T7-f2 ri-certificati in config simple + produzione (`--parallel 4 --kv-unified`).
Rollout produzione su approvazione utente + osservazione 24h (meccanica T7).

**Numeri attesi (stime report 07, dati veri, incertezze dichiarate §9):** E[round] det
7.6→11.1 (+46%), agentic-tools 6.6→9.1 (+38%); tok/s det 57→65-75 (power-law
18.4·E^0.57, NON lineare; calibrata a E≤8 → estrapolazione a 11.1 incerta).

**Requisiti vincolanti ereditati:** R1 lossless (verify exact prefix-match greedy; output
per-seq corretto vs target, caveat batched noto T7), R2 mai regressione cache/RS non
certificata, R3 switch produzione solo su approvazione.

## 2. Emendamento al sub-gate ncols (APPROVATO UTENTE 21/08 — dichiara, non nasconde)

La spec T8 §4.3 pre-registrava: "verify 14 col > +15% del tempo vs 8 col → fallback
(a) k1=1+DF6, (b) patch VK, (c) concat OFF". Il bench della fase 0
(`docs/research/2026-08-21-t8-ncols-curve.md`, A[4096×8192] Q4_0_ROCMFP4_FAST, 5 run ×
50 iter, mean±stddev, probe `GGML_VK_PERF_LOGGER` a conferma del path) ha misurato:

| Config verify | t/op | µs/colonna | numerica |
|---|---|---|---|
| 8 col dmmv (oggi) | 143.7±6.2 µs | 17.9 | f32 exact |
| 14 col dmmv con patch VK 16 | 210.8±9.5 µs | **15.0 (−16%)** | f32 exact |
| 14 col tiled (senza patch) | 249.3±2.9 µs | 17.8 (neutro) | **f16 accumulate** (rel ~0.11 sul mul_mat nel bench) |

La soglia pre-registrata era per-op (Δt +47.4% → FALLITA), MA la funzione obiettivo del
filone è token accettati / tok/s: **per token verificato la patch VK è la migliore
opzione (−16%) e l'unica che preserva il verify f32 exact** (il tiled f16 cambierebbe i
logits e quindi l'acceptance — inaccettabile per R1). Decisione utente 21/08: **sub-gate
emendato a costo-per-token-verificato; patch VK `mul_mat_vec_max_cols 8→16`
(`ggml/src/ggml-vulkan/ggml-vulkan.cpp:324`) ENTRA nello Stadio 2 e viene COMMITTATA nel
branch** (non più solo staging; 1 riga, costante compile-time). L'A/B end-to-end resta
l'unico giudice del gate numerico §1.

Scoperte accessorie del bench da tenere nel piano: first-use specializzazione pipeline
per ncols = 19-142 ms (one-off per processo → i bench fanno warm-up che la assorbe; il
primo round dopo boot di produzione la paga una volta); path tiled accumula f16 (rel
~0.11) — motivo per cui NON si usa il fallback tiled per il verify.

## 3. Design — meccanica engine per-round

**Cosa:** nello stesso round, catena MTP di k1 token + blocco DFlash di 7, **un solo
verify** su k1+7+1 colonne (14 con k1=6; 9 con k1=1).

**Meccanica multi-impl per seq (rottura del first-wins):**
- `common/speculative.cpp` `common_speculative_draft` (~:3340-3455, append a :3416):
  quando la policy dice CONCAT per la seq, l'impl **MTP genera k1 token**, poi l'impl
  **DFlash genera il suo blocco (n_max=7) condizionato** sui k1.
- **Meccanismo di condizionamento (esplicita — è il cuore dello stadio):** oggi DFlash
  riceve il contesto solo tramite `process()`/KV-injection di batch decodificati dal
  target (`llama_get_embeddings_layer_inp(ctx_tgt, ...)` a `:1255-1296`) e il suo blocco
  è ancorato a `(dp.id_last @ dp.n_past)` con input placeholder alle posizioni
  `n+1..n+7` (`:1330-1344`). Nel round concat: il blocco DFlash si ancora allo stesso
  `(id_last @ n_past)`, le posizioni `n+1..n+k1` del suo input di drafting ricevono **i
  token MTP draft** (invece del placeholder) e il blocco DFlash vero e proprio occupa
  `n+k1+1..n+k1+7`. Questo richiede plumbing DEDICATO (passaggio dei token MTP
  all'input di drafting DFlash + offset posizioni) — NON è "meccanica esistente".
  **Distinzione dal NO-GO S4 (handoff):** i token MTP entrano SOLO come condizionamento
  di drafting e sono verificati dal target nello STESSO round (mai confermati senza
  verify, mai estesi nel contesto confermato); l'off-distribution del condizionamento è
  il rischio dichiarato §9, quantificato dal gate fase A §7 PRIMA di qualunque rollout
  (S4 era NO-GO perché senza eval dedicata).
- `dp.result` unico per seq (append concatenato); `impl_last` (:2835-2842) attribuisce il
  round all'impl che lo chiude (DFlash).
- **Dispatch di `accept()` multi-impl (OBBLIGATORIO, fix del TODO esistente):** oggi
  `common_speculative_accept` (`:3457-3486`) invia `accept()` SOLO a `impl_last` (c'è
  un TODO esplicito a `:3459-3463` su come estenderlo ai round multi-impl). Nei round
  concat `accept()` va inviato a TUTTI gli impl contributori. **Definizione precisa
  (review 21/08):** `accept()` di MTP è position-based al boundary → riceve l'indice di
  riga TOTALE accettato (`pending = verify_pos_first + min(n_accepted_totale,
  n_rows-1)`), NON il conteggio del solo segmento `[0, k1)` (con accettazione ≥ k1 —
  il caso comune atteso — la lettura per-segmento lascerebbe `pending` indietro e il
  bridge `pending+1 == pos[beg]` di process() non scatterebbe → limite differito
  desincronizzato). Le conte per-segmento alimentano SOLO i contatori/statistiche
  (`concat_mtp_accepted_total`); DFlash riceve il resto. Senza il dispatch multi-impl,
  `pending_pos_last/pending_g_last` di MTP (aggiornati in `accept()` `:913-928`) non
  vengono corretti → limite differito MTP / `get_state()` (`:938`) / cache-checkpoint
  0005-0009 DESINCRONIZZATI → violazione R2. La semantica dell'update MTP con token
  accettati < k1 (prefix parziale) va definita nel piano con test T2 dedicato.
- Verify: `tools/server/server-context.cpp` ~4060-4262 invariato nella logica (il batch è
  già N-token); colonne totali k1+7+1 ≤ 16 col cap dmmv patchato.
- **Statistics per-impl:** con impl_last=DFlash round-closer, i token MTP accettati
  vengono attribuiti a DFlash nei contatori per-impl esistenti. Il marker spec-route
  esteso riporta `k1=<n>` + contatore dedicato `concat_mtp_accepted_total` (metrica
  Prometheus pre-registrata a 0, lezione T7 8b35a795f) per leggere il contributo MTP
  senza ambiguità.

**k1 parametrico:** flag `--spec-concat-k1 <n>` (default **0 = OFF** → comportamento
identico a oggi, bit-per-bit; 0 disattiva il concat ovunque). Valori validi 0..6.
k1=6 per-classe arriva dalla policy §4, non dall'utente.

**Process/trim/checkpoint:** process MTP ungated resta sacro (cache 0005-0009);
trailing-rollback invariato; il checkpoint resta taggato col drafter modello come oggi,
MA la correttezza del limite MTP dipende dal dispatch accept multi-impl sopra.

**Boot (flag REALI del T7 in produzione — MTP è il layer nextn del target, DFlash2 è il
draft-model):** `--spec-type draft-mtp,draft-dflash --spec-draft-model
<DFlash2.gguf> --spec-draft-n-max 7 --spec-concat-k1 <n>` (NON esiste un flag
`--spec-draft-dflash`; `--spec-draft-model` carica DFlash2 nel setup T7). Marker:
`spec-route: concat mode k1=<n>` accanto ai marker duali esistenti. k1=0 → nessun
marker concat, zero delta. k1>0 con routing non duale (override per-request o fallback
mono) → concat OFF per quella request con WARNING (stessa semantica fallback T7-f2
§3.1 T2: mai SRV_ERR).

## 4. Routing (policy F1 evoluta — prosa/reasoning intoccabili)

| Classe (signal policy T7) | Oggi (T7) | Stadio 2 (k1>0) |
|---|---|---|
| reasoning / prosa (no tools) | MTP6 puro | **invariato** — concat MAI attivo su queste classi |
| det / agentic-tools (tools signal) | DFlash7 | **MTP(k1)→DFlash7 concat** |

- Il routing T7 resta l'arbitro: CONCAT è un modo alternativo di comporre il round dove
  la policy già seleziona DFlash; il signal, la classificazione e il fallback mono non
  cambiano.
- Override per-request esistente (`--spec-type` override, check C4 del T1) invariato:
  l'override a singolo drafter disattiva il concat per quella request (concat richiede
  il routing duale).

**temp>0:** verify su segmento misto MTP+DFlash con `spec_dists` → noto assert
(`common/sampling.cpp:841-842` nell'overload con dists,
`GGML_ASSERT(dists.size() == draft.size())`). Comportamento: **fallback automatico al
drafter singolo della classe** (path T7 puro, DF7 su det/agentic a temp>0), WARNING una
volta per sessione nel log. A temp 0 (tutti i gate e la produzione attuale) il path
concat non è mai coinvolto. (NB: l'overload con dists NON è strumentato da
SPEC_VERIFY_LOG — nessun impatto sui gate, tutti a temp 0.)

## 5. Sizing e memoria (pattern T4 della T7-f2)

- `n_draft` effettivo = k1+7 (13 con k1=6): sizing `n_rs_seq`/`n_batch`/`n_ubatch` segue
  il pattern T4 della T7-f2; l'anchor reale del sizing RS è `need_n_rs_seq()`
  (`common/common.h:372`). Il sizing è interno al draft loop, NON tocca i default di boot.
- **ncols VARIABILE per round:** l'early-stop p_min (`--spec-draft-p-min`, confidenza
  draft) può accorciare k1 sotto il valore configurato round per round → colonne verify
  9..14 variabili. Conseguenze: (a) i costi first-use di specializzazione pipeline per
  ncols (§2, 19-142 ms one-off) si pagano per OGNI ncols incontrato — assorbiti dal
  warm-up dei bench, una tantum in produzione; (b) la misura fase A §7 normalizza per
  posizione sul segmento DFlash EFFETTIVO di ogni round (non su posizioni fisse).
- **Budget-check memoria RS (task obbligatorio del piano, PRIMA di qualunque run in
  config produzione):** sul target ibrido il costo RS è lineare in (1+n_rs_seq)
  (`src/llama-memory-recurrent.cpp:99`) → da 8 a 14 slot ≈ memoria RS ×1.75 con
  `--parallel 4`. Verifica headroom UMA con i numeri reali del target (report nel
  piano). NB: qwen35/qwen35moe ∈ `llm_arch_is_hybrid` e supporta rs_rollback
  (`src/llama-arch.cpp:1001-1002, 1024-1033`) — il costo è REALE sul target di
  produzione (smentita report 08 già registrata in spec T8 §4).
- Il cap strict-qwen KV-256-block (`server-context.cpp:2915-2930`) è attivo SOLO con
  `--spec-mtp-strict-qwen` (default false) AND np=1 → in produzione (`--parallel 4`)
  NON clippa k1.

## 6. Fix preliminare obbligatorio: build_post_sampling duplicata (dentro lo Stadio 2)

Finding fase 0 (confermato da review statica indipendente): `build_post_sampling()`
gira DUE volte sui grafi DFlash2 con selettore — chiamata diretta
`src/models/dflash.cpp:736` (introdotta da ebf1cc855) + hook framework
`src/llama-model.cpp:2232`; nessuna guard di idempotenza → lattice selettore costruito
2× nello stesso grafo (correttezza preservata, spreco reale: computo, nodi, pressure su
`graph_max_nodes` DFLASH a `src/llama-context.cpp:2441`).

**Primo task del piano** (decisione utente 21/08): rimuovere la chiamata diretta
(allineamento alla PR #90 che usa solo l'hook), commit separato, verifica = smoke T1
(14/14) + confronto tg su prompt det prima/dopo (atteso: pari o meglio, mai peggio).
Il risparmio di nodi supporta il verify esteso a 14 col.

## 7. Approccio incrementale (APPROVATO UTENTE 21/08): gate intermedio k1=1

Il rischio modello dominante dello stadio — **condizionamento DFlash-su-prefix-MTP mai
misurato** (spec T8 §4: ±12-20% su prosa/reasoning; allineamento blocco DFlash
block_size 8 vs punto di attacco k1 non modellato) — viene quantificato PRIMA di
costruire la policy completa:

- **Fase A (meccanismo + k1=1 su det):** implementare tutto il meccanismo §3 con
  `--spec-concat-k1 1` (9 col, dmmv f32). Misura isolata: acceptance per-posizione del
  blocco DFlash nel round concat (strumento: SPEC_VERIFY_LOG, già nel branch t8-ngram)
  vs acceptance del DF7 puro sulla stessa request det.
  **Gate informativo fase A — definizione precisa:** sia `acc_concat` la frazione media
  di token accettati per posizione sul segmento DFlash effettivo dei round concat
  (posizioni k1+1..fine del round, normalizzata sul segmento presente in ogni round per
  l'early-stop p_min), e `acc_df7` la stessa metrica sul segmento equivalente di una run
  DF7 pura (concat OFF). Il gate è **`acc_concat ≥ 0.90 × acc_df7`** (−10% RELATIVO).
  Dati: stesse 3+ request identiche (D1 + 2 agentic-tools standard T7), temp 0, stesso
  seed/parametri; ≥ 3 run per braccio, report mean±stddev. Decisore: l'utente col report
  in mano (gate informativo, non automatico). Se FALLISCE → lo Stadio 2 si ferma,
  report con causa, decisione utente (niente k1=6).
- **Fase B (k1=6 + policy + A/B finale):** se la fase A passa → k1=6 attivato dalla
  policy sulle classi §4, T1-T5 completi, A/B end-to-end col gate §1.

k1 parametrico serve comunque al k1 adattivo futuro (policy F1 evolve) — non è codice
speculativo.

## 8. Numerica (T5-style, ereditata)

Verify exact prefix-match greedy per-token su path dmmv **f32** (mantenuto dalla patch
VK): output del target deterministico indipendentemente dai draft (i draft anticipano,
non cambiano).

**Nota di implementazione (post Task 1, 21/08 — emersa dal code review):** la patch VK
`mul_mat_vec_max_cols 8→16` cambia il dispatch per OGNI `MUL_MAT` con `dst->ne[1]` in
9..16 a sequenza singola (`ggml-vulkan.cpp:9535`) — incluse eventuali code di prefill
non-speculative di 9-16 token (prima: path tiled con accumulo f16; ora: dmmv f32).
Conseguenze: (a) output potenzialmente NON bit-identici **tra build diverse**
(pre/post patch) per quei batch — i confronti deterministici di questo stadio usano
tutti la stessa immagine t8-concat e restano validi, ma confronti cross-build (vs
produzione T7) vanno letti con questo filtro; (b) i round verify non superano
k1+7+1=14 col < cap 16. Numerica dmmv verificata `rel=0.0000` a 8 e 14 col
(`docs/research/2026-08-21-t8-ncols-curve.md`); CHECK a 15/16 da aggiungere al prossimo
giro di ncols-bench (colonne mai usate dai round, misura di completezza). Test: char-identical del testo generato vs duale attuale sulla stessa
request a temp 0 su (a) prosa/reasoning (path invariato: identità di controllo) e
(b) det/agentic (path concat vs DF7 puro: stesso caveat batched noto del T7). SPEC_VERIFY_LOG
come strumento di diagnosi (per-pos, già validato in fase 0: 21 round + 78 per-pos).

## 9. Rischi dichiarati (i gate coprono; non trattare 11.1/9.1 come certi)

- Condizionamento DFlash-su-prefix-MTP mai misurato (±12-20%) → **gate fase A §7**.
- Power-law calibrata a E≤8, estrapolata a 11.1 → il costo del forward a 14 col nel
  workload reale (dmmv è una frazione del grafo) non è il +47% del kernel isolato →
  **A/B end-to-end è l'unico giudice** (gate §1).
- Budget RS ×1.75 (parallel 4) → **task headroom §5** prima della config produzione.
- bimodalità det (D1/D2 ±20%) → gate primario D1, reporting separato D2 (obbligatorio
  nel report A/B, già regola spec T8).
- graph_max_nodes col verify esteso → mitigato dal fix §6; se emergono assert nodi in
  fase A, il piano prevede l'analisi headroom come follow-up (non silent bump).

## 10. Error handling (ogni fallimento degrada allo status quo T7)

| Fallimento | Comportamento |
|---|---|
| `--spec-concat-k1 0` (default) | bit-identico a oggi (nessun codice concat eseguito) |
| k1>0 ma drafter singolo / override per-request | concat OFF per quella request, path T7 puro, WARNING boot/round |
| k1>0 ma routing seleziona MTP puro (prosa/reasoning) | round MTP6 puro, invariato |
| temp>0 su classe concat | fallback drafter singolo della classe (path T7), WARNING una volta per sessione |
| decode DFlash fallita a metà round concat | verify sul round parziale (k1'+m+1 col, k1' = token MTP effettivi): MAI SRV_ERR, il round si chiude corto |
| interazione `n_min` con `dp.result` condiviso | `effective_n_min`/`effective_n_max` del secondo impl (DFlash) conteggiano anche i token MTP nel round (il truncate a n_max sul result condiviso deve contare i k1): la semantica n_min resta per-SEQ (globale round), non per-impl — test nel T1 smoke |
| acceptance fase A sotto soglia | stadio 2 si ferma prima della fase B (gate §7) |
| headroom RS insufficiente (config produzione) | fase B si esegue in config simple; deploy produzione bloccato finché non risolto (mai OOM silenzioso) |
| assert nodi grafo a 14 col | analisi headroom graph_max_nodes (follow-up pianificato, mai silent bump) |

## 11. Testing (per fase, gate in sequenza)

- **Fase A:** T1 smoke concat (`spec-route: concat mode k1=1`, round 9 col, marker
  nelle statistics); SPEC_VERIFY_LOG confronto acceptance (gate §7); T2/T3 baseline
  ri-cert (k1=0 invariato bit-per-bit); numerica det char-identical.
- **Fase B:** T1 smoke k1=6 (14 col, marker policy per-classe); T2/T3 ri-certificazione
  COMPLETA (simple + config produzione `--parallel 4 --kv-unified`); T4 A/B end-to-end
  vs duale attuale (gate §1: D1 primario, D2 reporting, agentic ≥ 58, prosa ≥ −3);
  T5 numerica completa (prosa identità di controllo + det caveat T7).
- **Piano A/B:** piani esperimento .md dedicati PRIMA dei run GPU (regola lab), ≥5 run
  mean±stddev, marker TREATMENT, `--spec-draft-p-min 0.75` (confidenza del drafter,
  verify, flag T7 esistente), temp 0, c 16384.

## 12. Fuori scope (NO-GO confermati)

Handoff DFlash-su-token-MTP (S4, off-distribution); typical/lenient acceptance (NO-GO
T4); voting/best-of-2; stadio 3 union-tree (filone separato, prereq suoi); qualsiasi
modifica ai kernel verify OLTRE la costante `mul_mat_vec_max_cols` (nessun nuovo
kernel); ncols > 16 (cap architetturale dmmv patchato; round > 16 col fuori scope).

## 13. Vincoli di lavoro (memoria/CLAUDE.md, invariati)

AGENTS.md del repo ROCmFPX prima di toccare codice; branch **`t8-concat` da creare da
`t8-ngram`** (t8-ngram è su gitea al tip e6baf9518; t8-concat nasce in questa sessione
di impl e va pushato su gitea a fine stadio, su conferma utente); patch durature
`patches/t8-concat/` una-patch-per-feature,
format-patch --stdout nome esplicito; build SOLO pipeline staging validata in fase 0
(variante sed toolboxes, vedi `logs/test-t8-ngram/build-notes.txt`); llm-service =
produzione utente (stop/start solo con piano approvato + restart + health); modelli in
`~/llmodels/` mai toccati; attribuzione AI nei commit: `Co-Authored-By: GLM by z.ai
<glm@z.ai>` MAI Claude/Anthropic; suite T7 mai modificate in-place.

## 14. Riferimenti

- Spec padre: `2026-08-20-t8-dual-drafter-synergy-design.md` (§1 requisiti, §4 stadio 2
  originale, §8 vincoli)
- Fase 0: `docs/research/2026-08-21-t8-fase0-session-report.md` (gate, matrice PR #90,
  finding build_post_sampling), `docs/research/2026-08-21-t8-ncols-curve.md` (curva)
- Dati acceptance: report 07 (`docs/research/2026-08-20-t8-strategy-scan/`); engine
  anchor: report 08 (da leggere col filtro RS registrato in spec padre §4)
- Meccanica dual-drafter T7: `2026-08-19-t7f2-drafter-routing-design.md` (in produzione)
