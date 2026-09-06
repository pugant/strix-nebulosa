# T8 Stadio 2 — Concat MTP(k1)→DFlash Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** implementare lo Stadio 2 T8 — round concat MTP(k1)→DFlash(7) con verify unico — con approccio incrementale gate-ato: Fase A (meccanismo + k1=1, gate condizionamento) → Fase B (k1=6 + policy per-classe + A/B end-to-end col gate numerico det ≥ 65).

**Architecture:** il fork ROCmFPX ha il dual-drafter per-request T7 in produzione (MTP nextn ↔ DFlash2 n7 per policy). Lo Stadio 2 rompe il first-wins in `common_speculative_draft`: dove la policy seleziona DFlash, MTP genera k1 token e DFlash completa il blocco condizionato; `accept()` va dispatchato a TUTTI gli impl (position-based per MTP); verify unico su k1+7+1 colonne con patch VK `mul_mat_vec_max_cols 8→16` COMMITTATA (verify f32 exact preservato; emendamento sub-gate approvato utente 21/08).

**Tech Stack:** C++ (fork llama.cpp ROCmFPX), Docker (build pipeline staging documentata), bash smoke scripts (derivate dalle suite T7, MAI modify in-place), SPEC_VERIFY_LOG per acceptance analysis.

**Spec di riferimento (VINCOLANTE, approvata con review loop):** `docs/superpowers/specs/2026-08-21-t8-stadio2-concat-design.md`. Letta PRIMA di ogni task: contiene le definizioni precise (§3 meccanica/accept/boot, §7 gate fase A) emendate dalla review del 21/08.

**Vincoli non negoziabili (spec §13 + memoria lab):**
- Leggere `ROCmFPX/AGENTS.md` prima di toccare codice.
- llm-service = produzione utente: MAI stop/start negli script (eccezione: esperimenti GPU pianificati con approvazione utente; restart a fine run + health via `docker ps`/IP container, MAI host :1234). Al 21/08 sera era FERMO dall'utente: verifica lo stato a inizio sessione e chiedi se serve riavviarlo o tenerlo fermo.
- Modelli in `~/llmodels/` MAI toccati. Build SOLO con la pipeline staging validata in fase 0 (vedi Task 1).
- Ogni esperimento GPU: piano .md PRIMA del run, marker TREATMENT, `--spec-draft-p-min 0.75`, temp 0, c 16384, ≥5 run mean±stddev (bench end-to-end) o ≥3 run (gate fase A, come da spec §7).
- Commit col trailer `Co-Authored-By: GLM by z.ai <glm@z.ai>` MAI Claude/Anthropic. Branch `t8-concat` DA CREARE da `t8-ngram` (tip atteso `e6baf9518`, su gitea). Patch durature `patches/t8-concat/`.
- Suite T7 (`scripts/test-drafter-routing-t*.sh`) runnate ORIGINALE mai modificate in-place; i test nuovi sono script NUOVI derivati (copiati).
- Lezioni lab rigide: verify timestamp immagine DOPO build e PRIMA dei run; exit code veri (mai `cmd; echo $?` in wrapper); poll health con timeout esplicito (no sleep fissi lunghi); `docker logs` su stderr (`2>&1`); nessun carattere CJK nei testi italiani; numeri col punto decimale.
- SPEC_VERIFY_LOG è una env il cui valore è un PATH file (non "=1").

---

## Chunk 1: Fase A — meccanismo + k1=1 + gate condizionamento

### Task 1: Branch t8-concat + patch VK committata + build + smoke baseline

**Files:**
- ROCmFPX (git): branch `t8-concat` da `t8-ngram`
- Modify: `ROCmFPX/ggml/src/ggml-vulkan/ggml-vulkan.cpp:324` (`mul_mat_vec_max_cols = 8` → `16`)

- [ ] **Step 1: Crea il branch**: `cd ~/workspaces/rocmfpx-strix-lean/ROCmFPX && git checkout t8-ngram && git log -1 --oneline` (atteso `e6baf9518`; se diverso, FERMATI e riporta) `&& git checkout -b t8-concat`.
- [ ] **Step 2: Patch VK**: cambia `mul_mat_vec_max_cols = 8;` in `= 16;` a `ggml-vulkan.cpp:324` (unica assegnazione; le altre occorrenze sono usi — verify con grep). Commit `vulkan: raise mul_mat_vec_max_cols to 16 for wide spec-verify rounds [t8 stadio 2]` + trailer GLM. Motivo nel body: emendamento sub-gate 21/08 (costo-per-token −16% a 14 col, f32 exact preservato; riferimento spec §2 e curva `docs/research/2026-08-21-t8-ncols-curve.md`).
- [ ] **Step 3: Build**: riusa la pipeline documentata in `logs/test-t8-ngram/build-notes.txt` + `build.log` header (staging /tmp da `git archive HEAD`, sed toolboxes, cmake --install /usr, backends statici). Tag `docker-llm-service:vulkan-fork-t8-concat`. Verify: BUILD_EXIT=0, `docker images` mostra il tag con timestamp DI ADESSO, e double-tag temporaneo `docker-llm-service:vulkan-fork-dflash2-route` SOLO se serve per lo script T1 (hardcoded riga 25) — ricorda che il double-tag orfana l'immagine precedente.
- [ ] **Step 4: Smoke baseline**: `scripts/test-drafter-routing-t1.sh` ORIGINALE con l'immagine nuova (via double-tag o env se disponibile) → atteso **14/14** (la patch VK a ncols ≤ 8 non cambia il path dmmv: probe `GGML_VK_PERF_LOGGER=1` opzionale su una request per conferma `_VEC`).
- [ ] **Step 5: `docker rm -f`** dei container di test; log in `logs/test-t8-concat/` (crea la dir: build.log, t1-console.log).

### Task 2: Fix build_post_sampling duplicata

**Files:**
- Modify: `ROCmFPX/src/models/dflash.cpp` (~:736, rimozione chiamata diretta)
- Reference: `src/llama-model.cpp:2232` (hook framework — resta), override `dflash.cpp:740`

- [ ] **Step 1: Baseline tg**: 3 run di un prompt det (D1 standard T7, temp 0, p_min 0.75, c 16384) sull'immagine del Task 1; salva tok/s mean±stddev (TSV con marker TREATMENT) in `logs/test-t8-concat/dup-build-baseline.tsv`.
- [ ] **Step 2: Impl**: rimuovi la chiamata diretta `build_post_sampling()` nel costruttore `graph<false>` di dflash.cpp (~:736, introdotta da ebf1cc855). L'hook framework in llama-model.cpp:2232 la raggiunge via override (verificato da review statica fase 0: dispatch virtuale intatto).
- [ ] **Step 3: Build + verify**: rebuild (pipeline Task 1 Step 3), verify timestamp, smoke T1 duale → 14/14.
- [ ] **Step 4: Confronto tg**: stessi 3 run del Step 1 → atteso pari o meglio (MAI peggio oltre il rumore ±3%); salva `dup-build-after.tsv`.
- [ ] **Step 5: Commit** `dflash: remove duplicate build_post_sampling call (framework hook suffices) [t8 stadio 2]` + trailer GLM (nel body: finding fase 0, lattice 2×, RIFERIMENTO report fase 0 §Finding).

### Task 3: Flag --spec-concat-k1 (arg parsing + boot marker + guard)

**Files:**
- Modify: `ROCmFPX/common/common.h` / `common/arg.cpp` (nuovo campo `spec_concat_k1` default 0)
- Modify: `ROCmFPX/tools/server/server-context.cpp` (load_model zona draft: guard + marker)
- Test: `scripts/t8-concat-t1.sh` (NUOVO, copiato da `scripts/test-drafter-routing-t1.sh` — mai modify in-place)

- [ ] **Step 1: Scrivi il test di boot** (prima dell'impl): da `test-drafter-routing-t1.sh` copiato in `t8-concat-t1.sh`, aggiungi check: (a) boot con `--spec-type draft-mtp,draft-dflash --spec-draft-model <DFlash2.gguf> --spec-draft-n-max 7 --spec-concat-k1 1` → marker `spec-route: concat mode k1=1` nel log; (b) `--spec-concat-k1 0` (o assente) → NESSUN marker concat e boot identico a T7; (c) `--spec-concat-k1 1` con `--spec-type` singolo → WARNING + concat OFF (mai SRV_ERR); (d) valore invalido (>6) → WARNING + OFF o rifiuto config 400 (decidi e documenta nel test). Il path del GGUF DFlash2: LEGGILO dagli script T7 esistenti (`scripts/test-drafter-routing-t1.sh` / `lucebox-run/scripts/run-ds4.sh`) — non inventarlo.
- [ ] **Step 2: Run → FAIL** (flag non riconosciuto: server esce o 400).
- [ ] **Step 3: Impl**: campo `spec_concat_k1` (uint, default 0) in common.h; parsing in arg.cpp (accetta 0..6); in server-context.cpp load_model: se k1>0 E routing duale attivo E drafter DFlash caricato → marker INFO `spec-route: concat mode k1=<n>`; se k1>0 ma condizioni mancanti → WARNING + concat OFF. Il flag NON attiva ancora alcuna meccanica di round (Task 4): qui solo config+marker. ~30-40 LOC.
- [ ] **Step 4: Build + run test → PASS** (tutti i check a/b/c/d).
- [ ] **Step 5: Commit** `feat(spec): --spec-concat-k1 flag with boot marker and guards [t8 stadio 2]` + trailer GLM.

### Task 4: Meccanismo concat nel draft loop (IL CUORE)

**Files:**
- Modify: `ROCmFPX/common/speculative.cpp` (`common_speculative_draft` ~:3340-3455, first-wins :3416; KV-injection process :1255-1296; blocco DFlash :1330-1344)
- Modify: `ROCmFPX/common/common.h` (`need_n_rs_seq()` :372 — sizing RS, vedi Step 3)
- Input vincolante: spec §3 (meccanica di condizionamento + effective_n_min/n_max truncate) — NON reinventare, seguire le definizioni letterali.

- [ ] **Step 1: Test funzionale-first**: estendi `scripts/t8-concat-t1.sh` con request det (D1, tools signal ATTIVO — vedi come T1 C3 costruisce il body) con `--spec-concat-k1 1`: attese (a) il round nei log/statistics mostra 1+7+1=9 colonne max (SPEC_VERIFY_LOG: righe R con ncols ≤ 9); (b) marker round con `k1=1`; (c) request prosa (no tools) → round IDENTICO a T7 (zero impatto, 8 col max); (d) `--spec-concat-k1 0` → output bit-identico a run senza flag (char-diff del testo generato); (e) i round concat con rejection PERCORRONO il fast-path RS, NON checkpoint-restore completo (grep nel log server: nessun picco di restore nei round concat; vedi Step 3 — senza sizing RS ogni rejection uscirebbe dal fast-path, corretto ma lentissimo e silenzioso). Run → FAIL.
- [ ] **Step 2: Impl nel draft loop**: per la seq con policy CONCAT: (1) impl MTP genera k1 token; (2) plumbing: le posizioni `n+1..n+k1` dell'input di drafting DFlash ricevono i token MTP (invece del placeholder `mask_token_id` a :1330-1344), blocco DFlash a `n+k1+1..n+k1+7`, ancoraggio `(id_last @ n_past)` invariato; (3) append concatenato in `dp.result` (unico per seq); (4) `impl_last` = DFlash; (5) il truncate a n_max e `effective_n_min`/`effective_n_max` del secondo impl CONTEGGIANO i k1 token MTP (spec §10: semantica per-SEQ); (6) early-stop p_min può accorciare k1 round-per-round → ncols variabile 9..(k1+8) è COMPORTAMENTO ATTESO (spec §5); (7) **decode DFlash fallita a metà round** (path esistente `llama_encode` failed → return false ~:1283-1287): chiudi il round CORTO con i soli k1' token MTP già draftati, verify su k1'+1 col, MAI SRV_ERR (spec §10). Verify e round NON si toccano.
- [ ] **Step 3: Sizing RS (spec §5 — senza questo, ogni rejection nei round concat esce dal fast-path RS)**: estendi `need_n_rs_seq()` (`common/common.h:372`, alimenta `cparams.n_rs_seq` via `common.cpp:1596`) a coprire `draft.n_max + k1` quando concat attivo (n_draft effettivo = k1+7); `n_batch`/`n_ubatch` secondo il pattern T4 della T7-f2. Verify: con k1=6 e rejection, `n_rollback > llama_n_rs_seq` NON scatta (server-context.cpp:4091-4092) → checkpoint-restore completo NON percorso nei round concat normali.
- [ ] **Step 4: Build + run → PASS** (check a-e del Step 1; SPEC_VERIFY_LOG attivo: env = path file dentro il container, es. `/tmp/spec-verify-concat.log`).
- [ ] **Step 5: Spot-check numerica**: stessa request det 2× (k1=1 vs k1=0) → testo generato char-identical (verify exact greedy; caveat batched T7 noto — se differisce, INDAGA con SPEC_VERIFY_LOG per-posizione prima di accettare).
- [ ] **Step 6: Commit** `feat(spec): concat round MTP(k1) head + DFlash block conditioned [t8 stadio 2]` + trailer GLM. **Invarianti sacri di non-regressione** (verifica nel self-review): process MTP ungated resta sacro (cache 0005-0009), trailing-rollback invariato, checkpoint taggato col drafter modello come oggi.

### Task 5: Dispatch accept() multi-impl (fix TODO :3459-3463)

**Files:**
- Modify: `ROCmFPX/common/speculative.cpp` (`common_speculative_accept` :3457-3486; accept MTP :913-928; get_state :938)
- Input vincolante: spec §3 definizione precisa (position-based TOTALE per MTP, NON per-segmento).

- [ ] **Step 1: Test cache-first** (prima dell'impl): scenario T2-like con k1=1 su request det ripetute (stessa conversazione, 2+ turni): attese (a) cache reuse attivo (contatore cache-hit come nel C7 del T1: cached ≈ atteso, delta piccolo), (b) NESSUN cold reload spurio, (c) checkpoint round-trip integro. Deriva da `scripts/test-drafter-routing-t2.sh` in `scripts/t8-concat-t2.sh` (copia) aggiungendo `--spec-concat-k1 1`. Run → può già FALLIRE (accept non dispatchato → limite MTP desincronizzato).
- [ ] **Step 2: Impl**: nei round multi-impl, `accept()` a TUTTI i contributori. MTP: position-based al boundary — `pending = verify_pos_first + min(n_accepted_totale, n_rows-1)` (spec §3 verbatim); conte per-segmento SOLO nei contatori (Task 6). DFlash: resto come oggi. Il caso prefix parziale (n_accepted < k1): il limite MTP si ferma al boundary coerente — copri con un assert morbito/log di debug + test nel Step 3. **Invarianti sacri** (spec §3 process/checkpoint): process MTP ungated sacro (cache 0005-0009), trailing-rollback invariato, checkpoint taggato col drafter modello — la correttezza del limite MTP dipende da QUESTO dispatch.
- [ ] **Step 3: Run t8-concat-t2.sh → PASS** + prefix parziale forzato (request det con risposta che si interrompe a fine-round: max_tokens calibrato) → nessun desync (checkpoint salvato/ripristinato coerente).
- [ ] **Step 4: Commit** `fix(spec): dispatch accept() to all drafting impls in concat rounds [t8 stadio 2]` + trailer GLM.

### Task 6: temp>0 fallback + statistics/metriche

**Files:**
- Modify: `ROCmFPX/tools/server/server-context.cpp` (print_timing + registration metrics; zona verify ~4060-4262 per la guard temp>0)
- Modify: `ROCmFPX/common/speculative.cpp` (solo se la guard vive nel draft loop)

- [ ] **Step 1: Test**: (a) request det con `temperature 0.7` e k1=1 → atteso WARNING una volta per sessione + round DF7 puro (path T7), output valido; (b) `/metrics` (con `--metrics`) espone `spec_route_concat_mtp_accepted_total` pre-registrata a 0 al boot (lezione T7 8b35a795f) e > 0 dopo request concat con accettazione MTP; (c) marker INFO `ngram`-style accanto alle statistics per-slot: `statistics concat: k1=<n> rounds=<n> mtp_accepted=<n>/<k1_tot>`.
- [ ] **Step 2: Impl**: guard temp>0 PRIMA del round concat (spec §4: fallback drafter singolo della classe, WARNING once); contatore Prometheus `spec_route_concat_mtp_accepted_total` (pre-registrazione a 0) + marker statistics.
- [ ] **Step 3: Build + run → PASS 3/3.**
- [ ] **Step 4: Commit** `feat(server): concat observability + temp>0 single-drafter fallback [t8 stadio 2]` + trailer GLM.

### Task 7: Baseline ri-cert k1=0 + suite fase A completa

- [ ] **Step 1: T1 duale ORIGINALE** (14/14) sull'immagine t8-concat — regressione globale del meccanismo con concat OFF.
- [ ] **Step 2: t8-concat-t1.sh completo** (tutti i check dei Task 3-6) → PASS.
- [ ] **Step 3: t8-concat-t2.sh + t3 ORIGINALE** (7/7) in config simple → PASS.
- [ ] **Step 4: Log** in `logs/test-t8-concat/` (già aperta nel Task 1).

### Task 8: Gate fase A — condizionamento DFlash-su-prefix-MTP (DECISIONE UTENTE)

**Files:**
- Create: `docs/superpowers/plans/2026-08-2X-t8-stadio2-faseA-gate.md` (piano esperimento, PRIMA del run)
- Create: `scripts/bench-t8-fasea.sh` (derivato da `scripts/bench-routing-vs-mono.sh`)
- Create: `docs/research/2026-08-2X-t8-fasea-conditioning.md` (report)

- [ ] **Step 1: Piano esperimento .md**: bracci CONCAT-k1=1 vs DF7-puro (concat OFF); metrica `acc_concat`/`acc_df7` come spec §7 (frazione media di token accettati per posizione sul segmento DFlash effettivo, normalizzata per early-stop p_min); dati: stesse 3+ request identiche (D1 + 2 agentic-tools standard T7), temp 0, stesso seed; ≥3 run per braccio, mean±stddev; strumento SPEC_VERIFY_LOG (env path) + statistics; marker TREATMENT; gate `acc_concat ≥ 0.90 × acc_df7` (RELATIVO). Data concreta nel nome file al posto di 2026-08-2X.
- [ ] **Step 2: Run** (GPU: verifica stato llm-service e chiedi conferma utente se interferisce; i run sono brevi — 3 request × 3 run × 2 bracci).
- [ ] **Step 3: Report** con tabella per-posizione (SPEC_VERIFY_LOG), acc mean±stddev, verdetto gate, e lettura del condizionamento (dove cade l'acceptance: prime posizioni dopo il prefix MTP vs coda).
- [ ] **Step 4: DECISIONE UTENTE** (gate informativo, non automatico): PASS → Fase B (Chunk 2). FAIL → Stadio 2 si ferma: report con causa, update memoria, chiusura filone qui.

---

## Chunk 2: Fase B — k1=6 + policy + certificazione + A/B finale (SOLO se gate fase A PASS)

### Task 9: Policy per-classe k1=6

**Files:**
- Modify: `ROCmFPX/tools/server/server-context.cpp` (policy F1: dove seleziona DFlash → concat k1=6; override per-request invariato)
- Test: estensione `scripts/t8-concat-t1.sh`

- [ ] **Step 1: Test-first**: boot k1=6; request det/agentic-tools → round 6+7+1=14 col max (SPEC_VERIFY_LOG R con ncols ≤ 14, marker `k1=6` variabile per early-stop); request prosa → round MTP6 puro invariato (8 col max, ZERO marker concat); override per-request a singolo → concat OFF per quella request; **ripeti il check (e) del Task 4 a k1=6** (round concat con rejection sul fast-path RS, NON checkpoint-restore — qui la clausola verify "con k1=6" del Task 4 Step 3 trova esecuzione). NESSUN test con `--spec-mtp-strict-qwen` (cap KV-256-block attivo solo con quel flag AND np=1 — fuori dallo scope, default false).
- [ ] **Step 2: Impl**: la policy T7 resta l'arbitro; CONCAT è la composizione del round dove la policy seleziona DFlash. k1 configurato = CAP (6), effettivo per round variabile (early-stop).
- [ ] **Step 3: Build + run → PASS. Commit** `feat(spec): per-class concat k1=6 via routing policy [t8 stadio 2]` + trailer GLM.

### Task 10: Headroom RS/memoria in config produzione (PRIMA dei run prod-config)

**Files:**
- Create: `docs/research/2026-08-2X-t8-concat-rs-headroom.md`

- [ ] **Step 1: Misura**: boot in config produzione (`--parallel 4 --kv-unified` + flag duali T7, vedi `lucebox-run/scripts/run-ds4.sh` e piani T7 per la config esatta) su porta dedicata, k1=6: osserva VRAM/UMA (rocm-smi / docker stats) a riposo e sotto carico 4 seq; confronta col duale attuale. Anchor sizing: `need_n_rs_seq()` `common/common.h:372`; costo lineare (1+n_rs_seq) `src/llama-memory-recurrent.cpp:99`; atteso ≈ RS ×1.75 (spec §5). **EMENDATO 21/08 (post fix ring-window, decisione utente):** il floor RS `max(round_width, 16)` (commit 59982de3f) porta `n_rs_seq` 8→16/17 anche a k1=0 → **budget memoria RS atteso ≈ ×2.125 vs duale pre-T8** ((1+16)/(1+7)), non ×1.75; l'invariante "trailing-rollback invariato" è SUPERSEDATO dalla finestra ring MTP 17-posizioni (decisione utente 21/08 dopo triage S3-a: la patch VK f32 esponeva la finestra fragile preesistente). L'headroom va certificato contro questa nuova baseline.
- [ ] **Step 2: Report** con numeri e verdetto: headroom OK / insufficiente (in tal caso: fase B si certifica in config simple e il deploy produzione resta bloccato finché non risolto — mai OOM silenzioso, spec §10).
- [ ] **Step 3: Nessun commit** (docs). Se emergono assert graph_max_nodes a 14 col: analisi headroom come follow-up documentato (mai silent bump della costante).

### Task 11: Certificazione completa T1/T2/T3 (simple + produzione)

- [ ] **Step 1: T1 duale ORIGINALE 14/14** (regressione: il duale non rotto dal k1=6 in produzione-config).
- [ ] **Step 2: t8-concat-t1/t2 completo** in config simple E config produzione (`--parallel 4 --kv-unified`): marker, cache round-trip 4/4, checkpoint, temp>0 fallback.
- [ ] **Step 3: t3 ORIGINALE 7/7** in entrambe le config (percorsi sacri RS).
- [ ] **Step 4: Numerica T5**: char-identical prosa (identità di controllo: path invariato) + det concat vs DF7 puro entro caveat batched T7 (spec §8).
- [ ] **Step 5: Log** in `logs/test-t8-concat/`.

### Task 12: A/B end-to-end finale (gate numerico — GPU, piano dedicato + approvazione utente)

**Files:**
- Create: `docs/superpowers/plans/2026-08-2X-t8-stadio2-ab.md` (PRIMA del run)
- Create: `scripts/bench-t8-concat-vs-duale.sh` (derivato da `scripts/bench-routing-vs-mono.sh`)

- [ ] **Step 1: Piano esperimento**: bracci DUALE-attuale (produzione, immagine split-restore) vs CONCAT-k1=6 (immagine t8-concat); prompt: D1 primario + D2 reporting obbligatorio (bimodalità ±20%) + A1-A3 agentic + P1-P2 prosa (gate prosa ≥ −3%); ≥5 run mean±stddev, temp 0, c 16384, `--spec-draft-p-min 0.75`, marker TREATMENT; gate spec §1: **det ≥ 65 (D1), agentic ≥ 58, prosa ≥ −3%**; TSV + response salvate; data concreta nel nome file.
- [ ] **Step 2: APPROVAZIONE UTENTE** per GPU dedicata (stop llm-service se attivo; restart a fine run + health).
- [ ] **Step 3: Run A/B** → `logs/bench-t8-concat/`.
- [ ] **Step 4: Report** `docs/benchmarks/results-2026-08-2X-t8-concat.md`: gate PASS/FAIL per metrica, confronto E[round] misurato vs atteso (11.1/9.1), tok/s power-law check, analisi per-posizione (SPEC_VERIFY_LOG).

### Task 13: Chiusura Stadio 2

- [ ] **Step 1: Patch duratura (una-patch-per-feature, file per commit)**: `mkdir -p patches/t8-concat` POI `git format-patch t8-ngram..t8-concat -o patches/t8-concat/` (file numerati per commit, nomi espliciti generati da git — niente `ls -t|grep`, niente --stdout multi-commit in un solo file).
- [ ] **Step 2: Gate PASS → proponi rollout produzione** (GATED utente: switch immagine + backup config + osservazione 24h, meccanica T7 Task 12). Gate FAIL → report causa + decisione (ritoccare k1 / chiudere stadio 2).
- [ ] **Step 3: Push branch su gitea** (`git -c credential.helper=store push gitea t8-concat`) — SOLO su conferma utente.
- [ ] **Step 4: Update memoria** (`project-t8-sinergia-dual-drafter.md`: gate fase A e finale PASS/FAIL + decisioni) e spec se emendata in corsa.

---

## Note per l'esecutore

- **Ordine rigido**: i Task 4-5 dipendono dalle definizioni LITERALI della spec §3 (meccanica condizionamento, accept position-based totale, effective_n_min/n_max che contano i k1). Non reinventare: aprire la spec a ogni task engine.
- **Fermate obbligatorie**: Task 8 Step 4 (decisione utente gate fase A — se FAIL il Chunk 2 NON si esegue), Task 12 Step 2 (approvazione GPU), Task 13 Step 3 (push).
- Ogni build ~3-7 min: batchare build dove sensato (es. Task 4 e 5 insieme se il flusso lo consente, MA commit separati).
- Boot del modello ~1-2 min: poll health con timeout 300s, mai sleep fissi.
- Il testo dei report è italiano, i numeri col punto, zero CJK. Commit message in inglese (convenzione repo).
- Se un anchor file:riga drifta di poche righe (le modifiche del branch spostano il codice), cerca nelle vicinanze e verifica la SEMANTICA prima di procedere; se un anchor è strutturalmente diverso, FERMATI e riporta (NEEDS_CONTEXT) invece di forzare.
