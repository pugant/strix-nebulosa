# Wave-7 — FR-Spec, the router requant and the last host sweep (2026-09-11)

Wave-6 closed the round engine; wave-7 was built around one main lever — FR-Spec,
a frequency-ranked trim of the drafter's vocabulary — plus the one trunk change
that survived the wave, two bit-exact host-side cards and a zero-code
decomposition of the last big round gap. Deployed as
`qwen4exp-mtp-vk-optim-w7` (15 commits over `optim-w6` `f37da5c0c`, branch
`optim-w7` @ `b21239c06`; image `06c978c5fef5` built from that exact tree,
verified by `diff` of the source context).

**Final interleaved A/B (containers, prod-style config): +3.9% t/s on means,
+4.5% on medians** — B 28.07/27.60/27.18 vs A 26.38/26.42/26.94 tok/s (timed-gen
600 tok, 3+3 arms), **battery 12/12 bit-identical** to the pinned fingerprint set
(`6cc4f583…`) on the release image running the exact deploy config. The
pre-declared ≥5% gate was **not** reached: the FR-Spec card (a scoped work item,
the wave's main lever) measured NO-GO, so the cumulative delta rests on the
router requant alone (+4.24% in the ctx-32k session-regime cell). Deployed on an
explicit operator go-ahead below the gate, with that composition stated openly.
Live since 2026-09-11 01:33.

## What shipped

15 commits, merged in the order W7-5 → W7-3 → W7-4 → W7-2 → W7-1, each behind a
per-card review; shipped as the patch series `patches/optim-w7-series/` (one
patch per commit, `git am` on a clean tree — see [PATCHES.md](../../PATCHES.md)
for the apply chain).

- **W7-2 — MoE router requant, the lever that moved** (`2440e6001`, `221a237f0`):
  the 48 `blk.N.ffn_gate_inp.weight` routers go F32 → `Q4_0_ROCMFP4` (double
  UE4M3 scale, imatrix-guided, the same quant family as the rest of the trunk).
  Each router drops 5.000 MiB → 737,280 B (0.703 MiB), i.e. 240.0 → 33.75 MiB
  of router weights read per forward; counting the 48 one-dimensional
  `ffn_gate_inp_shexp` gates that stay F32 (+0.47 MiB), the router path falls
  240.5 → 34.2 MiB per forward (**7.03×**). This needed an opt-in quantizer
  patch: upstream excludes gating tensors from quantization and `--tensor-type`
  was inert under `--pure`; the guard now lifts *only* when a pattern explicitly
  names the tensor, and `--pure` + non-empty patterns converts exactly the named
  tensors while copying everything else verbatim (the W6-3 head-requant recipe
  still converts exactly its one output tensor). Gates: perplexity en −0.27% /
  it +0.37% (both inside the ±0.5% gate), battery 12/12 bit-identical, tg parity
  (27.105 vs 27.14 tok/s median), **c32k +4.24%** (27.01 → 28.155 median,
  recomputed from the raw logs). This artifact — 206 MiB smaller — is the
  deployed trunk.
- **W7-5 — allocation-free BPE session** (`d057d6864`): the census showed the
  merge loop at ~77% of tokenize with ~13 heap string allocations per merge; the
  bigram-stale check now compares symbol sizes (the pattern already in-tree for
  SPM), the scratch buffers are members, the byte-encoder drops its per-word
  decode→encode round-trip (valid UTF-8 by construction), and the 256-entry
  byte→UTF-8 table replaces the map lookups. Bit-identity proven on 16 real agent
  rounds with a three-leg chain (pre-change binary, post-change binary, and an
  independent pinned Python tokenizer): 16/16 identical token streams and
  formatted text. Full tokenize **1.80×** (741.2 → 412.3 ms on the 412 KB agent
  corpus), steady-state prefix-cache splice **~2.5×** (0.10–0.33 → 0.04–0.13 ms
  for a 66 KB round). The per-request path now sits at ~20 ms — the ≤15 ms
  informational target was *not* reached, because 75–80% of what remains is the
  Jinja template `execute` itself (structural; the incremental-render card is
  identified but out of budget). Multi-threaded tokenize: NO-GO, justified — it
  only helps the first round of a conversation and would contend with the HTTP
  thread pool.
- **W7-3 — k-slot graph-result cache, rev-2** (`ce5828405`, `bf06efb4d`; opt-in,
  default OFF: `LLAMA_GRAPH_CACHE_SLOTS`, 0/1 = legacy single-slot, ≥2 active,
  cap 8): the W6-9 crash (cached graph holding cross-backend copies from a freed
  context) is fixed by snapshotting the as-built topology and restoring it
  before re-binding. Mandatory sequence respected: lavapipe+CPU multi-backend
  gates first (hashes identical k=0 vs k=2/k=3, counters field-exact vs the
  control), then real Vulkan cells: zero crash signatures, 170 deterministic
  rebinds, acceptance identical (0.68681) in all four arms, trunk-context
  rebuilds 262 → 92. t/s +2.0% median — noise-dominated, the top of the honest
  0–2% band for a host lever in a GPU-bound round. One of the two graph contexts
  still thrashes at k=3 (989 rebuilds, 0 hits, 986 evictions) — the follow-up is
  a per-context k. Ships inert.
- **W7-1 — host-side element-wise fusions** (`71ce79400`, `bcfaf9044`,
  `b21239c06`): two metadata/graph-only changes — rechunk the hc-norm gammas to
  stream layout `[n_embd, hc]` (same storage bytes, loader-side, with contiguity
  asserts) and drop the identity reshape on the GDN alpha path — so two fusion
  matchers that already existed in the Vulkan backend finally see adjacent,
  same-shape nodes. On real VK the counters move: **RMS_NORM_MUL 292 → 371,
  MUL_MAT_ADD 0 → 12** per decode round. Bit-exact by construction (same shaders,
  same reduction order, bias folded in-register) and by gate: graph-hash diff
  empty, 813/813 state writes identical, mini-model GGUFs sha-identical to the
  wave-6 fixtures. Acceptance identical to the 4th decimal (0.68681/0.74286);
  t/s parity within noise (median −1.5%, spread ±1.5% — the census-predicted
  +0.35–0.6% is not resolvable at n=2). The remaining element-wise pairs need
  shaders that do not exist yet; documented, not attempted.
- **W7-4 — the FR-Spec port, merged as a lossless no-op** (`680ac40e7`,
  `0bb283c56`, `0ba73354f`, `103592153`, `76df54cfe`, plus disclosure and
  session-dedup `d98d2d60b`, `3911bf4cd`): five commits porting the d2t
  draft-vocabulary trim onto our qwen4exp loader and `graph_mtp` (net +55/−1,
  semantically identical to the reference series, two conflicts on the divergent
  drafter graph resolved by hand and documented). Without a `d2t` tensor in the
  GGUF, the code paths are inert; with one, the drafter head computes logits
  over only the kept rows and maps them back to full-vocabulary ids, while the
  target verifies at full vocabulary — lossless by construction.

## What did not ship, and why (measured)

- **The FR-Spec artifact — NO-GO on speed, and the story is worth keeping.**
  FR-Spec trims the drafter's output head to the K most frequent vocabulary rows
  (a d2t draft-to-target index maps back; the inverse t2d path is ported and
  verified but unused, matching the reference trimmer). Our frequency map is
  1.54 M tokens of the real workload mix (agent traces counted five times,
  C++/Python/shell code, English and Italian docs, agent JSON): 23,335 observed
  ids, plus the 33 control/user-defined specials deliberately pre-seeded so the
  drafter can still propose `<|im_end|>` and friends (a correctness-neutral,
  acceptance-preserving, opt-out deviation), plus BPE-ordered filler to K. The
  lossless gates all passed, twice (65k and 128k): greedy generation was
  bit-identical on 3 prompts including CJK whose tokens fall *outside* the keep
  set (the target verifies at full vocabulary), and a cross-backend harness
  showed the kept rows' logits bit-identical to the untrimmed head, with the
  other ids at exactly −inf. Then the cells: at **65k rows** (head −50%, 357.6 →
  178.3 MB) acceptance fell to 0.88× on the thinking-heavy 600-token workload
  and t/s lost 3.9% — on short prompts acceptance was *above* baseline, so the
  loss is workload-dependent, not mechanism damage. The pre-authorized fallback
  at **128k rows** raised acceptance to 1.049× (replicating the reference
  measurement) — but the Q8 trimmed head at 128k rows is 356.5 MB against our
  production Q4 full-vocab head's 357.6 MB: **byte parity by construction, no
  bandwidth margin left to win**. The lever only pays against a Q8 full-vocab
  head; ours is already Q4. Artifacts stay out of production and off HF; the
  port stays in the engine (no-op without the tensor) for future drafters or a
  thinking-heavy map.
- **W7-6 — the 20 ms draft→verify gap, decomposed (zero-code card)**: reproduced
  at 20.7 ms/round (baseline 20.0, +3.2%), generation bit-identical under the
  same seed, with 6,423 barriers and 290 submits per round unchanged. The
  decomposition — async run vs a forced-sync run that moves GPU drain inside the
  dispatch — splits it into **62% serial host code (12.9 ms)** and **38% GPU
  drain (7.8 ms of real overlapped work)**. The round is ~89% GPU-saturated
  (idle ~14 ms), so host-only fixes cap at **~4–8 ms/round (3–6% wall)**, not
  the nominal 20 ms. The slices map onto W7-3 (graph rebuild, ~1.7–3.5 ms) and
  W7-4 (the lm_head matvec that dominates drain time); an overlap card would be
  worth opening only if residual idle stayed ≥10 ms after those — it did not, so
  no card this wave.

## Final validation

| Gate | Result |
|---|---|
| Battery 12/12 on the release image, deploy config | bit-identical, set `6cc4f583…` |
| A/B final, interleaved 3+3, containers | A 26.38/26.42/26.94 · B 28.07/27.60/27.18 → **+3.9% means / +4.5% medians** |
| Perplexity (router trunk) | en 3.2820 → 3.2733 (−0.27%) · it 1.1768 → 1.1811 (+0.37%), both ≤ +0.5% |
| c32k session-regime cell (router) | 27.01 → 28.155 median (**+4.24%**) |
| Whole-range final review (spec→plan→code→gates) | APPROVED, 8 findings all non-blocking |
| GPU ledger, reconciled from process timestamps | ~10.6k s ≈ 2 h 56′, dominated by the perplexity gate (thermal-throttled, declared) |

The first attempt at the final window aborted (arm B dead on start: the park
mirror flag takes a MiB value, not a boolean — `stoi`); it was restarted clean
and only the completed window was read. Both attempts are recorded in the
dispatch registry.

## Deploy configuration

Image `qwen4exp-mtp-vk-optim-w7`; trunk = the router-requant artifact (sha256
`13625e32…`, verified after the swap); drafter Q4-head unchanged; park mirror ON
(`--cache-disk-park-ram-mirror 3072`). Boot markers checked post-deploy: park
enabled, mirror enabled at 3072 MiB, tokenizer prefix cache enabled, correct
trunk/drafter/mmproj filenames. Rollback: L1 — swap the trunk back to LEAN and
disable the mirror; L2 — re-pin the `optim-w6` image. The compose file and start
script were backed up before the switch.

## Credits

- **avifenesh** — the original d2t draft-vocabulary trim for the MTP sidecar
  (commit `047bfa508`), which our FR-Spec port descends from.
- **drluoto** — the qwen4exp port of that trim, the reference trimmer and
  frequency map, and the Strix Halo measurements that motivated the card (the
  drafter lm_head at ~81% of per-token draft bytes; acceptance 0.78 → 0.81 at
  65k rows).
- **quimmedes** — the community Q8_0 MTP head our requantized production
  sidecar derives from (lineage documented in our drafter research).

The d2t port is incorporated in the engine even though the FR-Spec artifact is
NO-GO — the credit is due for the code, and given.

## Process

Step-0 census and pinning before any code; two-stage review per card (numbers
vs sources, then language) plus a whole-range final review over the merged
series; interleaved verdicts only; GPU spend reconciled from process
timestamps (~2 h 56′ total; the ledger was dominated by the perplexity gate at
thermal throttle, declared). Full Italian report: `docs/optim/W7-REPORT.md` in
the workspace; per-card reports and the dispatch registry in
`docs/research/wave7/` there.

*Measured on the lab's [bare-metal Strix Halo](../../BARE-METAL.md), configuration as of the note's date.*

*AI-assisted: GLM by z.ai.*
