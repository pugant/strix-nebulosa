# Vulkan optimization wave 4: what the prefill GEMM stack can and cannot give on gfx1151

**Research note — September 2026 (thread T31, wave 4).** Companion to
[2026-09-02-qwen4exp-graph-reuse-and-dense-decode.md](2026-09-02-qwen4exp-graph-reuse-and-dense-decode.md)
(the first optimization campaign) and to the wave-2/wave-3 series in
[PATCHES.md](../../PATCHES.md): wave 4 went after the prefill GEMM paths of the
Vulkan backend on the production model `Qwen3.8-Flash-Next-Q4_0_ROCMFP4_STRIX_LEAN`
(qwen4exp MoE, 98.5 GiB, MTP drafter Q8_0, KV q5_1, ctx 32768).

Everything was measured with the LLM service frozen for the whole window and —
new for this wave — **every performance verdict taken on interleaved A/B arms
only** (see lesson 3).

**TL;DR**

- **One GO: small-m split-k prefill (`+2.33%` combined-32k, `+3.95%` pp8192).**
  `ggml_vk_guess_split_k` required `m >= wg_denoms[0]`, so the tall-skinny
  prefill GEMMs the qwen4exp hyper-connections inject (m=4 k=10240, m=48
  k=2560, router m=1) ran one wave-row grid — 16 workgroups on 40 CUs, whole
  k chain serial — at 34 GF/s (7% of the achievable rate). Allowing
  k-parallelism for that regime (split capped at 8, same tile pipeline and
  split_k_reduce path) takes m=4 to −56% µs/call. ppl improved, battery
  clean, decode untouched. Shipped as the default in
  [`patches/optim-w4-series/`](../../patches/optim-w4-series/) (env
  `GGML_VK_MM_SMALLM_SPLITK` keeps 0/1/2 as opt-out).
- **Five measured NO-GOs, each closing a thread with a number or a mechanism**
  — pooled-indexer keys (tap cost equals the gather it saves), int-dot MMQ
  pipelines (never instantiated on RADV; enabled, they lose by 8-33% to the
  f16-dequant tile-m), large-tile and small-tile MUL_MAT_ID variants (−15.7%,
  −1.9%), f16 flash-attention accumulator on quantized KV (moves a bilingual
  near-tie below the acceptance gabarit — a degenerate-output glitch the
  perplexity gate cannot see), and element-wise fusion batching (real
  ~0.95 µs/launch elision against a 2.35 µs async-submit floor: ceiling
  +1.5-2%, gate +2%).
- **Bit-identity on Vulkan is structurally unreachable for any patch that
  adds or removes graph nodes** — ggml-vulkan picks fusion kernels from the
  per-split node stream, so even a one-node delta perturbs selection for
  identical ops downstream and greedy fingerprints flip at long distance.
  Proven on 5 variants of the same patch (CPU 12/12 everywhere, VK always
  divergent). Every VK card that touches the graph is quality-class D
  (perplexity + output battery), never fingerprint-class N.
- **Thermal drift reached ±26% between serial A/B arms.** A "GO" of +3.07%
  measured back-to-back (pool arm first, GPU at 62 °C cold) dissolved to
  −0.10% under the interleaved protocol. Serial cells on this box are
  ordering-biased: the second arm always runs hot. All wave-4 verdicts above
  are interleaved.

## The cards

| Card | Idea | Verdict | The number |
|---|---|---|---|
| C2-3 IDX-POOL | Pool per-block indexer keys, tap instead of gather | **NO-GO** | interleaved −0.10% (serial +3.07% was thermal) |
| N4 MMID-PP-TILING | int-dot MMQ pipelines; L/S/MoE tile variants | **NO-GO** | dense −8.4%, id −33%, tile-L −15.7%, tile-s −1.9% |
| N1 MM-SMALLM-PP | split-k for tall-skinny prefill GEMMs | **GO** | c32k **+2.33%**, pp8192 **+3.95%** |
| N3 FA-F16ACC-Q8KV | f16 accumulator in FA over quantized KV | **NO-GO** | battery DANTE-2 1/4 vs 4/4, same degenerate sha |
| N2 EW-FUSE-BATCH | fuse element-wise chains (SIGLU et al.) | **NO-GO** | tg512 +1.15% vs ≥+2% gate; quality all PASS |

## Lesson 1 — bit identity vs. the node stream

The C2-3 indexer-pooling patch was first gated on greedy fingerprint identity
(9k-token sha), the class-N gate the previous waves used for decode patches.
It never passed on Vulkan — on CPU it was 12/12 immediately. Five rewrites
later the mechanism was isolated: `ggml_vk_run` selects fused
kernels/pipelines from the **sequence of nodes in each split**, so any patch
that adds, removes or reorders nodes — even one — changes which fusion a
*later, unmodified* op receives, changing its FP accumulation order. The flip
appears at long distance (token 6 of 9k; a spontaneous EOS at token 29), far
from the touched region.

Consequences:

- The W2/W3 patches passed fingerprint gates because they were CPU/IO-side or
  non-fusion-perturbing — luck of the draw, not a property to rely on.
- The only class-N escape on VK would be tapping outside the graph (a second
  compute per round), which the backend does not support.
- **Default for VK graph-touching cards is class D**: perplexity (±0.5%) plus
  an output battery with a pre-declared degenerate-output sensor.

## Lesson 2 — the int-dot pipelines were dead code on RADV (and lose anyway)

The fork carries MMQ int-dot pipelines for ROCmFP4 that look enabled —
env-gated creation blocks exist in `ggml_vk_load_shaders`. Two traps:

1. **Duplicated creation branches**: only one branch is reached at runtime on
   RADV (identifiable only with a runtime probe, not by reading — patches on
   the dead branch apply silently and do nothing).
2. `case VK_VENDOR_ID_AMD` hardcodes `mul_mat_id_l = false` — the grouped
   (MoE) int-dot path is switched off for AMD regardless.

Enabled properly (explicit creation appended after `load_shaders`,
section-independent), the pipelines **lose to the f16-dequant tile-m path**
at production shapes: dense q8_1 −8.4%, grouped id −33%, with clean ppl. The
upstream choice of f16-dequant on this GPU is correct — now measured, not
assumed. Along the way the sweep found a **latent fork bug**: the AMD-GCN
branch resizes `l_warptile_mmq_int` to BLOCK=256 against wg denominators
{128,128,1} — dense MMQ l-tile produced garbage for m>32 (inf/ERR at m=2560).
Fixed in the series (patch 0003) with 15 regression cases in test-backend-ops.

## Lesson 3 — ±26% thermal drift between serial arms

Back-to-back A/B cells on this box order-bias everything: the second arm runs
at 80 °C while the first ran at 62 °C. Observed drift over a 15-minute
cooldown: +26% on pp8192. The C2-3 serial GO (+3.07%, pool arm first on a
cold GPU) was *entirely* the first-arm advantage. Protocol adopted for every
wave-4 verdict: alternate arms off/on ×4, log the GPU temperature per arm,
verdict on means with per-arm spread; discard cold-start outliers explicitly
and re-read without them. Absolute serial cells are only comparable to other
serial cells of the same session.

## Lesson 4 — count the work you add, not just the work you remove

The C2-3 estimate projected +2.9-3.9% from eliminating a gather (7.4% of the
round). It omitted the cost of the added tap. Measured: tap ≈ gather saved,
net 0. The same accounting killed the N2 ceiling: 142 fused launches +
~70 dead SCALE launches elided ≈ 0.95 µs/launch of real work, but the
submit floor is 2.35 µs/launch (async, pipelined) — the wall-clock ceiling of
the whole element-wise fusion family is +1.5-2%, below any sane gate.

## Lesson 5 — f16 accumulators and the near-tie gabarit

Flash-attention with an f16 accumulator over q8_0 KV passed perplexity
(±0.2%) and failed the output battery on a single item: a bilingual
near-tie prompt collapsed to the *same* 19-word degenerate answer on 3 of 4
seeds (baseline: 4/4 distinct). The perturbation is enough to move borderline
logits across the sampling gabarito. The battery catches what perplexity
cannot; a degenerate-output sensor pre-declared before the measurement is
what makes the verdict trustworthy.

## Deployment verification

- **Image flip-check**: pp8192 inside the shipped container, N1 canone (KV
  q8_0, no-mmap), OFF arm (`GGML_VK_MM_SMALLM_SPLITK=0`) on the colder GPU
  first, ON (default) after: **374.70 → 384.50 t/s = +2.62%** — the default
  flip is live in production (the gap to the wave-binary's +3.95% is the
  second arm running hot, plus a uniform +7% toolchain offset present in
  both arms).
- **Server A/B (production config, interleaved rounds w3-image then
  w4-image)**: timed 600-token generation **w3 24.08/24.02 vs w4 24.02/24.08
  tok/s — exact parity**, as expected: the patch touches prefill, not the
  speculative decode path. Acceptance rounds and rates coherent (161/167
  spec rounds per arm).
- **Output battery**: baseline image 12/12; w4 11/12 with the DANTE-2
  near-tie probe stopping at the prompt-determined three tercines (19 words).
  Seed-retry (temp 0, seeds 42/1234/777/2026): the **unpatched baseline
  fails the same probe at seed 2026 with the same sha** the w4 showed at
  seed 42, and the two w4 failures carry different shas — the failure basin
  belongs to the model's near-tie, not to the patch (the N3 case it
  resembles was 3× the identical degenerate sha against a 4/4 baseline).
  Verdict: sampling border, pre-declared sensor working as designed.
- **Production**: deployed as `qwen4exp-mtp-vk-optim-w4`, health OK, no new
  flags — the optimization is a code default, `GGML_VK_MM_SMALLM_SPLITK`
  remains a runtime opt-out.

## Series contents

`patches/optim-w4-series/` (8 patches, `git am`-clean on the tree of
`65084bd1c`, tree-verified at `ff264cbc5`): five N4 diagnostic patches
(env-gated int-dot / tile variants, default-off, kept as reproducible
instrumentation for the measurements above, including the warptile fix +
regression tests), the N1 split-k patch, a test-skip collateral, and the N1
default flip. The N2/N3 experiments are documented here but dropped from the
deploy branch; the full experimental history (including them) is preserved
on branch `optim-w4-full` of the engine workspace.

*Measured on the lab's [bare-metal Strix Halo](../../BARE-METAL.md), configuration as of the note's date.*
