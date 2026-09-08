# Experiments — the story index

The raw notes are one file per experiment (plus `data/` for the datasets); this index is the story that connects them. Every note below is referenced from one of the six threads; numbers in the README of this repo trace back to these files (T-numbers are our internal task ids).

## Quantization & quality

This lab began as a quantization pipeline: take the community-born ROCmFP4-STRIX_LEAN preset and turn out measured, sanitized quants of the models we actually run. That pipeline era produced the lab's first HF model cards (grug, Ornith, Nemotron Lightning, the Qwen3.6 27B/35B pair, and the Qwen3.8-27B model + its imatrix) — each with its bench note and, for the early ones, an adversarially reviewed design spec.

Once the pipeline was routine, the question became quality: what exactly do the preset choices cost? That produced the preset comparison thread — FULL vs LEAN vs the `_EVEN` controls (the "+53% ROCm" reading of the era is resolved later as a protocol artifact, see [Performance & hardware](#performance--hardware)) — the discovery that the "3-bit" preset is not a 3-bit on hybrid gated-deltanet models, and the cross-check against Unsloth's UD family.

Timeline:

- `2026-08-10` [results-2026-08-10.md](results-2026-08-10.md) — Qwen3.6-27B STRIX_LEAN vs Q4_K_M: +61% with MTP active, −13% size → verdict "optimal format on Strix Halo".
- `2026-08-11` [results-2026-08-11-grug.md](results-2026-08-11-grug.md) — grug-35b-v2 quantized and benched vs Q4_K_M baseline.
- `2026-08-12` [results-2026-08-12-nemotron.md](results-2026-08-12-nemotron.md) — Nemotron 3.5 Lightning 30B-A3B: first Mamba-hybrid MoE through the preset.
- `2026-08-12` [results-2026-08-12-publish.md](results-2026-08-12-publish.md) — grug + Ornith-1.0 published to HF (84.28 tok/s record on Nemotron came from the same era).
- `2026-08-14` [results-2026-08-14-q6-base-35b.md](results-2026-08-14-q6-base-35b.md) — Qwen3.6-35B base → Q6_0_ROCMFPX.
- `2026-08-15` [results-2026-08-15-qwen38-27b.md](results-2026-08-15-qwen38-27b.md) — the canonical Qwen3.8-27B run: imatrix coverage gate 496/496, backend table of the era.
- `2026-08-17` [results-2026-08-17-ppl-strategy-phaseA.md](results-2026-08-17-ppl-strategy-phaseA.md) — first in-house perplexity delta on our own GGUF: LEAN at parity with Q4_K_M on the real (Italian-agentic) domain, 18% smaller.
- `2026-08-18` [results-2026-08-18-rocmfp4-full-vs-strix-lean.md](results-2026-08-18-rocmfp4-full-vs-strix-lean.md) — FULL vs LEAN vs `_EVEN` at unified protocol: block-16 scale granularity is worth ~1.2 ppl points per 0.86 GB; LEAN is near-pareto.
- `2026-08-23` [2026-08-23-rocmfp3-quality-speed.md](2026-08-23-rocmfp3-quality-speed.md) — **NO-GO**: ROCmFP3 is not a true 3-bit on this hybrid arch (4.44/5.72 effective bpw, no byte savings) and tg pays −15.6/−31.7% for the K-quant path.
- `2026-08-29` [2026-08-29-t22-lean-vs-ud-quant.md](2026-08-29-t22-lean-vs-ud-quant.md) — STRIX_LEAN vs Unsloth UD-IQ4_XS on Flash-Next: LEAN ahead everywhere on speed (prose margin 11.7–14.5%); quality side settled daily use on the UD-plain variant; official-master engine cannot load the external-drafter setup — that requires our fork.
- `2026-09-02` [2026-09-02-flashnext-fp2-64gb-and-lean-requant.md](2026-09-02-flashnext-fp2-64gb-and-lean-requant.md) — the Flash-Next requant from native BF16 + full unsloth imatrix (measured ppl −0.6%/−4.9%, swapped into production and re-uploaded same-name), and the honest 2-bit frontier: a 57.04 GB FP2/FP4 mix for the 64 GB machines that runs +4–18% faster but pays the full structural 2-bit quality bill (RMSE at the codebook's theoretical floor); includes the Q5_1 `{d, m, qh, qs}` decoder-layout forensics that nearly derailed the diagnosis.
- `2026-09-03` [2026-09-03-fp2mix-v2-requant.md](2026-09-03-fp2mix-v2-requant.md) — the FP2MIX v2: PLE → Q5_1 disk-resident restores the lost n-gram recall (Dante 8k ppl 37.90 → **2.07**), FP4/Q3 trunk upgrade brings EN ppl to **3.27** (within ~2% of the current LEAN) at −5.5% tg128; 75.87 GiB on disk, **55.94 GiB resident** measured at ctx 131k with `--ple-disk`; shipped in-place over the v1 shards (re-download both).

Design specs of the era: [design-2026-08-11-grug-35b-v2-strix-lean.md](design-2026-08-11-grug-35b-v2-strix-lean.md), [design-2026-08-11-hf-publish-grug-ornith.md](design-2026-08-11-hf-publish-grug-ornith.md), [design-2026-08-12-nemotron35-lightning-rocmfp4.md](design-2026-08-12-nemotron35-lightning-rocmfp4.md).

**What shipped:** the full `scripts/` pipeline (download → imatrix → quantize → sanitize → publish), eight HF model cards, and the NO-GO presets (ROCmFP3, and the T10 `Q2_3_ROCMFPX_MIX` family below) documented instead of deleted.

## Dual-drafter & speculative decoding

The 27B carries a built-in MTP layer; upstream DFlash2 (llama.cpp PR #27342) offered a block-diffusion drafter. The first question was simply *which drafter is better* — and the A/B answered *neither*: DFlash2 dominates deterministic/agentic content but loses ~26% on free prose, MTP is the reverse. That asymmetry is the entire motivation for per-request routing. Everything after that — same-round cooperation (T8), per-round switching (T9), the round budget (T10), the round software (T11) — was the attempt to squeeze more out of the round, each step closed by measurement rather than fatigue.

Timeline:

- `2026-08-19` [results-2026-08-19-dflash2-vs-mtp.md](results-2026-08-19-dflash2-vs-mtp.md) — T7 A/B: the asymmetry measured (DFlash2 deterministic record 57.4 tok/s, prose −26%); verify batch shown bimodal via the SPEC_VERIFY_LOG instrumentation.
- `2026-08-19` [results-2026-08-19-t0-reasoning-acceptance.md](results-2026-08-19-t0-reasoning-acceptance.md) — T0: DFlash2 does **not** harm agentic thinking content → the routing policy can stay keyed on `tools` with MTP as the conservative default.
- `2026-08-19/20` [design-2026-08-19-t7f2-drafter-routing.md](design-2026-08-19-t7f2-drafter-routing.md) + [results-2026-08-20-drafter-routing-t1-t5.md](results-2026-08-20-drafter-routing-t1-t5.md) — routing designed, gated T1–T5, **shipped to production 2026-08-20**.
- `T8` [dual-drafter-synergy.md](dual-drafter-synergy.md) — same-round cooperation (concat on head token, pattern exclusion, deeper drafts, two-root verify tree): **closed by measurement** at seven gated steps; realized synergy = the alternation that routing already provides.
- `T9` [per-round-drafter-switching.md](per-round-drafter-switching.md) — per-round drafter switching **NO-GO**: acceptance correlation ρ = +0.729 pooled (round difficulty dominates complementarity) and the shadow drafter costs ~83 ms/round to keep synchronized — every oracle loses to the best static drafter by 66–74%.
- follow-up [interleaved-decode-charidentity.md](interleaved-decode-charidentity.md) — the engine question T9 left open: char-divergence comes from multi-row interleaved decode on the target context itself; a rollback is not numerically neutral after a >1-row decode.
- `T10` [speculative-round-budget.md](speculative-round-budget.md) — the round decomposed to the millisecond (accounting identity closes at 136.43 vs 136.42 ms); the ~3 bpw door **closed** (perplexity gate failed ~5×; kernels verified healthy).
- `T11` [speculative-round-software.md](speculative-round-software.md) — both software levers **falsified**: fused draft chain −10.2/−18.6% (ggml has no data-dependent control flow, so the fused graph pays the full n_max chain), verify dispatch switch −55.3/−55.6% (batched MMQ is ~2.5× slower than VEC/DMMV at 2–9 columns). The ~38 ms/round residue is structural on Vulkan/RADV.
- `typical-acceptance MTP` — **NO-GO, no raw note in this repo**: typical sampling was always worse than exact match. The verify batch is strongly bimodal (match ≈ 1.0, mismatch ≈ p_draft), so loosening acceptance with a typical-sampling criterion can only lose; best case Δ −0.24 tok/round. Closed at step 0, before a note was worth writing.

### Main table — which drafter, which workload

| Setup | prose tg (tok/s) | deterministic tg (tok/s) | Source |
|---|---|---|---|
| MTP n6 (control) | 19.6 / 20.2 | 45.2 / 26.1 | [`results-2026-08-19-dflash2-vs-mtp.md:15`](results-2026-08-19-dflash2-vs-mtp.md) |
| DFlash2 n7 | 14.2 / 15.0 | **57.4 / 36.3** — record | [`:16`](results-2026-08-19-dflash2-vs-mtp.md) |
| DFlash2 n5 (best single drafter) | 17.5 / 15.9 | 52.2 / 39.5 | [`:17`](results-2026-08-19-dflash2-vs-mtp.md) |
| **Dual routing (production)** | 19.7 / 20.4 (≈ MTP) | 54.7 / 40.7 (+19.2% / +20.8%) | [`results-2026-08-20-drafter-routing-t1-t5.md:30-36`](results-2026-08-20-drafter-routing-t1-t5.md) |

Each cell is two runs. "Deterministic" = counting/alphabet-style prompts. Dual-routing prose values are prompts P1/P2 of the T4 table. The MTP control row and the T4 gate runs were **separate sessions** (e.g. mono prose 19.6/20.2 here vs 19.8/20.4 in the T4 table — the gate re-ran the prose pair) — compare deltas, not absolute cells, across sessions. Read the table as: routing keeps MTP-class prose speed *and* captures most of the DFlash2 deterministic win, in one server.

### Why routing works: workload and acceptance

Agentic workload, MTP n6 vs DFlash2 n7 (tok/s) — [`results-2026-08-19-dflash2-vs-mtp.md:73-75`](results-2026-08-19-dflash2-vs-mtp.md):

| Prompt | MTP n6 | DFlash2 n7 | Delta |
|---|---|---|---|
| coding (10 functions) | 27.5 | 33.8 | **+23%** |
| JSON (30 objects) | 35.0 | 36.2 | +3% |
| log (20 fixed-format lines) | 28.0 | 35.8 | **+28%** |

Acceptance, per draft position ([`:26-30`, `:77-79`](results-2026-08-19-dflash2-vs-mtp.md)):

- On deterministic/structured content, DFlash2 stays **≥ 0.90 through position 7** (0.99, 0.97, 0.96, 0.94, 0.92, 0.91, 0.90).
- On agentic content it decays but holds **≥ 0.50 at position 7** (0.92, 0.77, 0.71, 0.65, 0.56, 0.52, 0.51), where MTP drops to 0.25–0.50.
- MTP acceptance **collapses after position 1** on structured content (0.93 → 0.33): long drafts are wasted there, which is precisely the slack DFlash2 picks up.

### Validation gates T1–T5

The routing feature shipped only after these gates passed ([`results-2026-08-20-drafter-routing-t1-t5.md:19-24`](results-2026-08-20-drafter-routing-t1-t5.md)):

| Gate | What it checks | Result |
|---|---|---|
| T1 smoke dual-load | 14 checks: boot / policy / override / 400 / fallback / cache-switch / metrics | **PASS 14/14** |
| T2 cache round-trip | 4 gates per config (simple + production), drafter switched every turn | **PASS 4/4** per config |
| T3 sacred paths (patches 0005–0009) | budget-forced end, altered resend, trailing rollback, checkpoint restore — in dual mode | **PASS 7/7** |
| T4 routing vs mono | prose ≥ −3%; agentic ≥ +10% (mean of 3 prompts) | **PASS** — prose +1.0% / −0.2%; agentic +19.6% / +18.9% |
| T5 numerical spot-check | dual-MTP ≡ mono ckpt7, dual-DFlash ≡ mono DF7 (greedy) | **PASS** — char-identical, 4/4 |

### Routing per-prompt deltas (T4)

[`results-2026-08-20-drafter-routing-t1-t5.md:30-36`](results-2026-08-20-drafter-routing-t1-t5.md):

| Prompt | Class | Mono MTP6 | Dual | Delta | Routed to |
|---|---|---|---|---|---|
| P1 | prose | 19.8 | 19.7 | −0.5% | mtp |
| P2 | prose | 20.4 | 20.4 | +0.0% | mtp |
| D1 | deterministic | 45.9 | 54.7 | **+19.2%** | dflash |
| D2 | deterministic | 33.7 | 40.7 | **+20.8%** | dflash |
| A1 | agentic | 38.6 | 46.0 | **+19.2%** | dflash |
| A2 | agentic | 42.0 | 51.1 | **+21.7%** | dflash |
| A3 | agentic | 39.9 | 46.2 | **+15.8%** | dflash |

**What shipped:** the 14-commit [`patches/drafter-routing/`](../../patches/drafter-routing/) series (in production since 2026-08-20), the SPEC_VERIFY_LOG instrumentation behind every workload characterization here, and the T10/T11 NO-GOs kept as patch series ([`patches/t10/`](../../patches/t10/), [`patches/t11/`](../../patches/t11/)).

## qwen4exp runtime

The flagship line. Qwen3.8-Flash-Next (`qwen4exp`) is a 180B-class hybrid — gated-deltanet linear attention, QSA indexer attention with a third KV stream, a 51B-parameter PLE n-gram table, 4-stream low-rank hyper-connections — and the thread walked it from a quant, to a runtime, to the lab's daily production model: pipeline quant (98.5 GiB HF card) → architecture port onto the fork → external MTP drafter driven by its own GGUF → vision working together with MTP → rollback-restore correctness → the PLE table leaving RAM entirely (`--ple-disk`, deployed 2026-08-31) → and finally the prompt cache surviving restarts (`--cache-disk-persist`, deployed 2026-09-01: a 107k-token context restores in 1.57 s after a restart — end-to-end 14.3 s vs the 920 s re-prefill it replaces, 64×).

Timeline:

- **Quant + card** — 98.5 GiB ROCmFP4-STRIX_LEAN quant published with its mmproj; the largest model the lab runs.
- **Runtime port** — full `qwen4exp` support on the fork (HIP and Vulkan builds alike), branch `qwen4exp-rt`.
- **External MTP drafter** — upstream PR #27836 ported and adapted (converter side from PR #27742): the MTP/NextN head runs from a drafter GGUF via `-md`; four silent porting bugs fixed, the mmproj×MTP abort among them.
- **Vision × MTP** — image chunks are embd-only, so the drafter-side replay is skipped while the head still sees the image through the trunk hidden state; measured acceptance on a vision request: 98.3% cumulative.
- **RS-rollback + ring fix** — one line enables the RS ring-salvage rollback for qwen4exp; with the conv/PLE ring-slot writer fix (see [Infrastructure fixes](#infrastructure-fixes)) the stack sustains 41–44 tok/s at n=6, ~+70% over the pre-fix server.
- **Reasoning-budget warn window** — the 75% convergence nudge (patches 0016–0020, note: [reasoning-budget-warn75.md](reasoning-budget-warn75.md)).
- `2026-08-31` **PLE disk-offload deployed** — `--ple-disk`: the 35.76 GiB PLE table is read on demand from the GGUF itself, ~36 GB of RAM freed, output char-identical; guide: [`docs/guide/qwen38-flash-next-ple-disk.md`](../guide/qwen38-flash-next-ple-disk.md).
- `2026-09-01` **Persistent prompt cache deployed** — `--cache-disk-persist` (ds4-inspired): the on-SSD prompt library outlives the process; after a restart a 107k-token context restores in 1.57 s — end-to-end 14.3 s vs a 920 s cold re-prefill (**64×**) — deterministically (char-identical 111/111). Structural limit kept honest: with the MTP drafter the restore needs a token-exact boundary — raw verbatim replays restore, chat-template history replays do not; guide: [`docs/guide/qwen38-flash-next-prompt-cache-disk.md`](../guide/qwen38-flash-next-prompt-cache-disk.md).
- `2026-09-02` **PLE disk-offload learns FP2 + the production model becomes a true BF16 requant** — `--ple-disk` extended to the 2-bit PLE table (on-disk vs in-RAM lookup: 0-byte divergence, ≤4% tg gap), and the production STRIX_LEAN file replaced in place by a requant from the native BF16 export with the full unsloth imatrix (measured ppl −0.6% Italian / −4.9% English); note: [2026-09-02-flashnext-fp2-64gb-and-lean-requant.md](2026-09-02-flashnext-fp2-64gb-and-lean-requant.md).
- `2026-09-02` **Graph reuse + dense decode in the round loop** — the two production runtime speedups of the optimization campaign: the QSA/PLE graph inputs learned `can_reuse` (rebuilds 1025 → 6) and decode goes dense when the indexer budget covers the cache; plain tg512 +18.6% HIP / +32.5% Vulkan with bit-identical greedy fingerprints; note: [2026-09-02-qwen4exp-graph-reuse-and-dense-decode.md](2026-09-02-qwen4exp-graph-reuse-and-dense-decode.md).

### Outcome (dedicated GPU, 98.5 GiB target + 3.85 GiB Q8_0 drafter, ctx 8192, median of 3)

| workload | plain | +MTP n=3 | +MTP n=5 |
|---|---|---|---|
| deterministic (counting) | 22.1 | 46.0 (**+108%**) | **50.2 (+127%)** |
| deterministic (alphabet) | 20.9 | 32.0 (+53%) | 32.6 (+56%) |
| open prose | 22.6 | 22.8–25.4 (+1–12%) | — |

Draft acceptance 95.7% of tokens (mean accepted length 3.24 at n=3; still 5.20 of 6 at n=5 on deterministic text). The mechanical read: the round bottleneck is the **batched verify over the hybrid trunk** (KV + GDN + QSA index + PLE), not the drafter — deterministic work converts acceptance into speed almost 1:1, open prose decays after position 1. After the rollback-restore fix the same stack on Vulkan/RADV sustains 41–44 tok/s at n=6 on code-generation workloads (draft acceptance 0.91–0.95), and n=6 is the measured optimum (n=8 regresses to 35 tok/s: draft positions 7–8 do not pay for themselves).

The flagship note for this thread is [qwen38-flash-next-runtime.md](qwen38-flash-next-runtime.md); the quant comparison on this model is [2026-08-29-t22-lean-vs-ud-quant.md](2026-08-29-t22-lean-vs-ud-quant.md).

**What shipped:** [`patches/qwen4exp-mtp/`](../../patches/qwen4exp-mtp/) (20 commits), [`patches/t25-ple-disk/`](../../patches/t25-ple-disk/) (15 patches), [`patches/t23-kv-disk-persist/`](../../patches/t23-kv-disk-persist/) (12 patches), [`patches/optim-camp/`](../../patches/optim-camp/) (2 patches), the [98.5 GiB HF card](https://huggingface.co/pugant/Qwen3.8-Flash-Next-ROCMFP4_STRIX_LEAN-GGUF), and the production server running all of it since 2026-08-31, 09-01 and 09-03.

## pi-stack

The agent stack is the lab's own daily driver: a long-lived thinking agent with tool calls, running for hours at a time. The thread asks where its time and its quality actually go — measured on the production workload rather than on benches, and each candidate change gated before deploy.

Timeline:

- `2026-08-28/29` [2026-08-29-pi-stack-improvement.md](2026-08-29-pi-stack-improvement.md) — the improvement cycle, all gates closed, deployed as `rs2`:
  - hipCUB enablement (port of upstream PR #27874) — **NO-GO**: pp at 131k context **−42%** (0.58× the control); measure, don't assume.
  - KV quantization — **costs 19–26%** tg on this workload → f16 K/V for that cycle (the later Vulkan switch re-introduced q8_0 KV to fund `--no-mmap`, see [Performance & hardware](#performance--hardware)).
  - n-gram drafter — instrumented with per-drafter counters, **no headroom** — the drafter never engages (its engagement precondition fails).
  - drafter-state rollback fix (F4/H1) — resetting the MTP drafter state on partial-reject rollback cuts the position-0 rejection rate (p0-reject) 0.272 → 0.161.
- `2026-08-30` [reasoning-budget-warn75.md](reasoning-budget-warn75.md) — the warn window at 75% of the thinking budget: in a one-hour real agent session (48 requests) every request closed naturally, the single triggered warning converged in 1.3 s, zero budgets exhausted.
- **Quality matrix on the real agent** — out-of-format tool-calls turned out to be **independent of quant and spec** configuration (a model/parser-level issue, fixed client-side), and the quality verdict on Flash-Next quants moved daily use to the UD-plain variant ([2026-08-29-t22-lean-vs-ud-quant.md](2026-08-29-t22-lean-vs-ud-quant.md)). Full story: [pi-stack-followups.md](pi-stack-followups.md).
- **Thinking-cap steering — NO-GO** (soft redirect at the budget line: notice + squeeze, 2026-08-17): the pressure never stopped the exhaustion — 9/10 generations still hit the cap — and it broke the cache resend under pressure; the warn window that later shipped is the surviving idea (design [design-2026-08-17-reasoning-pressure.md](design-2026-08-17-reasoning-pressure.md), patch series [`patches/reasoning-pressure/`](../../patches/reasoning-pressure/)).
- **Thinking-cap budget-aware anchor line — NO-GO**: a line meant to hold the remaining reasoning budget stable mid-generation; it caused quality regressions and was dropped. What survived from the thinking-cap thread is the hard cap plus the warn window above.

**What shipped:** [`patches/f4-rollback-fix/`](../../patches/f4-rollback-fix/) and [`patches/ngram-drafter-instrumentation/`](../../patches/ngram-drafter-instrumentation/) (both in the `rocmfpx/` snapshot), the warn-75 window (qwen4exp-mtp patches 0016–0020) deployed on the real server, and the production switch to Vulkan that closes the cycle (next thread).

## Performance & hardware

One machine, one physics: 128 GB of unified memory behind a ~256 GB/s pipe. This thread is the honest characterization of the Radeon 8060S — what is memory-bound, what the backends actually differ on, and which knobs are already at their optimum.

Timeline:

- `2026-08-14` [results-2026-08-14-vulkan-vs-rocm.md](results-2026-08-14-vulkan-vs-rocm.md) — first in-house Vulkan vs ROCm validation on Qwen3.6-35B-A3B (ROCm stack ahead on both fronts that day — vs the Vulkan UD-Q5 stack; on the same GGUF, RADV already won tg +14%).
- `2026-08-14` [results-2026-08-14-vulkan-rocmfp4-fork.md](results-2026-08-14-vulkan-rocmfp4-fork.md) — ROCmFP4 types on the Vulkan build: the community 78–90 tok/s claim confirmed in-house (81.6); "Vulkan can't run ROCmFP4" falls (it is a fork capability).
- `2026-08-17` [results-2026-08-17-kv-quant-tg-context.md](results-2026-08-17-kv-quant-tg-context.md) — asymmetric q8_0 KV vs f16 — **NO-GO**: quantized KV worsens or ties everywhere; the long-ctx bottleneck is not KV read bytes.
- `2026-08-17` [results-2026-08-17-ckpt-ring-tg-24k.md](results-2026-08-17-ckpt-ring-tg-24k.md) — checkpoint/ring tuning is a **dead lever** (prefill checkpoint cost ≤4%, ring 32 vs 4 indistinguishable); MTP boost is **constant with context** (~3.09× at both 1k and 24k).
- `2026-08-17` [results-2026-08-17-rocm-vs-vulkan-tg.md](results-2026-08-17-rocm-vs-vulkan-tg.md) — at unified protocol the backends are **equivalent** on tg (1k and 24k); the historical "+53% ROCm dense" reading does not reproduce and is resolved as a protocol artifact — protocol-dependent, not physics.
- **Production switch to Vulkan + `--no-mmap`** — on Flash-Next prose Vulkan beats ROCm by ~+22%, but **mmap collapses Vulkan prompt processing ~3× (244 → 70 tok/s)**, so the production combo is Vulkan + `--no-mmap` + KV q8_0 (the KV quant here buys the RAM that no-mmap needs, at a cost the prose win pays for). Thread note: [vulkan-nommap-backend.md](vulkan-nommap-backend.md).
- **Agent-latency decomposition** ([agent-latency.md](agent-latency.md)) — of the round-trip latency budget, the dominant residual is **client-side** (agent stalls, not GPU); the cold re-prefill quota traced back to a client-side cache-breaking timestamp and went to zero cold fallbacks with the client fix, while C4 fell 64.6% (the salvage that absorbed the aborts: [Infrastructure fixes](#infrastructure-fixes)).
- **Measured 8060S facts** — tg is memory-bound (closed); attention is **44.8% of GPU time at 80k context** running at MFU 13% vs ~40% for plain matmul (time-crossover ~15–25k tokens, well before the FLOP crossover); GDN scan cost is 0.1–0.9% — not the ~44% sometimes claimed; clocks hold a 2220 MHz plateau with no thermal degradation across long runs.
- `2026-09-08` **Wave-4 prefill GEMM campaign** — small-m split-k for the tall-skinny prefill GEMMs the qwen4exp hyper-connections inject, shipped **on by default**: combined32k **+2.33%**, pp8192 **+3.95%**, ppl improved, decode untouched. Four measured NO-GOs close their threads (pooled indexer keys — the tap costs what the gather saved; int-dot MMQ — never instantiated on RADV and slower when forced; f16 FA accumulator on quantized KV — near-tie degenerate output; element-wise fusion — under the async-submit floor), and two protocol lessons now rule every future card: bit-identity on Vulkan is structurally unreachable for graph-touching patches (fusion picks follow the node stream) and serial A/B arms drift ±26% with temperature — interleaved verdicts only; note: [2026-09-08-vulkan-optimization-wave4.md](2026-09-08-vulkan-optimization-wave4.md).

### Backend choice by model class: ROCm vs Vulkan

On the 08-15 bench the ROCm build beat Vulkan on dense-27B generation (+53%) — a margin the unified-protocol re-run two days later did not reproduce (tg equivalent at 1k/24k); we keep dense → ROCm as our operating default, but the margin is protocol-dependent, not physics. On MoE FP4 the Vulkan build wins ([`results-2026-08-15-qwen38-27b.md:58-60`](results-2026-08-15-qwen38-27b.md)):

| Metric | ROCm | Vulkan RADV | Note |
|---|---|---|---|
| plain tg128 (dense 27B) | **13.8** | 9.0 | ROCm **+53%** |
| plain pp512 (dense 27B) | **354.7** | 319.8 | ROCm +11% |
| tg (MoE 35B FP4 class) | 71.2 | **81.6** | Vulkan fork wins |

Practical rule we follow: dense → ROCm; MoE FP4 → Vulkan fork. (The speculative-decoding tables above are all Vulkan RADV.)

### Methodology

- **One machine** — our Strix Halo (Ryzen AI MAX+ 395, 128 GB), dedicated GPU window (production service stopped during runs).
- temp 0, single stream, warm-up discarded, **2–5 runs, median, per note** — these are not statistical means; treat them as careful point measurements.
- Full raw data, logs and per-experiment setup live in [`docs/experiments/`](.) — they are the raw working notes. This index is the summary and entry point.

**What shipped:** the production serving config (Vulkan/RADV + `--no-mmap` + KV q8_0, cache-ram sized for the agent), and the closed levers (KV-quant, checkpoint/ring, hipCUB) documented so they stay closed. The optimization-campaign speedups (graph reuse + dense decode, waves 2-4 incl. the split-k default of `optim-w4`) are documented under [PATCHES.md](../../PATCHES.md) and the wave-4 note above.

## Infrastructure fixes

Production servers fail in ways benches never show. This thread collects the bugs that first *looked* like performance or quality problems and were in fact correctness bugs in the cache/rollback machinery — each found by measurement, each fixed behind a gate, each shipped.

Timeline:

- **Spec-boundary cache salvage** — when a speculative conversation crosses a cache boundary cold, the server used to re-process everything. Three patches fixed the surface: the forced-end sequence prefixed with a newline for a cache-friendly resend (`0005`), the vLLM-name alias (`0006`), and checkpoint-based salvage (`0007`) — **~91% of prefill tokens salvaged** on spec-boundary cold starts of our 12-request agent workload. The trailing-rollback part merged upstream as [charlie12345/ROCmFPX#69](https://github.com/charlie12345/ROCmFPX/pull/69). Dedicated note: [spec-boundary-cache.md](spec-boundary-cache.md).
- **conv/PLE ring-slot rollback fix** — the "bad quant" that was a bug: `build_conv_state_at` wrote only ring group 0, so every partial-reject rollback restored the GDN conv history and the PLE history from never-written (zeroed) slots while the SSM rewound correctly — lost digits in number runs, spliced fragments, premature stops, and an acceptance dip right after each rollback that had been misread as the ring's inherent cost. With the fix: acceptance ~0.74 → **0.91–0.95**, tg 24 → **41–44 tok/s**, the post-rollback dip gone. Full story: [conv-ple-ring-slots.md](conv-ple-ring-slots.md).
- **Device split-buffer checkpoint restore** — a rare (1-in-16 in production logs) `device checkpoint restore buffer mismatch` refused the draft-state restore and forced a **full** ~43k-token re-prefill (~2.5 min) although the target state was already fine. Root cause: the device-side write emits one block per contiguous cell range, the read one block per layer tensor — same bytes, different split. The fix copies through cursor views when total sizes match (`restored split device buffers`), plus `llama-state-split-test`, a deterministic repro/verification tool.
- `2026-09-03` **F4 reset keeps the speculative boundary** — the 08-29 drafter reset at partial-reject had been clearing the *whole* impl state (`set_state({})`), including the cache-facing boundary `get_state` must hand to `prompt_save`: a task whose final round is a partial reject went idle with an empty boundary, the save skipped (`required speculative state unavailable`) and the next session re-prefilled everything (run7 09-03: 5/5 full re-prefills, ~82k–139k tokens, 30 checkpoint deferrals, `spec_state_resets_total` 6577/120 tasks). The new `draft_sync_reset` clears only the draft-sync pair — eagle3 splits `boundary_*` from `pending_*`/`verify_*`, the production `draft-mtp` drafter snapshots the boundary blob; every other impl keeps the old behavior by default. Deployed 09-03 evening: 479 resets exercised in the smoke, 0 skipped saves, 0 deferrals; note: [2026-09-03-f4-boundary-save-fix.md](2026-09-03-f4-boundary-save-fix.md).
- `2026-09-04` **Tolerating the parameter that came back** — an agent turn that had spent four minutes generating a 222-line file write died with `The model produced output that does not match the expected peg-native format`, twice, identically: the model re-emitted a *required* `<parameter=path>` block after the content, and the auto-generated tagged grammar admits each required argument exactly once — the whole tool call failed to parse and the server threw the generation away (headless runs have no retry). The grammar now tolerates any argument repeated after the required sequence, the mapper resolves duplicates **last-wins** (surgical pair removal, other keys' order preserved), and the tolerance stays loud: `peg-native: leniency-hit: duplicate param '<name>' tolerated (last-wins)` — the slip rate is a model-behaviour signal worth keeping visible. This also fixed a latent cousin: repeated *optional* args were already legal but silently produced JSON with duplicate keys. Note: [2026-09-04-peg-native-duplicate-arg-lenient.md](2026-09-04-peg-native-duplicate-arg-lenient.md).

Also in this thread, the pipeline-era specs that defined the quantize/publish infrastructure the repo still runs: [design-2026-08-11-grug-35b-v2-strix-lean.md](design-2026-08-11-grug-35b-v2-strix-lean.md) and [design-2026-08-11-hf-publish-grug-ornith.md](design-2026-08-11-hf-publish-grug-ornith.md). Dedicated notes for the salvage and ring fixes: [spec-boundary-cache.md](spec-boundary-cache.md) and [conv-ple-ring-slots.md](conv-ple-ring-slots.md).

**What shipped:** [`patches/spec-cache-trailing-rollback/`](../../patches/spec-cache-trailing-rollback/) (9 patches, partially merged via PR #69), [`patches/ckpt-device-split-restore/`](../../patches/ckpt-device-split-restore/), [`patches/f4-boundary-save/`](../../patches/f4-boundary-save/) (2 patches), [`patches/peg-lenient-dup-args/`](../../patches/peg-lenient-dup-args/) (1 patch, deployed 09-04 as `qwen4exp-mtp-vk-optim3`), and the ring-slot writer inside [`patches/qwen4exp-mtp/`](../../patches/qwen4exp-mtp/) — all live in the production server and in the [`rocmfpx/`](../../rocmfpx/) snapshot.
