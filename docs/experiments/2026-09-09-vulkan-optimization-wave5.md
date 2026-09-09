# Vulkan optimization wave 5: the restore CRC was the win; the FA tile space closed by measurement

**Research note — September 2026 (thread T31, wave 5).** Companion to
[2026-09-08-vulkan-optimization-wave4.md](2026-09-08-vulkan-optimization-wave4.md)
and to the series entries in [PATCHES.md](../../PATCHES.md). Wave 5 ran
**roofline-first**: three wall-critical probes (F0) decided which cards were
worth implementing before any patch was written. Same production model as
wave 4, `Qwen3.8-Flash-Next-Q4_0_ROCMFP4_STRIX_LEAN` (qwen4exp MoE, 98.5 GiB,
MTP drafter Q8_0, KV q5_1 in production / q8_0 in the bench canon, ctx 32768).

**Identity.** Baseline = the deployed `optim-w4` at `286f51d4b` (same-binary
A/B). Worktree branch `optim-w5`; series = `4435d1855` (t23-crc) +
`e8a4d8c5f` (t23-crc-fix) on that base, tree-verified by `git am` on a clean
detached worktree (tree `449517003…`, zero conflicts). The full experimental
history — including the two dropped FA-TILE commits — lives on branch
`optim-w5-full` @ `5bab7f594` of the engine workspace; the snapshot in this
repo adds only the wave-5 `AI_CHANGES.md` session entry (`ae48440e6`).
Production image `qwen4exp-mtp-vk-optim-w5` (CPU-only build, rc=0 in 119 s).
LLM service frozen for the whole window; every performance verdict below is
interleaved A/B (the wave-4 thermal protocol).

**TL;DR**

- **One GO, deployed: T23-CRC.** The `--cache-disk-persist` restore verifies
  a CRC32 over the *whole* entry file (target + draft, 2 CRCs) before loading
  it — with a 256-entry byte-wise table at 0.63 GB/s. The canonical production
  entry is 107,283 tokens = **2,272,282,732 B**, and the historical 1.57 s
  restore figure was `load_ms` *without* the CRC: the real restore is
  ≈3.63 s CRC + 1.57 s load = **5.20 s, of which CRC is 69.8%**. A PCLMUL
  folding path (16 B/iter, same polynomial, **bit-exact**, runtime dispatch,
  table fallback verbatim) runs the CRC pass at 8.07 GB/s = **12.9×** —
  expected restore −64% (gate ≥30%). Host-only series: decode untouched, A/B
  parity confirmed, per-probe battery shas **bit-identical 12/12**. Deployed
  as `qwen4exp-mtp-vk-optim-w5`, no new flags.
- **One structural NO-GO: FA-PP-TILE.** The flash-attention prefill tile
  levers are exhausted on this GPU. The wave's opening premise (a coopmat2 FA
  path) was refuted at runtime: **coopmat2 is never active on gfx1151/RADV** —
  every FA call takes `FA_COOPMAT1` with `Br=16 Bc=64` hardcoded. The Br32
  dual-panel variant built on the real path is *correct* (5336/5336 ops,
  bit-identical greedy probe) but **slower on every FA shape** (c32k −5.96%
  vs a ≥+3.69% gate): LDS 25,968 → 42,672 B/WG halves occupancy (2→1 WG/CU)
  and dominates the halved KV rereads. The zero-reuse roofline that motivated
  the card is refuted by measurement — the ×384 KV rereads were being served
  by L2/MALL far better than the model assumed.
- **One thread closed in the opening probe: host sampling.** The sampling
  chain costs 0.375 ms/round (median) = **0.32% of the 117.353 ms round
  wall**, and the fork's top-k is already a radix-bucket partial sort over the
  full vocab. No AVX-512 card can pay for itself there.

## The cards

| Card | Idea | Verdict | The number |
|---|---|---|---|
| F0 wall-critical probes | ceiling + wall accounting before implementing | ran | GPU ~7 min vs 35 planned (−80%) |
| FA-PP-TILE | wider FA prefill tiles on the real coopmat path | **NO-GO (structural)** | c32k −5.96% vs ≥+3.69% gate; 0/4 separation; every FA shape slower |
| SAMPLING-TOPK AVX-512 | overlap/vectorize the host sampling chain | **closed in F0** | 0.375 ms/round = 0.32% of the round wall |
| T23-CRC | PCLMUL CRC32 for the cache-disk restore verify | **GO (deployed)** | bit-exact 12.9× (0.63 → 8.07 GB/s); CRC = 69.8% of the real restore |

## The one GO — T23-CRC (deployed)

**The wall it removes.** `llama_persist_crc32_file` streams the entry through
a Sarwate byte-wise table (IEEE polynomial `0xEDB88320`, check-value
`0xCBF43926`) at 0.63 GB/s, twice per restore (`crc_main` + `crc_drft`), and
the timer named `load_ms` starts only *after* this verify passes. The wave-5
F0 probe re-derived the entry sizes from production logs: the 107,283-token
canonical entry is 2,049,563,172 + 222,719,560 = **2,272,282,732 B** in CRC
(the earlier "133 MB" figure was the `draft_bytes` of a different entry), so
the celebrated 1.57 s restore was the load phase alone; the real wall was
≈5.20 s with CRC at 69.8%.

**The implementation** (patch 0001, `src/llama-persist-meta.cpp` only,
+240/−4): a 128-bit folding state carried as two scalars, 16 B absorbed per
iteration (2 `pclmulqdq`), folding constants `x^e mod F` **derived at first
use** (magic static) instead of transcribed literals — one of them re-derived
the slow way in an assert as a one-off guard. Runtime dispatch
`__builtin_cpu_supports("pclmul") && ("ssse3")` cached in a magic static; the
intrinsics are target-attributed, so the TU still builds without `-mpclmul`.
The fallback is the original table path, verbatim; a `< 16 B` remainder parks
in a carry so the streaming update is agnostic of the caller's chunking.

**Bit-exactness evidence** (harness over 4 paths — reference / table / fold /
dispatch): agreement on 36 sizes (0..1 MiB, every tail 0-15), check-value
`0xCBF43926`, 300 fuzz (offset,len) cases, ragged-chunk streaming both ways,
file-API semantics (exact/longer/shorter/missing); plus a read-only replay of
**14 real production entries** (target+draft ≈ 16.71 GB) whose recomputed
CRCs match the pre-PCLMUL sidecars. Smoke on 64 MiB: dispatch 8.30-8.33 GB/s
vs table 0.64 = 12.9-13.0×. Review minors (multi-16 assert, `const&`,
comments) landed as patch 0002; image identity check: `libllama.so` carries
14 `pclmul` instructions in w5 vs 0 in every w4 binary.

**Gates.** Restore-speed gate ≥30%: the CRC pass itself is measured at 12.9×
on the real entry (expected restore −64%); the end-to-end timing is
**provisional** — the A/B entries sat under the restore min-tokens, so the
first production restore lands with the next π turn. Final A/B (interleaved,
production config, timed 600-token generations): w4 24.37 vs w5 24.08 tok/s =
**−1.2%, inside the baseline's own intra-arm spread (2.4%)**, with spec
acceptance and rounds identical in every arm — expected parity, decode path
untouched. Output battery ×2: 11/12 both arms, the single fail being the
DANTE-2 near-tie basin already present in the w4 baseline, and the per-probe
`sha256_content16` values are **bit-identical between the w4 and w5 batteries
on 12/12 probes**. Deployed 2026-09-09 as `qwen4exp-mtp-vk-optim-w5`, health
64 s, all markers present, no new flags (host-only series, defaults
unchanged); rollback = repoint the compose file at `optim-w4`.

## The structural NO-GO — FA-PP-TILE

### The premise, refuted at runtime

The F0 ceiling analysis admitted the card: at 32k context the FA-pp phase
costs 111.90 s = 23.3% of prefill (3,120 calls, 75.7 ms/call at k=32768) at
an aggregate MFU of 12.7% against a 45.15 TF/s peak — a fillable headroom of
+5.76% (12.7%→35%) to +11.65% of the c32k cell. The v1 sketch assumed the
coopmat2 FA path with a Bc 32→64 lever. The runtime diagnostics refuted the
premise: the device line says `matrix cores: KHR_coopmat` on **both** the
host Mesa 26.0.8 and the production container's 25.3.6, and every FA call in
the diagnostic bench takes `path=FA_COOPMAT1 Br=16 Bc=64` (num_subgroups=4,
hardcoded, shape-blind — the coopmat2 tuning function is never invoked). The
real shader is `flash_attn_cm1.comp`; v1 was inert on this machine and was
dropped.

### The map on the real path (and its refutation)

With Br16/Bc64 fixed, one FA prefill call spawns 768 workgroups (32 row-tiles
× 24 q-heads), so each kv-head's K/V is re-read by 12 GQA heads × 32 tiles =
**384 workgroups**; the zero-reuse demand model B(k) ≈ 417,792×k bytes
(≈13.9 GB at k=33,280 vs 36.2 MB unique, ×382). A 3-point roofline fit
k=8192 at **94%** of the DRAM roof (214 GB/s effective) and k=33,280 at 78%
(183 GB/s) — the map's verdict: KV rereads are the neck, halve them with
Br16→32 dual-panel. The measurement below refutes exactly this: on the
production driver the rereads were already served by L2/MALL far better than
the zero-reuse model assumes.

### The correct implementation, measured

v2 (+101/−35, env `GGML_VK_FA_PP_TILE` 0=off/1/2, default off, engagement
gate `n_rows≥32 ∧ KV%64==0`): dual-panel `flash_attn_cm1.comp` + host Br=32
override + an amended LDS-accounting term, plus a probe fix that keeps the
gqa-fold probe immune to the override. Correctness proven: FLASH_ATTN_EXT ops
**5336/5336 PASS** in both modes, engagement markers 4/4 on / 0/4 off, greedy
host probe **bit-identical** between arms (sha `3e644a98…`, env verified in
/proc), LDS 42,672 B ≤ 65,536, default-off bit-identical to baseline.

Stage-1 cells (production container/driver, ×4 interleaved, q8_0 canon):

| Cell | off | on | delta |
|---|---:|---:|---|
| pp32768 (mean ± σ, n=4) | 279.73 ± 3.26 | 263.06 ± 3.43 | **−5.96%** (gate ≥+3.69%) |
| pp8192 (guard) | 361.38 ± 2.16 | 358.48 ± 2.13 | −0.81% |

Separation 0/4 (every off/on pair negative, sign constant across 8/8 cells);
without the declared cold-GPU outlier arm the delta is −5.43%, verdict
unchanged. Per-op evidence (perf-logger, mean of 4 replicas): the FA op is
slower on **every** shape — k=512 +41.9%, k=1024 **+251.7%**, k=1536 +29.3%,
k=8192 +12.2%, k=16384 +18.6%, k=24576 +30.4%, k=32768 **+50.5%**; FA_tot per
arm (1,920 calls) 59,057 ± 2,360 → 74,092 ± 4,056 ms = **+25.5%**.

### The mechanism, and why the space is closed

Measured, not presumed: LDS 25,968 → 42,672 B/WG (≈+220-230 VGPR) drops
occupancy **2→1 WG/CU**, and that cost dominates the halved K/V byte traffic.
The regression reproduces on two drivers (host −5.8%, container −5.96%) and
is not a patch defect (ops/probe/default-off all clean). The v3 fold-GQA
sketch dies the same way (LDS 61.3 KB = the same 1-WG/CU trap) and Br>32
exceeds the 64 KB LDS: **the tile-lever space on coopmat1 is exhausted** —
any Br>16 either breaks LDS or loses occupancy. Both FA-TILE commits were
dropped from the series; the full history is on `optim-w5-full`.

## Closed in F0 — host sampling

Same-binary A/B with a diagnostic null-sampling hook (uncommitted, preserved
as `s0b-sampling-null.patch`), timed 600-token generations per arm, SPEC_VERIFY_LOG
CSV extended with timestamps/chain. The real host sampling chain: **0.375
ms/round median (IQR 0.209)**, 0.196 ms/apply — 0.32% of the 117.353 ms
round wall, 0.42× the 0.9 ms/round admission threshold. Sign-coherence
denied (the control argmax over ~151k floats costs ~2× the real chain), and
the fork's top-k is already a 128-bucket radix partial sort (fixed range
[-10,10]) over the full vocab. The card was closed before Chunk 3 existed.

## Budget

| Movement | GPU minutes |
|---|---:|
| F0 probes (sanity + S0a 8k diagnostic + S0b) | ~7 (planned 35, −80%) |
| Card-1 FA-PP-TILE (v1 premise probe 13' + v2 host diag 50' + stage-1 cells 72') | ~135 |
| T23-CRC (harness + corpus on host/container gcc) | 0 |
| Series (git surgery) | 0 |
| Final A/B + battery + smoke | ~15 |
| **Total vs the 7 h cap** | **≈160 (38% of cap; point estimate was 350')** |

## Artifacts

Engine workspace, `logs/optim-w5/`: `s0a-ceiling.md`, `s0a-8k.md`,
`s0b-verdetto.md`, `s0c-crc.md`, `s0c-crc-bench.c`, `c1-levers.md`,
`c1-levers-v2.md`, `c1-gate1-verdetto.md`, `c1-verdetto.md`,
`t23crc-harness.cpp` + corpus log, `t14-*` image/smoke/A/B logs, the battery
`verdicts.tsv` per arm, and the row-by-row ledger `budget.md`.

Durable measurement notes: the perf-logger op tag is `FLASH_ATTN_EXT`
(substring `FA_EXT` does not match); non-verbose llama-bench nulls the log
callback — `GGML_LOG_*` markers need `-v`; FA diagnostic benches must run in
the q8_0 canon (q5_1 KV is −29% bytes, timings not comparable).

## Series contents

[`patches/optim-w5-series/`](../../patches/optim-w5-series/) — 2 patches,
`git am`-clean on the tree of `286f51d4b`, tree-verified at `449517003…`:
0001 the CRC32 PCLMUL fold + runtime dispatch, 0002 the review minors. The
FA-TILE experiments (v1 `cf0b4e9e0`, v2 `703e6197e`, both dropped) are
documented here and preserved on branch `optim-w5-full` @ `5bab7f594` of the
engine workspace; the `rocmfpx/` snapshot in this repo is the series plus the
wave-5 `AI_CHANGES.md` session entry (`ae48440e6`).

*Measured on the lab's [bare-metal Strix Halo](../../BARE-METAL.md), configuration as of the note's date.*
