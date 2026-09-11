# ROCmFPX — AI Change Log & Handoff

> **Instructions for any AI assistant reading this repository:**
>
> This file is the authoritative record of AI-assisted changes made to ROCmFPX.
> **Every time you make a meaningful change to any file in this repository, you must add an entry to the bottom of this file** under the next numbered session heading. Include: date, files touched, what changed, and why. Keep entries factual and terse — this is a technical log, not a narrative.
>
> Do not rewrite or remove existing entries. Append only.

---

## Session 001 — 2026-06-21

**Scope:** Audit and optimisation pass across `eagle3.cpp`, `rocmfpx.c`, and `scripts/rocmfpx-draft-profile.py`.

### `src/models/eagle3.cpp`

| Fix | Line(s) | Detail |
|-----|---------|--------|
| Log typo | 28 | `"EAGLE3gnorm_before_residual"` → `"EAGLE3 norm_before_residual"` (missing space broke log grep) |
| Style | 113 | Added missing space in `ggml_new_tensor_2d(ctx0, GGML_TYPE_F32,hparams…)` call |
| Concat dimension | 217 | Replaced loop variable `il` with explicit constant `0` as the `ggml_concat` dimension argument. `il` was always 0 for the single-layer EAGLE3, but using a bare loop variable made the intent invisible and would silently break if `n_layer > 1` were ever added. |
| Rename | 214, 257 | `inpSA` → `residual`. The variable is the residual connection target, not the self-attention input. Added clarifying comment explaining `norm_before_residual` behaviour. |

---

### `scripts/rocmfpx-draft-profile.py`

| Fix | Detail |
|-----|--------|
| Tokenize timeout | `urllib.urlopen(..., timeout=300)` → `timeout=10`. A stalled server previously hung the client for 5 minutes. |
| `p_split` emitted | All returned profile dicts now include `"speculative.p_split"`. Previously the key was absent, so `dense-coder` at `p_split=0.20` could not communicate that setting to callers. |
| 4-bracket context ladder | Replaced the single 48 k-token cliff with a stepped policy derived from acceptance-rate evidence in `ROCmFPX-EXPERIMENT.md`: |

**`fp3-mtp` policy ladder (new):**

```
< 16 384 tokens  → n_max=4, p_min=0.75, p_split=0.10
16 384–49 151    → n_max=4, p_min=0.25, p_split=0.10
49 152–98 303    → n_max=2, p_min=0.0,  p_split=0.10
≥ 98 304         → n_max=1, p_min=0.0,  p_split=0.10
```

**`dense-coder` policy (new — context-aware):**
```
< 98 304 tokens  → n_max=6, p_min=0.0, p_split=0.20
≥ 98 304         → n_max=3, p_min=0.0, p_split=0.10  (backed off at extreme context)
```

**`fp4-general` policy (new — context-aware):**
```
< 98 304 tokens  → n_max=4, p_min=0.0, p_split=0.10
≥ 98 304         → n_max=2, p_min=0.0, p_split=0.10
```

---

### `ggml/rocmfpx/rocmfpx.c`

#### C1 — Binary search for `rocmfpx_nearest_scale_ue4m3`

The original implementation was an O(126) linear scan over the UE4M3 table. Replaced with a binary search (matching the existing `rocmfp4_nearest_scale_ue4m3` in `rocmfp4.c`). The UE4M3 table is monotonically increasing, so the binary search narrows to a 2-element window then picks the closer neighbour. Tie-breaking kept identical to the old scan (prefer the lower byte value).

Estimated impact: ~18× reduction in `nearest_scale` call cost. Called ~(n_params/16) times during quantization — meaningful on large models.

#### C2 — Precomputed `rocmfpx_scale_table[127]`

Added a static `const float rocmfpx_scale_table[127]` initialised at compile time with all valid UE4M3 → FP32 values. Added `static inline rocmfpx_scale_lookup(uint8_t e)` for O(1) table access. All internal uses of `rocmfpx_ue4m3_to_fp32()` inside this file (MSE inner loops, scale-search clip checks, quantize/dequantize row functions) replaced with `rocmfpx_scale_lookup()`. The public `rocmfpx_ue4m3_to_fp32()` function is preserved for external callers.

#### C3 — `all_finite` fast path for MSE inner loops

`rocmfpx_prepare_mse_weights()` gained a `bool * all_finite` output parameter. Six `_finite` variants of the three MSE inner-loop functions were added (one unweighted + one weighted per format: FP3, FP6, FP8). These variants skip the `isfinite(x[i])` guard per element. All three scale-search dispatch functions (`_choose_scale_fp3_mse_impl`, `_choose_scale_fp6_mse_impl`, `_choose_scale_fp8_weighted_mse`) now propagate `all_finite` and route to the appropriate variant. In the common case of normal model weights (no NaN/Inf), this eliminates one conditional branch per element per MSE candidate evaluation.

#### C4 — Group pack/unpack replacing bit-by-bit `set_bits`/`get_bits`

**Removed:** `rocmfpx_set_bits` and `rocmfpx_get_bits`. Both looped over individual bits with a branch + read-modify-write per bit.

**Added:** Four `static inline` group functions:

| Function | Input → Output | Elements/call |
|---|---|---|
| `rocmfpx_fp3_pack8(dst, codes)` | 8 × uint8 → 3 bytes | 8 |
| `rocmfpx_fp3_unpack8(src, codes)` | 3 bytes → 8 × uint8 | 8 |
| `rocmfpx_fp6_pack4(dst, codes)` | 4 × uint8 → 3 bytes | 4 |
| `rocmfpx_fp6_unpack4(src, codes)` | 3 bytes → 4 × uint8 | 4 |

Both formats use the natural 3-byte group size (lcm(3-bits, 8) = 24 bits; lcm(6-bits, 8) = 24 bits). Every output byte is fully determined by the pack expressions — the `memset(yb->qs, 0, …)` call that preceded the old loop was removed.

**Rewritten:** All 4 quantize-row and 2 dequantize-row functions for FP3/FP6. Quantize collects codes into a stack array then calls pack in groups; dequantize unpacks all codes upfront then loops over elements with no bit arithmetic.

FP3 layout (3 bytes per 8 elements):
```
byte 0: v0[2:0] | v1[2:0]<<3 | v2[1:0]<<6
byte 1: v2[2]   | v3[2:0]<<1 | v4[2:0]<<4 | v5[0]<<7
byte 2: v5[2:1] | v6[2:0]<<2 | v7[2:0]<<5
```

FP6 layout (3 bytes per 4 elements):
```
byte 0: v0[5:0] | v1[1:0]<<6
byte 1: v1[5:2] | v2[3:0]<<4
byte 2: v2[5:4] | v3[5:0]<<2
```

---

## Future sessions — append below this line

## Session 002 — 2026-06-21

**Scope:** ROCmFPX production-preflight, agent quant, imatrix, DFlash capability, TurboQuant, and serving polish.

### `scripts/rocmfpx-model-capabilities.py`

| Fix | Detail |
|-----|--------|
| Capability helper | Added lightweight GGUF capability detection for MTP, diffusion, QAT, agent/coherent, and DFlash/DDFlash markers. |
| Serving profiles | Added model-aware serving profiles for known Nemotron agent, Qwen/Qwable MTP, generic MTP, diffusion, QAT, agent, and regular GGUF cases. |
| False-positive guard | Restricted short `qat` / `diffusion` matches to filenames and longer metadata markers to avoid raw tensor byte false positives. |

### `scripts/rocmfpx-production-preflight.sh`

| Fix | Detail |
|-----|--------|
| Preflight JSON | Added model kind/capability fields, full `launch_command`, warnings/errors, and `WRAPPER_OUT` generation. |
| Safety gates | Added hard-fail options for `REQUIRE_MTP=1` and `REQUIRE_PROFILE=1`. |

### `scripts/run-rocmfpx-mtp-server.sh`

| Fix | Detail |
|-----|--------|
| MTP auto-detect | Added model capability check so non-MTP models do not receive invalid `draft-mtp` flags unless `REQUIRE_MTP=1`. |
| Utilization knobs | Added `PERF_PRESET`, `PARALLEL`, polling, priority, GPU-layer, FlashAttention, fit, split, and host/offload toggles. |

### `scripts/quantize-rocmfpx-agent.sh`

| Fix | Detail |
|-----|--------|
| Imatrix support | Added `IMATRIX=/path/to/imatrix.gguf` pass-through to `llama-quantize --imatrix` with missing-file validation. |

### `ggml/rocmfpx/rocmfpx.c`

| Fix | Detail |
|-----|--------|
| Imatrix scale search | Added weighted quantization paths for ROCmFP3, ROCmFP6, and ROCmFP8 so accepted imatrix data affects ROCmFPX scale selection. |

### `ggml/rocmfpx/test_rocmfpx.c`

| Fix | Detail |
|-----|--------|
| Imatrix tests | Added weighted-MSE checks proving imatrix improves FP3/FP6/FP8 reconstruction on targeted calibration-weight cases. |

### `src/llama-kv-cache.cpp`

| Fix | Detail |
|-----|--------|
| TurboQuant policy | Added opt-in boundary-layer K protection for symmetric TurboQuant cache experiments. |

### Docs and gates

| File | Detail |
|------|--------|
| `README.md` | Added ROCmFPX family, agent quant, imatrix, TurboQuant, and contributor guidance. |
| `docs/ROCmFPX-SERVING.md` | Added preflight, model-kind guidance, utilization knobs, imatrix usage, TurboQuant asymmetric KV, and DFlash safety notes. |
| `docs/ROCmFPX-EXPERIMENT.md` | Documented ROCmFPX imatrix reference coverage. |
| `docs/ROCmFPX-HANDOFF.md` | Added handoff notes for ROCmFPX family usage and safety boundaries. |
| `scripts/check-rocmfpx-model-capabilities.sh` | Added synthetic capability/preflight/wrapper smoke coverage. |
| `scripts/check-rocmfpx-all.sh`, `scripts/check-rocmfpx-summary.sh` | Added capability gate integration. |
| `scripts/run-rocmfpx-turboquant-asym-server.sh` | Added safe asymmetric TurboQuant K/V serving wrapper. |

### Validation

| Check | Result |
|-------|--------|
| `scripts/build-strix-rocmfp4-mtp.sh llama-quantize` | passed |
| `scripts/check-rocmfpx-model-capabilities.sh` | passed |
| `scripts/check-rocmfpx-reference.sh` | passed, including imatrix weighted FP3/FP6/FP8 checks |
| Python/shell syntax checks | passed |
| DiffusionGemma BF16 → ROCmFP4 coherent agent quant | passed, `13,764.94 MiB / 4.57 BPW` |

## Session 003 — 2026-09-04

**Scope:** peg-native tagged tool calls rejected a duplicated required parameter, throwing away whole generations; made the grammar lenient (any arg may repeat after the required sequence) and the mapper duplicate-aware (last-wins + logged marker).

### `common/chat-auto-parser-generator.cpp`

| Fix | Line(s) | Detail |
|-----|---------|-------|
| Grammar leniency | 411-428 | The trailing repeat after the required-arg sequence now admits ANY arg (required included), not just optionals. A model re-emitting a required parameter at the end of a long generation (write emitted `path`, `content`, `path`) previously failed the whole tool-call rule. |

### `common/chat-peg-parser.h`

| Fix | Line(s) | Detail |
|-----|---------|-------|
| Mapper state | 29 | New `arg_pair_pos` (emitted pairs: start offset + name) to detect and resolve duplicate args; reset per tool open. |

### `common/chat-peg-parser.cpp`

| Fix | Line(s) | Detail |
|-----|---------|-------|
| Last-wins on duplicates | 348-395 | `is_arg_name` (marker at :366): a duplicate name logs `peg-native: leniency-hit: duplicate param '<name>' tolerated (last-wins)` and surgically removes the previous pair (other keys and their order preserved; leading-comma cleanup when the first pair is removed). Previously a duplicate produced a JSON string with a duplicate key (for grammar-allowed optional repeats) or a hard parse failure (required repeats). |

### `tests/test-chat-auto-parser.cpp`

| Fix | Line(s) | Detail |
|-----|---------|-------|
| Regression test | 2145+ | `tagged_duplicate_required_param`: drives the production builder `build_tool_parser_tag_tagged` with the Qwen-style tagged syntax — identical duplicate parses to the same args as the clean call; divergent duplicate resolves last-wins (equal to a clean call carrying the last value); the marker is asserted via a captured log file. |

### Validation

| Check | Result |
|-------|--------|
| `test-chat-auto-parser` | 71 tests / 420 assertions / 0 failures (includes the new test, RED before the fix) |
| `test-chat-peg-parser` | 32 tests / 198 assertions / 0 failures |
| Deploy smoke (production image) | pending at log time — image `qwen4exp-mtp-vk-optim3` in build; result recorded in the lab's deploy plan `2026-09-04-peg-lenient-dup-args-deploy-optim3.md` |

## Session 004 — 2026-09-05

**Scope:** peg-native tagged tool calls also failed when required parameters arrived out of schema order (05/09 task 190228: write emitted `content`, `path`); made the tagged grammar order-free and moved required-ness to a post-parse completeness check keyed by a serialized arena sidecar. Unknown parameter names still fail the grammar (05/09 task 181152: write with `<parameter=command>`).

### `common/peg-parser.h`

| Fix | Line(s) | Detail |
|-----|---------|--------|
| Arena sidecar | 310-316 | New public `tool_required_args` (tool name → required param names). The order-free grammar cannot express "each required at least once" (the PEG star is possessive), so required-ness travels with the parser arena instead. |

### `common/peg-parser.cpp`

| Fix | Line(s) | Detail |
|-----|---------|--------|
| Sidecar serialization | to_json/from_json | Optional `tool_required` field: emitted only when non-empty (arenas without it serialize byte-identical to before); read back if present (previously saved arenas stay loadable). Needed because the server ships the parser between threads as a string (`server-task.cpp` `parser.load(...)`). |

### `common/chat-auto-parser-generator.cpp`

| Fix | Line(s) | Detail |
|-----|---------|--------|
| Order-free grammar | 402-421 | The required-in-schema-order arg sequence is gone; args are a repeat of ANY parameter in any order/number. Required-ness is enforced post-parse instead (see chat.cpp). Unknown names still match no rule, so garbage params still fail the whole tool call. |
| Sidecar population | build_parser | After building the arena, tools in TAG_WITH_TAGGED mode get their required params recorded on `arena.tool_required_args` (same gating as the tool grammar itself). |

### `common/chat.cpp`

| Fix | Line(s) | Detail |
|-----|---------|--------|
| Completeness check | common_chat_peg_parse | After mapping (non-partial only): a tool call missing a required param logs `common_chat_peg_parse: rejected: required param '<name>' missing for tool '<tool>' (order-free leniency)` and throws the same "does not match the expected ... format" error the strict grammar used to raise — behavior parity for genuinely incomplete calls. |

### `tests/test-chat-auto-parser.cpp`

| Fix | Line(s) | Detail |
|-----|---------|--------|
| Regression test | tagged_reordered_required_params | Reversed required order parses to the same args as the clean call; optional-before-required parses; reorder+duplicate resolves last-wins; missing required still rejected with the logged marker; sidecar survives save/load; autoparser populates the sidecar through the real Hy3 template (wiring verified RED with the population disabled). |

### Validation

| Check | Result |
|-------|--------|
| `test-chat-auto-parser` | 78 tests / 441 assertions / 0 failures (new test RED before the fix: 5 sub-tests failing) |
| `test-chat-peg-parser` | 32 tests / 198 assertions / 0 failures |
| Deploy | NOT deployed — prepared only; the joint π test is live on `qwen4exp-mtp-vk-optim3` and a restart would cost the running session a ~20 min cold re-prefill |

## Session 005 — 2026-09-05

**Scope:** the 04/09 duplicate leniency re-serialized last-wins by ERASING the earlier pair, which reordered the accumulated args string and broke the append-only invariant of the streaming diff (`string_diff`, chat.cpp) — every leniency turn died client-side with "Invalid diff" (pi_agent errata 05/09 09:06, verified: 7/7 cancel↔marker in the server log). Duplicates now append as-is (duplicate JSON key, parsers resolve last-wins at parse time).

### `common/chat-peg-parser.h`

| Fix | Line(s) | Detail |
|-----|---------|--------|
| Mapper state | 29 | `arg_pair_pos` offset bookkeeping replaced by a simple `seen_args` name set (only used to log the leniency marker). |

### `common/chat-peg-parser.cpp`

| Fix | Line(s) | Detail |
|-----|---------|--------|
| Append-only duplicates | is_arg_name | On a duplicate name the marker still logs, but the pair is simply APPENDED instead of erasing/re-emitting the earlier pair. The mapper is now append-only by construction, so partial-parse args always extend the previous snapshot and `compute_diffs`/`string_diff` never throws. Final semantics unchanged: duplicate keys resolve last-wins at JSON parse time (server-side completeness check included). |

### `tests/test-chat-auto-parser.cpp`

| Fix | Line(s) | Detail |
|-----|---------|--------|
| Streaming regression test | duplicate_streaming_append_only | Simulates the server streaming loop (partial parses on a growing input, production `common_chat_parse` + `common_chat_msg_diff::compute_diffs`) over the 04/09 duplicate shape (path, content, path): every step must diff cleanly; full parse still resolves last-wins per key. RED before the fix: failed exactly at the duplicate step ("Invalid diff"). |

### Validation

| Check | Result |
|-------|--------|
| `test-chat-auto-parser` | 79 tests / 609 assertions / 0 failures (04/09 duplicate test still green: marker + last-wins semantics preserved) |
| `test-chat-peg-parser` | 32 tests / 198 assertions / 0 failures |
| Deploy | NOT deployed — prepared only; joint π test live on `qwen4exp-mtp-vk-optim3` |

## Session 006 — 2026-09-09

**Scope:** wave-4 optimization series (8 patches, branch `optim-w4` @`5bb661e1b`, deployed as `qwen4exp-mtp-vk-optim-w4`). One GO: split-k for tall-skinny prefill GEMMs (N1), promoted to default ceil mode. Four measured NO-GOs documented and dropped from the deploy branch (kept on `optim-w4-full` @`d284454f8`): pooled indexer keys (tap cost = gather saved), int-dot/tile MMQ variants (lose to f16-dequant on gfx1151; includes a latent warptile bug fix that stays), f16 FA accumulator on quantized KV (near-tie degenerate output), element-wise fusion batch (below the async-submit floor).

### `ggml/src/ggml-vulkan/ggml-vulkan.cpp`

| Change | Detail |
|--------|--------|
| Small-m split-k (N1, GO) | `ggml_vk_guess_split_k` gated k-parallelism on `m >= wg_denoms[0]`, so qwen4exp prefill GEMMs (m=4 k=10240, m=48 k=2560, m=1) ran one wave-row grid at 34 GF/s. New branch allows split-k (cap 8, k>=2048, n_tiles <= cores/2) for that regime on the existing tile pipeline + split_k_reduce path. Default = ceil mode (`GGML_VK_MM_SMALLM_SPLITK` 0/1/2 override). Measured interleaved: pp8192 +3.95%, c32k +2.33%, ppl improved, battery clean. |
| Env-gated int-dot/tile diagnostics (N4) | `GGML_VK_MMID_INTDOT` / `GGML_VK_MMID_F16_TILE_L` / `GGML_VK_MMID_F16_MOE_TILE` / BK_STEP=4 variants, default OFF — instrumentation kept for reproducibility; measured all slower than the f16-dequant tile-m on gfx1151/RADV. |
| Warptile fix (latent bug) | AMD-GCN branch resized `l_warptile_mmq_int` to BLOCK=256 against wg denominators {128,128,1} — dense MMQ l-tile garbage for m>32 (inf at m=2560). Explicit coherent warptile + 15 regression cases. |
| Dropped experiments | SIGLU op + element-wise fusion batch (N2) and f16 FA accumulator on quantized KV (N3) were committed during the wave, measured NO-GO, and dropped from the deploy series via rebase (tree-verified). |

### `tests/test-backend-ops.cpp`

| Change | Detail |
|--------|--------|
| Regression cases | Prod-shape small-m MUL_MAT cases (m=4/k=10240, m=48/k=2560, m=1 f32, repeated-row variant) + MMQ tile regression battery (15 cases incl. m=2560). |

### `tests/test-llama-archs.cpp`

| Change | Detail |
|--------|--------|
| qwen4exp skip | Fixture skip-list (FIXME) after the IDX-POOL revert dropped the C2-3 test fixture the arch test depended on (same pattern as DEEPSEEK4); full fixture to be re-landed separately. |

### Validation

| Check | Result |
|-------|--------|
| `test-backend-ops` (Vulkan0) | MUL_MAT 4/4 off/sk1/sk2 · rocmfp4 sweep 150/150 on/off · FA prec 3/3 · GLU 145/145 |
| `test-llama-archs` | rc=0 (137 lines, qwen4exp skipped) |
| ppl | Dante 1,1215 / wikitext 3,4927 (gates ±0,5% PASS) |
| Battery | card run 12/12; deploy re-check 11/12 with DANTE-2 at the near-tie boundary — seed-retry shows the same failure basin in the unpatched baseline (w3 seed 2026 fails with the same sha), verdict: sampling border, not systematic |
| Interleaved A/B | c32k +2.33% (4/4 arm separation) · server timed-gen parity w3 vs w4 (24.02–24.08 tok/s) |
| Deploy | LIVE on `qwen4exp-mtp-vk-optim-w4` (health OK, prod flags unchanged, no new flags) |

## Session 007 — 2026-09-09

**Scope:** wave-5 optimization series (branch `optim-w5` @`e8a4d8c5f`, series = 2 patches on top of `optim-w4` @`286f51d4b`; full in-wave history kept on `optim-w5-full` @`5bab7f594`). One GO: CRC32 PCLMUL for the persist-file checksum (T23-CRC; restore-check post-deploy pending). One card closed in F0 without code: sampling per-round cost. One structural NO-GO dropped from the series: FA prefill tile variants (card-1, v1 + v2).

### `src/llama-persist-meta.cpp` (T23-CRC, GO — restore-check post-deploy pending)

| Change | Detail |
|--------|--------|
| CRC32 PCLMUL folding | Streaming CRC of the persist file replaces the bytewise table loop on x86-64 with a PCLMULQDQ folding path (`__attribute__((target("pclmul,ssse3")))`, 16 B per iteration, fold state deg <= 94). Bit-exact with the table path: same poly/init/final-xor, reflected-table correspondence documented in-code; folding constants x^e mod F derived at first use (magic static) instead of transcribed literals, with a one-off assert re-deriving k96 the slow way. |
| Runtime dispatch + fallback | `crc32_pclmul_available()` probed once (magic static); non-x86-64 or no PCLMUL → table path unchanged. Stream API (`crc32_stream_update`/`_final`) owns the <16 B remainder on both paths. |
| Review fix (2nd commit of the series) | `assert((size & 15) == 0)` in `crc32_fold_update`; `crc32_stream_final` takes `const&`; comment fixes (reference-harness wording, "96 successive multiplications by x"). |
| Measured basis | CRC = 70% of real restore time; expected −64% restore time with PCLMUL active. |

### Closed in F0 (no code)

| Card | Result |
|------|--------|
| card-2 sampling | Closed in F0: 0.375 ms/round measured vs 0.9 ms/round gate. |

### Dropped from the series (card-1 FA-PP-TILE, NO-GO structural)

| Variant | Detail |
|---------|--------|
| v1 `GGML_VK_FA_PP_TILE=1` (Bc64 per-shape override, `ggml-vulkan.cpp`) and v2 `=2` (Br32 dual-panel coopmat1, `ggml-vulkan.cpp` + `flash_attn_cm1.comp`), both default OFF | Interleaved c32k: −5.96% (4/4 cell separation); FA op share +25.5%; occupancy LDS 2→1. coopmat2 absent on gfx1151/RADV; roofline re-read confuted the F0 hypothesis. Both commits dropped from the deploy series via cherry-pick rebuild on the base; full history on `optim-w5-full` @`5bab7f594`. |

### Validation

| Check | Result |
|-------|--------|
| Series rebuild | cherry-pick clean (no conflicts); `git diff --stat 286f51d4b..optim-w5` touches only `src/llama-persist-meta.cpp`; `ggml/` tree identical to base; `src/llama-persist-meta.cpp` bit-identical to the in-wave chain tip `5bab7f594` |
| `git am` gate | patches re-applied on a detached worktree at the base → resulting tree SHA identical to `optim-w5` (`449517003313aba8b97d4bcb22ccaf470a82f21e`) |
| Bit-exactness | PCLMUL path vs table path produce equal CRCs (reference harness in the wave-5 experiment notes) |
| Restore-check | PENDING post-deploy (real restore timing with PCLMUL active) |
| Deploy | NOT deployed — series prepared only |

## Session 008 — 2026-09-10

**Scope:** T32-b main-park series (branch `t32b-main-park` @`30d2c49be`, 11 commits on top of `optim-w5` @`ae48440e6`; linear reflog, nothing dropped/replaced, no `t32b-full` conservation branch). Park disk-library for the main-agent context: at a task-switch the main task's boundary disk-save is redirected to a dedicated park library (own MiB budget, own persist subdir), and on the next main load the park entry is a restore candidate under the existing cache rules. All behind two new flags, both default OFF.

### `common/common.h` / `common/arg.cpp`

| Change | Detail |
|--------|--------|
| New flags | `--cache-disk-park-mib` (default `0` = feature fully off; min 1024 when on) and `--cache-disk-park-heuristic` (default `none`; `none` \| `longest`). |

### `tools/server/llama-park-identity.{h,cpp}` (new)

| Change | Detail |
|--------|--------|
| Identity helpers | Pure functions, no server state / no llama runtime: `park_cfg` (mib + heuristic enum), `heur_state` (multiset of seen prompt token counts), `park_enabled`, `park_from_header` (`X-Pi-Role: main`), `park_from_heuristic` (LONGEST: >32768 and the longest seen), `park_task_is_main` (header wins over heuristic), `park_should_redirect` (task-switch redirect predicate, spec step 4.2; caller adds the live-park-instance conjunct). |

### `tools/server/server-context.cpp`

| Change | Detail |
|--------|--------|
| Park library instance | Persist subdir parametrized (`main-park`); park instance exists exactly when the disk cache is configured; a boot-failed park instance is fully inert (general library unaffected). |
| Save redirect at task-switch | Disk half of the boundary save goes to the park library (`persist park: redirect disk-save of main task tokens=N`), with supersede of the previous park entry and a safety-net fallback to the general library if the park save fails (`persist park: fallback to general reason=park-disk-save-failed`). |
| Park candidates in load() | Park entries are restore candidates under the existing selection rules; owner bookkeeping keeps RAM-cache vs park attribution explicit (who served the main-return). |
| LONGEST registry | `park_heur.seen` insert gated on (COMPLETION\|\|INFILL) && park_enabled && heuristic==LONGEST (the only reader) — no registry growth without a reader. |
| Metrics | `park_restore_crc_ms` exposed in the server metrics JSON. |

### `tools/server/server-task.{h,cpp}`

| Change | Detail |
|--------|--------|
| X-Pi-Role wiring | `park_main` explicit task-param field (client header `X-Pi-Role: main`); negative gate scoped to the `params_from_json_cmpl` body. |
| Park budget + logs | Park budget accounting separate from the general persist budget; `persist park:` log family (enabled / disabled:reason=no-persist-budget / save / supersede / fallback). |
| Gauge | `park_restore_crc_ms` (double) on the task result, fed by the pending park-restore verification. |

### `tests/`

| Change | Detail |
|--------|--------|
| 4 new binaries | `test-park-identity` (links `llama-park-identity.cpp` directly; combined header-vs-heuristic cases; source-scan gates surface their skip count in the verdict line instead of masking under ALL PASS), `test-park-library`, `test-park-redirect` (touch/supersede/fallback over two live cache instances), `test-park-load` (park candidate selection + owner routing) — the last three link `server-context`. |

### Validation

| Check | Result |
|-------|--------|
| Test binaries (Phase A, container) | 5/5 PASS — 4 new park binaries + pre-existing `test-persist-meta` (regression) |
| `git am` gate | series re-applied on a detached worktree at `optim-w5` → resulting tree SHA `b5bb20b1cfe961ab6657dd29cbe393840ba52790` identical to `t32b-main-park` |
| i3 byte-parity | 5 PASS / 0 FAIL — sha256 canone (w5-ref) vs ours identical (`f06d0e1fa…`, timed-gen 600 tok) |
| Park mechanics live (i1) | redirect + real supersede + save in the server log: `persist park: enabled: … limit_mib=10240`, `redirect disk-save of main task tokens=59161`, `supersede entry=1 new=2`, `save entry=2 bytes=936828332` |
| i1 / i2-i4 restore gates | NO-PASS (final runs: i1 6 PASS / 2 FAIL; i2/i4 4 PASS / 3 FAIL) — main-return restore hits the pre-existing T23 exact-boundary (lcp-esatto) limit: structural, not a T32-b defect, zero regression vs today (flags off = current behaviour). Details in `docs/t32b-REPORT.md` |
| Cells driver note | host Mesa 26.0.8 (non-prod): i-cells were functional gates, timings not comparable with prod (container 25.3.6) |
| Deploy | NOT deployed — default 0 = zero effect; deploy and flag activation = user GO (prerequisites in `docs/t32b-REPORT.md`) |

## Session 009 — 2026-09-10

**Scope:** W6-2 (wave-6 card A1-C1) — eliminate the eliminable CPY/CONT copies from the qwen4exp decode graph, host-side only; P3 fine attribution first.

### `tests/test-copy-elim.cpp` (new) + `tests/CMakeLists.txt`

| Fix | Detail |
|-----|--------|
| Attribution harness | Synthetic mini qwen4exp GGUF writer (trunk+MTP and draft-only; GDN + QSA + PLE + hc + MoE gate/up separate) + fixed decode schedule (prefill 8, N×1, one n=3 batch, 3 drafter evals with h input, target n_rs_seq=16) + per-node graph dump via the eval callback (op/name/shape/contiguity/FNV data-hash/src ops/named ancestor) + logits/state hashes for the bit-exact gate. Built via llama_build, not in ctest. |

### `src/models/qwen4exp.cpp`

| Fix | Line(s) | Detail |
|-----|---------|--------|
| conv-ring cont | build_conv_state_at | cpy the strided tail view directly (CPY reads through nb; delta-net-base always did); the per-slot `ggml_cont` added K−1 byte-identical CONT dispatches per layer-eval in decode. ~630 CONT/eval-target saved in prod |
| hc collapse | build_hc_mix | fold the streams straight off their views (ADD already consumed a strided src1); drops the cont of stream 0, 97 CONT/eval-target |
| QSA pooled | build_qsa_top_k | drop the per-slice cont (ADD reads views; the last ADD/scale yields the contiguous tensor the reshape needs) |
| QSA q/top_k | build_qsa_top_k | drop cont of already-contiguous fresh tensors ahead of reshapes |
| PLE tail | build_ple | drop cont of silu's fresh output before reshape_3d |
| kept (documented) | build_ple | the kernel-column cont is a stride-kern gather (view_1d = row, wrong data — caught by the harness, reverted); shifted/permute/gate conts stay for consumer/matcher layout |

**Verification:** bit-exact vs base `482d84333` (same harness both sides): 10/10 logits hashes + full state hash identical; 6.158/6.169 named tensor instances identical (the 11 diffs are the GDN fused result's never-read unwritten snapshot tail, proven by the matching cache writes). CONT per trunk eval 96→12, drafter 4→1, CPY unchanged (structural ring writes). Artifacts: `logs/wave6/w6m2-attribuzione.txt`, dumps and hashes alongside. VK build rc=0.

## Session 009b — 2026-09-10 (W6-9 gradino-2, A2-C2; rinumerato da 009 nel cherry-pick W7-3 su w7p3)

**Scope:** k-slot cache of `llm_graph_result` in `llama_context` (wave-6 card W6-9 / A2-C2). The MTP round alternates graph keys inside one context (verify `n=d+1` all-outputs -> plain `n=1` on the target; process -> step on the drafter), so the legacy single slot (`gf_res_prev`) missed almost every eval (measured 4-4.5 rebuilds / 7.7 evals per round). Base `optim-w6` @ `482d84333`, branch `optim-w6-c9`.

### `src/llama-context.h`

| Fix | Line(s) | Detail |
|-----|---------|-------|
| Cache members | ~383-427 | `gf_res_prev` replaced by `gf_res_cache` (slot 0 always exists after `sched_reserve()`, extra slots created lazily up to the cap), `gf_res_cur` (slot whose graph the sched currently has allocated), LRU clock, `graph_cache_max_nodes`, `graph_cache_slots` (env `LLAMA_GRAPH_CACHE_SLOTS`), helpers `graph_cache_reinit` / `graph_cache_reset_all`, perf counters `n_cache_hits` / `n_cache_rebinds` / `n_cache_evicts` + `t_graph_rebind_us`. |

### `src/llama-context.cpp`

| Fix | Line(s) | Detail |
|-----|---------|-------|
| Env flag | ~250-267 | `LLAMA_GRAPH_CACHE_SLOTS` parsed in the ctor (clamped to [1, `LLAMA_GRAPH_CACHE_SLOTS_MAX`=8]); 0/1 = legacy single-slot behavior (same-binary A/B, runtime rollback). |
| Slot lifecycle | ~490-520 | `graph_cache_reinit()` (sched re-created: drop all slots, re-create slot 0) called from `sched_reserve()`; `graph_cache_reset_all()` replaces the single-slot `gf_res_prev->reset()` at the two external-sched-reset sites (`memory_update`, `graph_reserve`) and at the top of the training path (`opt_epoch_iter` builds into slot 0). |
| Cache logic | ~1429-1600 | `process_ubatch`: warm slots tried most-recently-used first, each with the UNMODIFIED `llm_graph_result::can_reuse` predicate (the slot lookup only routes, it never widens the predicate). Hit on the current slot = legacy fast path (no sched work). Hit on a non-current slot = re-bind: `sched_reset` + `alloc_graph(cached_gf)` (the sched runs whatever it last allocated — `compute_splits` ignores the passed gf while `is_alloc` is set — and the cached tensors' data pointers went stale when another graph was allocated), skipping only `reset` + `build_graph`. Miss = LRU victim (or lazy new slot); rebuild as before. `pipeline_parallel` synchronize covers the re-bind too. |
| Perf | ~3684-3725 | `perf_print_graph_build_data` also prints `graph cache slots = used / cap`, `graph cache hits (rebinds, evictions)` and rebind timing; `perf_reset` clears them. Early-return condition extended so a pure-hit run still prints. Eviction counter only counts warm slots dropped while other slots exist (single-slot misses are the legacy rebuild). |

**Gate N (bit-exact of construction)**: harness `logs/wave6/w6c9/w6c9-cache-harness.cpp` + driver `run-gate.sh` (outside the repo, modeled on `tests/test-llama-archs.cpp`): seeded tiny llama fixture, CPU-only, MTP-shaped key alternation with a 5-key working set, two invalidation phases (sched re-reserve via `llama_set_sampler`, full KV clear via `llama_memory_seq_rm`). All per-eval output hashes identical for k=0 vs k=2 vs k=3 (two seeds); counters: k=0 = 25 rebuilds / 0 rebinds (legacy parity), k=3 = 14 rebuilds / 16 hits / 11 rebinds / 8 evictions. `GATE-N: PASS` (`logs/wave6/w6c9/gate-output.txt`). Memory: 15.86 MiB per slot at the prod `max_nodes` = 39168 (`ggml_tensor_overhead()` = 368 B measured in the builder container).
## Session 010 — 2026-09-10

**Scope:** W7-5 — per-request pi path S3/S4 residuals: census + tokenizer bit-identical optimisations (card docs/research/wave7/w7p5-report.md).

### `src/llama-vocab.cpp`

| Fix | Line(s) | Detail |
|-----|---------|--------|
| bigram text | ~274 | `llm_bigram_bpe` no longer carries `std::string text`; stale entries detected from symbol sizes (`left.n + right.n != bigram.size`), the same pattern as `llm_bigram_spm`. Symbol text pointers never move and `n` never shrinks, so the size check is exactly equivalent to the old concatenation compare. Kills ~13 heap allocations per merge |
| pop-loop | ~646 | drop the per-merge `left_token`/`right_token` constructions and concatenation check |
| add_new_bigram | ~724 | rank lookups go through member scratch strings (SSO covers BPE pieces): no allocation per call |
| final lookup | ~688 | reused `str.assign` instead of a fresh `std::string` per output token |
| queue reset | ~608 | `llama_priority_queue::clear()` keeps the vector storage across words |

### `src/unicode.cpp`

| Fix | Line(s) | Detail |
|-----|---------|--------|
| byte encode | ~196 | `unicode_byte_encoding_process`: removed the per-word decode->re-encode round-trip (words are concatenations of `unicode_cpt_to_utf8`, valid UTF-8 by construction, so the round-trip reproduced the same bytes) |
| byte table | ~1161 | `unicode_byte_to_utf8`: flat 256-entry table instead of a per-call `unordered_map` lookup |

### `tests/`

| Fix | Detail |
|-----|--------|
| new gate | `test-per-request-path.cpp` + CMake wiring: offline census/bit-identity harness for the server per-request path (json::parse -> chat params -> template apply -> prefix-cache tokenize -> task params) with per-component timings and sha256 of prompt/token streams |

**Verification:** token bit-identity 16/16 real pi prompts (T12 rounds), pre-change == post-change == pinned Python tokenizer (staging-w7/tokenizer-trunk); formatted-text sha unchanged 16/16; test-tokenizer-0 15/15 vocabs; test-token-prefix-cache (qwen35 + pi corpus 16/16, gpt2, spm) and test-chat-template-cache (62 templates / 3720 applies / 0 fail) all PASS. Tokenize full 741->412 ms on the 412 KB pi corpus (1.8x), steady splice 2.5x; llama-server builds rc=0. Artifacts: logs/wave7/w7p5-*.



## Session 009c — 2026-09-10 (W6-9 gradino-2 rev-2: multi-backend re-bind fix + default flip; rinumerato per deduplica 010 nel merge wave-7)

**Scope:** the W6-9 cells crashed the server on the real VK path with the cache enabled (k=3 default): `ggml_abort ← ggml_backend_sched_backend_id_from_cur ← split_graph ← alloc_graph ← process_ubatch`, message `pre-allocated tensor (attn_inp_kq_mask (copy)) in a buffer (Vulkan0) that cannot run the operation (CPY)` (`logs/wave6/w6c9c-slot3-r1.log`). Root cause analysis + fix + default flip to legacy. Perimeter still `src/llama-context.{h,cpp}` only.

**Root cause (two in-place mutations of the user graph by the allocation path, verified in `ggml-backend.cpp`/`ggml-alloc.c`):** a cached `llm_graph_result` holds the graph as it was LEFT by its last allocation, not as built —
1. `ggml_backend_sched_split_graph` replaces `node->src[j]` with its cross-backend copy tensors, created in `sched->ctx`, which split_graph **frees and re-initializes at every call** (`ggml-backend.cpp:1266-1268`) — the cached graph's srcs dangle into freed/reused memory;
2. the backend graph optimizer (`ggml_vk_graph_optimize`) reorders `gf->nodes[]` through the split views;
3. the tensors keep their gallocr-assigned `buffer`/`data`, which `ggml_gallocr_init_tensor` would NOT re-assign (`data != NULL` skip) and which `backend_id_from_cur` probes as pre-allocated (the cell abort).
A fresh build never sees this because freshly built tensors have `buffer == data == NULL` and pristine links.

### `src/llama-context.h`

| Fix | Detail |
|-----|-------|
| Default flip | `graph_cache_slots` default 3 → 1 (legacy). The cache is strictly opt-in (`LLAMA_GRAPH_CACHE_SLOTS >= 2`) until the cells re-pass; a default can never crash. |
| Slot snapshot | `gf_res_slot` gains `built_nodes` / `built_srcs` (as-built topology, ~0.9 MB per slot at 10k nodes) + `graph_result_restore_built()`. |

### `src/llama-context.cpp`

| Fix | Detail |
|-----|-------|
| Snapshot | taken in the miss path right after `model.build_graph()` and BEFORE `ggml_backend_sched_alloc_graph` (the mutation point): node order + all `src[]` links, via public API (`ggml_graph_nodes`). |
| Re-bind restore | `graph_result_restore_built()` puts the cached graph back to the exact as-built state (nodes order, src links, `buffer`/`data` scrubbed to NULL for every tensor in the result's ggml context — weights/KV/cross tensors are not in that context and keep their persistent buffers), then `sched_reset` + `alloc_graph` as before. The re-bind now goes through exactly the same split+alloc a fresh build would. |

**Repro + gate**: the crash is reproducible in the builder container with zero GPU via lavapipe (`VK_ICD_FILENAMES=lvp_icd...`, `GGML_VK_VISIBLE_DEVICES=0`) + CPU — pre-fix: abort at the FIRST rebind (`repro-prefix-k3.*`: rms_norm shape assert — the dangling-src manifestation; the RADV cell aborted earlier on the stale-buffer probe — same root). Post-fix gate (`logs/wave6/w6c9/gate-output.txt`): all arms bit-exact incl. the multi-backend arm (hashes k=0 ≡ k=3 on [VK-lvp, CPU], 25→14 rebuilds, 11 rebinds, 8 evictions); default without env = legacy (1 slot, 0 rebinds) on the same topology.
## Session 011 — 2026-09-10 (W7-4 FR-Spec; rinumerato da 010 nel rebase su optim-w7 con W7-5) — 2026-09-10

**Scope:** W7-4 FR-Spec — port of the 5-commit d2t series from the drluoto reference clone (`staging-w7-frspec-ref` @ `fb367b8cf2cf`) onto `w7p4` (= optim-w6 `f37da5c0c`): draft-vocab trim for the qwen4exp MTP sidecar. Tooling (map builder, trim wrapper, gate scripts) lives OUTSIDE the repo in the workspace (`scripts/w7p4-*`, `staging-w7/w7p4-*`), per wave convention.

### `src/models/qwen4exp.cpp`

| Fix | Line(s) | Detail |
|-----|---------|--------|
| d2t/t2d loader | load_arch_tensors | `d2t` meta-gated branch: forward map (len = n_draft rows) sizes `output.weight` to `n_vocab_out`; inverse map (len = n_vocab, I32) reads n_draft from `output.weight` ne[1]; assert when output falls back to a duplicated token_embd |
| MTP inputs as graph outputs | graph_mtp ctor | `ggml_set_output(inp->tokens/.h)` after `mtp_h_input` naming (CIRU H121: continuation step rewrites them; allocator could recycle storage between steps) |
| FR-Spec spread | graph_mtp head | after `build_lora_mm(head_w, ...)`: t2d inverse-gather (get_rows over [n_out, n_draft+1] with -inf sentinel row) or d2t forward scatter (`ggml_set_rows` into a -inf-filled full-vocab tensor), then `result_output` cb; TRUNK graph left untouched (the draft never runs it) |

**Port notes (per commit, full detail in `docs/research/wave7/w7p4-report.md`):** `7a3aa1dd5` (f01e30e42) and `6b7713dd6` (dfcaeb160) auto-merged clean (anchor rename `tf`→`trunk_flags`/`mtp_flags` absorbed); `cd9c998b0` (e0b617b71) auto-merged clean; `ff146108d` (53aef6bac) and `ebb3def77` (d786226ef) conflicted on the ref's divergent graph_mtp (dsv4 lineage) — resolved by keeping OUR graph_mtp and applying only the semantic change by hand; the trunk-graph deletion side of `ebb3def77` auto-merged. Series net on our tree +55/−1, identical to the reference series. Interaction with W6-2 copy-elim: none — W6-2 touches build_hc_mix/QSA/conv/PLE helpers, disjoint regions. No files outside `qwen4exp.cpp` were needed (`LLM_TENSOR_D2T` + `llama_model::d2t` already exist upstream for EAGLE3/dflash).

**Verification:** build in container `docker-llm-service:w2-vk-builder` rc=0 (`logs/wave7/w7p4-build.{log,err,rc}`); artifact gates 5/5 PASS (`logs/wave7/w7p4-sidecar-gates.txt`); gate N: see `docs/research/wave7/w7p4-report.md`.


## Session 012 — 2026-09-10 (W7-1 fusione EW host-side)

**Scope:** W7-1 (wave-7 Task 4) — fuse adjacent element-wise chains in the qwen4exp graph-builder where the cpu/vulkan matchers don't fire, unblocking EXISTING fusions only (no shader changes). Census first (W6-2 per-node dump + matcher simulation, `logs/wave7/w7p1-census.{py}` + report §1): the grouped-RMSNorm `rms_norm → reshape(flat) → mul` chain missed `RMS_NORM_MUL` at every hyper-connection mix (2/layer + final + drafter + 3 PLE sites) because the flat gamma relabel sat between the two ops; the GDN alpha `mul_mat → reshape(id) → add` missed `MUL_MAT_ADD` (mat-vec) the same way.

### `src/models/qwen4exp.cpp`

| Fix | Line(s) | Detail |
|-----|---------|--------|
| Gamma re-chunk at load | load_arch_tensors | `rechunk_hc_gamma` helper: metadata-only `[hc_dim] → [n_embd, hc]` for `output_hc_norm`, per-layer `hc_attn_norm`/`hc_ffn_norm`, `nextn.hnorm`, `ple_norm_{key,query,conv}` (same contiguous storage; asserts type/contiguity; null- and already-chunked-safe). Weights stay LEAFs so no relabel node interleaves. |
| hc mix mul in place | `build_hc_mix` | gamma multiplies the normed `[n_embd, hc, T]` tensor directly (broadcast over T); the flat `[hc_dim, nt]` relabel moves AFTER the mul → RMS_NORM+MUL adjacent, same shape, f32, contiguous ⇒ fusion fires on VK (and on CPU via `ggml_cpu_try_fuse_ops`). |
| drafter hnorm | `graph_mtp` ctor | mul directly on the normed streams; the two flat relabels collapse (consumer concat already wants `[n_embd, hc, T]`). |
| PLE grouped norm | `build_ple` `grouped_norm` | mul lands straight on the normed `[n_embd, hc, T]`; inner flat relabels dropped. |
| GDN alpha | `build_layer_attn_linear` | early identity `reshape_3d` between the `ssm_alpha` matmul and the `ssm_dt` bias add dropped; the closing `reshape_4d` already restores the mixer's shape; add/softplus/mul broadcast identically either way. |

**Bit-exactness (read from both matcher implementations before claiming):** VK `rms_norm.comp` `do_multiply` is the same shader with the same sum-of-squares reduction and the same left-to-right `(scale·a)·b` association (D_TYPE=f32 round-trip lossless); VK mat-vec `MAT_VEC_FUSION_FLAGS_BIAS0` adds the bias in-register after the identical dot accumulation; CPU fused template computes the same `(x·scale)·w` as unfused vec_scale+mul with the identical double accumulator. Broadcast pairing `gamma[k] ↔ element k` unchanged by the relabel move (row-major order preserved).

**Known boundary:** `llama_model_save_to_file` on a patched build writes the gammas already chunked (`[n_embd, hc]`); the loader's strict per-dim check rejects them (patched or not). No in-tree consumer saves qwen4exp (`test-llama-archs` skips the arch; quantizer works at loader/GGUF level, untouched).

**Verification:** gate N (W6-2 §3 pattern, `staging-w7/w7p1-gateN.sh`), VK build + interleaved cells (`staging-w7/w7p1-{build,base-build,celle}.sh`) — outcomes in `docs/research/wave7/w7p1-report.md` §3-§5.

<!-- TEMPLATE FOR FUTURE AI SESSIONS:

Addendum (post final-review): fix `423a7b59b` — euristica LONGEST nutrita con la size d'ARRIVO del task (task_prev->tokens, spec r3) invece del boundary di slot (che include i generati e rendeva l'eguaglianza irraggiungibile); gate sorgente anti-regressione nel test. Serie finale: 13 commit, tree SHA `808af2282` (gate git-am pari).

## Session NNN — YYYY-MM-DD

**Scope:** Brief description of what was changed and why.

### `path/to/file.ext`

| Fix | Line(s) | Detail |
|-----|---------|--------|
| Short label | L123 | What changed and why. |

*(Repeat per file. Keep entries factual. Do not remove or rewrite earlier sessions.)*

-->
