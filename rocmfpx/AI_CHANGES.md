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

<!-- TEMPLATE FOR FUTURE AI SESSIONS:

## Session NNN — YYYY-MM-DD

**Scope:** Brief description of what was changed and why.

### `path/to/file.ext`

| Fix | Line(s) | Detail |
|-----|---------|--------|
| Short label | L123 | What changed and why. |

*(Repeat per file. Keep entries factual. Do not remove or rewrite earlier sessions.)*

-->
