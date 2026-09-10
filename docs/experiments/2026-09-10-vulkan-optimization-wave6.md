# Wave-6 — the round engine, restore/save and the per-request path (2026-09-10)

Wave-5 closed the FA-tile door; wave-6 went where the S0 measurements said the
remaining value was: the speculative round itself, the persist/restore path,
and the per-request host work. Deployed as `qwen4exp-mtp-vk-optim-w6`
(19 commits over `t32b-main-park` 9191e2db7, branch `optim-w6` @ `f37da5c0c`).

**Final interleaved A/B (container, prod-style, driver 25.3.6): +8.7% t/s**
(25.44 vs 23.50 tok/s mediana, timed-gen 600 tok), **battery 12/12 bit-identical
to the pinned canone** (`6cc4f583`, DANTE-1/DANTE-2 PASS — zero numerics drift).
Prefill: ~0 by design (all PP-shaped cards were exhausted by W4/W5).

## The measurements that steered the wave (step-0, before any code)

- **p-min sweep** (4 values × 2 interleaved): the default 0.75 is optimal;
  P(accept | p_dft ∈ [0.6, 0.75)) = 0.193 measured, ≈0 below 0.6. This killed
  the "offer the discarded tail to the verifier" idea *before* it cost a cell —
  and later killed it again with a cell (see W6-1).
- **Barrier census**: 6,423 Vulkan sync barriers per decode round (counter
  validated 1:1 against the stderr logger: 29,286 = 29,286).
- **Per-eval timeline** (E-lines): the dominant non-dispatch gap is the
  draft→verify transition (20.0 ms/round, 33% of gaps), the draft loop itself
  only 15% — which closed the device-resident draft loop (W6-10) as structural.

## What shipped

- **W6-1 leva-2 — keep the accepted MTP boundary through the F4 partial-reject
  reset** (+5.4% t/s alone, battery bit-identical). The F4 reset used to clear
  the whole impl state, so the next round mirrored with a *zero* h — one
  poisoned KV cell per partial reject (~80/600 tokens), plus the drafting
  boundary restarted from scratch. `draft_sync_reset` now suppresses drafting
  for exactly one round while the accepted boundary feeds the mirror. The
  companion "verify-the-tail" lever was measured NO-GO (−4% when combined:
  the tail is almost always rejected at temp 0.7 and each rejection arms the
  one-round suppression) — archived on `optim-w6-m1`, not deployed.
- **W6-3 — head-only drafter requant** (`output.weight` Q8_0 → Q4_0_ROCMFP4,
  the quantizer flags were already enough: `--pure --output-tensor-type`):
  −7.68% drafter file, +4.5% t/s, acceptance *improved* (0.652→0.667).
  Recipe: `scripts/w6t3-drafter-head-requant.sh`.
- **W6-2 — copy elimination in the decode graph** (qwen4exp only, host-side):
  the conv-history ring slots are copied without the per-slot `ggml_cont`
  (−629 CONT/eval), hc streams collapse straight off their views, QSA/PLE
  redundant conts dropped: CONT 96→12 per trunk eval (−87.5%), bit-exact by
  construction (verified with a synthetic mini-qwen4exp graph dumper + hash
  comparison; the strided-view consumers were checked against the Vulkan
  kernels' `nb[]` addressing).
- **W6-5/W6-7 — single-read restore, in-stream save CRC**: `load_disk` reads
  the payload once (CRC folded in the same pass, verify-then-mutate
  preserved), the save folds the CRC while writing instead of re-reading the
  0.9–2.3 GB tmp after fsync. Gauges at 128 MiB scale: save −45.5%, restore
  −25% (the ratio grows with entry size), file format byte-identical.
- **W6-6 — token-prefix + chat-template caches**: conversation prefix tokens
  are reused across requests (same tokenizer stream, byte-identical), the
  autoparser/PEG analysis and the rendered generation prompt are cached per
  (template, tools, schema, ...) key. Bit-identity proven on the real π corpus
  (412 KB / 127.6k tokens, 16/16 rounds) and 62 templates × 6 scenarios; the
  per-request host path drops 148 → ~20 ms at π scale.
- **W6-8 — park hot-RAM mirror** (flag `--cache-disk-park-ram-mirror <MiB>`,
  **default OFF**): RAM copy of the parked entry captured at save (CRC-checked
  against the sidecar at birth), served on restore with zero disk traffic.
  9.7× vs disk at 128 MiB scale. Ships inert.
- **A5-C3 — memoized backend-support verdicts** in scheduler assignment
  (exact-signature memcmp, kill-switch `GGML_SCHED_ASSIGN_MEMO=0`): the
  assignment decisions are pure per (op, shapes, src types) — the review
  caught `GGML_OP_UPSCALE` reading `op_params[0]` (mode/antialias bits, used
  by the vision models) before the memo could ship.

## What did not ship, and why (measured)

- **W6-4 — scoped per-buffer barriers**: tracker + scoped `vk::BufferMemoryBarrier`
  emitter with a lockstep-tested hazard predicate (40k checks, negative
  control persisted), activation verified in-cell (46,273 scoped vs 29,286
  global)… and tg parity (+0.3% ≪ σ). Integrated **default OFF**
  (`GGML_VK_SYNC_SCOPE=1` to opt in): the barrier count was never the cost —
  RADV makes the wide barrier cheap enough.
- **W6-9 k-slot graph cache**: crashed the server on the real multi-backend VK
  path (`split_graph` abort — the cached graph holds cross-backend copy
  tensors from a freed context; the single-backend harness could not see it).
  The rev-2 fix (as-built snapshot + restore before re-bind, crash reproduced
  and fixed on a lavapipe+CPU topology) lives on `optim-w6-c9`, opt-in, and
  waits for its own cells before any deploy.
- **W6-10 device-resident draft loop**: 15% of the gaps + P(accept)≈0 below
  p_min 0.6 — nothing to accelerate into.
- **Driver Mesa 26 (T33)**: flat on every cell (+0.3–0.7% ≪ threshold), and it
  *changes numerics* (class-D break in the 25.3→26.0 cycle). Not deployed.

## Process

Step-0 measurements before code; two-stage review per card (11 dispatched
implementers/reviewers; five review-fixes — a real UPSCALE hazard, a mirror
rejection-path divergence, an env-gate, a missing negative-control log,
arithmetic amendments — all caught before the cells); interleaved verdicts
only; GPU ledger reconciled from processes (~81' spent against a 5–7 h
ceiling). Full Italian report: `docs/optim/W6-REPORT.md` in the workspace.

*AI-assisted: GLM by z.ai.*
