# T8 Stadio 2 (concat MTP→DFlash) — experiment artifacts

Archived 2026-08-22 from the lab workspace (`~/workspaces/rocmfpx-strix-lean`) as a
safety net on the gitea mirror. These are verbatim copies of the working documents;
internal paths refer to the lab workspace layout, not to this repo.

Chain: spec → implementation plan → pre-registered gate experiment plan → gate report
(FAIL, content-controlled) → pattern-exclusion handoff (next cycle).

| File (original path in the lab workspace) | Content |
|---|---|
| `2026-08-21-t8-stadio2-concat-design.md` (`docs/superpowers/specs/`) | Binding design spec (review-loop approved 21/08; §8 annotated post-Task-1 with the VK 9-16 col dispatch change) |
| `2026-08-21-t8-stadio2-concat.md` (`docs/superpowers/plans/`) | 13-task implementation plan (Task 10 amended post ring-window fix: RS budget ×2.125, trailing-rollback invariant superseded) |
| `2026-08-21-t8-stadio2-faseA-gate.md` (`docs/superpowers/plans/`) | Pre-registered phase-A gate experiment plan (written BEFORE the runs; §10/§10.1 amended with the content-controlled AG verification, user-approved) |
| `2026-08-21-t8-fasea-conditioning.md` (`docs/research/`) | Gate report: pooled 0.9815 PASS is an artifact of cross-arm content divergence; content-controlled combined ratio 0.831 < 0.90 → FAIL. Cause: the conditioned DFlash block clones the previous pattern token (off-by-one, p_dft ≈ 0); free reasoning unharmed (1.133) |
| `2026-08-22-t8-pattern-exclusion-handoff.md` (`docs/research/`) | Handoff for the next cycle (pattern-exclusion brainstorm): mechanism, numbers, hypotheses to falsify first, design directions, reusable instrumentation |

Code: branch `t8-concat` (11 commits, inert at k1=0 — bit-identical certified), patches
`patches/t8-concat/0001-0011` in the lab workspace. Raw experiment data (40 MB: run
dirs, speclogs, metrics JSON) lives in the lab workspace `logs/test-t8-concat/fasea/`
(not archived here by size).
