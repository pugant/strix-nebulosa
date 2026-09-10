<p align="center"><img src="logo.svg" alt="Strix Lean — the lab's mascot" width="220"></p>

This is a llama.cpp laboratory for one machine class: AMD Strix Halo (Ryzen AI MAX+ 395, 128 GB unified memory) — the owl above is **Strix Lean**, this lab's mascot. The headline is the **qwen4exp** line — Qwen3.8-Flash-Next, the 180B-class hybrid mixing gated-deltanet linear attention, 4-stream low-rank hyper-connections and a 51B-parameter PLE n-gram table. Our ROCmFP4 quant of it is 98.5 GiB and, with `--ple-disk`, runs on the same 128 GB machine **with ~36 GB of RAM to spare**: the PLE table is never loaded — its blocks are read on demand straight from the GGUF on disk, the output stays char-identical, and warm token generation costs ~3%.

Around that model runs a small production stack, every piece of it measured on this hardware: one `llama-server` holding two speculative drafters — the target's built-in MTP (multi-token prediction) layer and an external DFlash2 block-diffusion drafter — with routing that picks the drafter that fits each request; vision input working *together with* MTP speculation; a reasoning budget with a warn window at 75% instead of a hard cliff; and, behind all of it, the full ROCmFP4-STRIX_LEAN quantization pipeline (imatrix → quantize → sanitize → publish).

We are an unofficial community lab, not a vendor. We publish the measurements, the scripts and the patches — and the NO-GOs with the same care as the wins.

## Supported

- **Hardware** — AMD Strix Halo: Ryzen AI MAX+ 395, 128 GB unified LPDDR5X (~256 GB/s of UMA bandwidth — the budget everything here is tuned against), Radeon 8060S iGPU (gfx1151, RDNA 3.5) — [full dated configuration](BARE-METAL.md).
- **Backends** — Vulkan (RADV/Mesa, our primary) and ROCm/HIP 7.2.4.
- **Models** — the quants published by this lab, plus community GGUFs; see [Model weights](#model-weights) below.

Everything here is a thin layer on top of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) and [charlie12345's ROCmFPX fork](https://github.com/charlie12345/ROCmFPX) — see the Credits at the end.

## Read this first

Full transparency:

- **This is an unofficial, experimental community project.** It is *not* affiliated with, endorsed by, or sponsored by AMD, ggml-org, or any hardware vendor.
- **MIT license, "as is", NO WARRANTY.** Use it only if you accept full responsibility for what it does on your machine.
- **FP4 is software-emulated on RDNA 3.5.** gfx1151 has no FP4 silicon; the ROCmFP4-STRIX_LEAN preset wins on *memory bandwidth* (weights roughly one quarter the size on a bandwidth-starved unified-memory part), not on FP4 compute. Do not expect datacenter-FP4 numbers or semantics.
- **Your mileage WILL vary.** Every number here was measured on one specific machine (ours), one container stack, one set of models — careful point measurements, not statistics. Raw data for every number lives in [`docs/experiments/`](docs/experiments/).
- **Back up your data** before running any container or script from the internet, this repo included.
- **Do not blame the toolchain authors.** The containers come from kyuz0's toolboxes, the fork from charlie12345, the foundation from ggml-org. If something here breaks, the fault is ours, not theirs.

# So, what can I do with this software?

Each line below is one measured claim; nothing is merged or projected.

- Run ROCmFP4/ROCMFPX quants of Qwen3.x-class models on a 128 GB Strix Halo — the dense 27B (13.8 GiB) decodes at ~20 tok/s on free prose and 45–55 tok/s on structured text (Vulkan build, MTP n=6); Qwen3.6-35B-A3B reaches 81.6 tok/s on the same build.
- Route each request to the drafter that fits it, on the 27B server — MTP for prose, DFlash2 for agentic/deterministic work: **+19% agentic** throughput over MTP-only (measured), prose unchanged (−0.5% to +1%). In production since 2026-08-20.
- Run the qwen4exp 180B-class hybrid on the same 128 GB machine — with `--ple-disk`, ~36 GB of RAM stays free (see intro); output is char-identical, warm tg (token generation) costs ~3%.
- Restart the server without re-prefilling every session — `--cache-disk-persist` keeps an antirez-ds4-inspired prompt-cache library on disk across restarts: a 107k-token context restores in 1.57 s — end-to-end 14.3 s vs the measured 920 s cold re-prefill (**64×**) — deterministically. With the MTP drafter the restore needs a token-exact boundary: raw verbatim replays restore, chat-template history replays silently do not. In production since 2026-09-01.
- More than double deterministic decode on qwen4exp with the external MTP drafter (agentionai's, qwen4exp-specific) — 22.1 → 50.2 tok/s (+127%) on a counting task at n=5; 41–44 tok/s at n=6 for code generation on the Vulkan/RADV build after the rollback-restore fix.
- Use vision and speculative decoding together — `--mmproj` plus the MTP drafter in one server; draft acceptance measured on a vision request: 98.3% cumulative.
- Bound the reasoning budget without a cliff — hard thinking cap plus a warn window at 75% of the budget: in a real one-hour agent session (48 requests) every request closed naturally; the single triggered warning converged in 1.3 s; none exhausted the budget.
- Reproduce any published number — replication guides in `docs/guide/`, raw notes and data in [`docs/experiments/`](docs/experiments/).

## The engine

The quants listed under [Model weights](#model-weights) load on ggml-org/llama.cpp builds only where upstream supports
their tensor types — several of the features below exist only in our lab build.
Arriving from a model card: that "our lab build" recommendation is this repo —
build it from [`rocmfpx/`](rocmfpx/) or the [Vulkan
Dockerfile](docker/Dockerfile.vulkan-rocmfpx-local), then try the feature with
the model it was measured on. Full flag recipes (secondary flags, defaults,
commands) live in the [Feature guide](#feature-guide) below.

| Feature | Flag | What it buys | Try it with | Docs |
|---|---|---|---|---|
| PLE disk-offload | `--ple-disk` | ~36 GB RAM back, char-identical output | [Qwen3.8-Flash-Next](https://huggingface.co/pugant/Qwen3.8-Flash-Next-Q4_0_ROCMFP4_STRIX_LEAN-GGUF) | [guide](docs/guide/qwen38-flash-next-ple-disk.md) |
| Persistent prompt cache | `--cache-disk-persist` | restarts stop costing a re-prefill (64× measured) | any model | [guide](docs/guide/qwen38-flash-next-prompt-cache-disk.md) |
| Per-request dual-drafter routing | `--spec-type draft-mtp,draft-dflash` | +19% agentic (measured on the 27B), prose unchanged | [Qwen3.8-27B](https://huggingface.co/pugant/Qwen3.8-27B-MTP-Q4_0_ROCMFP4_STRIX_LEAN) | [replication guide](docs/guide/qwen38-27b.md) |
| External MTP drafter | `-md <mtp.gguf>` | +108/127% deterministic decode | [Qwen3.8-Flash-Next](https://huggingface.co/pugant/Qwen3.8-Flash-Next-Q4_0_ROCMFP4_STRIX_LEAN-GGUF) | [note](docs/experiments/qwen38-flash-next-runtime.md) |
| Matched-quant MTP sidecar | head-only requant (`--output-tensor-type Q4_0_ROCMFP4`) | +4.5% decode, acceptance 0.652→0.667, greedy bit-identical | [MTP draft head](https://huggingface.co/pugant/Qwen3.8-Flash-Next-MTP-DRAFT-HEAD-ROCMFP4-GGUF) | [note](docs/experiments/2026-09-10-vulkan-optimization-wave6.md) |
| Reasoning-budget warn window | `--reasoning-budget` | budgets converge instead of exhausting | any model | [note](docs/experiments/reasoning-budget-warn75.md) |
| qwen4exp arch + vision×MTP | — | the whole Flash-Next family, vision and drafter together | [Qwen3.8-Flash-Next](https://huggingface.co/pugant/Qwen3.8-Flash-Next-Q4_0_ROCMFP4_STRIX_LEAN-GGUF) | [note](docs/experiments/qwen38-flash-next-runtime.md) |
| Graph reuse + dense decode | — | plain tg512 +18.6% HIP / +32.5% Vulkan, identical outputs | [Qwen3.8-Flash-Next](https://huggingface.co/pugant/Qwen3.8-Flash-Next-ROCMFP4_STRIX_LEAN-GGUF) | [note](docs/experiments/2026-09-02-qwen4exp-graph-reuse-and-dense-decode.md) |
| Boundary save guard | — | partial-reject final rounds stop costing a full re-prefill (0 skipped saves over 479 resets) | any model with a drafter | [note](docs/experiments/2026-09-03-f4-boundary-save-fix.md) |

# Motivations

- **One machine, one physics.** Strix Halo puts 128 GB of unified memory behind a ~256 GB/s pipe. Everything that matters here — quant format, drafter choice, KV layout, what moves to disk — follows from that single bandwidth budget.
- **Niche formats get neglected.** ROCmFP4-STRIX_LEAN and hybrid architectures (gated-deltanet, PLE) live in community forks long before — if ever — the big runtimes. Without someone measuring them on real silicon, "FP4 on Strix Halo" stays folklore.
- **Measurements over impressions.** Every claim in this lab is a point measurement with its raw data kept in the repo; a number we cannot reproduce is a number we do not publish.
- **Negative results are results.** What did not work is documented just as carefully as what did, so nobody — ourselves included — has to re-run it.

# AI full disclosure

This lab's measurements, scripts and write-ups are a human + AI collaboration. The AI collaborator did real engineering work — porting, instrumentation, gate design, prose — under human direction and human review.
If that collaborator must be cited by name, it is **GLM by z.ai** — under that name only, never any other.

# Status

**Production-proven for us** (our daily-driver agent server):

- Per-request drafter routing (MTP ⇄ DFlash2) — live since 2026-08-20, shipped only after its validation gates passed.
- qwen4exp conv (convolution-state) and PLE ring-slot rollback fix — partial-reject rollbacks no longer restore the GDN (gated-deltanet)/PLE history from zeroed slots: acceptance 0.74 → 0.91–0.95, decode ~+70% at n=6.
- Reasoning-budget warn window at 75% — deployed and validated on the real agent workload (48/48 requests closed naturally).
- PLE disk-offload (`--ple-disk`) — on a real always-on agent server since 2026-08-31; rolling back is a matter of turning the flag off.
- Persistent prompt cache (`--cache-disk-persist`) — on the same server since 2026-09-01; a restart now restores served context from the on-disk library instead of re-prefilling it.

**Experimental or closed:**

- hipCUB enablement — NO-GO by measurement: pp (prompt processing) at 131k context costs ~42%.
- n-gram drafter — instrumented, NO-GO.
- Dual-drafter same-round cooperation (concat on head token, pattern exclusion, deeper drafts, two-root verify tree) — closed by measurement, not fatigue; the realized synergy is the per-request routing above.

# How to use this project?

## More documentation

- [`docs/experiments/README.md`](docs/experiments/README.md) — narrative index over every experiment note and its raw data.
- [`docs/guide/qwen38-27b.md`](docs/guide/qwen38-27b.md) — canonical end-to-end replication guide: BF16 GGUF → imatrix → ROCmFP4-STRIX_LEAN quant → sanitized GGUF → dual-drafter server.
- [`docs/guide/qwen38-flash-next-ple-disk.md`](docs/guide/qwen38-flash-next-ple-disk.md) — running Qwen3.8-Flash-Next on 128 GB: the PLE disk-offload flags, cache budget and measured cost.
- [`docs/guide/qwen38-flash-next-prompt-cache-disk.md`](docs/guide/qwen38-flash-next-prompt-cache-disk.md) — the persistent prompt cache: flags, the verbatim-replay requirement, monitoring and rollback.
- [`PATCHES.md`](PATCHES.md) — index of every patch series in `patches/`, with upstream status.
- Featured research note: [`docs/experiments/qwen38-flash-next-runtime.md`](docs/experiments/qwen38-flash-next-runtime.md).

## Model weights

No weights live in this repo — our quants are on Hugging Face under their own licenses:

| Model | HF repo | Notes |
|---|---|---|
| Qwen3.8-Flash-Next (qwen4exp) | [`pugant/Qwen3.8-Flash-Next-Q4_0_ROCMFP4_STRIX_LEAN-GGUF`](https://huggingface.co/pugant/Qwen3.8-Flash-Next-Q4_0_ROCMFP4_STRIX_LEAN-GGUF) | 180B-class hybrid (6B active + 51B PLE), 98.5 GiB — the flagship; pairs with the external MTP drafter; `--ple-disk` fits it in 128 GB |
| Qwen3.8-27B (dense) | [`pugant/Qwen3.8-27B-MTP-Q4_0_ROCMFP4_STRIX_LEAN`](https://huggingface.co/pugant/Qwen3.8-27B-MTP-Q4_0_ROCMFP4_STRIX_LEAN) | 13.8 GiB — the lab's canonical model, MTP layer included |
| Qwen3.8-27B imatrix | [`pugant/Qwen3.8-27B-imatrix`](https://huggingface.co/pugant/Qwen3.8-27B-imatrix) | the importance matrix behind the preset above |
| Qwen3.8-27B Q3_0 | [`pugant/Qwen3.8-27B-MTP-Q3_0_ROCMFPX`](https://huggingface.co/pugant/Qwen3.8-27B-MTP-Q3_0_ROCMFPX) | documented **NO-GO** quant — not a true 3-bit on this hybrid arch (4.44/5.72 effective bpw, no byte savings), tg (token generation) costs 15.6% on the base preset, 31.7% on the agent-tuned preset |
| grug-35b-v2 | [`pugant/grug-35b-v2-ROCmFP4-STRIX_LEAN`](https://huggingface.co/pugant/grug-35b-v2-ROCmFP4-STRIX_LEAN) | MoE 35B-A3B, reasoning/tool-call fine-tune |
| Ornith-1.0-35B | [`pugant/Ornith-1.0-35B-ROCmFP4-STRIX_LEAN`](https://huggingface.co/pugant/Ornith-1.0-35B-ROCmFP4-STRIX_LEAN) | MoE 35B-A3B, multimodal |
| Ornith-1.5-35B | [`pugant/Ornith-1.5-35B-ROCmFP4-STRIX_LEAN`](https://huggingface.co/pugant/Ornith-1.5-35B-ROCmFP4-STRIX_LEAN) | MoE 35B-A3B, multimodal — MTP head degraded by the fine-tune (pos-2 acceptance ~0.07): speculative decoding not recommended |
| Nemotron-3.5-Lightning-30B-A3B | [`pugant/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-ROCmFP4-STRIX_LEAN`](https://huggingface.co/pugant/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-ROCmFP4-STRIX_LEAN) | Mamba-hybrid MoE |
| Qwen3.6-35B-A3B | [`pugant/Qwen3.6-35B-A3B-MTP-Q6_0_ROCMFPX`](https://huggingface.co/pugant/Qwen3.6-35B-A3B-MTP-Q6_0_ROCMFPX) | MoE 35B-A3B at Q6_0 |

Most community Strix Halo GGUFs run unmodified; support beyond llama.cpp upstream (niche quant types, hybrid architectures) depends on the fork — the ROCmFP4-STRIX_LEAN preset itself was born in that community.

## Build

The engine ships as source, not binaries: the full buildable fork tree is [`rocmfpx/`](rocmfpx/). Fastest path is the local-source Vulkan Dockerfile (companion files come from [kyuz0's toolboxes](https://github.com/kyuz0/amd-strix-halo-toolboxes), `$TB` = your checkout of them):

```bash
mkdir -p /tmp/vkctx/src
git archive HEAD rocmfpx | tar -x --strip-components=1 -C /tmp/vkctx/src     # the fork source, no clone needed
cp "$TB"/toolboxes/{llama-grammar.patch,gguf-vram-estimator.py} /tmp/vkctx/  # required by the image
cp docker/Dockerfile.vulkan-rocmfpx-local /tmp/vkctx/Dockerfile
docker build -t docker-llm-service:vulkan-fork /tmp/vkctx
```

For ROCm/HIP use [`docker/Dockerfile.rocm-7.2.4-rocmfpx`](docker/Dockerfile.rocm-7.2.4-rocmfpx) (builds from a remote branch). Remember: HIP containers need `/dev/kfd` **and** `/dev/dri` — without `/dev/kfd`, ROCm init fails *silently* and the server falls back to CPU.

## Feature guide

- **Dual-drafter routing** — `--spec-type draft-mtp,draft-dflash` plus `--spec-draft-model <dflash.gguf>`, `--spec-draft-n-max 7`, `--spec-draft-p-min 0.75`; requests carrying `tools` are routed to DFlash2, the rest to MTP, with a per-request `"spec_drafter"` override. Full recipe in the replication guide ([`docs/guide/qwen38-27b.md`](docs/guide/qwen38-27b.md)).
- **qwen4exp + PLE disk-offload** — `--ple-disk` keeps the PLE n-gram table on disk (`--ple-cache-mib N` sets the block-cache budget, default 4096); guide with the measured cost: [`docs/guide/qwen38-flash-next-ple-disk.md`](docs/guide/qwen38-flash-next-ple-disk.md).
- **Persistent prompt cache** — `--cache-disk <dir>` plus `--cache-disk-persist` (`--cache-disk-persist-mib` budget, default 16384) keeps the prompt-cache library across restarts; guide with the token-exact boundary caveat: [`docs/guide/qwen38-flash-next-prompt-cache-disk.md`](docs/guide/qwen38-flash-next-prompt-cache-disk.md).
- **Reasoning-budget warn window** — `--reasoning-budget N` server-wide or `thinking_budget_tokens` per request; at 75% of the budget a mid-conversation message nudges the model to converge, and the forced end fires only if it ignores it.

Minimal smoke test (image from the Build section above, LEAN model from [Model weights](#model-weights)):

```bash
docker run -d --name owl --network host --device /dev/dri --group-add render \
  -v ~/llmodels:/llmodels:ro --entrypoint llama-server docker-llm-service:vulkan-fork \
  -m /llmodels/Qwen3.8-Flash-Next-Q4_0_ROCMFP4_STRIX_LEAN.gguf \
  -ngl 999 --jinja -c 16384 --no-mmap --ple-disk --metrics --port 1234

curl http://localhost:1234/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Who is Strix Lean?"}]}'
```

# Credits 🙏

This lab adds a thin layer on top of giants' work.

- **ggml-org / llama.cpp** — Georgi Gerganov and contributors. The main branch of llama.cpp is the foundation of everything here and is invaluable to the whole local-inference community. Thank you for keeping it open.
- **charlie12345** — maintainer of the ROCmFPX fork of llama.cpp, where the ROCmFP4 / `Q4_0_ROCMFP4_STRIX_LEAN` preset, the HIP kernels and the GGUF extensions live. Most of the code this lab builds upon — and merges upstream (PRs #67–#82, ours among them) — is their work.
- **kyuz0 (Donato Capitella)** — the `amd-strix-halo-toolboxes` / `docker-llm-service` containers this lab builds and runs on.
- **The Strix Halo community** — where the ROCmFP4-STRIX_LEAN preset was born and tuned collectively; its benchmarks, feedback and hardware knowledge shaped every decision in this repo.
- **Jian Chen** — author of the upstream DFlash2 support (llama.cpp PR #27342) that we ported onto the fork.
- **Unsloth** — the UD quantization family and the Flash-Next reference material; our pipeline builds on their published BF16 GGUFs and write-ups.
- **agentionai** — author of the external MTP drafter that pairs with our qwen4exp quant.
- **kingjones777** — reference tg/pp numbers on Strix Halo that we cross-checked our own measurements against.
- **avifenesh** — the original d2t draft-vocabulary trim for the MTP sidecar; our FR-Spec port descends from it.
- **drluoto** — the qwen4exp port of that trim, the reference trimmer and frequency map, and the Strix Halo drafter measurements that motivated the card.
- **quimmedes** — the community Q8_0 MTP head our requantized production sidecar derives from.
- **antirez / Salvatore Sanfilippo** — the ds4 README pattern this page follows, the
  disk-resident model state of his dwarstar (the direct inspiration for our PLE
  disk-offload, `--ple-disk`), and his `ds4_kvstore` — the cross-restart on-disk library
  with hit-decay eviction (6 h half-life) and little-endian sidecar records — which our
  persistent prompt cache (`--cache-disk-persist`) is modeled on.

If we forgot anyone: it is an omission, not intentional — open an issue and we will credit you.

## License & attribution

- Code and scripts in this repo: **MIT** — see [LICENSE](LICENSE).
- Credits and attribution: [NOTICE](NOTICE).
- **No model weights in this repo.** Weights live on Hugging Face under their own licenses; the base-model licenses carry through to the quants.
