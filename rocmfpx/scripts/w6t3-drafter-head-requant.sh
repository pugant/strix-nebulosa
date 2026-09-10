#!/usr/bin/env bash
# W6-3 (A2-C4): head-only requant of the qwen4exp MTP drafter.
#
# Requantizes ONLY the drafter LM head (output.weight, 83% of the per-step
# bytes read by the draft chain) from Q8_0 to Q4_0_ROCMFP4 (4.5 bpw, dual
# UE4M3 half-block scales) while every other tensor is copied bit-exact.
#
# No source patch is needed: llama-quantize already supports this via
#   --pure                    -> no recipe remixing: each tensor keeps its
#                                current type (same type == verbatim copy,
#                                no Q8_0 -> Q8_0 round trip, see
#                                src/llama-quant.cpp "quantize = cur_type != new_type")
#   --output-tensor-type T    -> applies to tensor_category::OUTPUT
#                                (output.weight and nextn.shared_head_head),
#                                honored before the --pure branch
#   --allow-requantize        -> Q8_0 is not in the NVFP4/Q4_0 whitelist of
#                                llama_tensor_allows_requantize_to_rocmfp4()
#   base ftype Q8_0           -> default type == current type for the rest
#
# The fork flag convention applies: all flags BEFORE the positional args.
#
# Usage:
#   scripts/w6t3-drafter-head-requant.sh <in.gguf> <out.gguf> [llama-quantize]
# Env:
#   HEAD_TYPE   (default Q4_0_ROCMFP4) - e.g. Q2_0_ROCMFPX for an FP2 arm

set -euo pipefail

IN="${1:?usage: $0 <in.gguf> <out.gguf> [llama-quantize]}"
OUT="${2:?usage: $0 <in.gguf> <out.gguf> [llama-quantize]}"
QUANTIZE="${3:-llama-quantize}"
HEAD_TYPE="${HEAD_TYPE:-Q4_0_ROCMFP4}"

# pre-flight: dry-run shows the selector perimeter (only output.weight converts)
"$QUANTIZE" --dry-run --allow-requantize --pure --output-tensor-type "$HEAD_TYPE" "$IN" "${OUT}.dryrun" Q8_0

"$QUANTIZE" --allow-requantize --pure --output-tensor-type "$HEAD_TYPE" "$IN" "$OUT" Q8_0

stat -c 'IN : %n %s bytes' "$IN"
stat -c 'OUT: %n %s bytes' "$OUT"
