#!/usr/bin/env bash
# W7-2 (T34 wave-7): router-only requant of the Qwen3.8-Flash-Next trunk.
#
# Requantizes ONLY the 96 MoE router tensors' 2D half (48x
# blk.N.ffn_gate_inp.weight, F32 [2560,512] = 240.5 MiB) to Q4_0_ROCMFP4
# (4.5 bpw, imatrix-guided, "stile LEAN") while every other tensor of the
# trunk is copied bit-exact. ffn_gate_inp_shexp stays F32 (1-D tensor,
# 0.47 MiB total: llama-quantize never quantizes n_dims<2 tensors and the
# byte saving would be 284 KiB = noise).
#
# Two upstream guards make this impossible with stock flags, hence the
# companion patch in src/llama-quant.cpp (W7-2):
#   1) tensor_allows_quantization() excludes ffn_gate_inp.weight from
#      quantization by default -> the patch lifts the guard ONLY when a
#      --tensor-type pattern explicitly names the tensor;
#   2) --tensor-type patterns are ignored in --pure mode -> the patch gives
#      --pure + patterns the exact per-tensor semantics: named tensors are
#      converted, everything else keeps its current type (verbatim copy).
# Without patterns --pure keeps the legacy ftype-default semantics (w6t3
# head requant); without --pure patterns keep recipe semantics.
#
# Base ftype Q4_0_ROCMFP4_STRIX_LEAN (=106) is chosen ONLY to make the writer
# re-emit general.file_type = 106, the value already in the trunk header: under
# --pure + patterns the base ftype is inert for tensor selection. The imatrix
# is passed WITHOUT --include-weights (entries_count KV = 926 = trunk) and the
# container mounts llmodels/models at /data/models (imatrix.file KV string =
# trunk's, T27 convention).
#
# NO disk copy of the trunk is made (the card's copy step is skipped: free
# space 110 GiB < the 130 GiB threshold): llama-quantize only READS the
# read-only mounted original and writes one ~98.3 GiB output.
#
# Usage (from the workspace root or anywhere; paths are absolute):
#   ROCmFPX-w7p2/scripts/w7p2-router-requant.sh [build|dryrun|regress|quant|all]
# Env:
#   ROUTER_TYPE (default Q4_0_ROCMFP4) - target type for the routers
#   SKIP_REGRESS=1 - skip the w6t3 backward-compat dry-run (drafter Q8)

set -euo pipefail

WS=/home/pugant/workspaces/rocmfpx-strix-lean
SRC=$WS/ROCmFPX-w7p2
BUILD=$SRC/build-w7p2-vk
QUANTIZE=$BUILD/bin/llama-quantize          # host path (existence check)
QUANTIZE_C=/ws/ROCmFPX-w7p2/build-w7p2-vk/bin/llama-quantize  # in-container path
IMG=docker-llm-service:w2-vk-builder

TRUNK=/data/models/QWEN3.8/Qwen3.8-Flash-Next-Q4_0_ROCMFP4_STRIX_LEAN.gguf
TRUNK_HOST=/home/pugant/llmodels/models/QWEN3.8/Qwen3.8-Flash-Next-Q4_0_ROCMFP4_STRIX_LEAN.gguf
IMATRIX=/data/models/QWEN3.8/imatrix-flashnext-unsloth.gguf
DRAFTER=/data/models/QWEN3.8/Qwen3.8-Flash-Next-MTP-Q8_0.gguf
# NOTE: llmodels/MODELS is mounted at /data/models (the T27 build convention):
# the writer stamps quantize.imatrix.file verbatim from this path, so the mount
# choice is what keeps that KV byte-identical to the trunk header.

ROUTER_TYPE=${ROUTER_TYPE:-Q4_0_ROCMFP4}
OUT_NAME=Qwen3.8-Flash-Next-Q4_0_ROCMFP4_STRIX_LEAN-ROUTER-${ROUTER_TYPE}.gguf
OUT=/ws/staging-w7/$OUT_NAME          # in-container path (docker args)
OUT_HOST=$WS/staging-w7/$OUT_NAME     # host path (stat / sha256sum)

# the pattern names exactly the 48 2D routers; shexp does not match
# ("ffn_gate_inp.weight" is not a substring of "ffn_gate_inp_shexp.weight")
TT_PATTERN='blk\.[0-9]+\.ffn_gate_inp\.weight'

STAGE=${1:-all}

log_dir=$WS/logs/wave7
mkdir -p "$log_dir" "$WS/staging-w7"

# ---------------------------------------------------------------- build ---
do_build () {
    mkdir -p "$BUILD"
    docker run --rm -v "$WS:/ws" -w /ws --user "$(id -u):$(id -g)" -e HOME=/tmp \
      $IMG bash -c "
        set -e
        cmake -S /ws/ROCmFPX-w7p2 -B /ws/ROCmFPX-w7p2/build-w7p2-vk \
          -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=OFF \
          -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
          -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=ON \
          -DBUILD_SHARED_LIBS=OFF \
          > /ws/logs/wave7/w7p2-build.log 2>&1
        cmake --build /ws/ROCmFPX-w7p2/build-w7p2-vk --target llama-quantize -j \$(nproc) \
          >> /ws/logs/wave7/w7p2-build.log 2>&1
        echo rc=\$? >> /ws/logs/wave7/w7p2-build.log
      "
    test -x "$QUANTIZE"
}

docker_quant () {  # $1=logfile, rest = llama-quantize argv (in-container paths)
    local log=$1; shift
    docker run --rm -v "$WS:/ws" -w /ws --user "$(id -u):$(id -g)" -e HOME=/tmp \
      -v /home/pugant/llmodels/models:/data/models:ro \
      $IMG bash -c "
        set -e
        $QUANTIZE_C \"\$@\" 2>&1
        echo rc=\$?
      " bash "$@" > "$log"
    grep -q 'rc=0$' "$log" || { echo "FAIL: see $log"; return 1; }
}

# --------------------------------------------------------------- dryrun ---
# Gate: the selector perimeter. Exactly the 48 blk.N.ffn_gate_inp.weight
# tensors may convert; anything else converting (or a router without its
# imatrix entry) is an abort.
do_dryrun () {
    local log=$log_dir/w7p2-dryrun.log
    docker_quant "$log" --dry-run --allow-requantize --pure \
      --tensor-type "${TT_PATTERN}=${ROUTER_TYPE}" \
      --imatrix "$IMATRIX" \
      "$TRUNK" "/tmp/ignored.gguf" Q4_0_ROCMFP4_STRIX_LEAN

    local n_conv n_gate
    n_conv=$(grep -c ' -> ' "$log" || true)
    n_gate=$(grep ' -> ' "$log" | grep -c 'ffn_gate_inp\.weight' || true)
    echo "converting tensors: $n_conv (ffn_gate_inp: $n_gate)"
    # no --include-weights: the full 926-entry imatrix is kept so that the
    # writer stamps quantize.imatrix.entries_count = 926 = trunk value
    local n_imx; n_imx=$(grep -o 'have [0-9]* importance matrix entries' "$log" | grep -o '[0-9]*' || true)
    echo "imatrix entries kept: ${n_imx:-0}"
    [ "$n_conv" = "48" ] && [ "$n_gate" = "48" ] && [ "$n_imx" = "926" ] \
      || { echo "GATE FAIL: perimeter is not exactly the 48 routers"; return 1; }
    echo "GATE perimeter: PASS"
}

# --------------------------------------------------------------- regress ---
# Backward compatibility of the patched --pure: the w6t3 head-requant command
# (no --tensor-type patterns) must keep converting exactly output.weight.
do_regress () {
    [ "${SKIP_REGRESS:-0}" = "1" ] && { echo "regress: skipped"; return 0; }
    local log=$log_dir/w7p2-regress.log
    docker_quant "$log" --dry-run --allow-requantize --pure \
      --output-tensor-type Q4_0_ROCMFP4 \
      "$DRAFTER" "/tmp/ignored.gguf" Q8_0

    local n_conv n_head
    n_conv=$(grep -c ' -> ' "$log" || true)
    n_head=$(grep ' -> ' "$log" | grep -c 'output\.weight' || true)
    echo "regress converting tensors: $n_conv (output.weight: $n_head)"
    [ "$n_conv" = "1" ] && [ "$n_head" = "1" ] \
      || { echo "REGRESS FAIL: --pure legacy behavior changed"; return 1; }
    echo "REGRESS w6t3 pure: PASS"
}

# ---------------------------------------------------------------- quant ---
do_quant () {
    # disk gate: the artifact (~98.3 GiB) + 5 GiB headroom must be free
    local avail_kb need_kb
    avail_kb=$(df --output=avail -k "$WS" | tail -1)
    need_kb=$((105537262976 / 1024 + 5 * 1024 * 1024))
    if [ "$avail_kb" -lt "$need_kb" ]; then
        echo "BLOCKED: free ${avail_kb}KiB < needed ${need_kb}KiB (artifact + 5 GiB headroom)"; return 1
    fi

    local log=$log_dir/w7p2-quant.log
    rm -f "$OUT_HOST"
    docker_quant "$log" --allow-requantize --pure \
      --tensor-type "${TT_PATTERN}=${ROUTER_TYPE}" \
      --imatrix "$IMATRIX" \
      "$TRUNK" "$OUT" Q4_0_ROCMFP4_STRIX_LEAN

    stat -c 'OUT: %n %s bytes' "$OUT_HOST"
    sha256sum "$OUT_HOST" | tee "$WS/staging-w7/${OUT_NAME}.sha256"
}

case $STAGE in
    build)   do_build ;;
    dryrun)  do_dryrun ;;
    regress) do_regress ;;
    quant)   do_quant ;;
    all)
        do_build
        do_dryrun
        do_regress
        do_quant
        ;;
    *) echo "usage: $0 [build|dryrun|regress|quant|all]"; exit 2 ;;
esac
