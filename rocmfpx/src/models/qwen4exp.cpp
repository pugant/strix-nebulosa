// Qwen3.8-Flash-Next (qwen4exp) — quantization-side port of upstream PR
// ggml-org/llama.cpp#27742 (danielhanchen), adapted to this fork's layout:
//   - hyper-connection count lives in hparams.n_hc (not dsv4_hc_mult)
//   - compress ratios in hparams.attn_compress_ratio (not dsv4_compress_ratios)
//   - recurrent layers via hparams.recurrent_layer_arr (no is_recr()/is_ple_impl)
//   - PLE layers via hparams.ple_layer_arr
//   - final HC mixer tensors are OUTPUT_HC_NORM/DOWN/UP (fork tensor naming)
// The inference graph is ported from PR #27742 (hyper-connections, QSA indexer,
// PLE hash lookup), adapted to this fork's field and tensor names.

#include "models.h"
#include "llama-memory-recurrent.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-hybrid-idx.h"
#include "llama-kv-cache.h"

void llama_model_qwen4exp::load_arch_hparams(llama_model_loader & ml) {
    // NextN/MTP: an extra decoder block appended past the trunk; the trunk is the first
    // n_layer - nextn_predict_layers blocks (fork n_layer includes it). Read this first,
    // since the per-layer array guard below needs it.
    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.nextn_predict_layers, false);
    GGML_ASSERT((hparams.nextn_predict_layers == 0 || hparams.nextn_predict_layers == 1) &&
            "qwen4exp supports at most one NextN/MTP block");

    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp, false);
    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);

    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS,    hparams.rope_sections, 4, true);

    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

    // HC; low_rank is qwen4exp-specific, DeepSeek-V4 leaves it absent (full rank)
    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT,    hparams.n_hc);
    ml.get_key(LLM_KV_HYPER_CONNECTION_LOW_RANK, hparams.hc_low_rank);
    GGML_ASSERT(hparams.n_hc      > 0 && "qwen4exp needs a hyper-connection count");
    GGML_ASSERT(hparams.hc_low_rank > 0 && "qwen4exp needs a hyper-connection low rank");
    hparams.n_embd_out_impl = hparams.n_hc * hparams.n_embd;

    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);
    ml.get_key_or_arr(LLM_KV_ATTENTION_COMPRESS_RATIOS, hparams.attn_compress_ratio, hparams.n_layer, false);

    // PLE n-gram hash embeddings; if the key group is absent every field stays zero
    std::fill(hparams.ple_layer_arr.begin(), hparams.ple_layer_arr.end(), false);
    hparams.ple_n_heads = 0;

    uint32_t n_ple = 0;
    ml.get_arr_n(LLM_KV_PLE_LAYERS, n_ple, false);
    if (n_ple > 0) {
        std::vector<uint32_t> ple_layers;
        ml.get_arr(LLM_KV_PLE_LAYERS, ple_layers);
        for (uint32_t il : ple_layers) {
            GGML_ASSERT(il < hparams.n_layer);
            hparams.ple_layer_arr[il] = true;
        }

        ml.get_key(LLM_KV_PLE_NGRAM_SIZE,      hparams.ple_ngram_size);
        ml.get_key(LLM_KV_PLE_HEADS_PER_NGRAM, hparams.ple_heads_per_ngram);
        ml.get_key(LLM_KV_PLE_CONV_KERNEL,     hparams.ple_conv_kernel);
        ml.get_key(LLM_KV_PLE_EOS_TOKEN_ID,    hparams.ple_eos_token_id);
        // optional: files written before this key fall back to the EOS token
        ml.get_key(LLM_KV_PLE_IMAGE_TOKEN_ID,  hparams.ple_image_token_id, false);
        ml.get_key(LLM_KV_EMBEDDING_LENGTH_PER_LAYER, hparams.n_embd_per_layer);

        hparams.ple_n_heads  = (hparams.ple_ngram_size - 1) * hparams.ple_heads_per_ngram;
        hparams.ple_head_dim = hparams.n_embd_per_layer;
        GGML_ASSERT(hparams.ple_ngram_size >= 2 && hparams.ple_ngram_size <= LLAMA_MAX_PLE_NGRAM);
        GGML_ASSERT(hparams.ple_n_heads > 0 && hparams.ple_n_heads <= LLAMA_MAX_PLE_HEADS);

        ml.get_arr(LLM_KV_PLE_LAYER_MULTIPLIERS, hparams.ple_layer_multipliers);
        ml.get_arr(LLM_KV_PLE_HEAD_OFFSETS,      hparams.ple_head_offsets);
        ml.get_arr(LLM_KV_PLE_HEAD_VOCAB_SIZES,  hparams.ple_head_vocab_sizes);
    }

    // linear attention everywhere except every full_attention_interval-th layer;
    // the trailing NextN/MTP block is a full-attention block, so keep it out of the
    // recurrent set (entries past the trunk are written false explicitly, since the
    // array has no initializer)
    {
        uint32_t full_attn_interval = 4;
        ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);
        const uint32_t n_main = hparams.n_layer - hparams.nextn_predict_layers;
        for (uint32_t i = 0; i < hparams.n_layer; ++i) {
            hparams.recurrent_layer_arr[i] = (i < n_main) && ((i + 1) % full_attn_interval != 0);
        }
    }

    // label by the trunk-only count: the draft carries one extra NextN/MTP block
    switch (hparams.n_layer - hparams.nextn_predict_layers) {
        case 48: type = LLM_TYPE_A3B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_qwen4exp::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t hc     = hparams.n_hc;
    const int64_t hc_dim = hc * n_embd;
    const int64_t hc_lr  = hparams.hc_low_rank;

    // draft-only GGUF (nextn block plus shared tables, no trunk blocks): qwen4exp has no
    // attn_norm, its first trunk tensor is hc_attn_norm
    const uint32_t n_main     = n_layer - hparams.nextn_predict_layers;
    const bool     mtp_only   = (hparams.nextn_predict_layers > 0) &&
                                (ml.get_weight("blk.0.hc_attn_norm.weight") == nullptr);
    const int      trunk_flags = mtp_only ? TENSOR_NOT_REQUIRED : 0;
    const int      mtp_flags   = mtp_only ? 0 : TENSOR_SKIP;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

    // there is no output_norm: the final hyper-connection mixer carries it
    output_hc_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_HC_NORM, "weight"), { hc_dim }, 0);
    output_hc_down = create_tensor(tn(LLM_TENSOR_OUTPUT_HC_DOWN, "weight"), { hc_dim, hc_lr }, 0);
    output_hc_up   = create_tensor(tn(LLM_TENSOR_OUTPUT_HC_UP,   "weight"), { hc_lr, hc_dim }, 0);

    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }

    // flat [ple_head_dim, n_rows] gather target; n_rows is padded, so read it back
    if (hparams.ple_n_heads > 0) {
        const std::string ple_name = tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight").str();
        const auto * ple_w = ml.get_weight(ple_name.c_str());
        GGML_ASSERT(ple_w != nullptr && "qwen4exp is missing the PLE n-gram table");
        const int64_t ple_rows = ple_w->tensor->ne[1];
        if (params.ple_disk) {
            // T25: the table stays on disk; the store serves rows on demand.
            // The tensor is deliberately NOT created: the loader skips it
            // (done_getting_tensors is called with partial=true for this arch).
            GGML_ASSERT((ple_w->tensor->type == GGML_TYPE_Q5_1 ||
                         ple_w->tensor->type == GGML_TYPE_Q2_0_ROCMFPX) &&
                        "T25 v2 supports Q5_1 and Q2_0_ROCMFPX PLE tables");
            char err[512];
            const enum ggml_type ple_type = (enum ggml_type) ple_w->tensor->type;
            ple_store_handle = ple_store_open(ml.file_path(ple_w->idx).c_str(), ple_w->offs,
                                              ple_type,
                                              hparams.ple_head_dim, ple_rows,
                                              (uint64_t) params.ple_cache_mib * 1024ull * 1024ull,
                                              err, sizeof(err));
            if (ple_store_handle == nullptr) {
                throw std::runtime_error(err); // loader idiom (loader.cpp:632): clean load failure the server can report
            }
            // T25: the tensor is externalized and will never be loaded, so keep its
            // bytes out of the progress total (same accounting as the unused-tensor
            // skip in llama_model_loader::create_tensor), or size_done never reaches
            // size_data and the final progress_callback(1.0f)/unmap tail is skipped.
            ml.size_data -= ggml_nbytes(ple_w->tensor);
            // bytes per row from the ggml type layout (24 B / 32 el Q5_1, 10 B / 32 el Q2_0_ROCMFPX)
            const double row_bytes = (double) (hparams.ple_head_dim / ggml_blck_size(ple_type) * ggml_type_size(ple_type));
            LLAMA_LOG_INFO("%s: PLE table disk-resident: %.2f GiB externalized (%s, %lld rows), cache %.2f GiB\n",
                    __func__, ple_rows * row_bytes / (1024.0 * 1024.0 * 1024.0), ggml_type_name(ple_type),
                    (long long) ple_rows, params.ple_cache_mib / 1024.0);
        } else {
            per_layer_tok_embd = create_tensor(tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight"),
                                               { hparams.ple_head_dim, ple_rows }, 0);
        }
    }

    for (int il = 0; il < n_layer; ++il) {
        auto & layer = layers[il];

        // the MTP block is structurally a trunk block: it is neither recurrent nor PLE, so it
        // takes the full-attention + MoE path below with no special casing
        const int flags = il < (int) n_main ? trunk_flags : mtp_flags;

        const int64_t n_ff_exp   = hparams.n_ff_exp   ? hparams.n_ff_exp   : n_ff / n_expert_used;
        const int64_t n_ff_shexp = hparams.n_ff_shexp ? hparams.n_ff_shexp : n_ff;

        const int64_t head_k_dim = hparams.ssm_d_state;
        const int64_t head_v_dim = hparams.ssm_d_state;
        const int64_t n_k_heads  = hparams.ssm_n_group;
        const int64_t n_v_heads  = hparams.ssm_dt_rank;
        const int64_t key_dim    = head_k_dim * n_k_heads;
        const int64_t value_dim  = head_v_dim * n_v_heads;
        const int64_t conv_dim   = key_dim * 2 + value_dim;

        // two HC modules per layer: before the token mixer, before the MoE
        layer.hc_attn_norm   = create_tensor(tn(LLM_TENSOR_HC_ATTN_NORM,   "weight", il), { hc_dim }, flags);
        layer.hc_attn_down   = create_tensor(tn(LLM_TENSOR_HC_ATTN_DOWN,   "weight", il), { hc_dim, hc_lr }, flags);
        layer.hc_attn_up     = create_tensor(tn(LLM_TENSOR_HC_ATTN_UP,     "weight", il), { hc_lr, hc_dim }, flags);
        layer.hc_attn_inject = create_tensor(tn(LLM_TENSOR_HC_ATTN_INJECT, "weight", il), { hc_dim, hc }, flags);
        layer.hc_ffn_norm    = create_tensor(tn(LLM_TENSOR_HC_FFN_NORM,    "weight", il), { hc_dim }, flags);
        layer.hc_ffn_down    = create_tensor(tn(LLM_TENSOR_HC_FFN_DOWN,    "weight", il), { hc_dim, hc_lr }, flags);
        layer.hc_ffn_up      = create_tensor(tn(LLM_TENSOR_HC_FFN_UP,      "weight", il), { hc_lr, hc_dim }, flags);
        layer.hc_ffn_inject  = create_tensor(tn(LLM_TENSOR_HC_FFN_INJECT,  "weight", il), { hc_dim, hc }, flags);

        if (!hparams.recurrent_layer_arr[il]) {
            // full attention: wq holds [q|gate] interleaved per head
            create_tensor_qkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, flags);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), { n_embd_head_k * n_head, n_embd }, flags);

            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, flags);
            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, flags);

            const int64_t idx_dim = hparams.indexer_head_size;
            layer.indexer_q_proj = create_tensor(tn(LLM_TENSOR_INDEXER_Q_PROJ, "weight", il), { n_embd, hparams.indexer_n_head * idx_dim }, flags);
            layer.indexer_k_proj = create_tensor(tn(LLM_TENSOR_INDEXER_K_PROJ, "weight", il), { n_embd, idx_dim }, flags);
            layer.indexer_q_norm = create_tensor(tn(LLM_TENSOR_INDEXER_Q_NORM, "weight", il), { idx_dim }, flags);
            layer.indexer_k_norm = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM, "weight", il), { idx_dim }, flags);
        } else {
            layer.wqkv       = create_tensor(tn(LLM_TENSOR_ATTN_QKV,   "weight", il), { n_embd, key_dim * 2 + value_dim }, flags);
            layer.wqkv_gate  = create_tensor(tn(LLM_TENSOR_ATTN_GATE,  "weight", il), { n_embd, value_dim }, flags);
            layer.ssm_conv1d = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", il), { hparams.ssm_d_conv, conv_dim }, flags);
            layer.ssm_dt     = create_tensor(tn(LLM_TENSOR_SSM_DT,     "bias",   il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_a      = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN,         il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_beta   = create_tensor(tn(LLM_TENSOR_SSM_BETA,   "weight", il), { n_embd, n_v_heads }, flags);
            layer.ssm_alpha  = create_tensor(tn(LLM_TENSOR_SSM_ALPHA,  "weight", il), { n_embd, n_v_heads }, flags);
            layer.ssm_norm   = create_tensor(tn(LLM_TENSOR_SSM_NORM,   "weight", il), { head_v_dim }, flags);
            layer.ssm_out    = create_tensor(tn(LLM_TENSOR_SSM_OUT,    "weight", il), { value_dim, n_embd }, flags);
        }

        if (hparams.ple_layer_arr[il]) {
            layer.ple_key        = create_tensor(tn(LLM_TENSOR_PLE_KEY,        "weight", il), { n_embd, hc_dim }, flags);
            layer.ple_value      = create_tensor(tn(LLM_TENSOR_PLE_VALUE,      "weight", il), { n_embd, n_embd }, flags);
            layer.ple_norm_key   = create_tensor(tn(LLM_TENSOR_PLE_NORM_KEY,   "weight", il), { hc_dim }, flags);
            layer.ple_norm_query = create_tensor(tn(LLM_TENSOR_PLE_NORM_QUERY, "weight", il), { hc_dim }, flags);
            layer.ple_norm_conv  = create_tensor(tn(LLM_TENSOR_PLE_NORM_CONV,  "weight", il), { hc_dim }, flags);
            layer.ple_conv1d     = create_tensor(tn(LLM_TENSOR_PLE_CONV1D,     "weight", il), { hparams.ple_conv_kernel, hc_dim }, flags);
        }

        layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", il), { n_embd, n_expert }, flags);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), { n_ff_exp, n_embd, n_expert }, flags);
        create_tensor_gate_up_exps(layer, il, n_embd, n_ff_exp, n_expert, flags);

        layer.ffn_gate_inp_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP_SHEXP, "weight", il), { n_embd }, flags);
        layer.ffn_gate_shexp     = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP,     "weight", il), { n_embd, n_ff_shexp }, flags);
        layer.ffn_up_shexp       = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,       "weight", il), { n_embd, n_ff_shexp }, flags);
        layer.ffn_down_shexp     = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP,     "weight", il), { n_ff_shexp, n_embd }, flags);
    }

    // NextN/MTP head. enorm/hnorm gate the two inputs; eh_proj is the checkpoint's
    // fc_embedding and fc_hidden fused side by side, so one matmul over
    // concat(e, h) computes fc_embedding@e + fc_hidden@h.
    //
    // This port reuses the trunk's output_hc_{norm,down,up} as the head mixer (D1): the
    // draft GGUF ships those, not blk.%d.nextn.hc_head_*, so no nextn.hc_head_* is created.
    if (hparams.nextn_predict_layers > 0) {
        const int il = (int) n_main;
        auto & layer = layers[il];

        layer.nextn.enorm   = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,   "weight", il), { n_embd }, mtp_flags);
        layer.nextn.hnorm   = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,   "weight", il), { hc_dim }, mtp_flags);
        layer.nextn.eh_proj = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ, "weight", il), { 2 * n_embd, n_embd }, mtp_flags);

        // qwen4exp sets mtp_use_dedicated_embeddings=false, so these are absent and the
        // head falls back to the trunk's embedding table and LM head
        layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS,     "weight", il), { n_embd, n_vocab }, mtp_flags | TENSOR_NOT_REQUIRED);
        layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", il), { n_embd, n_vocab }, mtp_flags | TENSOR_NOT_REQUIRED);
    }
}

std::unique_ptr<llm_graph_context> llama_model_qwen4exp::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }
    return std::make_unique<graph>(*this, params);
}

// Hyper-connections keep hc parallel residual streams [n_embd, hc, T] in place of layer norms.
// Returns the mixed [n_embd, T] stream; `inject` gets the [hc, T] scatter weights.
ggml_tensor * llama_model_qwen4exp::graph::build_hc_mix(
        ggml_tensor *  x,
        ggml_tensor *  w_norm,
        ggml_tensor *  w_down,
        ggml_tensor *  w_up,
        ggml_tensor *  w_inject,
        ggml_tensor ** inject,
        int            il) {
    const int64_t hc     = hparams.n_hc;
    const int64_t hc_dim = hc * n_embd;
    const int64_t nt     = x->ne[2];

    // grouped RMSNorm: reduce over one stream, then scale all streams with the [hc_dim] gamma
    // the converter folded each gamma to (1 + w)
    ggml_tensor * xn = ggml_rms_norm(ctx0, x, hparams.f_norm_rms_eps);
    xn = ggml_reshape_2d(ctx0, xn, hc_dim, nt);
    xn = ggml_mul(ctx0, xn, w_norm);
    cb(xn, "hc_norm", il);

    ggml_tensor * lo = build_lora_mm(w_down, xn);
    lo = ggml_silu(ctx0, ggml_scale(ctx0, lo, 1.0f / (float) hc));
    ggml_tensor * gate = ggml_sigmoid(ctx0, build_lora_mm(w_up, lo));
    cb(gate, "hc_gate", il);

    ggml_tensor * gated = ggml_mul(ctx0, xn, gate);
    gated = ggml_reshape_3d(ctx0, gated, n_embd, hc, nt);

    // collapse the streams by their mean
    // [w6-2] the streams are consumed straight as strided views: ADD reads through nb
    // (its src1 was already a strided view here), so the leading cont of stream 0 only
    // materialized data the first ADD would re-read anyway. With hc == 1 the loop body
    // never runs and `mixed` stays a view of `gated` until the scale allocates.
    ggml_tensor * mixed = ggml_view_2d(ctx0, gated, n_embd, nt,
            ggml_row_size(gated->type, n_embd) * hc, 0);
    for (int64_t c = 1; c < hc; ++c) {
        ggml_tensor * s = ggml_view_2d(ctx0, gated, n_embd, nt,
                ggml_row_size(gated->type, n_embd) * hc,
                ggml_row_size(gated->type, n_embd) * c);
        mixed = ggml_add(ctx0, mixed, s);
    }
    mixed = ggml_scale(ctx0, mixed, 1.0f / (float) hc);
    cb(mixed, "hc_mixed", il);

    if (inject) {
        *inject = build_lora_mm(w_inject, xn);
        cb(*inject, "hc_inject", il);
    }

    return mixed;
}

ggml_tensor * llama_model_qwen4exp::graph::build_hc_combine(
        ggml_tensor * residual,
        ggml_tensor * block_out,
        ggml_tensor * inject,
        int           il) {
    const int64_t hc = hparams.n_hc;
    const int64_t nt = residual->ne[2];

    // 2*sigmoid centres the scatter weights on 1, so a zero injection is a plain residual add
    ggml_tensor * w = ggml_sigmoid(ctx0, ggml_scale(ctx0, inject, 1.0f / (float) hc));
    w = ggml_scale(ctx0, w, 2.0f);
    w = ggml_reshape_3d(ctx0, w, 1, hc, nt);

    ggml_tensor * b = ggml_reshape_3d(ctx0, block_out, n_embd, 1, nt);
    b = ggml_repeat_4d(ctx0, b, n_embd, hc, nt, 1);

    ggml_tensor * cur = ggml_add(ctx0, residual, ggml_mul(ctx0, b, w));
    cb(cur, "hc_combine", il);

    return cur;
}

llama_model_qwen4exp::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {
    const int64_t hc = hparams.n_hc;

    GGML_ASSERT(hparams.n_embd_head_v() == hparams.n_embd_head_k());

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    cb(inpL, "model.input_embed", -1);

    auto * inp = build_inp_mem_hybrid();

    // qwen4exp always builds llama_memory_hybrid_idx, so this downcast is safe
    // the indexer cache inside it is absent when the GGUF has no indexer tensors
    const auto * mctx_hyb = static_cast<const llama_memory_hybrid_idx_context *>(inp->mctx);

    const llama_kv_cache_context * mctx_idx = mctx_hyb->get_idx();
    if (mctx_idx) {
        GGML_ASSERT(mctx_idx->get_n_kv() == inp->mctx->get_attn()->get_n_kv() &&
                "the indexer cache must track the attention cache cell for cell");
    }

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // the wide residual starts as hc identical copies of the embedding
    ggml_tensor * res_hc = ggml_repeat_4d(ctx0,
            ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens),
            n_embd, hc, n_tokens, 1);
    cb(res_hc, "hc_init", -1);

    // NextN/MTP layers are loaded as extra blocks but never run in the main pass
    const int n_main = n_layer - (int) hparams.nextn_predict_layers;

    for (int il = 0; il < n_main; ++il) {
        res->t_layer_inp[il] = res_hc;

        if (hparams.ple_layer_arr[il]) {
            res_hc = build_ple(inp->get_recr(), mctx_hyb, res_hc, il);
        }

        ggml_tensor * inject = nullptr;
        ggml_tensor * cur = build_hc_mix(res_hc,
                model.layers[il].hc_attn_norm,
                model.layers[il].hc_attn_down,
                model.layers[il].hc_attn_up,
                model.layers[il].hc_attn_inject,
                &inject, il);

        ggml_build_forward_expand(gf, cur);

        if (hparams.is_recurrent(il)) {
            cur = build_layer_attn_linear(inp->get_recr(), cur, il);
        } else {
            cur = build_layer_attn(inp->get_attn(), mctx_hyb, cur, inp_pos, sections, il);
        }

        // an unmasked MTP export needs a hidden row for every token, so in that case the
        // gather is deferred until after t_h_pre_norm is taken below
        const bool gather_now = !cparams.embeddings_pre_norm || cparams.embeddings_pre_norm_masked;

        if (il == n_main - 1 && inp_out_ids && gather_now) {
            // everything below is per token, so drop the rows that produce no output
            cur    = ggml_get_rows(ctx0, cur,    inp_out_ids);
            inject = ggml_get_rows(ctx0, inject, inp_out_ids);

            res_hc = ggml_reshape_2d(ctx0, res_hc, n_embd*hc, res_hc->ne[2]);
            res_hc = ggml_get_rows(ctx0, res_hc, inp_out_ids);
            res_hc = ggml_reshape_3d(ctx0, res_hc, n_embd, hc, res_hc->ne[1]);
        }

        res_hc = build_hc_combine(res_hc, cur, inject, il);

        cur = build_hc_mix(res_hc,
                model.layers[il].hc_ffn_norm,
                model.layers[il].hc_ffn_down,
                model.layers[il].hc_ffn_up,
                model.layers[il].hc_ffn_inject,
                &inject, il);

        cur = build_layer_ffn(cur, il);
        cb(cur, "ffn_out", il);

        res_hc = build_hc_combine(res_hc, cur, inject, il);

        // "l_last" is the layer output name that build_cvec and imatrix look for
        cb(res_hc, "l_last", il);
    }

    // The MTP head consumes the wide residual, before the head mixer collapses it. Export the
    // combine result itself rather than a reshape of it: a pure view gets no backend assignment
    // from the scheduler, and the readback in llama_context looks one up. It is contiguous, so
    // [n_embd, hc, rows] already has the [n_embd_out, rows] layout the reader expects, and it
    // carries exactly the right rows either way -- gathered above when masked, ungathered when not.
    if (cparams.embeddings_pre_norm) {
        cb(res_hc, "h_pre_norm", -1);
        res->t_h_pre_norm = res_hc;

        // deferred from the last layer: collapse to the output rows now that the export is taken
        if (!cparams.embeddings_pre_norm_masked && inp_out_ids) {
            res_hc = ggml_reshape_2d(ctx0, res_hc, n_embd*hc, res_hc->ne[2]);
            res_hc = ggml_get_rows(ctx0, res_hc, inp_out_ids);
            res_hc = ggml_reshape_3d(ctx0, res_hc, n_embd, hc, res_hc->ne[1]);
        }
    }

    // the final mixer is the output norm: there is no separate one
    ggml_tensor * cur = build_hc_mix(res_hc,
            model.output_hc_norm, model.output_hc_down, model.output_hc_up,
            nullptr, nullptr, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur, model.output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

// LLM_GRAPH_TYPE_DECODER_MTP draft head for qwen4exp.
//
// The head folds the next token's embedding into the trunk's wide hyper-connection residual,
// runs one trunk-style block over it, and collapses the result with its own mixer before
// reusing the trunk's LM head. The wide post-block residual is exported as t_h_pre_norm so the
// speculative driver can feed it straight back in for the next draft step.
//
// v1 simplification: the block attends densely. The trunk's QSA only prunes context past a
// 2048-token budget, so dense is a numerical superset; drafts are verified by the target
// either way. The indexer tensors are still loaded so the GGUF stays complete.
// TODO: wire up QSA here for long-context draft fidelity.
llama_model_qwen4exp::graph_mtp::graph_mtp(const llama_model & model, const llm_graph_params & params) :
    graph(model, params, no_build_t{}) {
    GGML_ASSERT(hparams.nextn_predict_layers > 0 && "QWEN4EXP MTP requires nextn_predict_layers > 0");
    GGML_ASSERT(hparams.nextn_predict_layers == 1 && "QWEN4EXP MTP currently only supports a single MTP block");
    GGML_ASSERT(ubatch.token && "QWEN4EXP MTP requires token input");

    const int64_t hc     = hparams.n_hc;
    const int64_t hc_dim = hc * n_embd;
    GGML_ASSERT(hparams.n_embd_out() == (uint32_t) hc_dim && "QWEN4EXP MTP hidden width mismatch");

    const int il = (int) hparams.n_layer - (int) hparams.nextn_predict_layers;
    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj && "MTP block missing nextn.eh_proj");
    GGML_ASSERT(layer.nextn.enorm   && "MTP block missing nextn.enorm");
    GGML_ASSERT(layer.nextn.hnorm   && "MTP block missing nextn.hnorm");

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    auto inp = std::make_unique<llm_graph_input_embd_h>(hc_dim);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hc_dim, n_tokens);
    ggml_set_input(inp->embd);

    inp->h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hc_dim, n_tokens);
    ggml_set_input(inp->h);
    ggml_set_name(inp->h, "mtp_h_input");

    ggml_tensor * tok_embd_w = layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd;
    ggml_tensor * tok_embd   = ggml_get_rows(ctx0, tok_embd_w, inp->tokens);
    cb(tok_embd, "mtp_tok_embd", il);

    ggml_tensor * h_state = ggml_reshape_3d(ctx0, inp->h, n_embd, hc, n_tokens);
    cb(h_state, "mtp_h_state", il);

    res->add_input(std::move(inp));

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    auto * inp_attn = build_attn_inp_kv();

    // grouped RMSNorm over the wide stream: normalise each hc stream, then scale the flattened
    // [hc_dim] vector with the head's gamma, exactly as build_hc_mix does
    ggml_tensor * h_norm = ggml_rms_norm(ctx0, h_state, hparams.f_norm_rms_eps);
    h_norm = ggml_reshape_2d(ctx0, h_norm, hc_dim, n_tokens);
    h_norm = ggml_mul(ctx0, h_norm, layer.nextn.hnorm);
    h_norm = ggml_reshape_3d(ctx0, h_norm, n_embd, hc, n_tokens);
    cb(h_norm, "mtp_hnorm", il);

    // the token embedding is shared across the streams, so broadcast it to hc copies
    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    e_norm = ggml_repeat_4d(ctx0,
            ggml_reshape_3d(ctx0, e_norm, n_embd, 1, n_tokens),
            n_embd, hc, n_tokens, 1);
    cb(e_norm, "mtp_enorm", il);

    // eh_proj holds fc_embedding and fc_hidden side by side, so this one matmul is
    // fc_embedding @ e_norm + fc_hidden @ h_norm, applied to each stream independently.
    // Keeping the streams distinct here is the point of the hyper-connection residual:
    // pooling them before the projection would throw that away.
    ggml_tensor * concat = ggml_concat(ctx0, e_norm, h_norm, /*dim=*/ 0);
    cb(concat, "mtp_concat", il);

    ggml_tensor * res_hc = build_lora_mm(layer.nextn.eh_proj, concat, layer.nextn.eh_proj_s);
    cb(res_hc, "mtp_eh_proj", il);

    ggml_tensor * inject = nullptr;
    ggml_tensor * cur = build_hc_mix(res_hc,
            layer.hc_attn_norm, layer.hc_attn_down, layer.hc_attn_up, layer.hc_attn_inject,
            &inject, il);
    cb(cur, "mtp_hc_attn_pre", il);

    // ---- dense attention, mirroring the trunk's full-attention branch ----
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * Qcur_full = build_lora_mm(layer.wq, cur, layer.wq_s);
    cb(Qcur_full, "mtp_Qcur_full", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
    Qcur = build_norm(Qcur, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "mtp_Qcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "mtp_gate", il);

    ggml_tensor * Kcur = build_lora_mm(layer.wk, cur, layer.wk_s);
    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "mtp_Kcur_normed", il);

    ggml_tensor * Vcur = build_lora_mm(layer.wv, cur, layer.wv_s);
    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
    cb(Vcur, "mtp_Vcur", il);

    // IMRoPE, same convention and freq_base as the trunk
    Qcur = ggml_rope_multi(ctx0, Qcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    Kcur = ggml_rope_multi(ctx0, Kcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    cb(Qcur, "mtp_Qcur", il);
    cb(Kcur, "mtp_Kcur", il);

    const float kq_scale = hparams.f_attention_scale == 0.0f
            ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    cur = build_attn(inp_attn,
            nullptr, nullptr, nullptr,
            Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    cb(cur, "mtp_attn_pregate", il);

    cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, gate));
    cb(cur, "mtp_attn_gated", il);

    cur = build_lora_mm(layer.wo, cur, layer.wo_s);
    cb(cur, "mtp_attn_out", il);

    if (inp_out_ids) {
        cur    = ggml_get_rows(ctx0, cur,    inp_out_ids);
        inject = ggml_get_rows(ctx0, inject, inp_out_ids);

        res_hc = ggml_reshape_2d(ctx0, res_hc, hc_dim, res_hc->ne[2]);
        res_hc = ggml_get_rows(ctx0, res_hc, inp_out_ids);
        res_hc = ggml_reshape_3d(ctx0, res_hc, n_embd, hc, res_hc->ne[1]);
    }

    res_hc = build_hc_combine(res_hc, cur, inject, il);
    cb(res_hc, "mtp_hc_attn_post", il);

    // ---- MoE, identical to the trunk's build_layer_ffn ----
    cur = build_hc_mix(res_hc,
            layer.hc_ffn_norm, layer.hc_ffn_down, layer.hc_ffn_up, layer.hc_ffn_inject,
            &inject, il);
    cb(cur, "mtp_hc_ffn_pre", il);

    cur = build_layer_ffn(cur, il);
    cb(cur, "mtp_ffn_out", il);

    res_hc = build_hc_combine(res_hc, cur, inject, il);
    cb(res_hc, "mtp_hc_ffn_post", il);

    // The next draft step re-enters here, so export the wide stream before it is collapsed.
    // As in the trunk, export the combine result rather than a reshape view of it.
    cb(res_hc, "h_pre_norm", -1);
    res->t_h_pre_norm = res_hc;

    // the head mixer collapses the streams and doubles as the output norm; this port reuses
    // the trunk's output_hc_* weights (the draft GGUF ships them, not nextn.hc_head_*)
    cur = build_hc_mix(res_hc,
            model.output_hc_norm, model.output_hc_down, model.output_hc_up,
            nullptr, nullptr, -1);
    cb(cur, "mtp_hc_head", -1);

    // deliberately no res->t_embd: it would be n_embd wide while the context sizes its
    // embedding buffer by n_embd_out (the wide stream). The driver reads t_h_pre_norm instead.

    ggml_tensor * head_w = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;
    ggml_tensor * head_s = layer.nextn.shared_head_head ? layer.nextn.shared_head_head_s : model.output_s;
    GGML_ASSERT(head_w && "QWEN4EXP MTP: missing LM head (nextn.shared_head_head or model.output)");

    cur = build_lora_mm(head_w, cur, head_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

std::pair<ggml_tensor *, ggml_tensor *> llama_model_qwen4exp::graph::build_qkvz(
                ggml_tensor * input,
                        int   il) {
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    ggml_tensor * qkv_mixed = build_lora_mm(model.layers[il].wqkv, input, model.layers[il].wqkv_s);
    qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
    cb(qkv_mixed, "linear_attn_qkv_mixed", il);

    ggml_tensor * z = build_lora_mm(model.layers[il].wqkv_gate, input, model.layers[il].wqkv_gate_s);
    cb(z, "z", il);

    return { qkv_mixed, z };
}

ggml_tensor * llama_model_qwen4exp::graph::build_norm_gated(
        ggml_tensor * input,
        ggml_tensor * weights,
        ggml_tensor * gate,
        int           layer) {
    // the one numerical difference from Qwen3.5's GDN: sigmoid output gate, not silu
    ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);
    ggml_tensor * gated = ggml_sigmoid(ctx0, gate);

    return ggml_mul(ctx0, normalized, gated);
}

// QSA attends to a budget of whole blocks of compress_ratio tokens, plus the incomplete tail
// one mean-pooled indexer key scores each block; set_input resolves the cache layout
class llm_graph_input_qsa : public llm_graph_input_i {
public:
    llm_graph_input_qsa(const llama_memory_hybrid_idx_context * mctx, uint32_t ratio, bool blk_bias) :
        mctx(mctx), ratio(ratio), blk_bias(blk_bias) {}
    virtual ~llm_graph_input_qsa() = default;

    void set_input(const llama_ubatch * ubatch) override {
        mctx->get_idx()->set_input_k_idxs(k_idxs, ubatch);
        mctx->set_input_qsa(cell_blk, blk_cells, blk_pos, bias, ubatch, ratio, blk_bias);
    }

    // reuse only needs the shapes to hold: set_input rewrites every tensor from the current
    // cache state each round, and the consumers downstream (the blk_cells gather, the pooled
    // block reshapes, the top_k width) are all fixed by them. n_kv moves in padded steps, so
    // the block count moves with it and the gate catches the step. The ratio is fixed per
    // layer at load, and blk_bias is a pure function of the kq mask shape, causal_attn and
    // use_alibi, all of which other gates on the same graph already cover (the hybrid
    // attention input and allow_reuse): a change there forces a rebuild, which recomputes
    // blk_bias together with the rest.
    bool can_reuse(const llm_graph_params & params) override {
        // dynamic_cast, not the build-time static_cast: this is a gate, and a wrong or null
        // context must fall back to a rebuild, never to a stale graph
        const auto * mctx = dynamic_cast<const llama_memory_hybrid_idx_context *>(params.mctx);

        if (!mctx) {
            return false;
        }

        // refresh first, like every can_reuse: the stored context belongs to a previous batch
        this->mctx = mctx;

        const llama_kv_cache_context * mctx_idx = mctx->get_idx();
        if (!mctx_idx) {
            return false;
        }

        const int64_t n_tokens = params.ubatch.n_tokens;
        const int64_t n_kv     = mctx_idx->get_n_kv();
        const int64_t n_stream = mctx->get_n_stream();
        const int64_t r        = ratio;

        if (r < 1 || n_stream < 1 || n_tokens % n_stream != 0) {
            return false;
        }

        const int64_t n_tps    = n_tokens/n_stream;
        const int64_t n_blocks = (n_kv + r - 1)/r;

        bool res = true;

        res &= k_idxs->ne[0] == n_tokens;

        res &= cell_blk->ne[0] == n_kv;
        res &= cell_blk->ne[1] == n_stream;

        res &= blk_cells->ne[0] == r*n_blocks;
        res &= blk_cells->ne[1] == n_stream;

        res &= blk_pos->ne[0] == 4*n_blocks*n_stream;

        res &= bias->ne[0] == (blk_bias ? n_blocks : n_kv);
        res &= bias->ne[1] == n_tps;
        res &= bias->ne[2] == n_stream;

        return res;
    }

    // per stream: a cell index names a different token in each stream
    ggml_tensor * k_idxs    = nullptr;   // I64 [n_tokens] (llama_kv_cache::build_input_k_idxs)
    ggml_tensor * cell_blk  = nullptr;   // I32 [n_kv, n_stream]
    ggml_tensor * blk_cells = nullptr;   // I32 [ratio*n_blocks, n_stream]
    ggml_tensor * blk_pos   = nullptr;   // I32 [4*n_blocks*n_stream]
    ggml_tensor * bias      = nullptr;   // F32 [n_blocks or n_kv, n_tokens/n_stream, n_stream]

    const llama_memory_hybrid_idx_context * mctx;
    const uint32_t ratio;

    // the per-cell half of the bias is the attention mask, so only the per-block half is uploaded
    const bool blk_bias;
};

// The short-attention shortcut still owes the indexer cache its keys: when n_kv grows past
// the budget the QSA selection must score every cell, including the ones stored while the
// shortcut was active (a rollback can also shrink the cache back into range). This input
// carries that write alone; the rest of the chain is skipped because at this size the
// selection cannot prune anything.
class llm_graph_input_qsa_idx : public llm_graph_input_i {
public:
    llm_graph_input_qsa_idx(const llama_memory_hybrid_idx_context * mctx, uint32_t ratio, uint32_t top_k) :
        mctx(mctx), ratio(ratio), top_k(top_k) {}
    virtual ~llm_graph_input_qsa_idx() = default;

    void set_input(const llama_ubatch * ubatch) override {
        mctx->get_idx()->set_input_k_idxs(k_idxs, ubatch);
    }

    // same discipline as llm_graph_input_qsa: dynamic_cast or rebuild, then re-check every
    // quantity the shortcut depends on (decode shape, one stream, budget covering the
    // cache), so a graph built short is never reused for a round that needs the selection
    bool can_reuse(const llm_graph_params & params) override {
        const auto * mctx = dynamic_cast<const llama_memory_hybrid_idx_context *>(params.mctx);

        if (!mctx) {
            return false;
        }

        this->mctx = mctx;

        const llama_kv_cache_context * mctx_idx = mctx->get_idx();

        if (!mctx_idx) {
            return false;
        }

        if (!params.ubatch.equal_seqs() || params.ubatch.n_seq_tokens != 1 || mctx->get_n_stream() != 1) {
            return false;
        }

        return k_idxs->ne[0] == params.ubatch.n_tokens &&
            (int64_t) mctx_idx->get_n_kv() <= (int64_t) top_k + ratio - 1;
    }

    ggml_tensor * k_idxs = nullptr;   // I64 [n_tokens] (llama_kv_cache::build_input_k_idxs)

    const llama_memory_hybrid_idx_context * mctx;

    const uint32_t ratio;
    const uint32_t top_k;
};

ggml_tensor * llama_model_qwen4exp::graph::build_qsa_top_k(
        const llama_memory_hybrid_idx_context * mctx_hyb,
        ggml_tensor *                           cur,
        ggml_tensor *                           inp_pos,
        ggml_tensor *                           kq_mask,
        int *                                   sections,
        int                                     il) {
    const llama_kv_cache_context * mctx_idx = mctx_hyb->get_idx();

    const int64_t idx_dim  = hparams.indexer_head_size;
    const int64_t n_idx_h  = hparams.indexer_n_head;
    const int64_t r        = hparams.attn_compress_ratio[il];
    const int64_t n_kv     = mctx_idx->get_n_kv();

    GGML_ASSERT(r > 0);

    const int64_t n_blocks = (n_kv + r - 1)/r;

    // build_attn_qsa and the KQ mask need the tokens to divide evenly across the streams
    const int64_t n_stream = mctx_hyb->get_n_stream();
    GGML_ASSERT(n_tokens % n_stream == 0);
    const int64_t n_tps = n_tokens/n_stream;

    // only the "which block is visible" half of the bias varies per block
    // the rest is the visible/not test the attention mask already carries, so upload the per-block half only: 1/ratio of the cells
    // alibi writes distances instead of a mask and non-causal keeps future cells, so both opt out
    // the mask also holds an mrope rule for the query's own position, but only 2d image positions can differ there
    const bool blk_bias = kq_mask != nullptr &&
        kq_mask->ne[0] == n_kv && kq_mask->ne[1] == n_tps && kq_mask->ne[3] == n_stream &&
        cparams.causal_attn && !hparams.use_alibi;

    auto qsa = std::make_unique<llm_graph_input_qsa>(mctx_hyb, (uint32_t) r, blk_bias);

    qsa->k_idxs    = mctx_idx->build_input_k_idxs(ctx0, ubatch);
    qsa->cell_blk  = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_kv, n_stream);
    qsa->blk_cells = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, r*n_blocks, n_stream);
    qsa->blk_pos   = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 4*n_blocks*n_stream);
    qsa->bias      = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, blk_bias ? n_blocks : n_kv, n_tps, n_stream);

    ggml_set_input(qsa->cell_blk);
    ggml_set_input(qsa->blk_cells);
    ggml_set_input(qsa->blk_pos);
    ggml_set_input(qsa->bias);

    llm_graph_input_qsa * inp = qsa.get();
    res->add_input(std::move(qsa));

    // cached indexer keys are raw: pooling precedes norm and rotation, so apply neither
    ggml_tensor * k_raw = build_lora_mm(model.layers[il].indexer_k_proj, cur);
    k_raw = ggml_reshape_3d(ctx0, k_raw, idx_dim, 1, n_tokens);
    cb(k_raw, "indexer_k_raw", il);

    ggml_build_forward_expand(gf, mctx_idx->cpy_k(ctx0, k_raw, inp->k_idxs, il));

    // one key head, so rows are contiguous. get_k gives [idx_dim, n_head_kv, n_kv, n_stream].
    ggml_tensor * k_all = mctx_idx->get_k(ctx0, il);
    k_all = ggml_view_3d(ctx0, k_all, idx_dim, n_kv, n_stream, k_all->nb[2], k_all->nb[3], 0);

    // gathers per stream: blk_cells row s indexes stream s's own cells
    ggml_tensor * members = ggml_get_rows(ctx0, k_all, inp->blk_cells);
    members = ggml_reshape_4d(ctx0, members, idx_dim, r, n_blocks, n_stream);

    // mean over the block members; r is small, so summing slices beats a transpose plus sum_rows
    // [w6-2] the slices feed ADD, which reads through nb, so the per-slice cont only added
    // a dispatch; the last ADD (or the scale when r == 1) still produces the contiguous
    // tensor the reshape below needs
    ggml_tensor * pooled = nullptr;
    for (int64_t i = 0; i < r; ++i) {
        ggml_tensor * slice = ggml_view_3d(ctx0, members, idx_dim, n_blocks, n_stream,
                members->nb[2], members->nb[3], i*members->nb[1]);
        pooled = pooled ? ggml_add(ctx0, pooled, slice) : slice;
    }
    pooled = ggml_scale(ctx0, pooled, 1.0f/(float) r);
    cb(pooled, "indexer_k_pooled", il);

    // rope wants [n_dims, n_head, n_tokens]: lay every stream's blocks flat, split after.
    pooled = ggml_reshape_3d(ctx0, pooled, idx_dim, 1, n_blocks*n_stream);
    pooled = build_norm(pooled, model.layers[il].indexer_k_norm, nullptr, LLM_NORM_RMS, il);
    pooled = ggml_rope_multi(ctx0, pooled, inp->blk_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    pooled = ggml_reshape_3d(ctx0, pooled, idx_dim, n_blocks, n_stream);
    cb(pooled, "indexer_k", il);

    ggml_tensor * q = build_lora_mm(model.layers[il].indexer_q_proj, cur);
    q = ggml_reshape_3d(ctx0, q, idx_dim, n_idx_h, n_tokens);
    q = build_norm(q, model.layers[il].indexer_q_norm, nullptr, LLM_NORM_RMS, il);
    q = ggml_rope_multi(ctx0, q, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    cb(q, "indexer_q", il);

    // rectify each head dot product before the sum, as in the DeepSeek lightning indexer
    // mul_mat matches ne[2], so the queries of stream s only meet the blocks of stream s
    // [w6-2] q comes fresh out of rope_multi and is already contiguous, so the cont was a
    // byte-identical copy ahead of a reshape that keeps it contiguous
    ggml_tensor * score = ggml_mul_mat(ctx0, pooled,
            ggml_reshape_3d(ctx0, q, idx_dim, n_idx_h*n_tps, n_stream));
    score = ggml_reshape_4d(ctx0, score, n_blocks, n_idx_h, n_tps, n_stream);
    score = ggml_relu(ctx0, score);
    score = ggml_cont(ctx0, ggml_permute(ctx0, score, 1, 0, 2, 3));
    score = ggml_sum_rows(ctx0, score);
    score = ggml_reshape_3d(ctx0, score, n_blocks, n_tps, n_stream);
    cb(score, "indexer_score", il);

    // one value per block, so it is cheaper to bias here than after the cells are expanded
    if (blk_bias) {
        score = ggml_add(ctx0, score, inp->bias);
    }

    // every token of a block gets the block score; the budget is whole blocks, so top-k cuts on a block boundary
    ggml_tensor * expanded = ggml_get_rows(ctx0,
            ggml_cont(ctx0, ggml_permute(ctx0, score, 1, 0, 2, 3)), inp->cell_blk);
    expanded = ggml_cont(ctx0, ggml_permute(ctx0, expanded, 1, 0, 2, 3));

    if (blk_bias) {
        // flash attention keeps the mask in f16; the scores are f32
        ggml_tensor * mask = kq_mask->type == GGML_TYPE_F32 ? kq_mask : ggml_cast(ctx0, kq_mask, GGML_TYPE_F32);
        expanded = ggml_add(ctx0, expanded, ggml_reshape_3d(ctx0, mask, n_kv, n_tps, n_stream));
    } else {
        expanded = ggml_add(ctx0, expanded, inp->bias);
    }
    cb(expanded, "indexer_score_tokens", il);

    // the reference returns indexer_top_k + compress_ratio - 1: whole blocks plus the tail
    const int64_t width = std::min<int64_t>(n_kv, (int64_t) hparams.indexer_top_k + r - 1);

    // [w6-2] top_k allocates its result, so the cont was a byte-identical copy before the
    // contiguous reshape below
    ggml_tensor * top_k = ggml_top_k(ctx0, expanded, width);

    // build_attn_qsa reads [n_top_k, n_batch, 1, n_stream], matching the KQ mask.
    top_k = ggml_reshape_4d(ctx0, top_k, width, n_tps, 1, n_stream);
    cb(top_k, "indexer_top_k", il);

    return top_k;
}

// The cache half of build_qsa_top_k: project the raw keys and store them, so the
// short-attention shortcut in build_layer_attn stays transparent to the rounds that
// run the full selection.
void llama_model_qwen4exp::graph::build_qsa_idx_write(
        const llama_memory_hybrid_idx_context * mctx_hyb,
        ggml_tensor *                           cur,
        int                                     il) {
    const llama_kv_cache_context * mctx_idx = mctx_hyb->get_idx();

    const int64_t idx_dim = hparams.indexer_head_size;

    auto inp = std::make_unique<llm_graph_input_qsa_idx>(mctx_hyb,
            hparams.attn_compress_ratio[il], hparams.indexer_top_k);

    inp->k_idxs = mctx_idx->build_input_k_idxs(ctx0, ubatch);

    llm_graph_input_qsa_idx * inp_ptr = inp.get();
    res->add_input(std::move(inp));

    // cached indexer keys are raw: pooling precedes norm and rotation, so apply neither
    ggml_tensor * k_raw = build_lora_mm(model.layers[il].indexer_k_proj, cur);
    k_raw = ggml_reshape_3d(ctx0, k_raw, idx_dim, 1, n_tokens);
    cb(k_raw, "indexer_k_raw", il);

    ggml_build_forward_expand(gf, mctx_idx->cpy_k(ctx0, k_raw, inp_ptr->k_idxs, il));
}

// Dense GQA self-attention restricted to the cells that top_k names.
// The mask build below copies the MLA sparse path in llm_graph_context::build_attn.
ggml_tensor * llama_model_qwen4exp::graph::build_attn_qsa(
        llm_graph_input_attn_kv * inp,
        ggml_tensor *             q_cur,
        ggml_tensor *             k_cur,
        ggml_tensor *             v_cur,
        ggml_tensor *             top_k,
        float                     kq_scale,
        int                       il) {
    // rotate q/k/v before they reach a quantized cache, as the dense path does. the indexer
    // has already scored with its own query in build_qsa_top_k, so top_k is unaffected.
    if (inp->self_k_rot) {
        q_cur = llama_mul_mat_hadamard(ctx0, q_cur, inp->self_k_rot);
        k_cur = llama_mul_mat_hadamard(ctx0, k_cur, inp->self_k_rot);
    }

    if (inp->self_v_rot) {
        v_cur = llama_mul_mat_hadamard(ctx0, v_cur, inp->self_v_rot);
    }

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // expand k later to enable rope fusion which directly writes into k-v cache
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx;

    // store to KV cache
    {
        const auto & k_idxs = inp->get_k_idxs();
        const auto & v_idxs = inp->get_v_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
        ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
    }

    // the F32 base mask: the _cnv one follows the cache dtype and ggml_fill wants F32
    ggml_tensor * kq_mask = inp->self_kq_mask;

    // prepare new kq mask - starts filled with -INFINITY
    ggml_tensor * kq_mask_all = ggml_fill(ctx0, kq_mask, -INFINITY);

    // reshape KQ mask into tensor with rows of size 1:
    // [n_kv, n_batch, 1, n_stream] -> [1, n_kv, n_batch, n_stream]
    kq_mask_all = ggml_view_4d(ctx0, kq_mask_all, 1, kq_mask_all->ne[0], kq_mask_all->ne[1], kq_mask_all->ne[3], kq_mask_all->nb[0], kq_mask_all->nb[1], kq_mask_all->nb[2], 0);

    // reshape top_k indices: [n_top_k, n_batch, 1, n_stream] -> [n_top_k, n_batch, n_stream, 1]
    ggml_tensor * top_k_3d = ggml_view_4d(ctx0, top_k, top_k->ne[0], top_k->ne[1], top_k->ne[3], 1, top_k->nb[1], top_k->nb[2], top_k->ne[3]*top_k->nb[3], 0);

    // prepare zero-filled tensor with rows of size 1: [1, n_top_k, n_batch, n_stream]
    // this will be our source of zero values for unmasking top k mask elements
    ggml_tensor * zeros = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, top_k_3d->ne[0], top_k_3d->ne[1], top_k_3d->ne[2]);
    zeros = ggml_fill(ctx0, zeros, 0.0f);

    // modify KQ mask by unmasking elements that are in top_k indices
    // ggml_set_rows([1, n_kv, n_batch, n_stream], [1, n_top_k, n_batch, n_stream], [n_top_k, n_batch, n_stream, 1])
    ggml_tensor * kq_mask_top_k = ggml_set_rows(ctx0, kq_mask_all, zeros, top_k_3d);

    // reshape to restore the original shape of KQ mask:
    // [1, n_kv, n_batch, n_stream] -> [n_kv, n_batch, 1, n_stream]
    kq_mask_top_k = ggml_view_4d(ctx0, kq_mask_top_k, kq_mask_top_k->ne[1], kq_mask_top_k->ne[2], 1, kq_mask_top_k->ne[3], kq_mask_top_k->nb[2], kq_mask_top_k->nb[3], kq_mask_top_k->nb[3], 0);

    // combine with the original kq mask
    kq_mask_top_k = ggml_add(ctx0, kq_mask_top_k, kq_mask);

    // the fork's flash-attn path takes an F16 mask, like every other input does
    if (cparams.flash_attn) {
        kq_mask_top_k = ggml_cast(ctx0, kq_mask_top_k, GGML_TYPE_F16);
    }

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = mctx_cur->get_v(ctx0, il);

    ggml_tensor * cur = build_attn_mha(q, k, v, nullptr, kq_mask_top_k, nullptr, nullptr, kq_scale, il);
    cb(cur, "kqv_out", il);

    // the rotation is its own inverse, so undo it on the value side of the output
    if (inp->self_v_rot) {
        cur = llama_mul_mat_hadamard(ctx0, cur, inp->self_v_rot);
    }

    return cur;
}

ggml_tensor * llama_model_qwen4exp::graph::build_layer_attn(
        llm_graph_input_attn_kv * inp,
        const llama_memory_hybrid_idx_context * mctx_hyb,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        int *                     sections,
        int                       il) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    // indexer reads the same block input as q/k/v; no cache or no ratio means dense
    const bool qsa = mctx_hyb->get_idx() != nullptr && hparams.attn_compress_ratio[il] > 0;

    // Short-attention shortcut: with n_kv at or below the budget the selection cannot
    // prune anything. build_qsa_top_k asks top_k for width = min(n_kv, indexer_top_k +
    // ratio - 1) cells, and width == n_kv returns the full cell set, so the mask surgery
    // in build_attn_qsa unmasks every row and the final mask is self_kq_mask again: the
    // dense build_attn path below is bit-identical without the indexer chain. Decode
    // only (one token per sequence, single stream): that is where the chain's node count
    // dominates the launch overhead. The indexer cache write survives via
    // build_qsa_idx_write, so no key is missing when the budget is outgrown.
    const int64_t idx_budget = (int64_t) hparams.indexer_top_k + hparams.attn_compress_ratio[il] - 1;

    const bool qsa_short = qsa
        && ubatch.equal_seqs() && ubatch.n_seq_tokens == 1
        && mctx_hyb->get_n_stream() == 1
        && (int64_t) mctx_hyb->get_idx()->get_n_kv() <= idx_budget;

    ggml_tensor * top_k = nullptr;

    if (qsa && !qsa_short) {
        top_k = build_qsa_top_k(mctx_hyb, cur, inp_pos, inp->get_kq_mask(), sections, il);
    } else if (qsa) {
        build_qsa_idx_write(mctx_hyb, cur, il);
    }

    // Qwen3Next uses a single Q projection that outputs query + gate
    ggml_tensor * Qcur_full = build_lora_mm(model.layers[il].wq, cur, model.layers[il].wq_s); // [ (n_embd_head * 2) * n_head, n_tokens ]
    cb(Qcur_full, "Qcur_full", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
    cb(Qcur, "Qcur_reshaped", il);

    Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "Qcur_normed", il);

    ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur, model.layers[il].wk_s);
    cb(Kcur, "Kcur", il);

    ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur, model.layers[il].wv_s);
    cb(Vcur, "Vcur", il);

    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "Kcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "gate_reshaped", il);

    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

    // Apply IMRoPE
    Qcur = ggml_rope_multi(
            ctx0, Qcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    Kcur = ggml_rope_multi(
            ctx0, Kcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    if (top_k) {
        cur = build_attn_qsa(inp, Qcur, Kcur, Vcur, top_k, kq_scale, il);
    } else {
        cur = build_attn(inp,
                    nullptr, nullptr, nullptr,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    }
    cb(cur, "attn_pregate", il);

    ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx0, gate);
    cb(gate_sigmoid, "gate_sigmoid", il);

    cur = ggml_mul(ctx0, cur, gate_sigmoid);
    cb(cur, "attn_gated", il);

    cur = build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s);
    cb(cur, "attn_output", il);

    return cur;
}

ggml_tensor * llama_model_qwen4exp::graph::build_layer_attn_linear(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const int64_t d_inner      = hparams.ssm_d_inner;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t head_k_dim   = hparams.ssm_d_state;
    const int64_t num_k_heads  = hparams.ssm_n_group;
    const int64_t num_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_v_dim   = d_inner / num_v_heads;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    auto qkvz = build_qkvz(cur, il);
    ggml_tensor * qkv_mixed = qkvz.first;
    ggml_tensor * z         = qkvz.second;

    ggml_tensor * beta = build_lora_mm(model.layers[il].ssm_beta, cur, model.layers[il].ssm_beta_s);
    beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
    cb(beta, "beta", il);

    beta = ggml_sigmoid(ctx0, beta);
    cb(beta, "beta_sigmoid", il);

    ggml_tensor * alpha = build_lora_mm(model.layers[il].ssm_alpha, cur, model.layers[il].ssm_alpha_s);
    alpha = ggml_reshape_3d(ctx0, alpha, num_v_heads, n_seq_tokens, n_seqs);
    cb(alpha, "alpha", il);

    ggml_tensor * alpha_biased   = ggml_add(ctx0, alpha, model.layers[il].ssm_dt);
    ggml_tensor * alpha_softplus = ggml_softplus(ctx0, alpha_biased);
    cb(alpha_softplus, "a_softplus", il);

    ggml_tensor * gate = ggml_mul(ctx0, alpha_softplus, model.layers[il].ssm_a);  // -A_log.exp() * softplus
    cb(gate, "gate", il);

    gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, n_seq_tokens, n_seqs);

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    ggml_tensor * conv_kernel      = model.layers[il].ssm_conv1d;
    const int64_t conv_kernel_size = conv_kernel->ne[0];

    // the channels must match how load_arch_tensors sizes wqkv, not ssm_d_inner
    const int64_t conv_channels    = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;

    ggml_tensor * conv_input = build_conv_state_at(inp, conv_states_all, qkv_mixed,
            conv_kernel_size - 1, conv_channels, il);

    ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);
    cb(state, "state_predelta", il);

    ggml_tensor * conv_output_proper = ggml_ssm_conv(ctx0, conv_input, conv_kernel);
    cb(conv_output_proper, "conv_output_raw", il);

    ggml_tensor * conv_output_silu = ggml_silu(ctx0, conv_output_proper);
    cb(conv_output_silu, "conv_output_silu", il);

    ggml_tensor * conv_qkv_mix = conv_output_silu;

    int64_t qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
    int64_t nb1_qkv = ggml_row_size(conv_qkv_mix->type, qkv_dim);

    // Extract the convolved Q, K, V from conv_output
    ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            0);

    ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            head_k_dim * num_k_heads * ggml_element_size(conv_qkv_mix));

    ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_v_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            ggml_row_size(conv_qkv_mix->type, 2 * head_k_dim * num_k_heads));

    cb(q_conv, "q_conv", il);
    cb(k_conv, "k_conv", il);
    cb(v_conv, "v_conv", il);

    const float eps_norm = hparams.f_norm_rms_eps;

    q_conv = ggml_l2_norm(ctx0, q_conv, eps_norm);
    k_conv = ggml_l2_norm(ctx0, k_conv, eps_norm);



    // repeat to match shapes when head keys != value keys; unneeded with the fused GDN
    if (num_k_heads != num_v_heads && (!cparams.fused_gdn_ar || !cparams.fused_gdn_ch)) {
        GGML_ASSERT(num_v_heads % num_k_heads == 0);
        q_conv = ggml_repeat_4d(ctx0, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        k_conv = ggml_repeat_4d(ctx0, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    }

    cb(q_conv, "q_conv_predelta", il);
    cb(k_conv, "k_conv_predelta", il);
    cb(v_conv, "v_conv_predelta", il);

    ggml_tensor * output = build_recurrent_attn(inp, ssm_states_all, q_conv, k_conv, v_conv, gate, beta, state, il);

    ggml_tensor * z_2d = ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    // gated normalization, as self.norm(core_attn_out, z) in the reference
    ggml_tensor * attn_out_norm = build_norm_gated(output, model.layers[il].ssm_norm, z_2d, il);

    ggml_tensor * final_output = ggml_reshape_3d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);
    cb(final_output, "final_output", il);

    cur = build_lora_mm(model.layers[il].ssm_out, final_output, model.layers[il].ssm_out_s);
    cb(cur, "linear_attn_out", il);

    cur = ggml_reshape_2d(ctx0, cur, n_embd, n_seq_tokens * n_seqs);

    return cur;
}

ggml_tensor * llama_model_qwen4exp::graph::build_layer_ffn(ggml_tensor * cur, const int il) {
    GGML_ASSERT(model.layers[il].ffn_gate_inp != nullptr);

    ggml_tensor * moe_out =
        build_moe_ffn(cur,
            model.layers[il].ffn_gate_inp,
            model.layers[il].ffn_up_exps,
            model.layers[il].ffn_gate_exps,
            model.layers[il].ffn_down_exps,
            nullptr,
            n_expert, n_expert_used,
            LLM_FFN_SILU, true,
            hparams.expert_weights_scale,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il,
            nullptr, model.layers[il].ffn_gate_up_exps,
            model.layers[il].ffn_up_exps_s,
            model.layers[il].ffn_gate_exps_s,
            model.layers[il].ffn_down_exps_s);
    cb(moe_out, "ffn_moe_out", il);

    // shared experts, as in the Qwen3Next reference
    if (model.layers[il].ffn_up_shexp != nullptr) {
        ggml_tensor * ffn_shexp =
            build_ffn(cur,
                model.layers[il].ffn_up_shexp, NULL, model.layers[il].ffn_up_shexp_s,
                model.layers[il].ffn_gate_shexp, NULL, model.layers[il].ffn_gate_shexp_s,
                model.layers[il].ffn_down_shexp, NULL, model.layers[il].ffn_down_shexp_s,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "ffn_shexp", il);

        // shared expert has its own sigmoided gate (ffn_gate_inp_shexp, one value per token)
        ggml_tensor * shared_gate = build_lora_mm(model.layers[il].ffn_gate_inp_shexp, cur);
        cb(shared_gate, "shared_expert_gate", il);

        shared_gate = ggml_sigmoid(ctx0, shared_gate);
        cb(shared_gate, "shared_expert_gate_sigmoid", il);


        ffn_shexp = ggml_mul(ctx0, ffn_shexp, shared_gate);
        cb(ffn_shexp, "ffn_shexp_gated", il);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il);
    } else {
        cur = moe_out;
    }

    return cur;
}

// PLE n-gram hash embedding: each token gathers ple_n_heads rows of a shared table.
//   mixed_n = (t[p]*m[0]) ^ ... ^ (t[p-n+1]*m[n-1]);  row = mixed_n % vocab[h] + offset[h]
// The hash runs host-side because ggml has no int64 and no xor. EOS resets the window.

class llm_graph_input_ple : public llm_graph_input_i {
public:
    llm_graph_input_ple(const llama_model_qwen4exp & pmodel,
                        const llama_kv_cache_context * mctx,
                        bool disk)
        : pmodel(pmodel), mctx(mctx), disk(disk) {}
    virtual ~llm_graph_input_ple() = default;

    void set_input(const llama_ubatch * ubatch) override;
    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * rows = nullptr;     // I32 [ple_n_heads * n_tokens]   (mode off)
    ggml_tensor * gathered = nullptr; // F32 [ple_head_dim, ple_n_heads * n_tokens] (mode on)

    const llama_model_qwen4exp & pmodel;

    // the predecessor tokens live in the attention KV cells (ext.tok)
    const llama_kv_cache_context * mctx;

    const bool disk; // T25: fetch rows from ple_store_handle instead of ggml_get_rows

    // scratch, reused across set_input() calls
    std::vector<llama_token> prev;
    std::vector<int32_t> idx;    // T25: host indices (mode on)
    std::vector<float>   scratch; // T25: host gather buffer (mode on)
};

void llm_graph_input_ple::set_input(const llama_ubatch * ubatch) {
    const auto & hp = pmodel.hparams;

    // an image arrives as an embd batch, so ubatch->token is null, but every position still needs a row for ggml_get_rows
    // stand in the image token id that the reference hashes, or EOS if the file has no such key
    // gemma3n and gemma4 do the same with a hardcoded row 0 of per_layer_token_embd.
    const llama_token img_tok = hp.ple_image_token_id != 0
        ? (llama_token) hp.ple_image_token_id
        : (llama_token) hp.ple_eos_token_id;
    auto tok_of = [&](int64_t k) -> llama_token {
        return ubatch->token ? ubatch->token[k] : img_tok;
    };

    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t n_gram   = hp.ple_ngram_size;
    const int64_t n_heads  = hp.ple_n_heads;
    const int64_t per_gram = hp.ple_heads_per_ngram;
    const int64_t eos      = hp.ple_eos_token_id;
    const int64_t n_prev   = n_gram - 1;

    std::vector<int32_t> idx(n_heads * n_tokens);

    GGML_ASSERT(mctx != nullptr);

    for (int64_t i = 0; i < n_tokens; ++i) {
        // the preceding tokens would be ambiguous, see get_prev_tokens()
        GGML_ASSERT(ubatch->n_seq_id[i] == 1 && "PLE n-gram embeddings do not support tokens shared by multiple sequences");
    }

    // predecessors come from the KV cells (ext.tok); apply_ubatch() already stored this ubatch, so its own tokens count too
    mctx->get_prev_tokens(*ubatch, n_prev, prev);

    for (int64_t i = 0; i < n_tokens; ++i) {
        // an EOS in the window resets everything at or before it
        // a missing predecessor (before the sequence start, or no cached cell) reads as EOS
        // the EOS of the token itself does not cut its own context, as in the reference
        std::vector<int64_t> ctx(n_gram);
        ctx[0] = tok_of(i);
        bool cut = false;
        for (int64_t s = 1; s < n_gram; ++s) {
            // predecessor s positions back; prev[] is oldest-first, missing entries are LLAMA_TOKEN_NULL
            const llama_token t = cut ? LLAMA_TOKEN_NULL : prev[i*n_prev + (n_prev - s)];
            cut = cut || t < 0 || t == eos;
            ctx[s] = cut ? eos : t;
        }

        for (int64_t n = 2; n <= n_gram; ++n) {
            uint64_t mixed = (uint64_t) ctx[0] * hp.ple_layer_multipliers[0];
            for (int64_t j = 1; j < n; ++j) {
                mixed ^= (uint64_t) ctx[j] * hp.ple_layer_multipliers[j];
            }
            const int64_t base = (n - 2) * per_gram;
            for (int64_t g = 0; g < per_gram; ++g) {
                const int64_t h_i = base + g;
                idx[i * n_heads + h_i] =
                    (int32_t) (mixed % hp.ple_head_vocab_sizes[h_i] + hp.ple_head_offsets[h_i]);
            }
        }
    }

    if (disk) {
        // T25: serve rows from the disk store, then upload the gathered block
        char err[512];
        scratch.resize((size_t) n_tokens * n_heads * hp.ple_head_dim);
        if (!ple_store_fetch(pmodel.ple_store_handle, idx.data(),
                             (int64_t) idx.size(), scratch.data(),
                             (size_t) hp.ple_head_dim, err, sizeof(err))) {
            GGML_ABORT("%s", err); // fatal WITH the collected message
        }
        ggml_backend_tensor_set(gathered, scratch.data(), 0, scratch.size() * sizeof(float));
    } else {
        ggml_backend_tensor_set(rows, idx.data(), 0, idx.size() * ggml_element_size(rows));
    }
}

// everything the hash reads (the ubatch tokens, the cache predecessors) is re-read by
// set_input on each round, so reuse only needs the row tensor to hold its shape: the head
// count and the row width are model constants, leaving n_tokens as the only variable
bool llm_graph_input_ple::can_reuse(const llm_graph_params & params) {
    // dynamic_cast, not the build-time static_cast: this is a gate, and a wrong or null
    // context must fall back to a rebuild, never to a stale graph
    const auto * mctx_hyb = dynamic_cast<const llama_memory_hybrid_idx_context *>(params.mctx);

    if (!mctx_hyb) {
        return false;
    }

    // refresh first, like every can_reuse: set_input reads the predecessor tokens through
    // the attention cache context of the current batch
    mctx = mctx_hyb->get_attn();

    if (!mctx) {
        return false;
    }

    const int64_t n_rows = (int64_t) pmodel.hparams.ple_n_heads * params.ubatch.n_tokens;

    bool res = true;

    if (disk) {
        // the gathered block is the only graph input: it must carry the rows the
        // ggml_get_rows path would have gathered, at the table's fixed row width
        if (!gathered) {
            return false;
        }

        res &= gathered->ne[0] == pmodel.hparams.ple_head_dim;
        res &= gathered->ne[1] == n_rows;
    } else {
        if (!rows) {
            return false;
        }

        res &= rows->ne[0] == n_rows;
    }

    return res;
}

// Read a conv history out of its own recurrent row and write the new tail back.
// The shared build_conv_state cannot do this: qwen4exp has two such rows per layer.
ggml_tensor * llama_model_qwen4exp::graph::build_conv_state_at(
        llm_graph_input_rs * inp,
        ggml_tensor *        conv_states_all,
        ggml_tensor *        x,
        int64_t              state_cols,
        int64_t              channels,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const auto kv_head = mctx_cur->get_head();

    const int64_t n_seqs    = ubatch.n_seqs;
    const int64_t row_total = conv_states_all->ne[0];

    // the row is exactly this convolution's state, so the gather is reused as a whole
    GGML_ASSERT(state_cols * channels == row_total);

    auto it = rs_rows.find(conv_states_all);
    if (it == rs_rows.end()) {
        it = rs_rows.emplace(conv_states_all, build_rs(inp, conv_states_all, row_total, n_seqs)).first;
    }
    ggml_tensor * rows = it->second;

    ggml_tensor * state = ggml_reshape_3d(ctx0, rows, state_cols, channels, n_seqs);
    cb(state, "conv_state_at", il);

    ggml_tensor * conv_input = ggml_concat(ctx0, state, ggml_transpose(ctx0, x), 0);

    // Write one conv-history snapshot per ring slot: slot s must hold the state
    // after the token s positions before the end of the ubatch, because the
    // rollback gather (build_rs) restores group rs_idx = number of rolled-back
    // rows. Port of the K-slot loop in delta-net-base.cpp build_conv_state.
    // Before this, only group 0 was written, so a partial-reject rollback
    // restored r_l (conv GDN) and p_l (PLE) from never-written ring groups
    // (zeroed buffer): the SSM rewound correctly but the conv/PLE history came
    // back as zeros, leaving the hybrid recurrent state incoherent.
    // [TAG_RECURRENT_ROLLBACK_SPLITS] same assumption as delta-net-base.cpp:
    // the last (n_rs_seq + 1) tokens of a seq must stay in the same ubatch.
    const auto   mem_size = mctx_cur->get_size();
    const size_t row_size = ggml_row_size(conv_states_all->type, row_total);

    const int64_t K = (int64_t) cparams.n_rs_seq + 1;

    for (int64_t t = 1; t <= K; ++t) {
        const int64_t s_idx  = std::max<int64_t>(0, conv_input->ne[0] - state_cols - K + t);
        const int64_t s_slot = K - t;

        ggml_tensor * tail = ggml_view_3d(ctx0, conv_input,
                state_cols, channels, n_seqs,
                conv_input->nb[1], conv_input->nb[2],
                ggml_row_size(conv_input->type, s_idx));

        ggml_tensor * dst = ggml_view_2d(ctx0, conv_states_all,
                state_cols * channels, n_seqs,
                conv_states_all->nb[1],
                (s_slot * mem_size + kv_head) * row_size);

        // [w6-2] the tail view is strided (rows of the wider conv_input), but CPY reads
        // through nb on every backend (delta-net-base.cpp build_conv_state has cpy'd the
        // raw view since the start), so materializing a contiguous copy of it first just
        // added one CONT dispatch per ring slot: K-1 of them carry byte-identical data in
        // decode (n_tokens == 1 leaves a single distinct s_idx). Copy the view directly.
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, tail, dst));
    }

    return conv_input;
}

ggml_tensor * llama_model_qwen4exp::graph::build_ple(
        llm_graph_input_rs * inp,
        const llama_memory_hybrid_idx_context * mctx_hyb,
        ggml_tensor *        hidden,
        int                  il) {
    GGML_UNUSED(inp);

    const int64_t hc      = hparams.n_hc;
    const int64_t hc_dim  = hc * n_embd;
    const int64_t n_heads = hparams.ple_n_heads;

    // the attention cells see every ubatch regardless of the layer types
    // (in the graph, `model` is a const llama_model & — downcast once for the T25 store handle)
    const llama_model_qwen4exp & pmodel = static_cast<const llama_model_qwen4exp &>(model);
    auto ple_inp = std::make_unique<llm_graph_input_ple>(
            pmodel, mctx_hyb->get_attn(),
            /*disk=*/ pmodel.ple_store_handle != nullptr);

    ggml_tensor * emb = nullptr;
    if (ple_inp->disk) {
        // T25: `rows` is NOT created as a graph input (llama-context.cpp FIXME:
        // unused inputs are not allocated and crash); the single input is the
        // gathered F32 tensor, filled host-side by set_input via ple_store_fetch.
        ple_inp->gathered = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32,
                                               hparams.ple_head_dim, n_heads * n_tokens);
        ggml_set_input(ple_inp->gathered);
        emb = ple_inp->gathered;
        res->add_input(std::move(ple_inp));
    } else {
        ple_inp->rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_heads * n_tokens);
        ggml_set_input(ple_inp->rows);
        ggml_tensor * rows = ple_inp->rows;
        res->add_input(std::move(ple_inp));
        // gather then flatten the heads: get_rows lays the head dimension out slowest, as the reference does
        emb = ggml_get_rows(ctx0, model.per_layer_tok_embd, rows);
    }
    emb = ggml_reshape_2d(ctx0, emb, hparams.ple_head_dim * n_heads, n_tokens);
    cb(emb, "ple_embd", il);

    ggml_tensor * key   = build_lora_mm(model.layers[il].ple_key,   emb);
    ggml_tensor * value = build_lora_mm(model.layers[il].ple_value, emb);

    // both norms group over one hc stream, with a weight over the whole hc*n_embd layout
    auto grouped_norm = [&](ggml_tensor * x, ggml_tensor * w) {
        ggml_tensor * t = ggml_reshape_3d(ctx0, x, n_embd, hc, n_tokens);
        t = ggml_rms_norm(ctx0, t, hparams.f_norm_rms_eps);
        t = ggml_reshape_2d(ctx0, t, hc_dim, n_tokens);
        t = ggml_mul(ctx0, t, w);
        return ggml_reshape_3d(ctx0, t, n_embd, hc, n_tokens);
    };

    key = grouped_norm(key, model.layers[il].ple_norm_key);
    ggml_tensor * query = grouped_norm(hidden, model.layers[il].ple_norm_query);

    // per-stream dot product, then a signed square root before the sigmoid
    ggml_tensor * s = ggml_sum_rows(ctx0, ggml_mul(ctx0, key, query));
    s = ggml_scale(ctx0, s, 1.0f / sqrtf((float) n_embd));

    ggml_tensor * mag  = ggml_sqrt(ctx0, ggml_clamp(ctx0, ggml_abs(ctx0, s), 1e-6f, INFINITY));
    ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_mul(ctx0, ggml_sgn(ctx0, s), mag));
    cb(gate, "ple_gate", il);

    // [n_embd, 1, T] value broadcast across the hc streams, scaled by the gate
    ggml_tensor * v3 = ggml_reshape_3d(ctx0, value, n_embd, 1, n_tokens);
    v3 = ggml_repeat_4d(ctx0, v3, n_embd, hc, n_tokens, 1);

    ggml_tensor * gated = ggml_mul(ctx0, v3, gate);
    cb(gated, "ple_gated_value", il);

    ggml_tensor * normalized = grouped_norm(
            ggml_reshape_2d(ctx0, gated, hc_dim, n_tokens),
            model.layers[il].ple_norm_conv);
    normalized = ggml_reshape_2d(ctx0, normalized, hc_dim, n_tokens);

    // depthwise causal conv, dilated by the n-gram size, as a sum of shifted copies
    // ggml_conv_1d_dw is documented as unreliable:
    //   out[c, t] = sum_k w[k, c] * x[c, t - (K-1-k)*dilation]
    // The history of the earlier ubatches is prepended, so a chunked prefill matches a single-shot one.
    const int64_t kern = hparams.ple_conv_kernel;
    const int64_t dil  = hparams.ple_ngram_size;
    const int64_t hist = (kern - 1) * dil;

    // the conv history is per sequence, so the input carries the sequence axis too
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    // [hist + n_seq_tokens, hc_dim, n_seqs], tokens on ne[0]
    ggml_tensor * padded = build_conv_state_at(inp, inp->mctx->get_p_l(il),
            ggml_reshape_3d(ctx0, normalized, hc_dim, n_seq_tokens, n_seqs),
            hist, hc_dim, il);

    ggml_tensor * conv_out = nullptr;
    for (int64_t k = 0; k < kern; ++k) {
        // tap k reads (kern-1-k)*dilation positions back
        const int64_t start = hist - (kern - 1 - k) * dil;

        ggml_tensor * shifted = ggml_cont(ctx0,
                ggml_transpose(ctx0,
                        ggml_view_3d(ctx0, padded, n_seq_tokens, hc_dim, n_seqs,
                                padded->nb[1], padded->nb[2],
                                ggml_row_size(padded->type, start))));

        // column k of the [kern, hc_dim] kernel is one weight per channel
        // [w6-2] tried a contiguous view_1d of row k here: wrong data, the tap needs
        // COLUMN k (a stride-kern gather), so the cont that materializes it stays
        ggml_tensor * wk = ggml_cont(ctx0,
                ggml_view_2d(ctx0, model.layers[il].ple_conv1d, 1, hc_dim,
                        model.layers[il].ple_conv1d->nb[1],
                        k * model.layers[il].ple_conv1d->nb[0]));
        // this kernel keeps the file type, so cast it before it multiplies an f32 activation
        wk = ggml_reshape_1d(ctx0, wk, hc_dim);
        if (wk->type != GGML_TYPE_F32) {
            wk = ggml_cast(ctx0, wk, GGML_TYPE_F32);
        }

        ggml_tensor * term = ggml_mul(ctx0, shifted, wk);
        conv_out = conv_out ? ggml_add(ctx0, conv_out, term) : term;
    }

    conv_out = ggml_silu(ctx0, conv_out);
    // [w6-2] silu allocates a fresh contiguous tensor, so the cont before the reshape was
    // a byte-identical copy
    conv_out = ggml_reshape_3d(ctx0, conv_out, n_embd, hc, n_tokens);
    cb(conv_out, "ple_conv_out", il);

    return ggml_add(ctx0, hidden, ggml_add(ctx0, gated, conv_out));
}
