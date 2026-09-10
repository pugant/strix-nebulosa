#include "speculative.h"

#include "common.h"
#include "ggml.h"
#include "llama.h"
#include "../src/llama-ext.h" // staging API: llama_set_embeddings_pre_norm / llama_get_embeddings_pre_norm_ith (used by MTP)
#include "log.h"
#include "ngram-cache.h"
#include "ngram-map.h"
#include "ngram-mod.h"
#include "sampling.h"
#include "spec-concat-exclusion.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <map>
#include <numeric> // std::iota
#include <random>
#include <stdexcept>

#define SPC_DBG(fmt, ...) LOG_DBG("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_TRC(fmt, ...) LOG_TRC("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_INF(fmt, ...) LOG_INF("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_WRN(fmt, ...) LOG_WRN("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_ERR(fmt, ...) LOG_ERR("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_CNT(fmt, ...) LOG_CNT(""              fmt,               __VA_ARGS__)

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

const std::map<std::string, common_speculative_type> common_speculative_type_from_name_map = {
    {"none",          COMMON_SPECULATIVE_TYPE_NONE},
    {"draft-simple",  COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE},
    {"draft-eagle3",  COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3},
    {"draft-mtp",     COMMON_SPECULATIVE_TYPE_DRAFT_MTP},
    {"draft-dflash",  COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH},
    {"draft-dspark",  COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK},
    {"ngram-simple",  COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE},
    {"ngram-map-k",   COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K},
    {"ngram-map-k4v", COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V},
    {"ngram-mod",     COMMON_SPECULATIVE_TYPE_NGRAM_MOD},
    {"ngram-cache",   COMMON_SPECULATIVE_TYPE_NGRAM_CACHE}
};

static int32_t common_speculative_effective_n_max(
        const common_params_speculative_draft & params,
        const common_speculative_draft_params & dp) {
    int32_t n_max = std::max(0, params.n_max);
    if (dp.n_max >= 0) {
        n_max = std::min(n_max, dp.n_max);
    }
    return n_max;
}

static int32_t common_speculative_effective_n_min(
        const common_params_speculative_draft & params,
        const common_speculative_draft_params & dp,
        const int32_t n_max) {
    const int32_t n_min = dp.n_min >= 0 ? dp.n_min : params.n_min;
    return std::max(0, std::min(n_min, n_max));
}

static float common_speculative_effective_p_min(
        const common_params_speculative_draft & params,
        const common_speculative_draft_params & dp) {
    const float p_min = dp.p_min >= 0.0f ? dp.p_min : params.p_min;
    return std::max(0.0f, std::min(p_min, 1.0f));
}

struct common_speculative_config {
    common_speculative_type type;
    common_params_speculative params;

    common_speculative_config(common_speculative_type t,
            const common_params_speculative & p = common_params_speculative{}) : type(t), params(p) {}
};

static bool common_speculative_are_compatible(
    const llama_model * model_tgt,
    const llama_model * model_dft) {
    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);

    const auto vocab_type_tgt = llama_vocab_type(vocab_tgt);
    LOG_DBG("%s: vocab_type tgt: %d\n", __func__, vocab_type_tgt);

    const auto vocab_type_dft = llama_vocab_type(vocab_dft);
    LOG_DBG("%s: vocab_type dft: %d\n", __func__, vocab_type_dft);

    if (vocab_type_tgt != vocab_type_dft) {
        LOG_WRN("%s: draft model vocab type must match target model to use speculation but "
                "vocab_type_dft = %d while vocab_type_tgt = %d\n", __func__, vocab_type_dft, vocab_type_tgt);
        return false;
    }

    if (llama_vocab_get_add_bos(vocab_tgt) != llama_vocab_get_add_bos(vocab_dft) ||
        (llama_vocab_get_add_bos(vocab_tgt) && llama_vocab_bos(vocab_tgt) != llama_vocab_bos(vocab_dft))) {
        LOG_WRN("%s: draft model bos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
                __func__,
                llama_vocab_get_add_bos(vocab_tgt), llama_vocab_get_add_bos(vocab_dft),
                llama_vocab_bos(vocab_tgt), llama_vocab_bos(vocab_dft));
        return false;
    }

    if (llama_vocab_get_add_eos(vocab_tgt) != llama_vocab_get_add_eos(vocab_dft) ||
        (llama_vocab_get_add_eos(vocab_tgt) && llama_vocab_eos(vocab_tgt) != llama_vocab_eos(vocab_dft))) {
        LOG_WRN("%s: draft model eos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
                __func__,
                llama_vocab_get_add_eos(vocab_tgt), llama_vocab_get_add_eos(vocab_dft),
                llama_vocab_eos(vocab_tgt), llama_vocab_eos(vocab_dft));
        return false;
    }

    {
        const int n_vocab_tgt = llama_vocab_n_tokens(vocab_tgt);
        const int n_vocab_dft = llama_vocab_n_tokens(vocab_dft);
        const int vocab_diff  = n_vocab_tgt > n_vocab_dft
            ? n_vocab_tgt - n_vocab_dft
            : n_vocab_dft - n_vocab_tgt;

        if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
            LOG_DBG("%s: draft model vocab must closely match target model to use speculation but ", __func__);
            LOG_DBG("target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n",
                    n_vocab_tgt, llama_vocab_n_tokens(vocab_dft), vocab_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
            return false;
        }

        for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
            const char * token_text_tgt = llama_vocab_get_text(vocab_tgt, i);
            const char * token_text_dft = llama_vocab_get_text(vocab_dft, i);

            if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
                LOG_DBG("%s: draft model vocab must match target model to use speculation but ", __func__);
                LOG_DBG("token %d content differs - target '%s', draft '%s'\n", i,
                        common_token_to_piece(vocab_tgt, i).c_str(),
                        common_token_to_piece(vocab_dft, i).c_str());
                return false;
            }
        }
    }

    return true;
}

using common_speculative_draft_params_vec = std::vector<common_speculative_draft_params>;

// state of an implementation of speculative decoding
//
// each implementation has a unique type and a state that is implementation-specific
// in a subclass of common_speculative_impl
struct common_speculative_impl {
    const common_speculative_type type;

    uint32_t n_seq;

    // per-seq drafter routing (nullptr = no routing active): points at the owner's
    // per-seq draft params, indexed by seq_id, and is set by common_speculative_begin()
    // / common_speculative_process() right before the call - not set for draft(),
    // which is routed via the `drafting` stash in common_speculative_draft(). The
    // vector is never resized after the ctor and the use sites index it with
    // seq_id < n_seq as guaranteed by their asserts. Implementations that
    // mirror the target batch into their own draft context (DFlash/DSpark) use it to
    // skip the sequences drafted by another implementation; the others (e.g. MTP)
    // must ignore it and always observe every sequence.
    const common_speculative_draft_params_vec * dparams_routing = nullptr;

    size_t n_call_begin  = 0; // number of times this implementation was called for refresh.
    size_t n_call_draft  = 0; // number of times this implementation was called for generation.
    size_t n_call_accept = 0; // number of times this implementation was called for accumulation.

    size_t n_gen_drafts = 0; // number of times a draft or part was generated by this implementation.
    size_t n_acc_drafts = 0; // number of times a draft or part was accepted by the target model.
    size_t n_gen_tokens = 0; // number of tokens generated by this implementation.
    size_t n_acc_tokens = 0; // number of tokens accepted by the target model.

    std::vector<size_t> n_acc_tokens_per_pos; // number of tokens accepted per draft position.

    // TODO: track performance of most recent calls
    const bool gen_perf = true; // whether to generate performance stats.

    int64_t t_begin_us  = 0; // total time spent in refresh of this implementation in microseconds.
    int64_t t_draft_us  = 0; // total time spent in generating drafts in this implementation in microseconds.
    int64_t t_accept_us = 0; // total time spent in accumulation of this implementation in microseconds.

    common_speculative_impl(common_speculative_type type, uint32_t n_seq) : type(type), n_seq(n_seq) {}

    virtual ~common_speculative_impl() = default;

    virtual void begin(llama_seq_id seq_id, const llama_tokens & prompt) = 0;

    virtual bool process(const llama_batch & batch) = 0;

    virtual void draft(common_speculative_draft_params_vec & dparams) = 0;

    virtual void accept(llama_seq_id seq_id, uint16_t n_accepted) = 0;

    // (optional) serialize/restore per-seq internal state (e.g. eagle3's deferred boundary).
    virtual bool get_state(llama_seq_id /*seq_id*/, std::vector<uint8_t> & /*data*/) const { return false; }
    virtual bool set_state(llama_seq_id /*seq_id*/, const std::vector<uint8_t> & /*data*/) { return true; }
    virtual bool state_required() const { return false; }
    virtual void shift_state(llama_seq_id /*seq_id*/, llama_pos /*delta*/) {}

    // (optional) partial-reject draft-sync reset: clear the round bookkeeping the
    // next draft() would misuse (ghost rows) while keeping any cache-facing
    // boundary valid for get_state(). Default = full set_state({}) — i.e. exactly
    // the pre-split behavior; only eagle3 overrides (its boundary must survive).
    virtual void draft_sync_reset(llama_seq_id seq_id) {
        set_state(seq_id, {});
    }

    // (optional) rewind the per-seq state to a previously seen position after a
    // bounded memory rollback; see common_speculative_rollback_state
    virtual bool rollback_state(llama_seq_id /*seq_id*/, llama_pos /*pos*/) { return false; }

    // true if this implementation requires the target context to extract post-norm embeddings
    virtual bool need_embd() const = 0;

    // true if this implementation requires the target context to extract pre-norm embeddings
    virtual bool need_embd_pre_norm() const { return false; }

    // spec-route: effective (post-clamp) max draft size this implementation will
    // use (its own clamped params copy); -1 = no draft-size notion
    virtual int32_t draft_n_max() const { return -1; }
};

static void common_speculative_batch_add_one_seq(
        llama_batch & batch, llama_token id, llama_pos pos, llama_seq_id seq_id, bool logits) {
    GGML_ASSERT(batch.seq_id[batch.n_tokens] && "llama_batch size exceeded");

    batch.token   [batch.n_tokens] = id;
    batch.pos     [batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = 1;
    batch.seq_id  [batch.n_tokens][0] = seq_id;
    batch.logits  [batch.n_tokens] = logits ? 1 : 0;

    batch.n_tokens++;
}

struct common_speculative_impl_draft_simple : public common_speculative_impl {
    common_params_speculative_draft params;

    llama_batch batch;

    std::vector<common_sampler_ptr> smpls;
    std::vector<uint8_t> drafting;

    common_speculative_impl_draft_simple(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE, n_seq)
        , params(params.draft)
    {
        auto * ctx_dft = this->params.ctx_dft;
        auto * ctx_tgt = this->params.ctx_tgt;

        batch = llama_batch_init(llama_n_batch(ctx_dft), 0, 1);

        // TODO: optimize or pass from outside?
        // {
        //     common_params_sampling params;
        //     params.no_perf = false;
        //
        //     params.top_k = 40;
        //     params.top_p = 0.9;
        //
        //     params.samplers = {
        //         COMMON_SAMPLER_TYPE_TOP_K,
        //         COMMON_SAMPLER_TYPE_TOP_P,
        //         COMMON_SAMPLER_TYPE_INFILL,
        //     };
        //
        //     result->smpl = common_sampler_init(llama_get_model(ctx_dft), params);
        // }

        smpls.resize(n_seq);
        for (auto & smpl : smpls) {
            common_params_sampling params;
            params.no_perf = false;
            params.top_k = 10;
            params.samplers = {
                COMMON_SAMPLER_TYPE_TOP_K,
            };

            smpl.reset(common_sampler_init(llama_get_model(ctx_dft), params));
        }

        const bool vocab_cmpt = common_speculative_are_compatible(llama_get_model(ctx_tgt), llama_get_model(ctx_dft));
        LOG_DBG("%s: vocab_cmpt = %d\n", __func__, vocab_cmpt);

        if (!vocab_cmpt) {
            LOG_ERR("%s: the target and draft vocabs are not compatible\n", __func__);

            throw std::runtime_error("draft model vocab type must match target model to use speculation");
        }

        if (n_seq != llama_n_seq_max(ctx_dft)) {
            LOG_ERR("%s: n_seq mismatch: %d != %d\n", __func__, n_seq, llama_n_seq_max(ctx_dft));

            throw std::runtime_error("the draft model number of sequences is incompatible with the speculative n_seq");
        }

        drafting.assign(n_seq, 0);
    }

    ~common_speculative_impl_draft_simple() override {
        llama_batch_free(batch);
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    bool process(const llama_batch & batch) override {
        auto * ctx_dft = params.ctx_dft;

        llama_batch batch_dft = batch;
        batch_dft.logits = nullptr;

        const int ret = llama_decode(ctx_dft, batch_dft);

        if (ret != 0) {
            LOG_ERR("%s: failed to decode draft batch, ret = %d\n", __func__, ret);

            return false;
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        common_batch_clear(batch);

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::fill(drafting.begin(), drafting.end(), 0);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }

            n_drafting++;
            drafting[seq_id] = 1;
            common_sampler_reset(smpls[seq_id].get());

            common_speculative_batch_add_one_seq(batch, dp.id_last, dp.n_past, seq_id, true);
        }

        if (n_drafting == 0) {
            return;
        }

        int ret = llama_decode(ctx_dft, batch);
        if (ret != 0) {
            LOG_WRN("%s: llama_decode returned %d\n", __func__, ret);
            return;
        }

        int i = 0;
        const bool log_debug = LOG_LEVEL_DEBUG <= common_log_get_verbosity_thold();

        while (n_drafting > 0) {
            int i_batch = 0;

            common_batch_clear(batch);

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (!drafting[seq_id]) {
                    continue;
                }

                auto & dp = dparams[seq_id];
                auto & result = *dp.result;
                const int32_t n_max = common_speculative_effective_n_max(params, dp);
                if (n_max <= (int) result.size()) {
                    drafting[seq_id] = 0;
                    n_drafting--;
                    continue;
                }

                auto * smpl = smpls[seq_id].get();

                common_sampler_sample(smpl, ctx_dft, i_batch, true);
                ++i_batch;

                const auto * cur_p = common_sampler_get_candidates(smpl, true);

                if (log_debug) {
                    for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                        LOG_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                                seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                                common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                    }
                }

                // add drafted token for each sequence
                const llama_token id = cur_p->data[0].id;

                // only collect very high-confidence draft tokens
                if (cur_p->data[0].p < common_speculative_effective_p_min(params, dp)) {
                    drafting[seq_id] = 0;
                    n_drafting--;

                    continue;
                }

                common_sampler_accept(smpl, id, true);

                result.push_back(id);

                if (n_max <= (int) result.size()) {
                    drafting[seq_id] = 0;
                    n_drafting--;
                    continue;
                }

                common_speculative_batch_add_one_seq(batch, id, dp.n_past + i + 1, seq_id, true);
            }

            if (batch.n_tokens == 0) {
                break;
            }

            // evaluate the drafted tokens on the draft model
            ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                LOG_WRN("%s: llama_decode[%d] returned %d\n", __func__, i, ret);
                break;
            }

            ++i;
        }

        for (auto & dp : dparams) {
            if (!dp.drafting) {
                continue;
            }

            const int32_t n_max = common_speculative_effective_n_max(params, dp);
            const int32_t n_min = common_speculative_effective_n_min(params, dp, n_max);
            if (dp.result->size() < (size_t) n_min) {
                dp.result->clear();
            }
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/) override {
        // noop
    }

    bool need_embd() const override {
        return false;
    }

    int32_t draft_n_max() const override {
        return std::max(0, params.n_max);
    }
};


// EAGLE3 speculative decoding state
//
// Input of draft decoder: (This is different compared to MTP)
//   At "pos P", the decoder takes input pair (t_{P+1}, g_P), with RoPE at P.
//     - t_{P+1} = token at sequence pos P+1 (the *next* token after P)
//     - g_P     = encoder output = projection of target's extracted hidden states at P
//
// Deferred boundary (MTP doesn't have this issue):
//   Within a single process() call with n_tokens, we can only write decoder KV for
//   training pos 0..n_tokens-2. The last training pos (n_tokens-1) needs t_{n_tokens}
//   which lies *outside* this batch — it is the token target will sample next or the first token from next ubatch.
//   So the last training pos of each process() call is *deferred* to whichever next call has
//   the missing token in hand:
//     - multi-ubatch prefill: the next process()'s first token completes the pair
//                              (handled by the per-seq "cross-ubatch bridge")
//     - single-ubatch prefill / after verify: draft()'s seed step uses "dp.id_last"
//                              (target's freshest sample) to complete the pair
//
// Per-seq carry-over state:
//   pending_g_last    [n_embd_dec]  ┐  the deferred boundary's (g, pos). Set by
//   pending_pos_last  llama_pos     ┘  process() at end of ubatch (= last row);
//                                       rebased by accept() to first-non-accepted pos.
//   verify_g          [N × n_embd_dec] snapshot of process()'s encoder output;
//   verify_pos_first  llama_pos         consumed by accept() to recover the right
//   verify_g_rows     int32_t           pending_g_last row for any n_accepted value.
//
// Performance is overall good but there is waste in verify cycle:
//   process() runs encoder + decoder on the *full* verify batch including rows for
//   rejected drafts. The KV at those positions is then dropped.
//
// TODO: Not sure if we need optimization for this waste?
// If so we may need hybrid stash:
//      in verify mode, have process() only stash features and let draft() seed run
//      encoder+decoder on n_accepted+1 rows).
struct common_speculative_impl_draft_eagle3 : public common_speculative_impl {
    common_params_speculative_draft params;
    llama_batch batch;

    std::vector<common_sampler_ptr> smpls;

    // backend sampler chain per seq, attached to ctx_dft
    std::vector<llama_sampler *> backend_chains;

    int32_t n_embd_dec = 0;       // draft hidden size
    int32_t n_embd_enc = 0;       // target_layer_ids_n * target_hidden_size
    int32_t n_embd_tgt = 0;       // target model hidden size

    const int32_t * target_layer_ids   = nullptr; // model_dft's extract layer indices
    uint32_t        target_layer_ids_n = 0;

    // [per-seq] deferred boundary state
    std::vector<std::vector<float>> pending_g_last;
    std::vector<llama_pos>          pending_pos_last;

    // [per-seq] cache-facing boundary state — same row pending_* holds at quiescent
    // points, but NOT cleared by draft_sync_reset (F4 partial-reject). A task whose
    // final verify round is a partial reject goes idle with the draft-sync pair
    // empty; prompt_save must still succeed or the slot holds the ONLY copy of the
    // history (run7 03/09: 5 skipped saves → full re-prefills).
    std::vector<std::vector<float>> boundary_g_last;
    std::vector<llama_pos>          boundary_pos_last;

    // [per-seq] snapshot of the most recent process()'s encoder output
    std::vector<std::vector<float>> verify_g;         // [n_seq][n_rows * n_embd_dec]
    std::vector<llama_pos>          verify_pos_first; // [n_seq] — pos of verify_g[seq][0]
    std::vector<int32_t>            verify_g_rows;    // [n_seq] — number of rows

    // scratch buffer for concatenated target features [n_tokens, n_embd_enc]
    std::vector<float> features_buf;
    std::vector<float> g_embd_buf;

    common_speculative_impl_draft_eagle3(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3, n_seq)
        , params(params.draft)
    {
        LOG_INF("%s: adding speculative implementation 'draft-eagle3'\n", __func__);
        LOG_INF("%s: - n_max=%d, n_min=%d, p_min=%f, backend_sampling=%d\n", __func__, params.draft.n_max, params.draft.n_min, params.draft.p_min, (int) params.draft.backend_sampling);

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;
        GGML_ASSERT(ctx_tgt && ctx_dft && "EAGLE3 requires ctx_tgt and ctx_dft to be set");

        const llama_model * model_dft = llama_get_model(ctx_dft);
        const llama_model * model_tgt = llama_get_model(ctx_tgt);

        target_layer_ids   = llama_model_target_layer_ids  (model_dft);
        target_layer_ids_n = llama_model_target_layer_ids_n(model_dft);
        if (target_layer_ids_n != 3) {
            throw std::runtime_error("draft model is not eagle3 (expected 3 extract layers, got " +
                                     std::to_string(target_layer_ids_n) + ")");
        }

        n_embd_tgt = llama_model_n_embd(model_tgt);
        n_embd_dec = llama_model_n_embd(model_dft);
        n_embd_enc = (int32_t) target_layer_ids_n * n_embd_tgt;

        const int32_t n_b = (int32_t) llama_n_batch(ctx_dft);
        batch = llama_batch_init(/*n_tokens=*/ n_b, /*embd=*/ n_embd_dec, /*n_seq_max=*/ 1);
        // llama_batch_init allocates only one of token/embd; eagle3 decoder needs both.
        // TODO: fix, how to call without malloc
        batch.token = (llama_token *) malloc(sizeof(llama_token) * n_b);

        smpls.resize(n_seq);
        for (auto & s : smpls) {
            common_params_sampling sparams;
            sparams.no_perf  = false;
            sparams.top_k    = 10;
            sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
            s.reset(common_sampler_init(llama_get_model(ctx_dft), sparams));
        }

        // offload draft sampling to the backend
        backend_chains.assign(n_seq, nullptr);
        if (this->params.backend_sampling) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
                llama_sampler_chain_add(chain, llama_sampler_init_top_k(10));

                if (!llama_set_sampler(ctx_dft, seq_id, chain)) {
                    LOG_WRN("%s: backend offload failed for seq_id=%d; using CPU sampler\n", __func__, (int) seq_id);
                    llama_sampler_free(chain);
                    chain = nullptr;
                }
                backend_chains[seq_id] = chain;
            }
        }

        // turn on extraction of the target layers' input embeddings
        for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
            llama_set_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k], true);
        }

        // turn on extraction of the draft model's pre-norm hidden state
        // (used both for the encoder output g_embd and the decoder pre-norm output).
        llama_set_embeddings_pre_norm(ctx_dft, true, /*masked*/ true);

        pending_g_last.assign(n_seq, std::vector<float>(n_embd_dec, 0.0f));
        pending_pos_last.assign(n_seq, -1);

        boundary_g_last.assign(n_seq, std::vector<float>(n_embd_dec, 0.0f));
        boundary_pos_last.assign(n_seq, -1);

        verify_g.assign(n_seq, std::vector<float>());
        verify_pos_first.assign(n_seq, -1);
        verify_g_rows.assign(n_seq, 0);
    }

    ~common_speculative_impl_draft_eagle3() override {
        auto * ctx_dft = this->params.ctx_dft;
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) backend_chains.size(); ++seq_id) {
            if (backend_chains[seq_id] == nullptr) {
                continue;
            }
            if (ctx_dft) {
                llama_set_sampler(ctx_dft, seq_id, nullptr);
            }
            llama_sampler_free(backend_chains[seq_id]);
        }
        backend_chains.clear();

        if (batch.token != nullptr) {
            free(batch.token);
            batch.token = nullptr;
        }
        llama_batch_free(batch);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        const int32_t N = (int32_t) prompt.size();
        if (N <= 0) {
            return;
        }
        // expected state after prefill: ctx_dft has pos 0..N-2 (last position is deferred to
        // draft()'s seed step). Warn only if more than one position is missing.
        auto * ctx_dft = this->params.ctx_dft;
        const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id);
        if (pos_max < N - 2) {
            LOG_WRN("%s: ctx_dft pos_max=%d < N-2=%d — process() did not run on every prefill ubatch. "
                    "Drafts may degrade.\n",
                    __func__, (int) pos_max, N - 2);
        }
    }

    bool process(const llama_batch & batch_in) override {
        if (batch_in.n_tokens <= 0) {
            return true;
        }

        if (batch_in.token == nullptr || batch_in.embd != nullptr) {
            return true;
        }

        const int32_t n_tokens = batch_in.n_tokens;

        // i_batch_beg[seq] / i_batch_end[seq]: inclusive batch indices of this seq's
        // first/last token in batch_in. Assumes per-seq tokens are contiguous within
        // the ubatch (server's default ordering).
        std::vector<int32_t> i_batch_beg(n_seq, -1);
        std::vector<int32_t> i_batch_end(n_seq, -1);
        for (int k = 0; k < n_tokens; ++k) {
            GGML_ASSERT(batch_in.n_seq_id[k] == 1);
            const llama_seq_id seq_id = batch_in.seq_id[k][0];
            if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
                continue;
            }
            i_batch_end[seq_id] = k;
            if (i_batch_beg[seq_id] < 0) {
                i_batch_beg[seq_id] = k;
            }
        }

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;

        // Interleave each extract_layer's hidden state into a contiguous buffer of
        // shape [n_tokens, target_layer_ids_n * n_embd_tgt]. Then run EAGLE3 encoder
        // to get one g_embd row per token.
        features_buf.resize((size_t) n_tokens * n_embd_enc, 0.0f);

        for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
            const float * layer = llama_get_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k]);
            if (!layer) {
                GGML_ABORT("EAGLE3: target layer %d input not extracted.", target_layer_ids[k]);
            }
            for (int32_t i = 0; i < n_tokens; ++i) {
                float * dst = features_buf.data() + (size_t) i * n_embd_enc + k * (size_t) n_embd_tgt;
                const float * src = layer + (size_t) i * n_embd_tgt;
                std::memcpy(dst, src, (size_t) n_embd_tgt * sizeof(float));
            }
        }

        g_embd_buf.resize((size_t) n_tokens * n_embd_dec);

        // llama_encode() requires the full encoder batch to fit in n_ubatch.
        // Allow batch > ubatch: eagle3's per-token encoder can be chunked safely.
        const int32_t n_ubatch_dft = (int32_t) llama_n_ubatch(ctx_dft);
        for (int32_t i = 0; i < n_tokens; i += n_ubatch_dft) {
            const int32_t n_chunk = std::min(n_ubatch_dft, n_tokens - i);

            llama_batch enc_batch = {
                /*.n_tokens =*/ n_chunk,
                /*.token    =*/ nullptr,
                /*.embd     =*/ features_buf.data() + (size_t) i * n_embd_enc,
                /*.pos      =*/ nullptr,
                /*.n_seq_id =*/ nullptr,
                /*.seq_id   =*/ nullptr,
                /*.logits   =*/ nullptr,
            };
            const int32_t rc = llama_encode(ctx_dft, enc_batch);
            if (rc != 0) {
                LOG_ERR("%s: llama_encode(ctx_dft) failed rc=%d (n_tokens=%d, offset=%d)\n",
                        __func__, rc, (int) n_chunk, (int) i);
                return false;
            }

            // g_embd has shape [n_chunk, n_embd_dec] in ctx_dft's pre-norm embeddings buffer.
            const float * g_embd_chunk = llama_get_embeddings_pre_norm(ctx_dft);
            GGML_ASSERT(g_embd_chunk && "EAGLE3 encoder produced no output.");
            std::memcpy(g_embd_buf.data() + (size_t) i * n_embd_dec,
                        g_embd_chunk,
                        (size_t) n_chunk * n_embd_dec * sizeof(float));
        }

        const float * g_embd = g_embd_buf.data();

        const size_t row_bytes = (size_t) n_embd_dec * sizeof(float);

        // EAGLE3 decoder input convention: at memory pos P the input pair is
        // (token[P+1], g_embd[P]). This shifts the token index "left by one" relative to g_embd.
        //
        // Per seq, in order:
        //   (a) cross-ubatch bridge — when applicable, write the previously-deferred
        //       pos using this ubatch's first token + pending_g_last.
        //   (b) main write loop — for k in [beg, end-1], write (token[k+1], g_embd[k])
        //       at pos[k]. The last training pos (k=end) is left unwritten = new
        //       deferred boundary, completed by the next process() or draft() call.
        //   (c) refresh deferred state — stash this ubatch's full g_embd into verify_g,
        //       update pending_g_last / pending_pos_last to the last row.
        common_batch_clear(batch);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            const int32_t beg = i_batch_beg[seq_id];
            const int32_t end = i_batch_end[seq_id];
            if (beg < 0 || end < 0) {
                continue;
            }

            // cross-ubatch bridge — complete the prior ubatch's deferred boundary.
            // Fires iff all three preconditions hold:
            //   1) pending_pos_last >= 0
            //   2) pending_pos_last + 1 == pos[beg]
            //   3) pending_pos_last > dft_pos_max // TODO: is this check needed?
            const llama_pos pending_pos = pending_pos_last[seq_id];
            if (pending_pos >= 0 && pending_pos + 1 == batch_in.pos[beg]) {
                const llama_pos dft_pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id);
                if (pending_pos > dft_pos_max) {
                    common_batch_add(batch, batch_in.token[beg], pending_pos, { seq_id }, /*logits=*/ false);
                    std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd_dec,
                                pending_g_last[seq_id].data(), row_bytes);
                }
            }

            for (int32_t k = beg; k < end; ++k) {
                common_batch_add(batch, batch_in.token[k + 1], batch_in.pos[k], { seq_id }, /*logits=*/ false);
                std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd_dec,
                            g_embd + (size_t) k * n_embd_dec, row_bytes);
            }

            // refresh deferred state
            const int32_t n_rows = end - beg + 1;
            verify_pos_first[seq_id] = batch_in.pos[beg];
            pending_pos_last[seq_id] = batch_in.pos[end];
            verify_g_rows[seq_id]    = n_rows;
            verify_g[seq_id].resize((size_t) n_rows * n_embd_dec, 0.0f);
            std::memcpy(verify_g[seq_id].data(),       g_embd + (size_t) beg * n_embd_dec, row_bytes * n_rows);
            std::memcpy(pending_g_last[seq_id].data(), g_embd + (size_t) end * n_embd_dec, row_bytes);

            // refresh the cache-facing boundary at the same position
            boundary_pos_last[seq_id] = batch_in.pos[end];
            std::memcpy(boundary_g_last[seq_id].data(), g_embd + (size_t) end * n_embd_dec, row_bytes);
        }

        if (batch.n_tokens > 0) {
            const int32_t rc = llama_decode(ctx_dft, batch);
            if (rc != 0) {
                LOG_ERR("%s: llama_decode(ctx_dft) failed rc=%d (n_tokens=%d, ubatch_pos[0]=%d)\n",
                        __func__, rc, (int) batch.n_tokens, (int) batch_in.pos[0]);
                return false;
            }
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        common_batch_clear(batch);

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

        const size_t row_bytes = (size_t) n_embd_dec * sizeof(float);

        // Complete the deferred boundary pair (dp.id_last, pending_g_last) at memory
        // pos pending_pos_last. dp.id_last is target's freshest sample (= corrected
        // token after verify, or first generated token after prefill), matching the
        // EAGLE3 input convention (token[P+1], g_embd[P]) at pos P.
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }
            if (pending_pos_last[seq_id] < 0) {
                continue;
            }

            n_drafting++;
            drafting[seq_id] = true;
            common_sampler_reset(smpls[seq_id].get());

            llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, pending_pos_last[seq_id], -1);

            common_batch_add(batch, dp.id_last, pending_pos_last[seq_id], { seq_id }, true);
            std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd_dec,
                        pending_g_last[seq_id].data(),
                        row_bytes);
        }

        if (batch.n_tokens == 0) {
            return;
        }

        int ret = llama_decode(ctx_dft, batch);
        if (ret != 0) {
            LOG_WRN("%s: llama_decode returned %d\n", __func__, ret);
            return;
        }

        int i = 0;

        while (n_drafting > 0) {
            int i_batch = 0;

            common_batch_clear(batch);

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (!drafting[seq_id]) {
                    continue;
                }

                auto & dp = dparams[seq_id];
                auto & result = *dp.result;
                const int32_t n_max = common_speculative_effective_n_max(params, dp);
                if (n_max <= (int) result.size()) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                auto * smpl = smpls[seq_id].get();

                common_sampler_sample(smpl, ctx_dft, i_batch, true);
                // pre-norm hidden state of this position becomes g_embd for the next step
                const float * prenorm = llama_get_embeddings_pre_norm_ith(ctx_dft, i_batch);
                ++i_batch;

                const auto * cur_p = common_sampler_get_candidates(smpl, true);

                for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                    LOG_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                            common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                const llama_token id = cur_p->data[0].id;

                // only collect very high-confidence draft tokens
                // (configurable via --spec-draft-p-min, set to 0.0 to disable early-stop)
                if (cur_p->data[0].p < common_speculative_effective_p_min(params, dp)) {
                    drafting[seq_id] = false;
                    n_drafting--;

                    continue;
                }

                common_sampler_accept(smpl, id, true);

                result.push_back(id);

                if (n_max <= (int) result.size()) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                common_batch_add(batch, id, pending_pos_last[seq_id] + (i + 1), { seq_id }, true);
                std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd_dec, prenorm, row_bytes);
            }

            if (batch.n_tokens == 0) {
                break;
            }

            ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                LOG_WRN("%s: llama_decode[%d] returned %d\n", __func__, i, ret);
                break;
            }

            ++i;
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            const int32_t n_max = common_speculative_effective_n_max(params, dp);
            const int32_t n_min = common_speculative_effective_n_min(params, dp, n_max);
            if (dp.result->size() < (size_t) n_min) {
                dp.result->clear();
            }
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        const int32_t n_rows = verify_g_rows[seq_id];
        if (n_rows <= 0) {
            return;
        }

        const int32_t i_g = std::min<int32_t>(n_accepted, n_rows - 1);
        pending_pos_last[seq_id] = verify_pos_first[seq_id] + i_g;
        std::memcpy(pending_g_last[seq_id].data(),
                    verify_g[seq_id].data() + (size_t) i_g * n_embd_dec,
                    (size_t) n_embd_dec * sizeof(float));

        // the accepted row is the cache-facing boundary (a save right after this
        // partial rejection — e.g. at task end — must still serve get_state)
        boundary_pos_last[seq_id] = pending_pos_last[seq_id];
        std::memcpy(boundary_g_last[seq_id].data(),
                    verify_g[seq_id].data() + (size_t) i_g * n_embd_dec,
                    (size_t) n_embd_dec * sizeof(float));
    }

    // we only need to stash the deferred boundary's g_embd row for recurrent/hybrid targets:
    // their single-position checkpoints drop it on restore
    bool need_boundary_stash() const {
        const llama_model * model_tgt = llama_get_model(params.ctx_tgt);
        return llama_model_is_recurrent(model_tgt) || llama_model_is_hybrid(model_tgt);
    }

    bool get_state(llama_seq_id seq_id, std::vector<uint8_t> & data) const override {
        if (!need_boundary_stash()) {
            return false;
        }
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq || boundary_pos_last[seq_id] < 0) {
            return false;
        }

        const llama_pos          pos = boundary_pos_last[seq_id];
        const std::vector<float> & g = boundary_g_last[seq_id];

        data.resize(sizeof(llama_pos) + g.size() * sizeof(float));
        std::memcpy(data.data(),                     &pos,     sizeof(llama_pos));
        std::memcpy(data.data() + sizeof(llama_pos), g.data(), g.size() * sizeof(float));
        return true;
    }

    bool set_state(llama_seq_id seq_id, const std::vector<uint8_t> & data) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return false;
        }
        if (data.empty()) {
            pending_pos_last[seq_id] = -1;
            std::fill(pending_g_last[seq_id].begin(), pending_g_last[seq_id].end(), 0.0f);
            verify_g[seq_id].clear();
            verify_pos_first[seq_id] = -1;
            verify_g_rows[seq_id] = 0;
            boundary_pos_last[seq_id] = -1;
            std::fill(boundary_g_last[seq_id].begin(), boundary_g_last[seq_id].end(), 0.0f);
            return !need_boundary_stash();
        }
        if (!need_boundary_stash()) {
            return true;
        }
        if (data.size() != sizeof(llama_pos) + (size_t) n_embd_dec * sizeof(float)) {
            return false;
        }

        llama_pos pos = -1;
        std::memcpy(&pos, data.data(), sizeof(llama_pos));

        pending_pos_last[seq_id] = pos;
        pending_g_last[seq_id].resize(n_embd_dec);
        std::memcpy(pending_g_last[seq_id].data(), data.data() + sizeof(llama_pos), (size_t) n_embd_dec * sizeof(float));

        boundary_pos_last[seq_id] = pos;
        boundary_g_last[seq_id].resize(n_embd_dec);
        std::memcpy(boundary_g_last[seq_id].data(), data.data() + sizeof(llama_pos), (size_t) n_embd_dec * sizeof(float));
        return true;
    }

    bool state_required() const override {
        return need_boundary_stash();
    }

    // F4: clear only the draft-sync pair; the cache boundary survives a
    // partial-reject reset (prompt_save at idle must still succeed — see the
    // boundary_* members' note). set_state({}) remains the FULL invalidation
    // for cold-fallback paths where the KV itself is gone.
    void draft_sync_reset(llama_seq_id seq_id) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }
        pending_pos_last[seq_id] = -1;
        std::fill(pending_g_last[seq_id].begin(), pending_g_last[seq_id].end(), 0.0f);
        verify_g[seq_id].clear();
        verify_pos_first[seq_id] = -1;
        verify_g_rows[seq_id] = 0;
    }

    void shift_state(llama_seq_id seq_id, llama_pos delta) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq || delta == 0) {
            return;
        }

        if (pending_pos_last[seq_id] >= 0) {
            pending_pos_last[seq_id] += delta;
        }
        if (verify_pos_first[seq_id] >= 0) {
            verify_pos_first[seq_id] += delta;
        }
        if (boundary_pos_last[seq_id] >= 0) {
            boundary_pos_last[seq_id] += delta;
        }
    }

    bool need_embd() const override {
        return false;
    }

    int32_t draft_n_max() const override {
        return std::max(0, params.n_max);
    }
};

struct common_speculative_impl_draft_dflash : public common_speculative_impl {
    common_params_speculative_draft params;

    llama_batch batch;
    llama_batch batch_inject;

    std::vector<common_sampler_ptr> smpls;

    int32_t n_embd_dec = 0;
    int32_t n_embd_enc = 0;
    int32_t n_embd_tgt = 0;

    int32_t     block_size    = 0;
    llama_token mask_token_id = 0;

    // backend sampler chain per seq, attached to ctx_dft
    std::vector<llama_sampler *> backend_chains;

    bool    is_dflash2     = false;
    int32_t selector_top_k = 0;
    std::vector<std::mt19937> selector_rng;
    std::vector<bool> selector_reset;

    // t8 stadio 2 (Task 5 test coverage): one-shot INFO marker the first time a
    // draft decode is split into more than one width group (mixed plain/concat
    // rounds of concurrent sequences) - the grouped decode path is otherwise
    // silent, and the smoke tests assert on this marker
    bool width_split_logged = false;

    // draft-dspark: the draft carries a Markov head and uses an anchor-first block layout
    const bool is_dspark;

    // dspark speculators
    bool sample_from_anchor = true;

    const int32_t * target_layer_ids   = nullptr;
    uint32_t        target_layer_ids_n = 0;
    std::vector<int32_t> target_layer_ids_adjusted;

    std::vector<float> features_buf;

    common_speculative_impl_draft_dflash(const common_params_speculative & params, uint32_t n_seq,
            common_speculative_type type = COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH)
        : common_speculative_impl(type, n_seq)
        , params(params.draft)
        , is_dspark(type == COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK) {
        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;
        GGML_ASSERT(ctx_tgt && ctx_dft && "DFlash requires ctx_tgt and ctx_dft to be set");

        const llama_model * model_dft = llama_get_model(ctx_dft);
        const llama_model * model_tgt = llama_get_model(ctx_tgt);

        target_layer_ids   = llama_model_target_layer_ids  (model_dft);
        target_layer_ids_n = llama_model_target_layer_ids_n(model_dft);
        GGML_ASSERT(target_layer_ids_n > 0 && "DFlash model has no target_layer_ids");
        target_layer_ids_adjusted.assign(target_layer_ids, target_layer_ids + target_layer_ids_n);
        const int32_t target_layer_offset = []() {
            const char * env = std::getenv("LLAMA_DFLASH_TARGET_LAYER_OFFSET");
            return env ? std::atoi(env) : 0;
        }();
        if (target_layer_offset != 0) {
            for (int32_t & id : target_layer_ids_adjusted) {
                id += target_layer_offset;
                if (id < 0) {
                    throw std::runtime_error("DFlash target layer offset produced a negative layer id");
                }
            }
        }

        n_embd_tgt = llama_model_n_embd(model_tgt);
        n_embd_dec = llama_model_n_embd(model_dft);
        n_embd_enc = (int32_t) target_layer_ids_n * n_embd_tgt;

        block_size = 16;
        {
            char buf[32] = {};
            if (llama_model_meta_val_str(model_dft, "dflash.block_size", buf, sizeof(buf)) >= 0 ||
                llama_model_meta_val_str(model_dft, "dflash-draft.dflash.block_size", buf, sizeof(buf)) >= 0) {
                block_size = std::atoi(buf);
            }
            if (llama_model_meta_val_str(model_dft, "dflash.sample_from_anchor", buf, sizeof(buf)) >= 0) {
                sample_from_anchor = std::strcmp(buf, "true") == 0;
            }
            if (llama_model_meta_val_str(model_dft, "dflash.selector_top_k", buf, sizeof(buf)) >= 0) {
                selector_top_k = std::atoi(buf);
                is_dflash2 = selector_top_k > 0;
            }
        }
        mask_token_id = llama_vocab_mask(llama_model_get_vocab(model_dft));
        if (mask_token_id < 0) {
            char buf[32] = {};
            if (llama_model_meta_val_str(model_dft, "dflash.mask_token_id", buf, sizeof(buf)) >= 0 ||
                llama_model_meta_val_str(model_dft, "dflash-draft.dflash.mask_token_id", buf, sizeof(buf)) >= 0) {
                mask_token_id = std::atoi(buf);
            }
        }

        LOG_INF("%s: adding speculative implementation '%s'\n", __func__, common_speculative_type_to_str(type).c_str());
        LOG_INF("%s: - n_max=%d, n_min=%d, p_min=%.2f\n", __func__, this->params.n_max, this->params.n_min, this->params.p_min);
        LOG_INF("%s: - block_size=%d, mask_token_id=%d, n_extract=%u\n", __func__, block_size, mask_token_id, target_layer_ids_n);
        if (target_layer_offset != 0) {
            LOG_INF("%s: - target_layer_offset=%d\n", __func__, target_layer_offset);
        }
        const int32_t n_draft_max = is_dspark ? block_size : block_size - 1;
        if (this->params.n_max > n_draft_max || this->params.n_min > n_draft_max) {
            LOG_WRN("%s: requested draft size (n_max=%d, n_min=%d) exceeds the trained block size %d -- clamping to %d\n",
                    __func__, this->params.n_max, this->params.n_min, block_size, n_draft_max);
            this->params.n_max = std::min(this->params.n_max, n_draft_max);
            this->params.n_min = std::min(this->params.n_min, n_draft_max);
        }

        batch        = llama_batch_init(llama_n_batch(ctx_dft), 0,          n_seq);
        batch_inject = llama_batch_init(llama_n_batch(ctx_dft), n_embd_dec, n_seq);

        smpls.resize(n_seq);
        for (auto & s : smpls) {
            common_params_sampling sparams;
            sparams.no_perf  = false;
            sparams.top_k    = is_dflash2 ? selector_top_k : (is_dspark ? 10 : 1);
            sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
            s.reset(common_sampler_init(model_dft, sparams));
        }

        selector_rng.resize(n_seq);
        selector_reset.assign(n_seq, true);

        // offload draft sampling to the backend
        backend_chains.assign(n_seq, nullptr);
        if (this->params.backend_sampling && !is_dflash2) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
                llama_sampler_chain_add(chain, llama_sampler_init_top_k(10));

                if (!llama_set_sampler(ctx_dft, seq_id, chain)) {
                    SPC_WRN("backend offload failed for seq_id=%d; using CPU sampler\n", (int) seq_id);
                    llama_sampler_free(chain);
                    chain = nullptr;
                }
                backend_chains[seq_id] = chain;
            }
        }

        // turn on extraction of the target layers' input embeddings
        for (const int32_t id : target_layer_ids_adjusted) {
            llama_set_embeddings_layer_inp(ctx_tgt, (uint32_t) id, true);
        }

        // DFlash2 reads its selector lattice from h_nextn and never consumes raw logits.
        llama_set_embeddings_pre_norm(ctx_dft, true, /*masked*/ !is_dflash2);
        llama_set_causal_attn(ctx_dft, false); // DFlash needs non-causal attention
    }

    ~common_speculative_impl_draft_dflash() override {
        auto * ctx_dft = this->params.ctx_dft;
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) backend_chains.size(); ++seq_id) {
            if (backend_chains[seq_id] == nullptr) {
                continue;
            }
            if (ctx_dft) {
                llama_set_sampler(ctx_dft, seq_id, nullptr);
            }
            llama_sampler_free(backend_chains[seq_id]);
        }
        backend_chains.clear();

        llama_batch_free(batch);
        llama_batch_free(batch_inject);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        const int32_t N = (int32_t) prompt.size();
        if (N <= 0) {
            return;
        }

        selector_reset[seq_id] = true;

        if (dparams_routing != nullptr) {
            const common_speculative_type drafter = (*dparams_routing)[seq_id].drafter;
            if (drafter != COMMON_SPECULATIVE_TYPE_NONE && drafter != type) {
                // this sequence is drafted by another implementation and never
                // advances in this context - the pos_max check below would always trip
                return;
            }
        }

        const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(params.ctx_dft), seq_id);
        if (pos_max < N - 1) {
            LOG_WRN("%s: ctx_dft pos_max=%d < N-1=%d - process() did not run on every prefill ubatch. "
                    "Drafts may degrade.\n",
                    __func__, (int) pos_max, N - 1);
        }
    }

    bool process(const llama_batch & batch_in) override {
        if (batch_in.n_tokens <= 0) {
            return true;
        }

        if (batch_in.token == nullptr || batch_in.embd != nullptr) {
            return true;
        }

        const int32_t n_tokens = batch_in.n_tokens;

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;

        const int32_t n_ubatch = (int32_t) llama_n_ubatch(ctx_dft);

        // indices, within each chunk, of the tokens this implementation ingests:
        // with per-seq drafter routing active, only the sequences routed here (or
        // unrouted) advance in this external draft context - the others are drafted
        // by another implementation. Without routing the mapping is the identity,
        // i.e. the pre-routing behavior.
        std::vector<int32_t> sel;

        // Flatten token-wise encoder work into shared chunks while preserving each row's position and sequence.
        for (int32_t offset = 0; offset < n_tokens; offset += n_ubatch) {
            const int32_t n_chunk = std::min(n_ubatch, n_tokens - offset);

            sel.clear();
            if (dparams_routing == nullptr) {
                sel.resize(n_chunk);
                std::iota(sel.begin(), sel.end(), 0);
            } else {
                for (int32_t i = 0; i < n_chunk; ++i) {
                    const int32_t j = offset + i;
                    GGML_ASSERT(batch_in.n_seq_id[j] == 1);
                    const llama_seq_id seq_id = batch_in.seq_id[j][0];
                    GGML_ASSERT(seq_id >= 0 && seq_id < (llama_seq_id) n_seq);
                    const common_speculative_type drafter = (*dparams_routing)[seq_id].drafter;
                    if (drafter == COMMON_SPECULATIVE_TYPE_NONE || drafter == type) {
                        sel.push_back(i);
                    }
                }
            }

            if (sel.empty()) {
                // no sequence of this chunk is drafted by this implementation
                continue;
            }

            const int32_t n_sel = (int32_t) sel.size();

            // note: the compaction below is always between distinct buffers
            // (features_buf vs the target embedding buffer), so memcpy is safe
            features_buf.resize((size_t) n_sel * n_embd_enc);
            for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
                const int32_t layer_id = target_layer_ids_adjusted[k];
                const float * layer = llama_get_embeddings_layer_inp(ctx_tgt, (uint32_t) layer_id);
                if (!layer) {
                    GGML_ABORT("DFlash: target layer %d input not extracted.", layer_id);
                }
                for (int32_t s = 0; s < n_sel; ++s) {
                    float       * dst = features_buf.data() + (size_t) s * n_embd_enc + k * (size_t) n_embd_tgt;
                    const float * src = layer + (size_t) (offset + sel[s]) * n_embd_tgt;
                    std::memcpy(dst, src, (size_t) n_embd_tgt * sizeof(float));
                }
            }

            llama_batch enc_batch = {
                /*.n_tokens =*/ n_sel,
                /*.token    =*/ nullptr,
                /*.embd     =*/ features_buf.data(),
                /*.pos      =*/ nullptr,
                /*.n_seq_id =*/ nullptr,
                /*.seq_id   =*/ nullptr,
                /*.logits   =*/ nullptr,
            };

            int32_t rc = llama_encode(ctx_dft, enc_batch);
            if (rc != 0) {
                LOG_ERR("%s: llama_encode(ctx_dft) failed rc=%d (n_tokens=%d, offset=%d)\n",
                        __func__, rc, (int) n_sel, (int) offset);
                return false;
            }

            const float * inp_g = llama_get_embeddings_pre_norm(ctx_dft);
            GGML_ASSERT(inp_g && "DFlash encoder produced no output.");

            batch_inject.n_tokens = n_sel;
            std::memcpy(batch_inject.embd, inp_g, (size_t) n_sel * n_embd_dec * sizeof(float));
            for (int32_t s = 0; s < n_sel; ++s) {
                const int32_t j = offset + sel[s];
                GGML_ASSERT(batch_in.n_seq_id[j] == 1);
                const llama_seq_id seq_id = batch_in.seq_id[j][0];
                GGML_ASSERT(seq_id >= 0 && seq_id < (llama_seq_id) n_seq);
                batch_inject.pos[s]       = batch_in.pos[j];
                batch_inject.n_seq_id[s]  = 1;
                batch_inject.seq_id[s][0] = seq_id;
                batch_inject.logits[s]    = false;
            }

            rc = llama_decode(ctx_dft, batch_inject);
            if (rc != 0) {
                LOG_ERR("%s: llama_decode(ctx_dft) failed rc=%d (n_tokens=%d, offset=%d)\n",
                        __func__, rc, (int) n_sel, (int) offset);
                return false;
            }
            // The server may switch contexts before the next draft decode.
            llama_synchronize(ctx_dft);
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        std::vector<int32_t> i_block_beg(n_seq, -1);
        std::vector<int32_t> n_block    (n_seq,  0);
        std::vector<int32_t> n_dft_v    (n_seq,  0);

        // t8 stadio 2 (spec §3): per-seq MTP head tokens conditioning the block
        // (0 outside concat rounds; dspark never receives a head)
        std::vector<int32_t> n_head     (n_seq,  0);
        // t8: rows this sequence adds to the draft batch (head + block)
        std::vector<int32_t> n_rows     (n_seq,  0);

        // t8 stadio 2: sample one sequence from the logits/embeddings of the LAST
        // decode (the next decode overwrites the host buffers). Extracted from
        // the former single-pass loop so the width-grouped decode below can
        // sample each group right after its own decode.
        auto sample_seq = [&](llama_seq_id seq_id, int32_t beg) {
            auto & dp = dparams[seq_id];

            const int32_t n_block_tokens = n_block[seq_id];

            auto * smpl = smpls[seq_id].get();
            auto & result = *dp.result;

            const int32_t n_draft = n_dft_v[seq_id];
            const int32_t n_min   = common_speculative_effective_n_min(params, dp, n_draft);
            const float   p_min   = common_speculative_effective_p_min(params, dp);

            if (dp.dists) {
                dp.dists->clear();
            }

            if (is_dflash2) {
                GGML_ASSERT(dp.temperature <= 0.0f || dp.dists);
                const float * lattice = llama_get_embeddings_pre_norm(ctx_dft);
                GGML_ASSERT(lattice && "DFlash2 selector produced no lattice");

                if (selector_reset[seq_id]) {
                    uint32_t seed = dp.seed;
                    if (seed == LLAMA_DEFAULT_SEED) {
                        seed = (uint32_t) std::chrono::high_resolution_clock::now().time_since_epoch().count();
                    }
                    selector_rng[seq_id].seed(seed ^ 0x85ebca6bU);
                    selector_reset[seq_id] = false;
                }

                int32_t predecessor = 0;
                for (int32_t i = 1; i < n_block_tokens; ++i) {
                    // t8 concat: skip the head rows - the selector lattice walks
                    // the noise block only, which starts after the head
                    const float * row = lattice + (size_t) (beg + n_head[seq_id] + i) * n_embd_dec;
                    const float * scores = row + selector_top_k + (size_t) predecessor * selector_top_k;

                    if (dp.temperature > 0.0f) {
                        common_speculative_token_dist dist;
                        dist.ids.resize(selector_top_k);
                        dist.probs.resize(selector_top_k);
                        const float max_score = *std::max_element(scores, scores + selector_top_k);
                        float sum = 0.0f;
                        for (int32_t k = 0; k < selector_top_k; ++k) {
                            dist.ids[k] = (llama_token) row[k];
                            dist.probs[k] = std::exp((scores[k] - max_score) / dp.temperature);
                            sum += dist.probs[k];
                        }
                        for (float & p : dist.probs) {
                            p /= sum;
                        }
                        std::discrete_distribution<int32_t> sample(dist.probs.begin(), dist.probs.end());
                        predecessor = sample(selector_rng[seq_id]);
                        result.push_back(dist.ids[predecessor]);
                        // t8 concat: dists cover ONLY the noise-block rows - the
                        // head tokens carry no distribution - so a composed round
                        // with temp>0 has dists.size() < draft.size() and the
                        // server's size-guard falls back to exact accept; the
                        // explicit temp>0 single-drafter fallback is Task 6
                        dp.dists->push_back(std::move(dist));
                    } else {
                        predecessor = (int32_t) std::distance(scores,
                                std::max_element(scores, scores + selector_top_k));
                        result.push_back((llama_token) row[predecessor]);
                    }
                }

                // t8 concat: the closing arm's n_min counts the WHOLE round, head
                // included (result holds head + block), so a high n_min can clear
                // also a composed head -> round lost (spec §10 "never a lost
                // round" tension: the target still decodes one token unperturbed,
                // but the composed round is dropped; pre-existing semantics)
                if (result.size() < (size_t) n_min) {
                    result.clear();
                    if (dp.dists) {
                        dp.dists->clear();
                    }
                }
                return;
            }

            if (is_dspark) {
                // DSpark predicts the next token at position zero.  Its first
                // extracted pre-norm component carries the confidence score.
                const float * conf = p_min > 0.0f ? llama_get_embeddings_pre_norm(ctx_dft) : nullptr;

                for (int32_t i = 0; i < n_block_tokens; ++i) {
                    // t8 concat: n_head is always 0 for dspark (never a head) -
                    // the offset is kept for symmetry with the other layouts
                    const int32_t idx = beg + n_head[seq_id] + i;
                    if (conf && conf[(size_t) idx * n_embd_dec] < p_min) {
                        break;
                    }

                    common_sampler_sample(smpl, ctx_dft, idx, true);
                    const auto * cur_p = common_sampler_get_candidates(smpl, true);

                    for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                        LOG_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                                seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                                common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                    }

                    const llama_token id = cur_p->data[0].id;
                    common_sampler_accept(smpl, id, true);
                    result.push_back(id);
                }
            } else {
                for (int32_t i = 1; i < n_block_tokens; ++i) {
                    // t8 concat: the noise block starts after the head rows
                    common_sampler_sample(smpl, ctx_dft, beg + n_head[seq_id] + i, true);

                    const auto * cur_p = common_sampler_get_candidates(smpl, true);

                    for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                        LOG_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                                seq_id, k, i - 1, cur_p->data[k].id, cur_p->data[k].p,
                                common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                    }

                    const llama_token id = cur_p->data[0].id;
                    if (cur_p->data[0].p < p_min) {
                        break;
                    }

                    common_sampler_accept(smpl, id, true);
                    result.push_back(id);
                }
            }

            if (result.size() < (size_t) n_min) {
                result.clear();
            }
        };

        // pass 1: per-seq sizing
        std::vector<llama_seq_id> seqs;
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            common_sampler_reset(smpls[seq_id].get());

            const int32_t n_draft = common_speculative_effective_n_max(params, dp);

            const int32_t n_block_tokens = n_draft + (is_dspark ? 0 : 1);
            n_block[seq_id] = n_block_tokens;
            n_dft_v[seq_id] = n_draft;
            n_head [seq_id] = (!is_dspark && dp.concat_head) ? (int32_t) dp.concat_head->size() : 0;
            n_rows [seq_id] = n_head[seq_id] + n_block_tokens;
            seqs.push_back(seq_id);
        }

        if (seqs.empty()) {
            return;
        }

        // t8 stadio 2: the DFlash2 selector graph requires a uniform per-seq width
        // within one decode - it derives tokens_per_block = n_tokens / n_seqs_unq
        // and asserts the exact division, so a batch mixing a concat round
        // (1 + k1' + n_max rows) with a plain block (1 + n_max rows) would abort.
        // Group the sequences by width and decode one group at a time, sampling
        // each group before the next decode overwrites the host buffers. With a
        // single width (every pre-t8 round) this is exactly the former
        // one-batch one-decode flow.
        std::sort(seqs.begin(), seqs.end(), [&](llama_seq_id a, llama_seq_id b) {
            return n_rows[a] < n_rows[b];
        });

        // t8 stadio 2 (Task 5 test coverage): count the DISTINCT round widths in
        // this decode BEFORE the grouping loop - one group holding 2+ sequences
        // is the same-width case (the former one-decode flow), while a real
        // split (e.g. a composed concat round next to a plain block) shows up
        // as multiple single-width groups. The first real split is logged once
        // at INFO so the concurrency smoke tests can assert the grouped decode
        // path actually ran.
        size_t n_width_groups = 0;
        for (size_t s = 0; s < seqs.size(); ++s) {
            if (s == 0 || n_rows[seqs[s]] != n_rows[seqs[s - 1]]) {
                n_width_groups++;
            }
        }

        size_t g = 0;
        while (g < seqs.size()) {
            size_t h = g;
            while (h < seqs.size() && n_rows[seqs[h]] == n_rows[seqs[g]]) {
                ++h;
            }

            if (g == 0 && n_width_groups > 1) {
                if (!width_split_logged) {
                    SPC_INF("dflash decode split into %zu width groups (n_seqs=%zu)\n", n_width_groups, seqs.size());
                    width_split_logged = true;
                } else {
                    SPC_DBG("dflash decode split into %zu width groups (n_seqs=%zu)\n", n_width_groups, seqs.size());
                }
            }

            common_batch_clear(batch);

            for (size_t s = g; s < h; ++s) {
                const llama_seq_id seq_id = seqs[s];
                auto & dp = dparams[seq_id];

                const int32_t n       = (int32_t) dp.n_past;
                const int32_t n_draft = n_dft_v[seq_id];

                i_block_beg[seq_id] = batch.n_tokens;

                if (n_head[seq_id] > 0) {
                    // t8 stadio 2 (spec §3) - concat round: the drafting input
                    // carries the MTP head tokens at positions n+1..n+k1' and the
                    // noise block shifts to n+k1'+1..n+k1'+n_draft; the anchor
                    // (id_last @ n) is unchanged. The head replaces the mask
                    // placeholder at those positions and conditions the block
                    // through this context's non-causal attention - the head
                    // tokens are verified by the target in the SAME round, never
                    // confirmed unverified. All rows keep logits on so
                    // t_logits->ne[1] stays equal to n_tokens (DFlash2
                    // build_post_sampling reads the full-block logits).
                    // dp.result already holds the k1' head tokens: the sampling
                    // loop APPENDS to it, giving one concatenated draft per seq.
                    common_batch_add(batch, dp.id_last, n, { seq_id }, true);
                    for (int32_t j = 0; j < n_head[seq_id]; ++j) {
                        common_batch_add(batch, (*dp.concat_head)[j], n + 1 + j, { seq_id }, true);
                    }
                    for (int32_t i = 1; i <= n_draft; ++i) {
                        common_batch_add(batch, mask_token_id, n + n_head[seq_id] + i, { seq_id }, true);
                    }
                } else {
                    for (int32_t i = 0; i < n_block[seq_id]; ++i) {
                        // NOTE: unlike upstream, request logits on every noise position
                        // also for DFlash2: build_post_sampling() reads the full-block
                        // logits to build the selector lattice (t_logits->ne[1] must
                        // equal n_tokens).
                        common_batch_add(batch, i == 0 ? dp.id_last : mask_token_id, n + i, { seq_id }, true);
                    }
                }
            }

            GGML_ASSERT(batch.n_tokens > 0);

            // DFlash1 needs no pre-norm output during the noise-block decode; DFlash2
            // instead reads its selector lattice from the unmasked pre-norm rows of
            // exactly this decode, so the flag must stay on (set in the constructor).
            if (!is_dspark && !is_dflash2) {
                llama_set_embeddings_pre_norm(ctx_dft, false, /*masked*/ false);
            }
            const int ret = llama_decode(ctx_dft, batch);
            if (!is_dspark && !is_dflash2) {
                llama_set_embeddings_pre_norm(ctx_dft, true, /*masked*/ true);
            }
            if (ret != 0) {
                // t8 stadio 2 (spec §10): a failed decode never errors the request
                // - this group's sequences keep what they already drafted (a
                // concat head closes a SHORT round of k1' tokens; a plain block
                // leaves them without a draft this round) and the remaining
                // groups still run.
                LOG_WRN("%s: llama_decode returned %d\n", __func__, ret);
                g = h;
                continue;
            }

            for (size_t s = g; s < h; ++s) {
                sample_seq(seqs[s], i_block_beg[seqs[s]]);
            }

            g = h;
        }
    }

    void accept(llama_seq_id, uint16_t) override {
    }

    bool need_embd() const override {
        return false;
    }

    int32_t draft_n_max() const override {
        return std::max(0, params.n_max);
    }
};

struct common_speculative_state_draft_mtp : public common_speculative_impl {
    static constexpr int32_t MTP_DRAFT_TOP_K = 10;

    common_params_speculative_draft params; // reuses the draft-model params slot (ctx_tgt/ctx_dft)

    llama_batch batch;

    std::vector<common_sampler_ptr> smpls;

    // backend sampler chain per seq, attached to ctx_dft
    std::vector<llama_sampler *> backend_chains;

    int32_t n_embd = 0;

    // One MTP draft driver, three modes (set once in the ctor):
    //   is_mem_shared (gemma4): shares the target KV, runs all heads in one graph.
    //   chain_heads (step35): n_mtp_layers trained heads, one per draft step.
    //   neither (qwen35 / qwen35moe): a single trained MTP head.
    int32_t n_mtp_layers  = 1;
    bool    is_mem_shared = false;   // gemma4
    bool    chain_heads   = false;   // derived in the ctor: n_mtp_layers > 1 && !is_mem_shared

    // Per-sequence cross-batch carryover: pair (h_p, x_{p+1}) at MTP pos p+1.
    // The last h-row of one process() call needs the first token of the NEXT
    // call to pair with, so it's stashed here until that next call fires.
    std::vector<std::vector<float>> pending_h;   // [n_seq][n_embd]
    std::vector<std::vector<float>> pending_h_prev;
    std::vector<uint8_t> pending_h_valid;
    std::vector<uint8_t> pending_h_prev_valid;
    std::vector<llama_pos> pending_h_pos;
    std::vector<llama_pos> pending_h_prev_pos;

    // Boundary selected for the most recent process() batch. This is needed
    // when verification accepts row zero, and when a one-token replay advances
    // the current boundary while retaining the prior one.
    std::vector<std::vector<float>> process_boundary_h;
    std::vector<uint8_t> process_boundary_valid;
    std::vector<llama_pos> process_boundary_pos;

    std::vector<int32_t> i_batch_beg;
    std::vector<int32_t> i_batch_end;

    // Hidden rows from the most recent target verification batch, grouped by seq.
    // Row 0 corresponds to the sampled token, row N to the Nth accepted draft token.
    std::vector<std::vector<float>> verify_h;
    std::vector<int32_t> verify_h_rows;
    std::vector<llama_pos> verify_pos_first;

    // Per-seq draft length from the last draft() call, used in accept() to
    // roll back ctx_dft's recurrent state past the AR draft's redundant
    // pre-advancement before process() mirrored the verify batch.
    std::vector<uint16_t> last_n_drafted;
    std::vector<uint8_t> drafting;
    // W6-1 leva-2: one-shot F4 drafting suppression. Set by draft_sync_reset()
    // after a partial rejection; consumed by the next draft() which skips the
    // sequence for that single round (the measured H1 benefit) while the
    // accepted boundary stays valid so process() feeds the plain round's
    // mirror with the real h instead of the neutral zero row.
    std::vector<uint8_t> suppress_draft;
    std::vector<int>                i_last;
    std::vector<std::vector<float>> chain_h;

    // Ring of the most recent boundary h-rows per seq, so that a bounded
    // memory rollback (prompt-cache boundary salvage, see the server) can
    // rewind pending_h to any of the last RING_N positions without a full
    // cold reprocessing.
    //
    // Window sizing (2026-08-21 triage, t8 stadio 2): a client resend that
    // truncates or mutates the tail of the last assistant turn must rewind
    // pending_h to lcp-1, and that tail is the regenerated last word: its
    // token length follows the greedy stream, which shifts with kernel
    // numerics (the VK mul_mat_vec_max_cols 8->16 change moved 9-16-column
    // MUL_MAT batches from the tiled f16 kernel to the f32 vec kernel).
    // Causal tail-chop evidence on both the base image and the VK image:
    // trailing rollback succeeded at tail deltas (cached_tokens - lcp) of
    // 3-5 tokens and degraded to a cold fallback from 6 tokens on, because
    // RING_N=8 without position-dedup retained only ~5 distinct positions
    // (every verify round pushes its rows twice: process() mirrors the
    // batch, accept() re-pushes verify_h; a partially rejected round also
    // re-pushes its rejected tail). The window must cover ~2x a typical
    // word (mutated word plus its template tail) with margin, so the fix
    // widens the ring rather than masking the symptom.
    //
    // Sizing arithmetic (measured on the causal sweep): with the
    // ring_push() position-dedup below, one slot is allocated per DISTINCT
    // decoded position, so the ring holds the last RING_N positions of the
    // stream MINUS the phantom tail: the rejected rows of the FINAL verify
    // round are never re-decoded and keep occupying the newest ~n_max
    // slots beyond the stream end (measured extent 8 with n_max 7). Real
    // distinct-position coverage is therefore RING_N - ~n_max: 8 - 8 ~ 0-5
    // (the original bug), 16 - 8 ~ 8 (cold fallback still at delta >= 8),
    // 32 - 8 ~ 24 (covers deltas well past 14; >= 12 required). A no-dedup
    // ring does not help: re-pushes consume slots at ~2x the batch width
    // per round and low-acceptance tails measured only ~8 distinct
    // positions even at 32 slots. The same rollback is ALSO gated by the
    // target's recurrent snapshot depth (common.h need_n_rs_seq, floored
    // at N_RS_SEQ_FLOOR by this same triage): the effective window is the minimum of
    // the two. The serialized ring grows accordingly,
    // which bumps MTP_STATE_VERSION: pre-widening checkpoints (whose
    // newest-position ring entries were stale re-pushes) fail validation
    // and take the existing cold-recreate path.
    static constexpr uint32_t RING_N = 32;
    std::vector<std::vector<float>> ring_h;      // [n_seq][RING_N * n_embd]
    std::vector<std::vector<llama_pos>> ring_pos; // [n_seq][RING_N]
    std::vector<uint32_t> ring_len;              // [n_seq]
    std::vector<uint32_t> ring_head;             // [n_seq] next write slot

    // F4-boundary (run7 03/09): snapshot of the last get_state()-valid blob,
    // captured by draft_sync_reset() right before the partial-reject reset
    // clears the live state. A task whose final verify round is a partial
    // reject goes idle with pending_h_valid == 0; prompt_save must still
    // succeed or the slot holds the ONLY copy of the history (5 skipped
    // saves → full re-prefills). Cleared whenever the live state is restored
    // or fully invalidated (set_state) or positions shift (shift_state).
    std::vector<std::vector<uint8_t>> boundary_snapshot; // [n_seq]

    common_speculative_state_draft_mtp(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_MTP, n_seq)
        , params(params.draft)
    {
        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;
        GGML_ASSERT(ctx_tgt && ctx_dft && "MTP requires ctx_tgt and ctx_dft to be set");

        n_embd = llama_model_n_embd_pre_norm(llama_get_model(ctx_dft));
        n_mtp_layers = std::max(1, (int) llama_model_n_layer_nextn(llama_get_model(ctx_dft)));

        LOG_INF("%s: adding speculative implementation 'draft-mtp'\n", __func__);
        LOG_INF("%s: - n_max=%d, n_min=%d, p_min=%.2f, n_embd=%d, backend_sampling=%d\n", __func__, this->params.n_max, this->params.n_min, this->params.p_min, n_embd, (int) this->params.backend_sampling);
        LOG_INF("%s: - gpu_layers=%d, cache_k=%s, cache_v=%s, ctx_tgt=%s, ctx_dft=%s\n", __func__,
                this->params.n_gpu_layers,
                ggml_type_name(this->params.cache_type_k),
                ggml_type_name(this->params.cache_type_v),
                ctx_tgt ? "yes" : "no",
                ctx_dft ? "yes" : "no");

        const int32_t n_b = (int32_t) llama_n_batch(ctx_dft);
        batch = llama_batch_init(/*n_tokens=*/ n_b, /*embd=*/ n_embd, /*n_seq_max=*/ 1);
        // llama_batch_init allocates only one of token/embd; MTP needs both.
        // TODO: fix, how to call without malloc
        batch.token = (llama_token *) malloc(sizeof(llama_token) * n_b);

        smpls.resize(n_seq);
        for (auto & s : smpls) {
            common_params_sampling sparams;
            sparams.no_perf  = false;
            // Keep several candidates so --spec-draft-p-min has a meaningful
            // probability distribution while the draft loop still selects the
            // top candidate after sorting below.
            sparams.top_k    = MTP_DRAFT_TOP_K;
            sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
            s.reset(common_sampler_init(llama_get_model(ctx_dft), sparams));
        }

        // offload draft sampling to the backend
        backend_chains.assign(n_seq, nullptr);
        if (this->params.backend_sampling) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
                llama_sampler_chain_add(chain, llama_sampler_init_top_k(MTP_DRAFT_TOP_K));

                if (!llama_set_sampler(ctx_dft, seq_id, chain)) {
                    LOG_WRN("%s: backend offload failed for seq_id=%d; using CPU sampler\n", __func__, (int) seq_id);
                    llama_sampler_free(chain);
                    chain = nullptr;
                }
                backend_chains[seq_id] = chain;
            }
        }

        char arch_dft[64] = {};
        llama_model_meta_val_str(llama_get_model(ctx_dft), "general.architecture", arch_dft, sizeof(arch_dft));
        const bool full_hidden_rows =
            std::strcmp(arch_dft, "gemma4_assistant") == 0 ||
            std::strcmp(arch_dft, "gemma4-assistant") == 0;

        llama_set_embeddings_pre_norm(ctx_tgt, true, /*masked*/ false);
        llama_set_embeddings_pre_norm(ctx_dft, true, full_hidden_rows ? /*masked*/ false : /*masked*/ true);

        // Only Gemma4 assistants share the target KV cache. Embedded MTP
        // contexts also use ctx_other to read target hidden states, but keep
        // their own MTP-layer cache and must advance draft positions.
        is_mem_shared = full_hidden_rows && llama_get_ctx_other(ctx_dft) == ctx_tgt;
        chain_heads   = n_mtp_layers > 1 && !is_mem_shared;

        if (chain_heads) {
            this->params.n_max = std::min(this->params.n_max, n_mtp_layers);

            chain_h.assign(n_seq, {});
            for (auto & c : chain_h) {
                c.reserve((size_t) (this->params.n_max + 1) * n_embd);
            }
        }

        pending_h.assign(n_seq, std::vector<float>(n_embd, 0.0f));
        pending_h_prev.assign(n_seq, std::vector<float>(n_embd, 0.0f));
        pending_h_valid.assign(n_seq, 0);
        pending_h_prev_valid.assign(n_seq, 0);
        pending_h_pos.assign(n_seq, -1);
        pending_h_prev_pos.assign(n_seq, -1);

        ring_h.assign(n_seq, std::vector<float>((size_t) RING_N * n_embd, 0.0f));
        ring_pos.assign(n_seq, std::vector<llama_pos>(RING_N, -1));
        ring_len.assign(n_seq, 0);
        ring_head.assign(n_seq, 0);

        boundary_snapshot.assign(n_seq, std::vector<uint8_t>());

        process_boundary_h.assign(n_seq, std::vector<float>(n_embd, 0.0f));
        process_boundary_valid.assign(n_seq, 0);
        process_boundary_pos.assign(n_seq, -1);

        i_last.assign(n_seq, -1);
        i_batch_beg.assign(n_seq, -1);
        i_batch_end.assign(n_seq, -1);

        verify_h.assign(n_seq, {});
        for (auto & h : verify_h) {
            h.reserve((size_t) std::max<int32_t>(1, this->params.n_max) * n_embd);
        }
        verify_h_rows.assign(n_seq, 0);
        verify_pos_first.assign(n_seq, -1);

        last_n_drafted.assign(n_seq, 0);
        drafting.assign(n_seq, 0);
        suppress_draft.assign(n_seq, 0);
    }

    ~common_speculative_state_draft_mtp() override {
        auto * ctx_dft = this->params.ctx_dft;
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) backend_chains.size(); ++seq_id) {
            if (backend_chains[seq_id] == nullptr) {
                continue;
            }
            if (ctx_dft) {
                llama_set_sampler(ctx_dft, seq_id, nullptr);
            }
            llama_sampler_free(backend_chains[seq_id]);
        }
        backend_chains.clear();

        if (batch.token != nullptr) {
            free(batch.token);
            batch.token = nullptr;
        }
        llama_batch_free(batch);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        // W6-1 leva-2: a new generation never inherits the previous task's
        // one-shot F4 suppression (a task ending on a partial rejection would
        // otherwise burn a plain round at the start of the next one).
        if (seq_id >= 0 && seq_id < (llama_seq_id) n_seq) {
            suppress_draft[seq_id] = 0;
        }

        const int32_t N = (int32_t) prompt.size();
        if (N <= 0) {
            return;
        }
        auto * ctx_dft = this->params.ctx_dft;
        const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id);
        if (pos_max < N - 1) {
            LOG_WRN("%s: ctx_dft pos_max=%d < N-1=%d — "
                    "process() hook may not have run on every prefill ubatch "
                    "(need_embd / logits=1 on every prompt position?). "
                    "Drafts may degrade.\n",
                    __func__, (int) pos_max, N - 1);
        }
    }

    bool process(const llama_batch & batch_in) override {
        if (batch_in.n_tokens <= 0) {
            return true;
        }

        // TODO: how to make it work with vision tokens?
        if (batch_in.token == nullptr || batch_in.embd != nullptr) {
            return true;
        }

        const int32_t n_tokens = batch_in.n_tokens;

        // remember the first and last batch index for each sequence
        std::fill(i_batch_beg.begin(), i_batch_beg.end(), -1);
        std::fill(i_batch_end.begin(), i_batch_end.end(), -1);
        std::fill(verify_h_rows.begin(), verify_h_rows.end(), 0);
        std::fill(verify_pos_first.begin(), verify_pos_first.end(), -1);
        std::fill(process_boundary_valid.begin(), process_boundary_valid.end(), 0);
        std::fill(process_boundary_pos.begin(), process_boundary_pos.end(), -1);

        for (int k = 0; k < n_tokens; ++k) {
            GGML_ASSERT(batch_in.n_seq_id[k] == 1);

            const llama_seq_id seq_id = batch_in.seq_id[k][0];
            if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
                continue;
            }

            i_batch_end[seq_id] = k;
            if (i_batch_beg[seq_id] < 0) {
                i_batch_beg[seq_id] = k;
            }
        }

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;

        const size_t row_bytes = (size_t) n_embd * sizeof(float);

        common_batch_clear(batch);

        for (int k = 0; k < n_tokens; ++k) {
            common_speculative_batch_add_one_seq(batch, batch_in.token[k], batch_in.pos[k], batch_in.seq_id[k][0], false);
        }

        // shift the tgt embeddings to the right by one position
        // assumes that the tokens in the batch are sequential for each sequence
        // i.e. we cannot have seq_id like this: [0, 0, 0, 1, 1, 0, 1, 1]
        //                                                       ^--- this is a problem
        // TODO:this is generally true, but would be nice to assert it
        {
            const float * h_tgt = llama_get_embeddings_pre_norm(ctx_tgt);
            std::memcpy(batch.embd + (size_t) 1 * n_embd, h_tgt, row_bytes * (n_tokens-1));

            //{
            //    // string with seq_ids in the batch
            //    std::stringstream ss;
            //    for (int i = 0; i < n_tokens; ++i) {
            //        ss << batch_in.seq_id[i][0] << ",";
            //    }
            //    LOG_WRN("%s: batch_in.seq_id = %s\n", __func__, ss.str().c_str());
            //}
        }

        // fill the pending embeddings from a previous run
        auto set_h = [&](int idx, const float * h_row) {
            std::memcpy(batch.embd + (size_t) idx * n_embd, h_row, row_bytes);
        };

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_batch_beg[seq_id] < 0) {
                continue;
            }

            const llama_pos pos_needed = batch_in.pos[i_batch_beg[seq_id]] - 1;
            const float * h_boundary = nullptr;
            if (pending_h_valid[seq_id] && pending_h_pos[seq_id] == pos_needed) {
                h_boundary = pending_h[seq_id].data();
            } else if (pending_h_prev_valid[seq_id] && pending_h_prev_pos[seq_id] == pos_needed) {
                h_boundary = pending_h_prev[seq_id].data();
            } else if (pos_needed < 0) {
                // A new request can reach BOS without prompt_clear(), notably
                // through an explicit id_slot or with prompt caching disabled.
                // Never reuse the prior request's deferred MTP boundary there.
                std::fill(pending_h[seq_id].begin(), pending_h[seq_id].end(), 0.0f);
                std::fill(pending_h_prev[seq_id].begin(), pending_h_prev[seq_id].end(), 0.0f);
                pending_h_valid[seq_id] = 0;
                pending_h_prev_valid[seq_id] = 0;
                pending_h_pos[seq_id] = -1;
                pending_h_prev_pos[seq_id] = -1;
                h_boundary = pending_h[seq_id].data();
            } else {
                // The sequence advanced through positions this impl never observed, so no
                // boundary h was ever captured for pos_needed. This happens with mtmd/vision
                // chunks: process_chunk() decodes the image directly on ctx_tgt and ctx_dft
                // (see server-context.cpp, "maybe we simply need to call
                // common_speculative_process() on the mtmd batches"), so process() is never
                // called for those positions and pending_h stays at the last text token.
                //
                // Failing here aborts the server on the first image. Resync instead: use a
                // neutral boundary for this step -- the resulting draft is verified by the
                // target like any other and simply gets rejected, so output stays correct --
                // and let the capture at the end of this call record a valid pending_h so
                // speculation resumes normally on the next step. Costs one rejected draft
                // per image, not a crash.
                //
                // Dual-mode note: for a sequence routed to another drafter this is the
                // EXPECTED outcome of every partially-rejected verify round. process() runs
                // ungated on all sequences (prompt-cache invariant), captures pending_h at
                // the end of the drafted block, and the rollback of the rejected tail then
                // rewinds the sequence behind that capture. The neutral resync below costs
                // nothing here - this impl never drafts that sequence - so demote to debug
                // and only warn where a real gap (e.g. vision chunk) costs a rejected draft.
                const bool drafts_this_seq =
                    dparams_routing == nullptr ||
                    (*dparams_routing)[seq_id].drafter == COMMON_SPECULATIVE_TYPE_NONE ||
                    (*dparams_routing)[seq_id].drafter == type;

                if (drafts_this_seq) {
                    // PI F4 follow-up (29/08): with the partial-reject state reset
                    // (server-side, f4 2d9ca97e1) this fires on the round that follows
                    // every partial rejection - ~20% of draft rounds, routine by
                    // design. Demoted to debug; the /metrics counter
                    // spec_state_resets_total carries the frequency in production.
                    LOG_DBG("%s: MTP boundary missing for seq_id=%d pos=%d (current=%d/%d previous=%d/%d); "
                            "resyncing after a non-token batch (e.g. vision chunk)\n",
                            __func__, (int) seq_id, (int) pos_needed,
                            (int) pending_h_pos[seq_id], (int) pending_h_valid[seq_id],
                            (int) pending_h_prev_pos[seq_id], (int) pending_h_prev_valid[seq_id]);
                } else {
                    LOG_DBG("%s: MTP boundary ahead of the rolled-back position for seq_id=%d pos=%d "
                            "(current=%d/%d) - sequence routed to another drafter, neutral resync\n",
                            __func__, (int) seq_id, (int) pos_needed,
                            (int) pending_h_pos[seq_id], (int) pending_h_valid[seq_id]);
                }
                std::fill(pending_h[seq_id].begin(), pending_h[seq_id].end(), 0.0f);
                std::fill(pending_h_prev[seq_id].begin(), pending_h_prev[seq_id].end(), 0.0f);
                pending_h_valid[seq_id]      = 0;
                pending_h_prev_valid[seq_id] = 0;
                pending_h_pos[seq_id]        = -1;
                pending_h_prev_pos[seq_id]   = -1;
                h_boundary = pending_h[seq_id].data();
            }

            set_h(i_batch_beg[seq_id], h_boundary);
            if (pos_needed >= 0) {
                std::memcpy(process_boundary_h[seq_id].data(), h_boundary, row_bytes);
                process_boundary_valid[seq_id] = 1;
                process_boundary_pos[seq_id] = pos_needed;
            }
        }

        auto * mem_dft = llama_get_memory(ctx_dft);

        bool ok = true;
        for (int head = 0; head < n_mtp_layers; ++head) {
            if (chain_heads) {
                // ref: https://github.com/ggml-org/llama.cpp/pull/24340/changes#r3413498544
                for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                    if (i_batch_beg[seq_id] < 0) {
                        continue;
                    }
                    llama_memory_seq_rm(mem_dft, seq_id, batch_in.pos[i_batch_beg[seq_id]], -1);
                }
                llama_set_nextn_layer_offset(ctx_dft, head);
                llama_set_mtp_speculative_step(head);
            }

            const int32_t rc = llama_decode(ctx_dft, batch);
            if (rc != 0) {
                LOG_ERR("%s: llama_decode(ctx_dft) head=%d failed rc=%d (pos=%d)\n",
                        __func__, head, (int) rc, (int) batch_in.pos[0]);
                ok = false;
                break;
            }
        }

        if (chain_heads) {
            llama_set_nextn_layer_offset(ctx_dft, 0); // restore default for non-draft decodes
            llama_set_mtp_speculative_step(0);
        }
        if (!ok) {
            return false;
        }

        const float * h_tgt = llama_get_embeddings_pre_norm(ctx_tgt);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_batch_end[seq_id] < 0) {
                continue;
            }

            const int32_t n_rows = i_batch_end[seq_id] - i_batch_beg[seq_id] + 1;
            const float * h_seq = h_tgt + (size_t) i_batch_beg[seq_id] * n_embd;
            const llama_pos pos_last = batch_in.pos[i_batch_end[seq_id]];

            if (n_rows > 1) {
                const float * h_prev = h_seq + (size_t) (n_rows - 2) * n_embd;
                std::memcpy(pending_h_prev[seq_id].data(), h_prev, row_bytes);
                pending_h_prev_valid[seq_id] = 1;
                pending_h_prev_pos[seq_id] = batch_in.pos[i_batch_end[seq_id] - 1];
            } else if (process_boundary_valid[seq_id]) {
                std::memcpy(pending_h_prev[seq_id].data(), process_boundary_h[seq_id].data(), row_bytes);
                pending_h_prev_valid[seq_id] = 1;
                pending_h_prev_pos[seq_id] = process_boundary_pos[seq_id];
            } else {
                std::fill(pending_h_prev[seq_id].begin(), pending_h_prev[seq_id].end(), 0.0f);
                pending_h_prev_valid[seq_id] = 0;
                pending_h_prev_pos[seq_id] = -1;
            }

            // keep the last rows in the rollback ring at their contiguous
            // positions, so a bounded trailing rollback can find any of them
            for (int32_t i = 0; i < n_rows; ++i) {
                ring_push(seq_id, batch_in.pos[i_batch_beg[seq_id] + i], h_seq + (size_t) i * n_embd);
            }

            if (last_n_drafted[seq_id] == 0) {
                const float * h_last = h_seq + (size_t) (n_rows - 1) * n_embd;
                std::memcpy(pending_h[seq_id].data(), h_last, row_bytes);
                pending_h_valid[seq_id] = 1;
                pending_h_pos[seq_id] = pos_last;
                continue;
            }

            verify_h_rows[seq_id] = n_rows;
            verify_pos_first[seq_id] = batch_in.pos[i_batch_beg[seq_id]];
            const size_t n_verify_floats = (size_t) (n_rows - 1) * n_embd;
            if (verify_h[seq_id].size() < n_verify_floats) {
                verify_h[seq_id].resize(n_verify_floats);
            }
            if (n_verify_floats > 0) {
                std::memcpy(verify_h[seq_id].data(), h_seq, n_verify_floats * sizeof(float));
            }

            const float * h_last = h_seq + (size_t) (n_rows - 1) * n_embd;
            std::memcpy(pending_h[seq_id].data(), h_last, row_bytes);
            pending_h_valid[seq_id] = 1;
            pending_h_pos[seq_id] = pos_last;
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        common_batch_clear(batch);

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::fill(drafting.begin(), drafting.end(), 0);

        const size_t row_bytes = (size_t) n_embd * sizeof(float);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }

            last_n_drafted[seq_id] = 0;

            // W6-1 leva-2: F4 one-shot suppression after a partial rejection.
            // Skip drafting for exactly this round (the reset's measured H1
            // benefit - proposing from the just-rejected boundary is
            // counterproductive) but do NOT invalidate the boundary: the plain
            // round's process() mirror uses it, so the MTP KV row for the new
            // position is written with the real accepted h instead of the
            // neutral zero row that reset_seq_state() used to force here.
            if (suppress_draft[seq_id]) {
                suppress_draft[seq_id] = 0;

                SPC_DBG("F4 one-shot draft suppression (seq %d) - boundary kept at pos %d/%d\n",
                        (int) seq_id, (int) pending_h_pos[seq_id], (int) pending_h_valid[seq_id]);

                dp.drafting = false;
                continue;
            }

            // Draft positions must be in the target's RoPE space. They coincide
            // with n_past for text-only prompts; after an image chunk they do not,
            // and using n_past left pos_needed permanently offset so drafting was
            // disabled for the rest of the sequence.
            const llama_pos p_next = dp.pos_next >= 0 ? dp.pos_next : dp.n_past;

            const llama_pos pos_needed = p_next - 1;
            if (!pending_h_valid[seq_id] || pending_h_pos[seq_id] != pos_needed) {
                // t8 stadio 2 (spec §3/§10): for a sequence routed to draft-dflash
                // this fires on every concat round that follows a partial
                // rejection - accept() is dispatched to the round-closing arm only
                // (multi-impl dispatch is Task 5), so pending_h is not rewound to
                // the accepted boundary. The head is skipped and the harness
                // degrades the round to a plain draft-dflash block: every token is
                // still target-verified, so demote to debug - same treatment as the
                // neutral resync in process() for sequences routed elsewhere.
                const bool drafts_this_seq = dparams_routing == nullptr ||
                    (*dparams_routing)[seq_id].drafter == COMMON_SPECULATIVE_TYPE_NONE ||
                    (*dparams_routing)[seq_id].drafter == type;
                if (drafts_this_seq) {
                    // PI F4 follow-up (29/08): companion of the process() "MTP boundary
                    // missing" demotion above - fires on the round after every partial
                    // rejection while the reset state is re-synced. Debug level; the
                    // spec_state_resets_total counter observes it in production.
                    LOG_DBG("%s: disabling MTP draft for seq_id=%d: boundary pos=%d/%d, needed=%d\n",
                            __func__, (int) seq_id, (int) pending_h_pos[seq_id],
                            (int) pending_h_valid[seq_id], (int) pos_needed);
                } else {
                    LOG_DBG("%s: MTP boundary behind the accepted position for seq_id=%d - concat head skipped, plain dflash round (pos=%d/%d, needed=%d)\n",
                            __func__, (int) seq_id, (int) pending_h_pos[seq_id],
                            (int) pending_h_valid[seq_id], (int) pos_needed);
                }
                dp.drafting = false;
                continue;
            }

            n_drafting++;
            drafting[seq_id] = 1;
            common_sampler_reset(smpls[seq_id].get());

            common_speculative_batch_add_one_seq(batch, dp.id_last, p_next, seq_id, true);
            std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd, pending_h[seq_id].data(), row_bytes);

            i_last[seq_id] = batch.n_tokens - 1;

            if (chain_heads) {
                chain_h[seq_id].assign(pending_h[seq_id].begin(), pending_h[seq_id].end());
            }
        }

        if (n_drafting == 0) {
            return;
        }

        int i = 0;
        const bool log_debug = LOG_LEVEL_DEBUG <= common_log_get_verbosity_thold();

        while (n_drafting > 0) {
            // each step decodes under a different head, i.e. a different decoder layer, and
            // KV is per layer. process() filled this layer's KV only for positions < n_past
            // (prompt + accepted prefix) — nothing in the draft region yet. so reset the
            // draft region (the seq_rm lower bound is n_past, leaving the prompt KV intact)
            // and select head i so it rebuilds its own layer's KV there; decoding just the
            // latest token would leave its attention reading cells only another head wrote.
            if (chain_heads) {
                auto * mem_dft = llama_get_memory(ctx_dft);
                for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                    if (drafting[seq_id]) {
                        llama_memory_seq_rm(mem_dft, seq_id, dparams[seq_id].pos_next >= 0 ? dparams[seq_id].pos_next : dparams[seq_id].n_past, -1);
                    }
                }
                llama_set_nextn_layer_offset(ctx_dft, i);
            }
            llama_set_mtp_speculative_step(i);

            int ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                LOG_WRN("%s: llama_decode[%d] returned %d\n", __func__, i, ret);
                break;
            }

            // rebuild the batch for the next step: the growing-KV paths re-add only the
            // new token (the KV already holds the prefix), while chained heads re-add the
            // whole prefix at the next head. dropped sequences are simply not re-added.
            common_batch_clear(batch);

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (!drafting[seq_id]) {
                    continue;
                }

                auto & dp = dparams[seq_id];
                auto & result = *dp.result;
                const int32_t n_max = common_speculative_effective_n_max(params, dp);
                if (n_max <= (int) result.size()) {
                    drafting[seq_id] = 0;
                    n_drafting--;
                    continue;
                }

                auto * smpl = smpls[seq_id].get();

                const auto * cur_p = common_sampler_sample_top_k_probs(smpl, ctx_dft, i_last[seq_id], MTP_DRAFT_TOP_K);
                const float * h_row = llama_get_embeddings_pre_norm_ith(ctx_dft, i_last[seq_id]);

                if (log_debug) {
                    for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                        LOG_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                                seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                                common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                    }
                }

                // add drafted token for each sequence
                const llama_token id = cur_p->data[0].id;

                if (cur_p->data[0].p < common_speculative_effective_p_min(params, dp)) {
                    drafting[seq_id] = 0;
                    n_drafting--;
                    continue;
                }

                common_sampler_accept(smpl, id, true);

                result.push_back(id);

                if (n_max <= (int) result.size()) {
                    drafting[seq_id] = 0;
                    n_drafting--;
                    continue;
                }

                if (chain_heads) {
                    // ref: https://github.com/ggml-org/llama.cpp/pull/24340#discussion_r3448031546
                    chain_h[seq_id].insert(chain_h[seq_id].end(), h_row, h_row + n_embd);

                    const int n_rows = (int) result.size() + 1; // id_last + tokens drafted so far
                    for (int t = 0; t < n_rows; ++t) {
                        const llama_token tok = (t == 0) ? dp.id_last : result[t - 1];
                        common_speculative_batch_add_one_seq(batch, tok, (dp.pos_next >= 0 ? dp.pos_next : dp.n_past) + t, seq_id, t == n_rows - 1);
                        std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd,
                                    chain_h[seq_id].data() + (size_t) t * n_embd, row_bytes);
                    }
                } else if (is_mem_shared) {
                    // note: with shared memory (e.g. Gemma4 assistants) we use the same position for all draft tokens
                    // ref: https://github.com/huggingface/transformers/blob/effde20942e3f82a1b97449f60b3a48c5ff96145/docs/source/en/model_doc/gemma4_assistant.md?plain=1#L36-L37
                    common_speculative_batch_add_one_seq(batch, id, dp.pos_next >= 0 ? dp.pos_next : dp.n_past, seq_id, true);
                    std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd, h_row, row_bytes);
                } else {
                    common_speculative_batch_add_one_seq(batch, id, (dp.pos_next >= 0 ? dp.pos_next : dp.n_past) + i + 1, seq_id, true);
                    std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd, h_row, row_bytes);
                }

                i_last[seq_id] = batch.n_tokens - 1;
            }

            if (batch.n_tokens == 0) {
                break;
            }

            ++i;
        }

        if (chain_heads) {
            llama_set_nextn_layer_offset(ctx_dft, 0); // restore default for non-draft decodes
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            const int32_t n_max = common_speculative_effective_n_max(params, dp);
            const int32_t n_min = common_speculative_effective_n_min(params, dp, n_max);
            if (dp.result->size() < (size_t) n_min) {
                dp.result->clear();
            }

            last_n_drafted[seq_id] = (uint16_t) dp.result->size();
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        const int32_t n_rows = verify_h_rows[seq_id];
        if (n_rows <= 0) {
            last_n_drafted[seq_id] = 0;
            return;
        }

        const int32_t i_h = std::min<int32_t>(n_accepted, n_rows - 1);
        const size_t row_bytes = (size_t) n_embd * sizeof(float);

        // t8 stadio 2 (spec §3, Task 5): partial head prefix - this impl drafted
        // a concat head (last_n_drafted = k1') and the round verified more rows
        // than that (the closing arm's block), but fewer tokens than the head
        // were accepted: the boundary simply stops INSIDE the head at row i_h,
        // which pairs with the target's freshest sample, so the next draft()
        // resyncs from there exactly like a partially accepted mono-MTP round.
        // Debug log only (soft check): a partial acceptance is a normal
        // rejection, not an invariant violation.
        if (n_rows > 1 + last_n_drafted[seq_id] && n_accepted < last_n_drafted[seq_id]) {
            SPC_DBG("concat head partially accepted: n_accepted=%d < head k1'=%d (seq %d) - boundary at row %d/%d\n",
                    (int) n_accepted, (int) last_n_drafted[seq_id], (int) seq_id, (int) i_h, (int) n_rows);
        }
        if (i_h != n_rows - 1) {
            std::memcpy(pending_h[seq_id].data(), verify_h[seq_id].data() + (size_t) i_h * n_embd, row_bytes);
        }
        pending_h_valid[seq_id] = 1;
        pending_h_pos[seq_id] = verify_pos_first[seq_id] + i_h;

        // t8 stadio 2 (spec §3, Task 5): reconcile this impl's draft context with
        // the accepted boundary. A partially accepted round leaves the mirrored
        // verify rows of the rejected tail in ctx_dft. Invariant: this never
        // trims below pending_h_pos (the freshest valid row), so the next
        // process()/draft() always mirrors contiguously from there. In concat
        // rounds this is the ONLY trim of this context - the server's rollback
        // (server-context.cpp:4254-4257) trims the closing arm's context only,
        // and the harness's draft-phase trim of the other context is skipped by
        // any iteration that does not draft (e.g. the plain single-token decode
        // at n_predict exhaustion): without this trim the stale tail fails the
        // KV position contiguity check (Y = X + 1) and derails the request into
        // the error loop (dispatch-only image d3bdf3497f46 evidence in
        // logs/test-t8-concat/t5-dormancy-numbers.txt). Gemma4 assistants share
        // the draft context with the target: the server trim covers them, keep
        // out.
        if (!is_mem_shared) {
            if (!llama_memory_seq_rm(llama_get_memory(params.ctx_dft), seq_id, pending_h_pos[seq_id] + 1, -1)) {
                // t8: a silent failure here would reintroduce the KV-contiguity
                // runaway invisibly - warn loudly instead (no abort: the
                // aggressive common_context_seq_rm wrapper is not appropriate
                // for this best-effort reconciliation)
                SPC_WRN("draft-context trim to the accepted boundary failed (seq %d, pos=%d)\n",
                        (int) seq_id, (int) (pending_h_pos[seq_id] + 1));
            }
        }

        // all verification rows are valid boundary candidates at their
        // contiguous positions: keep them in the rollback ring.
        // verify_h mirrors only rows [0, n_rows - 1) - the last row lives
        // in pending_h (see process()) and was already pushed by process()
        // with the authoritative embedding. Pushing row n_rows - 1 here
        // read past the copied region and planted a stale duplicate of the
        // newest position that shadowed the authoritative row in ring_get
        // (which scans newest first), so bound the loop to the mirrored
        // rows.
        for (int32_t i = 0; i + 1 < n_rows; ++i) {
            ring_push(seq_id, verify_pos_first[seq_id] + i, verify_h[seq_id].data() + (size_t) i * n_embd);
        }

        if (i_h == 0) {
            if (process_boundary_valid[seq_id]) {
                std::memcpy(pending_h_prev[seq_id].data(), process_boundary_h[seq_id].data(), row_bytes);
                pending_h_prev_valid[seq_id] = 1;
                pending_h_prev_pos[seq_id] = process_boundary_pos[seq_id];
            } else {
                std::fill(pending_h_prev[seq_id].begin(), pending_h_prev[seq_id].end(), 0.0f);
                pending_h_prev_valid[seq_id] = 0;
                pending_h_prev_pos[seq_id] = -1;
            }
        } else {
            std::memcpy(pending_h_prev[seq_id].data(), verify_h[seq_id].data() + (size_t) (i_h - 1) * n_embd, row_bytes);
            pending_h_prev_valid[seq_id] = 1;
            pending_h_prev_pos[seq_id] = verify_pos_first[seq_id] + i_h - 1;
        }
        last_n_drafted[seq_id] = 0;
    }

    static constexpr uint32_t MTP_STATE_MAGIC       = 0x3250544d; // "MTP2" in little-endian byte order
    // v4: rollback ring widened 8 -> 32 rows with position-dedup (same
    // per-entry layout) - old blobs fail the version check below and take
    // the cold-recreate path
    static constexpr uint16_t MTP_STATE_VERSION     = 4;
    static constexpr uint16_t MTP_STATE_CURRENT     = 1u << 0;
    static constexpr uint16_t MTP_STATE_PREVIOUS    = 1u << 1;
    static constexpr size_t   MTP_STATE_HEADER      = sizeof(uint32_t) + 2*sizeof(uint16_t) + sizeof(uint32_t) + 2*sizeof(llama_pos);

    void ring_push(llama_seq_id seq_id, llama_pos pos, const float * row) {
        const size_t row_bytes = (size_t) n_embd * sizeof(float);

        // position-dedup: a verify round re-pushes rows that process() already
        // mirrored, and after a partial rejection the next round re-decodes the
        // rejected tail at the same positions with new content. Update the
        // existing slot in place so a re-push never consumes a slot: without
        // dedup RING_N slots hold only ~RING_N/2 (high acceptance) down to
        // ~RING_N/4 (low acceptance) distinct positions and the rollback
        // window shrinks with the acceptance rate (causal evidence 2026-08-21:
        // cold fallback at delta 8/10 with a 32-slot no-dedup ring in a
        // low-acceptance tail). With dedup RING_N is a guaranteed count of
        // DISTINCT positions.
        for (uint32_t i = 0; i < ring_len[seq_id]; ++i) {
            const uint32_t idx = (ring_head[seq_id] + RING_N - 1 - i) % RING_N;
            if (ring_pos[seq_id][idx] == pos) {
                std::memcpy(ring_h[seq_id].data() + (size_t) idx * n_embd, row, row_bytes);
                return;
            }
        }

        std::memcpy(ring_h[seq_id].data() + (size_t) ring_head[seq_id] * n_embd, row, row_bytes);
        ring_pos[seq_id][ring_head[seq_id]] = pos;
        ring_head[seq_id] = (ring_head[seq_id] + 1) % RING_N;
        ring_len[seq_id] = std::min(ring_len[seq_id] + 1, RING_N);
    }

    const float * ring_get(llama_seq_id seq_id, llama_pos pos) const {
        for (uint32_t i = 0; i < ring_len[seq_id]; ++i) {
            const uint32_t idx = (ring_head[seq_id] + RING_N - 1 - i) % RING_N;
            if (ring_pos[seq_id][idx] == pos) {
                return ring_h[seq_id].data() + (size_t) idx * n_embd;
            }
        }
        return nullptr;
    }

    bool rollback_state(llama_seq_id seq_id, llama_pos pos) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq || pos < 0) {
            return false;
        }

        const float * row = ring_get(seq_id, pos);
        if (row == nullptr) {
            return false;
        }

        const size_t row_bytes = (size_t) n_embd * sizeof(float);

        std::memcpy(pending_h[seq_id].data(), row, row_bytes);
        pending_h_valid[seq_id] = 1;
        pending_h_pos[seq_id] = pos;

        if (pos >= 1) {
            if (const float * prev = ring_get(seq_id, pos - 1)) {
                std::memcpy(pending_h_prev[seq_id].data(), prev, row_bytes);
                pending_h_prev_valid[seq_id] = 1;
                pending_h_prev_pos[seq_id] = pos - 1;
            } else {
                std::fill(pending_h_prev[seq_id].begin(), pending_h_prev[seq_id].end(), 0.0f);
                pending_h_prev_valid[seq_id] = 0;
                pending_h_prev_pos[seq_id] = -1;
            }
        } else {
            std::fill(pending_h_prev[seq_id].begin(), pending_h_prev[seq_id].end(), 0.0f);
            pending_h_prev_valid[seq_id] = 0;
            pending_h_prev_pos[seq_id] = -1;
        }

        // generation-time bookkeeping refers to positions beyond the rollback point
        process_boundary_valid[seq_id] = 0;
        process_boundary_pos[seq_id] = -1;
        verify_h[seq_id].clear();
        verify_h_rows[seq_id] = 0;
        verify_pos_first[seq_id] = -1;
        last_n_drafted[seq_id] = 0;

        // drop ring entries beyond the rollback point: their rows belong to
        // tokens that are about to be reprocessed with new content
        for (uint32_t i = 0; i < RING_N; ++i) {
            if (ring_pos[seq_id][i] > pos) {
                ring_pos[seq_id][i] = -1;
            }
        }

        return true;
    }

    void reset_seq_state(llama_seq_id seq_id) {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        std::fill(pending_h[seq_id].begin(), pending_h[seq_id].end(), 0.0f);
        std::fill(pending_h_prev[seq_id].begin(), pending_h_prev[seq_id].end(), 0.0f);
        pending_h_valid[seq_id] = 0;
        pending_h_prev_valid[seq_id] = 0;
        pending_h_pos[seq_id] = -1;
        pending_h_prev_pos[seq_id] = -1;
        std::fill(process_boundary_h[seq_id].begin(), process_boundary_h[seq_id].end(), 0.0f);
        process_boundary_valid[seq_id] = 0;
        process_boundary_pos[seq_id] = -1;
        verify_h[seq_id].clear();
        verify_h_rows[seq_id] = 0;
        verify_pos_first[seq_id] = -1;

        std::fill(ring_pos[seq_id].begin(), ring_pos[seq_id].end(), -1);
        ring_len[seq_id] = 0;
        ring_head[seq_id] = 0;
        last_n_drafted[seq_id] = 0;
        drafting[seq_id] = 0;
        // W6-1 leva-2: a restored/rewound state drafts normally - the one-shot
        // F4 suppression belongs to the live partial-reject flow only.
        suppress_draft[seq_id] = 0;
        i_last[seq_id] = -1;
        if (chain_heads) {
            chain_h[seq_id].clear();
        }
    }

    bool get_state(llama_seq_id seq_id, std::vector<uint8_t> & data) const override {
        data.clear();
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return false;
        }
        if (!pending_h_valid[seq_id]) {
            // F4 window: serve the snapshot captured at the partial-reject
            // reset (the accepted-boundary blob) — see boundary_snapshot
            if (!boundary_snapshot[seq_id].empty()) {
                data = boundary_snapshot[seq_id];
                return true;
            }
            return false;
        }

        const uint16_t flags = MTP_STATE_CURRENT |
            (pending_h_prev_valid[seq_id] ? MTP_STATE_PREVIOUS : 0);
        const uint32_t width = (uint32_t) n_embd;
        const size_t row_bytes = (size_t) n_embd * sizeof(float);

        // ring entries, oldest first
        std::vector<std::pair<llama_pos, const float *>> ring_entries;
        for (uint32_t i = 0; i < ring_len[seq_id]; ++i) {
            const uint32_t idx = (ring_head[seq_id] + RING_N - (ring_len[seq_id] - i)) % RING_N;
            if (ring_pos[seq_id][idx] >= 0) {
                ring_entries.emplace_back(ring_pos[seq_id][idx], ring_h[seq_id].data() + (size_t) idx * n_embd);
            }
        }

        const uint32_t ring_count = (uint32_t) ring_entries.size();

        data.resize(MTP_STATE_HEADER + 2*row_bytes + sizeof(uint32_t) + (size_t) ring_count * (sizeof(llama_pos) + row_bytes));

        size_t off = 0;
        std::memcpy(data.data() + off, &MTP_STATE_MAGIC,   sizeof(MTP_STATE_MAGIC));   off += sizeof(MTP_STATE_MAGIC);
        std::memcpy(data.data() + off, &MTP_STATE_VERSION, sizeof(MTP_STATE_VERSION)); off += sizeof(MTP_STATE_VERSION);
        std::memcpy(data.data() + off, &flags,             sizeof(flags));             off += sizeof(flags);
        std::memcpy(data.data() + off, &width,             sizeof(width));             off += sizeof(width);
        std::memcpy(data.data() + off, &pending_h_pos[seq_id],      sizeof(llama_pos)); off += sizeof(llama_pos);
        std::memcpy(data.data() + off, &pending_h_prev_pos[seq_id], sizeof(llama_pos)); off += sizeof(llama_pos);
        std::memcpy(data.data() + off, pending_h[seq_id].data(), row_bytes); off += row_bytes;
        std::memcpy(data.data() + off, pending_h_prev[seq_id].data(), row_bytes); off += row_bytes;

        std::memcpy(data.data() + off, &ring_count, sizeof(ring_count)); off += sizeof(ring_count);
        for (const auto & entry : ring_entries) {
            std::memcpy(data.data() + off, &entry.first, sizeof(llama_pos)); off += sizeof(llama_pos);
            std::memcpy(data.data() + off, entry.second, row_bytes); off += row_bytes;
        }
        return true;
    }

    bool set_state(llama_seq_id seq_id, const std::vector<uint8_t> & data) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return false;
        }

        // the live state becomes authoritative (restored) or the KV is gone
        // (empty data) — either way the F4-window snapshot no longer applies
        boundary_snapshot[seq_id].clear();

        reset_seq_state(seq_id);
        if (data.empty() || data.size() < MTP_STATE_HEADER) {
            return false;
        }

        uint32_t magic = 0;
        uint16_t version = 0;
        uint16_t flags = 0;
        uint32_t width = 0;
        llama_pos pos_current = -1;
        llama_pos pos_previous = -1;
        size_t off = 0;
        std::memcpy(&magic,   data.data() + off, sizeof(magic));   off += sizeof(magic);
        std::memcpy(&version, data.data() + off, sizeof(version)); off += sizeof(version);
        std::memcpy(&flags,   data.data() + off, sizeof(flags));   off += sizeof(flags);
        std::memcpy(&width,   data.data() + off, sizeof(width));   off += sizeof(width);
        std::memcpy(&pos_current,  data.data() + off, sizeof(pos_current));  off += sizeof(pos_current);
        std::memcpy(&pos_previous, data.data() + off, sizeof(pos_previous)); off += sizeof(pos_previous);

        const size_t row_bytes = (size_t) n_embd * sizeof(float);
        if (magic != MTP_STATE_MAGIC || version != MTP_STATE_VERSION ||
            (flags & MTP_STATE_CURRENT) == 0 || (flags & ~(MTP_STATE_CURRENT | MTP_STATE_PREVIOUS)) != 0 ||
            width != (uint32_t) n_embd || pos_current < 0 ||
            ((flags & MTP_STATE_PREVIOUS) != 0 && pos_previous < 0) ||
            data.size() < MTP_STATE_HEADER + 2*row_bytes + sizeof(uint32_t)) {
            return false;
        }

        std::memcpy(pending_h[seq_id].data(), data.data() + off, row_bytes); off += row_bytes;
        std::memcpy(pending_h_prev[seq_id].data(), data.data() + off, row_bytes); off += row_bytes;
        pending_h_valid[seq_id] = 1;
        pending_h_prev_valid[seq_id] = (flags & MTP_STATE_PREVIOUS) != 0;
        pending_h_pos[seq_id] = pos_current;
        pending_h_prev_pos[seq_id] = pending_h_prev_valid[seq_id] ? pos_previous : -1;

        // v3: trailing ring of recent boundary rows, oldest first
        // v4: ring capacity 32 with position-dedup (window widening)
        uint32_t ring_count = 0;
        std::memcpy(&ring_count, data.data() + off, sizeof(ring_count)); off += sizeof(ring_count);

        if (ring_count > RING_N ||
            data.size() != off + (size_t) ring_count * (sizeof(llama_pos) + row_bytes)) {
            return false;
        }

        ring_len[seq_id] = 0;
        ring_head[seq_id] = 0;
        for (uint32_t i = 0; i < ring_count; ++i) {
            llama_pos pos = -1;
            std::memcpy(&pos, data.data() + off, sizeof(pos)); off += sizeof(pos);
            std::memcpy(ring_h[seq_id].data() + (size_t) i * n_embd, data.data() + off, row_bytes); off += row_bytes;

            ring_pos[seq_id][i] = pos;
            ring_len[seq_id] = i + 1;
        }
        ring_head[seq_id] = ring_count % RING_N;
        return true;
    }

    bool state_required() const override {
        return true;
    }

    void shift_state(llama_seq_id seq_id, llama_pos delta) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq || delta == 0) {
            return;
        }

        // the snapshot's embedded positions cannot be rewritten in place —
        // drop it (a save in this window skips, as before the split)
        boundary_snapshot[seq_id].clear();

        if (pending_h_valid[seq_id]) {
            pending_h_pos[seq_id] += delta;
        }
        if (pending_h_prev_valid[seq_id]) {
            pending_h_prev_pos[seq_id] += delta;
        }
        if (process_boundary_valid[seq_id]) {
            process_boundary_pos[seq_id] += delta;
        }
        if (verify_h_rows[seq_id] > 0 && verify_pos_first[seq_id] >= 0) {
            verify_pos_first[seq_id] += delta;
        }
        for (auto & p : ring_pos[seq_id]) {
            if (p >= 0) {
                p += delta;
            }
        }
    }

    // F4 (+ W6-1 leva-2 amendment): suppress the next round's drafting - the
    // measured H1 intervention (bench 29/08: proposing from the just-rejected
    // boundary raised p0-reject 0.272 vs 0.099 base; suppression 0.161) - while
    // keeping the cache boundary available: snapshot the accepted-boundary blob
    // (the live state right after accept() is coherent with it). The amendment:
    // the full reset_seq_state() that used to run here also invalidated
    // pending_h, which forced the next process() into the NEUTRAL resync - the
    // plain round's mirror row was then written with a zero h and permanently
    // poisoned one MTP KV cell per partial rejection (the unexplained residual
    // excess of the F4 bench). accept() has already rewound pending_h to the
    // accepted boundary, so keeping it valid lets that same mirror row carry
    // the real h: same round structure as the measured fix, no state
    // corruption. The boundary re-arms on the plain round's process() and
    // get_state serves fresh data again (the snapshot only covers the window
    // where the task ends before the plain round runs).
    void draft_sync_reset(llama_seq_id seq_id) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        std::vector<uint8_t> snap;
        if (get_state(seq_id, snap)) {
            boundary_snapshot[seq_id] = std::move(snap);
        }

        suppress_draft[seq_id] = 1;
    }

    bool need_embd() const override {
        return false;
    }

    bool need_embd_pre_norm() const override {
        return true;
    }

    int32_t draft_n_max() const override {
        return std::max(0, params.n_max);
    }
};

// state of self-speculation (simple implementation, not ngram-map)
struct common_speculative_impl_ngram_simple : public common_speculative_impl {
    common_params_speculative_ngram_map params;

    // shared across all sequences
    common_ngram_simple_config config;

    common_speculative_impl_ngram_simple(
            const common_params_speculative & params, uint32_t n_seq,
            common_ngram_simple_config config)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, n_seq)
        , params(params.ngram_simple)
        , config(config) {}

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            *dp.result = common_ngram_simple_draft(config, *dp.prompt, dp.id_last);
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/) override {
        // noop
    }

    bool need_embd() const override {
        return false;
    }
};

struct common_speculative_impl_ngram_map_k : public common_speculative_impl {
    common_params_speculative_ngram_map params;

    // n_seq configs
    std::vector<common_ngram_map> config;

    common_speculative_impl_ngram_map_k(
            const common_params_speculative & params,
            const common_ngram_map & config,
            uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K, n_seq)
        , params(params.ngram_map_k) {
        for (uint32_t i = 0; i < n_seq; i++) {
            this->config.push_back(config);
        }
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        GGML_ASSERT(seq_id < (llama_seq_id) n_seq);

        common_ngram_map_begin(config[seq_id], prompt);
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            common_ngram_map_draft(config[seq_id], *dp.prompt, dp.id_last, *dp.result);
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted) override {
        GGML_ASSERT((seq_id < (llama_seq_id) config.size()));

        common_ngram_map_accept(config[seq_id], n_accepted);
    }

    bool need_embd() const override {
        return false;
    }
};

struct common_speculative_impl_ngram_mod : public common_speculative_impl {
    common_params_speculative_ngram_mod params;

    // shared across all sequences
    common_ngram_mod mod;

    // enable trace logging if LLAMA_TRACE is set
    const bool verbose;

    struct seq_info {
        // the last position in the prompt that was added to the ngram container
        size_t i_last = 0;

        // length of the last drafted n‑gram (number of tokens returned by draft)
        size_t n_draft_last = 0;

        // consecutive accept rounds with low acceptance fraction (< 0.5)
        int n_low = 0;
    };

    std::vector<seq_info> sinfos;

    common_speculative_impl_ngram_mod(
            const common_params_speculative & params,
            uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_MOD, n_seq)
        , params(params.ngram_mod)
        , mod(params.ngram_mod.n_match, 4*1024*1024)
        , verbose(std::getenv("LLAMA_TRACE") != nullptr) {
        static_assert(sizeof(llama_token) == sizeof(common_ngram_mod::entry_t));

        LOG_INF("%s: initialized ngram_mod with n_match=%d, size=%zu (%.3f MB)\n", __func__,
                this->params.n_match, mod.size(), (float)(mod.size_bytes())/1024/1024);

        if (this->params.n_match < 16) {
            LOG_WRN("%s: ngram_mod n_match=%d is too small - poor quality is possible, "
                    "see: https://github.com/ggml-org/llama.cpp/pull/19164\n", __func__, this->params.n_match);
        }

        sinfos.resize(n_seq);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        auto & sinfo = sinfos[seq_id];

        sinfo.i_last = 0;
        sinfo.n_draft_last = 0;

        const size_t n = mod.get_n();
        if (prompt.size() < n) {
            return;
        }

        for (size_t i = 0; i < prompt.size() - n; ++i) {
            mod.add(prompt.data() + i);
        }

        sinfo.i_last = prompt.size() - n;

        const double f = (double)mod.get_used() / (double)mod.size();
        LOG_INF("%s: ngram_mod occupancy = %zu/%zu (%.2f)\n", __func__, mod.get_used(), mod.size(), f);

        constexpr double f_thold = 0.25;
        if (f > f_thold) {
            LOG_WRN("%s: ngram_mod occupancy %.2f exceeds threshold (%.2f) - resetting\n", __func__, f, f_thold);

            mod.reset();
        }
    }

    void draft_one(
            llama_seq_id seq_id,
            common_speculative_draft_params & dparams) {
        auto & sinfo = sinfos[seq_id];
        auto & result = *dparams.result;

        const auto & prompt = *dparams.prompt;

        sinfo.n_draft_last = 0;

        const size_t cur_len = prompt.size();
        if (cur_len < mod.get_n()) {
            return;
        }

        const size_t n = mod.get_n();

        // add new ngrams in chunks
        if (sinfo.i_last + 32 < cur_len) {
            for (size_t i = sinfo.i_last; i < cur_len - n; ++i) {
                mod.add(prompt.data() + i);
            }

            sinfo.i_last = cur_len - n;
        }

        result.resize(n + params.n_max);
        for (size_t i = 0; i < n - 1; ++i) {
            result[i] = prompt.at(cur_len - n + 1 + i);
        }
        result[n - 1] = dparams.id_last;

        for (int i = 0; i < params.n_max; ++i) {
            const llama_token token = mod.get(result.data() + i);
            if (token == common_ngram_mod::EMPTY) {
                if (i < params.n_min) {
                    result.clear();
                    return;
                }

                result.resize(n + i);
                break;
            }
            result[n + i] = token;
        }

        // only return the m tokens that were drafted
        for (size_t i = 0; n + i < result.size(); ++i) {
            result[i] = result[n + i];
        }
        result.resize(result.size() - n);

        // store length of drafted n‑gram for later acceptance analysis
        sinfo.n_draft_last = result.size();
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            draft_one(seq_id, dp);
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted) override {
        auto & sinfo = sinfos[seq_id];

        // compute acceptance fraction if we have a recorded draft length
        if (sinfo.n_draft_last > 0) {
            const double f_acc = (double)n_accepted / (double)sinfo.n_draft_last;
            if (f_acc < 0.5) {
                sinfo.n_low++;
                if (sinfo.n_low >= 3) {
                    if (verbose) {
                        LOG_WRN("%s: low acceptance streak (%d) – resetting ngram_mod\n", __func__, sinfo.n_low);
                    }

                    mod.reset();
                    sinfo.n_low = 0;
                    sinfo.i_last = 0;
                }
            } else {
                sinfo.n_low = 0;
            }
        }
    }

    bool need_embd() const override {
        return false;
    }
};

struct common_speculative_impl_ngram_cache : public common_speculative_impl {
    common_params_speculative_ngram_cache params;

    uint16_t n_draft;

    bool save_dynamic;
    bool save_static;

    struct seq_info {
        size_t cache_size = 0; // number of tokens in n-gram cache

        common_ngram_cache ngram_cache_context;
        common_ngram_cache ngram_cache_dynamic;
        common_ngram_cache ngram_cache_static;
    };

    std::vector<seq_info> sinfos;

    common_speculative_impl_ngram_cache(
            const common_params_speculative & params,
            uint32_t n_seq,
            uint16_t n_draft,
            const std::string & path_static,
            const std::string & path_dynamic,
            bool save_dynamic,
            bool save_static)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE, n_seq)
        , params(params.ngram_cache)
        , n_draft(n_draft)
        , save_dynamic(save_dynamic)
        , save_static(save_static)
    {
        sinfos.resize(n_seq);

        if (!path_static.empty()) {
            try {
                auto ngram_cache_static = common_ngram_cache_load(path_static);

                for (auto & sinfo : sinfos) {
                    sinfo.ngram_cache_static = ngram_cache_static;
                }
            } catch (...) {
                LOG_ERR("failed to open static lookup cache: %s", path_static.c_str());
                GGML_ABORT("Couldn't read static lookup cache");
            }
        }

        if (!path_dynamic.empty()) {
            try {
                auto ngram_cache_dynamic = common_ngram_cache_load(path_dynamic);

                for (auto & sinfo : sinfos) {
                    sinfo.ngram_cache_dynamic = ngram_cache_dynamic;
                }
            } catch (...) {
                LOG_ERR("failed to open dynamic lookup cache: %s", path_dynamic.c_str());
                GGML_ABORT("Couldn't read dynamic lookup cache");
            }
        }
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    void draft_one(
            llama_seq_id seq_id,
            common_speculative_draft_params & dparams) {
        auto & sinfo = sinfos[seq_id];
        auto & result = *dparams.result;

        const auto & prompt = *dparams.prompt;

        if (sinfo.cache_size < prompt.size() + 1) {
            llama_tokens tokens_new;
            tokens_new.reserve(prompt.size() + 1 - sinfo.cache_size);
            for (size_t j = sinfo.cache_size; j < prompt.size(); ++j) {
                tokens_new.push_back(prompt[j]);
            }
            tokens_new.push_back(dparams.id_last); // add the last token

            // Update context ngram cache with new dparams.prompt:
            common_ngram_cache_update(
                    sinfo.ngram_cache_context,
                    LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                    tokens_new, tokens_new.size(), false);
            sinfo.cache_size = prompt.size() + 1;
        }

        llama_tokens inp;
        inp.reserve(prompt.size() + 1);
        for (size_t j = 0; j < prompt.size(); ++j) {
            inp.push_back(prompt[j]);
        }
        inp.push_back(dparams.id_last);

        result.push_back(dparams.id_last);

        common_ngram_cache_draft(
                inp, result, n_draft, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                sinfo.ngram_cache_context,
                sinfo.ngram_cache_dynamic,
                sinfo.ngram_cache_static);

        if (result.size() > 0) {
            // delete first token in result (which is the id_last token)
            result.erase(result.begin());
        }
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            draft_one(seq_id, dp);
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/) override {
        // noop
    }

    bool need_embd() const override {
        return false;
    }
};

struct common_speculative {
    common_speculative_draft_params_vec dparams;

    // list of implementations to use and their states
    std::vector<std::unique_ptr<common_speculative_impl>> impls;

    // which implementaion was used for a given seq_id
    std::vector<common_speculative_impl *> impl_last;

    // t8 stadio 2 (spec §3, Task 5): the implementation that drafted the head of
    // the concat round in flight for each seq (nullptr outside concat rounds).
    // Set when the head is published in common_speculative_draft(), consumed and
    // cleared by common_speculative_accept() to dispatch accept() to every round
    // contributor - and cleared wholesale at the top of the next draft() call so
    // a round that is never verified (lost round) cannot leak its contributor.
    std::vector<common_speculative_impl *> impl_head;

    // t8 stadio 2 (spec §3): concat round configuration (--spec-concat-k1, 0 = off
    // - every concat branch below is gated on it and k1 = 0 boots run zero new
    // code) and the per-seq storage backing common_speculative_draft_params::
    // concat_head during the round in flight.
    int32_t concat_k1 = 0;
    std::vector<llama_tokens> concat_head;

    // diagnostics: number of concat rounds composed (first one logs at INFO)
    size_t n_concat_rounds = 0;

    // t8 stadio 2 (spec §4, Task 6): the temp > 0 single-drafter fallback
    // warning is logged once per session (server lifetime), not per request
    bool concat_temp_fallback_warned = false;

    // t8 branch-2 (spec §3): pattern-window exclusion for the concat round
    // (--spec-concat-exclusion, default enabled). Inert unless concat rounds
    // are armed: the gate lives inside the concat branch of
    // common_speculative_draft(), so a k1 = 0 boot runs none of it
    // (structural inertia; the bit-identical certification is gate G0 of the
    // branch-2 plan).
    bool concat_exclusion = true;

    // t8 branch-2 (spec §3): cumulative exclusion counters - TELEMETRY ONLY:
    // they feed no decision and no mechanism state (zero hysteresis by
    // design) and are read as per-request deltas like the impl counters.
    // total counts the rounds the gate evaluated (concat-eligible windows),
    // gated the rounds routed down the plain path; both stay zero while the
    // exclusion flag is off (the gate does not run at all).
    size_t n_concat_excl_total = 0;
    size_t n_concat_excl_gated = 0;
};

static common_ngram_map get_common_ngram_map(
        common_speculative_type type,
        const common_params_speculative_ngram_map & config) {
    uint16_t size_key   = config.size_n;
    uint16_t size_value = config.size_m;
    bool     key_only   = type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K;
    uint16_t min_hits   = config.min_hits;

    return common_ngram_map(size_key, size_value, key_only, min_hits);
}

static common_speculative_impl_ngram_cache create_state_ngram_cache(
        const common_speculative_config & config,
        uint32_t n_seq,
        const std::string & path_static,
        const std::string & path_dynamic) {
    uint16_t n_draft = 8; // TODO get from config?

    // TODO bool param in common/common.h to set save_static/save_dynamic?
    bool save_static = false;
    bool save_dynamic = false;

    common_speculative_impl_ngram_cache state(config.params, n_seq, n_draft, path_static, path_dynamic, save_static, save_dynamic);

    return state;
}

std::string common_speculative_type_name_str(const std::vector<common_speculative_type> & types) {
    std::string result;

    for (size_t i = 0; i < types.size(); i++) {
        if (i > 0) {
            result += ",";
        }
        result += common_speculative_type_to_str(types[i]);
    }
    return result;
}

const char * common_speculative_all_types_str() {
    static std::string all_types_str = []() {
        std::vector<common_speculative_type> types;
        types.reserve(COMMON_SPECULATIVE_TYPE_COUNT);
        for (int i = 0; i < COMMON_SPECULATIVE_TYPE_COUNT; i++) {
            types.push_back((common_speculative_type) i);
        }
        return common_speculative_type_name_str(types);
    }();
    return all_types_str.c_str();
}

std::string common_speculative_type_to_str(common_speculative_type type) {
    switch (type) {
        case COMMON_SPECULATIVE_TYPE_NONE:          return "none";
        case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE:  return "draft-simple";
        case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3:  return "draft-eagle3";
        case COMMON_SPECULATIVE_TYPE_DRAFT_MTP:     return "draft-mtp";
        case COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH:  return "draft-dflash";
        case COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK:  return "draft-dspark";
        case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:  return "ngram-simple";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:   return "ngram-map-k";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: return "ngram-map-k4v";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:     return "ngram-mod";
        case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:   return "ngram-cache";
        default:                                    return "unknown";
    }
}

std::vector<common_speculative_type> common_speculative_types_from_names(const std::vector<std::string> & names) {
    std::vector<common_speculative_type> types;
    types.reserve(names.size());

    for (const auto & name : names) {
        auto type = common_speculative_type_from_name_map.find(name);
        if (type != common_speculative_type_from_name_map.end()) {
            if (type->second == COMMON_SPECULATIVE_TYPE_NONE) {
                return std::vector<common_speculative_type> { COMMON_SPECULATIVE_TYPE_NONE };
            }
            types.push_back(type->second);
            continue;
        }
        throw std::invalid_argument("unknown speculative type: " + name);
    }

    return types;
}

common_speculative_type common_speculative_type_from_name(const std::string & name) {
    const auto it = common_speculative_type_from_name_map.find(name);
    if (it == common_speculative_type_from_name_map.end()) {
        return COMMON_SPECULATIVE_TYPE_COUNT;
    }
    return it->second;
}

static uint32_t common_get_enabled_speculative_configs(const std::vector<common_speculative_type> & configs) {
    uint32_t result = 0;
    for (size_t i = 0; i < configs.size(); i++) {
        result |= (1u << configs[i]);
    }
    return result;
}

int32_t common_speculative_n_max(const common_params_speculative * spec) {
    int32_t n_max = 0;

    for (const auto type : spec->types) {
        switch (type) {
            case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE:
            case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3:
            case COMMON_SPECULATIVE_TYPE_DRAFT_MTP:
            case COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH:
            case COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK:
                n_max = std::max(n_max, std::max(0, spec->draft.n_max));
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:
                n_max = std::max(n_max, (int32_t) spec->ngram_simple.size_m);
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:
                n_max = std::max(n_max, (int32_t) spec->ngram_map_k.size_m);
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V:
                n_max = std::max(n_max, (int32_t) spec->ngram_map_k4v.size_m);
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:
                n_max = std::max(n_max, std::max(0, spec->ngram_mod.n_max));
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:
                n_max = std::max(n_max, (int32_t) 8);
                break;
            case COMMON_SPECULATIVE_TYPE_NONE:
            case COMMON_SPECULATIVE_TYPE_COUNT:
                break;
        }
    }

    return n_max;
}

common_params common_base_params_to_speculative(const common_params & params) {
    const bool has_draft = params.speculative.has_dft();

    const auto & params_spec = params.speculative.draft;
    common_params result = params;

    if (has_draft) {
        result.devices               = params_spec.devices;
        result.model                 = params_spec.mparams;
        result.n_gpu_layers          = params_spec.n_gpu_layers;
        result.tensor_buft_overrides = params_spec.tensor_buft_overrides;

        if (params_spec.cpuparams.n_threads > 0) {
            result.cpuparams.n_threads       = params_spec.cpuparams.n_threads;
            result.cpuparams_batch.n_threads = params_spec.cpuparams_batch.n_threads;
        }
    }

    result.cache_type_k = params_spec.cache_type_k;
    result.cache_type_v = params_spec.cache_type_v;

    // dflash/dspark decode every sequence's full noise block in one pass
    // TODO: refactor such properties to be announced by the speculative types
    //       something like `struct common_speculative_type_props common_speculative_type_get_props(...);`
    const bool has_block_draft = std::any_of(
        params.speculative.types.begin(), params.speculative.types.end(),
        [](common_speculative_type t) {
            return t == COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH || t == COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK;
        });
    if (has_block_draft) {
        // per-seq output positions: DFlash decodes anchor + n_max masks (n_max + 1); DSpark n_max -> +1 covers both
        const int32_t per_seq       = std::max(1, params_spec.n_max + 1);
        const int32_t n_outputs_max = params.n_parallel * per_seq;
        result.n_batch  = std::max(result.n_batch,  n_outputs_max);
        result.n_ubatch = std::max(result.n_ubatch, n_outputs_max);
    }

    return result;
}

common_speculative_output_limits common_speculative_get_output_limits(
        int32_t n_batch, int32_t n_parallel, int32_t n_draft) {
    const int64_t per_seq = 1 + (int64_t) std::max(0, n_draft);
    const int64_t total   = (int64_t) n_parallel * per_seq;

    return {
        /* .total   = */ (int32_t) std::min<int64_t>(n_batch, total),
        /* .per_seq = */ (int32_t) std::min<int64_t>(n_batch, per_seq),
    };
}

// initialization of the speculative decoding system
//
common_speculative * common_speculative_init(common_params_speculative & params, uint32_t n_seq) {
    // Compute the implementations to use based on the config and their order of preference
    std::vector<common_speculative_config> configs = {}; // list of speculative configs to try
    {
        uint32_t enabled_configs = common_get_enabled_speculative_configs(params.types);

        bool has_draft_model_path = !params.draft.mparams.path.empty();

        // dual mode: the MTP implementation runs on its own context (on the target
        // model), while ctx_dft stays bound to the external draft model (e.g. DFlash);
        // in legacy mono-MTP mode ctx_dft_mtp is not set and ctx_dft is used instead
        llama_context * ctx_dft_mtp = params.draft.ctx_dft_mtp != nullptr ? params.draft.ctx_dft_mtp : params.draft.ctx_dft;

        bool has_draft_simple = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE));
        bool has_draft_eagle3 = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3)) && params.draft.ctx_dft != nullptr;
        bool has_draft_mtp    = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_DRAFT_MTP))    && ctx_dft_mtp != nullptr;
        bool has_draft_dflash = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH)) && params.draft.ctx_dft != nullptr;
        bool has_draft_dspark = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK)) && params.draft.ctx_dft != nullptr;



        bool has_ngram_cache   = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_NGRAM_CACHE));
        bool has_ngram_simple  = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE));
        bool has_ngram_map_k   = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K));
        bool has_ngram_map_k4v = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V));
        bool has_ngram_mod     = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_NGRAM_MOD));

        // when adding a new type - update here the logic above
        static_assert(COMMON_SPECULATIVE_TYPE_COUNT == 11);

        // this list here defines the priority of the speculators
        // the one with highest priority are listed first
        if (has_ngram_simple) {
            // This implementation can guess a lot of tokens without any draft model.
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, params));
        }
        if (has_ngram_map_k) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K, params));
        }
        if (has_ngram_map_k4v) {
            // This implementation can guess tokens with high acceptance rate but is more expensive.
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V, params));
        }
        if (has_ngram_mod) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MOD, params));
        }
        if (has_ngram_cache) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE, params));
        }
        if (has_draft_simple) {
            if (!has_draft_model_path) {
                LOG_WRN("%s: draft model is not specified - cannot use 'draft' type\n", __func__);
                has_draft_simple = false;
            }
        } else if (has_draft_model_path && !has_draft_mtp && !has_draft_eagle3 && !has_draft_dflash && !has_draft_dspark) {
            LOG_WRN("%s: draft model is specified but 'draft' speculative type is not explicitly enabled - enabling it\n", __func__);
            has_draft_simple = true;
        }

        if (has_draft_simple) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE, params));
        }
        if (has_draft_eagle3) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3, params));
        }
        if (has_draft_mtp) {
            // the MTP implementation always runs on the MTP context (its own in dual
            // mode, ctx_dft in legacy mono mode), never on the external draft model
            common_speculative_config config_mtp(COMMON_SPECULATIVE_TYPE_DRAFT_MTP, params);
            config_mtp.params.draft.ctx_dft = ctx_dft_mtp;
            // spec-route (spec §3.1 T4): in dual mode the global --spec-draft-n-max
            // is sized on the largest drafter (7); the MTP arm runs at its production
            // optimum instead. The ctor's chain_heads clamp does not apply to
            // single-head nextn models (e.g. Qwen3.8), so clamp here - dual mode
            // only, mono keeps honoring the requested value verbatim.
            //
            // dual_mtp_n_max: production-optimal MTP draft size for the dual
            // configuration, measured on Qwen3.8-27B in the T7 A/B benchmarks
            // (spec §1: "MTP n6", 19.6-21.0 tok/s on prose) - the value the dual
            // boot marker of spec §6 reports as well.
            constexpr int32_t dual_mtp_n_max = 6;
            if (has_draft_dflash || has_draft_dspark || has_draft_eagle3) {
                config_mtp.params.draft.n_max = std::min(config_mtp.params.draft.n_max, dual_mtp_n_max);
            }
            configs.push_back(std::move(config_mtp));
        }
        if (has_draft_dflash) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH, params));
        }
        if (has_draft_dspark) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK, params));
        }
    }

    std::vector<std::unique_ptr<common_speculative_impl>> impls = {};

    for (const common_speculative_config & config : configs) {
        LOG_INF("%s: adding speculative implementation '%s'\n", __func__, common_speculative_type_to_str(config.type).c_str());
        try {
            switch (config.type) {
            case COMMON_SPECULATIVE_TYPE_NONE:
                break;
            case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_simple>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_eagle3>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_MTP: {
                impls.push_back(std::make_unique<common_speculative_state_draft_mtp>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_dflash>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_dflash>(
                        config.params, n_seq, COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE: {
                common_ngram_map ngram_map = get_common_ngram_map(config.type, config.params.ngram_simple);

                uint16_t ngram_size_key   = ngram_map.size_key;
                uint16_t mgram_size_value = ngram_map.size_value;

                auto config_simple = common_ngram_simple_config {
                    /* .size_ngram = */ ngram_size_key,
                    /* .size_mgram = */ mgram_size_value
                };
                auto state = std::make_unique<common_speculative_impl_ngram_simple>(
                    /* .params = */ config.params,
                    /* .n_seq  = */ n_seq,
                    /* .state  = */ config_simple
                );
                impls.push_back(std::move(state));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_map_k>(
                            config.params, get_common_ngram_map(config.type, config.params.ngram_map_k), n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MOD: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_mod>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE: {
                auto state = create_state_ngram_cache(
                        config, n_seq,
                        params.ngram_cache.lookup_cache_static,
                        params.ngram_cache.lookup_cache_dynamic);
                impls.push_back(std::make_unique<common_speculative_impl_ngram_cache>(state));
                break;
            }
            default:
                break;
            }
        } catch (const std::exception & err) {
            LOG_WRN("%s: failed to initialize speculative implementation '%s': %s\n",
                    __func__, common_speculative_type_to_str(config.type).c_str(), err.what());
        }
    }

    if (impls.empty()) {
        LOG_WRN("%s: no implementations specified for speculative decoding\n", __func__);
        return nullptr;
    }

    auto * result = new common_speculative {
        /* .dparams   = */ common_speculative_draft_params_vec(n_seq),
        /* .impls     = */ std::move(impls),
        /* .impl_last = */ std::vector<common_speculative_impl *>(n_seq, nullptr)
    };

    // t8 stadio 2 (spec §3): wire the concat configuration into the harness. The
    // implementations keep receiving only their own draft params (they must not
    // observe concat state), so the head tokens travel via dparams::concat_head,
    // backed by this per-seq storage.
    result->concat_k1 = params.concat_k1;
    result->concat_exclusion = params.concat_exclusion;
    result->concat_head.assign(n_seq, {});
    result->impl_head.assign(n_seq, nullptr);
    if (result->concat_k1 > 0) {
        // gate the "armed" wording on the same requested-types check as the server
        // boot marker (common_params_speculative::concat_armed()) - with a mono or
        // otherwise non-dual boot this branch must not claim the mode is armed
        // while the server logs "concat mode disabled" (quality review: the two
        // lines said the opposite of each other)
        if (params.concat_armed()) {
            LOG_INF("%s: concat mode armed, k1=%d (per-request routing picks the rounds)\n", __func__, result->concat_k1);
        } else {
            LOG_INF("%s: concat k1=%d accepted but the requested speculative types are not the dual draft-mtp,draft-dflash routing - concat stays inert\n",
                    __func__, result->concat_k1);
        }
    }

    return result;
}

void common_speculative_free(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    delete spec;
}

common_speculative_draft_params & common_speculative_get_draft_params(
        common_speculative * spec,
        llama_seq_id seq_id) {
    GGML_ASSERT(spec);
    GGML_ASSERT(seq_id < (llama_seq_id) spec->dparams.size());

    return spec->dparams[seq_id];
}

std::vector<common_speculative_type> common_speculative_types(const common_speculative * spec) {
    if (spec == nullptr) {
        return {};
    }

    std::vector<common_speculative_type> types;
    types.reserve(spec->impls.size());
    for (const auto & impl : spec->impls) {
        types.push_back(impl->type);
    }
    return types;
}

int32_t common_speculative_n_max_type(const common_speculative * spec, common_speculative_type type) {
    if (spec == nullptr) {
        return -1;
    }

    for (const auto & impl : spec->impls) {
        if (impl->type == type) {
            return impl->draft_n_max();
        }
    }

    return -1;
}

static bool common_speculative_routing_active(const common_speculative_draft_params_vec & dparams) {
    return std::any_of(dparams.begin(), dparams.end(),
            [](const common_speculative_draft_params & dp) {
                return dp.drafter != COMMON_SPECULATIVE_TYPE_NONE;
            });
}

void common_speculative_begin(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & prompt) {
    if (spec == nullptr) {
        return;
    }

    const bool routing_active = common_speculative_routing_active(spec->dparams);

    for (auto & impl : spec->impls) {
        impl->dparams_routing = routing_active ? &spec->dparams : nullptr;

        common_time_meas tm(impl->t_begin_us, !impl->gen_perf);
        impl->begin(seq_id, prompt);
        impl->n_call_begin++;
    }
}

bool common_speculative_process(common_speculative * spec, const llama_batch & batch) {
    bool result = true;

    if (spec == nullptr) {
        return result;
    }

    // per-seq drafter routing (see common_speculative_draft_params::drafter): while
    // active, the implementations that mirror the target batch into their own draft
    // context (DFlash/DSpark) skip the sequences drafted by another implementation.
    // note: MTP's process() must observe every sequence of the batch, including the
    // ones routed to another drafter: it captures the per-seq boundary that the
    // server relies on when saving the prompt cache - gating it per-seq would
    // silently drop cache entries for the routed-away sequences
    const bool routing_active = common_speculative_routing_active(spec->dparams);

    for (auto & impl : spec->impls) {
        impl->dparams_routing = routing_active ? &spec->dparams : nullptr;

        result = result && impl->process(batch);
    }

    return result;
}

bool common_speculative_need_embd(common_speculative * spec) {
    if (spec == nullptr) {
        return false;
    }

    for (auto & impl : spec->impls) {
        if (impl->need_embd()) {
            return true;
        }
    }

    return false;
}

bool common_speculative_need_embd_pre_norm(common_speculative * spec) {
    if (spec == nullptr) {
        return false;
    }

    for (auto & impl : spec->impls) {
        if (impl->need_embd_pre_norm()) {
            return true;
        }
    }

    return false;
}

void common_speculative_draft(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    auto & dparams = spec->dparams;

    {
        int n_drafting = 0;

        // t8 concat (Task 5): a new round starts - drop any head contributor
        // left over from a round that was never accepted (lost round). A
        // replayed round between draft() and accept() also loses its head
        // contributor this way: benign dormancy, resync at the next full
        // acceptance.
        std::fill(spec->impl_head.begin(), spec->impl_head.end(), nullptr);

        for (auto & dp : dparams) {
            GGML_ASSERT(!dp.drafting || dp.result->empty());
            // t8 concat: never leak a head pointer across rounds
            dp.concat_head = nullptr;

            if (dp.drafting) {
                n_drafting++;
            }
        }

        if (n_drafting == 0) {
            return;
        }
    }

    // t8 stadio 2 (spec §3): concat round support. In a dual MTP+DFlash routing
    // armed with concat_k1 > 0, a sequence routed to draft-dflash composes its
    // round as an MTP head of up to k1 tokens followed by the draft-dflash block
    // conditioned on that head (the head tokens sit at positions n_past+1..n_past+k1'
    // of the dflash drafting input, where the mask placeholder used to be). This
    // is the one deliberate break of the first-wins chaining below: the MTP arm
    // runs first for those sequences but does NOT close the round - the
    // draft-dflash arm completes and closes it, so impl_last attributes the round
    // to draft-dflash exactly as in the T7 routing.
    common_speculative_impl * impl_mtp = nullptr;
    common_speculative_impl * impl_dflash = nullptr;
    for (auto & impl : spec->impls) {
        if (impl->type == COMMON_SPECULATIVE_TYPE_DRAFT_MTP) {
            impl_mtp = impl.get();
        }
        if (impl->type == COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH) {
            impl_dflash = impl.get();
        }
    }
    // degenerate guard: with a draft-dflash arm that cannot draft at all (n_max
    // clamped to 0 post-load) the head would close head-only rounds attributed to
    // dflash - keep the plain T7 routing instead (concat inert).
    // NB deliberately NOT named concat_armed(): this adds impl existence and
    // draft_n_max() > 0 on top of common_params_speculative::concat_armed()
    // (requested types only) - different semantics, different name
    const bool concat_rounds_ready = spec->concat_k1 > 0 && impl_mtp != nullptr &&
            impl_dflash != nullptr && impl_dflash->draft_n_max() > 0;

    for (auto & impl : spec->impls) {
        // per-seq drafter routing (see common_speculative_draft_params::drafter):
        // a sequence with an active drafter is drafted only by the matching
        // implementation. The `drafting` flag is the only per-seq input that the
        // implementations consume, so the routing is applied by stashing the flag
        // of the sequences owned by another implementation for the duration of the
        // call and restoring it right after - the chaining logic below operates on
        // the true flags and is unaffected. With no routing (NONE on every seq)
        // the behavior is identical to before.
        int n_drafting_impl = 0;
        std::vector<llama_seq_id> seq_routed;

        // t8 concat: sequences of this MTP-arm iteration that must draft the
        // round's head (spec §3) - kept drafting (NOT stashed away like the
        // routed ones) with dp.n_max temporarily clamped to k1 so the MTP arm
        // produces at most the head
        std::vector<llama_seq_id> seq_concat;
        std::vector<int32_t>      seq_concat_n_max;
        const bool concat_mtp_phase = concat_rounds_ready && impl.get() == impl_mtp;

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) dparams.size(); ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }

            if (dp.drafter == COMMON_SPECULATIVE_TYPE_NONE || dp.drafter == impl->type) {
                n_drafting_impl++;
            } else if (concat_mtp_phase && dp.drafter == COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH) {
                // t8 stadio 2 (spec §4, Task 6): temp > 0 single-drafter fallback,
                // BEFORE any composition. A composed round mixes MTP head rows
                // (which carry no proposal distribution) with DFlash noise rows
                // (which do), so spec-dists acceptance is impossible on the mixed
                // segment - the server's size-guard would silently degrade every
                // such round to exact accept. Route the sequence to the class
                // drafter instead: a plain draft-dflash round, exactly the
                // concat_off path, whose dists cover the whole draft so sample
                // acceptance works at temp > 0. WARNING once per session (spec §4).
                if (dp.temperature > 0.0f) {
                    if (!spec->concat_temp_fallback_warned) {
                        spec->concat_temp_fallback_warned = true;
                        SPC_WRN("concat mode fallback: temperature > 0 request (seq %d) - spec-dists acceptance is impossible on mixed MTP+DFlash segments; using single-drafter (class) rounds for every temp > 0 request - this warning is logged once per session\n",
                                (int) seq_id);
                    }
                    seq_routed.push_back(seq_id);
                } else {
                    // t8 branch-2 (spec §3): pattern-window exclusion gate - it runs
                    // BEFORE the round becomes a concat round (the head is published,
                    // and its injection armed, only through seq_concat below). The
                    // detector reads the last W confirmed tokens ending at the anchor
                    // (dp.id_last @ dp.n_past, wiring map section 4.2): the drafting
                    // prompt tail by pointer plus the anchor token, no per-round copy
                    // of the slot buffer. A gated (pattern-like / copy-mode) window
                    // routes the sequence down the PLAIN path - the exact k1 = 0
                    // behavior for this round: stashed away from this MTP arm like
                    // any other foreign-drafter sequence (the same mechanism as the
                    // temp > 0 fallback above), so the draft-dflash arm drafts it
                    // from the mask placeholder (dp.concat_head stays null: no head
                    // injection, no "concat round composed" marker, no impl_head
                    // attribution, no n_concat_rounds bump). Ungated windows compose
                    // the concat round unchanged. --no-spec-concat-exclusion skips
                    // the gate entirely (A-noexclusion test arm). The counters below
                    // are TELEMETRY ONLY (spec §3): they feed no decision and no
                    // mechanism state.
                    bool excl_round_gated = false;
                    if (spec->concat_exclusion && dp.prompt != nullptr) {
                        const auto & prompt = *dp.prompt;
                        const auto gres = spec_concat_exclusion::gating(
                                prompt.data(), prompt.size(), dp.id_last);
                        spec->n_concat_excl_total++;
                        if (gres.gated) {
                            excl_round_gated = true;
                            spec->n_concat_excl_gated++;
                            SPC_INF("spec-concat: gated p=%d segnale=%s (seq %d)\n",
                                    gres.p_best, gres.segnale, (int) seq_id);
                        }
                    }

                    if (excl_round_gated) {
                        // plain round for this sequence in this round (k1 = 0 path)
                        seq_routed.push_back(seq_id);
                    } else {
                        // concat round: the MTP arm heads it. The arm's effective n_max
                        // already folds dp.n_max in, so clamping dp.n_max to k1 caps the
                        // head at k1 tokens; the original per-request value is restored
                        // right after the call, like the drafting stash below.
                        seq_concat.push_back(seq_id);
                        seq_concat_n_max.push_back(dp.n_max);
                        dp.n_max = dp.n_max >= 0 ? std::min(dp.n_max, spec->concat_k1) : spec->concat_k1;
                        n_drafting_impl++;
                    }
                }
            } else {
                seq_routed.push_back(seq_id);
            }
        }

        if (n_drafting_impl == 0) {
            // every drafting sequence is routed to another implementation
            continue;
        }

        for (const llama_seq_id seq_id : seq_routed) {
            dparams[seq_id].drafting = false;
        }

        {
            common_time_meas tm(impl->t_draft_us, !impl->gen_perf);
            impl->draft(dparams);
            impl->n_call_draft++;
        }

        for (const llama_seq_id seq_id : seq_routed) {
            dparams[seq_id].drafting = true;
        }

        // t8 concat: restore the per-request n_max and publish the head. With a
        // head of k1' > 0 tokens the round stays open - the draft-dflash arm below
        // consumes dp.concat_head and closes the round (the first-wins close is
        // skipped for this sequence in this iteration). With no head (p_min early
        // stop or a boundary miss) the round degrades to a plain draft-dflash
        // round, which requires re-arming drafting here because the MTP arm
        // disables the sequence on a boundary miss (spec §10: never a lost round).
        for (size_t i = 0; i < seq_concat.size(); ++i) {
            auto & dp = dparams[seq_concat[i]];

            dp.n_max       = seq_concat_n_max[i];
            dp.concat_head = nullptr;

            if (!dp.result->empty()) {
                spec->concat_head[seq_concat[i]] = *dp.result;
                dp.concat_head = &spec->concat_head[seq_concat[i]];
                // t8 concat (Task 5): record the head contributor - accept()
                // will dispatch to it as well (see common_speculative_accept)
                spec->impl_head[seq_concat[i]] = impl_mtp;

                spec->n_concat_rounds++;
                if (spec->n_concat_rounds == 1) {
                    SPC_INF("concat round composed, k1=%d (head=%zu tokens, seq %d)\n",
                            spec->concat_k1, dp.result->size(), (int) seq_concat[i]);
                } else {
                    SPC_TRC("concat round composed, k1=%d (head=%zu tokens, seq %d)\n",
                            spec->concat_k1, dp.result->size(), (int) seq_concat[i]);
                }
            } else {
                dp.drafting = true;
            }
        }

        int n_drafting = 0;

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) dparams.size(); ++seq_id) {
            auto & dp = dparams[seq_id];

            auto & result = *dp.result;

            // a new draft has been sampled
            if (dp.drafting && !result.empty()) {
                // t8 concat: while a head is stashed the round stays open - it is
                // conditioning, not a completed round - and ONLY the draft-dflash
                // arm may close it (spec §3: impl_last attributes the round to the
                // closing arm). Closing on "any other impl" instead would depend
                // on the init priority list keeping MTP first: with the arms
                // reordered (or an intermediate arm drafting the sequence), the
                // head would be closed - and accept() dispatched - to an arm that
                // never drafted it. With this form a reordered list simply leaves
                // the round unclosed here and concat degrades to plain rounds.
                if (dp.concat_head == nullptr || impl.get() == impl_dflash) {
                    dp.drafting = false;

                    // t8 concat (spec §10): the round-level n_max is per-SEQ and
                    // counts the whole round - the k1' head tokens drafted by the
                    // MTP arm sit on top of the closing arm's own budget, so the
                    // shared-result truncate extends the cap by the head size
                    // (zero extra on every non-concat round: identical behavior)
                    const int32_t n_max_round = dp.n_max + (dp.concat_head ? (int32_t) dp.concat_head->size() : 0);
                    if (dp.n_max >= 0 && (int) result.size() > n_max_round) {
                        LOG_DBG("%s: truncating draft to %d tokens\n", __func__, n_max_round);
                        result.resize(n_max_round);
                    }

                    // t8 concat: the round is closed - release the head plumbing
                    dp.concat_head = nullptr;

                    if (!result.empty()) {
                        LOG_DBG("%s: called impl %s, hist size = %zu, call_count = %zu, gen = %zu\n", __func__,
                                common_speculative_type_to_str(impl.get()->type).c_str(), dp.prompt->size(),
                                impl.get()->n_call_draft, result.size());

                        // remember which implementation was used
                        spec->impl_last[seq_id] = impl.get();

                        impl->n_gen_drafts++;
                        impl->n_gen_tokens += result.size();
                    }
                }
            }

            if (dp.drafting) {
                n_drafting++;
            }
        }

        if (n_drafting == 0) {
            break;
        }
    }

    // these sequences failed to generate a draft
    for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) dparams.size(); ++seq_id) {
        auto & dp = dparams[seq_id];

        if (dp.drafting) {
            dp.drafting = false;
        }

        // t8 concat: never leak a head pointer across rounds
        dp.concat_head = nullptr;
    }
}

void common_speculative_accept(common_speculative * spec, llama_seq_id seq_id, uint16_t n_accepted) {
    common_speculative_impl * impl = spec->impl_last[seq_id];

    GGML_ASSERT(impl);

    // t8 stadio 2 (spec §3, Task 5 - fixes the former accept-dispatch TODO): in a
    // concat round the draft is composed by more than one implementation, and every
    // contributor with per-seq boundary state must also "see" the accepted
    // tokens - without this dispatch the head drafter's deferred boundary (MTP
    // pending_h / eagle3 pending_g_last) stays at the end of the verify batch
    // and goes stale after every partial rejection (deferred limit / get_state /
    // cache-checkpoint desync, R2). The closing arm keeps the whole accounting
    // (impl_last attribution, spec §3 "Statistics per-impl"); the head
    // contributor receives the same TOTAL n_accepted and reconciles its
    // boundary positionally: MTP computes pending = verify_pos_first +
    // min(n_accepted, n_rows - 1), which is correct for any acceptance value
    // including a partial head prefix (n_accepted < k1', boundary inside the
    // head). The dispatch is per-round (impl_head is set only while this seq's
    // round was composed), so implementations whose accept() consumes their own
    // draft (e.g. the ngram maps) never receive another implementation's round.
    {
        common_time_meas tm(impl->t_accept_us, !impl->gen_perf);

        if (impl->n_acc_tokens_per_pos.size() < n_accepted) {
            impl->n_acc_tokens_per_pos.resize(n_accepted, 0);
        }

        for (size_t i = 0; i < n_accepted; ++i) {
            impl->n_acc_tokens_per_pos[i]++;
        }

        if (n_accepted > 0) {
            impl->n_acc_drafts++;
            impl->n_acc_tokens += n_accepted;
        }

        impl->accept(seq_id, n_accepted);
        impl->n_call_accept++;
    }

    common_speculative_impl * impl_contrib = spec->impl_head[seq_id];
    spec->impl_head[seq_id] = nullptr;

    if (impl_contrib != nullptr && impl_contrib != impl) {
        common_time_meas tm(impl_contrib->t_accept_us, !impl_contrib->gen_perf);

        impl_contrib->accept(seq_id, n_accepted);
        impl_contrib->n_call_accept++;
    }
}

int32_t common_speculative_concat_head_size(const common_speculative * spec, llama_seq_id seq_id) {
    if (spec == nullptr || spec->concat_k1 <= 0) {
        return 0;
    }

    if (seq_id < 0 || seq_id >= (llama_seq_id) spec->impl_head.size()) {
        return 0;
    }

    // impl_head is non-null only while this sequence's round was composed as a
    // concat round: set when the head is published in common_speculative_draft(),
    // consumed here by accept() and cleared wholesale at the top of the next
    // draft() call. It intentionally survives the server's checkpoint-replay
    // path (no draft() runs between the restore and the replayed accept), where
    // the replayed round keeps its original composition attribution.
    if (spec->impl_head[seq_id] == nullptr) {
        return 0;
    }

    return (int32_t) spec->concat_head[seq_id].size();
}

// t20 f3 (pi-stack): impl_last[seq_id] is set when a round closes in
// common_speculative_draft() (first non-empty draft of the closing arm) and is
// NOT cleared by common_speculative_accept() - only the in-flight impl_head
// attribution is - so this readout keeps naming the drafter of the round the
// server is about to close (or has just closed) with that accept call.
enum common_speculative_type common_speculative_round_drafter_type(const common_speculative * spec, llama_seq_id seq_id) {
    if (spec == nullptr || seq_id < 0 || seq_id >= (llama_seq_id) spec->impl_last.size()) {
        return COMMON_SPECULATIVE_TYPE_NONE;
    }

    if (spec->impl_last[seq_id] == nullptr) {
        return COMMON_SPECULATIVE_TYPE_NONE;
    }

    return spec->impl_last[seq_id]->type;
}

int32_t common_speculative_concat_k1(const common_speculative * spec) {
    return spec ? spec->concat_k1 : 0;
}

// TODO: support the case of more than one speculative implementations having a state
bool common_speculative_get_state(common_speculative * spec, llama_seq_id seq_id, std::vector<uint8_t> & data) {
    if (spec == nullptr) {
        return false;
    }

    for (auto & impl : spec->impls) {
        if (impl->get_state(seq_id, data)) {
            return true;
        }
    }

    return false;
}

bool common_speculative_set_state(common_speculative * spec, llama_seq_id seq_id, const std::vector<uint8_t> & data) {
    if (spec == nullptr) {
        return true;
    }

    bool ok = true;
    for (auto & impl : spec->impls) {
        const bool restored = impl->set_state(seq_id, data);
        ok = ok && (!impl->state_required() || restored);
    }

    if (seq_id >= 0 && seq_id < (llama_seq_id) spec->dparams.size()) {
        spec->dparams[seq_id].drafting = false;
    }

    return ok;
}

void common_speculative_draft_sync_reset(common_speculative * spec, llama_seq_id seq_id) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        impl->draft_sync_reset(seq_id);
    }

    if (seq_id >= 0 && seq_id < (llama_seq_id) spec->dparams.size()) {
        spec->dparams[seq_id].drafting = false;
    }
}

bool common_speculative_state_required(const common_speculative * spec) {
    if (spec == nullptr) {
        return false;
    }

    for (const auto & impl : spec->impls) {
        if (impl->state_required()) {
            return true;
        }
    }

    return false;
}

bool common_speculative_rollback_state(common_speculative * spec, llama_seq_id seq_id, llama_pos pos) {
    if (spec == nullptr) {
        return false;
    }

    bool ok = true;

    for (auto & impl : spec->impls) {
        const bool rolled = impl->rollback_state(seq_id, pos);
        ok = ok && (!impl->state_required() || rolled);
    }

    return ok;
}

void common_speculative_shift_state(common_speculative * spec, llama_seq_id seq_id, llama_pos delta) {
    if (spec == nullptr || delta == 0) {
        return;
    }

    for (auto & impl : spec->impls) {
        impl->shift_state(seq_id, delta);
    }
}

void common_speculative_print_stats(const common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    for (const auto & impl : spec->impls) {
        std::string str_perf;
        if (impl->gen_perf) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3) << impl->t_begin_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_draft_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_accept_us / 1000.0;
            str_perf = ", dur(b,g,a) = " + oss.str() + " ms";
        } else {
            str_perf = "";
        }

        std::string str_stats;
        if (impl->n_call_accept > 0) {
            const double mean =
                1.0 + (double) impl->n_acc_tokens / (double) impl->n_call_accept;
            std::ostringstream tmp;
            tmp << std::fixed << std::setprecision(3);
            for (size_t i = 0; i < impl->n_acc_tokens_per_pos.size(); ++i) {
                if (i > 0) {
                    tmp << ", ";
                }
                tmp << (double) impl->n_acc_tokens_per_pos[i] / (double) impl->n_call_accept;
            }
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2) << mean;
            str_stats = ", #mean acc len = " + oss.str() + ", #acc rate/pos = (" + tmp.str() + ")";
        }

        LOG_INF("statistics %16s: #calls(b,g,a) = %4zu %6zu %6zu, #gen drafts = %6zu, #acc drafts = %5zu, #gen tokens = %6zu, #acc tokens = %5zu%s%s\n",
                common_speculative_type_to_str(impl->type).c_str(),
                impl->n_call_begin, impl->n_call_draft, impl->n_call_accept,
                impl->n_gen_drafts,
                impl->n_acc_drafts,
                impl->n_gen_tokens,
                impl->n_acc_tokens,
                str_stats.c_str(),
                str_perf.c_str());
    }

    // t8 branch-2 (spec §3): cumulative concat-exclusion telemetry, printed
    // next to the per-impl counters above. TELEMETRY ONLY - never a decision
    // input, never mechanism state (zero hysteresis by design); read as
    // per-request deltas like the impl counters. total = rounds the gate
    // evaluated (concat-eligible windows), gated = rounds sent down the plain
    // path. Double-gated on the same condition the counters accumulate under:
    // k1 > 0 (a k1 = 0 boot never runs the gate and must not see new
    // statistics output) AND the exclusion flag on - with the flag off the
    // gate never runs, the counters stay zero and the line is not printed
    // (quality review: do not report empty exclusion telemetry).
    if (spec->concat_k1 > 0 && spec->concat_exclusion) {
        LOG_INF("statistics %16s: #excl rounds = %6zu, #gated rounds = %5zu\n",
                "concat-exclusion",
                spec->n_concat_excl_total,
                spec->n_concat_excl_gated);
    }
}
