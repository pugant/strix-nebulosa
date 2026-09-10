#pragma once

#include "common.h"
#include "llama.h"

#include <string>
#include <unordered_set>
#include <list>
#include <map>

// TODO: prevent including the whole server-common.h as we only use server_tokens
#include "server-common.h"

using json = nlohmann::ordered_json;

enum server_task_type {
    SERVER_TASK_TYPE_COMPLETION,
    SERVER_TASK_TYPE_EMBEDDING,
    SERVER_TASK_TYPE_RERANK,
    SERVER_TASK_TYPE_INFILL,
    SERVER_TASK_TYPE_CANCEL,
    SERVER_TASK_TYPE_NEXT_RESPONSE,
    SERVER_TASK_TYPE_METRICS,
    SERVER_TASK_TYPE_SLOT_SAVE,
    SERVER_TASK_TYPE_SLOT_RESTORE,
    SERVER_TASK_TYPE_SLOT_ERASE,
    SERVER_TASK_TYPE_GET_LORA,
    SERVER_TASK_TYPE_SET_LORA,
};

// TODO: change this to more generic "response_format" to replace the "format_response_*" in server-common
enum task_response_type {
    TASK_RESPONSE_TYPE_NONE, // llama.cpp native format
    TASK_RESPONSE_TYPE_OAI_CHAT,
    TASK_RESPONSE_TYPE_OAI_CMPL,
    TASK_RESPONSE_TYPE_OAI_RESP,
    TASK_RESPONSE_TYPE_OAI_ASR, // transcriptions API
    TASK_RESPONSE_TYPE_OAI_EMBD,
    TASK_RESPONSE_TYPE_ANTHROPIC,
};

enum stop_type {
    STOP_TYPE_NONE,
    STOP_TYPE_EOS,
    STOP_TYPE_WORD,
    STOP_TYPE_LIMIT,
};

struct task_params {
    bool stream          = true;
    bool include_usage   = false;
    bool cache_prompt    = true; // remember the prompt to avoid reprocessing all prompt
    bool return_tokens   = false;
    bool return_progress = false;

    int32_t n_keep    =  0; // number of tokens to keep from initial prompt
    int32_t n_discard =  0; // number of tokens after n_keep that may be discarded when shifting context, 0 defaults to half
    int32_t n_predict = -1; // new tokens to predict
    int32_t n_indent  =  0; // minimum line indentation for the generated text in number of whitespace characters
    int32_t n_cmpl    =  1; // number of completions to generate from this prompt

    int32_t n_cache_reuse = 0; // min chunk size to attempt reusing from the cache via KV shifting (0 = disabled)

    int64_t t_max_prompt_ms  = -1; // TODO: implement
    int64_t t_max_predict_ms = -1; // if positive, limit the generation phase to this time limit

    std::map<int, float> lora; // mapping adapter ID -> scale

    std::vector<std::string> antiprompt;
    std::vector<std::string> response_fields;

    bool timings_per_token   = false;
    bool post_sampling_probs = false;

    struct common_params_sampling sampling;
    struct common_params_speculative speculative;

    // spec-route: per-request drafter selection (spec §3.1 T5)
    // -1 = no override and no tools signal: the server routing policy applies
    // (mono: identity; dual: default MTP). Otherwise a resolved
    // COMMON_SPECULATIVE_TYPE_DRAFT_* - either the explicit "spec_drafter" body
    // override or the tools/tool_choice signal policy (-> DFLASH).
    int spec_drafter = -1;

    // spec-route: true only when spec_drafter came from a valid explicit
    // "spec_drafter" body override - gates the no-silent-fallback rejection
    // (spec §5). The tools-signal policy never sets it.
    bool spec_drafter_is_override = false;

    // spec-route: why spec_drafter got its value - "tools" (policy signal),
    // "none" (no signal) or "override:<val>" (explicit body override).
    // Purely diagnostic: consumed by logging only, never by control flow.
    std::string spec_drafter_signal;

    // T32-b park: this task belongs to the main agent. Set ONLY by the caller
    // (server-context.cpp) from the X-Pi-Role HTTP header at the completions
    // convergence point; never read from the request JSON, so a client body
    // with "park_main": true has no effect (spec §4).
    bool park_main = false;

    // response formatting
    bool               verbose  = false;
    task_response_type res_type = TASK_RESPONSE_TYPE_NONE;
    std::string        oaicompat_model;
    std::string        oaicompat_cmpl_id;

    // per-request parameters for chat parsing
    common_chat_parser_params chat_parser_params;

    // Embeddings
    int32_t embd_normalize = 2; // (-1=none, 0=max absolute int16, 1=taxicab, 2=Euclidean/L2, >2=p-norm)

    json format_logit_bias(const std::vector<llama_logit_bias> & logit_bias) const;
    json to_json(bool only_metrics = false) const;
};

// struct for tracking the state of a task (e.g., for streaming)
struct task_result_state {
    // tracking diffs for partial tool calls
    std::vector<common_chat_msg_diff> diffs;
    common_chat_parser_params chat_parser_params;
    common_chat_msg chat_msg;
    std::string generated_text; // append new chunks of generated text here
    std::vector<std::string> generated_tool_call_ids;
    std::unordered_set<size_t> sent_tool_call_names;

    // for OpenAI Responses and Anthropic streaming API:
    // track output item / content block state across chunks
    bool thinking_block_started = false;
    bool text_block_started = false;

    // for OpenAI Responses streaming API
    const std::string oai_resp_id;
    const std::string oai_resp_reasoning_id;
    const std::string oai_resp_message_id;
    std::string oai_resp_fc_id; // function call ID for current args delta

    task_result_state(const common_chat_parser_params & chat_parser_params)
        : chat_parser_params(chat_parser_params)
        , oai_resp_id("resp_" + random_string())
        , oai_resp_reasoning_id("rs_" + random_string())
        , oai_resp_message_id("msg_" + random_string()) {}

    // parse partial tool calls and update the internal state
    common_chat_msg update_chat_msg(
        const std::string & text_added,
        bool is_partial,
        std::vector<common_chat_msg_diff> & diffs,
        bool filter_tool_calls = false);
};

struct server_task {
    int id = -1; // to be filled by server_queue

    // TODO @ngxson : remove this field and implement a mapping task_id -> idx in the response_reader
    size_t index = 0; // used when there are multiple prompts (batch request)

    // used by SERVER_TASK_TYPE_CANCEL
    int id_target = -1;
    int id_slot   = -1;

    // used by parallel sampling (multiple completions from same prompt)
    int id_parent  = -1;
    // temporary store of child tasks for scheduling
    // note: accessing to elements is invalid after the task is moved to server_slot
    std::vector<server_task> child_tasks;

    // used by SERVER_TASK_TYPE_INFERENCE
    task_params   params;
    server_tokens tokens;

    // only used by CLI, this allow tokenizing CLI inputs on server side
    // we need this because mtmd_context and vocab are not accessible outside of server_context
    bool                    cli = false;
    std::string             cli_prompt;
    std::vector<raw_buffer> cli_files;

    server_task_type type;

    // used by SERVER_TASK_TYPE_SLOT_SAVE, SERVER_TASK_TYPE_SLOT_RESTORE, SERVER_TASK_TYPE_SLOT_ERASE
    struct slot_action {
        int id_slot;
        std::string filename;
        std::string filepath;
    };
    slot_action slot_action;

    // used by SERVER_TASK_TYPE_METRICS
    bool metrics_reset_bucket = false;

    // used by SERVER_TASK_TYPE_SET_LORA
    std::map<int, float> set_lora; // mapping adapter ID -> scale

    server_task() = default;

    server_task(server_task_type type) : type(type) {}

    int32_t n_tokens() const {
        return tokens.size();
    }

    bool need_embd() const {
        switch (type) {
            case SERVER_TASK_TYPE_EMBEDDING:
            case SERVER_TASK_TYPE_RERANK:
                return true;
            default:
                return false;
        }
    }

    bool need_logits() const {
        switch (type) {
            case SERVER_TASK_TYPE_COMPLETION:
            case SERVER_TASK_TYPE_INFILL:
                return true;
            default:
                return false;
        }
    }

    bool need_sampling() const {
        switch (type) {
            case SERVER_TASK_TYPE_COMPLETION:
            case SERVER_TASK_TYPE_INFILL:
                return true;
            default:
                return false;
        }
    }

    static task_params params_from_json_cmpl(
        const llama_vocab * vocab,
        const common_params & params_base,
        const int n_ctx_slot,
        const std::vector<llama_logit_bias> & logit_bias_eog,
        const json & data);

    // utility function
    static std::unordered_set<int> get_list_id(const std::vector<server_task> & tasks) {
        std::unordered_set<int> ids(tasks.size());
        for (size_t i = 0; i < tasks.size(); i++) {
            ids.insert(tasks[i].id);
            for (auto & child : tasks[i].child_tasks) {
                ids.insert(child.id);
            }
        }
        return ids;
    }

    void add_child(int id_parent, int id_child) {
        server_task copy;

        copy.id        = id_child;
        copy.id_parent = id_parent;
        copy.params    = params;
        copy.type      = type;
        copy.tokens    = tokens.clone();
        copy.id_slot   = -1; // child tasks cannot specify slot

        // use different sampling seed for each child
        // note: https://github.com/ggml-org/llama.cpp/pull/18700#discussion_r2675115723
        if (copy.params.sampling.seed != LLAMA_DEFAULT_SEED) {
            copy.params.sampling.seed += (uint32_t)child_tasks.size() + 1;
        }

        child_tasks.push_back(std::move(copy));
    }

    // the task will be moved into queue, then onto slots
    // however, the state must be kept by caller (e.g., HTTP thread)
    task_result_state create_state() const {
        return task_result_state(params.chat_parser_params);
    }

    bool is_parent() const {
        return child_tasks.size() > 0;
    }

    bool is_child() const {
        return id_parent != -1;
    }
};

struct result_timings {
    int32_t cache_n = -1;

    int32_t prompt_n = -1;
    double prompt_ms = 0.0;
    double prompt_per_token_ms = 0.0;
    double prompt_per_second = 0.0;

    int32_t predicted_n = -1;
    double predicted_ms = 0.0;
    double predicted_per_token_ms = 0.0;
    double predicted_per_second = 0.0;

    // Optional speculative metrics - only included when > 0
    int32_t draft_n = 0;
    int32_t draft_n_accepted = 0;

    json to_json() const;
};

struct result_prompt_progress {
    int32_t total = 0;
    int32_t cache = 0;
    int32_t processed = 0;
    int64_t time_ms = 0;

    json to_json() const;
};

struct server_task_result {
    int id           = -1;
    int id_slot      = -1;

    // TODO @ngxson : remove this field and implement a mapping task_id -> idx in the response_reader
    size_t index = 0; // to be used for batched tasks

    virtual bool is_error() {
        // only used by server_task_result_error
        return false;
    }
    virtual bool is_stop() {
        // only used by server_task_result_cmpl_*
        return true;
    }
    virtual void update(task_result_state &) {
        // only used by server_task_result_cmpl_*
    }
    virtual json to_json() = 0;
    virtual ~server_task_result() = default;
};

// using shared_ptr for polymorphism of server_task_result
using server_task_result_ptr = std::unique_ptr<server_task_result>;

struct completion_token_output {
    llama_token tok;
    float prob;
    std::string text_to_send;
    struct prob_info {
        llama_token tok;
        std::string txt;
        float prob;
    };
    std::vector<prob_info> probs;

    json to_json(bool post_sampling_probs) const;

    static json probs_vector_to_json(const std::vector<completion_token_output> & probs, bool post_sampling_probs);

    static float logarithm(float x);

    static std::vector<unsigned char> str_to_bytes(const std::string & str);

};

struct server_task_result_cmpl_final : server_task_result {
    std::string content;
    llama_tokens tokens;

    bool stream;
    bool include_usage;
    result_timings timings;
    std::string prompt;

    bool truncated;
    int32_t n_decoded;
    int32_t n_prompt_tokens;
    int32_t n_prompt_tokens_cache;
    int32_t n_tokens_cached;
    bool has_new_line;
    std::string stopping_word;
    stop_type stop = STOP_TYPE_NONE;

    bool post_sampling_probs;
    std::vector<completion_token_output> probs_output;
    std::vector<std::string>  response_fields;

    task_params generation_params;

    // response formatting
    bool               verbose  = false;
    task_response_type res_type = TASK_RESPONSE_TYPE_NONE;
    std::string        oaicompat_model;
    std::string        oaicompat_cmpl_id;
    common_chat_msg    oaicompat_msg; // to be populated by update()

    std::vector<common_chat_msg_diff> oaicompat_msg_diffs; // to be populated by update()
    bool is_updated = false;

    // for OpenAI Responses API
    std::string oai_resp_id;
    std::string oai_resp_reasoning_id;
    std::string oai_resp_message_id;

    virtual bool is_stop() override {
        return true; // in stream mode, final responses are considered stop
    }

    virtual json to_json() override;

    virtual void update(task_result_state & state) override {
        is_updated = true;
        oaicompat_msg = state.update_chat_msg(content, false, oaicompat_msg_diffs);

        oai_resp_id = state.oai_resp_id;
        oai_resp_reasoning_id = state.oai_resp_reasoning_id;
        oai_resp_message_id = state.oai_resp_message_id;
    }

    json to_json_non_oaicompat();

    json usage_json_oaicompat();

    json to_json_oaicompat();

    json to_json_oaicompat_chat();

    json to_json_oaicompat_chat_stream();

    json to_json_oaicompat_resp();

    json to_json_oaicompat_resp_stream();

    json to_json_oaicompat_asr();

    json to_json_anthropic();

    json to_json_anthropic_stream();
};

struct server_task_result_cmpl_partial : server_task_result {
    std::string  content;
    llama_tokens tokens;

    int32_t n_decoded;
    int32_t n_prompt_tokens;
    int32_t n_prompt_tokens_cache;

    bool post_sampling_probs;
    bool is_progress = false;
    completion_token_output prob_output;
    result_timings timings;
    result_prompt_progress progress;

    // response formatting
    bool               verbose  = false;
    task_response_type res_type = TASK_RESPONSE_TYPE_NONE;
    std::string        oaicompat_model;
    std::string        oaicompat_cmpl_id;
    std::vector<common_chat_msg_diff> oaicompat_msg_diffs; // to be populated by update()
    bool is_updated = false;

    // Streaming state copied from task_result_state for this chunk
    bool thinking_block_started = false;
    bool text_block_started     = false;

    // for OpenAI Responses API
    std::string oai_resp_id;
    std::string oai_resp_reasoning_id;
    std::string oai_resp_message_id;
    std::string oai_resp_fc_id;

    // for Anthropic API: track if any reasoning content has been generated
    bool anthropic_has_reasoning = false;

    virtual bool is_stop() override {
        return false; // in stream mode, partial responses are not considered stop
    }

    virtual void update(task_result_state & state) override;

    virtual json to_json() override;

    json to_json_non_oaicompat();

    json to_json_oaicompat();

    json to_json_oaicompat_chat();

    json to_json_oaicompat_resp();

    json to_json_oaicompat_asr();

    json to_json_anthropic();
};

struct server_task_result_embd : server_task_result {
    std::vector<std::vector<float>> embedding;

    int32_t n_tokens;

    // response formatting
    task_response_type res_type = TASK_RESPONSE_TYPE_NONE;

    virtual json to_json() override;

    json to_json_non_oaicompat();

    json to_json_oaicompat();
};

struct server_task_result_rerank : server_task_result {
    float score = -1e6;

    int32_t n_tokens;

    virtual json to_json() override;
};

struct server_task_result_error : server_task_result {
    error_type err_type = ERROR_TYPE_SERVER;
    std::string err_msg;

    // for ERROR_TYPE_EXCEED_CONTEXT_SIZE
    int32_t n_prompt_tokens = 0;
    int32_t n_ctx           = 0;

    virtual bool is_error() override {
        return true;
    }

    virtual json to_json() override;
};

struct server_task_result_metrics : server_task_result {
    int n_idle_slots;
    int n_processing_slots;
    int n_tasks_deferred;
    int64_t t_start;

    // TODO: somehow reuse server_metrics in the future, instead of duplicating the fields
    uint64_t n_prompt_tokens_processed_total = 0;
    uint64_t t_prompt_processing_total       = 0;
    uint64_t n_tokens_predicted_total        = 0;
    uint64_t t_tokens_generation_total       = 0;

    uint64_t n_tokens_max = 0;

    uint64_t n_prompt_tokens_processed = 0;
    uint64_t t_prompt_processing       = 0;

    uint64_t n_tokens_predicted  = 0;
    uint64_t t_tokens_generation = 0;

    uint64_t n_decode_total     = 0;
    uint64_t n_busy_slots_total = 0;

    // spec-route counters (label -> count): drafter = resolved drafter of the
    // task, kind = cache rebuild reason on a drafter tag mismatch
    std::map<std::string, uint64_t> spec_route_requests       = {};
    std::map<std::string, uint64_t> spec_route_cache_rebuilds = {};
    uint64_t spec_route_overrides_total = 0;

    // t8 stadio 2 (spec §6, Task 6): MTP head tokens accepted in concat rounds
    uint64_t spec_route_concat_mtp_accepted_total = 0;

    // t20 f3 (pi-stack): draft rounds attributed per drafter family - the
    // ngram drafter (pure prompt lookup, no model) vs the model drafter (MTP
    // or DFlash per the per-request routing)
    uint64_t spec_route_ngram_drafts_total  = 0;
    uint64_t spec_route_model_drafts_total = 0;

    // t25: PLE disk-store counters (cumulative since model load)
    uint64_t ple_hits_total        = 0;
    uint64_t ple_misses_total      = 0;
    uint64_t ple_blocks_read_total = 0;
    uint64_t ple_read_bytes_total  = 0;
    uint64_t ple_fetch_us_total    = 0;
    uint64_t ple_fetches_total     = 0;

    // t23: persistent library counters (cumulative since model load; all zero
    // without --cache-disk-persist). The persist_* family is the persist subset
    // of the disk_* counters - see persist_stats.
    uint64_t persist_saves_total         = 0;
    uint64_t persist_loads_total         = 0;
    uint64_t persist_hits_total          = 0;
    uint64_t persist_touches_total       = 0;
    uint64_t persist_evictions_total     = 0;
    uint64_t persist_gc_orphans_total    = 0;
    uint64_t persist_bytes_written_total = 0;
    uint64_t persist_bytes_read_total    = 0;
    uint64_t persist_save_failures_total = 0;

    // t23: gauges of the last cross-restart restore
    double   persist_last_restore_ms       = 0.0;
    uint64_t persist_last_tokens_restored  = 0;
    uint64_t persist_last_tokens_prefilled = 0;

    // T32-b: CRC-verification wall time of the last PARK-library restore
    // (read from the park instance; 0 while the park is off or unused)
    double   park_restore_crc_ms          = 0.0;

    // PI F4 follow-up (29/08): speculative drafter state resets (the partial-reject
    // reset window), exported as spec_state_resets_total
    uint64_t spec_state_resets_total = 0;

    // while we can also use std::vector<server_slot> this requires copying the slot object which can be quite messy
    // therefore, we use json to temporarily store the slot.to_json() result
    json slots_data = json::array();

    virtual json to_json() override;
};

struct server_task_result_slot_save_load : server_task_result {
    std::string filename;
    bool is_save; // true = save, false = load

    size_t n_tokens;
    size_t n_bytes;
    double t_ms;

    virtual json to_json() override;
};

struct server_task_result_slot_erase : server_task_result {
    size_t n_erased;

    virtual json to_json() override;
};

struct server_task_result_get_lora : server_task_result {
    struct lora {
        common_adapter_lora_info info;
        std::string  alora_invocation_string;
        llama_tokens alora_invocation_tokens;
    };
    std::vector<lora> loras;

    virtual json to_json() override;
};

struct server_task_result_apply_lora : server_task_result {
    virtual json to_json() override;
};

struct server_prompt_data {
    std::vector<uint8_t> main;
    std::vector<uint8_t> drft;
    std::vector<uint8_t> spec;

    size_t size() const {
        return main.size() + drft.size() + spec.size();
    }
};

struct server_prompt {
    server_tokens tokens;

    server_prompt_data data;

    // spec-route: drafter whose context produced data.drft - the draft KV is only
    // restorable into that drafter's context (a dual server keeps one draft
    // context per drafter). NONE = untagged (legacy/pre-routing), restored as before.
    common_speculative_type drafter = COMMON_SPECULATIVE_TYPE_NONE;

    std::list<common_prompt_checkpoint> checkpoints;

    size_t size() const {
        size_t res = 0;

        res += data.size();

        for (const auto & ckpt : checkpoints) {
            res += ckpt.size();
        }

        return res;
    }

    int n_tokens() const {
        return tokens.size();
    }

    server_prompt clone() const {
        return server_prompt {
            tokens.clone(),
            data,
            drafter,
            checkpoints,
        };
    }
};

// The context checkpoint payloads may contain large host vectors and cloned
// backend buffers. Disk-cache entries keep only the small scheduling metadata;
// a restored disk state starts with an empty live checkpoint list and creates
// fresh checkpoints as prompt processing continues.
struct server_prompt_checkpoint_meta {
    int64_t   n_tokens = 0;
    llama_pos pos_min  = 0;
    llama_pos pos_max  = 0;
};

struct server_prompt_disk_state {
    server_tokens tokens;
    std::vector<server_prompt_checkpoint_meta> checkpoints;
    std::vector<uint8_t> spec;

    // spec-route: drafter whose context produced path_drft (see server_prompt::drafter)
    common_speculative_type drafter = COMMON_SPECULATIVE_TYPE_NONE;

    std::string path_main;
    std::string path_drft;

    size_t size_main = 0;
    size_t size_drft = 0;

    uint64_t id = 0;
    bool usable = true;

    // T23 persistent library bookkeeping (persist only). Neutral defaults keep
    // the per-run path byte-for-byte untouched.
    uint32_t hits       = 0;
    uint64_t created_at = 0;
    uint64_t last_used  = 0;
    uint32_t crc_main   = 0;
    uint32_t crc_drft   = 0;

    size_t size() const {
        return size_main + size_drft;
    }

    int n_tokens() const {
        return tokens.size();
    }
};

// spec-route: may an entry's draft/spec payload be restored into the drafter the
// incoming task routed to? NONE on either side means "no routing information" and
// keeps the legacy restore behavior (wildcard).
static inline bool spec_route_tag_compatible(common_speculative_type entry, common_speculative_type active) {
    return entry == active ||
        entry == COMMON_SPECULATIVE_TYPE_NONE || active == COMMON_SPECULATIVE_TYPE_NONE;
}

// spec-route: strict drafter identity for the cache bookkeeping (dedup / disk
// touch / obsolete-prefix reclaim): an entry of another drafter - untagged ones
// included - never contains this state's draft KV, so it must neither suppress a
// save nor be superseded by it.
static inline bool spec_route_same_drafter(common_speculative_type a, common_speculative_type b) {
    return a == b;
}

// T32-b: the load() admission and ranking rules, split out as pure functions so
// every candidate source (the slot's base prompt, the RAM states, the disk
// states of the general AND the park library) shares literally ONE selection
// implementation - the park joins the existing selection, never a parallel one
// - and so tests/test-park-load.cpp can drive the rules without a llama
// runtime. server_prompt_cache::load wires its spec_boundary_valid lambda and
// its candidate loops onto these two; the semantics are byte-for-byte the ones
// that lived inline in the loops before.

// may an entry of `cached_tokens` tokens whose common prefix with the new
// prompt is `lcp` still be restored while stateful speculative state is
// required? An exact boundary is always fine; otherwise the diverging tail
// (cached_tokens - lcp tokens) must be removable: dense KV (n_rs_min ==
// UINT32_MAX, no bound known) accepts any bounded rollback, RS-bounded
// recurrent state only up to n_rs_min snapshot positions.
static inline bool server_prompt_cache_spec_boundary_valid(
        bool spec_state_required, bool spec_trailing_rm,
        size_t cached_tokens, int lcp, uint32_t n_rs_min) {
    if (!spec_state_required || lcp == (int) cached_tokens) {
        return true;
    }
    if (!spec_trailing_rm || lcp < 0 || cached_tokens <= (size_t) lcp) {
        return false;
    }
    const size_t delta = cached_tokens - (size_t) lcp;
    return n_rs_min == UINT32_MAX || delta <= (size_t) n_rs_min;
}

// one candidate step of the best-candidate ranking (shared by every source):
// given the candidate's common prefix and entry length, decide whether it is
// STRICTLY better than the running best and, if so, update the best trackers.
// f_keep < 0.25 ("don't trash large prompts") rejects before the ranking; in
// spec mode the longest valid boundary wins, otherwise BOTH f_keep and sim
// must strictly improve (so at parity the earlier-scanned source - the hot RAM
// copy before either disk library - keeps the win).
static inline bool server_prompt_cache_candidate_better(
        bool spec_state_required, size_t tokens_new_size,
        int lcp_cur, size_t cached_tokens,
        float & f_keep_best, float & sim_best, size_t & spec_boundary_best) {
    const float f_keep_cur = float(lcp_cur) / std::max<size_t>(1, cached_tokens);
    const float sim_cur    = float(lcp_cur) / std::max<size_t>(1, tokens_new_size);

    if (f_keep_cur < 0.25f) {
        return false;
    }

    const bool is_better = spec_state_required
        ? cached_tokens > spec_boundary_best
        : f_keep_best < f_keep_cur && sim_best < sim_cur;
    if (!is_better) {
        return false;
    }

    f_keep_best = f_keep_cur;
    sim_best    = sim_cur;
    spec_boundary_best = cached_tokens;
    return true;
}

// W6-8 (A3-5): the park library's optional RAM mirror of its live entry. The
// T32-b park is persist-only BY DESIGN (bounded RAM); --cache-disk-park-ram-mirror
// (default 0 = off) re-enables a bounded single-entry RAM copy behind an
// explicit flag: every park save that fits the cap also tees its payload into
// this mirror (verified against the sidecar CRC at capture time), and the next
// restore of that entry serves from RAM - no disk read, no CRC pass. The
// mirror dies with its entry (supersede / eviction / GC); a fresh boot starts
// mirror-less. Default-off leaves the save/restore paths byte-for-byte as
// they are: every mirror code path is gated on park_ram_mirror_limit > 0.
struct park_ram_mirror {
    bool        valid    = false; // a captured entry is live (entry_id names it)
    uint64_t    entry_id = 0;
    uint32_t    crc_main = 0;     // informational: the CRCs verified at capture
    uint32_t    crc_drft = 0;
    std::vector<uint8_t> main;    // the exact state-file bytes of the entry
    std::vector<uint8_t> drft;

    void reset() {
        *this = park_ram_mirror{};
    }
};

// admission: the park holds one live entry, so the mirror is all-or-nothing -
// the entry's total payload (target + draft state files) either fits the
// configured cap or no mirror is kept at all
static inline bool server_prompt_cache_park_mirror_admits(size_t limit_bytes, size_t payload_bytes) {
    return limit_bytes > 0 && payload_bytes > 0 && payload_bytes <= limit_bytes;
}

// serve: the mirror answers for exactly the entry it was captured from, with
// byte counts equal to the entry record. The draft is only consulted when the
// caller will actually restore it: a drafter-tag mismatch skips the draft by
// design, in which case the mirror's draft half is simply never looked at
static inline bool server_prompt_cache_park_mirror_matches(
        const park_ram_mirror & mirror, uint64_t entry_id,
        size_t target_bytes, bool drft_wanted, size_t draft_bytes, bool drft_present) {
    if (!mirror.valid || mirror.entry_id != entry_id) {
        return false;
    }
    if (mirror.main.size() != target_bytes || target_bytes == 0) {
        return false;
    }
    if (drft_wanted) {
        return drft_present && draft_bytes > 0 && mirror.drft.size() == draft_bytes;
    }
    return true;
}

// t23: live stats of the persistent library; zeroed when persist is disabled.
// The persist_* values are a strict subset of the disk_* counters (every persist
// save/load/eviction bumps both), so a dashboard watching only persist_* sees no
// double counting.
struct persist_stats {
    uint64_t saves         = 0;
    uint64_t loads         = 0;
    uint64_t hits          = 0;
    uint64_t touches       = 0;
    uint64_t evictions     = 0;
    uint64_t gc_orphans    = 0;
    uint64_t bytes_written = 0;
    uint64_t bytes_read    = 0;
    uint64_t save_failures = 0;

    double   last_restore_ms       = 0.0; // wall time of the last cross-restart restore
    uint64_t last_tokens_restored  = 0;   // of it, the tokens served from the library
    uint64_t last_tokens_prefilled = 0;   // of it, the tokens still prefilled by the model

    // T32-b: CRC-verification wall time of the last PARK-library restore
    // (meaningful only on a park instance, zero elsewhere)
    double   park_restore_crc_ms   = 0.0;
};

struct server_prompt_cache {
    // T32-b: persist_subdir selects the persistent library directory below the
    // cache root ("persist" keeps the historical layout); park marks a
    // persist-only instance (the park library) - no per-run directory is
    // created or cleaned up, disk_owned_path names the library directory
    // itself and save_disk/eviction operate on it like on any persist library.
    server_prompt_cache(
            int32_t limit_size_mib,
             size_t limit_tokens,
        const std::string & disk_base_path = {},
            int32_t disk_limit_size_mib = 0,
        const uint8_t    fingerprint[16] = {},
            int32_t persist_mib = 0,
            int32_t persist_min_tokens = 1024,
        const std::string & persist_subdir = "persist",
             bool park = false,
        // W6-8 (A3-5): RAM mirror budget in bytes for a PARK instance's live
        // entry; 0 (the default everywhere today) keeps the T32-b
        // RAM-disabled park exactly as it is
             size_t park_ram_mirror_bytes = 0);

    ~server_prompt_cache();

    std::list<server_prompt> states;

    // Cold automatic cache. Entries own only token/checkpoint metadata in RAM;
    // target and draft context payloads live in owner-only files.
    std::list<server_prompt_disk_state> disk_states;

    bool ram_enabled = false;

    // in bytes, 0 = no limit
    size_t limit_size = 0;

    // in tokens, 0 = no limit
    size_t limit_tokens = 0;

    // Disk fields are disabled when disk_owned_path is empty.
    std::string disk_base_path;
    std::string disk_owned_path;
    size_t disk_limit_size = 0;
    size_t disk_size_total = 0;
    int disk_lock_fd = -1;

    uint64_t disk_next_id       = 1;
    uint64_t disk_saves         = 0;
    uint64_t disk_loads         = 0;
    uint64_t disk_evictions     = 0;
    uint64_t disk_bytes_written = 0;
    uint64_t disk_bytes_read    = 0;
    uint64_t disk_bytes_evicted = 0;
    uint64_t disk_save_failures = 0;

    // A durable write/removal failure opens this run-level circuit breaker.
    // Existing valid entries remain readable, but no further state files are
    // created for this server process.
    bool disk_save_disabled = false;

    // T32-b: this instance is a persist-only park library (ctor park flag) -
    // no per-run directory exists for it and the destructor never removes
    // disk_owned_path (it IS the persistent library directory).
    bool park_library = false;

    // W6-8 (A3-5): park RAM mirror state (see park_ram_mirror above). Inert
    // unless park_library && park_ram_mirror_limit > 0 - every reader/writer
    // of park_mirror is behind that gate, so default-off is behavior-neutral.
    size_t          park_ram_mirror_limit = 0; // bytes, 0 = off (default)
    park_ram_mirror park_mirror;

    // T23 persistent library (all inert when persist_path is empty)
    std::string persist_path;                 // <base>/<ns>/<subdir> ("persist" or the park's subdir)
    size_t      persist_limit_size = 0;
    int         persist_min_tokens = 1024;
    uint8_t     persist_fingerprint[16] = {0};
    uint64_t    persist_saves = 0, persist_loads = 0, persist_hits = 0;
    uint64_t    persist_touches = 0, persist_evictions = 0, persist_gc_orphans = 0;
    uint64_t    persist_bytes_written = 0, persist_bytes_read = 0;
    uint64_t    persist_save_failures = 0;
    double      persist_last_restore_ms = 0.0;   // last cross-restart restore
    uint64_t    persist_last_tokens_restored = 0, persist_last_tokens_prefilled = 0;
    double      persist_park_restore_crc_ms = 0.0; // T32-b: last park restore's CRC-verify wall time

    // advisory lock on persist_path, held for the whole server lifetime
    int persist_lock_fd = -1;

    size_t size() const;

    size_t n_tokens() const;

    size_t disk_size() const;

    size_t disk_n_tokens() const;

    // t23: copy-out of the persistent library counters/gauges (zeros when the
    // persist library is disabled); read from the metrics task like get_ple_stats
    persist_stats get_persist_stats() const;

    // T32-b: park (optional, default nullptr = the historical combined flow,
    // byte-for-byte). When set, this save is REDIRECTED: the RAM half still
    // runs on this (the general) instance exactly as before, but the disk half
    // goes to the park library - and if that park disk save fails for ANY
    // reason (inert park, write error, the multimodal skip), it is retried on
    // this instance's own disk library, so a state is never dropped because
    // the park was chosen (safety-net, spec t32-b step 4.2). The combined
    // return keeps save()'s historical meaning: disk_ok || ram_ok.
    bool save(
        const server_prompt & prompt,
              llama_context * ctx_main,
              llama_context * ctx_drft,
               llama_seq_id   id_slot,
        const std::vector<uint8_t> & state_spec,
        common_speculative_type drafter,
        server_prompt_cache * park = nullptr);

    // T32-b: the park max-1-live-entry invariant keeper - erase every entry
    // except keep_id (the just-committed one), one "persist park: supersede"
    // log per removed entry. Public on purpose: unit-tested directly
    // (tests/test-park-redirect.cpp); driving it through a real save needs a
    // llama runtime. In save_disk it runs strictly AFTER the new entry's
    // commit point (fsync'd bins + meta rename + entry-dir fsync), which is
    // what makes the erase-after-write crash ordering hold: a crash in between
    // never leaves a missing entry - only a stale second one that lives until
    // the boundary grows past it (a same-boundary save is a touch, no
    // supersede) or a boot eviction drops it. Returns false if an erase fails
    // (caller: warn + stop; the new entry stays committed).
    bool persist_park_supersede(uint64_t keep_id);

    server_prompt * alloc(
        const server_prompt & prompt,
                    size_t state_size_main,
                    size_t state_size_drft,
        const std::vector<uint8_t> & state_spec,
        common_speculative_type drafter);

    bool load(
              server_prompt & prompt,
        const server_tokens & tokens_new,
              llama_context * ctx_main,
              llama_context * ctx_drft,
                    int32_t   id_slot,
                       bool   spec_state_required,
                       bool   spec_trailing_rm,
                       bool * cache_hit,
                   uint64_t * disk_entry_id,
              // spec-route: drafter the incoming task routed to - on a tag
              // mismatch the target KV is still restored, the entry's
              // draft/spec payload is not (set on mismatch, cleared otherwise)
              common_speculative_type drafter_active,
                    bool * tag_mismatch,
              // T32-b: park (optional, default nullptr = today's selection,
              // byte-for-byte). The park library's disk entries join the SAME
              // best-candidate selection (same lcp/similarity machinery, same
              // spec-boundary and f_keep rules; at parity the RAM copy and
              // then the general library win - the park is scanned last).
              // When a park entry wins, the physical restore AND all its
              // bookkeeping run on the PARK instance: load_disk, the
              // pending-load stash, and - through disk_entry_owner below -
              // the caller's accept_disk_load/reject_disk_load (hits/LRU
              // splice/sidecar rewrite/erase) operate on the park's paths and
              // budget. The park's entry file paths are absolute, which is
              // the only reason a cross-instance restore is safe at all.
              server_prompt_cache * park = nullptr,
              // T32-b: out-param - the instance OWNING the disk entry this
              // load selected (this or the park above; left at `this` when no
              // disk entry won). The caller MUST route accept/reject there.
              server_prompt_cache ** disk_entry_owner = nullptr);

    // Called when a stateful speculative implementation rejects a blob after
    // the target/draft files themselves restored successfully.
    void accept_disk_load(uint64_t entry_id);

    void reject_disk_load(uint64_t entry_id, const char * reason);

    void update();

private:
    // T32-b: the RAM half of save(), split out so the redirected (park) save
    // runs it on the general instance alone. ram_enabled gates FIRST (a
    // RAM-disabled instance never touches the contexts), mirroring the early
    // return of the pre-split save().
    bool save_ram(
        const server_prompt & prompt,
              llama_context * ctx_main,
              llama_context * ctx_drft,
               llama_seq_id   id_slot,
        const std::vector<uint8_t> & state_spec,
        common_speculative_type drafter);

    bool save_disk(
        const server_prompt & prompt,
              llama_context * ctx_main,
              llama_context * ctx_drft,
               llama_seq_id   id_slot,
        const std::vector<uint8_t> & state_spec,
        common_speculative_type drafter);

    bool load_disk(
        std::list<server_prompt_disk_state>::iterator it,
        server_prompt & prompt,
        llama_context * ctx_main,
        llama_context * ctx_drft,
         llama_seq_id   id_slot,
              size_t   lcp,
            uint64_t * entry_id_out,
        common_speculative_type drafter_active,
              bool * tag_mismatch,
            // T23: restore measurements for the pending-load stash (see below)
              double * load_ms_out,
              size_t * n_tokens_main_out,
            // T32-b: wall time of the sidecar CRC verification of this restore
            // (0 when the persist library is off or the load rejected earlier)
              double * crc_ms_out = nullptr);

    bool erase_disk_state(std::list<server_prompt_disk_state>::iterator it, bool eviction, const char * reason);

    void disable_disk_saves(const char * reason, const std::string & path);

    void update_disk();

    void log_disk_state() const;

    // T23: adopt the cross-restart persistent library at boot (persist only).
    // Deterministic fixed order: create the library directory (subdir below the
    // cache root - "persist" for the general library, the park's own name for a
    // park instance), take the exclusive advisory
    // lock, copy the identity fingerprint, scan entry-* directories, rebuild
    // disk_states oldest-first, evict by decayed-hit score down to the library
    // budget, GC foreign (orphan) entries only under budget pressure, then
    // publish the next entry id. Fail-safe: never throws (for a general
    // instance the run-dir lock is already held; a park instance owns no run
    // directory); every failure disables persist only - persist_path cleared,
    // and for a park instance the owned path too, making it fully inert -
    // leaving the per-run cache of a general instance fully working.
    void boot_persist(const std::string & cache_root_utf8, const std::string & subdir_utf8, const uint8_t * fingerprint);

    // T23: eviction-victim choice for the persistent library - minimum decayed
    // hit score; ties break to the older last_used, then the smaller id.
    // Returns end() when nothing is eligible. exclude_id = 0 excludes nothing.
    std::list<server_prompt_disk_state>::iterator persist_pick_victim(uint64_t exclude_id);

    // T23: write (or rewrite) an entry's metadata sidecar: serialize -> meta.tmp
    // -> fsync -> atomic rename to meta. The meta rename is the entry's commit
    // point (before it, a crashed save leaves only tmp/partial files which the
    // next boot discards). Returns false on any IO failure; the caller decides
    // the severity (save-time = failure, touch-time = best-effort).
    bool persist_write_meta(const server_prompt_disk_state & state);

    // T23: measurements of the in-flight disk restore. load_disk is the only
    // place that measures it (load_ms, tokens restored) and load() the only one
    // that knows the full new-prompt token count - the stash carries both to
    // accept_disk_load/reject_disk_load, which the caller invokes on the same
    // main-loop thread right after the load. The restore gauges and the
    // persist-load marker report these stashed numbers, never recomputed ones.
    struct persist_pending_load {
        uint64_t entry_id            = 0;
        double   load_ms             = 0.0;
        double   crc_ms              = 0.0; // T32-b: CRC-verify wall time (park gauge)
        size_t   n_tokens_main       = 0;  // tokens the entry restored
        size_t   lcp                 = 0;  // of those, the prefix the new prompt reuses
        size_t   prompt_tokens_total = 0;  // full length of the new prompt
        bool     valid               = false;
    };

    persist_pending_load pending_load = {};
};
