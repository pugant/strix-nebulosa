#pragma once

#include "common.h"
#include "log.h"
#include "llama.h"
#include "chat.h"
#include "mtmd.h"

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

#include <string>
#include <vector>
#include <cinttypes>
#include <cstdint>
#include <memory>

using json = nlohmann::ordered_json;

#define SLT_DBG(slot, fmt, ...) LOG_DBG("slot %12.*s: id %2d | task %d | " fmt, 12, __func__, (slot).id, ((slot).task ? (slot).task->id : -1), __VA_ARGS__)
#define SLT_TRC(slot, fmt, ...) LOG_TRC("slot %12.*s: id %2d | task %d | " fmt, 12, __func__, (slot).id, ((slot).task ? (slot).task->id : -1), __VA_ARGS__)
#define SLT_INF(slot, fmt, ...) LOG_INF("slot %12.*s: id %2d | task %d | " fmt, 12, __func__, (slot).id, ((slot).task ? (slot).task->id : -1), __VA_ARGS__)
#define SLT_WRN(slot, fmt, ...) LOG_WRN("slot %12.*s: id %2d | task %d | " fmt, 12, __func__, (slot).id, ((slot).task ? (slot).task->id : -1), __VA_ARGS__)
#define SLT_ERR(slot, fmt, ...) LOG_ERR("slot %12.*s: id %2d | task %d | " fmt, 12, __func__, (slot).id, ((slot).task ? (slot).task->id : -1), __VA_ARGS__)
#define SLT_CNT(slot, fmt, ...) LOG_CNT(""                                 fmt,                                                                __VA_ARGS__)

#define SRV_DBG(fmt, ...) LOG_DBG("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SRV_TRC(fmt, ...) LOG_TRC("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SRV_INF(fmt, ...) LOG_INF("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SRV_WRN(fmt, ...) LOG_WRN("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SRV_ERR(fmt, ...) LOG_ERR("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SRV_CNT(fmt, ...) LOG_CNT(""              fmt,               __VA_ARGS__)

using raw_buffer = std::vector<uint8_t>;

template <typename T>
static T json_value(const json & body, const std::string & key, const T & default_value) {
    // Fallback null to default value
    if (body.contains(key) && !body.at(key).is_null()) {
        try {
            return body.at(key);
        } catch (NLOHMANN_JSON_NAMESPACE::detail::type_error const & err) {
            LOG_WRN("Wrong type supplied for parameter '%s'. Expected '%s', using default value: %s\n", key.c_str(), json(default_value).type_name(), err.what());
            return default_value;
        }
    } else {
        return default_value;
    }
}

// https://community.openai.com/t/openai-chat-list-of-error-codes-and-types/357791/11
enum error_type {
    ERROR_TYPE_INVALID_REQUEST,
    ERROR_TYPE_AUTHENTICATION,
    ERROR_TYPE_SERVER,
    ERROR_TYPE_NOT_FOUND,
    ERROR_TYPE_PERMISSION,
    ERROR_TYPE_UNAVAILABLE, // custom error
    ERROR_TYPE_NOT_SUPPORTED, // custom error
    ERROR_TYPE_EXCEED_CONTEXT_SIZE, // custom error
};

// thin wrapper around common_grammar_trigger with (de)serialization functions
struct server_grammar_trigger {
    common_grammar_trigger value;

    server_grammar_trigger() = default;
    server_grammar_trigger(const common_grammar_trigger & value) : value(value) {}
    server_grammar_trigger(const json & in) {
        value.type = (common_grammar_trigger_type) in.at("type").get<int>();
        value.value = in.at("value").get<std::string>();
        if (value.type == COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN) {
            value.token = (llama_token) in.at("token").get<int>();
        }
    }

    json to_json() const {
        json out {
            {"type", (int) value.type},
            {"value", value.value},
        };
        if (value.type == COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN) {
            out["token"] = (int) value.token;
        }
        return out;
    }
};

json format_error_response(const std::string & message, const enum error_type type);

//
// random string / id
//

std::string random_string();
std::string gen_chatcmplid();
std::string gen_tool_call_id();

// get a random marker; note: each time the server restarts, the marker will be different
const char * get_media_marker();

//
// lora utils
//

// check whether the given lora set has only aloras activated (empty => false)
bool lora_all_alora(const std::vector<common_adapter_lora_info> & loras);

// if the two sets of loras are different, they require a cache clear unless the
// change is only from aloras to aloras.
bool lora_should_clear_cache(
        const std::vector<common_adapter_lora_info> & current,
        const std::vector<common_adapter_lora_info> & next);

std::map<int, float> parse_lora_request(const json & data);

bool are_lora_equal(
        const std::vector<common_adapter_lora_info> & l1,
        const std::vector<common_adapter_lora_info> & l2);

// get the ids of all enabled loras
std::vector<size_t> lora_get_enabled_ids(const std::vector<common_adapter_lora_info> & loras);

//
// server_tokens
//

/**
 * server_tokens is a helper to manage the input tokens and image for the server.
 * it is made this way to simplify the logic of KV cache management.
 */
struct server_tokens {
    bool has_mtmd = false;

private: // disallow accessing these members directly, risking out-of-sync

    // map a **start** index in tokens to the image chunk
    // note: the order need to be in-sync with tokens
    std::map<size_t, mtmd::input_chunk_ptr> map_idx_to_media;

    // list of tokens
    //   if the token is LLAMA_TOKEN_NULL, it indicates that this position is occupied by media chunk
    //   otherwise, it is a normal text token
    // note: a non-text chunk can occupy multiple tokens (aka memory cells) in the token list
    // note(2): for M-RoPE, an image can occupy different number of pos; do not assume 1-to-1 mapping tokens <-> pos
    llama_tokens tokens;

    // for ex. with input of 5 text tokens and 2 images (each image occupies 3 tokens and 2 pos):
    //      [0] [1] [2] [3] [4] [img0] [img0] [img0] [img1] [img1] [img1]
    // idx  0   1   2   3   4   5      6      7      8      9      10
    // pos  0   1   2   3   4   5      5      5      7      7      7
    // map_idx_to_media will contain: {5, img0}, {8, img1}

public:
    server_tokens() = default;
    ~server_tokens() = default;

    // Prevent copying
    // TODO: server_tokens should be copyable - remove this:
    server_tokens(const server_tokens&) = delete;
    server_tokens& operator=(const server_tokens&) = delete;

    // Allow moving (usually implicitly generated if members are movable)
    server_tokens(server_tokens&&) = default;
    server_tokens& operator=(server_tokens&&) = default;

    // Allow accessing elements using [] operator
    llama_token operator[](size_t index) { return tokens[index]; }
    const llama_token& operator[](size_t index) const { return tokens[index]; }

    server_tokens(mtmd::input_chunks & mtmd_chunks, bool has_mtmd);
    server_tokens(const llama_tokens & tokens, bool has_mtmd);

    // for debugging
    std::string str() const;

    // the next position after n_tokens. if n_tokens < 0, return the next position after all tokens.
    llama_pos pos_next(int64_t n_tokens = -1) const;

    // number of tokens with position < max_pos
    size_t size_up_to_pos(llama_pos max_pos) const;

    const mtmd::input_chunk_ptr & find_chunk(size_t idx) const;

    void push_back(llama_token tok);

    // will create a copy of the chunk if it contains non-text data
    void push_back(const mtmd_input_chunk * chunk);

    // appends server tokens, updates the media map. copies media chunks.
    void push_back(server_tokens & tokens);

    // for compatibility with context shift and prompt truncation
    void insert(const llama_tokens & inp_tokens);

    // for compatibility with speculative decoding, ctx shift, slot save/load
    const llama_tokens & get_tokens() const;

    llama_tokens get_text_tokens() const;

    // for compatibility with speculative decoding
    void set_token(llama_pos pos, llama_token id);

    size_t size() const { return tokens.size(); }

    bool empty() const { return tokens.empty(); }

    // T23: does this prompt actually carry media chunks? Distinct from has_mtmd,
    // which is also set at slot creation on multimodal servers (mctx != nullptr)
    // even for pure-text prompts - the disk cache must only skip real media.
    bool has_media() const { return !map_idx_to_media.empty(); }

    void clear() {
        map_idx_to_media.clear();
        tokens.clear();
    }

    void keep_first(size_t n);

    std::string detokenize(const llama_context * ctx, bool special) const;

    size_t get_common_prefix(const server_tokens & b) const;

    // make sure all text tokens are within the vocab range
    bool validate(const struct llama_context * ctx) const;

    // encode and decode the image chunk
    int32_t process_chunk(
                llama_context * ctx,
                mtmd_context * mctx,
                size_t idx,
                llama_pos pos,
                int32_t seq_id,
                size_t & n_tokens_out) const;

    server_tokens clone() const;
};


//
// tokenizer and input processing utils
//

bool json_is_array_of_numbers(const json & data);

// is array having BOTH numbers & strings?
bool json_is_array_of_mixed_numbers_strings(const json & data);

// does array have any individual integers/tokens?
bool json_is_array_and_contains_numbers(const json & data);

// get value by path(key1 / key2)
json json_get_nested_values(const std::vector<std::string> & paths, const json & js);

/**
 * this handles 2 cases:
 * - only string, example: "string"
 * - mixed string and tokens, example: [12, 34, "string", 56, 78]
 */
llama_tokens tokenize_mixed(const llama_vocab * vocab, const json & json_prompt, bool add_special, bool parse_special,
                            class server_token_prefix_cache * prefix_cache = nullptr);

//
// W6-6 / A4-S1: token-prefix cache for conversation prompts
//

struct server_token_prefix_cache_stats {
    uint64_t n_lookups       = 0;  // total tokenize() calls seen
    uint64_t n_full          = 0;  // full tokenizations served (miss / bypass)
    uint64_t n_reuse         = 0;  // spliced results (cached prefix reused)
    uint64_t n_reuse_full_hit = 0; // identical prompt served straight from cache
    uint64_t n_bypass_type   = 0;  // non-BPE vocab: splice not supported, bypassed
    uint64_t n_bypass_flag   = 0;  // (add_special, parse_special) differs from the entry
    uint64_t n_bypass_short  = 0;  // common prefix too small to be worth a splice
    uint64_t n_selfcheck_ok  = 0;  // self-check splices that matched the full tokenization
    uint64_t n_selfcheck_fail = 0; // self-check mismatches (cache disabled when > 0)
    uint64_t bytes_reused    = 0;  // total prefix bytes served from cache
    bool     enabled         = true; // false after a self-check failure or env disable
};

/**
 * Reuses the tokenized prefix of the previous prompt when a stateless client
 * (e.g. pi) resends the same conversation with new turns appended: only the
 * new tail is tokenized and spliced onto the cached token stream.
 *
 * Bit-identity with a full tokenization is by construction for BPE vocabularies:
 * the splice point is always the start of a special-token fragment (a hard
 * boundary of tokenizer_st_partition), and BPE fragments tokenize independently.
 * Every entry is verified at creation (sum of token pieces must re-spell the
 * prompt exactly) and the first splices are self-checked against a full
 * tokenization; any mismatch disables the cache permanently for the process.
 *
 * Thread-safe (HTTP worker threads call tokenize() concurrently).
 * Disable with LLAMA_SERVER_TOKEN_PREFIX_CACHE=0.
 */
class server_token_prefix_cache {
  public:
    // max_entries: LRU entries kept (one per interleaved conversation is enough)
    // selfcheck_n: first splices verified against a full tokenization
    server_token_prefix_cache(const llama_vocab * vocab, size_t max_entries = 4, uint32_t selfcheck_n = 3);
    ~server_token_prefix_cache();

    server_token_prefix_cache(const server_token_prefix_cache &) = delete;
    server_token_prefix_cache & operator=(const server_token_prefix_cache &) = delete;

    // same semantics as common_tokenize(vocab, text, add_special, parse_special)
    llama_tokens tokenize(const std::string & text, bool add_special, bool parse_special);

    server_token_prefix_cache_stats get_stats() const;
    void reset(); // drop cached entries (enabled state is kept)

  private:
    struct entry;
    struct impl;
    std::unique_ptr<impl> pimpl;
};

// return the last index of character that can form a valid string
// if the last character is potentially cut in half, return the index before the cut
// if validate_utf8(text) == text.size(), then the whole text is valid utf8
size_t validate_utf8(const std::string& text);

// process mtmd prompt, return the server_tokens containing both text tokens and media chunks
server_tokens process_mtmd_prompt(mtmd_context * mctx, std::string prompt, std::vector<raw_buffer> files);

/**
 * break the input "prompt" object into multiple prompt if needed, then tokenize them
 * this supports these cases:
 * - "prompt": "string"
 * - "prompt": [12, 34, 56]
 * - "prompt": [12, 34, "string", 56, 78]
 * - "prompt": { "prompt_string": "string", "multimodal_data": [ "base64" ] }
 * and multiple prompts (multi-tasks):
 * - "prompt": ["string1", "string2"]
 * - "prompt": ["string1", [12, 34, 56]]
 * - "prompt": [[12, 34, 56], [78, 90, 12]]
 * - "prompt": [[12, 34, "string", 56, 78], [12, 34, 56], { "prompt_string": "string", "multimodal_data": [ "base64" ]}]
 */
std::vector<server_tokens> tokenize_input_prompts(
                                        const llama_vocab * vocab,
                                        mtmd_context * mctx,
                                        const json & json_prompt,
                                        bool add_special,
                                        bool parse_special,
                                        class server_token_prefix_cache * prefix_cache = nullptr);

//
// OAI utils
//

// global server parameters for chat formatting / parsing
struct server_chat_params {
    bool use_jinja;
    bool prefill_assistant;
    common_reasoning_format reasoning_format;
    std::map<std::string, std::string> chat_template_kwargs; // mapping key --> json value
    common_chat_templates_ptr tmpls;
    bool allow_image;
    bool allow_audio;
    bool enable_thinking = true;
    int  reasoning_budget = -1;
    std::string reasoning_budget_message;
    float reasoning_budget_warn_ratio = 0.75f; // consumed share before the warn message (0 = warn off)
    std::string media_path;
    bool force_pure_content = false;
};

// used by /completions endpoint
json oaicompat_completion_params_parse(const json & body);

// used by /chat/completions endpoint
json oaicompat_chat_params_parse(
    json & body, /* openai api json semantics */
    const server_chat_params & opt,
    std::vector<raw_buffer> & out_files);

// TODO: move it to server-task.cpp
json format_embeddings_response_oaicompat(
    const json & request,
    const std::string & model_name,
    const json & embeddings,
    bool use_base64 = false);

// TODO: move it to server-task.cpp
json format_response_rerank(
        const json & request,
        const std::string & model_name,
        const json & ranks,
        bool is_tei_format,
        std::vector<std::string> & texts,
        int top_n);

//
// other utils
//

std::vector<llama_token_data> get_token_probabilities(llama_context * ctx, int idx);

std::string safe_json_to_str(const json & data);

std::string tokens_to_str(llama_context * ctx, const llama_tokens & tokens);
std::string tokens_to_str(const llama_vocab * vocab, const llama_tokens & tokens);

// format incomplete utf-8 multibyte character for output
std::string tokens_to_output_formatted_string(const llama_context * ctx, const llama_token token);

// format server-sent event (SSE), return the formatted string to send
// note: if data is a json array, it will be sent as multiple events, one per item
std::string format_oai_sse(const json & data);

std::string format_oai_resp_sse(const json & data);

// format Anthropic-style SSE with event types
std::string format_anthropic_sse(const json & data);

bool is_valid_utf8(const std::string & str);

//
// formatting output responses
// TODO: move these to server-task.cpp
//

llama_tokens format_prompt_infill(
        const llama_vocab * vocab,
        const json & input_prefix,
        const json & input_suffix,
        const json & input_extra,
        const int n_batch,
        const int n_predict,
        const int n_ctx,
        const bool spm_infill,
        const llama_tokens & tokens_prompt);

// format rerank task: [BOS]query[EOS][SEP]doc[EOS].
server_tokens format_prompt_rerank(
        const struct llama_model * model,
        const struct llama_vocab * vocab,
        mtmd_context * mctx,
        const std::string & query,
        const std::string & doc);
