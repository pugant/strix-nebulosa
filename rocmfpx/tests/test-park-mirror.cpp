// W6-8 (A3-5) end-to-end test of the park RAM mirror: a REAL tiny model (the
// test-state-io-buffer recipe, CPU-only, in-memory gguf) driven through
// server_prompt_cache's own save()/load() on park instances.
//
// The mirror discriminator is physical: after a park save the entry's state
// files are DELETED from disk. A mirror-enabled park must still restore the
// entry (RAM copy serves it, read_path=ram-mirror) with a payload
// byte-identical to the ordinary file restore; a mirror-OFF park (the default)
// must fail the same scenario exactly like today (size-mismatch rejection).
// Invalidation is exercised through persist_park_supersede: the mirror dies
// with its entry, and a superseding save that no longer fits the cap leaves
// the park mirror-less (same delete-files scenario fails again).
//
// A W6-GAUGE section at the end measures the card's wall metrics on the same
// code paths (bigger ~128 MiB state, disk-backed dir - see run_gauge).

#include "../src/llama-arch.h"
#include "../src/llama-model-saver.h"
#include "../src/llama-persist-meta.h"
#include "../tools/server/server-task.h"

#include "llama.h"
#include "llama-cpp.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "gguf.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#include <cstdlib> // mkdtemp, getenv

namespace fs = std::filesystem;

static int g_failures = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("FAIL: %s\n", msg); g_failures++; } \
} while (0)

static bool silent_model_load_progress(float /*progress*/, void * /*user_data*/) {
    return true;
}

static void test_log(ggml_log_level level, const char * text, void * /*user_data*/) {
    if (level == GGML_LOG_LEVEL_ERROR) {
        fputs(text, stderr);
    }
}

// CPU-only device list: zero GPU, ever
static ggml_backend_dev_t * cpu_only_devices() {
    static std::vector<ggml_backend_dev_t> devs;
    if (devs.empty()) {
        ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (!cpu) {
            throw std::runtime_error("no CPU backend device found");
        }
        devs.push_back(cpu);
        devs.push_back(nullptr);
    }
    return devs.data();
}

static void set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    std::hash<std::string> hasher;
    std::mt19937 gen(hasher(tensor->name) + *(const size_t *) userdata);
    std::normal_distribution<float> dis(0.0f, 1.0e-2f);

    const int64_t ne = ggml_nelements(tensor);
    if (tensor->type == GGML_TYPE_F32) {
        std::vector<float> tmp(ne);
        for (int64_t i = 0; i < ne; i++) {
            tmp[i] = dis(gen);
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(ne);
        for (int64_t i = 0; i < ne; i++) {
            tmp[i] = ggml_fp32_to_fp16(dis(gen));
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else {
        GGML_ABORT("fatal error");
    }
}

static gguf_context_ptr get_gguf_ctx() {
    gguf_context_ptr ret(gguf_init_empty());
    llama_model_saver ms(LLM_ARCH_LLAMA, ret.get());
    const uint32_t n_ctx    = 128;
    const uint32_t n_vocab  = 128;
    const uint32_t n_embd   = 256;
    const uint32_t n_head   = 2;
    const uint32_t n_ff     = 384;
    const uint32_t n_layer  = 2;

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,      llm_arch_name(LLM_ARCH_LLAMA));
    ms.add_kv(LLM_KV_VOCAB_SIZE,                n_vocab);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,            n_ctx);
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,          n_embd);
    ms.add_kv(LLM_KV_FEATURES_LENGTH,           n_embd);
    ms.add_kv(LLM_KV_BLOCK_COUNT,               n_layer);
    ms.add_kv(LLM_KV_LEADING_DENSE_BLOCK_COUNT, uint32_t(1));
    ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH,       n_ff);
    ms.add_kv(LLM_KV_USE_PARALLEL_RESIDUAL,     false);
    ms.add_kv(LLM_KV_LOGIT_SCALE,               1.0f);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT,      n_head);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV,   n_head);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_EPS,   1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, 1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW,  n_ctx / 8);
    ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS,   std::vector<uint32_t>({n_embd / 8, n_embd / 8, n_embd / 8, n_embd / 8}));
    ms.add_kv(LLM_KV_TOKENIZER_MODEL,           "no_vocab");

    for (uint32_t il = 0; il < n_layer; il++) {
        ggml_tensor t;
        memset(&t, 0, sizeof(t));
        t.type = GGML_TYPE_F16;
        ggml_format_name(&t, "conv%" PRIu32 "d.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
    }
    return ret;
}

// W6-GAUGE: the bigger sibling of get_gguf_ctx - 8 layers x 1024 embd with
// f16 KV gives ~32 KiB/token, so a 4096-token state lands at ~128 MiB of
// payload: the same order as the park entry the live gauge host exercised
// (127 MB), big enough for the disk wall to dominate the noise
static gguf_context_ptr get_gguf_ctx_gauge() {
    gguf_context_ptr ret(gguf_init_empty());
    llama_model_saver ms(LLM_ARCH_LLAMA, ret.get());
    const uint32_t n_ctx    = 8192;
    const uint32_t n_vocab  = 128;
    const uint32_t n_embd   = 1024;
    const uint32_t n_head   = 8;
    const uint32_t n_ff     = 2816;
    const uint32_t n_layer  = 8;

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,      llm_arch_name(LLM_ARCH_LLAMA));
    ms.add_kv(LLM_KV_VOCAB_SIZE,                n_vocab);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,            n_ctx);
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,          n_embd);
    ms.add_kv(LLM_KV_FEATURES_LENGTH,           n_embd);
    ms.add_kv(LLM_KV_BLOCK_COUNT,               n_layer);
    ms.add_kv(LLM_KV_LEADING_DENSE_BLOCK_COUNT, uint32_t(1));
    ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH,       n_ff);
    ms.add_kv(LLM_KV_USE_PARALLEL_RESIDUAL,     false);
    ms.add_kv(LLM_KV_LOGIT_SCALE,               1.0f);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT,      n_head);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV,   n_head);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_EPS,   1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, 1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW,  n_ctx / 8);
    // rope sections must sum to n_embd/n_head (the per-head rope dim)
    ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS,   std::vector<uint32_t>({n_embd / n_head / 4, n_embd / n_head / 4,
                                                                        n_embd / n_head / 4, n_embd / n_head / 4}));
    ms.add_kv(LLM_KV_TOKENIZER_MODEL,           "no_vocab");

    for (uint32_t il = 0; il < n_layer; il++) {
        ggml_tensor t;
        memset(&t, 0, sizeof(t));
        t.type = GGML_TYPE_F16;
        ggml_format_name(&t, "conv%" PRIu32 "d.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
    }
    return ret;
}

static std::pair<llama_model_ptr, llama_context_ptr> get_model_and_ctx(gguf_context * gguf_ctx, size_t seed,
        uint32_t n_ubatch = 64, int n_threads = 4) {
    llama_model_params model_params = llama_model_default_params();
    model_params.progress_callback = silent_model_load_progress;
    model_params.devices = cpu_only_devices();

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx         = 0;
    ctx_params.n_threads     = n_threads;
    ctx_params.n_threads_batch = n_threads;
    ctx_params.n_ubatch      = n_ubatch;

    llama_model_ptr model(llama_model_init_from_user(gguf_ctx, set_tensor_data, &seed, model_params));
    if (!model) {
        throw std::runtime_error("failed to create llama model");
    }
    llama_context_ptr lctx(llama_init_from_model(model.get(), ctx_params));
    if (!lctx) {
        throw std::runtime_error("failed to create llama context");
    }
    return {std::move(model), std::move(lctx)};
}

static std::vector<llama_token> get_tokens(uint32_t n_tokens, uint32_t n_vocab, size_t seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dis(0, n_vocab - 1);
    std::vector<llama_token> ret;
    ret.reserve(n_tokens);
    for (uint32_t i = 0; i < n_tokens; i++) {
        ret.push_back(dis(gen));
    }
    return ret;
}

// contexts stay alive for the whole test (the model each one was created
// from must outlive it; holding the pair together is the simplest contract)
static std::vector<std::pair<llama_model_ptr, llama_context_ptr>> g_keep_alive;

// a fresh, EMPTY context (restore target)
static llama_context * fresh_ctx(gguf_context * gguf_ctx, size_t seed, uint32_t n_ubatch = 64, int n_threads = 4) {
    g_keep_alive.push_back(get_model_and_ctx(gguf_ctx, seed, n_ubatch, n_threads));
    return g_keep_alive.back().second.get();
}

// decode `tokens` on seq `id_slot` of a fresh context (state source).
// Chunked at the context's logical batch limit (the one-shot variant asserts
// n_tokens <= n_batch for large gauge prompts)
static llama_context * decode_tokens(gguf_context * gguf_ctx, size_t seed, const std::vector<llama_token> & tokens, llama_seq_id id_slot,
        uint32_t n_ubatch = 64, int n_threads = 4) {
    llama_context * ctx = fresh_ctx(gguf_ctx, seed, n_ubatch, n_threads);

    const size_t chunk = 1024; // < n_batch default (2048), multiple-friendly
    for (size_t off = 0; off < tokens.size(); off += chunk) {
        const size_t n = std::min(chunk, tokens.size() - off);
        llama_batch batch = llama_batch_init((int32_t) n, 0, 1);
        for (size_t j = 0; j < n; j++) {
            batch.token    [batch.n_tokens] = tokens[off + j];
            batch.pos      [batch.n_tokens] = (llama_pos) (off + j);
            batch.n_seq_id [batch.n_tokens] = 1;
            batch.seq_id   [batch.n_tokens][0] = id_slot;
            batch.logits   [batch.n_tokens] = false;
            batch.n_tokens++;
        }
        batch.logits[n - 1] = true;
        if (llama_decode(ctx, batch)) {
            llama_batch_free(batch);
            throw std::runtime_error("failed to decode");
        }
        llama_batch_free(batch);
    }
    return ctx;
}

static std::vector<uint8_t> payload_of(llama_context * ctx, llama_seq_id seq) {
    const size_t size = llama_state_seq_get_size(ctx, seq);
    std::vector<uint8_t> blob(size);
    const size_t got = llama_state_seq_get_data(ctx, blob.data(), blob.size(), seq);
    if (got != size) {
        throw std::runtime_error("get_data short");
    }
    return blob;
}

// remove every state file of the live entries (simulates the payload going
// cold/away on disk - the sidecar meta stays, the bins do not)
static void delete_entry_files(const server_prompt_cache & cache) {
    for (const auto & st : cache.disk_states) {
        std::error_code ec;
        if (!st.path_main.empty()) {
            fs::remove(fs::u8path(st.path_main), ec);
        }
        if (!st.path_drft.empty()) {
            fs::remove(fs::u8path(st.path_drft), ec);
        }
    }
}

// ---------------------------------------------------------------------------
// W6-GAUGE: wall-clock measurements of the card metrics inside THIS e2e test
// (CPU-only, real tiny model, real park library). Everything prints with the
// `W6-GAUGE:` prefix so a harness can grep the log. NOTE: the gauge directory
// MUST be disk-backed (/tmp is tmpfs on the target host - a RAM filesystem
// would make every "cold" read a memcpy); the default is /var/tmp, override
// with W6A3_GAUGE_DIR. The card's absolute expectations are for the 2.27 GB
// production entry: these numbers are the same code paths at a ~128 MiB
// payload scale (the live host gauge exercised a 127 MB park entry), so the
// RATIOS and the decomposition carry, the absolute walls do not.
// ---------------------------------------------------------------------------

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static double median_of(std::vector<double> v) {
    if (v.empty()) {
        return 0.0;
    }
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// fdatasync + POSIX_FADV_DONTNEED, the same discipline the server uses after
// writing a state file (server_prompt_cache_disk_flush_and_drop)
static void sync_and_drop(const std::string & path) {
    const int fd = open(path.c_str(), O_RDWR | O_CLOEXEC);
    if (fd < 0) {
        return;
    }
    (void) fdatasync(fd);
#if defined(POSIX_FADV_DONTNEED)
    (void) posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
#endif
    close(fd);
}

static void run_gauge() {
    printf("W6-GAUGE: section start (CPU-only, disk-backed dir; ratios carry to prod scale, absolute walls do not)\n");

    const char * env_dir = getenv("W6A3_GAUGE_DIR");
    std::string base;
    if (env_dir != nullptr && *env_dir != '\0') {
        base = env_dir;
        std::error_code ec;
        fs::create_directories(fs::u8path(base), ec);
    } else {
        char tmpl[] = "/var/tmp/llama-test-park-gauge-XXXXXX";
        const char * tmp = mkdtemp(tmpl);
        if (tmp == nullptr) {
            printf("W6-GAUGE: SKIPPED (no tmpdir)\n");
            return;
        }
        base = tmp;
    }
    printf("W6-GAUGE: dir=%s\n", base.c_str());

    const size_t seed_g = 0x6A0E5EEDull;
    const uint32_t n_tokens_g = 4096;
    const uint32_t n_ubatch_g = 1024;
    const int n_threads_g = 8;

    gguf_context_ptr gguf_g = get_gguf_ctx_gauge();
    const std::vector<llama_token> tokens_g = get_tokens(n_tokens_g, 128, seed_g);

    llama_context * ctx_g = nullptr;
    {
        const double t0 = now_ms();
        ctx_g = decode_tokens(gguf_g.get(), seed_g, tokens_g, 0, n_ubatch_g, n_threads_g);
        printf("W6-GAUGE: setup decode_tokens=%u threads=%d ms=%.1f\n", n_tokens_g, n_threads_g, now_ms() - t0);
    }

    server_prompt prompt_g;
    prompt_g.tokens = server_tokens(tokens_g, false);

    uint8_t fp_g[16];
    for (size_t i = 0; i < sizeof(fp_g); i++) {
        fp_g[i] = (uint8_t) (0x3C + i);
    }

    // ---- (d) SAVE: in-stream CRC (W6-7) vs the old two-pass sequence -------
    // old = save_file (write) + fdatasync+DONTNEED + crc32_file (whole-file
    //      cold re-read, exactly what the server used to do after the drop)
    // new = save_file_crc (write with the CRC folded in) + fdatasync+DONTNEED
    {
        std::vector<double> old_ms, new_ms;
        size_t payload = 0;
        for (int i = 0; i < 3; i++) {
            const std::string path_old = base + "/gauge-save-old-" + std::to_string(i) + ".bin";
            const std::string path_new = base + "/gauge-save-new-" + std::to_string(i) + ".bin";

            double t0 = now_ms();
            const size_t n_old = llama_state_seq_save_file(ctx_g, path_old.c_str(), 0, tokens_g.data(), tokens_g.size());
            const double t_write = now_ms() - t0;
            t0 = now_ms();
            sync_and_drop(path_old);
            const double t_sync = now_ms() - t0;
            t0 = now_ms();
            uint32_t crc_old = 0;
            const bool crc_ok = llama_persist_crc32_file(path_old.c_str(), (uint64_t) n_old, &crc_old);
            const double t_crc = now_ms() - t0;
            const double old_total = t_write + t_sync + t_crc;
            CHECK(n_old > 0 && crc_ok, "gauge: old two-pass save readable");
            payload = n_old;

            t0 = now_ms();
            uint32_t crc_new = 0;
            const size_t n_new = llama_state_seq_save_file_crc(ctx_g, path_new.c_str(), 0, tokens_g.data(), tokens_g.size(), &crc_new);
            const double t_write_crc = now_ms() - t0;
            t0 = now_ms();
            sync_and_drop(path_new);
            const double t_sync_new = now_ms() - t0;
            const double new_total = t_write_crc + t_sync_new;
            CHECK(n_new == n_old && crc_new == crc_old, "gauge: in-stream CRC == read-back CRC at scale");

            old_ms.push_back(old_total);
            new_ms.push_back(new_total);
            printf("W6-GAUGE: save iter=%d bytes=%zu twopass_ms=%.1f (write=%.1f sync=%.1f crc_reread=%.1f) instream_ms=%.1f (write_crc=%.1f sync=%.1f)\n",
                    i, payload, old_total, t_write, t_sync, t_crc, new_total, t_write_crc, t_sync_new);
        }
        const double m_old = median_of(old_ms);
        const double m_new = median_of(new_ms);
        printf("W6-GAUGE: save MEDIAN bytes=%zu twopass_ms=%.1f instream_ms=%.1f delta_pct=%.1f (card: -30..-45%%)\n",
                payload, m_old, m_new, m_old > 0 ? 100.0 * (m_old - m_new) / m_old : 0.0);
    }

    // ---- parks: mirror OFF and mirror ON on the same live state -----------
    const size_t mirror_cap_g = 512ull << 20; // 512 MiB, far above the payload
    server_prompt_cache park_off(0, 8192, base + "/g-off", 4096, fp_g, 4096, 0, "persist", true, 0);
    server_prompt_cache park_on(0, 8192, base + "/g-on", 4096, fp_g, 4096, 0, "persist", true, mirror_cap_g);

    // ---- (c) mirror capture overhead: e2e save ON vs save OFF -------------
    {
        double t0 = now_ms();
        CHECK(park_off.save(prompt_g, ctx_g, nullptr, 0, {}, COMMON_SPECULATIVE_TYPE_NONE), "gauge: save OFF lands");
        const double off_ms = now_ms() - t0;
        t0 = now_ms();
        CHECK(park_on.save(prompt_g, ctx_g, nullptr, 0, {}, COMMON_SPECULATIVE_TYPE_NONE), "gauge: save ON lands");
        const double on_ms = now_ms() - t0;
        CHECK(park_on.park_mirror.valid, "gauge: mirror captured at scale");
        const size_t payload = park_on.disk_states.back().size_main;
        printf("W6-GAUGE: mirror capture overhead bytes=%zu save_off_ms=%.1f save_on_ms=%.1f capture_delta_ms=%.1f\n",
                payload, off_ms, on_ms, on_ms - off_ms);
    }

    const std::string gauge_state_path = park_off.disk_states.back().path_main;

    // ---- (a) restore from disk, single cold read (W6-5) -------------------
    llama_context * ctx_r_off = fresh_ctx(gguf_g.get(), seed_g, n_ubatch_g, n_threads_g);
    {
        std::vector<double> ms;
        for (int i = 0; i < 3; i++) {
            sync_and_drop(gauge_state_path); // force the cold regime
            server_prompt prompt_x;
            bool hit = false;
            uint64_t id = 0;
            const double t0 = now_ms();
            const bool ok = park_off.load(prompt_x, prompt_g.tokens, ctx_r_off, nullptr, 0,
                    false, false, &hit, &id, COMMON_SPECULATIVE_TYPE_NONE, nullptr, nullptr);
            const double dt = now_ms() - t0;
            CHECK(ok && hit, "gauge: disk restore succeeded");
            ms.push_back(dt);
            printf("W6-GAUGE: restore disk iter=%d ms=%.1f\n", i, dt);
        }
        printf("W6-GAUGE: restore disk MEDIAN ms=%.1f (W6-5: single cold read + in-buffer restore)\n", median_of(ms));
    }

    // ---- (a-old) the two-pass restore the W6-5 change removed, reconstructed
    //      from the same building blocks the server used: a whole-file CRC
    //      verify pass, then the file-IO load pass (both cold)
    llama_context * ctx_r_old = fresh_ctx(gguf_g.get(), seed_g, n_ubatch_g, n_threads_g);
    {
        std::vector<double> ms;
        for (int i = 0; i < 3; i++) {
            sync_and_drop(gauge_state_path);
            double t0 = now_ms();
            uint32_t crc = 0;
            CHECK(llama_persist_crc32_file(gauge_state_path.c_str(),
                        (uint64_t) park_off.disk_states.back().size_main, &crc),
                    "gauge: old verify pass readable");
            const double t_verify = now_ms() - t0;
            sync_and_drop(gauge_state_path); // the old CRC pass ended with DONTNEED
            t0 = now_ms();
            std::vector<llama_token> restored(tokens_g.size() + 8, LLAMA_TOKEN_NULL);
            size_t n_restored = 0;
            const size_t n = llama_state_seq_load_file(ctx_r_old, gauge_state_path.c_str(), 0,
                    restored.data(), restored.size(), &n_restored);
            const double t_load = now_ms() - t0;
            CHECK(n > 0 && n_restored == tokens_g.size(), "gauge: old load pass worked");
            ms.push_back(t_verify + t_load);
            printf("W6-GAUGE: restore twopass-old iter=%d ms=%.1f (verify_pass=%.1f load_pass=%.1f)\n",
                    i, t_verify + t_load, t_verify, t_load);
        }
        printf("W6-GAUGE: restore twopass-old MEDIAN ms=%.1f (what W6-5 removed: the verify-pass re-read)\n", median_of(ms));
    }

    // ---- (b) restore from the RAM mirror (W6-8) ---------------------------
    llama_context * ctx_r_on = fresh_ctx(gguf_g.get(), seed_g, n_ubatch_g, n_threads_g);
    {
        std::vector<double> ms;
        for (int i = 0; i < 3; i++) {
            sync_and_drop(gauge_state_path); // even cold disk must not matter here
            server_prompt prompt_x;
            bool hit = false;
            uint64_t id = 0;
            const double t0 = now_ms();
            const bool ok = park_on.load(prompt_x, prompt_g.tokens, ctx_r_on, nullptr, 0,
                    false, false, &hit, &id, COMMON_SPECULATIVE_TYPE_NONE, nullptr, nullptr);
            const double dt = now_ms() - t0;
            CHECK(ok && hit, "gauge: mirror restore succeeded");
            ms.push_back(dt);
            printf("W6-GAUGE: restore mirror iter=%d ms=%.1f\n", i, dt);
        }
        printf("W6-GAUGE: restore mirror MEDIAN ms=%.1f (W6-8 hit: memcpy+meta, zero disk, zero CRC)\n", median_of(ms));
    }

    std::error_code rm_ec;
    fs::remove_all(fs::u8path(base), rm_ec);
    printf("W6-GAUGE: section end\n");
}

int main() {
    llama_log_set(test_log, nullptr);

    char tmpl[] = "/tmp/llama-test-park-mirror-XXXXXX";
    const char * tmp = mkdtemp(tmpl);
    CHECK(tmp != nullptr, "mkdtemp created the test tmpdir");
    if (tmp == nullptr) {
        return 1;
    }
    const std::string base = tmp;

    const size_t seed = 0x5EED5EED;
    gguf_context_ptr gguf_ctx = get_gguf_ctx();

    const std::vector<llama_token> tokens = get_tokens(70, 128, seed);

    uint8_t fp[16];
    for (size_t i = 0; i < sizeof(fp); i++) {
        fp[i] = (uint8_t) (0x5A + i);
    }

    const size_t mirror_cap = 8ull << 20; // 8 MiB - far above the tiny entry

    try {
        // the source state: decode once, save the same live state to both
        // libraries below
        llama_context * ctx_src = decode_tokens(gguf_ctx.get(), seed, tokens, 0);

        server_prompt prompt;
        prompt.tokens = server_tokens(tokens, false);

        // --- 1. reference: the ordinary FILE restore of the saved entry ---
        std::vector<uint8_t> blob_ref;
        std::string path_state_ref;
        {
            server_prompt_cache park_off(0, 4096, base + "/off", 64, fp, 64, 0, "persist", true, 0);
            CHECK(park_off.save(prompt, ctx_src, nullptr, 0, {}, COMMON_SPECULATIVE_TYPE_NONE),
                "mirror-OFF park saves the state");
            CHECK(park_off.park_ram_mirror_limit == 0, "OFF park: mirror limit 0");
            CHECK(!park_off.park_mirror.valid, "OFF park: no mirror captured (default off = today's behavior)");
            CHECK(park_off.disk_states.size() == 1, "OFF park: one entry on disk");
            path_state_ref = park_off.disk_states.back().path_main;

            llama_context * ctx_ref = fresh_ctx(gguf_ctx.get(), seed);
            std::vector<llama_token> restored(tokens.size() + 8, LLAMA_TOKEN_NULL);
            size_t n_restored = 0;
            const size_t n = llama_state_seq_load_file(ctx_ref, path_state_ref.c_str(), 0,
                    restored.data(), restored.size(), &n_restored);
            CHECK(n > 0 && n_restored == tokens.size(), "reference file restore works");
            blob_ref = payload_of(ctx_ref, 0);
            CHECK(blob_ref.size() > 0, "reference payload extracted");

            // --- 2. OFF discriminator: same library with the bins deleted
            //        must fail exactly like today (size-mismatch rejection) ---
            delete_entry_files(park_off);
            server_prompt prompt_off;
            bool cache_hit_off = true; // expect it to stay false
            uint64_t entry_off = 0;
            llama_context * ctx_off = fresh_ctx(gguf_ctx.get(), seed);
            const bool ok_off = park_off.load(prompt_off, prompt.tokens, ctx_off, nullptr, 0,
                    false, false, &cache_hit_off, &entry_off, COMMON_SPECULATIVE_TYPE_NONE, nullptr, nullptr);
            CHECK(!ok_off && !cache_hit_off, "OFF park: bins gone -> restore fails (no RAM copy to serve)");
        }

        // --- 3. ON: the mirror serves when the bins are gone, and the payload
        //     is byte-identical to the file restore (gate N) ---
        {
            server_prompt_cache park_on(0, 4096, base + "/on", 64, fp, 64, 0, "persist", true, mirror_cap);
            CHECK(park_on.save(prompt, ctx_src, nullptr, 0, {}, COMMON_SPECULATIVE_TYPE_NONE),
                "mirror-ON park saves the state");
            CHECK(park_on.park_ram_mirror_limit == mirror_cap, "ON park: mirror budget carried");
            CHECK(park_on.park_mirror.valid, "ON park: mirror captured for the live entry");
            CHECK(park_on.park_mirror.entry_id == park_on.disk_states.back().id, "ON park: mirror names the entry");
            CHECK(park_on.park_mirror.main.size() == park_on.disk_states.back().size_main,
                "ON park: mirror holds the entry's exact target bytes");

            delete_entry_files(park_on);

            server_prompt prompt_on;
            bool cache_hit_on = false;
            uint64_t entry_on = 0;
            llama_context * ctx_on = fresh_ctx(gguf_ctx.get(), seed);
            const bool ok_on = park_on.load(prompt_on, prompt.tokens, ctx_on, nullptr, 0,
                    false, false, &cache_hit_on, &entry_on, COMMON_SPECULATIVE_TYPE_NONE, nullptr, nullptr);
            CHECK(ok_on && cache_hit_on, "ON park: bins gone -> restore SUCCEEDS from the RAM mirror");

            const std::vector<uint8_t> blob_mirror = payload_of(ctx_on, 0);
            CHECK(blob_mirror.size() == blob_ref.size() &&
                  memcmp(blob_mirror.data(), blob_ref.data(), blob_mirror.size()) == 0,
                "mirror restore payload is byte-identical to the file restore (gate N)");
            CHECK(park_on.disk_states.size() == 1, "ON park: the entry survived (no rejection happened)");

            // --- 4. invalidation by supersede with a cap the new entry no
            //        longer fits: the mirror must be gone ---
            park_on.park_ram_mirror_limit = 64; // smaller than the entry payload
            const std::vector<llama_token> tokens_b = get_tokens(80, 128, seed ^ 0xB);
            server_prompt prompt_b;
            {
                // grow the boundary: decode a longer prompt and save it
                llama_context * ctx_b = decode_tokens(gguf_ctx.get(), seed, tokens_b, 0);
                prompt_b.tokens = server_tokens(tokens_b, false);
                CHECK(park_on.save(prompt_b, ctx_b, nullptr, 0, {}, COMMON_SPECULATIVE_TYPE_NONE),
                    "grown-boundary save lands");
            }
            CHECK(!park_on.park_mirror.valid, "entry over the cap -> no mirror lives");
            CHECK(park_on.disk_states.size() == 1, "the park still holds exactly one (superseded) entry");
        }

        // --- 5. review-fix (lead): drafter-tag mismatch with a drafter context
        //     active. The entry was saved WITH a draft (tagged DFLASH) but the
        //     load runs under a DIFFERENT concrete drafter (DSPARK): the draft
        //     gate must reject the entry IDENTICALLY on the mirror path and on
        //     the file path - the mirror is a pure optimization, never a
        //     semantic. Expected everywhere: load fails, the entry is
        //     erase-rejected, and on the mirror park the mirror dies with it.
        {
            llama_context * ctx_drft_src = fresh_ctx(gguf_ctx.get(), seed);

            const auto save_tagged = [&](server_prompt_cache & park) {
                return park.save(prompt, ctx_src, ctx_drft_src, 0, {},
                                 COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH);
            };
            const auto load_under_other_drafter = [&](server_prompt_cache & park, llama_context * ctx_t) {
                server_prompt prompt_x;
                bool cache_hit = true; // must stay false
                uint64_t entry_id = 0;
                const bool ok = park.load(prompt_x, prompt.tokens, ctx_t, ctx_drft_src, 0,
                        false, false, &cache_hit, &entry_id, COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK, nullptr, nullptr);
                return std::pair<bool, bool>(ok, cache_hit);
            };

            // mirror OFF: today's file behavior - rejection + erase
            {
                server_prompt_cache park_off(0, 4096, base + "/off-tag", 64, fp, 64, 0, "persist", true, 0);
                CHECK(save_tagged(park_off), "tagged save lands on the OFF park");
                CHECK(park_off.disk_states.size() == 1, "OFF-tag park holds the entry");
                llama_context * ctx_t = fresh_ctx(gguf_ctx.get(), seed);
                const auto [ok, hit] = load_under_other_drafter(park_off, ctx_t);
                CHECK(!ok && !hit, "OFF path: tag mismatch with active drafter -> load fails (file behavior)");
                CHECK(park_off.disk_states.empty(), "OFF path: the entry was erase-rejected");
            }

            // mirror ON: the mirror must NOT turn the rejection into a
            // main-only serve - same outcome as the file path
            {
                server_prompt_cache park_on(0, 4096, base + "/on-tag", 64, fp, 64, 0, "persist", true, mirror_cap);
                CHECK(save_tagged(park_on), "tagged save lands on the ON park");
                CHECK(park_on.park_mirror.valid, "ON-tag park captured the mirror (main+drft)");
                CHECK(park_on.park_mirror.drft.size() == park_on.disk_states.back().size_drft,
                    "ON-tag mirror holds the draft bytes too");
                llama_context * ctx_t = fresh_ctx(gguf_ctx.get(), seed);
                const auto [ok, hit] = load_under_other_drafter(park_on, ctx_t);
                CHECK(!ok && !hit, "MIRROR path: tag mismatch with active drafter -> load fails IDENTICALLY to the file path");
                CHECK(park_on.disk_states.empty(), "MIRROR path: the entry was erase-rejected like on the file path");
                CHECK(!park_on.park_mirror.valid, "MIRROR path: the mirror died with the rejected entry");
            }
        }
    } catch (const std::exception & e) {
        printf("FAIL: unexpected exception: %s\n", e.what());
        g_failures++;
    }

    // W6-GAUGE: wall measurements of the card metrics (CPU-only, own section,
    // failures there count like any other)
    try {
        run_gauge();
    } catch (const std::exception & e) {
        printf("FAIL: unexpected exception (gauge): %s\n", e.what());
        g_failures++;
    }

    std::error_code rm_ec;
    fs::remove_all(fs::u8path(base), rm_ec);

    if (g_failures == 0) {
        printf("test-park-mirror: ALL PASS\n");
        return 0;
    }
    printf("test-park-mirror: %d FAILURES\n", g_failures);
    return 1;
}
