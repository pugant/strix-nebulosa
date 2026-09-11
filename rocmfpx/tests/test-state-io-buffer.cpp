// W6-5/W6-7 (A3-1/A3-3) bit-exactness harness for the sequence-state IO paths.
//
// Builds tiny deterministic models fully in memory (the test-llama-archs
// recipe: gguf metadata + llama_model_init_from_user with seeded tensor data),
// decodes a fixed token sequence on seq 0 and then drives the state
// save/restore paths a server prompt-cache restore uses:
//
//   1. llama_state_seq_save_file        -> state file on disk (the artifact)
//   2. llama_state_seq_load_file        -> baseline restore (today's path)
//   3. restore determinism              -> two independent restores agree
//   4. restore -> resave                -> byte-identical state file (the
//                                          on-disk format is stable)
//   5. llama_state_seq_get_data         -> identical payload blob after every
//                                          restore path (gate N: bit-exact)
//   6. [W6-5] llama_state_seq_load_buffer   -> restore from a host RAM buffer
//      must equal the file restore byte-for-byte, incl. restored tokens
//   7. [W6-7] llama_state_seq_save_file_crc -> CRC folded while writing must
//      equal llama_persist_crc32_file over the file just written, and the
//      file bytes must be identical to the plain save's
//
// Artifacts (state file + payload blob) are dumped into argv[1] so a harness
// can compare them across builds: the golden run happens BEFORE the change,
// the post-change run must reproduce the same bytes.
//
// Zero GPU: CPU backend only, no model files, no network.

#include "../src/llama-persist-meta.h"

// TODO: replace with #include "llama-ext.h" in the future
#include "../src/llama-arch.h"
#include "../src/llama-model-saver.h"

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
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

static int g_failures = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("FAIL [%s]: %s\n", g_arch.c_str(), msg); g_failures++; } \
} while (0)

static std::string g_arch = "?";

static bool silent_model_load_progress(float /*progress*/, void * /*user_data*/) {
    return true;
}

// keep the harness output readable: errors only, everything else dropped
static void test_log(ggml_log_level level, const char * text, void * /*user_data*/) {
    if (level == GGML_LOG_LEVEL_ERROR) {
        fputs(text, stderr);
    }
}

// CPU-only device list: the harness must never land compute or buffers on an
// accelerator (W6 rule: zero GPU) - the model sees exactly one device, the CPU
static ggml_backend_dev_t * cpu_only_devices() {
    static std::vector<ggml_backend_dev_t> devs;
    if (devs.empty()) {
        ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (!cpu) {
            throw std::runtime_error("no CPU backend device found");
        }
        devs.push_back(cpu);
        devs.push_back(nullptr); // terminator
    }
    return devs.data();
}

// deterministic tensor fill: seeded by tensor NAME + the global seed, so two
// builds (or two contexts) of the same arch produce identical weights
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

// the generic tiny-model recipe of tests/test-llama-archs.cpp (get_gguf_ctx),
// trimmed to the two archs exercised here: plain attention (llama) and M-RoPE
// (qwen2vl, n_pos_per_embd() > 1 - exercises the llama_kv_cell_ext record)
static gguf_context_ptr get_gguf_ctx(llm_arch arch) {
    gguf_context_ptr ret(gguf_init_empty());
    llama_model_saver ms(arch, ret.get());
    const uint32_t n_ctx    = 128;
    const uint32_t n_vocab  = 128;
    const uint32_t n_embd   = 256;
    const uint32_t n_head   = 2;
    const uint32_t n_ff     = 384;
    const uint32_t n_layer  = 2;

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,      llm_arch_name(arch));
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

    if (arch == LLM_ARCH_QWEN2VL) {
        ms.add_kv(LLM_KV_ROPE_FREQ_BASE, 1000000.0f);
    }

    for (uint32_t il = 0; il < n_layer; il++) {
        ggml_tensor t;
        memset(&t, 0, sizeof(t));
        t.type = GGML_TYPE_F16;
        ggml_format_name(&t, "conv%" PRIu32 "d.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
    }
    return ret;
}

static std::pair<llama_model_ptr, llama_context_ptr> get_model_and_ctx(gguf_context * gguf_ctx, size_t seed) {
    llama_model_params model_params = llama_model_default_params();
    model_params.progress_callback = silent_model_load_progress;
    model_params.devices = cpu_only_devices();

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx         = 0;
    ctx_params.n_threads     = 4;
    ctx_params.n_threads_batch = 4;
    ctx_params.n_ubatch      = 64;

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

static bool write_file(const std::string & path, const uint8_t * data, size_t size) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        return false;
    }
    const bool ok = size == 0 || fwrite(data, 1, size, f) == size;
    fclose(f);
    return ok;
}

static bool read_file(const std::string & path, std::vector<uint8_t> & out) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        return false;
    }
    fseek(f, 0, SEEK_END);
    const long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (size < 0) {
        fclose(f);
        return false;
    }
    out.resize(size);
    const bool ok = size == 0 || fread(out.data(), 1, size, f) == (size_t) size;
    fclose(f);
    return ok;
}

static bool files_equal(const std::string & a, const std::string & b) {
    std::vector<uint8_t> da, db;
    return read_file(a, da) && read_file(b, db) && da.size() == db.size() && memcmp(da.data(), db.data(), da.size()) == 0;
}

// one arch end-to-end. `dir` collects the artifacts, `arch` picks the recipe.
static void run_arch(llm_arch arch, const char * arch_name, const std::string & dir, size_t seed) {
    g_arch = arch_name;

    gguf_context_ptr gguf_ctx = get_gguf_ctx(arch);

    // source context: decode the prompt on seq 0
    auto [model, ctx] = get_model_and_ctx(gguf_ctx.get(), seed);
    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    const uint32_t n_ctx   = llama_n_ctx(ctx.get());
    std::vector<llama_token> tokens = get_tokens(70, n_vocab, seed ^ 0x5eed);

    {
        llama_batch batch = llama_batch_init(n_ctx, 0, 1);
        for (int pos = 0; pos < (int) tokens.size(); pos++) {
            batch.token    [batch.n_tokens] = tokens[pos];
            batch.pos      [batch.n_tokens] = pos;
            batch.n_seq_id [batch.n_tokens] = 1;
            batch.seq_id   [batch.n_tokens][0] = 0;
            batch.logits   [batch.n_tokens] = false;
            batch.n_tokens++;
        }
        batch.logits[tokens.size() - 1] = true;
        if (llama_decode(ctx.get(), batch)) {
            llama_batch_free(batch);
            throw std::runtime_error("failed to decode");
        }
        llama_batch_free(batch);
    }

    // 1. save the seq-0 state (the artifact under test)
    const std::string path_state = dir + "/state-" + arch_name + ".bin";
    const size_t n_saved = llama_state_seq_save_file(ctx.get(), path_state.c_str(), 0, tokens.data(), tokens.size());
    CHECK(n_saved > 0, "state save produced bytes");
    std::vector<uint8_t> state_file;
    CHECK(read_file(path_state, state_file), "state file readable");
    CHECK(state_file.size() == n_saved, "state file size == save return");
    printf("%s: state file %zu bytes\n", arch_name, state_file.size());

    // 2. baseline restore into a fresh context + payload extraction
    const auto restore_and_dump = [&](const std::string & tag, std::vector<llama_token> & tokens_out) -> std::vector<uint8_t> {
        auto [model2, ctx2] = get_model_and_ctx(gguf_ctx.get(), seed);
        std::vector<llama_token> restored(tokens.size() + 16, LLAMA_TOKEN_NULL);
        size_t n_restored = 0;
        const size_t n = llama_state_seq_load_file(ctx2.get(), path_state.c_str(), 0,
                restored.data(), restored.size(), &n_restored);
        CHECK(n == n_saved, (tag + ": load_file consumed the whole file").c_str());
        CHECK(n_restored == tokens.size(), (tag + ": token count matches").c_str());
        restored.resize(n_restored);
        CHECK(restored == tokens, (tag + ": restored tokens are identical").c_str());
        tokens_out = restored;

        const size_t size = llama_state_seq_get_size(ctx2.get(), 0);
        std::vector<uint8_t> blob(size);
        const size_t got = llama_state_seq_get_data(ctx2.get(), blob.data(), blob.size(), 0);
        CHECK(got == size, (tag + ": get_data filled the blob").c_str());
        return blob;
    };

    std::vector<llama_token> tokens_a, tokens_b;
    const std::vector<uint8_t> blob_a = restore_and_dump("restore-1", tokens_a);
    const std::vector<uint8_t> blob_b = restore_and_dump("restore-2", tokens_b);

    // 3. two independent restores agree bit-for-bit (determinism)
    CHECK(blob_a.size() == blob_b.size() && memcmp(blob_a.data(), blob_b.data(), blob_a.size()) == 0,
          "two file restores produce identical payloads (gate N baseline)");

    // 4. restore -> resave reproduces the exact same file (format stability)
    {
        auto [model2, ctx2] = get_model_and_ctx(gguf_ctx.get(), seed);
        std::vector<llama_token> restored(tokens.size() + 16, LLAMA_TOKEN_NULL);
        size_t n_restored = 0;
        const size_t n = llama_state_seq_load_file(ctx2.get(), path_state.c_str(), 0,
                restored.data(), restored.size(), &n_restored);
        CHECK(n == n_saved, "resave: reload consumed the whole file");

        const std::string path_state2 = dir + "/state-" + arch_name + "-resaved.bin";
        const size_t n_saved2 = llama_state_seq_save_file(ctx2.get(), path_state2.c_str(), 0,
                restored.data(), n_restored);
        CHECK(n_saved2 == n_saved, "resave produced the same byte count");
        // NOTE: byte-identity holds only for n_pos_per_embd() == 1. M-RoPE
        // records carry llama_kv_cell_ext, and the restore path does not
        // rebuild ext.tok (ubatch_reserve has no token ids - upstream TODO in
        // state_read_meta), so a restored->resaved M-RoPE state legitimately
        // differs in the ext.tok byte of every cell record. Pre-existing
        // baseline behavior, not a format change.
        if (strcmp(arch_name, "qwen2vl") != 0) {
            CHECK(files_equal(path_state, path_state2), "restore->resave is byte-identical (on-disk format stable)");
        }
    }

    // 5. dump the payload blob as an artifact (cross-build golden comparison)
    const std::string path_blob = dir + "/payload-" + arch_name + ".bin";
    CHECK(write_file(path_blob, blob_a.data(), blob_a.size()), "payload artifact written");

    // 5b. [W6-7] save with the in-stream CRC: the produced file must be
    //     byte-identical to the plain save and the folded CRC must equal the
    //     CRC a whole-file read-back computes (gate N: sidecar equivalence)
    {
        uint32_t crc_stream = 0;
        const std::string path_state3 = dir + "/state-" + arch_name + "-crcsave.bin";
        const size_t n3 = llama_state_seq_save_file_crc(ctx.get(), path_state3.c_str(), 0,
                tokens.data(), tokens.size(), &crc_stream);
        CHECK(n3 == n_saved, "save_file_crc produced the same byte count");
        CHECK(files_equal(path_state, path_state3), "save_file_crc file bytes identical to the plain save (gate N)");

        uint32_t crc_file = 0;
        CHECK(llama_persist_crc32_file(path_state3.c_str(), (uint64_t) n3, &crc_file), "file read-back CRC computable");
        CHECK(crc_stream == crc_file, "in-stream CRC == read-back CRC (gate N)");
        CHECK(crc_stream != 0, "crc is nonzero");

        // crc_out == nullptr must behave exactly like the plain save
        const std::string path_state4 = dir + "/state-" + arch_name + "-nullcrc.bin";
        const size_t n4 = llama_state_seq_save_file_crc(ctx.get(), path_state4.c_str(), 0,
                tokens.data(), tokens.size(), nullptr);
        CHECK(n4 == n_saved && files_equal(path_state, path_state4), "save_file_crc with NULL crc_out is the plain save");
    }

    // 6. [W6-5] buffer restore: the same payload bytes consumed from a host
    //    buffer must produce the identical restored state (gate N: the
    //    load_buffer path is byte-equivalent to the file path)
    {
        auto [model2, ctx2] = get_model_and_ctx(gguf_ctx.get(), seed);
        std::vector<llama_token> restored(tokens.size() + 16, LLAMA_TOKEN_NULL);
        size_t n_restored = 0;
        const size_t n = llama_state_seq_load_buffer(ctx2.get(), state_file.data(), state_file.size(), 0,
                restored.data(), restored.size(), &n_restored);
        CHECK(n == n_saved, "load_buffer consumed the whole payload");
        CHECK(n_restored == tokens.size(), "load_buffer token count matches");
        restored.resize(n_restored);
        CHECK(restored == tokens, "load_buffer restored tokens are identical");

        const size_t size2 = llama_state_seq_get_size(ctx2.get(), 0);
        std::vector<uint8_t> blob_c(size2);
        const size_t got2 = llama_state_seq_get_data(ctx2.get(), blob_c.data(), blob_c.size(), 0);
        CHECK(got2 == size2, "load_buffer: get_data filled the blob");
        CHECK(blob_a.size() == blob_c.size() && memcmp(blob_a.data(), blob_c.data(), blob_a.size()) == 0,
              "buffer restore == file restore, payload bit-identical (gate N)");

        // error paths: every malformed buffer must fail clean (return 0), the
        // public wrapper converts IO over-read exceptions like the file one.
        // (skip when the artifact dir was unwritable and no state was saved -
        // the checks above have already reported that)
        if (!state_file.empty()) {
            std::vector<uint8_t> bad_magic = state_file;
            bad_magic[0] ^= 0xFF;
            CHECK(llama_state_seq_load_buffer(ctx2.get(), bad_magic.data(), bad_magic.size(), 0,
                    restored.data(), restored.size(), &n_restored) == 0, "load_buffer rejects bad magic");
        }
        if (!state_file.empty()) {
            std::vector<uint8_t> trunc = state_file;
            trunc.pop_back(); // state data one byte short
            CHECK(llama_state_seq_load_buffer(ctx2.get(), trunc.data(), trunc.size(), 0,
                    restored.data(), restored.size(), &n_restored) == 0, "load_buffer rejects truncated payload");
        }
        {
            std::vector<uint8_t> empty;
            CHECK(llama_state_seq_load_buffer(ctx2.get(), empty.data(), 0, 0,
                    restored.data(), restored.size(), &n_restored) == 0, "load_buffer rejects empty buffer");
        }
        {
            // capacity one token short of the stored count
            std::vector<llama_token> small(tokens.size() - 1);
            CHECK(llama_state_seq_load_buffer(ctx2.get(), state_file.data(), state_file.size(), 0,
                    small.data(), small.size(), &n_restored) == 0, "load_buffer rejects short token capacity");
        }
    }
}

int main(int argc, char ** argv) {
    llama_log_set(test_log, nullptr);

    std::string dir = ".";
    if (argc > 1) {
        dir = argv[1];
    }

    // fixed seed: the artifacts must be reproducible across builds
    const size_t seed = 0xC0FFEE42;

    try {
        run_arch(LLM_ARCH_LLAMA,   "llama",   dir, seed);
        run_arch(LLM_ARCH_QWEN2VL, "qwen2vl", dir, seed);
    } catch (const std::exception & err) {
        fprintf(stderr, "runtime error: %s\n", err.what());
        return 1;
    }

    if (g_failures == 0) {
        printf("test-state-io-buffer: ALL PASS\n");
        return 0;
    }
    printf("test-state-io-buffer: %d FAILURES\n", g_failures);
    return 1;
}
