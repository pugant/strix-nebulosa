// Unit tests for the T23 persistent prompt-cache metadata sidecar. Zero GPU,
// zero llama runtime, zero server: the serialize/parse round-trip, the parse
// error paths, CRC-32 (in-memory + whole-file), the eviction score and the
// config fingerprint are all pure functions exercised directly.

#include "../src/llama-persist-meta.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <utility>
#include <vector>

#include <unistd.h>

static int g_failures = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("FAIL: %s\n", msg); g_failures++; } \
} while (0)

static void put_le32(uint8_t * p, uint32_t v) {
    p[0] = (uint8_t) (v);
    p[1] = (uint8_t) (v >> 8);
    p[2] = (uint8_t) (v >> 16);
    p[3] = (uint8_t) (v >> 24);
}

int main() {
    CHECK(LLAMA_PERSIST_META_HEADER == 88u, "fixed header is 88 bytes (layout contract)");
    CHECK(LLAMA_PERSIST_HALF_LIFE_S == 6ull * 60ull * 60ull, "half-life constant is 6h");

    // --- 1. bit-exact round-trip: pseudorandom tokens (137) + 64-byte spec,
    //        arbitrary scalars -> serialize -> parse -> identical fields,
    //        serialize again -> byte-identical to the first encoding ---
    llama_persist_meta m;
    {
        std::mt19937 rng(4242);
        std::uniform_int_distribution<int32_t> tok_dist(INT32_MIN, INT32_MAX);
        std::uniform_int_distribution<int>      byte_dist(0, 255);

        for (uint8_t & b : m.fingerprint) b = (uint8_t) byte_dist(rng);
        m.tokens.resize(137);
        for (auto & t : m.tokens) t = tok_dist(rng); // negatives included (i32 LE path)
        m.spec.resize(64);
        for (auto & b : m.spec) b = (uint8_t) byte_dist(rng);

        m.drafter    = 3;
        m.size_main  = 0xDEADBEEFCAFEBABEull;
        m.size_drft  = 0x0123456789ABCDEFull;
        m.crc_main   = 0x13579BDFu;
        m.crc_drft   = 0x2468ACE0u;
        m.hits       = 4242;
        m.created_at = 1756684800ull;
        m.last_used  = 1756684900ull;
    }

    std::vector<uint8_t> enc1;
    CHECK(llama_persist_meta_serialize(m, enc1), "serialize ok");
    CHECK(enc1.size() == 88u + 137u * 4u + 64u, "encoded size = header + tokens*4 + spec");

    llama_persist_meta p;
    CHECK(llama_persist_meta_parse(enc1.data(), enc1.size(), p), "parse ok");
    CHECK(memcmp(p.fingerprint, m.fingerprint, 16) == 0, "round-trip: fingerprint");
    CHECK(p.drafter    == m.drafter,    "round-trip: drafter");
    CHECK(p.size_main  == m.size_main,  "round-trip: size_main");
    CHECK(p.size_drft  == m.size_drft,  "round-trip: size_drft");
    CHECK(p.crc_main   == m.crc_main,   "round-trip: crc_main");
    CHECK(p.crc_drft   == m.crc_drft,   "round-trip: crc_drft");
    CHECK(p.hits       == m.hits,       "round-trip: hits");
    CHECK(p.created_at == m.created_at, "round-trip: created_at");
    CHECK(p.last_used  == m.last_used,  "round-trip: last_used");
    CHECK(p.tokens.size() == m.tokens.size() && p.tokens == m.tokens, "round-trip: tokens (incl. negatives)");
    CHECK(p.spec.size() == m.spec.size() && p.spec == m.spec, "round-trip: spec");

    std::vector<uint8_t> enc2;
    CHECK(llama_persist_meta_serialize(p, enc2), "re-serialize ok");
    CHECK(enc1.size() == enc2.size() && memcmp(enc1.data(), enc2.data(), enc1.size()) == 0,
          "re-serialize is byte-identical to first encoding");

    // --- 2. parse error paths: every corruption is rejected ---
    {
        std::vector<uint8_t> trunc(enc1.begin(), enc1.end() - 1);
        CHECK(!llama_persist_meta_parse(trunc.data(), trunc.size(), p), "truncated by 1 byte rejected");

        std::vector<uint8_t> bad_magic(enc1);
        bad_magic[0] ^= 0xFF;
        CHECK(!llama_persist_meta_parse(bad_magic.data(), bad_magic.size(), p), "altered magic rejected");

        std::vector<uint8_t> bad_ver(enc1);
        put_le32(bad_ver.data() + 4, 999);
        CHECK(!llama_persist_meta_parse(bad_ver.data(), bad_ver.size(), p), "version 999 rejected");

        std::vector<uint8_t> bad_pad(enc1);
        bad_pad[80] = 1; // first byte of the trailing u64 pad
        CHECK(!llama_persist_meta_parse(bad_pad.data(), bad_pad.size(), p), "nonzero pad rejected");

        std::vector<uint8_t> extra(enc1);
        extra.push_back(0);
        CHECK(!llama_persist_meta_parse(extra.data(), extra.size(), p), "extra trailing byte rejected");
    }

    // --- 3. CRC-32: repeatability, sensitivity, canonical check-value,
    //        whole-file path with exact-size enforcement ---
    std::vector<uint8_t> blob(4096);
    {
        std::mt19937 rng(777);
        std::uniform_int_distribution<int> byte_dist(0, 255);
        for (auto & b : blob) b = (uint8_t) byte_dist(rng);
    }
    const uint32_t crc_a = llama_persist_crc32(blob.data(), blob.size());
    const uint32_t crc_b = llama_persist_crc32(blob.data(), blob.size());
    CHECK(crc_a != 0, "crc of random blob is nonzero");
    CHECK(crc_a == crc_b, "crc is repeatable");

    std::vector<uint8_t> flipped(blob);
    flipped[2048] ^= 0x01; // single-bit flip
    CHECK(llama_persist_crc32(flipped.data(), flipped.size()) != crc_a, "1-bit flip changes the crc");

    const char * check = "123456789";
    CHECK(llama_persist_crc32((const uint8_t *) check, 9) == 0xCBF43926u, "canonical check-value 123456789 -> 0xCBF43926");

    char tmpl[] = "/tmp/t23-persist-meta-XXXXXX";
    const int fd = mkstemp(tmpl);
    assert(fd >= 0);
    CHECK(write(fd, blob.data(), blob.size()) == (ssize_t) blob.size(), "write crc fixture");
    close(fd);
    const char * tmp_path = tmpl;

    uint32_t crc_file = 0;
    CHECK(llama_persist_crc32_file(tmp_path, blob.size(), &crc_file), "crc32_file ok on exact size");
    CHECK(crc_file == crc_a, "crc32_file equals in-memory crc");

    uint32_t sentinel = 0xDEADBEEFu;
    CHECK(!llama_persist_crc32_file(tmp_path, blob.size() - 1, &sentinel), "file LONGER than declared rejected");
    CHECK(sentinel == 0xDEADBEEFu, "crc_out untouched on failure (longer)");
    CHECK(!llama_persist_crc32_file(tmp_path, blob.size() + 1, &sentinel), "file SHORTER than declared rejected");
    CHECK(sentinel == 0xDEADBEEFu, "crc_out untouched on failure (shorter)");
    CHECK(!llama_persist_crc32_file("/tmp/t23-persist-meta-does-not-exist", 1, &sentinel), "missing file rejected");

    unlink(tmp_path);

    // --- 3b. W6-5/W6-7 (A3): incremental folder - chunked updates must equal
    //         the one-shot and the file API for ANY chunking, incl. ragged
    //         (non 16-multiple) boundaries and empty updates ---
    {
        // fixed splits at every boundary type around the fold's 16-byte block
        const size_t chunks1[] = { 0, 1, 15, 16, 17, 31, blob.size() / 2, blob.size() / 2 }; // ends exactly at EOF if even
        {
            llama_persist_crc32_folder f;
            size_t off = 0;
            for (size_t c : chunks1) {
                if (off + c > blob.size()) {
                    c = blob.size() - off;
                }
                f.update(blob.data() + off, c);
                off += c;
            }
            f.update(blob.data(), 0); // empty update is a no-op
            f.update(blob.data() + off, blob.size() - off);
            CHECK(f.finalize() == crc_a, "folder: fixed ragged chunking == one-shot");
        }

        // pseudo-random ragged chunking over a larger buffer
        std::vector<uint8_t> big(1 << 18);
        {
            std::mt19937 rng(31337);
            std::uniform_int_distribution<int> byte_dist(0, 255);
            for (auto & b : big) b = (uint8_t) byte_dist(rng);
        }
        const uint32_t crc_big = llama_persist_crc32(big.data(), big.size());
        {
            llama_persist_crc32_folder f;
            size_t off = 0;
            std::mt19937 rng(424242);
            while (off < big.size()) {
                std::uniform_int_distribution<size_t> len_dist(1, 33331); // ragged on purpose
                size_t len = std::min(len_dist(rng), big.size() - off);
                f.update(big.data() + off, len);
                off += len;
            }
            CHECK(f.finalize() == crc_big, "folder: random ragged chunking == one-shot (256 KiB)");
        }

        // single-shot through the folder == one-shot API
        {
            llama_persist_crc32_folder f;
            f.update(big.data(), big.size());
            CHECK(f.finalize() == crc_big, "folder: single update == one-shot");
        }

        // move-construction/assignment keep the fold state alive
        {
            llama_persist_crc32_folder f;
            f.update(big.data(), big.size() / 2);
            llama_persist_crc32_folder g = std::move(f);
            g.update(big.data() + big.size() / 2, big.size() - big.size() / 2);
            CHECK(g.finalize() == crc_big, "folder: moved handle continues the same stream");
        }

        // the canonical check-value through the folder
        {
            llama_persist_crc32_folder f;
            const char * chk = "123456789";
            for (int i = 0; i < 9; i++) {
                f.update(chk + i, 1); // byte-at-a-time: worst-case chunking
            }
            CHECK(f.finalize() == 0xCBF43926u, "folder: check-value byte-at-a-time");
        }
    }

    // --- 4. eviction score: exact half-life points (fixtures are the
    //        analytic values, eps 1e-9; ds4 semantics: 6h half-life) ---
    {
        const uint64_t T = 1756684800ull;
        const double eps = 1e-9;
        // age 0: no decay
        CHECK(fabs(llama_persist_eviction_score(10, T, T) - 10.0) < eps, "score hits=10 age=0 -> 10.0");
        // age 6h = one half-life -> 10 * 2^-1 = 5
        CHECK(fabs(llama_persist_eviction_score(10, T, T + 21600ull) - 5.0) < eps, "score hits=10 age=6h -> 5.0");
        // age 12h = two half-lives -> 10 * 2^-2 = 2.5
        CHECK(fabs(llama_persist_eviction_score(10, T, T + 43200ull) - 2.5) < eps, "score hits=10 age=12h -> 2.5");
        CHECK(fabs(llama_persist_eviction_score(0, T, T)) < eps, "score hits=0 -> 0.0");
        // clock-skew guard: now < last_used and now == last_used both return hits
        CHECK(fabs(llama_persist_eviction_score(10, T, T - 100ull) - 10.0) < eps, "score now < last_used -> hits (skew guard)");
        CHECK(fabs(llama_persist_eviction_score(7, T, T) - 7.0) < eps, "score now == last_used -> hits");
    }

    // --- 5. fingerprint: deterministic, sensitive to each identity part ---
    {
        uint8_t fp1[16], fp2[16], fp3[16], fp4[16], fp_null[16], fp_empty[16];
        llama_persist_fingerprint("/a/b/model.gguf", 123456ull, "/x/y/drafter.gguf", 789ull, "q8_0/q8_0", 2u, fp1);
        llama_persist_fingerprint("/a/b/model.gguf", 123456ull, "/x/y/drafter.gguf", 789ull, "q8_0/q8_0", 2u, fp2);
        CHECK(memcmp(fp1, fp2, 16) == 0, "fingerprint deterministic for identical inputs");

        llama_persist_fingerprint("/a/b/model.gguf", 123456ull, "/x/y/drafter.gguf", 789ull, "q5_1/q5_1", 2u, fp3);
        CHECK(memcmp(fp1, fp3, 16) != 0, "fingerprint differs when only cache_kv changes");

        llama_persist_fingerprint("/a/b/model.gguf", 123457ull, "/x/y/drafter.gguf", 789ull, "q8_0/q8_0", 2u, fp4);
        CHECK(memcmp(fp1, fp4, 16) != 0, "fingerprint differs when only model_size changes");

        llama_persist_fingerprint("/a/b/model.gguf", 123456ull, nullptr, 0ull, "q8_0/q8_0", 2u, fp_null);
        llama_persist_fingerprint("/a/b/model.gguf", 123456ull, "",       0ull, "q8_0/q8_0", 2u, fp_empty);
        CHECK(memcmp(fp_null, fp_empty, 16) == 0, "drafter NULL and \"\" hash identically (absent semantics)");

        uint8_t fp_kv_null[16], fp_kv_empty[16];
        llama_persist_fingerprint("/a/b/model.gguf", 123456ull, "/x/y/drafter.gguf", 789ull, nullptr, 2u, fp_kv_null);
        llama_persist_fingerprint("/a/b/model.gguf", 123456ull, "/x/y/drafter.gguf", 789ull, "",       2u, fp_kv_empty);
        CHECK(memcmp(fp_kv_null, fp_kv_empty, 16) == 0, "cache_kv NULL and \"\" hash identically (absent semantics)");

        CHECK(strcmp(llama_persist_basename("/a/b/model.gguf"), "model.gguf") == 0, "basename strips directories");
        CHECK(strcmp(llama_persist_basename(nullptr), "") == 0, "basename NULL -> empty string");
    }

    if (g_failures == 0) {
        printf("test-persist-meta: ALL PASS\n");
        return 0;
    }
    printf("test-persist-meta: %d FAILURES\n", g_failures);
    return 1;
}
