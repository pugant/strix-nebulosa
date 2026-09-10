// T23: metadata sidecar for the persistent (cross-restart) prompt cache.
// Pure data + pure functions: no server, no llama_context. Design inspired by
// antirez's ds4_kvstore (https://github.com/antirez/ds4 - disk KV cache).
#pragma once

#include "llama.h"

#include <cstdint>
#include <vector>

#define LLAMA_PERSIST_META_MAGIC   0x4D504350u  // on-disk LE bytes read "PCPM" (hexdump-friendly)
#define LLAMA_PERSIST_META_VERSION 1u
#define LLAMA_PERSIST_HALF_LIFE_S  (6ull * 60ull * 60ull)  // ds4: hit half-life 6h
#define LLAMA_PERSIST_META_HEADER  88u  // fixed header size in bytes (8-byte aligned)

// On-disk layout (all little-endian, fixed header then arrays):
//   u32 magic | u32 version | u8[16] fingerprint | u32 drafter | u32 n_tokens
//   u64 size_main | u64 size_drft | u32 crc_main | u32 crc_drft
//   u32 hits | u64 created_at | u64 last_used | u32 n_spec | u64 pad(=0)
//   llama_token tokens[n_tokens] (i32 LE) | u8 spec[n_spec]
// (4+4+16+4+4+8+8+4+4+4+8+8+4+8 = 88 bytes, 8-byte aligned)
struct llama_persist_meta {
    uint8_t  fingerprint[16] = {0};
    uint32_t drafter         = 0;   // common_speculative_type (tag T7 spec-route)
    uint64_t size_main       = 0;
    uint64_t size_drft       = 0;
    uint32_t crc_main        = 0;
    uint32_t crc_drft        = 0;
    uint32_t hits            = 0;
    uint64_t created_at      = 0;   // epoch seconds
    uint64_t last_used       = 0;   // epoch seconds

    std::vector<llama_token> tokens;
    std::vector<uint8_t>     spec;  // stateful-MTP blob (state_spec)
};

// serialize + parse; parse validates magic/version/length and returns false on
// any inconsistency (truncated buffer, array overrun, nonzero pad)
bool llama_persist_meta_serialize(const llama_persist_meta & m, std::vector<uint8_t> & out);
bool llama_persist_meta_parse(const uint8_t * data, size_t size, llama_persist_meta & out);

// CRC-32 (IEEE 802.3, reflected, poly 0xEDB88320) - table built at first use
uint32_t llama_persist_crc32(const uint8_t * data, size_t size);
// streaming CRC of a file; reads the WHOLE file and requires exactly
// expected_size bytes (shorter OR longer = corruption -> false; IO error ->
// false; crc_out untouched in those cases)
bool llama_persist_crc32_file(const char * path, uint64_t expected_size, uint32_t * crc_out);

// W6-5/W6-7 (A3): incremental CRC-32 over a byte stream - the same value
// llama_persist_crc32 returns for the concatenated bytes. For callers that
// already hold the bytes as they move (a state file being written, a payload
// being streamed in from disk): the fold composes across update() calls, so
// the chunk boundaries are irrelevant to the result. finalize() may be called
// once; the destructor releases the (tiny) fold state either way.
class llama_persist_crc32_folder {
public:
    llama_persist_crc32_folder();
    ~llama_persist_crc32_folder();

    llama_persist_crc32_folder(const llama_persist_crc32_folder &) = delete;
    llama_persist_crc32_folder & operator=(const llama_persist_crc32_folder &) = delete;

    llama_persist_crc32_folder(llama_persist_crc32_folder && other) noexcept;
    llama_persist_crc32_folder & operator=(llama_persist_crc32_folder && other) noexcept;

    void update(const void * data, size_t size);

    uint32_t finalize(); // returns the CRC of everything updated so far

private:
    struct impl;
    impl * p = nullptr;
};

// eviction score: hits decayed with a 6h half-life on time since last_used
double llama_persist_eviction_score(uint32_t hits, uint64_t last_used, uint64_t now_s);

// 16-byte fingerprint = FNV-1a 64 (two seeds) over the config identity.
// NOT cryptographic: a config disambiguation guard, not a security boundary.
//   parts: model basename, model file size, drafter basename, drafter file size,
//          cache-type string "K/V" (e.g. "q8_0/q8_0"), state format version
void llama_persist_fingerprint(const char * model_path, uint64_t model_size,
                               const char * drafter_path, uint64_t drafter_size,
                               const char * cache_kv, uint32_t state_version,
                               uint8_t out[16]);

// helpers for basename extraction used by the fingerprint (NULL-safe)
const char * llama_persist_basename(const char * path);
