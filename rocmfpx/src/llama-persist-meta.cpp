// T23: implementation of the persistent prompt-cache metadata sidecar.
// Pure functions only - no server, no llama_context, no logging. See
// llama-persist-meta.h for the on-disk layout contract.

#include "llama-persist-meta.h"

#include <array>
#include <cassert>
#include <cerrno>
#include <cmath>
#include <cstring>

#if defined(__x86_64__)
#include <immintrin.h> // pclmulqdq CRC fold below (target-attributed, no global -m flag)
#endif

#include <fcntl.h>
#include <unistd.h>

// the sidecar layout spells llama_token as a fixed 4-byte i32 LE
static_assert(sizeof(llama_token) == 4, "llama_token must be int32_t");

// ---------------------------------------------------------------------------
// little-endian put/get (ds4_kvstore-style): the sidecar never depends on the
// host byte order, every field is spelled out byte by byte
// ---------------------------------------------------------------------------

static void llama_persist_le_put32(uint8_t * p, uint32_t v) {
    p[0] = (uint8_t) (v);
    p[1] = (uint8_t) (v >> 8);
    p[2] = (uint8_t) (v >> 16);
    p[3] = (uint8_t) (v >> 24);
}

static uint32_t llama_persist_le_get32(const uint8_t * p) {
    return (uint32_t) p[0]
         | ((uint32_t) p[1] << 8)
         | ((uint32_t) p[2] << 16)
         | ((uint32_t) p[3] << 24);
}

static void llama_persist_le_put64(uint8_t * p, uint64_t v) {
    for (int i = 0; i < 8; i++) {
        p[i] = (uint8_t) (v >> (8 * i));
    }
}

static uint64_t llama_persist_le_get64(const uint8_t * p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) {
        v |= (uint64_t) p[i] << (8 * i);
    }
    return v;
}

// header field offsets (see layout comment in llama-persist-meta.h; the u64
// fields are intentionally NOT 8-byte aligned - serialized byte-wise, never
// cast to a struct, so alignment is irrelevant)
enum : size_t {
    OFF_MAGIC      = 0,
    OFF_VERSION    = 4,
    OFF_FINGERPRINT= 8,   // 16 bytes
    OFF_DRAFTER    = 24,
    OFF_N_TOKENS   = 28,
    OFF_SIZE_MAIN  = 32,
    OFF_SIZE_DRFT  = 40,
    OFF_CRC_MAIN   = 48,
    OFF_CRC_DRFT   = 52,
    OFF_HITS       = 56,
    OFF_CREATED_AT = 60,
    OFF_LAST_USED  = 68,
    OFF_N_SPEC     = 76,
    OFF_PAD        = 80,  // 8 bytes, must be 0
};

// ---------------------------------------------------------------------------
// serialize / parse
// ---------------------------------------------------------------------------

bool llama_persist_meta_serialize(const llama_persist_meta & m, std::vector<uint8_t> & out) {
    // loud rather than silently truncating into the u32 header fields
    assert(m.tokens.size() <= UINT32_MAX);
    assert(m.spec.size()   <= UINT32_MAX);

    const uint32_t n_tokens = (uint32_t) m.tokens.size();
    const uint32_t n_spec   = (uint32_t) m.spec.size();

    const size_t payload = (size_t) LLAMA_PERSIST_META_HEADER
                         + m.tokens.size() * 4  // llama_token i32 LE (static_assert above)
                         + m.spec.size();

    out.clear();
    out.reserve(payload);
    out.resize((size_t) LLAMA_PERSIST_META_HEADER, 0);

    llama_persist_le_put32(out.data() + OFF_MAGIC,       LLAMA_PERSIST_META_MAGIC);
    llama_persist_le_put32(out.data() + OFF_VERSION,     LLAMA_PERSIST_META_VERSION);
    memcpy(out.data() + OFF_FINGERPRINT, m.fingerprint, 16);
    llama_persist_le_put32(out.data() + OFF_DRAFTER,     m.drafter);
    llama_persist_le_put32(out.data() + OFF_N_TOKENS,    n_tokens);
    llama_persist_le_put64(out.data() + OFF_SIZE_MAIN,   m.size_main);
    llama_persist_le_put64(out.data() + OFF_SIZE_DRFT,   m.size_drft);
    llama_persist_le_put32(out.data() + OFF_CRC_MAIN,    m.crc_main);
    llama_persist_le_put32(out.data() + OFF_CRC_DRFT,    m.crc_drft);
    llama_persist_le_put32(out.data() + OFF_HITS,        m.hits);
    llama_persist_le_put64(out.data() + OFF_CREATED_AT,  m.created_at);
    llama_persist_le_put64(out.data() + OFF_LAST_USED,   m.last_used);
    llama_persist_le_put32(out.data() + OFF_N_SPEC,      n_spec);
    llama_persist_le_put64(out.data() + OFF_PAD,         0); // reserved, must stay 0

    // tokens: llama_token is int32_t (include/llama.h) -> i32 LE each
    const size_t tokens_off = out.size();
    out.resize(tokens_off + m.tokens.size() * 4);
    for (size_t i = 0; i < m.tokens.size(); i++) {
        llama_persist_le_put32(out.data() + tokens_off + i * 4, (uint32_t) m.tokens[i]);
    }

    out.insert(out.end(), m.spec.begin(), m.spec.end());

    assert(out.size() == payload);
    return true;
}

bool llama_persist_meta_parse(const uint8_t * data, size_t size, llama_persist_meta & out) {
    if (size < (size_t) LLAMA_PERSIST_META_HEADER) {
        return false;
    }
    if (llama_persist_le_get32(data + OFF_MAGIC) != LLAMA_PERSIST_META_MAGIC) {
        return false;
    }
    if (llama_persist_le_get32(data + OFF_VERSION) != LLAMA_PERSIST_META_VERSION) {
        return false;
    }
    if (llama_persist_le_get64(data + OFF_PAD) != 0) {
        return false;
    }

    const uint32_t n_tokens = llama_persist_le_get32(data + OFF_N_TOKENS);
    const uint32_t n_spec   = llama_persist_le_get32(data + OFF_N_SPEC);

    // overflow check BEFORE the multiply: a corrupted n_tokens must not wrap
    // the size_t arithmetic below into accepting a truncated buffer
    if ((size_t) n_tokens > (SIZE_MAX - (size_t) LLAMA_PERSIST_META_HEADER - (size_t) n_spec) / 4) {
        return false;
    }
    // exact length: no truncated arrays, no extra trailing bytes
    if (size != (size_t) LLAMA_PERSIST_META_HEADER + (size_t) n_tokens * 4 + (size_t) n_spec) {
        return false;
    }

    llama_persist_meta m;
    memcpy(m.fingerprint, data + OFF_FINGERPRINT, 16);
    m.drafter    = llama_persist_le_get32(data + OFF_DRAFTER);
    m.size_main  = llama_persist_le_get64(data + OFF_SIZE_MAIN);
    m.size_drft  = llama_persist_le_get64(data + OFF_SIZE_DRFT);
    m.crc_main   = llama_persist_le_get32(data + OFF_CRC_MAIN);
    m.crc_drft   = llama_persist_le_get32(data + OFF_CRC_DRFT);
    m.hits       = llama_persist_le_get32(data + OFF_HITS);
    m.created_at = llama_persist_le_get64(data + OFF_CREATED_AT);
    m.last_used  = llama_persist_le_get64(data + OFF_LAST_USED);

    const uint8_t * tokens_data = data + (size_t) LLAMA_PERSIST_META_HEADER;
    m.tokens.resize(n_tokens);
    for (size_t i = 0; i < m.tokens.size(); i++) {
        m.tokens[i] = (llama_token) llama_persist_le_get32(tokens_data + i * 4);
    }

    m.spec.assign(tokens_data + m.tokens.size() * 4, data + size);

    out = std::move(m);
    return true;
}

// ---------------------------------------------------------------------------
// CRC-32 (IEEE 802.3, reflected, poly 0xEDB88320)
// ---------------------------------------------------------------------------

// magic-static initialization: thread-safe even if called from server task threads
static const std::array<uint32_t, 256> & crc32_table() {
    static const auto table = [] {
        std::array<uint32_t, 256> t{};
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t c = i;
            for (int k = 0; k < 8; k++) {
                c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            }
            t[i] = c;
        }
        return t;
    }();
    return table;
}

static uint32_t crc32_update(uint32_t c, const uint8_t * data, size_t size) {
    const auto & table = crc32_table();
    for (size_t i = 0; i < size; i++) {
        c = table[(c ^ data[i]) & 0xFFu] ^ (c >> 8);
    }
    return c;
}

// 128-bit fold state, kept as two scalars so no vector type crosses the
// target-attribute boundary below (the intrinsics live inside the functions)
struct crc32_fold_state {
    uint64_t lo = 0;
    uint64_t hi = 0;
};

#if defined(__x86_64__)
// pclmulqdq folding path, bit-exact with the table path above (same poly,
// init, final xor - derivation validated against a reference harness in
// the wave-5 experiment notes). The reflected
// table CRC is the byte-bit-reversed image of a CRC in the normal ring
// F = x^32 + 0x04C11DB7, where norm(init, M) = (init*x^8n + M(x)*x^32) mod F
// with M laid out MSB-first. The fold keeps T == prefix(x) (deg <= 94),
// absorbing 16 B per iteration (2 pclmul); the final reduction folds in
// init*x^8n and bit-reverses back into the reflected domain, so the < 16 tail
// bytes (and the whole fallback path) continue through crc32_update unchanged.

// a mod F for deg(a) <= 63 (the reduction constant carries the x^32 term)
static uint32_t crc32_gf_mod64(uint64_t a) {
    for (int i = 63; i >= 32; i--) {
        if (a & (1ull << i)) {
            a ^= ((((uint64_t) 1) << 32) | (uint64_t) 0x04C11DB7ull) << (i - 32);
        }
    }
    return (uint32_t) a;
}

// a mod F for deg(a) <= 127 (no shift beyond bit 127: no UB)
static uint32_t crc32_gf_mod128(unsigned __int128 a) {
    for (int i = 127; i >= 32; i--) {
        if (a & ((unsigned __int128) 1 << i)) {
            a ^= ((((unsigned __int128) 1) << 32) | (unsigned __int128) 0x04C11DB7ull) << (i - 32);
        }
    }
    return (uint32_t) a;
}

// (a*b) mod F, a,b < 2^32
static uint32_t crc32_gf_mul32(uint32_t a, uint32_t b) {
    uint64_t acc = 0;
    uint64_t bb   = b;
    while (a) {
        if (a & 1) {
            acc ^= bb;
        }
        bb <<= 1;
        a >>= 1;
    }
    return crc32_gf_mod64(acc);
}

// x^e mod F
static uint32_t crc32_gf_powx(uint64_t e) {
    uint32_t r    = 1;
    uint32_t base = 2;
    while (e) {
        if (e & 1) {
            r = crc32_gf_mul32(r, base);
        }
        base = crc32_gf_mul32(base, base);
        e >>= 1;
    }
    return r;
}

static uint32_t crc32_bitrev32(uint32_t v) {
    v = ((v & 0x55555555u) << 1) | ((v & 0xAAAAAAAAu) >> 1);
    v = ((v & 0x33333333u) << 2) | ((v & 0xCCCCCCCCu) >> 2);
    v = ((v & 0x0F0F0F0Fu) << 4) | ((v & 0xF0F0F0F0u) >> 4);
    v = ((v & 0x00FF00FFu) << 8) | ((v & 0xFF00FF00u) >> 8);
    return (v << 16) | (v >> 16);
}

// folding constants x^e mod F, derived at first use (magic static, like the
// table) instead of transcribed literals - a typo in a magic constant would
// silently break bit-exactness; the assert re-derives one of them the slow
// way (96 successive multiplications by x) as a one-off guard
struct crc32_fold_k {
    uint32_t k32;
    uint32_t k96;
    uint32_t k128;
    uint32_t k192;
};

static const crc32_fold_k & crc32_fold_constants() {
    static const crc32_fold_k k = {
        crc32_gf_powx(32),
        crc32_gf_powx(96),
        crc32_gf_powx(128),
        crc32_gf_powx(192),
    };
    assert(k.k96 == [] {
        uint32_t r = 1;
        for (int i = 0; i < 96; i++) {
            r = crc32_gf_mul32(r, 2);
        }
        return r;
    }());
    return k;
}

// folds size bytes into st; size must be a multiple of 16 (crc32_stream below
// owns the < 16 remainder), 16 B per iteration
__attribute__((target("pclmul,ssse3")))
static void crc32_fold_update(crc32_fold_state & st, const uint8_t * data, size_t size) {
    assert((size & 15) == 0);
    const crc32_fold_k & k = crc32_fold_constants();
    const __m128i k128 = _mm_set_epi64x(0, (int64_t) k.k128);
    const __m128i k192 = _mm_set_epi64x(0, (int64_t) k.k192);

    // per-byte bit reversal via nibble LUT, then byte-order reversal: the
    // chunk as M' = [br8(b)] laid out MSB-first, ready for the normal ring
    const __m128i lut = _mm_setr_epi8(0x0, 0x8, 0x4, 0xC, 0x2, 0xA, 0x6, 0xE,
                                      0x1, 0x9, 0x5, 0xD, 0x3, 0xB, 0x7, 0xF);
    const __m128i m0f = _mm_set1_epi8((char) 0x0F);
    const __m128i brb = _mm_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8,
                                      7, 6, 5, 4, 3, 2, 1, 0);

    __m128i T = _mm_set_epi64x((int64_t) st.hi, (int64_t) st.lo);

    const uint8_t * p     = data;
    const uint8_t * end16 = data + size;
    while (p < end16) {
        __m128i x  = _mm_loadu_si128((const __m128i *) p);
        __m128i lo = _mm_and_si128(x, m0f);
        __m128i hi = _mm_and_si128(_mm_srli_epi16(x, 4), m0f);
        __m128i rb = _mm_or_si128(_mm_slli_epi16(_mm_shuffle_epi8(lut, lo), 4),
                                  _mm_shuffle_epi8(lut, hi));
        __m128i C  = _mm_shuffle_epi8(rb, brb);

        __m128i a = _mm_clmulepi64_si128(T, k192, 0x01); // hi64(T) * x^192
        __m128i b = _mm_clmulepi64_si128(T, k128, 0x00); // lo64(T) * x^128
        T = _mm_xor_si128(_mm_xor_si128(a, b), C);
        p += 16;
    }

    st.lo = (uint64_t) _mm_cvtsi128_si64(T);
    st.hi = (uint64_t) _mm_cvtsi128_si64(_mm_unpackhi_epi64(T, T));
}

// reduces the fold state together with init*x^8n: returns the REFLECTED
// running register after the n_folded bytes already absorbed
__attribute__((target("pclmul,ssse3")))
static uint32_t crc32_fold_final(const crc32_fold_state & st, uint64_t n_folded) {
    const crc32_fold_k & k = crc32_fold_constants();
    const __m128i k32 = _mm_set_epi64x(0, (int64_t) k.k32);
    const __m128i k96 = _mm_set_epi64x(0, (int64_t) k.k96);
    const __m128i T   = _mm_set_epi64x((int64_t) st.hi, (int64_t) st.lo);

    // T(x)*x^32 = lo64(T)*x^32 + hi64(T)*x^96; each product fits in 96 bits,
    // so XOR them in-register before extracting (no high bits can be lost)
    __m128i pl = _mm_clmulepi64_si128(T, k32, 0x00);
    __m128i ph = _mm_clmulepi64_si128(T, k96, 0x01);
    __m128i ps = _mm_xor_si128(pl, ph);
    unsigned __int128 U = (unsigned __int128) (uint64_t) _mm_cvtsi128_si64(ps)
                        | ((unsigned __int128) (uint64_t) _mm_cvtsi128_si64(
                              _mm_unpackhi_epi64(ps, ps)) << 64);
    U ^= (unsigned __int128) crc32_gf_mul32(0xFFFFFFFFu, crc32_gf_powx(8ull * n_folded));

    return crc32_bitrev32(crc32_gf_mod128(U));
}

// runtime dispatch, cached in a magic static (thread-safe); the TU builds
// without -mpclmul: the intrinsics are behind the target attributes above
static bool crc32_pclmul_available() {
    static const bool ok = __builtin_cpu_supports("pclmul")
                        && __builtin_cpu_supports("ssse3");
    return ok;
}
#else
// no pclmulqdq path on this target: always the table
static bool crc32_pclmul_available() {
    return false;
}
static inline void crc32_fold_update(crc32_fold_state &, const uint8_t *, size_t) {
    assert(false); // unreachable: crc32_stream only folds when pclmulqdq is available
}
static inline uint32_t crc32_fold_final(const crc32_fold_state &, uint64_t) {
    assert(false);
    return 0;
}
#endif

// streaming CRC-32: table fallback or pclmulqdq fold, picked once per stream.
// The fold path parks a < 16 B remainder in carry so update() is agnostic of
// the caller's chunking; both paths produce the identical reflected state
// after every byte (the normal-domain fold composes across calls exactly like
// the reflected table update does).
struct crc32_stream {
    bool             fold   = crc32_pclmul_available();
    uint32_t         c      = 0xFFFFFFFFu; // reflected state (table path)
    crc32_fold_state st;                   // fold path
    uint64_t         folded = 0;           // bytes absorbed by the fold
    uint8_t          carry[16] = {0};      // fold-path remainder (< 16 B)
    size_t           ncarry = 0;
};

static void crc32_stream_update(crc32_stream & s, const uint8_t * data, size_t size) {
    if (!s.fold) {
        s.c = crc32_update(s.c, data, size);
        return;
    }
    // top the carry up to a full 16-byte block first: fold blocks always sit
    // on absolute 16-byte boundaries of the stream, like in the one-shot call
    while (s.ncarry > 0 && size > 0) {
        s.carry[s.ncarry++] = *data++;
        size--;
        if (s.ncarry == 16) {
            crc32_fold_update(s.st, s.carry, 16);
            s.folded += 16;
            s.ncarry = 0;
        }
    }
    const size_t bulk = size & ~(size_t) 15;
    crc32_fold_update(s.st, data, bulk);
    s.folded += bulk;
    data += bulk;
    size -= bulk;
    for (size_t i = 0; i < size; i++) {
        s.carry[s.ncarry++] = data[i];
    }
}

static uint32_t crc32_stream_final(const crc32_stream & s) {
    if (!s.fold) {
        return s.c ^ 0xFFFFFFFFu;
    }
    const uint32_t core = crc32_fold_final(s.st, s.folded);
    return crc32_update(core, s.carry, s.ncarry) ^ 0xFFFFFFFFu;
}

// sanity check-value (CRC-32 IEEE): "123456789" -> 0xCBF43926
uint32_t llama_persist_crc32(const uint8_t * data, size_t size) {
    crc32_stream s;
    crc32_stream_update(s, data, size);
    return crc32_stream_final(s);
}

// W6-5/W6-7 (A3): incremental wrapper over crc32_stream - see the header. The
// pimpl keeps the fold state (and its pclmul-vs-table dispatch decision) out
// of the ABI; update() composes exactly like feeding the whole buffer at once.
struct llama_persist_crc32_folder::impl {
    crc32_stream st;
    bool finalized = false;
};

llama_persist_crc32_folder::llama_persist_crc32_folder() : p(new impl) {}

llama_persist_crc32_folder::~llama_persist_crc32_folder() {
    delete p;
}

llama_persist_crc32_folder::llama_persist_crc32_folder(llama_persist_crc32_folder && other) noexcept : p(other.p) {
    other.p = nullptr;
}

llama_persist_crc32_folder & llama_persist_crc32_folder::operator=(llama_persist_crc32_folder && other) noexcept {
    if (this != &other) {
        delete p;
        p = other.p;
        other.p = nullptr;
    }
    return *this;
}

void llama_persist_crc32_folder::update(const void * data, size_t size) {
    assert(p != nullptr && !p->finalized);
    if (size > 0) {
        crc32_stream_update(p->st, (const uint8_t *) data, size);
    }
}

uint32_t llama_persist_crc32_folder::finalize() {
    assert(p != nullptr && !p->finalized);
    p->finalized = true;
    return crc32_stream_final(p->st);
}

bool llama_persist_crc32_file(const char * path, uint64_t expected_size, uint32_t * crc_out) {
    const int fd = open(path, O_RDONLY);
    if (fd < 0) {
        return false;
    }

    // read the WHOLE file in 1 MiB blocks; it must contain exactly
    // expected_size bytes - shorter OR longer means corruption.
    // heap buffer: this runs on server task threads, keep the stack flat
    std::vector<uint8_t> buf(1 << 20);
    crc32_stream crc;
    uint64_t    total = 0;
    bool        ok    = true;

    for (;;) {
        ssize_t n;
        do {
            n = read(fd, buf.data(), buf.size());
        } while (n < 0 && errno == EINTR);
        if (n < 0) {
            ok = false; // IO error
            break;
        }
        if (n == 0) {
            break; // EOF
        }
        crc32_stream_update(crc, buf.data(), (size_t) n);
        total += (uint64_t) n;
        if (total > expected_size) {
            ok = false; // longer than declared: stop early instead of streaming a bad file
            break;
        }
    }

    // drop what we just streamed from the page cache (pattern: llama-mmap.cpp,
    // llama-ple-store.cpp) - a CRC check must not evict hot cache pages
    (void) posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    close(fd);

    if (!ok || total != expected_size) {
        return false; // crc_out untouched on failure
    }
    *crc_out = crc32_stream_final(crc);
    return true;
}

// ---------------------------------------------------------------------------
// eviction score
// ---------------------------------------------------------------------------

double llama_persist_eviction_score(uint32_t hits, uint64_t last_used, uint64_t now_s) {
    if (now_s <= last_used) return (double) hits;          // clock skew guard
    const double age = (double)(now_s - last_used);
    return (double) hits * std::exp2(-age / (double) LLAMA_PERSIST_HALF_LIFE_S);
}

// ---------------------------------------------------------------------------
// config fingerprint: FNV-1a 64 with two seeds (not cryptographic - a config
// disambiguation guard, so the seeds just halve the collision odds of one hash)
// ---------------------------------------------------------------------------

static const uint64_t FNV1A_PRIME   = 0x100000001b3ull;        // 1099511628211
static const uint64_t FNV1A_SEED_A  = 0xcbf29ce484222325ull;   // FNV-1a offset basis
static const uint64_t FNV1A_SEED_B  = 0x9E3779B97F4A7C15ull;   // golden-ratio constant

struct fnv1a_pair {
    uint64_t h0 = FNV1A_SEED_A;
    uint64_t h1 = FNV1A_SEED_B;
};

static void fnv1a_update(fnv1a_pair & p, uint8_t b) {
    p.h0 = (p.h0 ^ (uint64_t) b) * FNV1A_PRIME;
    p.h1 = (p.h1 ^ (uint64_t) b) * FNV1A_PRIME;
}

static void fnv1a_update_u32(fnv1a_pair & p, uint32_t v) {
    uint8_t b[4];
    llama_persist_le_put32(b, v);
    for (int i = 0; i < 4; i++) fnv1a_update(p, b[i]);
}

static void fnv1a_update_u64(fnv1a_pair & p, uint64_t v) {
    uint8_t b[8];
    llama_persist_le_put64(b, v);
    for (int i = 0; i < 8; i++) fnv1a_update(p, b[i]);
}

// variable-length string + one trailing NUL: without the terminator the
// boundary between a basename and the fixed-size field that follows it would
// be ambiguous ("ab" + u32(0x63...) == "abc" + u32(0x...))
static void fnv1a_update_str(fnv1a_pair & p, const char * s) {
    for (const char * c = s; *c != '\0'; c++) {
        fnv1a_update(p, (uint8_t) *c);
    }
    fnv1a_update(p, 0);
}

void llama_persist_fingerprint(const char * model_path, uint64_t model_size,
                               const char * drafter_path, uint64_t drafter_size,
                               const char * cache_kv, uint32_t state_version,
                               uint8_t out[16]) {
    fnv1a_pair p;

    fnv1a_update_str(p, llama_persist_basename(model_path));
    fnv1a_update_u64(p, model_size);
    fnv1a_update_str(p, llama_persist_basename(drafter_path)); // NULL -> "" (absent)
    fnv1a_update_u64(p, drafter_size);
    fnv1a_update_str(p, cache_kv ? cache_kv : ""); // NULL -> "" (absent), like the drafter
    fnv1a_update_u32(p, state_version);

    llama_persist_le_put64(out + 0, p.h0);
    llama_persist_le_put64(out + 8, p.h1);
}

const char * llama_persist_basename(const char * path) {
    if (path == nullptr) {
        return "";
    }
    const char * base = path;
    for (const char * c = path; *c != '\0'; c++) {
        if (*c == '/' || *c == '\\') {
            base = c + 1;
        }
    }
    return base;
}
