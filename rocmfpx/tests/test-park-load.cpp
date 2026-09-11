// Unit tests for the T32-b park candidate in load() (park: park candidate in
// load with existing rules). Zero GPU, zero llama runtime: real files in a
// mkdtemp tmpdir, real server_prompt_cache instances, real boot adoption, real
// advisory locks - the SELECTION runs on the genuine
// server_prompt_cache::load() path.
//
// WHAT LEVEL EACH INVARIANT IS PROVEN AT (be explicit, per the plan):
//   SELECTION + OWNER ROUTING (cases a, c, parity, e, park=nullptr): proven
//   on the REAL production path - general.load(..., park, &owner) with two
//   live instances and injected candidates. A winning disk candidate is
//   arranged to stop at the load-time CRC gate (the fixture's sidecar carries
//   a deliberately wrong crc_main): the physical restore then never touches a
//   llama_context (which the test does not have), and the rejection's erase -
//   which entry disappeared from WHICH library - is the observable proof of
//   both the winner and the owning instance.
//
//   (b) RAM/PARK PARITY -> RAM WINS: NOT driven end-to-end here - a winning
//   RAM candidate restores through llama_state_seq_set_data_ext, which
//   dereferences the context (the known Task-4 test limit). Proven instead at
//   the pure-ranking level: server_prompt_cache_candidate_better() is strictly
//   better, so at parity the earlier-scanned source (RAM) keeps the win; the
//   scan order (RAM -> general disk -> park) is inspection-covered in load().
//
//   (d) STATEFUL PARK ENTRY WITH BOUNDED TRAILING-RM: the trailing-rollback
//   branch reads the memories' RS capacities (llama_n_rs_seq), unreachable
//   without a context. The RULE itself is proven purely on
//   server_prompt_cache_spec_boundary_valid(); that the park scan applies the
//   very same lambda is structural (one shared scan_disk_library) and
//   inspection-covered.
//
//   ACCEPT-SIDE BOOKKEEPING (hits++/last_used/LRU splice/sidecar rewrite +
//   the "persist restore: park" marker + park_restore_crc_ms): lives in
//   accept_disk_load, which the slot calls on the instance load() names in
//   disk_entry_owner - the routing asserted below. accept_disk_load itself is
//   pre-existing T23 code; driving it needs a successful physical restore
//   (a llama runtime). Inspection-covered here.
//
// f_keep/sim for disk candidates derive from token overlap - fixtures use
// distinct token values so the common prefixes are exactly the intended ones.

#include "../src/llama-persist-meta.h"
#include "../tools/server/server-task.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <cstdlib> // mkdtemp

namespace fs = std::filesystem;

static int g_failures = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("FAIL: %s\n", msg); g_failures++; } \
} while (0)

// one valid persist entry below `persist_dir` (fixture shape of
// tests/test-park-redirect.cpp, extended): entry-<id>/meta +
// state-<id>-target.bin of EXACTLY `target_bytes` bytes (boot scan and the
// load-time size check are both exact), optional spec bytes, and a
// DELIBERATELY WRONG crc_main - a winning candidate then stops at load_disk's
// CRC gate, before any llama_context use (see the file header).
static bool write_fixture(const std::string & persist_dir, uint64_t id, const uint8_t fingerprint[16],
                          size_t target_bytes, uint64_t last_used, const std::vector<llama_token> & tokens,
                          const std::vector<uint8_t> & spec) {
    const std::string entry_dir = persist_dir + "/entry-" + std::to_string(id);
    std::error_code ec;
    fs::create_directories(entry_dir, ec);
    if (ec) {
        return false;
    }

    const std::string stem = "state-" + std::to_string(id);
    std::vector<char> payload(target_bytes, '\x5a');
    {
        std::ofstream target(entry_dir + "/" + stem + "-target.bin", std::ios::binary | std::ios::trunc);
        target.write(payload.data(), (std::streamsize) payload.size());
        if (!target.good()) {
            return false;
        }
    }

    llama_persist_meta meta;
    memcpy(meta.fingerprint, fingerprint, sizeof(meta.fingerprint));
    meta.drafter    = 0; // COMMON_SPECULATIVE_TYPE_NONE
    meta.size_main  = target_bytes;
    meta.size_drft  = 0;
    // wrong on purpose: guarantees the load-time CRC gate rejects a selected
    // entry deterministically (xor keeps it a valid nonzero-looking crc)
    meta.crc_main   = llama_persist_crc32((const uint8_t *) payload.data(), payload.size()) ^ 1u;
    meta.hits       = 1;
    meta.created_at = last_used - 10;
    meta.last_used  = last_used;
    meta.tokens     = tokens;
    meta.spec       = spec;

    std::vector<uint8_t> blob;
    if (!llama_persist_meta_serialize(meta, blob)) {
        return false;
    }
    {
        std::ofstream m(entry_dir + "/meta", std::ios::binary | std::ios::trunc);
        m.write(reinterpret_cast<const char *>(blob.data()), (std::streamsize) blob.size());
        if (!m.good()) {
            return false;
        }
    }
    return true;
}

// sorted names of entry-* directories currently below a library dir
static std::vector<std::string> entry_dirs(const std::string & dir) {
    std::vector<std::string> res;
    std::error_code ec;
    for (const auto & entry : fs::directory_iterator(fs::u8path(dir), ec)) {
        if (ec) {
            break;
        }
        const std::string name = entry.path().filename().string();
        if (name.rfind("entry-", 0) == 0 && entry.is_directory()) {
            res.push_back(name);
        }
    }
    std::sort(res.begin(), res.end());
    return res;
}

// token streams with the intended exact prefixes: `prefix` shared tokens then
// distinct tails, so cross-entry common prefixes are never accidental
static std::vector<llama_token> toks_prefix(size_t n, llama_token base) {
    std::vector<llama_token> res(n);
    for (size_t i = 0; i < n; i++) {
        res[i] = base + (llama_token) i;
    }
    return res;
}

static std::vector<llama_token> concat(const std::vector<llama_token> & a, const std::vector<llama_token> & b) {
    std::vector<llama_token> res = a;
    res.insert(res.end(), b.begin(), b.end());
    return res;
}

// a RAM candidate for the general instance's states list: tokens + a spec
// blob (required in spec mode), empty KV payloads (it must only ever LOSE -
// a winning RAM candidate would restore through the absent context)
static server_prompt ram_candidate(const std::vector<llama_token> & toks) {
    server_prompt p;
    p.tokens = server_tokens(toks, /*has_mtmd=*/ false);
    p.data.spec = {0x1, 0x2, 0x3};
    return p;
}

int main() {
    char tmpl[] = "/tmp/llama-test-park-load-XXXXXX";
    const char * tmp = mkdtemp(tmpl);
    CHECK(tmp != nullptr, "mkdtemp created the test tmpdir");
    if (tmp == nullptr) {
        printf("test-park-load: %d FAILURES\n", g_failures + 1);
        return 1;
    }

    const std::string base = tmp;                                  // as --cache-disk PATH
    const std::string root = base + "/.llama-prompt-cache-v1";     // ctor cache namespace

    uint8_t fp[16];
    for (size_t i = 0; i < sizeof(fp); i++) {
        fp[i] = (uint8_t) (0xA0 + i);
    }

    // scaled-down shape of the plan's case (a): request 40, RAM 30,
    // general 31, park 32 - all exact prefixes of the request (spec mode
    // ranks by boundary length)
    const std::vector<llama_token> request40 = toks_prefix(40, 1000);
    const std::vector<llama_token> toks_ram30   (request40.begin(), request40.begin() + 30);
    const std::vector<llama_token> toks_gen31   (request40.begin(), request40.begin() + 31);
    const std::vector<llama_token> toks_park32  (request40.begin(), request40.begin() + 32);
    const std::vector<uint8_t> spec3 = {0x1, 0x2, 0x3};

    // --- A. case (a): RAM 30 / general 31 / park 32 -> PARK wins (longest
    //        boundary, spec mode). The park entry is the one taken: its
    //        load attempt (CRC stop) erases it from the PARK library, the
    //        general entry and the RAM copy stay untouched, and load() names
    //        the park as the owning instance ---
    {
        const std::string gen_dir   = root + "/gen-a";
        const std::string park_dir  = root + "/park-a";
        CHECK(write_fixture(gen_dir, 5, fp, 512, 1756684800ull, toks_gen31, spec3),
            "A: general entry-5 (31-token boundary) written");
        CHECK(write_fixture(park_dir, 7, fp, 512, 1756684800ull, toks_park32, spec3),
            "A: park entry-7 (32-token boundary) written");

        server_prompt_cache general(0, 4096, base, 64, fp, 64, 0, "gen-a");
        server_prompt_cache park(0, 4096, base, 64, fp, 64, 0, "park-a", true);
        general.states.push_back(ram_candidate(toks_ram30));
        CHECK(general.disk_states.size() == 1 && park.disk_states.size() == 1,
            "A: both libraries adopted their fixture entry");

        server_prompt prompt;
        bool cache_hit = true;
        uint64_t disk_entry_id = 12345; // poison: must stay 0 (the arranged reject reports no id)
        bool tag_mismatch = true;
        server_prompt_cache * owner = nullptr;

        const bool res = general.load(prompt, server_tokens(request40, false),
                nullptr, nullptr, /*id_slot=*/ 0,
                /*spec_state_required=*/ true, /*spec_trailing_rm=*/ true,
                &cache_hit, &disk_entry_id, COMMON_SPECULATIVE_TYPE_NONE, &tag_mismatch,
                &park, &owner);

        CHECK(!res, "A: load returns false (the arranged CRC-stop reject)");
        CHECK(!cache_hit && disk_entry_id == 0 && !tag_mismatch,
            "A: no cache hit, no entry id, no tag mismatch reported");
        CHECK(owner == &park, "A: the PARK instance is named owner of the selected candidate");
        CHECK(park.disk_states.empty() && entry_dirs(park_dir).empty(),
            "A: the PARK entry was taken (attempted, rejected, erased from the park)");
        CHECK(general.disk_states.size() == 1 && entry_dirs(gen_dir) == std::vector<std::string>({"entry-5"}),
            "A: the general entry lost and is untouched");
        CHECK(general.states.size() == 1 && general.states.front().tokens.size() == 30,
            "A: the RAM candidate lost and stays in the RAM states");
        CHECK(park.get_persist_stats().loads == 0,
            "A: a rejected restore is no persist load (park counters)");
    }

    // --- B. case (c): park STALE (boundary shorter than the general's) ->
    //        existing rules apply, the GENERAL entry wins. Mirror observable
    //        of A on the other instance ---
    {
        const std::string gen_dir   = root + "/gen-b";
        const std::string park_dir  = root + "/park-b";
        CHECK(write_fixture(gen_dir, 5, fp, 512, 1756684800ull, toks_park32, spec3),
            "B: general entry-5 (32-token boundary) written");
        CHECK(write_fixture(park_dir, 7, fp, 512, 1756684800ull, toks_ram30, spec3),
            "B: park entry-7 (30-token stale boundary) written");

        server_prompt_cache general(0, 4096, base, 64, fp, 64, 0, "gen-b");
        server_prompt_cache park(0, 4096, base, 64, fp, 64, 0, "park-b", true);

        server_prompt prompt;
        bool cache_hit = false;
        uint64_t disk_entry_id = 0;
        bool tag_mismatch = false;
        server_prompt_cache * owner = nullptr;

        const bool res = general.load(prompt, server_tokens(request40, false),
                nullptr, nullptr, 0, true, true,
                &cache_hit, &disk_entry_id, COMMON_SPECULATIVE_TYPE_NONE, &tag_mismatch,
                &park, &owner);

        CHECK(!res, "B: load returns false (the arranged CRC-stop reject)");
        CHECK(owner == &general, "B: the GENERAL instance owns the selected candidate");
        CHECK(general.disk_states.empty() && entry_dirs(gen_dir).empty(),
            "B: the GENERAL entry was taken (attempted, rejected, erased)");
        CHECK(park.disk_states.size() == 1 && entry_dirs(park_dir) == std::vector<std::string>({"entry-7"}),
            "B: the stale park entry lost and is untouched");
    }

    // --- B2. general/park parity on disk -> the GENERAL library wins (scan
    //         order + strictly-better ranking; the disk-level mirror of the
    //         RAM-first preference) ---
    {
        const std::string gen_dir   = root + "/gen-b2";
        const std::string park_dir  = root + "/park-b2";
        CHECK(write_fixture(gen_dir, 5, fp, 512, 1756684800ull, toks_park32, spec3),
            "B2: general entry-5 (32-token boundary) written");
        CHECK(write_fixture(park_dir, 7, fp, 512, 1756684800ull, toks_park32, spec3),
            "B2: park entry-7 (same 32-token boundary) written");

        server_prompt_cache general(0, 4096, base, 64, fp, 64, 0, "gen-b2");
        server_prompt_cache park(0, 4096, base, 64, fp, 64, 0, "park-b2", true);

        server_prompt prompt;
        bool cache_hit = false;
        uint64_t disk_entry_id = 0;
        bool tag_mismatch = false;
        server_prompt_cache * owner = nullptr;

        const bool res = general.load(prompt, server_tokens(request40, false),
                nullptr, nullptr, 0, true, true,
                &cache_hit, &disk_entry_id, COMMON_SPECULATIVE_TYPE_NONE, &tag_mismatch,
                &park, &owner);

        CHECK(!res, "B2: load returns false (the arranged CRC-stop reject)");
        CHECK(owner == &general, "B2: at parity the GENERAL library wins (park scanned last)");
        CHECK(general.disk_states.empty(), "B2: the general entry was taken");
        CHECK(park.disk_states.size() == 1, "B2: the equal park entry is untouched");
    }

    // --- C. case (e): park f_keep < 0.25 -> DISCARDED by the existing floor.
    //        Park entry of 100 tokens sharing only the first 10 with the
    //        request (f_keep = 0.1); no other candidate anywhere ---
    {
        const std::string park_dir = root + "/park-c";
        const std::vector<llama_token> park_tail  = toks_prefix(90, 2000);
        const std::vector<llama_token> req_tail   = toks_prefix(40, 3000);
        const std::vector<llama_token> toks_park100 = concat(toks_prefix(10, 1000), park_tail);
        const std::vector<llama_token> request50    = concat(toks_prefix(10, 1000), req_tail);
        CHECK(write_fixture(park_dir, 7, fp, 512, 1756684800ull, toks_park100, {}),
            "C: park entry-7 (100 tokens, 10-token overlap) written");

        server_prompt_cache general(0, 4096, base, 64, fp, 64, 0, "gen-c");
        server_prompt_cache park(0, 4096, base, 64, fp, 64, 0, "park-c", true);

        server_prompt prompt;
        bool cache_hit = false;
        uint64_t disk_entry_id = 0;
        bool tag_mismatch = false;
        server_prompt_cache * owner = nullptr;

        const bool res = general.load(prompt, server_tokens(request50, false),
                nullptr, nullptr, 0, /*spec_state_required=*/ false, /*spec_trailing_rm=*/ false,
                &cache_hit, &disk_entry_id, COMMON_SPECULATIVE_TYPE_NONE, &tag_mismatch,
                &park, &owner);

        CHECK(res, "C: with every candidate below the floor the base (empty slot prompt) stays valid");
        CHECK(!cache_hit, "C: no cache hit (the park candidate was discarded)");
        CHECK(owner == &general, "C: no park candidate selected - owner stays the general instance");
        CHECK(park.disk_states.size() == 1 && entry_dirs(park_dir) == std::vector<std::string>({"entry-7"}),
            "C: the low-overlap park entry was never attempted (no erase)");
    }

    // --- D. park = nullptr: the general library alone, exactly as before the
    //        park existed. Same shape as A minus the park (and its entry):
    //        the general entry wins over the RAM copy on the general library ---
    {
        const std::string gen_dir = root + "/gen-d";
        CHECK(write_fixture(gen_dir, 5, fp, 512, 1756684800ull, toks_gen31, spec3),
            "D: general entry-5 (31-token boundary) written");

        server_prompt_cache general(0, 4096, base, 64, fp, 64, 0, "gen-d");
        general.states.push_back(ram_candidate(toks_ram30));

        server_prompt prompt;
        bool cache_hit = false;
        uint64_t disk_entry_id = 0;
        bool tag_mismatch = false;
        server_prompt_cache * owner = nullptr;

        const bool res = general.load(prompt, server_tokens(request40, false),
                nullptr, nullptr, 0, true, true,
                &cache_hit, &disk_entry_id, COMMON_SPECULATIVE_TYPE_NONE, &tag_mismatch,
                /*park=*/ nullptr, &owner);

        CHECK(!res, "D: load returns false (the arranged CRC-stop reject)");
        CHECK(owner == &general, "D: without a park the owner is the general instance");
        CHECK(general.disk_states.empty(), "D: the general entry was taken exactly as before");
        CHECK(general.states.size() == 1, "D: the RAM candidate lost and stays");
    }

    // --- E. pure-rule battery (header helpers): the shared ranking and
    //        boundary rules, driven directly - this is where the RAM-parity
    //        (b) and the bounded trailing-rm (d) admissions are proven ---
    {
        // (b) RAM/park parity in spec mode: an equal boundary is NOT strictly
        // better - the earlier-scanned RAM candidate keeps the win
        {
            // trackers as load() initializes them for an empty slot prompt
            float f_keep = -1.0f;
            float sim = 0.0f;
            size_t boundary = 0;
            CHECK(server_prompt_cache_candidate_better(/*spec=*/ true, 40, 30, 30, f_keep, sim, boundary),
                "E(b): the RAM candidate (30/40) improves on the empty base");
            CHECK(boundary == 30, "E(b): the RAM candidate took the win");
            CHECK(!server_prompt_cache_candidate_better(/*spec=*/ true, 40, 30, 30, f_keep, sim, boundary),
                "E(b): an equal-boundary park candidate does NOT supersede the RAM copy");
            CHECK(boundary == 30, "E(b): the best boundary is unchanged by the parity attempt");
            // ...and a strictly longer park candidate does
            CHECK(server_prompt_cache_candidate_better(/*spec=*/ true, 40, 32, 32, f_keep, sim, boundary),
                "E(b): a longer park boundary does supersede");
            CHECK(boundary == 32, "E(b): the best boundary moved to the park's");
        }
        // (b) non-spec parity: both f_keep AND sim must strictly improve
        {
            float f_keep = 0.5f;
            float sim = 0.5f;
            size_t boundary = 0;
            CHECK(!server_prompt_cache_candidate_better(/*spec=*/ false, 40, 20, 40, f_keep, sim, boundary),
                "E(b): equal f_keep/sim (20/40 vs a 0.5/0.5 best) is not better");
            CHECK(server_prompt_cache_candidate_better(/*spec=*/ false, 40, 30, 40, f_keep, sim, boundary),
                "E(b): strictly better on both axes is better");
        }
        // (e) the f_keep floor rejects before any ranking
        {
            float f_keep = -1.0f;
            float sim = -1.0f;
            size_t boundary = 0;
            CHECK(!server_prompt_cache_candidate_better(/*spec=*/ false, 50, 10, 100, f_keep, sim, boundary),
                "E(e): f_keep = 10/100 < 0.25 discards even against an empty base");
            CHECK(f_keep == -1.0f && boundary == 0, "E(e): the trackers are untouched by a discarded candidate");
        }
        // (d) stateful entry, bounded trailing rollback: delta <= n_rs_min
        // admits, delta > n_rs_min rejects - the park obeys the same rule
        // because the park scan applies this very predicate
        {
            CHECK(server_prompt_cache_spec_boundary_valid(true, true, 100, 96, 8),
                "E(d): delta 4 <= bound 8 admits the entry");
            CHECK(!server_prompt_cache_spec_boundary_valid(true, true, 100, 96, 2),
                "E(d): delta 4 > bound 2 rejects the entry");
            CHECK(server_prompt_cache_spec_boundary_valid(true, true, 100, 96, UINT32_MAX),
                "E(d): dense KV (no known bound) admits any bounded rollback");
            CHECK(!server_prompt_cache_spec_boundary_valid(true, false, 100, 96, UINT32_MAX),
                "E(d): no trailing-rm support + boundary mismatch rejects");
            CHECK(server_prompt_cache_spec_boundary_valid(true, true, 100, 100, 0),
                "E(d): an exact boundary needs no rollback (bound irrelevant)");
            CHECK(server_prompt_cache_spec_boundary_valid(false, false, 100, 0, 0),
                "E(d): stateless mode never applies the boundary rule");
        }
    }

    std::error_code rm_ec;
    fs::remove_all(fs::u8path(base), rm_ec);

    if (g_failures == 0) {
        printf("test-park-load: ALL PASS\n");
        return 0;
    }
    printf("test-park-load: %d FAILURES\n", g_failures);
    return 1;
}
