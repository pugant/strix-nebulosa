// Unit tests for the T32-b task-switch save redirect (park: redirect
// disk-save of main at task-switch). Zero GPU, zero llama runtime: real
// files in a mkdtemp tmpdir, real server_prompt_cache instances, real boot
// adoption, real advisory locks.
//
// WHAT LEVEL EACH INVARIANT IS PROVEN AT (be explicit, per the plan):
//   u3 TOUCH (same boundary -> touch, hits++, zero state bytes, no new
//   entry): proven on the REAL production path - server_prompt_cache::save()
//   (park instance, no park argument) drives the existing save_disk
//   dedup/touch loop. Reaching the touch requires NO llama_context (the loop
//   runs before save_disk's first context use), so this is the genuine
//   code path, not a reimplementation.
//
//   u3 SUPERSEDE (grown boundary -> new entry + old erased, max 1 in the
//   park dir): the erase step is persist_park_supersede() - public exactly
//   so this battery can drive it - called here on a park that adopted a
//   hand-written old entry + a stand-in for a just-committed new entry. NOT
//   provable here: the write-new -> fsync -> erase-old ORDER, which lives in
//   save_disk's call placement (strictly after the commit block); driving a
//   real new-entry write needs a llama runtime (llama_state_seq_save_file).
//
//   u5 REDIRECT (park ok -> general library untouched; park fails ->
//   fallback lands on the general): proven on the REAL production sequence -
//   server_prompt_cache::save(prompt, ..., park) with two live instances.
//   Success is arranged as a park touch-hit and the fallback as a general
//   touch-hit so no llama_context is dereferenced; the touch bookkeeping
//   (hits/touches counters) is the observable proof of WHICH instance took
//   the save.
//
//   NOT covered at any level here (documented gaps): the RAM half of the
//   redirect with RAM enabled (llama_state_seq_get_size_ext derefs the
//   context - the general test instances therefore run with RAM disabled and
//   the RAM half is exercised in production by every cache-ram-enabled run);
//   the multimodal park skip log (needs mtmd chunks to build a media-carrying
//   server_tokens; the skip shares the safety-net path proven inert below);
//   the server_slot::prompt_save wrapper and the get_available_slot redirect
//   decision (server_slot is internal to server-context.cpp; the DECISION is
//   park_should_redirect, unit-tested in tests/test-park-identity.cpp §14).

#include "../src/llama-persist-meta.h"
#include "../tools/server/server-task.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include <cstdlib> // mkdtemp

namespace fs = std::filesystem;

static int g_failures = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("FAIL: %s\n", msg); g_failures++; } \
} while (0)

// one valid persist entry below `persist_dir` (same fixture shape as
// tests/test-park-library.cpp): entry-<id>/meta + state-<id>-target.bin of
// EXACTLY `target_bytes` bytes (the boot scan and the touch-path size check
// are both exact; the payload content itself is irrelevant)
static bool write_fixture(const std::string & persist_dir, uint64_t id, const uint8_t fingerprint[16],
                          size_t target_bytes, uint64_t last_used, const std::vector<llama_token> & tokens) {
    const std::string entry_dir = persist_dir + "/entry-" + std::to_string(id);
    std::error_code ec;
    fs::create_directories(entry_dir, ec);
    if (ec) {
        return false;
    }

    const std::string stem = "state-" + std::to_string(id);
    {
        std::ofstream target(entry_dir + "/" + stem + "-target.bin", std::ios::binary | std::ios::trunc);
        const std::vector<char> payload(target_bytes, '\x5a');
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
    meta.hits       = 1;
    meta.created_at = last_used - 10;
    meta.last_used  = last_used;
    meta.tokens     = tokens;

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

// full recursive file -> size map of a library dir (meta sidecars and the
// .persist.lock included): two equal maps mean the library is byte-for-byte
// untouched - not even a metadata rewrite happened
static std::map<std::string, size_t> dir_file_sizes(const std::string & dir) {
    std::map<std::string, size_t> res;
    std::error_code ec;
    for (const auto & entry : fs::recursive_directory_iterator(fs::u8path(dir), ec)) {
        if (ec) {
            break;
        }
        if (entry.is_regular_file()) {
            res[fs::relative(entry.path(), fs::u8path(dir)).string()] = entry.file_size();
        }
    }
    return res;
}

// raw bytes of a single file (for content identity, not just size)
static std::string file_bytes(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

static server_prompt make_prompt(const std::vector<llama_token> & toks) {
    server_prompt p;
    p.tokens = server_tokens(toks, /*has_mtmd=*/ false);
    return p;
}

int main() {
    char tmpl[] = "/tmp/llama-test-park-redirect-XXXXXX";
    const char * tmp = mkdtemp(tmpl);
    CHECK(tmp != nullptr, "mkdtemp created the test tmpdir");
    if (tmp == nullptr) {
        printf("test-park-redirect: %d FAILURES\n", g_failures + 1);
        return 1;
    }

    const std::string base = tmp;                                  // as --cache-disk PATH
    const std::string root = base + "/.llama-prompt-cache-v1";     // ctor cache namespace
    const std::string general_dir = root + "/persist";

    uint8_t fp[16];
    for (size_t i = 0; i < sizeof(fp); i++) {
        fp[i] = (uint8_t) (0xA0 + i);
    }

    const std::vector<llama_token> tokens_8 = {1, 2, 3, 4, 5, 6, 7, 8};
    const std::vector<llama_token> tokens_unrelated = {21, 22, 23, 24};

    // --- A. u3 TOUCH: a second save at the SAME boundary reuses the existing
    //        exact-token dedup/touch - hits++, one touch counter, zero new
    //        state bytes, no new entry directory. Real save() on the park
    //        instance; the touch loop is reached before any context use ---
    const std::string park_dir_a = root + "/park-a";
    CHECK(write_fixture(park_dir_a, 1, fp, 4096, 1756684800ull, tokens_8),
        "fixture entry-1 (boundary of 8 tokens) written below the park dir");
    try {
        server_prompt_cache park(0, 4096, base, 64, fp, 64, 0, "park-a", true);
        CHECK(park.disk_states.size() == 1, "park adopts its fixture entry");
        const auto before = park.get_persist_stats();
        const std::string target_before = file_bytes(park_dir_a + "/entry-1/state-1-target.bin");

        const bool rc = park.save(make_prompt(tokens_8), /*ctx_main=*/ nullptr, /*ctx_drft=*/ nullptr,
                                  /*id_slot=*/ 0, /*state_spec=*/ {}, COMMON_SPECULATIVE_TYPE_NONE);

        CHECK(rc, "same-boundary save returns true (the touch IS a successful save)");
        const auto after = park.get_persist_stats();
        CHECK(after.touches == before.touches + 1, "same-boundary save is a TOUCH (touches++)");
        CHECK(park.disk_states.size() == 1 && park.disk_states.back().hits == 2,
            "the existing entry's hits went 1 -> 2");
        CHECK(entry_dirs(park_dir_a).size() == 1 && entry_dirs(park_dir_a)[0] == "entry-1",
            "no new entry directory in the park");
        CHECK(park.disk_size() == 4096, "accounted library bytes unchanged (zero new state bytes)");
        CHECK(after.bytes_written == before.bytes_written, "touch writes no accounted bytes");
        CHECK(file_bytes(park_dir_a + "/entry-1/state-1-target.bin") == target_before,
            "the state file content is byte-identical (only the meta sidecar is rewritten)");
        CHECK(park.disk_states.back().last_used >= 1756684800ull, "touch refreshed last_used");
    } catch (const std::exception & e) {
        printf("FAIL: unexpected exception: %s\n", e.what());
        g_failures++;
    }

    // --- B. u3 SUPERSEDE: at a GROWN boundary the new entry supersedes the
    //        old one - the park never holds more than ONE entry. The erase
    //        step (persist_park_supersede) is driven directly: a real
    //        new-entry write needs a llama runtime (see the file header) ---
    const std::string park_dir_b = root + "/park-b";
    CHECK(write_fixture(park_dir_b, 1, fp, 4096, 1756684800ull, tokens_8),
        "fixture entry-1 (old boundary, 8 tokens) written");
    CHECK(write_fixture(park_dir_b, 2, fp, 2048, 1756684900ull, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}),
        "fixture entry-2 (grown boundary, 10 tokens - stand-in for the just-committed entry) written");
    try {
        server_prompt_cache park(0, 4096, base, 64, fp, 64, 0, "park-b", true);
        CHECK(park.disk_states.size() == 2, "park adopts both fixtures (the pre-supersede state)");

        const bool ok = park.persist_park_supersede(2);

        CHECK(ok, "supersede succeeds on a healthy park");
        CHECK(park.disk_states.size() == 1 && park.disk_states.front().id == 2,
            "only the new entry remains in the bookkeeping");
        const std::vector<std::string> dirs = entry_dirs(park_dir_b);
        CHECK(dirs.size() == 1 && dirs[0] == "entry-2",
            "the old entry directory is erased on disk, the new one survives (max 1 entry)");
        CHECK(park.disk_size() == 2048, "accounted bytes re-accounted to the surviving entry");
        CHECK(park.get_persist_stats().evictions == 0,
            "supersession is a remove, not a budget eviction (eviction counter untouched)");
    } catch (const std::exception & e) {
        printf("FAIL: unexpected exception: %s\n", e.what());
        g_failures++;
    }

    // --- C. u5 REDIRECT SUCCESS: the park disk-save succeeds -> the GENERAL
    //        library stays byte-for-byte untouched. The real production
    //        sequence (server_prompt_cache::save with a park argument) runs;
    //        the park takes the save as a touch-hit so no context is needed.
    //        The general instance runs with RAM disabled (see the file
    //        header for the RAM-half adaptation) ---
    CHECK(write_fixture(general_dir, 5, fp, 4096, 1756684800ull, tokens_unrelated),
        "fixture entry-5 (an unrelated subagent-sized prompt) written below the GENERAL library");
    const std::string park_dir_c = root + "/park-c";
    CHECK(write_fixture(park_dir_c, 7, fp, 4096, 1756684800ull, tokens_8),
        "fixture entry-7 (the parked main boundary) written below the park dir");
    try {
        server_prompt_cache general(0, 4096, base, 64, fp, 64, 0); // RAM disabled (test adaptation)
        server_prompt_cache park(0, 4096, base, 64, fp, 64, 0, "park-c", true);
        CHECK(general.disk_states.size() == 1 && park.disk_states.size() == 1,
            "both libraries adopt their fixture entry");
        const auto general_files_before = dir_file_sizes(general_dir);
        const auto general_stats_before = general.get_persist_stats();
        const auto park_stats_before = park.get_persist_stats();

        const bool rc = general.save(make_prompt(tokens_8), nullptr, nullptr, 0, {},
                                     COMMON_SPECULATIVE_TYPE_NONE, &park);

        CHECK(rc, "redirected save returns true (park touch-hit; RAM half disabled)");
        const auto park_stats_after = park.get_persist_stats();
        CHECK(park_stats_after.touches == park_stats_before.touches + 1 &&
              park.disk_states.back().hits == 2,
            "the PARK received the disk half (touch on the parked boundary)");
        CHECK(dir_file_sizes(general_dir) == general_files_before,
            "the GENERAL library is byte-for-byte unchanged - not even a meta rewrite");
        const auto general_stats_after = general.get_persist_stats();
        CHECK(general_stats_after.saves == general_stats_before.saves &&
              general_stats_after.touches == general_stats_before.touches &&
              general_stats_after.bytes_written == general_stats_before.bytes_written,
            "no save/touch/byte counters moved on the general instance");
    } catch (const std::exception & e) {
        printf("FAIL: unexpected exception: %s\n", e.what());
        g_failures++;
    }

    // --- D. u5 REDIRECT FAILURE -> SAFETY-NET: the park save fails (a fully
    //        inert park - boot-conflicted by a second instance holding the
    //        same library lock) and the entry lands on the GENERAL library
    //        exactly as before the redirect existed. The fallback's landing
    //        is arranged as a general touch-hit so no context is needed; the
    //        touch bookkeeping is the proof the general disk-save really ran ---
    CHECK(write_fixture(general_dir, 9, fp, 4096, 1756684800ull, tokens_8),
        "fixture entry-9 (the main boundary, pre-parked in the general library) written");
    const std::string park_dir_d = root + "/park-d";
    try {
        server_prompt_cache general(0, 4096, base, 64, fp, 64, 0); // RAM disabled (test adaptation)
        server_prompt_cache park_locker(0, 4096, base, 64, fp, 64, 0, "park-d", true);
        server_prompt_cache park_inert(0, 4096, base, 64, fp, 64, 0, "park-d", true);
        CHECK(park_inert.persist_path.empty() && park_inert.disk_owned_path.empty(),
            "precondition: the boot-conflicted park is fully inert - its save_disk false IS the safety-net trigger");
        const auto general_stats_before = general.get_persist_stats();
        const std::string target_before = file_bytes(general_dir + "/entry-9/state-9-target.bin");

        const bool rc = general.save(make_prompt(tokens_8), nullptr, nullptr, 0, {},
                                     COMMON_SPECULATIVE_TYPE_NONE, &park_inert);

        CHECK(rc, "redirected save still returns true: the safety-net recovered the state on the general library");
        const auto general_stats_after = general.get_persist_stats();
        CHECK(general_stats_after.touches == general_stats_before.touches + 1 &&
              general.disk_states.back().hits == 2,
            "the FALLBACK ran: the general library received the save (touch on entry-9)");
        CHECK(file_bytes(general_dir + "/entry-9/state-9-target.bin") == target_before,
            "the pre-existing general entry is intact (content untouched, only meta rewritten)");
        CHECK(entry_dirs(park_dir_d).empty(),
            "the inert park wrote nothing anywhere");
    } catch (const std::exception & e) {
        printf("FAIL: unexpected exception: %s\n", e.what());
        g_failures++;
    }

    std::error_code rm_ec;
    fs::remove_all(fs::u8path(base), rm_ec);

    if (g_failures == 0) {
        printf("test-park-redirect: ALL PASS\n");
        return 0;
    }
    printf("test-park-redirect: %d FAILURES\n", g_failures);
    return 1;
}
