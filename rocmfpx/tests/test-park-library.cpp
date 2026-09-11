// Unit tests for the T32-b park library instance (server_prompt_cache with a
// parametrized persist subdir + the park ctor flag). Zero GPU, zero llama
// runtime: real files in a mkdtemp tmpdir, real boot scan, real advisory
// locks. The invariants under test:
//   1. the subdir parametrization separates the libraries - an instance on a
//      custom subdir never adopts entries of the general "persist" subdir;
//   2. the ctor creates the library directory + its .persist.lock;
//   3. the identity fingerprint still governs adoption (same fingerprint
//      adopts, different fingerprint leaves the entries on disk as orphans);
//   4. a park instance is persist-only: no per-run directory, RAM disabled,
//      disk_owned_path names the park directory itself, and both the park dir
//      and the general library survive its destruction;
//   5. default-off: without the park flag (the existing call signature) no
//      park directory is ever created. The server_context-level guarantee
//      ("--cache-disk-park-mib 0 -> no park instance, no dir, no log") is the
//      `if (park_cache_enabled)` gate in server-context.cpp load_model() -
//      too heavy to construct here; verified by the gate being the ONLY
//      writer of server_context_impl::park_cache.

#include "../src/llama-persist-meta.h"
#include "../tools/server/server-task.h"

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

// one valid persist entry below `persist_dir`: entry-<id>/meta +
// state-<id>-target.bin of EXACTLY `target_bytes` bytes (the boot scan size
// check is exact; the content itself is irrelevant for adoption)
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

// names of run-* / .deleting-run-* directories currently below the cache root
// (a park instance must never create one)
static std::vector<std::string> run_dirs(const std::string & cache_root) {
    std::vector<std::string> res;
    std::error_code ec;
    for (const auto & entry : fs::directory_iterator(fs::u8path(cache_root), ec)) {
        if (ec) {
            break;
        }
        const std::string name = entry.path().filename().string();
        if (name.rfind("run-", 0) == 0 || name.rfind(".deleting-run-", 0) == 0 || name.rfind(".deleting-", 0) == 0) {
            res.push_back(name);
        }
    }
    return res;
}

int main() {
    char tmpl[] = "/tmp/llama-test-park-library-XXXXXX";
    const char * tmp = mkdtemp(tmpl);
    CHECK(tmp != nullptr, "mkdtemp created the test tmpdir");
    if (tmp == nullptr) {
        printf("test-park-library: %d FAILURES\n", g_failures + 1);
        return 1;
    }

    const std::string base = tmp;                                  // as --cache-disk PATH
    const std::string root = base + "/.llama-prompt-cache-v1";     // ctor cache namespace
    const std::string general_dir = root + "/persist";
    const std::string park_dir     = root + "/test-park";

    uint8_t fp[16];
    for (size_t i = 0; i < sizeof(fp); i++) {
        fp[i] = (uint8_t) (0xA0 + i);
    }
    uint8_t fp_other[16];
    for (size_t i = 0; i < sizeof(fp_other); i++) {
        fp_other[i] = (uint8_t) (0x10 + i);
    }

    // two valid entries in the GENERAL subdir, same fingerprint
    CHECK(write_fixture(general_dir, 1, fp, 4096,  1756684800ull, {1, 2, 3, 4, 5, 6, 7, 8}),
        "fixture entry-1 written below the general subdir");
    CHECK(write_fixture(general_dir, 2, fp, 8192, 1756684900ull, {11, 12, 13, 14, 15, 16, 17, 18}),
        "fixture entry-2 written below the general subdir");

    // --- 1 + 4: park instance on a custom subdir, alive at the same time as a
    //            general instance (both hold their own per-directory lock) ---
    try {
        server_prompt_cache general(0, 4096, base, 64, fp, 64, 0);
        server_prompt_cache park(0, 4096, base, 64, fp, 64, 0, "test-park", true);

        // scan separation: same cache root, same fingerprint, different subdir
        CHECK(general.persist_path == general_dir, "default subdir instance points at the general library");
        CHECK(park.persist_path == park_dir, "custom subdir instance points at its own library");
        CHECK(general.disk_states.size() == 2, "general instance adopts both same-fingerprint entries");
        CHECK(park.disk_states.empty(), "custom-subdir instance adopts NOTHING from the general subdir");
        CHECK(park.disk_size() == 0, "park instance accounts zero library bytes");

        // ctor effects: dir + per-dir lock, no per-run dir, RAM off, budget on
        CHECK(fs::is_directory(fs::u8path(park_dir)), "park ctor created its library directory");
        CHECK(fs::is_regular_file(fs::u8path(park_dir + "/.persist.lock")), "park ctor created .persist.lock");
        CHECK(run_dirs(root).size() == 1, "exactly one run dir exists (the general instance's) - the park created none");
        CHECK(!park.ram_enabled, "ram 0 -> RAM disabled (ram_enabled = limit_mib != 0)");
        CHECK(park.park_library, "park ctor flag recorded");
        CHECK(park.disk_owned_path == park_dir, "disk_owned_path is the park dir itself (save_disk guard needs it non-empty)");
        CHECK(park.disk_limit_size == 64ull*1024ull*1024ull, "park disk budget reflects the constructor mib figure");
        CHECK(memcmp(park.persist_fingerprint, fp, sizeof(fp)) == 0, "park instance carries the identity fingerprint");
    } catch (const std::exception & e) {
        printf("FAIL: unexpected exception: %s\n", e.what());
        g_failures++;
    }

    // --- 4: destruction semantics - the park library is persist-only ---
    CHECK(fs::is_directory(fs::u8path(park_dir)), "park directory survives the park instance destruction");
    CHECK(fs::is_regular_file(fs::u8path(park_dir + "/.persist.lock")), "park lock file survives destruction (released, never removed)");
    CHECK(fs::is_directory(fs::u8path(general_dir + "/entry-1")), "general entry-1 untouched by the park instance");
    CHECK(fs::is_directory(fs::u8path(general_dir + "/entry-2")), "general entry-2 untouched by the park instance");
    CHECK(run_dirs(root).empty(), "no per-run directories remain after both instances are destroyed");

    // --- 3: fingerprint identity still governs adoption ---
    try {
        server_prompt_cache stranger(0, 4096, base, 64, fp_other, 64, 0);
        CHECK(stranger.disk_states.empty(), "different fingerprint -> no adoption");
        CHECK(fs::is_directory(fs::u8path(general_dir + "/entry-1")) &&
              fs::is_directory(fs::u8path(general_dir + "/entry-2")),
            "mismatched-fingerprint entries stay on disk (orphans, no budget pressure)");
    } catch (const std::exception & e) {
        printf("FAIL: unexpected exception: %s\n", e.what());
        g_failures++;
    }

    // --- 6: positive adoption - fixtures below the PARK'S OWN subdir are
    //         adopted by the park instance (mirror of the separation check) ---
    const std::string own_dir = root + "/test-park-own";
    CHECK(write_fixture(own_dir, 7, fp, 2048, 1756685000ull, {21, 22, 23, 24}),
        "fixture entry-7 written below the park's own subdir");
    CHECK(write_fixture(own_dir, 8, fp, 2048, 1756685010ull, {31, 32, 33, 34}),
        "fixture entry-8 written below the park's own subdir");
    try {
        server_prompt_cache own(0, 4096, base, 64, fp, 64, 0, "test-park-own", true);
        CHECK(own.persist_path == own_dir, "own-subdir park points at its library");
        CHECK(own.disk_states.size() == 2, "park instance ADOPTS the entries of its own subdir");
        CHECK(own.disk_size() == 4096, "adopted bytes accounted");
    } catch (const std::exception & e) {
        printf("FAIL: unexpected exception: %s\n", e.what());
        g_failures++;
    }

    // --- 7: a park instance whose boot fails (library lock held by a second
    //         server on the SAME subdir) must come out FULLY inert: no
    //         persist_path AND no disk_owned_path - with the owned path still
    //         set, save_disk would silently fall back to the per-run layout
    //         and litter the park directory with unaccounted state files ---
    const std::string lock_dir = root + "/test-park-lock";
    try {
        server_prompt_cache park1(0, 4096, base, 64, fp, 64, 0, "test-park-lock", true);
        CHECK(park1.persist_path == lock_dir, "first park on the subdir holds the library");
        {
            server_prompt_cache park2(0, 4096, base, 64, fp, 64, 0, "test-park-lock", true);
            CHECK(park2.persist_path.empty(), "boot-conflicted park: persist disabled (persist_path cleared)");
            CHECK(park2.disk_owned_path.empty(), "boot-conflicted park: owned path cleared too - the instance is fully inert");
            CHECK(park2.disk_states.empty(), "boot-conflicted park: nothing adopted");
            CHECK(!park2.ram_enabled, "boot-conflicted park: RAM stays disabled");
        }
        CHECK(park1.persist_path == lock_dir, "the conflicting park did not disturb the lock holder");
        CHECK(fs::is_directory(fs::u8path(lock_dir)), "the lock holder's library directory survives the conflicted park");

        // --- 8: the destructor releases the per-dir lock - a new park on the
        //         SAME subdir must acquire it (park1 dies at this scope end,
        //         park3 below must not see the library as locked) ---
    } catch (const std::exception & e) {
        printf("FAIL: unexpected exception: %s\n", e.what());
        g_failures++;
    }
    try {
        server_prompt_cache park3(0, 4096, base, 64, fp, 64, 0, "test-park-lock", true);
        CHECK(park3.persist_path == lock_dir, "a new park on the same subdir acquires the released lock");
        CHECK(park3.disk_states.empty(), "the re-locked library is otherwise intact (empty as left)");
    } catch (const std::exception & e) {
        printf("FAIL: unexpected exception: %s\n", e.what());
        g_failures++;
    }

    // --- 5: default-off at the class level: no park flag -> no park dir ---
    CHECK(!fs::exists(fs::u8path(root + "/main-park")),
        "instances without the park flag never create a main-park directory");

    // --- W6-8 (A3-5): the park RAM mirror ---
    //  a. pure admission rule: all-or-nothing on the entry payload vs the cap
    CHECK(!server_prompt_cache_park_mirror_admits(0, 4096), "admits: limit 0 (default off) admits nothing");
    CHECK(!server_prompt_cache_park_mirror_admits(1024, 0), "admits: empty payload is never mirrored");
    CHECK(server_prompt_cache_park_mirror_admits(1024, 1024), "admits: payload exactly at the cap");
    CHECK(server_prompt_cache_park_mirror_admits(1024, 1), "admits: payload below the cap");
    CHECK(!server_prompt_cache_park_mirror_admits(1024, 1025), "admits: payload one byte over the cap is not mirrored");

    //  b. pure serve rule: entry id + exact byte counts, draft only when wanted
    {
        park_ram_mirror m;
        CHECK(!server_prompt_cache_park_mirror_matches(m, 7, 100, false, 0, false),
            "matches: an invalid mirror never answers");
        m.valid = true;
        m.entry_id = 7;
        m.main.resize(100);
        CHECK(server_prompt_cache_park_mirror_matches(m, 7, 100, false, 0, false),
            "matches: right entry, right size, no draft wanted");
        CHECK(!server_prompt_cache_park_mirror_matches(m, 8, 100, false, 0, false),
            "matches: another entry never answers");
        CHECK(!server_prompt_cache_park_mirror_matches(m, 7, 101, false, 0, false),
            "matches: target byte count must equal the entry record");
        CHECK(!server_prompt_cache_park_mirror_matches(m, 7, 0, false, 0, false),
            "matches: zero-size target never answers");
        // draft wanted: the draft half must be present with the exact size
        CHECK(!server_prompt_cache_park_mirror_matches(m, 7, 100, true, 50, true),
            "matches: draft wanted but the mirror has no draft bytes");
        m.drft.resize(50);
        CHECK(server_prompt_cache_park_mirror_matches(m, 7, 100, true, 50, true),
            "matches: draft wanted and mirrored at the exact size");
        CHECK(!server_prompt_cache_park_mirror_matches(m, 7, 100, true, 51, true),
            "matches: draft byte count must equal the entry record");
        // a tag mismatch (draft NOT wanted) serves from the target half alone
        CHECK(server_prompt_cache_park_mirror_matches(m, 7, 100, false, 50, true),
            "matches: tag mismatch serves target-only, draft never consulted");
    }

    //  c. default-off at the instance level: the ctor without the mirror
    //     budget leaves the gate closed (load_disk reads park_ram_mirror_limit
    //     before the mirror can serve)
    const std::string park_dir_w68 = root + "/test-park-mirror";
    try {
        write_fixture(park_dir_w68, 3, fp, 4096, 1756684800ull, {1, 2, 3});
        {
            server_prompt_cache park_plain(0, 4096, base, 64, fp, 64, 0, "test-park-mirror", true);
            CHECK(park_plain.park_ram_mirror_limit == 0, "default ctor: mirror limit is 0 (off)");
            CHECK(!park_plain.park_mirror.valid, "default ctor: no mirror is live");
            CHECK(park_plain.disk_states.size() == 1, "mirror test park adopts its fixture entry");
        } // release the subdir lock before re-opening the library below

        //  d. the mirror budget reaches the park (and only the park)
        {
            server_prompt_cache park_mir(0, 4096, base, 64, fp, 64, 0, "test-park-mirror", true, 8ull << 20);
            CHECK(park_mir.park_ram_mirror_limit == (8ull << 20), "park ctor carries the mirror budget");
            CHECK(park_mir.disk_states.size() == 1, "budgeted park adopts the same fixture entry");

            //  e. invalidation: superseding the mirrored entry drops the mirror
            //     (persist_park_supersede -> erase_disk_state -> park_mirror.reset)
            park_mir.park_mirror.valid = true;
            park_mir.park_mirror.entry_id = 3;
            park_mir.park_mirror.main.resize(4096);
            CHECK(park_mir.persist_park_supersede(999), "supersede keeps id 999, erases the rest");
            CHECK(park_mir.disk_states.empty(), "the fixture entry was erased by the supersede");
            CHECK(!park_mir.park_mirror.valid, "the mirror died with its entry");
        }
        server_prompt_cache general_mir(0, 4096, base, 64, fp, 64, 0, "persist", false, 8ull << 20);
        CHECK(general_mir.park_ram_mirror_limit == 0, "a non-park instance ignores the mirror budget (T32-b semantics)");
    } catch (const std::exception & e) {
        printf("FAIL: unexpected exception (mirror section): %s\n", e.what());
        g_failures++;
    }

    std::error_code rm_ec;
    fs::remove_all(fs::u8path(base), rm_ec);

    if (g_failures == 0) {
        printf("test-park-library: ALL PASS\n");
        return 0;
    }
    printf("test-park-library: %d FAILURES\n", g_failures);
    return 1;
}
