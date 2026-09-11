// Unit tests for the per-buffer unsynchronized-access tracker of the Vulkan
// backend (W6-4 / A6-S2), ggml-vulkan-sync.hpp.
//
// Two layers:
//  1. Pattern tests for the hazard rules (RAW/WAR/WAW per buffer, cross-buffer
//     independence, disjoint/adjacent ranges, zero-size edges).
//  2. A randomized equivalence run against a reference implementation of the
//     PREVIOUS global flat-list semantics (unsynced_nodes_written/read scanned
//     linearly for dst and srcs of every node group). The per-buffer tracker
//     must take the identical sync decision on every node group: this is the
//     no-semantic-drift gate for the restructure.
//
// The tracker is header-only and free of Vulkan types, so this test needs no
// device: buffer identity is an opaque pointer. It does NOT exercise the
// Vulkan barrier emission itself (that is device-side; the numeric bit-exact
// proof runs on the GPU cells).

#include "ggml-vulkan-sync.hpp"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

static int g_failures = 0;
static int g_checks = 0;

#define CHECK(cond, msg) do { \
    g_checks++; \
    if (!(cond)) { printf("FAIL: %s\n", msg); g_failures++; } \
} while (0)

// Distinct fake device buffers.
static const void * BUF_A = (const void *) 0x1000;
static const void * BUF_B = (const void *) 0x2000;
static const void * BUF_C = (const void *) 0x3000;

// ---------------------------------------------------------------------------
// Reference implementation: the previous flat-list semantics.
// ---------------------------------------------------------------------------

struct ref_access {
    const void * buf;
    uint64_t base;
    uint64_t size;
    bool write;
};

struct ref_tracker {
    // The previous code kept one "written" list and one "read" list; dst was
    // checked against both, srcs only against "written".
    std::vector<ref_access> written;
    std::vector<ref_access> read;

    bool conflicts(const void * buf, uint64_t base, uint64_t size, bool is_write) const {
        if (overlaps_any(written, buf, base, size)) return true;
        if (is_write && overlaps_any(read, buf, base, size)) return true;
        return false;
    }

    void record(const void * buf, uint64_t base, uint64_t size, bool is_write) {
        if (is_write) written.push_back({buf, base, size, true});
        else          read.push_back({buf, base, size, false});
    }

    void clear_all() { written.clear(); read.clear(); }

  private:
    static bool overlaps_any(const std::vector<ref_access> & list, const void * buf,
                             uint64_t base, uint64_t size) {
        for (const auto & e : list) {
            if (e.buf != buf) continue;
            if ((e.base <= base && base < e.base + e.size) ||
                (base <= e.base && e.base < base + size)) {
                return true;
            }
        }
        return false;
    }
};

// ---------------------------------------------------------------------------
// Pattern tests.
// ---------------------------------------------------------------------------

static void test_hazard_patterns() {
    {
        // RAW: write then read the same region in the same buffer.
        ggml_vk::unsynced_tracker t;
        t.record(BUF_A, 100, 50, true);
        std::vector<ggml_vk::sync_conflict> conflicts;
        CHECK(t.find_conflicts(BUF_A, 120, 10, false, &conflicts), "RAW same buffer fires");
        CHECK(conflicts.size() == 1, "RAW produces one conflict");
        CHECK(conflicts[0].buf == BUF_A && conflicts[0].base == 100 && conflicts[0].size == 50,
              "RAW conflict covers the enclosing range of the pair");
    }
    {
        // Cross-buffer: identical offsets, different buffers -> no sync.
        ggml_vk::unsynced_tracker t;
        t.record(BUF_A, 100, 50, true);
        CHECK(!t.find_conflicts(BUF_B, 100, 50, false, nullptr), "cross-buffer RAW does not fire");
        CHECK(!t.find_conflicts(BUF_B, 100, 50, true, nullptr), "cross-buffer WAW does not fire");
    }
    {
        // WAR: read then write the same region.
        ggml_vk::unsynced_tracker t;
        t.record(BUF_A, 0, 64, false);
        CHECK(t.find_conflicts(BUF_A, 0, 64, true, nullptr), "WAR same buffer fires");
    }
    {
        // WAW: write then write the same region.
        ggml_vk::unsynced_tracker t;
        t.record(BUF_A, 0, 64, true);
        CHECK(t.find_conflicts(BUF_A, 32, 64, true, nullptr), "WAW overlapping fires");
    }
    {
        // Read after read never conflicts.
        ggml_vk::unsynced_tracker t;
        t.record(BUF_A, 0, 64, false);
        CHECK(!t.find_conflicts(BUF_A, 0, 64, false, nullptr), "RAR does not fire");
    }
    {
        // Disjoint and exactly-adjacent ranges in the same buffer do not
        // conflict (strict-inequality semantics of the original check).
        ggml_vk::unsynced_tracker t;
        t.record(BUF_A, 0, 64, true);
        CHECK(!t.find_conflicts(BUF_A, 64, 64, false, nullptr), "adjacent ranges do not conflict");
        CHECK(!t.find_conflicts(BUF_A, 1000, 64, false, nullptr), "disjoint ranges do not conflict");
    }
    {
        // Zero-size query with base strictly inside a tracked region still
        // conflicts (first clause of the original predicate); a zero-size
        // tracked region conflicts when a later access strictly contains
        // its base (second clause).
        ggml_vk::unsynced_tracker t;
        t.record(BUF_A, 0, 64, true);
        CHECK(t.find_conflicts(BUF_A, 10, 0, false, nullptr), "zero-size query inside region fires");
        CHECK(!t.find_conflicts(BUF_A, 64, 0, false, nullptr), "zero-size query at region end does not fire");
        t.clear_all();
        t.record(BUF_A, 100, 0, true);
        CHECK(!t.empty(), "zero-size regions are tracked");
        CHECK(t.find_conflicts(BUF_A, 50, 100, false, nullptr), "zero-size tracked region conflicts when contained");
        CHECK(!t.find_conflicts(BUF_A, 100, 0, false, nullptr), "zero-size vs zero-size at same base does not fire");
    }
    {
        // Only the queried buffer's regions are scanned.
        ggml_vk::unsynced_tracker t;
        t.record(BUF_A, 0, 64, true);
        t.record(BUF_B, 0, 64, false);
        std::vector<ggml_vk::sync_conflict> conflicts;
        CHECK(t.find_conflicts(BUF_A, 0, 64, true, &conflicts), "query hits its own buffer only");
        CHECK(conflicts.size() == 1 && conflicts[0].buf == BUF_A,
              "conflicts never reference other buffers");
    }
    {
        // clear_all drops everything (device-wide barrier).
        ggml_vk::unsynced_tracker t;
        t.record(BUF_A, 0, 64, true);
        t.record(BUF_B, 0, 64, false);
        t.clear_all();
        CHECK(t.empty(), "clear_all empties the tracker");
        CHECK(!t.find_conflicts(BUF_A, 0, 64, true, nullptr), "no conflict after clear_all");
    }
}

// ---------------------------------------------------------------------------
// Randomized equivalence against the reference (previous) semantics.
// ---------------------------------------------------------------------------

struct rng {
    uint64_t s;
    explicit rng(uint64_t seed) : s(seed) {}
    uint64_t next() {
        // xorshift64*
        s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
        return s * 2685821657736338717ULL;
    }
    uint64_t below(uint64_t n) { return n ? next() % n : 0; }
};

// Simulates ggml_vk_build_graph's tracker block for one "node group":
// returns whether a sync fires, applies it (clear-all on sync, like the
// device-wide barrier) and records the group accesses. Runs both the new
// tracker and the reference in lockstep and compares the decisions.
static bool run_group(ggml_vk::unsynced_tracker & t, ref_tracker & r,
                      const void * const * bufs, size_t n_bufs,
                      bool dst_write, size_t dst_buf_idx, uint64_t dst_base, uint64_t dst_size,
                      const uint64_t (&src)[4][3], bool src_used[4]) {
    // Mirror the production loop: dst (if it writes memory) against both
    // hazard classes, then each src against prior writes.
    bool need_new = false;
    if (dst_write && t.find_conflicts(bufs[dst_buf_idx], dst_base, dst_size, true, nullptr)) {
        need_new = true;
    }
    if (!need_new) {
        for (int j = 0; j < 4; ++j) {
            if (!src_used[j]) continue;
            if (t.find_conflicts(bufs[src[j][0] % n_bufs], src[j][1], src[j][2], false, nullptr)) {
                need_new = true;
                break;
            }
        }
    }

    bool need_ref = false;
    if (dst_write && r.conflicts(bufs[dst_buf_idx], dst_base, dst_size, true)) {
        need_ref = true;
    }
    if (!need_ref) {
        for (int j = 0; j < 4; ++j) {
            if (!src_used[j]) continue;
            if (r.conflicts(bufs[src[j][0] % n_bufs], src[j][1], src[j][2], false)) {
                need_ref = true;
                break;
            }
        }
    }

    CHECK(need_new == need_ref, "sync decision identical to reference semantics");

    if (need_new) {
        t.clear_all();
        r.clear_all();
    }
    if (dst_write) {
        t.record(bufs[dst_buf_idx], dst_base, dst_size, true);
        r.record(bufs[dst_buf_idx], dst_base, dst_size, true);
    }
    for (int j = 0; j < 4; ++j) {
        if (!src_used[j]) continue;
        t.record(bufs[src[j][0] % n_bufs], src[j][1], src[j][2], false);
        r.record(bufs[src[j][0] % n_bufs], src[j][1], src[j][2], false);
    }
    return need_new;
}

static void test_randomized_equivalence() {
    const void * bufs[4] = { BUF_A, BUF_B, BUF_C, BUF_A };  // A appears twice on purpose

    for (uint64_t seed = 1; seed <= 20; ++seed) {
        rng g(seed * 0x9E3779B97F4A7C15ULL);
        ggml_vk::unsynced_tracker t;
        ref_tracker r;

        const int n_groups = 2000;
        int syncs = 0;
        for (int grp = 0; grp < n_groups; ++grp) {
            // Access layout that mimics a decode graph: mostly small regions
            // in a narrow offset window so overlaps are frequent, with some
            // far-apart regions and occasional zero-size tensors.
            const uint64_t window = (g.below(4) == 0) ? 65536 : 512;
            const size_t dst_buf_idx = g.below(4);
            const uint64_t dst_base = g.below(window);
            const uint64_t dst_size = g.below(4) == 0 ? 0 : (1 + g.below(256));
            const bool dst_write = g.below(8) != 0;  // most groups write memory

            uint64_t src[4][3];
            bool src_used[4];
            const int n_src = 1 + (int) g.below(4);
            for (int j = 0; j < 4; ++j) {
                src_used[j] = j < n_src;
                src[j][0] = g.below(1024);
                src[j][1] = g.below(window);
                src[j][2] = g.below(8) == 0 ? 0 : (1 + g.below(256));
            }

            if (run_group(t, r, bufs, 4, dst_write, dst_buf_idx, dst_base, dst_size, src, src_used)) {
                syncs++;
            }
        }
        CHECK(syncs > n_groups / 10, "randomized sequence actually exercises syncs");
        (void) syncs;
    }
}

// ---------------------------------------------------------------------------
// Scoped-clear hazard oracle (A6-S1 gate): simulate the scoped path exactly
// as production runs it (collect the group's conflicts -> merge into one
// enclosing range per buffer -> clear_covered -> record), then verify by
// brute force over the whole access trace that every inter-group RAW/WAR/WAW
// pair had a barrier covering the overlap between the two accesses. A missed
// dependency would be a lost barrier: a correctness bug, not a slowdown.
// ---------------------------------------------------------------------------

struct oracle_access {
    const void * buf;
    uint64_t base;
    uint64_t size;
    bool write;
    int group;  // node group the access belongs to
};

static void test_scoped_hazard_oracle(bool sabotage_clear_all = false) {
    int total_barriers = 0;
    int total_misses = 0;

    for (uint64_t seed = 1; seed <= 25; ++seed) {
        rng g(seed * 0xD1B54A32D192ED03ULL);
        const void * bufs[3] = { BUF_A, BUF_B, BUF_C };

        ggml_vk::unsynced_tracker t;
        std::vector<oracle_access> trace;
        // Barrier emitted at group boundary: (position = number of accesses
        // recorded before the group that emitted it, covered range).
        std::vector<std::pair<int, ggml_vk::sync_conflict>> barriers;

        const int n_groups = 1500;
        for (int grp = 0; grp < n_groups; ++grp) {
            // One node group: a destination (mostly writing) plus sources.
            std::vector<std::tuple<const void *, uint64_t, uint64_t, bool>> group;
            const uint64_t window = (g.below(4) == 0) ? 4096 : 384;
            group.push_back({ bufs[g.below(3)], g.below(window), 1 + g.below(192), g.below(8) != 0 });
            const int n_src = 1 + (int) g.below(3);
            for (int j = 0; j < n_src; ++j) {
                group.push_back({ bufs[g.below(3)], g.below(window), 1 + g.below(192), false });
            }

            bool need_sync = false;
            std::vector<ggml_vk::sync_conflict> conflicts;
            for (const auto & a : group) {
                if (t.find_conflicts(std::get<0>(a), std::get<1>(a), std::get<2>(a), std::get<3>(a), &conflicts)) {
                    need_sync = true;
                }
            }

            if (need_sync) {
                // Production merge: one enclosing range per buffer.
                std::vector<ggml_vk::sync_conflict> merged;
                for (const auto & c : conflicts) {
                    bool done = false;
                    for (auto & r : merged) {
                        if (r.buf == c.buf) {
                            const uint64_t end = std::max(r.base + r.size, c.base + c.size);
                            r.base = std::min(r.base, c.base);
                            r.size = end - r.base;
                            done = true;
                            break;
                        }
                    }
                    if (!done) {
                        merged.push_back(c);
                    }
                }
                for (const auto & m : merged) {
                    barriers.push_back({ (int) trace.size(), m });
                }
                total_barriers += (int) merged.size();
                if (sabotage_clear_all) {
                    // Negative control: simulate the over-clearing bug
                    // (drop every tracked region on every barrier, not just
                    // the covered ones) and let the oracle below prove it
                    // counts as lost dependencies.
                    t.clear_all();
                } else {
                    t.clear_covered(merged);
                }
            }

            for (const auto & a : group) {
                trace.push_back({ std::get<0>(a), std::get<1>(a), std::get<2>(a), std::get<3>(a), grp });
                t.record(std::get<0>(a), std::get<1>(a), std::get<2>(a), std::get<3>(a));
            }
        }

        // Brute-force hazard verification. For every pair of accesses from
        // DIFFERENT groups (intra-group ordering is a fusion property the
        // tracker never barriered, before or after this change) that overlap
        // in the same buffer with at least one write, some barrier emitted
        // strictly between them must cover the overlapping bytes.
        for (size_t i = 0; i < trace.size(); ++i) {
            for (size_t j = i + 1; j < trace.size(); ++j) {
                const auto & a = trace[i];
                const auto & b = trace[j];
                if (a.group == b.group) continue;
                if (a.buf != b.buf) continue;
                if (!a.write && !b.write) continue;
                if (!ggml_vk::sync_ranges_overlap(a.base, a.size, b.base, b.size)) continue;

                const uint64_t lo = std::max(a.base, b.base);
                const uint64_t hi = std::min(a.base + a.size, b.base + b.size);

                bool covered = false;
                for (const auto & bar : barriers) {
                    if (bar.first <= (int) i) continue;          // not after access i
                    if (bar.first > (int) j) break;              // after access j (vector is position-ordered)
                    if (bar.second.buf == a.buf &&
                        bar.second.base <= lo &&
                        bar.second.base + bar.second.size >= hi) {
                        covered = true;
                        break;
                    }
                }
                if (!covered) {
                    total_misses++;
                }
            }
        }
    }

    CHECK(total_misses == 0, "scoped clearing never drops an unsynchronized dependency");
    CHECK(total_barriers > 1000, "oracle run actually emitted barriers");
    printf("oracle: %d barrier ranges, %d missed hazards\n", total_barriers, total_misses);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char ** argv) {
    // Negative-control mode ("negctl"): runs the scoped-clear hazard oracle
    // with a deliberately over-clearing clear step (simulated in THIS test's
    // stub, never in the production header) and must fail with missed
    // hazards > 0. Exit code 1 in this mode is the expected, successful
    // outcome of the control; the normal ctest invocation (no arguments)
    // is unaffected.
    const bool negctl = argc > 1 && strcmp(argv[1], "negctl") == 0;

    test_hazard_patterns();
    test_randomized_equivalence();
    test_scoped_hazard_oracle(negctl);

    if (g_failures == 0) {
        printf("test-vk-sync-tracker: OK (%d checks)%s\n", g_checks, negctl ? " [NEGCTL: UNEXPECTED]" : "");
        return 0;
    }
    printf("test-vk-sync-tracker: %d FAILURES out of %d checks%s\n", g_failures, g_checks,
           negctl ? " [NEGCTL: expected - the oracle caught the sabotage]" : "");
    return 1;
}
