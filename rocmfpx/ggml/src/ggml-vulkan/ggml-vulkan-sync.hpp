#pragma once

// W6-4 / A6-S2: per-buffer tracker of unsynchronized device-buffer accesses.
//
// The Vulkan backend does not emit a barrier per dispatch. Instead it tracks,
// per device buffer, which byte regions have been accessed since the last
// barrier that covered them, split into reads and writes. Before a node group
// is recorded, its destination and sources are checked against this state
// (hazard test below); on a hazard a barrier is emitted and the covered
// regions are dropped from the tracker.
//
// This header is deliberately free of Vulkan types so the hazard logic can be
// unit-tested without a device (tests/test-vk-sync-tracker.cpp): a buffer is
// identified by an opaque pointer, a region is a byte range within it. The
// production adapter in ggml-vulkan.cpp derives (buffer, base, size) from
// ggml_tensor exactly like the previous flat unsynced_nodes_written/read
// lists did:
//   base = vk_tensor_offset(t) + t->view_offs
//   size = ggml_nbytes(t)
//   buffer identity = the ggml_backend_vk_buffer_context::dev_buffer.
//
// Hazard rules (identical to the previous implementation):
//  - a WRITE conflicts with a prior unsynced READ (WAR) or WRITE (WAW)
//    overlapping it in the same buffer;
//  - a READ conflicts only with a prior unsynced WRITE (RAW);
//  - READ vs READ never conflicts;
//  - two accesses conflict only within the same device buffer;
//  - ranges overlap per the strict-inequality test of the original
//    overlaps_unsynced() (touching-but-disjoint ranges do NOT conflict).

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace ggml_vk {

// A byte range [base, base + size) inside one device buffer.
struct sync_region {
    uint64_t base;
    uint64_t size;

    uint64_t end() const { return base + size; }
};

// A resolved hazard: the barrier must order the tracked access and the new
// access against each other, so the covered range is the enclosing range of
// the two overlapping accesses.
struct sync_conflict {
    const void * buf;  // opaque identity of the device buffer
    uint64_t base;     // enclosing range of the tracked region and the query
    uint64_t size;
};

// Exact overlap semantics of the previous per-tensor check:
//   (o_base <= n_base && n_base < o_base + o_size) ||
//   (n_base <= o_base && o_base < n_base + n_size)
// Note the asymmetry this implies for zero-size regions (empty tensors):
// a zero-size QUERY only conflicts when its base falls strictly inside a
// tracked range (first clause), while a zero-size TRACKED region conflicts
// whenever a later access strictly contains its base (second clause). Both
// behaviors are preserved.
inline bool sync_ranges_overlap(uint64_t a_base, uint64_t a_size, uint64_t b_base, uint64_t b_size) {
    return (b_base <= a_base && a_base < b_base + b_size) ||
           (a_base <= b_base && b_base < a_base + a_size);
}

class unsynced_tracker {
  public:
    // Does an access of `size` bytes at `base` in buffer `buf` conflict with
    // unsynchronized accesses? When `out` is non-null, append one
    // sync_conflict per conflicting tracked region (the enclosing range of
    // the tracked region and the queried range). Returns whether any
    // conflict was found.
    bool find_conflicts(const void * buf, uint64_t base, uint64_t size, bool is_write,
                        std::vector<sync_conflict> * out) const {
        auto it = per_buffer.find(buf);
        if (it == per_buffer.end()) {
            return false;
        }
        bool found = false;
        // A write conflicts with prior reads and writes; a read only with
        // prior writes.
        if (check_list(it->second.written, base, size, buf, out)) {
            found = true;
        }
        if (is_write && check_list(it->second.read, base, size, buf, out)) {
            found = true;
        }
        return found;
    }

    // Register an access as unsynchronized. Zero-size regions are kept: the
    // strict-inequality overlap test still reports a conflict when a later
    // access strictly contains their base.
    void record(const void * buf, uint64_t base, uint64_t size, bool is_write) {
        auto & regions = is_write ? per_buffer[buf].written : per_buffer[buf].read;
        regions.push_back({base, size});
    }

    // Drop all tracked state (a device-wide barrier makes every prior access
    // synchronized).
    void clear_all() { per_buffer.clear(); }

    // Drop every tracked region fully covered by one of `covered`. Regions
    // only partially covered are kept: keeping a region that was in fact
    // synchronized is conservative (it can only cause a later barrier),
    // dropping one that was not is a correctness bug.
    void clear_covered(const std::vector<sync_conflict> & covered) {
        for (const auto & c : covered) {
            auto it = per_buffer.find(c.buf);
            if (it == per_buffer.end()) {
                continue;
            }
            drop_covered(it->second.written, c);
            drop_covered(it->second.read, c);
            if (it->second.written.empty() && it->second.read.empty()) {
                per_buffer.erase(it);
            }
        }
    }

    bool empty() const {
        for (const auto & kv : per_buffer) {
            if (!kv.second.written.empty() || !kv.second.read.empty()) {
                return false;
            }
        }
        return true;
    }

    // Total number of tracked accesses (diagnostics / tests).
    size_t region_count() const {
        size_t n = 0;
        for (const auto & kv : per_buffer) {
            n += kv.second.written.size() + kv.second.read.size();
        }
        return n;
    }

    // Buffers that still have at least one tracked region (used to prune
    // side tables that resolve buffer identities to vk_buffer handles).
    std::vector<const void *> tracked_buffers() const {
        std::vector<const void *> bufs;
        for (const auto & kv : per_buffer) {
            if (!kv.second.written.empty() || !kv.second.read.empty()) {
                bufs.push_back(kv.first);
            }
        }
        return bufs;
    }

  private:
    struct buffer_state {
        std::vector<sync_region> written;
        std::vector<sync_region> read;
    };

    static bool check_list(const std::vector<sync_region> & regions, uint64_t base, uint64_t size,
                           const void * buf, std::vector<sync_conflict> * out) {
        bool found = false;
        for (const auto & r : regions) {
            if (!sync_ranges_overlap(r.base, r.size, base, size)) {
                continue;
            }
            if (out != nullptr) {
                const uint64_t lo  = r.base < base ? r.base : base;
                const uint64_t hi  = r.end() > base + size ? r.end() : base + size;
                out->push_back({buf, lo, hi - lo});
            }
            found = true;
        }
        return found;
    }

    static void drop_covered(std::vector<sync_region> & regions, const sync_conflict & covered) {
        size_t keep = 0;
        for (size_t i = 0; i < regions.size(); ++i) {
            const bool is_covered = regions[i].base >= covered.base &&
                                    regions[i].end() <= covered.base + covered.size;
            if (!is_covered) {
                regions[keep++] = regions[i];
            }
        }
        regions.resize(keep);
    }

    std::unordered_map<const void *, buffer_state> per_buffer;
};

}  // namespace ggml_vk
