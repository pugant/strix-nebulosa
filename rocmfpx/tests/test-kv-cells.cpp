// Unit tests for llama_kv_cells (W6-5 / A3-4): the per-cell sequence bitset
// bookkeeping. The bit-set iteration rework (_Find_first/_Find_next on
// libstdc++, plain scan elsewhere) must keep every observable behavior
// identical: pos/ext lifecycle, seq membership, seq_pos min/max counters with
// DUPLICATE positions (the map counts them), rm/seq_rm/seq_keep transitions,
// cp/set round-trips sharing cells between sequences, and the pos_add/pos_div
// shift path. Zero llama runtime - the struct is a plain header.

#include "../src/llama-kv-cells.h"

#include <cstdio>
#include <vector>

static int g_failures = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("FAIL: %s\n", msg); g_failures++; } \
} while (0)

int main() {
    llama_kv_cells cells;
    cells.resize(16);

    CHECK(cells.size() == 16, "resize sets the cell count");
    CHECK(cells.get_used() == 0, "fresh cache holds no used cells");
    for (uint32_t i = 0; i < cells.size(); i++) {
        CHECK(cells.is_empty(i), "fresh cells are empty");
    }

    // --- single-sequence lifecycle: pos_set -> seq_add -> counters ---
    cells.pos_set(3, 100);
    cells.seq_add(3, 7);
    CHECK(!cells.is_empty(3), "pos_set marks the cell used");
    CHECK(cells.seq_has(3, 7), "seq_add registers the sequence");
    CHECK(cells.seq_count(3) == 1, "one sequence in the cell");
    CHECK(cells.seq_get(3) == 7, "seq_get returns the single sequence");
    CHECK(cells.seq_pos_min(7) == 100, "seq_pos_min after first add");
    CHECK(cells.seq_pos_max(7) == 100, "seq_pos_max after first add");
    CHECK(cells.used_min() == 3 && cells.used_max_p1() == 4, "used range tracks the single cell");

    // --- shared cell: two sequences in one cell ---
    cells.seq_add(3, 9);
    CHECK(cells.seq_count(3) == 2, "shared cell holds two sequences");
    CHECK(cells.seq_pos_min(9) == 100 && cells.seq_pos_max(9) == 100, "second sequence sees the same pos");

    // --- duplicate positions: the seq_pos map must COUNT them (two cells,
    //     same pos, same seq) - removing one leaves the other visible ---
    cells.pos_set(5, 100);
    cells.seq_add(5, 7);
    CHECK(cells.seq_pos_min(7) == 100 && cells.seq_pos_max(7) == 100, "duplicate pos still bounded");
    bool res = cells.seq_rm(5, 7);
    CHECK(res, "removing the last seq of a cell frees it");
    CHECK(cells.is_empty(5), "cell 5 is empty again");
    CHECK(cells.seq_pos_min(7) == 100 && cells.seq_pos_max(7) == 100,
          "pos 100 still present for seq 7 (the duplicate in cell 3)");

    // --- seq_rm with one of two sequences: cell stays, other survives ---
    {
        bool freed = cells.seq_rm(3, 7);
        CHECK(!freed, "cell with a remaining sequence is not freed");
        CHECK(!cells.seq_has(3, 7), "seq 7 is gone from the cell");
        CHECK(cells.seq_has(3, 9), "seq 9 survives");
        CHECK(cells.seq_pos_min(7) == -1, "seq 7 has no positions left");
    }

    // --- seq_keep from a two-sequence cell keeps only the named one ---
    {
        cells.seq_add(3, 7); // back to {7, 9}
        bool freed = cells.seq_keep(3, 9);
        CHECK(!freed, "seq_keep on a member is not a free");
        CHECK(cells.seq_count(3) == 1 && cells.seq_has(3, 9), "only the kept sequence remains");
        freed = cells.seq_keep(2, 9); // cell 2 is empty
        CHECK(!freed, "seq_keep on an empty cell is a no-op");
        // cell 1: put seq 4 in, keep seq 5 -> the cell is freed
        cells.pos_set(1, 50);
        cells.seq_add(1, 4);
        freed = cells.seq_keep(1, 5);
        CHECK(freed, "seq_keep of an absent sequence frees the cell");
        CHECK(cells.is_empty(1), "cell freed by seq_keep");
    }

    // --- ext + cp/set round-trip with a shared cell ---
    {
        llama_kv_cells src;
        src.resize(2);
        src.pos_set(0, 10);
        src.pos_set(1, 11);
        src.seq_add(0, 0);
        src.seq_add(0, 1); // cell 0 shared by seqs 0 and 1
        src.seq_add(1, 1);
        llama_kv_cell_ext ext;
        ext.x = 7;
        ext.y = 8;
        ext.tok = 12345;
        src.ext_set(0, ext);

        llama_kv_cells dst;
        dst.resize(2);
        dst.pos_set(0, 999); // occupied: set() must overwrite fully
        dst.seq_add(0, 3);
        dst.set(0, src.cp(0, 2));

        CHECK(dst.pos_get(0) == 10, "set() overwrites pos");
        CHECK(dst.seq_has(0, 0) && dst.seq_has(0, 1) && !dst.seq_has(0, 3), "set() replaces the seq membership");
        CHECK(dst.ext_get(0).tok == 12345 && dst.ext_get(0).x == 7 && dst.ext_get(0).y == 8, "set() carries ext");
        // src cell 1 (pos 11, seq 1) also lands in dst via the range copy, so
        // seq 1 spans pos 10 (cell 0) and pos 11 (cell 1)
        CHECK(dst.seq_pos_min(0) == 10, "set() updates the seq_pos counters (seq 0)");
        CHECK(dst.seq_pos_min(1) == 10 && dst.seq_pos_max(1) == 11, "set() updates the seq_pos counters (seq 1, both cells)");

        // and back: set on empty destination via the idxs overload
        llama_kv_cells dst2;
        dst2.resize(2);
        const std::vector<uint32_t> idxs = { 1, 0 }; // reversed on purpose
        dst2.set(idxs, src.cp(idxs));
        CHECK(dst2.pos_get(1) == 11 && dst2.pos_get(0) == 10, "idxs set() maps cell by cell");
        CHECK(dst2.seq_has(1, 1) && dst2.seq_has(0, 0) && dst2.seq_has(0, 1), "idxs set() carries seq membership");
    }

    // --- shift path: pos_add / pos_div keep counters coherent ---
    {
        llama_kv_cells sh;
        sh.resize(4);
        sh.pos_set(0, 10);
        sh.pos_set(1, 20);
        sh.seq_add(0, 2);
        sh.seq_add(1, 2);
        CHECK(sh.get_has_shift() == false, "no shift yet");
        sh.pos_add(0, 5);
        CHECK(sh.get_has_shift() == true, "pos_add flags the shift");
        CHECK(sh.get_shift(0) == 5, "shift accumulates");
        CHECK(sh.pos_get(0) == 15, "pos_add moves the position");
        CHECK(sh.seq_pos_min(2) == 15 && sh.seq_pos_max(2) == 20, "counters follow the moved pos");
        sh.pos_div(0, 3);
        CHECK(sh.pos_get(0) == 5, "pos_div floors the division");
        CHECK(sh.seq_pos_min(2) == 5, "counters follow the divided pos");
        // pos_add past zero frees the cell
        sh.pos_add(0, -100);
        CHECK(sh.is_empty(0), "pos_add past zero frees the cell");
        CHECK(sh.seq_pos_min(2) == 20 && sh.seq_pos_max(2) == 20, "freed cell leaves the counters");
    }

    // --- reset clears everything ---
    {
        cells.reset();
        CHECK(cells.get_used() == 0, "reset empties the used set");
        CHECK(cells.seq_pos_min(9) == -1, "reset clears the seq_pos maps");
        for (uint32_t i = 0; i < cells.size(); i++) {
            CHECK(cells.is_empty(i), "reset empties every cell");
        }
    }

    if (g_failures == 0) {
        printf("test-kv-cells: ALL PASS\n");
        return 0;
    }
    printf("test-kv-cells: %d FAILURES\n", g_failures);
    return 1;
}
