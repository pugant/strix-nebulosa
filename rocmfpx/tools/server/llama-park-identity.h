// T32-b: pure identity helpers for the park library, the dedicated persistent
// store where the main-agent context survives on disk while subagent tasks
// use the regular prompt cache. Pure functions only: no server state, no
// llama runtime (unit-tested directly by tests/test-park-identity.cpp).
#pragma once

#include <cstdint>
#include <set>
#include <string>

// park configuration (mirrors --cache-disk-park-mib / --cache-disk-park-heuristic)
struct park_cfg {
    int32_t mib = 0;                          // library budget in MiB, 0 = feature fully off
    enum { NONE, LONGEST } heuristic = NONE;  // how the main-agent task is detected
};

// registry of the prompt token counts observed so far, updated on task arrival
struct heur_state {
    std::multiset<uint64_t> seen;
};

bool park_enabled(const park_cfg & cfg);                                  // cfg.mib > 0
bool park_from_header(const std::string & v);                             // v == "main"
bool park_header_value(const std::string & v);                            // raw X-Pi-Role value, "main" case-insensitive
bool park_from_heuristic(const heur_state & st, uint64_t prompt_tokens);  // > 32768 and the longest seen

// header wins over any heuristic; park_enabled() is NOT checked here - the
// caller gates on it before asking who the main task is
bool park_task_is_main(const park_cfg & cfg, const heur_state & st, bool header_main, uint64_t tokens);

// T32-b: the task-switch redirect predicate (spec t32-b step 4.2): the disk
// half of the boundary save goes to the park library iff the feature is on,
// the slot was serving a task, and that task is the main agent's. Pure on
// purpose - the combined cases are unit-tested (tests/test-park-identity.cpp).
// has_prev_task is a bool (not a server_task pointer) so the helper stays free
// of server types; prompt_tokens is the boundary the slot holds at the switch.
// The call site adds one conjunct that is not pure state: a live park library
// instance, which exists exactly when the disk cache is configured.
bool park_should_redirect(const park_cfg & cfg, const heur_state & st, bool has_prev_task, bool header_main, uint64_t prompt_tokens);
