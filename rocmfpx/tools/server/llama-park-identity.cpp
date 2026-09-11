#include "llama-park-identity.h"

#include <algorithm>
#include <cctype>

// T32-b spec: only a task strictly longer than 32768 tokens can be the main
// agent's (a subagent prompt never reaches a half-context-class size); the
// comparison is strict, so exactly 32768 does not qualify
static constexpr uint64_t park_min_tokens = 32768;

bool park_enabled(const park_cfg & cfg) {
    return cfg.mib > 0;
}

// the header value must arrive already lowercased/normalized by the caller,
// so the match here is exact
bool park_from_header(const std::string & v) {
    return v == "main";
}

// the raw X-Pi-Role value as it arrived on the wire: HTTP leaves the value
// casing to the sender and the client role is a fixed token, so the value is
// lowercased and then exact-matched - "Main"/"MAIN" park, anything else does
// not (including an empty/absent header)
bool park_header_value(const std::string & v) {
    std::string lowered(v.size(), '\0');
    std::transform(v.begin(), v.end(), lowered.begin(), [](unsigned char c) { return std::tolower(c); });
    return park_from_header(lowered);
}

bool park_from_heuristic(const heur_state & st, uint64_t prompt_tokens) {
    if (st.seen.empty()) {
        return false;
    }
    return prompt_tokens > park_min_tokens && prompt_tokens == *st.seen.rbegin();
}

bool park_task_is_main(const park_cfg & cfg, const heur_state & st, bool header_main, uint64_t tokens) {
    if (header_main) {
        return true; // an explicit "main" header is authoritative
    }
    if (cfg.heuristic != park_cfg::LONGEST) {
        return false; // a non-main header value is not a veto, but with no heuristic nothing else can say main
    }
    return park_from_heuristic(st, tokens);
}

bool park_should_redirect(const park_cfg & cfg, const heur_state & st, bool has_prev_task, bool header_main, uint64_t prompt_tokens) {
    return park_enabled(cfg) && has_prev_task && park_task_is_main(cfg, st, header_main, prompt_tokens);
}
