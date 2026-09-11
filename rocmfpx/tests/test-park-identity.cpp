// Unit tests for the T32-b park-library identity helpers. Zero GPU, zero
// llama runtime, zero server: park_enabled / park_from_header /
// park_header_value / park_from_heuristic / park_task_is_main /
// park_should_redirect are pure functions over the config, the client header
// value and the registry of the prompt lengths seen so far.

#include "../tools/server/llama-park-identity.h"

#include "arg.h"
#include "common.h"

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

static int g_failures = 0;

// source-scan gates (sections 13/15/16) that no-oped because the source tree
// was not reachable from the test binary - surfaced in the final verdict line
// so a bare "ALL PASS" can never mask an unexecuted gate
static int g_source_gate_skips = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("FAIL: %s\n", msg); g_failures++; } \
} while (0)

// brace-matched body of the definition of `sig` inside `src`: from the first
// '{' after the signature to its matching '}', skipping string/char literals
// and comments so braces inside them cannot unbalance the count. Returns the
// empty string when the signature or the matching brace is not found.
static std::string function_body(const std::string & src, const std::string & sig) {
    const size_t sig_pos = src.find(sig);
    if (sig_pos == std::string::npos) {
        return "";
    }
    const size_t open_pos = src.find('{', sig_pos);
    if (open_pos == std::string::npos) {
        return "";
    }
    int depth = 0;
    bool in_str = false, in_chr = false, in_line = false, in_block = false;
    for (size_t i = open_pos; i < src.size(); i++) {
        const char c = src[i];
        const char n = i + 1 < src.size() ? src[i + 1] : '\0';
        if (in_line)  { if (c == '\n')                     { in_line  = false; } continue; }
        if (in_block) { if (c == '*' && n == '/')          { in_block = false; i++; } continue; }
        if (in_str)   { if (c == '\\')                     { i++; } else if (c == '"')  { in_str = false; } continue; }
        if (in_chr)   { if (c == '\\')                     { i++; } else if (c == '\'') { in_chr  = false; } continue; }
        if (c == '/' && n == '/') { in_line  = true; i++; continue; }
        if (c == '/' && n == '*') { in_block = true; i++; continue; }
        if (c == '"')  { in_str = true; continue; }
        if (c == '\'') { in_chr  = true; continue; }
        if (c == '{') { depth++; } else if (c == '}') {
            depth--;
            if (depth == 0) {
                return src.substr(open_pos, i - open_pos + 1);
            }
        }
    }
    return "";
}

// whole file as one string ("" when unreadable)
static std::string read_source(const std::string & path) {
    std::ifstream in(path);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

// the source with comment characters blanked (string/char literals kept
// verbatim): searching the RESULT finds code only, never a mention that was
// commented out - the wiring gate below depends on this (a commented-out
// insert must FAIL the gate, not pass it by textual presence)
static std::string strip_comments(const std::string & src) {
    std::string res(src.size(), ' ');
    bool in_str = false, in_chr = false, in_line = false, in_block = false;
    for (size_t i = 0; i < src.size(); i++) {
        const char c = src[i];
        const char n = i + 1 < src.size() ? src[i + 1] : '\0';
        if (in_line)  { if (c == '\n') { in_line = false; res[i] = c; } continue; }
        if (in_block) { if (c == '*' && n == '/') { in_block = false; i++; } continue; }
        if (in_str || in_chr) {
            res[i] = c;
            if (c == '\\') { if (i + 1 < src.size()) { res[i + 1] = src[i + 1]; } i++; }
            else if ((in_str && c == '"') || (in_chr && c == '\'')) { in_str = in_chr = false; }
            continue;
        }
        if (c == '/' && n == '/') { in_line  = true; i++; continue; }
        if (c == '/' && n == '*') { in_block = true; i++; continue; }
        if (c == '"')  { in_str = true; res[i] = c; continue; }
        if (c == '\'') { in_chr = true; res[i] = c; continue; }
        res[i] = c;
    }
    return res;
}

// one collected log statement: every string literal of the call
struct srv_log_statement {
    std::vector<std::string> literals;
};

// T32-b task 6 (step 6.2): collect the string literals of every
// SRV_INF/SRV_WRN/SRV_ERR/SLT_INF/SLT_WRN/SLT_ERR statement in `src` (the
// log levels the park family lives at; DBG chatter is out of scope). One
// comment/string-aware pass - the same discipline as function_body - so macro
// names inside comments or strings never match, and each statement stops at
// its own ';' (paren/brace depth 0, lambdas in arguments included). Returns
// literal CONTENTS without the surrounding quotes.
static std::vector<srv_log_statement> scan_srv_log_statements(const std::string & src) {
    auto is_log_macro = [](const std::string & id) {
        return id == "SRV_INF" || id == "SRV_WRN" || id == "SRV_ERR" ||
               id == "SLT_INF" || id == "SLT_WRN" || id == "SLT_ERR";
    };

    std::vector<srv_log_statement> res;
    bool in_str = false, in_chr = false, in_line = false, in_block = false;
    size_t i = 0;
    while (i < src.size()) {
        const char c = src[i];
        const char n = i + 1 < src.size() ? src[i + 1] : '\0';
        if (in_line)  { if (c == '\n')                     { in_line  = false; } i++; continue; }
        if (in_block) { if (c == '*' && n == '/')          { in_block = false; i++; } i++; continue; }
        if (in_str)   { if (c == '\\')                     { i++; } else if (c == '"')  { in_str  = false; } i++; continue; }
        if (in_chr)   { if (c == '\\')                     { i++; } else if (c == '\'') { in_chr  = false; } i++; continue; }
        if (c == '/' && n == '/') { in_line  = true; i++; continue; }
        if (c == '/' && n == '*') { in_block = true; i++; continue; }
        if (c == '"')  { in_str = true;  i++; continue; }
        if (c == '\'') { in_chr = true;  i++; continue; }
        if (std::isalpha((unsigned char) c) || c == '_') {
            size_t j = i;
            while (j < src.size() && (std::isalnum((unsigned char) src[j]) || src[j] == '_')) {
                j++;
            }
            if (is_log_macro(src.substr(i, j - i))) {
                // the call's '(' may sit after whitespace and comments; any
                // other token means this occurrence is not a call
                size_t k = j;
                bool is_call = false;
                while (k < src.size()) {
                    if (std::isspace((unsigned char) src[k])) { k++; continue; }
                    if (src[k] == '/' && k + 1 < src.size() && src[k + 1] == '/') {
                        while (k < src.size() && src[k] != '\n') { k++; }
                        continue;
                    }
                    if (src[k] == '/' && k + 1 < src.size() && src[k + 1] == '*') {
                        k += 2;
                        while (k + 1 < src.size() && !(src[k] == '*' && src[k + 1] == '/')) { k++; }
                        k += 2;
                        continue;
                    }
                    is_call = (src[k] == '(');
                    break;
                }
                if (is_call) {
                    srv_log_statement stmt;
                    int depth = 0, braces = 0;
                    bool s_str = false, s_chr = false, s_ln = false, s_bl = false, collecting = false;
                    std::string cur;
                    size_t end = src.size();
                    for (size_t m = k; m < src.size(); m++) {
                        const char d = src[m];
                        const char e = m + 1 < src.size() ? src[m + 1] : '\0';
                        if (s_ln) { if (d == '\n') { s_ln = false; } continue; }
                        if (s_bl) { if (d == '*' && e == '/') { s_bl = false; m++; } continue; }
                        if (s_str) {
                            if (d == '\\') { if (collecting) { cur += d; cur += e; } m++; }
                            else if (d == '"') { s_str = false; if (collecting) { stmt.literals.push_back(cur); collecting = false; } }
                            else if (collecting) { cur += d; }
                            continue;
                        }
                        if (s_chr) { if (d == '\\') { m++; } else if (d == '\'') { s_chr = false; } continue; }
                        if (d == '/' && e == '/') { s_ln = true; m++; continue; }
                        if (d == '/' && e == '*') { s_bl = true; m++; continue; }
                        if (d == '"')  { s_str = true; collecting = true; cur.clear(); continue; }
                        if (d == '\'') { s_chr = true; continue; }
                        if (d == '(' || d == '[') { depth++; }
                        else if (d == ')' || d == ']') { depth--; }
                        else if (d == '{') { braces++; }
                        else if (d == '}') { braces--; }
                        else if (d == ';' && depth == 0 && braces == 0) { end = m; break; }
                    }
                    res.push_back(std::move(stmt));
                    i = end; // resume the outer pass after this statement
                    continue;
                }
            }
            i = j;
            continue;
        }
        i++;
    }
    return res;
}

int main() {
    // --- 1. feature gate: the park library only exists with a positive budget ---
    {
        park_cfg cfg; // defaults: mib = 0, heuristic NONE
        CHECK(cfg.mib == 0, "default park budget is 0");
        CHECK(!park_enabled(cfg), "default park-mib 0 -> park disabled (identity irrelevant)");

        park_cfg on = cfg;
        on.mib = 1024;
        CHECK(park_enabled(on), "mib > 0 -> park enabled");
    }

    // --- 2. header identity: exact "main" marks the main-agent task ---
    CHECK(park_from_header("main"), "header \"main\" -> main");
    CHECK(!park_from_header("sub"), "header \"sub\" -> not main");
    CHECK(!park_from_header(""), "empty header -> not main");
    CHECK(!park_from_header("MAIN"), "header match is exact (caller normalizes case)");

    // --- 3. heuristic NONE: never identifies a main task ---
    {
        park_cfg cfg;
        cfg.mib = 1024; // heuristic stays NONE
        heur_state st;
        st.seen.insert(10000);
        st.seen.insert(40000);
        st.seen.insert(33000);
        CHECK(!park_task_is_main(cfg, st, false, 40000), "heuristic disabled -> false even for the longest");
        CHECK(!park_task_is_main(cfg, st, false, 100000), "heuristic disabled -> false for any token count");
    }

    // --- 4. heuristic longest, registry {10k, 40k, 33k}: only 40k is main ---
    {
        heur_state st;
        st.seen.insert(10000);
        st.seen.insert(40000);
        st.seen.insert(33000);
        CHECK(!park_from_heuristic(st, 10000), "10k: not the longest -> not main");
        CHECK(park_from_heuristic(st, 40000), "40k: longest and > 32768 -> main");
        CHECK(!park_from_heuristic(st, 33000), "33k: not the longest -> not main");
    }

    // --- 5. two entries above the threshold: only the larger one is main ---
    {
        heur_state st;
        st.seen.insert(40000);
        st.seen.insert(50000);
        CHECK(park_from_heuristic(st, 50000), "50k (the larger) -> main");
        CHECK(!park_from_heuristic(st, 40000), "40k above threshold but not the max -> not main");
    }

    // --- 6. nothing above 32768: no main task at all ---
    {
        heur_state st;
        st.seen.insert(100);
        st.seen.insert(1000);
        st.seen.insert(32767);
        CHECK(!park_from_heuristic(st, 32767), "max is 32767 <= 32768 -> not main");
        CHECK(!park_from_heuristic(st, 1000), "not the max -> not main");
    }

    // --- 7. exactly 32768: the threshold is strictly greater ---
    {
        heur_state st;
        st.seen.insert(32768);
        CHECK(!park_from_heuristic(st, 32768), "exactly 32768 is not > 32768 -> not main");
        st.seen.insert(32769);
        CHECK(park_from_heuristic(st, 32769), "32769 > 32768 and the max -> main");
    }

    // --- 8. empty registry: no main task (max would not exist) ---
    {
        heur_state st;
        CHECK(!park_from_heuristic(st, 100000), "empty registry -> not main");
    }

    // --- 9. combined: a non-main header is not a veto, the heuristic decides ---
    {
        park_cfg cfg;
        cfg.mib = 1024;
        cfg.heuristic = park_cfg::LONGEST;
        heur_state st;
        st.seen.insert(10000);
        st.seen.insert(40000);
        st.seen.insert(33000);
        CHECK(park_task_is_main(cfg, st, park_from_header("sub"), 40000),
            "header != \"main\" + longest + max tokens -> heuristic decides main");
        CHECK(!park_task_is_main(cfg, st, park_from_header("sub"), 33000),
            "header != \"main\" + not the longest -> not main");
    }

    // --- 10. combined: header "main" wins over the heuristic ---
    {
        park_cfg cfg;
        cfg.mib = 1024;
        cfg.heuristic = park_cfg::LONGEST;
        heur_state st;
        st.seen.insert(10000);
        st.seen.insert(40000);
        st.seen.insert(33000);
        CHECK(park_task_is_main(cfg, st, true, 10000),
            "header main + tokens not the max -> still main (header wins)");
        CHECK(!park_task_is_main(cfg, st, false, 10000),
            "no header + tokens not the max -> not main");

        park_cfg cfg_none;
        cfg_none.mib = 1024; // heuristic NONE
        CHECK(park_task_is_main(cfg_none, st, true, 100),
            "header main + heuristic NONE -> still main");
    }

    // --- 11. flag parsing: --cache-disk-park-mib (0 = off, otherwise >= 1024)
    //         and --cache-disk-park-heuristic (none | longest) ---
    {
        auto list_str_to_char = [](std::vector<std::string> & argv) -> std::vector<char *> {
            std::vector<char *> res;
            for (auto & arg : argv) {
                res.push_back(const_cast<char *>(arg.data()));
            }
            return res;
        };
        auto parse = [&](std::vector<std::string> argv, common_params & params) -> bool {
            return common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SERVER);
        };

        common_params params;
        CHECK(params.cache_disk_park_mib == 0, "default --cache-disk-park-mib is 0");
        CHECK(params.cache_disk_park_heuristic == "none", "default --cache-disk-park-heuristic is none");

        // accepted values
        params = common_params();
        CHECK(parse({"binary_name", "--cache-disk-park-mib", "0"}, params), "parse --cache-disk-park-mib 0");
        CHECK(params.cache_disk_park_mib == 0, "--cache-disk-park-mib 0 -> off");

        params = common_params();
        CHECK(parse({"binary_name", "--cache-disk-park-mib", "10240"}, params), "parse --cache-disk-park-mib 10240");
        CHECK(params.cache_disk_park_mib == 10240, "--cache-disk-park-mib 10240 -> accepted");

        params = common_params();
        CHECK(parse({"binary_name", "--cache-disk-park-mib", "1024"}, params), "parse --cache-disk-park-mib 1024");
        CHECK(params.cache_disk_park_mib == 1024, "--cache-disk-park-mib 1024 -> accepted (the minimum itself)");

        params = common_params();
        CHECK(parse({"binary_name", "--cache-disk-park-heuristic", "longest"}, params), "parse --cache-disk-park-heuristic longest");
        CHECK(params.cache_disk_park_heuristic == "longest", "--cache-disk-park-heuristic longest -> accepted");

        params = common_params();
        CHECK(parse({"binary_name", "--cache-disk-park-heuristic", "none"}, params), "parse --cache-disk-park-heuristic none");
        CHECK(params.cache_disk_park_heuristic == "none", "--cache-disk-park-heuristic none -> accepted");

        // rejected values (the park budget inherits the --cache-disk-persist-mib minimum)
        params = common_params();
        CHECK(!parse({"binary_name", "--cache-disk-park-mib", "-1"}, params), "reject --cache-disk-park-mib -1");
        CHECK(params.cache_disk_park_mib == 0, "rejected value leaves the default untouched");

        params = common_params();
        CHECK(!parse({"binary_name", "--cache-disk-park-mib", "1023"}, params), "reject --cache-disk-park-mib 1023 (one below the minimum)");
        CHECK(params.cache_disk_park_mib == 0, "rejected boundary leaves the default untouched");

        params = common_params();
        CHECK(!parse({"binary_name", "--cache-disk-park-mib", "500"}, params), "reject --cache-disk-park-mib 500 (below minimum)");

        params = common_params();
        CHECK(!parse({"binary_name", "--cache-disk-park-mib", "foo"}, params), "reject non-numeric --cache-disk-park-mib");

        params = common_params();
        CHECK(!parse({"binary_name", "--cache-disk-park-heuristic", "foo"}, params), "reject unknown --cache-disk-park-heuristic value");
    }

    // --- 12. raw header VALUE: X-Pi-Role is matched case-insensitively ---
    // park_from_header stays exact-match (section 2, the caller normalizes);
    // park_header_value is the normalizing wrapper the server calls on the
    // raw value as it arrived on the wire
    CHECK(park_header_value("main"), "raw header \"main\" -> main");
    CHECK(park_header_value("Main"), "raw header \"Main\" -> main (value case-insensitive)");
    CHECK(park_header_value("MAIN"), "raw header \"MAIN\" -> main (value case-insensitive)");
    CHECK(!park_header_value("subagent"), "raw header \"subagent\" -> not main");
    CHECK(!park_header_value(""), "absent/empty header -> not main");

    // --- 13. NEGATIVE TEST (spec §5 trust): "park_main": true in the request
    //         BODY must have ZERO effect ---
    // The caller flow (handle_completions_impl -> params_from_json_cmpl ->
    // task.params.park_main) needs a full server, which is not linkable here,
    // so the guarantee is enforced as a gate instead: the BODY of the JSON
    // parser (server_task::params_from_json_cmpl, tools/server/server-task.cpp)
    // must never mention park - the only writer of task_params.park_main is
    // the convergence point in server-context.cpp, from the X-Pi-Role header.
    // The scan is scoped to the brace-matched function body on purpose: other
    // functions in the same file WILL legitimately touch park (the park
    // library wiring), and the invariant we want is exactly "the JSON parser
    // never reads/writes the park field". The parser source is re-scanned at
    // runtime so the gate cannot rot; if the source tree is not reachable
    // from the test binary it is skipped.
    {
        const std::string self = __FILE__;
        const std::string suffix = "tests/test-park-identity.cpp";
        const std::string root = self.size() > suffix.size() ? self.substr(0, self.size() - suffix.size()) : "./";
        std::ifstream in(root + "tools/server/server-task.cpp");
        std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        if (!in || content.empty()) {
            printf("note: negative-test gate skipped (source tree not reachable from the test binary)\n");
            g_source_gate_skips++;
        } else {
            const std::string body = function_body(content, "task_params server_task::params_from_json_cmpl");
            CHECK(!body.empty(), "gate sanity: the params_from_json_cmpl body was located");
            CHECK(body.find("return") != std::string::npos && body.size() > 1000 && body.size() < 100000,
                "gate sanity: the body slice is non-degenerate");
            CHECK(body.find("park") == std::string::npos,
                "negative test: the params_from_json_cmpl body never reads or writes a park field");
        }
    }

    // --- 14. park_should_redirect: the combined task-switch predicate (T32-b
    //         step 4.3a). Conjuncts: park on AND a previous task on the slot
    //         AND that task is the main agent's (header authoritative, LONGEST
    //         heuristic via the registry). The call site adds one more
    //         conjunct outside the pure helper - a live park instance, which
    //         exists exactly when the disk cache is configured.
    {
        park_cfg off;                    // defaults: mib = 0, heuristic NONE
        park_cfg on;   on.mib = 1024;    // heuristic NONE (the default)
        park_cfg on_l; on_l.mib = 1024; on_l.heuristic = park_cfg::LONGEST;

        heur_state empty;
        heur_state longest;
        longest.seen.insert(10000);
        longest.seen.insert(50000);

        CHECK(!park_should_redirect(off, longest, true,  true,  50000),
            "park disabled -> never redirect (even header main + heuristic hit)");
        CHECK(!park_should_redirect(on,   empty,  false, true,  50000),
            "no previous task on the slot -> never redirect (even header main)");
        CHECK( park_should_redirect(on,   empty,  true,  true,  100),
            "header main -> redirect (heuristic NONE, short prompt, empty registry)");
        CHECK(!park_should_redirect(on,   empty,  true,  false, 100),
            "no header + heuristic NONE + a previous task -> not main");
        CHECK(!park_should_redirect(on_l, longest, true,  false, 40000),
            "LONGEST: 40000 is not the registry max -> not main");
        CHECK( park_should_redirect(on_l, longest, true,  false, 50000),
            "LONGEST: 50000 is the registry max and > 32768 -> redirect");
        CHECK(!park_should_redirect(on,   longest, true,  false, 50000),
            "LONGEST-shaped registry but heuristic NONE -> not main");
        CHECK(!park_should_redirect(on_l, empty,   true,  false, 100),
            "LONGEST with an empty registry never fires on its own -> false");
        CHECK( park_should_redirect(on_l, longest, true,  true,  100),
            "header main wins even where the heuristic would say no");
    }

    // --- 15. TASK 6 WIRING GATES: the registry insert lives at task ARRIVAL.
    //         The pure LONGEST semantics are sections 4-8 above; what task 6
    //         adds is the prod wiring - seen.insert when a task enters the
    //         scheduler, never at save time. Constructing a real
    //         server_context is not linkable here, so (in the section-13
    //         style) the guarantee is enforced as source gates, re-scanned at
    //         runtime so they cannot rot.
    //
    //         Where the insert landed and why (declared): tokens ARE already
    //         materialized where the task object is built
    //         (handle_completions_impl tokenizes before the task loop), but
    //         that code runs on HTTP worker threads, which hold a CONST view
    //         of the server context and run concurrently with the task loop;
    //         impl state is mutated only from the loop. process_single_task
    //         is the arrival point ON the loop thread - the same thread that
    //         later evaluates park_heur in get_available_slot - so the
    //         multiset needs no locking, and the insert still strictly
    //         precedes every save evaluation that involves the task.
    {
        const std::string self = __FILE__;
        const std::string suffix = "tests/test-park-identity.cpp";
        const std::string root = self.size() > suffix.size() ? self.substr(0, self.size() - suffix.size()) : "./";

        const std::string ctx_src  = read_source(root + "tools/server/server-context.cpp");
        const std::string task_src = read_source(root + "tools/server/server-task.cpp");
        if (ctx_src.empty() || task_src.empty()) {
            printf("note: task-6 wiring gates skipped (source tree not reachable from the test binary)\n");
            g_source_gate_skips++;
        } else {
            // comment-blind: the insert must be LIVE code, not a mention
            const std::string ctx_code = strip_comments(ctx_src);

            size_t n_inserts = 0;
            for (size_t p = ctx_code.find("park_heur.seen.insert"); p != std::string::npos;
                    p = ctx_code.find("park_heur.seen.insert", p + 1)) {
                n_inserts++;
            }
            CHECK(n_inserts == 1, "wiring: exactly one live park_heur.seen.insert in server-context.cpp");

            const std::string body = function_body(ctx_code, "void process_single_task(server_task && task)");
            CHECK(!body.empty() && body.find("get_available_slot") != std::string::npos,
                "wiring gate sanity: the process_single_task body was located");
            CHECK(body.find("park_heur.seen.insert") != std::string::npos,
                "wiring: the registry insert lives at task arrival (process_single_task), not at save time");
            CHECK(body.find("SERVER_TASK_TYPE_COMPLETION") != std::string::npos &&
                  body.find("park_enabled(park_base)") != std::string::npos &&
                  body.find("park_base.heuristic == park_cfg::LONGEST") != std::string::npos,
                "wiring: the insert is gated to completion traffic, the feature flag and the LONGEST heuristic (the only registry reader)");

            // task 6.1: the successful park disk-save carries its own marker
            CHECK(strip_comments(task_src).find("persist park: save entry=") != std::string::npos,
                "wiring: a successful park disk-save logs the persist park: save marker");

            // final-review fix gate: the heuristic must be fed the task's
            // ARRIVAL size (what the registry recorded), never the slot
            // boundary (which includes generated tokens and would make the
            // strict-equality check unreachable for any task that generated)
            {
                const size_t call = ctx_code.find("park_should_redirect(park_base");
                CHECK(call != std::string::npos,
                    "wiring sanity: the redirect predicate call site was located");
                const std::string args = ctx_code.substr(call, 400);  // spans the nested casts/ternaries of the full call
                CHECK(args.find("task_prev->tokens.size()") != std::string::npos &&
                      args.find("prompt.tokens.size()") == std::string::npos,
                    "wiring: the heuristic input is the task ARRIVAL size (task_prev->tokens), not the slot boundary");
            }
        }
    }

    // --- 16. STEP 6.2 LOG-PREFIX GATE: every SRV/SLT INF/WRN/ERR log line of
    //         the park feature carries the family prefix. Declared rule: in
    //         tools/server/server-context.cpp and server-task.cpp, every
    //         scanned log statement whose string literals mention "park"
    //         (case-insensitive) must have one of those literals start with
    //         "persist park:", "park " or "persist restore: park" (the task-5
    //         restore marker, whose exact name the plan's integration cells
    //         grep for). One exception: literals starting with "--" are
    //         startup flag-validation warnings naming CLI flags (e.g.
    //         "--cache-disk-park-mib ignored: ..."), a pre-existing general
    //         family that merely mentions the flag - out of scope. A floor of
    //         8 family lines keeps the gate from silently matching nothing
    //         (enabled, disabled, redirect, skip, fallback, supersede,
    //         supersede-failed, save, restore = 9 at the time of writing).
    {
        const std::string self = __FILE__;
        const std::string suffix = "tests/test-park-identity.cpp";
        const std::string root = self.size() > suffix.size() ? self.substr(0, self.size() - suffix.size()) : "./";

        const std::string ctx_src  = read_source(root + "tools/server/server-context.cpp");
        const std::string task_src = read_source(root + "tools/server/server-task.cpp");
        if (ctx_src.empty() || task_src.empty()) {
            printf("note: log-prefix gate skipped (source tree not reachable from the test binary)\n");
            g_source_gate_skips++;
        } else {
            auto mentions_park = [](const std::string & lit) {
                for (size_t p = 0; p + 4 <= lit.size(); p++) {
                    if (std::tolower((unsigned char) lit[p])     == 'p' &&
                        std::tolower((unsigned char) lit[p + 1]) == 'a' &&
                        std::tolower((unsigned char) lit[p + 2]) == 'r' &&
                        std::tolower((unsigned char) lit[p + 3]) == 'k') {
                        return true;
                    }
                }
                return false;
            };
            size_t family_lines = 0;
            for (const auto * src : { &ctx_src, &task_src }) {
                for (const auto & stmt : scan_srv_log_statements(*src)) {
                    bool in_family = false;
                    for (const std::string & lit : stmt.literals) {
                        if (!mentions_park(lit)) {
                            continue;
                        }
                        const bool ok = lit.rfind("persist park:", 0) == 0 ||
                                        lit.rfind("park ", 0)         == 0 ||
                                        lit.rfind("persist restore: park", 0) == 0 ||
                                        lit.rfind("--", 0)            == 0;
                        CHECK(ok, "step 6.2: park log line without the family prefix");
                        if (lit.rfind("persist park:", 0) == 0 || lit.rfind("park ", 0) == 0 ||
                                lit.rfind("persist restore: park", 0) == 0) {
                            in_family = true;
                        }
                    }
                    if (in_family) {
                        family_lines++;
                    }
                }
            }
            CHECK(family_lines >= 8,
                "step 6.2 gate sanity: the scan found the park log family (>= 8 lines)");
        }
    }

    if (g_failures == 0) {
        if (g_source_gate_skips > 0) {
            printf("test-park-identity: ALL PASS (%d source gates skipped - run inside the builder container for full coverage)\n",
                g_source_gate_skips);
        } else {
            printf("test-park-identity: ALL PASS\n");
        }
        return 0;
    }
    if (g_source_gate_skips > 0) {
        printf("test-park-identity: %d FAILURES (%d source gates skipped - run inside the builder container for full coverage)\n",
            g_failures, g_source_gate_skips);
    } else {
        printf("test-park-identity: %d FAILURES\n", g_failures);
    }
    return 1;
}
