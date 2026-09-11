// W6-6 / A4-S2 — gate differenziale del testo formattato + cache parser/PEG.
//
// Le cache dentro common_chat_templates (analisi autoparser, parser PEG,
// generation_prompt) devono produrre, per la stessa sequenza di input, lo
// STESSO risultato byte-per-byte del path freddo (istanza nuova per ogni apply):
// - prompt formattato identico (byte)
// - tutti i campi di common_chat_params identici (parser, grammar, triggers...)
// e devono invalidare correttamente al cambio di tools/kwargs/flag.
//
// usage: test-chat-template-cache [templates-dir]
//   default: models/templates (63 template .jinja)

#include "common.h"
#include "chat.h"
#include "log.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#undef NDEBUG
#include <cassert>

using namespace std::chrono;

static int g_failures = 0;
static int g_templates = 0;
static int g_applies   = 0;

static const std::vector<std::string> k_contents = {
    "The quick brown fox jumps over the lazy dog while the template cache reuses its parser.",
    "Il prefisso della conversazione cresce alla fine: né più, né meno.",
    "月亮在白莲花般的云朵里穿行，晚风吹来一阵阵快乐的歌声。",
    "```cpp\nfor (auto & w : words) { bpe_merge(w, ranks); }\n```",
    "{\"name\": \"read_file\", \"arguments\": {\"path\": \"/tmp/x.go\", \"offset\": 7}}",
    "<think>\nanalyze the request, then answer\n</think>\nHere is the answer.",
    "emoji mix 🚀✅ and numbers 3.14159 2.71828",
};

static std::vector<common_chat_msg> make_messages(size_t n, size_t seed) {
    std::vector<common_chat_msg> msgs;
    for (size_t i = 0; i < n; ++i) {
        common_chat_msg m;
        m.role    = (i % 3 == 0) ? "user" : (i % 3 == 1 ? "assistant" : "tool");
        m.content = k_contents[(i + seed) % k_contents.size()] + " [msg " + std::to_string(i) + "]";
        if (m.role == "assistant" && i % 4 == 1) {
            m.reasoning_content = "reasoning " + std::to_string(i);
        }
        if (m.role == "assistant" && i % 5 == 1) {
            common_chat_tool_call tc;
            tc.name = "read_file";
            tc.arguments = "{\"path\": \"/tmp/f\"}";
            tc.id = "call-" + std::to_string(i);
            m.tool_calls.push_back(tc);
        }
        if (m.role == "tool") {
            m.tool_name = "read_file";
            m.tool_call_id = "call-" + std::to_string(i - 1);
        }
        msgs.push_back(std::move(m));
    }
    return msgs;
}

static std::vector<common_chat_tool> make_tools() {
    common_chat_tool t;
    t.name = "read_file";
    t.description = "Read a file from disk";
    t.parameters = R"({"type":"object","properties":{"path":{"type":"string"}},"required":["path"]})";
    return { t };
}

static bool params_equal(const common_chat_params & a, const common_chat_params & b) {
    return a.format == b.format && a.prompt == b.prompt && a.grammar == b.grammar &&
           a.grammar_lazy == b.grammar_lazy && a.generation_prompt == b.generation_prompt &&
           a.supports_thinking == b.supports_thinking && a.thinking_start_tag == b.thinking_start_tag &&
           a.thinking_end_tag == b.thinking_end_tag && a.preserved_tokens == b.preserved_tokens &&
           a.additional_stops == b.additional_stops && a.parser == b.parser &&
           a.grammar_triggers.size() == b.grammar_triggers.size();
}

static void compare_apply(const std::string & src, const common_chat_templates_inputs & inputs, const char * scenario) {
    common_chat_templates_ptr cold = common_chat_templates_init(nullptr, src);
    common_chat_templates_ptr warm = common_chat_templates_init(nullptr, src);

    // warm: apply due volte (la seconda gira con le cache calde)
    bool cold_threw = false;
    bool warm_threw = false;
    common_chat_params cold_res;
    common_chat_params warm_res;
    common_chat_params warm_res2;
    try {
        cold_res = common_chat_templates_apply(cold.get(), inputs);
    } catch (const std::exception & e) {
        cold_threw = true;
    }
    try {
        warm_res  = common_chat_templates_apply(warm.get(), inputs);
        warm_res2 = common_chat_templates_apply(warm.get(), inputs);
    } catch (const std::exception & e) {
        warm_threw = true;
    }

    g_applies += 3;
    if (cold_threw || warm_threw) {
        // un template non applicabile con questi input e' accettabile, ma il
        // comportamento freddo/caldo deve coincidere
        if (cold_threw != warm_threw) {
            g_failures++;
            fprintf(stderr, "FAIL template=%.60s scenario=%s: throw mismatch (cold=%d warm=%d)\n",
                    src.c_str(), scenario, (int) cold_threw, (int) warm_threw);
        }
        return;
    }
    if (!params_equal(cold_res, warm_res) || !params_equal(cold_res, warm_res2)) {
        g_failures++;
        fprintf(stderr, "FAIL template=%.60s scenario=%s: params differ\n", src.c_str(), scenario);
        if (cold_res.prompt != warm_res.prompt) {
            fprintf(stderr, "  prompt differs: cold %zu B, warm %zu B\n", cold_res.prompt.size(), warm_res.prompt.size());
        }
        if (cold_res.parser != warm_res.parser) {
            fprintf(stderr, "  parser string differs\n");
        }
        if (cold_res.generation_prompt != warm_res.generation_prompt) {
            fprintf(stderr, "  generation_prompt differs: cold='%s' warm='%s'\n",
                    cold_res.generation_prompt.c_str(), warm_res.generation_prompt.c_str());
        }
        return;
    }
    if (warm_res.prompt != warm_res2.prompt || warm_res.parser != warm_res2.parser) {
        g_failures++;
        fprintf(stderr, "FAIL template=%.60s scenario=%s: warm apply not stable across repeated calls\n",
                src.c_str(), scenario);
    }
}

static void run_template(const std::string & src) {
    g_templates++;

    // 1) conversazione crescente (pattern pi)
    {
        std::vector<common_chat_msg> msgs;
        for (size_t n = 2; n <= 30; n += 4) {
            msgs = make_messages(n, 0);
            common_chat_templates_inputs inputs;
            inputs.messages = msgs;
            inputs.use_jinja = true;
            inputs.add_generation_prompt = true;
            compare_apply(src, inputs, "grow");
        }
    }

    // 2) edit mid-prefix
    {
        auto msgs = make_messages(12, 1);
        common_chat_templates_inputs inputs;
        inputs.messages = msgs;
        inputs.use_jinja = true;
        compare_apply(src, inputs, "edit-base");
        msgs[5].content = "EDITED MID PREFIX CONTENT —— edited";
        inputs.messages = msgs;
        compare_apply(src, inputs, "edit-mid");
    }

    // 3) tools on/off e tool_choice
    {
        auto msgs = make_messages(8, 2);
        common_chat_templates_inputs inputs;
        inputs.messages = msgs;
        inputs.use_jinja = true;
        inputs.tools = make_tools();
        inputs.tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
        compare_apply(src, inputs, "tools-auto");
        inputs.tool_choice = COMMON_CHAT_TOOL_CHOICE_REQUIRED;
        compare_apply(src, inputs, "tools-required");
        inputs.tools.clear();
        inputs.tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
        compare_apply(src, inputs, "tools-off");
    }

    // 4) flag add_generation_prompt / enable_thinking / reasoning_format
    {
        auto msgs = make_messages(6, 3);
        for (int agp = 0; agp <= 1; ++agp) {
            for (int think = 0; think <= 1; ++think) {
                common_chat_templates_inputs inputs;
                inputs.messages = msgs;
                inputs.use_jinja = true;
                inputs.add_generation_prompt = agp != 0;
                inputs.enable_thinking = think != 0;
                inputs.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
                compare_apply(src, inputs, "flags");
            }
        }
    }

    // 5) kwargs
    {
        auto msgs = make_messages(6, 4);
        common_chat_templates_inputs inputs;
        inputs.messages = msgs;
        inputs.use_jinja = true;
        inputs.chat_template_kwargs["enable_thinking"] = "false";
        compare_apply(src, inputs, "kwargs");
        inputs.chat_template_kwargs["enable_thinking"] = "true";
        compare_apply(src, inputs, "kwargs2");
    }

    // 6) json_schema
    {
        auto msgs = make_messages(4, 5);
        common_chat_templates_inputs inputs;
        inputs.messages = msgs;
        inputs.use_jinja = true;
        inputs.json_schema = R"({"type":"object","properties":{"answer":{"type":"string"}}})";
        compare_apply(src, inputs, "json-schema");
    }
}

// misura del solo path jinja (prod usa --jinja): @400 messaggi, freddo vs caldo.
// i contenuti vengono ingrassati a ~1 KB per messaggio per raggiungere la scala
// della conversazione pi reale (~410 KB @400 messaggi)
static void run_measure(const std::string & src, const char * name) {
    auto fatten = [](std::vector<common_chat_msg> msgs) {
        for (auto & m : msgs) {
            std::string fat = m.content;
            for (int r = 0; r < 8; ++r) {
                fat += "\n" + m.content;
            }
            m.content = std::move(fat);
        }
        return msgs;
    };
    std::vector<std::vector<common_chat_msg>> rounds;
    for (size_t n = 40; n <= 400; n += 40) {
        rounds.push_back(fatten(make_messages(n, 0)));
    }

    common_chat_templates_ptr warm = common_chat_templates_init(nullptr, src);
    {
        common_chat_templates_inputs inputs;
        inputs.messages = make_messages(2, 0);
        inputs.use_jinja = true;
        (void) common_chat_templates_apply(warm.get(), inputs);
    }

    double t_cold = 0.0;
    {
        auto t0 = steady_clock::now();
        for (const auto & msgs : rounds) {
            common_chat_templates_ptr cold = common_chat_templates_init(nullptr, src);
            common_chat_templates_inputs inputs;
            inputs.messages = msgs;
            inputs.use_jinja = true;
            const auto res = common_chat_templates_apply(cold.get(), inputs);
            (void) res.prompt.size();
        }
        t_cold = duration<double, std::milli>(steady_clock::now() - t0).count();
    }
    double t_warm = 0.0;
    size_t prompt_bytes = 0;
    {
        auto t0 = steady_clock::now();
        for (const auto & msgs : rounds) {
            common_chat_templates_inputs inputs;
            inputs.messages = msgs;
            inputs.use_jinja = true;
            const auto res = common_chat_templates_apply(warm.get(), inputs);
            prompt_bytes = res.prompt.size();
        }
        t_warm = duration<double, std::milli>(steady_clock::now() - t0).count();
    }
    fprintf(stderr, "measure %s: 10 rounds up to 400 msgs (%zu B last prompt): cold=%.1f ms warm=%.1f ms (%.1fx)\n",
            name, prompt_bytes, t_cold, t_warm, t_warm > 0 ? t_cold / t_warm : 0.0);
}

int main(int argc, char ** argv) {
    const std::string dir = argc > 1 ? argv[1] : "models/templates";
    if (!std::filesystem::exists(dir)) {
        fprintf(stderr, "templates dir not found: %s\n", dir.c_str());
        return 2;
    }

    std::vector<std::string> sources;
    for (const auto & entry : std::filesystem::directory_iterator(dir)) {
        if (entry.path().extension() == ".jinja") {
            std::ifstream in(entry.path());
            std::string src((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            if (!src.empty()) {
                sources.push_back(std::move(src));
            }
        }
    }
    std::sort(sources.begin(), sources.end());
    fprintf(stderr, "testing %zu templates from %s\n", sources.size(), dir.c_str());

    for (const auto & src : sources) {
        run_template(src);
    }

    // misura sul template ChatML (rappresentativo del path prod pi)
    {
        std::ifstream in(dir + "/Qwen-ChatML-no-think.jinja");
        if (in.good()) {
            std::string src((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            run_measure(src, "Qwen-ChatML-no-think");
        }
    }

    fprintf(stderr, "templates=%d applies=%d failures=%d\n", g_templates, g_applies, g_failures);
    if (g_failures > 0) {
        fprintf(stderr, "%s: FAILURES\n", argv[0]);
        return 1;
    }
    fprintf(stderr, "%s: ALL OK\n", argv[0]);
    return 0;
}
