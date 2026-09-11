// W7-5 — census residui S3/S4 del path per-request pi + gate bit-identita' token.
//
// Il path host per-request (post-W6-6, lato HTTP worker) e':
//   json::parse(body) -> oaicompat_chat_params_parse (validazione + parse
//   messaggi + apply template jinja con cache W6-6) -> tokenize_input_prompts
//   (token-prefix cache W6-6 -> tokenize BPE) -> server_task::params_from_json_cmpl.
// Questo test esercita ESATTAMENTE quelle funzioni su conversazioni crescenti
// (pattern client pi stateless) e:
//
//   1. IDENTITA': per ogni round stampa sha256 del testo formattato e dello
//      stream di token prodotto dall'intero path (dump del prompt formattato
//      su disco per il cross-check col tokenizer pinnato in Python);
//   2. CENSUS: misura per-componente (body parse / chat params / parse
//      messaggi / apply template / tokenize piena vs cache / task params /
//      copia prompt) a scala del round piu' grande disponibile;
//   3. CENSUS-SINTETICO: conversazione crescente alla scala pi (~412 KB,
//      metodologia W6-6 run_e2e_measure) attraverso lo STESSO path body->token,
//      per numeri comparabili col report W6-6.
//
// usage: test-per-request-path <vocab.gguf> <rounds.jsonl> <out-dir> [--census <pi-corpus.txt>] [--iters N] [--skip-synth]

#include "server-common.h"
#include "server-task.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

extern "C" {
#include "sha256/sha256.h"
}

using namespace std::chrono;

static steady_clock::time_point t_now() {
    return steady_clock::now();
}

static double ms_since(steady_clock::time_point t0) {
    return duration<double, std::milli>(steady_clock::now() - t0).count();
}

static std::string sha256_hex(const void * data, size_t len) {
    unsigned char digest[SHA256_DIGEST_SIZE];
    sha256_t st;
    sha256_init(&st);
    sha256_update(&st, (const unsigned char *) data, len);
    sha256_final(&st, digest);
    static const char * hex = "0123456789abcdef";
    std::string out;
    out.reserve(SHA256_DIGEST_SIZE * 2);
    for (unsigned char b : digest) {
        out.push_back(hex[b >> 4]);
        out.push_back(hex[b & 0xf]);
    }
    return out;
}

struct round_result {
    size_t n_msgs      = 0;
    size_t body_bytes  = 0;
    size_t prompt_bytes = 0;
    size_t n_tokens    = 0;
    std::string sha_prompt;
    std::string sha_tokens;
    // component timings (ms)
    double t_body = 0, t_chat_params = 0, t_tok_cache = 0, t_task_params = 0;
    double t_msgs = 0, t_apply = 0, t_tok_full = 0, t_prompt_copy = 0;
};

// un round del path reale: body string -> json -> chat params -> tokens.
// ritorna anche il prompt formattato (per il dump di identita').
static round_result run_round(const std::string & body_str, const llama_vocab * vocab,
                              const server_chat_params & opt, server_token_prefix_cache & cache,
                              const common_params & params_base, std::string * prompt_out) {
    round_result r;

    auto t0 = t_now();
    json body = json::parse(body_str);
    r.t_body = ms_since(t0);
    r.body_bytes = body_str.size();
    r.n_msgs = body.at("messages").size();

    std::vector<raw_buffer> files;
    t0 = t_now();
    json data = oaicompat_chat_params_parse(body, opt, files);
    r.t_chat_params = ms_since(t0);

    // sottocomponenti (passata strumentale separata, stesso input: non alimenta
    // il path dei token - serve solo a ripartire i ms di t_chat_params)
    {
        const json & messages = body.at("messages");
        t0 = t_now();
        auto msgs = common_chat_msgs_parse_oaicompat(messages);
        r.t_msgs = ms_since(t0);

        common_chat_templates_inputs inputs;
        inputs.messages              = std::move(msgs);
        inputs.use_jinja             = true;
        inputs.add_generation_prompt = true;
        inputs.reasoning_format      = COMMON_REASONING_FORMAT_DEEPSEEK;
        t0 = t_now();
        const auto cp = common_chat_templates_apply(opt.tmpls.get(), inputs);
        r.t_apply = ms_since(t0);
        if (prompt_out != nullptr) {
            *prompt_out = cp.prompt;
        }
    }

    t0 = t_now();
    std::vector<server_tokens> inputs = tokenize_input_prompts(vocab, nullptr, data.at("prompt"), true, true, &cache);
    r.t_tok_cache = ms_since(t0);
    const auto & tokens = inputs[0].get_tokens();
    r.n_tokens = tokens.size();

    // copia del prompt fuori dal DOM (quella che fa tokenize_mixed internamente)
    {
        t0 = t_now();
        std::string s = data.at("prompt").get<std::string>();
        r.t_prompt_copy = ms_since(t0);
        r.prompt_bytes = s.size();
        r.sha_prompt = sha256_hex(s.data(), s.size());
        // tokenize piena di riferimento (S3-domain, path senza cache)
        t0 = t_now();
        auto full = common_tokenize(vocab, s, true, true);
        r.t_tok_full = ms_since(t0);
        if (full != tokens) {
            fprintf(stderr, "FATAL: token stream del path != tokenize piena (%zu vs %zu)\n",
                    tokens.size(), full.size());
            abort();
        }
    }

    t0 = t_now();
    auto task_params = server_task::params_from_json_cmpl(vocab, params_base, 32768, {}, data);
    (void) task_params;
    r.t_task_params = ms_since(t0);

    r.sha_tokens = sha256_hex(tokens.data(), tokens.size() * sizeof(llama_token));
    return r;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <vocab.gguf> <rounds.jsonl> <out-dir> [--census <pi-corpus.txt>] [--iters N] [--skip-synth]\n", argv[0]);
        return 2;
    }
    const char * vocab_path = argv[1];
    const char * rounds_path = argv[2];
    const char * out_dir = argv[3];
    const char * census_corpus = nullptr;
    int iters = 3;
    bool skip_synth = false;
    for (int i = 4; i < argc; ++i) {
        if (strcmp(argv[i], "--census") == 0 && i + 1 < argc) {
            census_corpus = argv[++i];
        } else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            iters = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--skip-synth") == 0) {
            skip_synth = true;
        }
    }

    llama_model_params mp = llama_model_default_params();
    mp.vocab_only    = true;   // zero allocazioni device (gate ZERO GPU)
    mp.use_mmap      = true;
    mp.check_tensors = false;

    llama_model * model = llama_model_load_from_file(vocab_path, mp);
    common_log_set_verbosity_thold(1); // silenzia SRV_INF per-round (inquina le misure)
    if (model == nullptr) {
        fprintf(stderr, "failed to load vocab from %s\n", vocab_path);
        return 1;
    }
    std::filesystem::create_directories(out_dir);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // template del modello (= prod: --jinja, template embedded nel GGUF)
    server_chat_params opt;
    opt.use_jinja        = true;
    opt.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK; // default prod
    opt.tmpls            = common_chat_templates_init(model, "");

    common_params params_base; // defaulti per params_from_json_cmpl

    // round reali (client pi): una riga json = array messaggi del round
    std::vector<std::string> body_strs;
    {
        std::ifstream in(rounds_path);
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            body_strs.push_back(json({{ "messages", json::parse(line) }, { "stream", false }}).dump());
        }
    }
    if (body_strs.size() < 2) {
        fprintf(stderr, "servono almeno 2 round in %s\n", rounds_path);
        return 1;
    }
    fprintf(stderr, "round reali: %zu (body max %zu B)\n", body_strs.size(),
            std::max_element(body_strs.begin(), body_strs.end(), [](auto & a, auto & b) { return a.size() < b.size(); })->size());

    // ============ 1) identita' + census sui round reali ============
    // sequenza completa per iter: cache fresche alla prima iterazione, poi il
    // warm e' quello steady-state del server (istanze persistenti)
    std::vector<round_result> last(body_strs.size());
    for (int it = 0; it < iters; ++it) {
        server_token_prefix_cache cache(vocab);
        for (size_t i = 0; i < body_strs.size(); ++i) {
            std::string prompt_out;
            auto r = run_round(body_strs[i], vocab, opt, cache, params_base, &prompt_out);
            if (it == 0) {
                // dump del prompt formattato per il cross-check col tokenizer pinnato
                char path[512];
                snprintf(path, sizeof(path), "%s/round-%02zu.txt", out_dir, i);
                std::ofstream o(path, std::ios::binary);
                o.write(prompt_out.data(), prompt_out.size());
            }
            last[i] = r;
            fprintf(stderr,
                    "ROUND it=%d r=%02zu msgs=%3zu body=%7zu prompt=%7zu tokens=%6zu | "
                    "body=%.2f chat_params=%.2f (msgs=%.2f apply=%.2f) tok_cache=%.2f tok_full=%.2f task_params=%.2f prompt_copy=%.2f | "
                    "sha_tok=%s\n",
                    it, i, r.n_msgs, r.body_bytes, r.prompt_bytes, r.n_tokens,
                    r.t_body, r.t_chat_params, r.t_msgs, r.t_apply, r.t_tok_cache, r.t_tok_full,
                    r.t_task_params, r.t_prompt_copy, r.sha_tokens.c_str());
        }
    }

    // tabella di identita' (ultima passata)
    {
        char path[512];
        snprintf(path, sizeof(path), "%s/identity.tsv", out_dir);
        std::ofstream o(path);
        o << "round\tn_msgs\tbody_bytes\tprompt_bytes\tn_tokens\tsha_prompt\tsha_tokens\n";
        for (size_t i = 0; i < last.size(); ++i) {
            o << i << '\t' << last[i].n_msgs << '\t' << last[i].body_bytes << '\t' << last[i].prompt_bytes
              << '\t' << last[i].n_tokens << '\t' << last[i].sha_prompt << '\t' << last[i].sha_tokens << '\n';
        }
        fprintf(stderr, "identity -> %s\n", path);
    }

    // ============ 2) census tokenize-domain sul corpus pi reale ============
    if (census_corpus != nullptr) {
        std::ifstream in(census_corpus, std::ios::binary);
        std::string prompt((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        if (prompt.size() > 4096) {
            const auto specials = [&]() {
                std::vector<std::pair<llama_token, std::string>> out;
                const int32_t n = llama_vocab_n_tokens(vocab);
                for (llama_token tok = 0; tok < n; ++tok) {
                    const auto attr = llama_vocab_get_attr(vocab, tok);
                    if (attr & (LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_USER_DEFINED | LLAMA_TOKEN_ATTR_UNKNOWN)) {
                        const char * text = llama_vocab_get_text(vocab, tok);
                        if (text != nullptr && text[0] != '\0') {
                            out.emplace_back(tok, std::string(text));
                        }
                    }
                }
                std::sort(out.begin(), out.end(), [](auto & a, auto & b) { return a.second.size() > b.second.size(); });
                return out;
            }();

            std::vector<size_t> cut_points;
            for (const auto & [tok, text] : specials) {
                size_t pos = 0;
                while ((pos = prompt.find(text, pos)) != std::string::npos) {
                    cut_points.push_back(pos);
                    pos += text.size();
                }
            }
            std::sort(cut_points.begin(), cut_points.end());
            cut_points.erase(std::unique(cut_points.begin(), cut_points.end()), cut_points.end());
            std::vector<size_t> rounds;
            for (int i = 1; i <= 16; ++i) {
                rounds.push_back(cut_points[(cut_points.size() - 1) * i / 16]);
            }

            // tokenize piena vs prefix-cache (delta) sul corpus: S3-domain @scala pi
            double t_full = 0, t_cached = 0;
            size_t n_tok_last = 0;
            {
                std::vector<std::string> texts;
                for (size_t c : rounds) {
                    texts.push_back(prompt.substr(0, c));
                }
                (void) common_tokenize(vocab, texts.back(), true, true); // warmup
                auto t0 = t_now();
                for (auto & t : texts) {
                    n_tok_last = common_tokenize(vocab, t, true, true).size();
                }
                t_full = ms_since(t0);
                server_token_prefix_cache cache(vocab);
                t0 = t_now();
                for (auto & t : texts) {
                    n_tok_last = cache.tokenize(t, true, true).size();
                }
                t_cached = ms_since(t0);
            }
            fprintf(stderr, "CENSUS-PI corpus=%zu B (%zu token): tokenize piena=%.1f ms/16round, prefix-cache=%.1f ms/16round\n",
                    prompt.size(), n_tok_last, t_full, t_cached);

            // per-delta: round b dopo lo stato di round a (= round pi di grown
            // conversation) -> costo marginale steady-state del solo splice.
            // le cache in stato-a sono preparate FUORI dal timer
            {
                const std::string a = prompt.substr(0, rounds[rounds.size() - 2]);
                const std::string b = prompt.substr(0, rounds[rounds.size() - 1]);
                const int rep = 20;
                std::vector<std::unique_ptr<server_token_prefix_cache>> prepared;
                for (int i = 0; i < rep; ++i) {
                    auto c = std::make_unique<server_token_prefix_cache>(vocab);
                    (void) c->tokenize(a, true, true);
                    prepared.push_back(std::move(c));
                }
                auto t0 = t_now();
                for (auto & c : prepared) {
                    (void) c->tokenize(b, true, true);
                }
                fprintf(stderr, "CENSUS-PI delta-round (b=%zu B dopo a=%zu B, delta %zu B): %.2f ms/splice medio\n",
                        b.size(), a.size(), b.size() - a.size(), ms_since(t0) / rep);
            }
        }
    }

    // ============ 3) census e2e sintetico alla scala pi (~412 KB) ============
    if (!skip_synth) {
        const std::vector<std::string> contents = {
            "Analizza il file src/llama-vocab.cpp e riduci le passate ridondanti del tokenizer BPE.",
            "月亮在白莲花般的云朵里穿行，晚风吹来一阵阵快乐的歌声，我们坐在高高的谷堆旁边。",
            "```cpp\nauto words = unicode_regex_split(text, regex);\nfor (auto & w : words) { merge(w); }\n```",
            "{\"name\": \"edit_file\", \"arguments\": {\"path\": \"server-common.cpp\", \"old\": \"x\", \"new\": \"y\"}}",
            "<think>\nthe prefix is unchanged, only the tail grows; reuse the tokens\n</think>",
        };
        std::vector<json> round_msgs;
        json msgs = json::array();
        for (size_t r = 0; r < 16; ++r) {
            for (size_t j = 0; j < 25; ++j) {
                json m;
                m["role"] = ((r * 25 + j) % 3 == 0) ? "user" : (((r * 25 + j) % 3 == 1) ? "assistant" : "tool");
                std::string content = contents[(r + j) % contents.size()];
                for (int k = 0; k < 20; ++k) {
                    content += "\nline " + std::to_string(k) + ": " + contents[(r + j + k) % contents.size()];
                }
                m["content"] = content;
                if (m["role"] == "assistant") {
                    m["reasoning_content"] = "reasoning round " + std::to_string(r);
                }
                msgs.push_back(m);
            }
            round_msgs.push_back(msgs);
        }
        // una passata warm (cache parser W6-6) poi la passata di misura
        {
            server_token_prefix_cache warm(vocab);
            (void) run_round(json({{ "messages", round_msgs.back() }, { "stream", false }}).dump(), vocab, opt, warm, params_base, nullptr);
        }
        server_token_prefix_cache cache(vocab);
        double tot_body = 0, tot_chat_params = 0, tot_msgs = 0, tot_apply = 0, tot_tok = 0, tot_task = 0, tot_full = 0;
        round_result last_synth;
        for (size_t r = 0; r < round_msgs.size(); ++r) {
            std::string body_str = json({{ "messages", round_msgs[r] }, { "stream", false }}).dump();
            auto res = run_round(body_str, vocab, opt, cache, params_base, nullptr);
            tot_body += res.t_body;
            tot_chat_params += res.t_chat_params;
            tot_msgs += res.t_msgs;
            tot_apply += res.t_apply;
            tot_tok += res.t_tok_cache;
            tot_task += res.t_task_params;
            tot_full += res.t_tok_full;
            last_synth = res;
            fprintf(stderr,
                    "SYNTH r=%02zu body=%7zu prompt=%7zu tokens=%6zu | body=%.2f chat_params=%.2f (msgs=%.2f apply=%.2f) tok_cache=%.2f tok_full=%.2f task=%.2f\n",
                    r, res.body_bytes, res.prompt_bytes, res.n_tokens,
                    res.t_body, res.t_chat_params, res.t_msgs, res.t_apply, res.t_tok_cache, res.t_tok_full, res.t_task_params);
        }
        fprintf(stderr,
                "SYNTH-TOT (16 round, ultimo prompt %zu B): body=%.1f chat_params=%.1f (msgs=%.1f apply=%.1f) tok_cache=%.1f tok_full=%.1f task=%.1f | e2e ultimo round=%.2f ms\n",
                last_synth.prompt_bytes, tot_body, tot_chat_params, tot_msgs, tot_apply, tot_tok, tot_full, tot_task,
                last_synth.t_body + last_synth.t_chat_params + last_synth.t_tok_cache + last_synth.t_task_params);
    }

    llama_model_free(model);
    fprintf(stderr, "%s: ALL OK\n", argv[0]);
    return 0;
}
