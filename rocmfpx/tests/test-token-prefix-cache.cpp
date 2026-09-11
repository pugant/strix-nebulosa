// W6-6 / A4-S1 — gate bit-identita' del token-prefix cache conversazione.
//
// Il cache (server_token_prefix_cache) deve produrre, per prompt conversazionali
// che crescono alla fine (client stateless tipo pi), un token stream IDENTICO
// bit-per-bit alla tokenizzazione piena (common_tokenize), e deve invalidare
// correttamente su edit mid-prefix / cambio flag / testo non correlato.
//
// usage: test-token-prefix-cache <vocab.gguf> [pi-prompt.txt]
//   - senza [pi-prompt.txt] gira solo il corpus sintetico (modalita' CI)
//   - con [pi-prompt.txt] aggiunge il corpus reale pi (412 KB, ~131k token) e
//     stampa la misura ms prima/dopo (tokenizzazione piena vs cache warm)

#include "server-common.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#undef NDEBUG
#include <cassert>

static bool expect_bypass_vocab(const llama_vocab * vocab) {
    return llama_vocab_type(vocab) != LLAMA_VOCAB_TYPE_BPE;
}

using namespace std::chrono;

static steady_clock::time_point t_now() {
    return steady_clock::now();
}

static double ms_since(steady_clock::time_point t0) {
    return duration<double, std::milli>(steady_clock::now() - t0).count();
}

// tokenizzazione piena di riferimento (path server: add_special+parse_special)
static std::vector<llama_token> ref_tokenize(const llama_vocab * vocab, const std::string & text, bool add_special = true) {
    return common_tokenize(vocab, text, add_special, true);
}

static bool tokens_equal(const std::vector<llama_token> & a, const std::vector<llama_token> & b) {
    return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
}

// raccoglie i special token del vocab (attr CONTROL|USER_DEFINED|UNKNOWN), testi non vuoti
static std::vector<std::pair<llama_token, std::string>> vocab_specials(const llama_vocab * vocab) {
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
    // i piu' lunghi prima (stesso criterio di ordinamento del partitioner)
    std::sort(out.begin(), out.end(), [](const auto & a, const auto & b) { return a.second.size() > b.second.size(); });
    return out;
}

// pool di contenuti multilingue / casi limite per i messaggi sintetici
static const std::vector<std::string> k_contents = {
    // prosa EN
    "The quick brown fox jumps over the lazy dog while the tokenizer caches its prefix across rounds.",
    // CJK
    "月亮在白莲花般的云朵里穿行，晚风吹来一阵阵快乐的歌声，我们坐在高高的谷堆旁边，听妈妈讲那过去的事情。",
    "日本語のテキストです。トークナイザーは接頭辞を再利用します。必要なのは末尾の差分だけです。",
    // italiano accentato
    "Perché la cache funziona, basta che il prefisso conversazione cresca alla fine: né più, né meno.",
    // codice
    "```cpp\nfor (size_t i = 0; i < n_words; ++i) {\n    auto & word = words[i];\n    bpe_merge(word, ranks);\n}\n```",
    // tool-call-like JSON
    "{\"name\": \"read_file\", \"arguments\": {\"path\": \"/tmp/a4/wave6.go\", \"offset\": 1024}}",
    // thinking block
    "<think>\nLet me analyze the request: the conversation prefix is unchanged, only the tail grows.\n</think>\n",
    // emoji + misto
    "mix 🚀🧠✅ with ASCII and 数字 123 and symbols @#$%^&*()",
    // spazi e ritorni
    "line1\n\n\nline3   with   runs\t\tof\twhitespace and trailing spaces   ",
    // numeri
    "3.141592653589793 2718281828459045 1e-9 0xDEADBEEF 0b1010",
    // quasi-special (prefisso di special ma NON completo: deve restare nel delta)
    "incomplete marker <|im_ and <|end and <|tool",
    // special dentro una parola
    "abc",
    // stringa vuota e punteggiatura
    "...",
    "",
};

// costruisce un messaggio in stile ChatML usando i special REALI del vocab
static std::string synth_message(const std::string & open, const std::string & close, size_t i, bool with_close = true) {
    const auto & content = k_contents[i % k_contents.size()];
    std::string role = (i % 3 == 0) ? "user" : (i % 3 == 1 ? "assistant" : "tool");
    std::string msg = open + role + "\n" + content;
    if (with_close) {
        msg += close + "\n";
    }
    // talvolta incolla un special dentro il contenuto (sub-word)
    if (i % 7 == 3) {
        msg += "tail" + close + "after\n";
    }
    return msg;
}

static void check_round(const llama_vocab * vocab, server_token_prefix_cache & cache, const std::string & text,
                        const char * scenario, size_t round, bool add_special = true) {
    const auto expected = ref_tokenize(vocab, text, add_special);
    const auto got      = cache.tokenize(text, add_special, true);
    if (!tokens_equal(expected, got)) {
        fprintf(stderr, "FAIL %s round %zu: token stream mismatch (expected %zu tokens, got %zu)\n",
                scenario, round, expected.size(), got.size());
        for (size_t i = 0; i < std::min(expected.size(), got.size()); ++i) {
            if (expected[i] != got[i]) {
                fprintf(stderr, "  first diff at token %zu: expected %d, got %d\n", i, expected[i], got[i]);
                break;
            }
        }
        assert(false && "bit-identity gate");
    }
}

static void run_synthetic(const llama_vocab * vocab, const std::vector<std::pair<llama_token, std::string>> & specials) {
    assert(!specials.empty() && "il fixture deve avere >= 1 special token per esercitare lo splice");
    const std::string open  = specials[0].second;
    const std::string close = specials.size() > 1 ? specials[1].second : specials[0].second;
    // vocabs non-BPM (SPM/WPM/UGM) tengono stato cross-frammento: la cache deve
    // essere DISABILITATA e servire sempre la tokenizzazione piena (bypass)
    const bool expect_bypass = llama_vocab_type(vocab) != LLAMA_VOCAB_TYPE_BPE;

    // 1) conversazione crescente (pattern pi) — 40 round
    {
        server_token_prefix_cache cache(vocab);
        std::string conv;
        for (size_t r = 0; r < 40; ++r) {
            conv += synth_message(open, close, r);
            // ogni 5 round aggiungi anche un blocco thinking con special adiacenti
            if (r % 5 == 2) {
                conv += open + close + open; // special adiacenti senza testo in mezzo
                conv += "assistant\ncompact " + std::to_string(r) + close + "\n";
            }
            check_round(vocab, cache, conv, "growing", r);
        }
        const auto stats = cache.get_stats();
        if (expect_bypass) {
            assert(!stats.enabled && "vocab non-BPE: la cache deve restare disabilitata");
            assert(stats.n_reuse == 0 && stats.n_reuse_full_hit == 0);
        } else {
            assert(stats.n_reuse > 0 && "lo splice deve essere stato esercitato (growing)");
            assert(stats.n_selfcheck_fail == 0);
            assert(stats.enabled);
        }
        fprintf(stderr, "growing 40 rounds: lookups=%llu reuse=%llu full=%llu bytes_reused=%llu\n",
                (unsigned long long) stats.n_lookups, (unsigned long long) stats.n_reuse,
                (unsigned long long) stats.n_full, (unsigned long long) stats.bytes_reused);
    }

    // 2) edit mid-prefix: modifica un messaggio gia' inviato e reinvia
    {
        server_token_prefix_cache cache(vocab);
        std::string conv;
        std::vector<std::string> msgs;
        for (size_t r = 0; r < 12; ++r) {
            msgs.push_back(synth_message(open, close, r));
            conv += msgs.back();
            check_round(vocab, cache, conv, "edit-build", r);
        }
        for (size_t m = 0; m < msgs.size(); m += 3) {
            std::string edited;
            for (size_t j = 0; j < msgs.size(); ++j) {
                edited += (j == m) ? (open + "user\nEDITED MID PREFIX " + std::to_string(j) + close + "\n") : msgs[j];
            }
            edited += synth_message(open, close, msgs.size());
            check_round(vocab, cache, edited, "edit-mid", m);
        }
        const auto stats = cache.get_stats();
        assert(stats.n_selfcheck_fail == 0);
    }

    // 3) full-hit (regenerate: stesso testo identico)
    {
        server_token_prefix_cache cache(vocab);
        std::string conv;
        for (size_t r = 0; r < 8; ++r) {
            conv += synth_message(open, close, r);
        }
        check_round(vocab, cache, conv, "fullhit-1", 0);
        check_round(vocab, cache, conv, "fullhit-2", 1);
        check_round(vocab, cache, conv, "fullhit-3", 2);
        const auto stats = cache.get_stats();
        if (!expect_bypass) {
            assert(stats.n_reuse_full_hit > 0);
        }
    }

    // 4) testo non correlato (nessun prefisso condiviso) + alternanza di due conversazioni
    {
        server_token_prefix_cache cache(vocab);
        std::string conv_a, conv_b;
        for (size_t r = 0; r < 10; ++r) {
            conv_a += synth_message(open, close, r);
            conv_b += open + "user\nConversazione B round " + std::to_string(r) + " 雪の降る町を歩く。\n" + close + "\n";
            check_round(vocab, cache, conv_a, "interleave-a", r);
            check_round(vocab, cache, conv_b, "interleave-b", r);
        }
        // testo completamente diverso
        check_round(vocab, cache, "unrelated short prompt with no shared prefix at all", "unrelated", 0);
        const auto stats = cache.get_stats();
        assert(stats.n_selfcheck_fail == 0);
    }

    // 5) flag add_special=false (entry distinte per flag)
    {
        server_token_prefix_cache cache(vocab);
        std::string conv;
        for (size_t r = 0; r < 6; ++r) {
            conv += synth_message(open, close, r);
            check_round(vocab, cache, conv, "nospecial-first", r, false);
            check_round(vocab, cache, conv, "nospecial-true", r, true);
        }
        const auto stats = cache.get_stats();
        assert(stats.n_selfcheck_fail == 0);
    }

    // 6) prompt senza alcun special (prosa pura): nessun cut disponibile, deve comunque restare identico
    {
        server_token_prefix_cache cache(vocab);
        std::string prose;
        for (size_t r = 0; r < 10; ++r) {
            prose += "Round " + std::to_string(r) + ": plain prose without any special token, just words and spaces. ";
            prose += k_contents[r % k_contents.size()];
            check_round(vocab, cache, prose, "prose-only", r);
        }
    }

    // 7) concorrenza: piu' thread sulla stessa cache, risultati identici al riferimento
    {
        server_token_prefix_cache cache(vocab);
        std::string conv;
        for (size_t r = 0; r < 24; ++r) {
            conv += synth_message(open, close, r);
        }
        std::vector<std::thread> workers;
        std::vector<int> failures(4, 0);
        for (int w = 0; w < 4; ++w) {
            workers.emplace_back([&, w]() {
                for (size_t r = 0; r < 6; ++r) {
                    // ogni worker taglia la conversazione a lunghezze diverse (prefissi condivisi)
                    std::string partial = conv.substr(0, conv.size() / 2 + (w * 137 + r * 271) % (conv.size() / 2));
                    if (!tokens_equal(ref_tokenize(vocab, partial), cache.tokenize(partial, true, true))) {
                        failures[w]++;
                    }
                }
            });
        }
        for (auto & t : workers) {
            t.join();
        }
        for (int f : failures) {
            assert(f == 0 && "concurrency mismatch");
        }
        const auto stats = cache.get_stats();
        assert(stats.n_selfcheck_fail == 0);
    }
}

// corpus reale pi: taglia il prompt ai confini dei special (round crescenti) e misura
static void run_pi_corpus(const llama_vocab * vocab, const std::string & prompt,
                          const std::vector<std::pair<llama_token, std::string>> & specials) {
    fprintf(stderr, "pi corpus: %zu bytes\n", prompt.size());

    // round = prefissi crescenti del prompt, tagliati all'inizio di un special
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
    assert(!cut_points.empty());

    // ~16 round sparsi sull'intervallo (ogni round e' un prefisso del successivo)
    std::vector<size_t> rounds;
    for (int i = 1; i <= 16; ++i) {
        size_t idx = (cut_points.size() - 1) * i / 16;
        rounds.push_back(cut_points[idx]);
    }

    // gate bit-identita' sul corpus reale
    {
        server_token_prefix_cache cache(vocab);
        for (size_t r = 0; r < rounds.size(); ++r) {
            std::string partial = prompt.substr(0, rounds[r]);
            check_round(vocab, cache, partial, "pi-grow", r);
        }
        const auto stats = cache.get_stats();
        assert(stats.n_reuse > 0);
        assert(stats.n_selfcheck_fail == 0);
        fprintf(stderr, "pi bit-identity: 16/16 round OK (reuse=%llu full=%llu)\n",
                (unsigned long long) stats.n_reuse, (unsigned long long) stats.n_full);
    }

    // misura: piena vs cache warm (stessa sequenza di round, cache nuova)
    {
        std::vector<std::string> texts;
        for (size_t r = 0; r < rounds.size(); ++r) {
            texts.push_back(prompt.substr(0, rounds[r]));
        }
        const std::string & last = texts.back();

        // warmup
        (void) ref_tokenize(vocab, last);

        // prima: tokenizzazione piena per ogni round
        double t_full = 0.0;
        {
            auto t0 = t_now();
            size_t n_tok = 0;
            for (const auto & t : texts) {
                n_tok += ref_tokenize(vocab, t).size();
            }
            t_full = ms_since(t0);
            fprintf(stderr, "pi measure FULL: %.1f ms over %zu rounds (%zu tokens on last round)\n",
                    t_full, texts.size(), ref_tokenize(vocab, texts.back()).size());
        }

        // dopo: cache warm (round 0 miss, tutti gli altri splice)
        double t_cached = 0.0;
        {
            server_token_prefix_cache cache(vocab);
            auto t0 = t_now();
            size_t n_tok = 0;
            for (const auto & t : texts) {
                n_tok += cache.tokenize(t, true, true).size();
            }
            t_cached = ms_since(t0);
            const auto stats = cache.get_stats();
            fprintf(stderr, "pi measure CACHED: %.1f ms over %zu rounds (reuse=%llu full=%llu bytes_reused=%llu)\n",
                    t_cached, texts.size(), (unsigned long long) stats.n_reuse, (unsigned long long) stats.n_full,
                    (unsigned long long) stats.bytes_reused);
        }
        fprintf(stderr, "pi measure: full=%.1f ms cached=%.1f ms speedup=%.1fx\n",
                t_full, t_cached, t_full / t_cached);
    }
}

// misura end-to-end del path host per-request (W6-6 completo): parse messaggi ->
// apply template (--jinja) -> tokenize, conversazione crescente alla scala pi.
// cold  = template fresco + tokenizzazione piena (comportamento pre-card)
// warm  = template con cache parser/gen-prompt + token-prefix cache (post-card)
static void run_e2e_measure(const llama_vocab * vocab) {
    const char * tmpl_path = "models/templates/Qwen-ChatML-no-think.jinja";
    std::ifstream tin(tmpl_path, std::ios::binary);
    if (!tin.good()) {
        tmpl_path = "../models/templates/Qwen-ChatML-no-think.jinja";
        tin.open(tmpl_path, std::ios::binary);
    }
    if (!tin.good()) {
        fprintf(stderr, "e2e: template Qwen-ChatML-no-think.jinja non trovato, skip\n");
        return;
    }
    const std::string tmpl_src((std::istreambuf_iterator<char>(tin)), std::istreambuf_iterator<char>());

    const std::vector<std::string> contents = {
        "Analizza il file src/llama-vocab.cpp e riduci le passate ridondanti del tokenizer BPE.",
        "月亮在白莲花般的云朵里穿行，晚风吹来一阵阵快乐的歌声，我们坐在高高的谷堆旁边。",
        "```cpp\nauto words = unicode_regex_split(text, regex);\nfor (auto & w : words) { merge(w); }\n```",
        "{\"name\": \"edit_file\", \"arguments\": {\"path\": \"server-common.cpp\", \"old\": \"x\", \"new\": \"y\"}}",
        "<think>\nthe prefix is unchanged, only the tail grows; reuse the tokens\n</think>",
    };

    // conversazione crescente: 16 round, ~25 KB per round -> ~410 KB finale
    std::vector<std::vector<common_chat_msg>> rounds;
    std::vector<common_chat_msg>              msgs;
    for (size_t r = 0; r < 16; ++r) {
        for (size_t j = 0; j < 25; ++j) {
            common_chat_msg m;
            m.role    = ((r * 25 + j) % 3 == 0) ? "user" : (((r * 25 + j) % 3 == 1) ? "assistant" : "tool");
            m.content = contents[(r + j) % contents.size()];
            for (int k = 0; k < 20; ++k) {
                m.content += "\nline " + std::to_string(k) + ": " + contents[(r + j + k) % contents.size()];
            }
            if (m.role == "assistant") {
                m.reasoning_content = "reasoning round " + std::to_string(r);
            }
            msgs.push_back(std::move(m));
        }
        rounds.push_back(msgs);
    }

    auto build_inputs = [](const std::vector<common_chat_msg> & m) {
        common_chat_templates_inputs inputs;
        inputs.messages              = m;
        inputs.use_jinja             = true;
        inputs.add_generation_prompt = true;
        inputs.reasoning_format      = COMMON_REASONING_FORMAT_DEEPSEEK;
        return inputs;
    };

    // warmup
    {
        common_chat_templates_ptr t = common_chat_templates_init(nullptr, tmpl_src);
        (void) common_chat_templates_apply(t.get(), build_inputs(rounds.front()));
        (void) common_tokenize(vocab, common_chat_templates_apply(t.get(), build_inputs(rounds.back())).prompt, true, true);
    }

    double t_cold = 0.0;
    size_t last_tokens = 0, last_bytes = 0;
    {
        std::vector<double> per_round;
        for (const auto & m : rounds) {
            auto t0 = t_now();
            common_chat_templates_ptr t = common_chat_templates_init(nullptr, tmpl_src);
            const auto res = common_chat_templates_apply(t.get(), build_inputs(m));
            const auto toks = common_tokenize(vocab, res.prompt, true, true);
            per_round.push_back(ms_since(t0));
            last_tokens = toks.size();
            last_bytes  = res.prompt.size();
        }
        t_cold = std::accumulate(per_round.begin(), per_round.end(), 0.0);
        fprintf(stderr, "e2e cold per-round ms:");
        for (double v : per_round) fprintf(stderr, " %.1f", v);
        fprintf(stderr, "\n");
    }

    double t_warm = 0.0;
    {
        common_chat_templates_ptr  t = common_chat_templates_init(nullptr, tmpl_src);
        server_token_prefix_cache  cache(vocab);
        std::vector<double> per_round;
        for (const auto & m : rounds) {
            auto t0 = t_now();
            const auto res  = common_chat_templates_apply(t.get(), build_inputs(m));
            const auto toks = cache.tokenize(res.prompt, true, true);
            per_round.push_back(ms_since(t0));
            last_tokens     = toks.size();
        }
        t_warm = std::accumulate(per_round.begin(), per_round.end(), 0.0);
        fprintf(stderr, "e2e warm per-round ms:");
        for (double v : per_round) fprintf(stderr, " %.1f", v);
        fprintf(stderr, "\n");
        const auto stats = cache.get_stats();
        fprintf(stderr, "e2e warm: reuse=%llu full=%llu selfcheck_ok=%llu fail=%llu\n",
                (unsigned long long) stats.n_reuse, (unsigned long long) stats.n_full,
                (unsigned long long) stats.n_selfcheck_ok, (unsigned long long) stats.n_selfcheck_fail);
    }

    fprintf(stderr, "e2e (template apply + tokenize): %zu rounds, last prompt %zu B / %zu tokens\n",
            rounds.size(), last_bytes, last_tokens);
    fprintf(stderr, "e2e COLD: %.1f ms   WARM: %.1f ms   speedup %.1fx\n", t_cold, t_warm, t_warm > 0 ? t_cold / t_warm : 0.0);
}
static void run_big_synthetic_measure(const llama_vocab * vocab, const std::vector<std::pair<llama_token, std::string>> & specials) {
    const std::string open  = specials[0].second;
    const std::string close = specials.size() > 1 ? specials[1].second : specials[0].second;

    // costruisce ~500 KB di conversazione
    std::vector<std::string> texts;
    std::string conv;
    for (size_t r = 0; conv.size() < 500 * 1024; ++r) {
        conv += synth_message(open, close, r);
        if (r % 10 == 0) {
            texts.push_back(conv);
        }
    }
    texts.push_back(conv);
    // tieni ~24 round
    std::vector<std::string> keep;
    for (size_t i = 0; i < texts.size(); ++i) {
        if (i % std::max<size_t>(1, texts.size() / 24) == 0 || i + 1 == texts.size()) {
            keep.push_back(texts[i]);
        }
    }
    texts = keep;

    (void) ref_tokenize(vocab, texts.back());
    const size_t n_tokens_last = ref_tokenize(vocab, texts.back()).size();
    fprintf(stderr, "synthetic big: %zu bytes, last round %zu tokens\n", texts.back().size(), n_tokens_last);

    double t_full = 0.0;
    {
        auto t0 = t_now();
        for (const auto & t : texts) {
            (void) ref_tokenize(vocab, t);
        }
        t_full = ms_since(t0);
        fprintf(stderr, "synthetic FULL: %.1f ms over %zu rounds\n", t_full, texts.size());
    }
    double t_cached = 0.0;
    {
        server_token_prefix_cache cache(vocab);
        auto t0 = t_now();
        for (const auto & t : texts) {
            (void) cache.tokenize(t, true, true);
        }
        t_cached = ms_since(t0);
        const auto stats = cache.get_stats();
        fprintf(stderr, "synthetic CACHED: %.1f ms over %zu rounds (reuse=%llu full=%llu)\n", t_cached, texts.size(),
                (unsigned long long) stats.n_reuse, (unsigned long long) stats.n_full);
    }
    fprintf(stderr, "synthetic: full=%.1f ms cached=%.1f ms speedup=%.1fx\n", t_full, t_cached, t_full / t_cached);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <vocab.gguf> [pi-prompt.txt]\n", argv[0]);
        return 2;
    }

    llama_model_params mp = llama_model_default_params();
    mp.vocab_only   = true;  // nessun tensore: zero allocazioni device (gate ZERO GPU)
    mp.use_mmap     = true;
    mp.check_tensors = false;

    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (model == nullptr) {
        fprintf(stderr, "failed to load vocab from %s\n", argv[1]);
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const auto specials = vocab_specials(vocab);
    fprintf(stderr, "vocab type=%d tokens=%d specials=%zu (longest: '%s')\n",
            (int) llama_vocab_type(vocab), llama_vocab_n_tokens(vocab), specials.size(),
            specials.empty() ? "-" : specials[0].second.c_str());

    run_synthetic(vocab, specials);

    if (argc >= 3) {
        std::ifstream in(argv[2], std::ios::binary);
        std::string prompt((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        if (prompt.size() > 4096) {
            run_pi_corpus(vocab, prompt, specials);
        } else {
            fprintf(stderr, "pi prompt too short (%zu bytes), skipping pi corpus\n", prompt.size());
        }
    } else {
        run_big_synthetic_measure(vocab, specials);
    }

    if (!expect_bypass_vocab(vocab)) {
        run_e2e_measure(vocab);
    }

    llama_model_free(model);

    fprintf(stderr, "%s: ALL OK\n", argv[0]);
    return 0;
}
