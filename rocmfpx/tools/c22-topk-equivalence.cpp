// W3-C22 TOPK-QSA-GPU: GPU-vs-CPU equivalence test for ggml_top_k with
// k > 1024 (k=2051, the qwen4exp QSA indexer selection at 32k context).
//
// Standalone tool, intentionally NOT wired into any default build target.
// Compile it inside the usual build container, e.g.:
//
//   g++ -std=c++17 -O2 -I ggml/include tools/c22-topk-equivalence.cpp \
//       -o build/c22-topk-equivalence \
//       -L <build>/bin -lggml -lggml-base -lggml-cpu
//
// (add -lggml-vulkan or set GGML_BACKEND_PATH if the Vulkan backend is built
// as a shared module; with a static build it is already registered).
//
// PASS criteria (class D, as declared in the patch):
//   - the top-k VALUE sequences of GPU and CPU reference match within 1e-6;
//   - the index selections differ at most on exact ties: per distinct value
//     the selected multiplicities must match, so any index difference is a
//     swap inside a tie group of the input;
//   - the GPU output is bit-identical across two identical runs (stability).
//
// Cases: [1x32768] k=2051 (the QSA shape), [2x32768] k=2051 (multi row),
// [1x4096] k=2051 (multi chunk, short row), plus a tie-stress case on
// [2x32768] with heavy value quantization.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

struct test_case_t {
    const char * name;
    int nrows;
    int ncols;
    int k;
    bool tie_stress;
};

static uint32_t f2u_key(float v) {
    // group key: bit pattern, with -0.0 normalized to +0.0 (comparisons treat
    // them as equal, so selections may swap them)
    if (v == 0.0f) {
        v = 0.0f;
    }
    uint32_t u;
    std::memcpy(&u, &v, sizeof(u));
    return u;
}

struct selection_t {
    std::vector<float>    values;  // sorted descending, size k
    std::vector<int32_t>  indices; // as produced by the reference
    std::vector<uint32_t> keys;    // f2u_key(values[i])
};

static selection_t reference_top_k(const std::vector<float> & x, int k) {
    selection_t sel;
    const int ncols = (int)x.size();
    sel.values.resize(k);
    sel.indices.resize(k);
    sel.keys.resize(k);

    std::vector<int32_t> idx(ncols);
    for (int i = 0; i < ncols; ++i) {
        idx[i] = i;
    }
    const float * data = x.data();
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [data](int32_t a, int32_t b) { return data[a] > data[b]; });
    for (int i = 0; i < k; ++i) {
        sel.indices[i] = idx[i];
        sel.values[i]  = data[idx[i]];
        sel.keys[i]    = f2u_key(data[idx[i]]);
    }
    return sel;
}

static bool check_row(const std::vector<float> & x, const selection_t & cpu,
                      const int32_t * gpu_idx, int k, std::string & err) {
    const int ncols = (int)x.size();

    // indices in range and distinct
    {
        std::vector<int32_t> s(gpu_idx, gpu_idx + k);
        for (int i = 0; i < k; ++i) {
            if (s[i] < 0 || s[i] >= ncols) {
                err = "index out of range: " + std::to_string(s[i]);
                return false;
            }
        }
        std::sort(s.begin(), s.end());
        for (int i = 1; i < k; ++i) {
            if (s[i] == s[i - 1]) {
                err = "duplicated index " + std::to_string(s[i]);
                return false;
            }
        }
    }

    // 1) value sequences (GPU values gathered by index, sorted descending)
    std::vector<float> gpu_sorted(k);
    for (int i = 0; i < k; ++i) {
        gpu_sorted[i] = x[gpu_idx[i]];
    }
    std::sort(gpu_sorted.begin(), gpu_sorted.end(), std::greater<float>());
    for (int i = 0; i < k; ++i) {
        if (std::fabs(gpu_sorted[i] - cpu.values[i]) > 1e-6f) {
            err = "value mismatch at " + std::to_string(i) + ": gpu " +
                  std::to_string(gpu_sorted[i]) + " cpu " + std::to_string(cpu.values[i]);
            return false;
        }
    }

    // 2) per-value multiplicities (=> index diffs only inside exact ties)
    std::vector<std::pair<uint32_t, int>> gpu_groups, cpu_groups;
    {
        std::vector<uint32_t> gk(k);
        for (int i = 0; i < k; ++i) {
            gk[i] = f2u_key(x[gpu_idx[i]]);
        }
        std::sort(gk.begin(), gk.end());
        for (int i = 0; i < k;) {
            int j = i;
            while (j < k && gk[j] == gk[i]) {
                ++j;
            }
            gpu_groups.push_back({gk[i], j - i});
            i = j;
        }
    }
    {
        std::vector<uint32_t> ck = cpu.keys;
        std::sort(ck.begin(), ck.end());
        for (int i = 0; i < k;) {
            int j = i;
            while (j < k && ck[j] == ck[i]) {
                ++j;
            }
            cpu_groups.push_back({ck[i], j - i});
            i = j;
        }
    }
    if (gpu_groups.size() != cpu_groups.size()) {
        err = "distinct value count differs: gpu " + std::to_string(gpu_groups.size()) +
              " cpu " + std::to_string(cpu_groups.size());
        return false;
    }
    for (size_t i = 0; i < gpu_groups.size(); ++i) {
        if (gpu_groups[i].first != cpu_groups[i].first ||
            gpu_groups[i].second != cpu_groups[i].second) {
            err = "value group mismatch at group " + std::to_string(i) +
                  " (key gpu 0x" + std::to_string(gpu_groups[i].first) +
                  " x" + std::to_string(gpu_groups[i].second) +
                  ", cpu 0x" + std::to_string(cpu_groups[i].first) +
                  " x" + std::to_string(cpu_groups[i].second) + ")";
            return false;
        }
    }

    return true;
}

static bool run_case(ggml_backend_t backend, const test_case_t & tc, std::mt19937 & rng, std::string & err) {
    const int nrows = tc.nrows;
    const int ncols = tc.ncols;
    const int k     = tc.k;

    // host input
    std::vector<float> x((size_t)nrows * ncols);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto & v : x) {
        v = dist(rng);
    }
    if (tc.tie_stress) {
        // quantize to a coarse grid -> many exact ties around the cut
        for (auto & v : x) {
            v = std::round(v * 8.0f) / 8.0f;
        }
    }

    // build the graph: a [ncols, nrows] f32 -> top_k -> [k, nrows] i32
    ggml_init_params ip = {};
    ip.mem_size = ggml_tensor_overhead() * 4 + ggml_graph_overhead();
    ip.no_alloc = true;
    ggml_context_ptr ctx(ggml_init(ip));
    GGML_ASSERT(ctx != nullptr);

    ggml_tensor * a   = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, ncols, nrows);
    ggml_tensor * out = ggml_top_k(ctx.get(), a, k);

    // the gate must accept k > 1024 on this device
    if (!ggml_backend_supports_op(backend, out)) {
        err = "backend does not support TOP_K with k=" + std::to_string(k) +
              " (gate rejected; is this the patched VK backend?)";
        return false;
    }

    ggml_cgraph * gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    if (buf == nullptr) {
        err = "alloc failed";
        return false;
    }

    ggml_backend_tensor_set(a, x.data(), 0, ggml_nbytes(a));

    std::vector<int32_t> out_run1((size_t)k * nrows);
    std::vector<int32_t> out_run2((size_t)k * nrows);
    for (int run = 0; run < 2; ++run) {
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            err = "graph_compute failed (run " + std::to_string(run) + ")";
            return false;
        }
        ggml_backend_synchronize(backend);
        std::vector<int32_t> & dst = run == 0 ? out_run1 : out_run2;
        ggml_backend_tensor_get(out, dst.data(), 0, ggml_nbytes(out));
    }

    // 3) run-to-run stability
    if (std::memcmp(out_run1.data(), out_run2.data(), out_run1.size() * sizeof(int32_t)) != 0) {
        err = "GPU output differs between identical runs (not stable)";
        return false;
    }

    // reference + comparison, row by row
    for (int r = 0; r < nrows; ++r) {
        std::vector<float> row(x.begin() + (size_t)r * ncols, x.begin() + (size_t)(r + 1) * ncols);
        selection_t cpu = reference_top_k(row, k);
        std::string row_err;
        if (!check_row(row, cpu, out_run1.data() + (size_t)r * k, k, row_err)) {
            err = "row " + std::to_string(r) + ": " + row_err;
            return false;
        }
    }

    return true;
}

int main(int argc, char ** argv) {
    GGML_UNUSED(argc);
    GGML_UNUSED(argv);

    ggml_backend_load_all();

    ggml_backend_dev_t vk_dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        std::string name = ggml_backend_dev_name(dev);
        if (name.find("Vulkan") != std::string::npos) {
            vk_dev = dev;
            break;
        }
    }
    if (vk_dev == nullptr) {
        fprintf(stderr, "c22-topk-equivalence: no Vulkan device found\n");
        return 1;
    }

    ggml_backend_ptr backend(ggml_backend_dev_init(vk_dev, nullptr));
    if (backend == nullptr) {
        fprintf(stderr, "c22-topk-equivalence: Vulkan backend init failed\n");
        return 1;
    }

    printf("c22-topk-equivalence: backend %s (%s)\n",
           ggml_backend_name(backend.get()), ggml_backend_dev_description(vk_dev));

    const std::vector<test_case_t> cases = {
        { "1x32768 k=2051 (QSA shape)",   1, 32768, 2051, false },
        { "2x32768 k=2051 (multi row)",   2, 32768, 2051, false },
        { "1x4096  k=2051 (multi chunk)", 1,  4096, 2051, false },
        { "2x32768 k=2051 (tie stress)",  2, 32768, 2051, true  },
    };

    int n_pass = 0;
    int n_fail = 0;
    for (const auto & tc : cases) {
        std::mt19937 rng(20260906u);
        std::string err;
        const bool ok = run_case(backend.get(), tc, rng, err);
        printf("  [%s] %s\n", ok ? "PASS" : "FAIL", tc.name);
        if (!ok) {
            printf("    -> %s\n", err.c_str());
            ++n_fail;
        } else {
            ++n_pass;
        }
    }

    printf("c22-topk-equivalence: %d passed, %d failed\n", n_pass, n_fail);
    return n_fail == 0 ? 0 : 1;
}
