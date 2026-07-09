#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include "ggml.h"
#include "../src/llama-kv-cache-dkvt.h"
#include "../ggml/src/ggml-quants.h"

int test_v_trans_layout();

void fill_v_trans_tensors(
    struct ggml_tensor * v_t, struct ggml_tensor * v_f,
    int64_t kv_size, int64_t head_size_v,
    int64_t n_head_kv, int64_t n_ctx) {
    for (int64_t h = 0; h < n_head_kv; ++h) {
        for (int64_t d = 0; d < head_size_v; ++d) {
            for (int64_t s = 0; s < kv_size; ++s) {
                for (int64_t t = 0; t < n_ctx; ++t) {
                    float val = (float)(h * 10000 + d * 100 +
                                         s * 10 + t);
                    int64_t off_t = s + d * kv_size +
                        h * kv_size * head_size_v +
                        t * kv_size * head_size_v * n_head_kv;
                    ((float *)v_t->data)[off_t] = val;
                    int64_t off_f = d + s * head_size_v +
                        h * head_size_v * kv_size +
                        t * head_size_v * kv_size * n_head_kv;
                    ((float *)v_f->data)[off_f] = val;
                }
            }
        }
    }
}

int verify_v_trans_tensors(
    struct ggml_tensor * v_t, struct ggml_tensor * v_f,
    int64_t kv_size, int64_t head_size_v,
    int64_t n_head_kv, int64_t n_ctx) {
    const int64_t n_mc = 10000;
    unsigned int seed_mc = 12345;
    int64_t mc_failures = 0;
    for (int64_t iter = 0; iter < n_mc; ++iter) {
        int64_t h = rand_r(&seed_mc) % n_head_kv;
        int64_t d = rand_r(&seed_mc) % head_size_v;
        int64_t s = rand_r(&seed_mc) % kv_size;
        int64_t t = rand_r(&seed_mc) % n_ctx;

        int64_t off_t = s + d * kv_size +
            h * kv_size * head_size_v +
            t * kv_size * head_size_v * n_head_kv;
        float val_t = ((float *)v_t->data)[off_t];

        int64_t off_f = d + s * head_size_v +
            h * head_size_v * kv_size +
            t * head_size_v * kv_size * n_head_kv;
        float val_f = ((float *)v_f->data)[off_f];

        if (val_t != val_f) {
            mc_failures++;
        }
    }
    return mc_failures > 0 ? 1 : 0;
}

int test_v_trans_layout() {
    printf("=== test_v_trans_layout ===\n");
    const int64_t kv_size    = 256;
    const int64_t head_size_v = 128;
    const int64_t n_head_kv   = 8;
    const int64_t n_ctx       = 4;

    size_t mem_sz = 2 * kv_size * head_size_v *
                    n_head_kv * n_ctx * sizeof(float) +
                    2 * 1024 * 1024;
    struct ggml_init_params params = {
        .mem_size   = mem_sz,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return 1;
    }
    struct ggml_tensor * v_t = ggml_new_tensor_4d(
        ctx, GGML_TYPE_F32, kv_size, head_size_v, n_head_kv, n_ctx);
    struct ggml_tensor * v_f = ggml_new_tensor_4d(
        ctx, GGML_TYPE_F32, head_size_v, kv_size, n_head_kv, n_ctx);

    fill_v_trans_tensors(
        v_t, v_f, kv_size, head_size_v, n_head_kv, n_ctx);
    int res = verify_v_trans_tensors(
        v_t, v_f, kv_size, head_size_v, n_head_kv, n_ctx);

    ggml_free(ctx);
    return res;
}

int main() {
    int res = test_v_trans_layout();
    if (res == 0) {
        printf("  test_v_trans_layout PASSED\n");
    } else {
        printf("  test_v_trans_layout FAILED\n");
    }
    return res;
}
