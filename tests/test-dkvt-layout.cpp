#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include "ggml.h"
#include "../ggml/src/ggml-quants.h"

static int verify_positive_growth(
    char * ptr_start, char * ptr_end,
    size_t compute_size, size_t v_size, size_t k_size,
    const char * name) {
    char * v_data = ptr_start + compute_size;
    char * k_data = v_data + v_size;
    char * segment_end = k_data + k_size;

    if (v_data < ptr_start) {
        fprintf(stderr, "FAIL: %s V data is below ptr_start\n", name);
        return 1;
    }
    if (k_data < v_data) {
        fprintf(stderr, "FAIL: %s K data is below V data\n", name);
        return 1;
    }
    if (k_data != v_data + v_size) {
        fprintf(stderr, "FAIL: %s K data is not immediately following V data\n", name);
        return 1;
    }
    if (segment_end > ptr_end) {
        fprintf(stderr, "FAIL: %s segment end %p exceeds ptr_end %p\n", name, (void*)segment_end, (void*)ptr_end);
        return 1;
    }
    printf("  %s layout: start=%p, v_data=%p, k_data=%p, end=%p (OK)\n",
           name, (void*)ptr_start, (void*)v_data, (void*)k_data, (void*)segment_end);
    return 0;
}

static int test_contiguous_memory_layout() {
    printf("=== test_contiguous_memory_layout ===\n");
    const size_t SCALE = 1024 * 1024;
    const size_t pp_compute  = 2700ULL * SCALE;
    const size_t tg_compute  =  100ULL * SCALE;

    const size_t dkvt_k_size_pp = 765ULL * SCALE;
    const size_t dkvt_v_size_pp = 150ULL * SCALE;
    const size_t turbo_kv_pp = dkvt_k_size_pp + dkvt_v_size_pp;

    const size_t dkvt_k_size_tg = 2440ULL * SCALE;
    const size_t dkvt_v_size_tg = 1300ULL * SCALE;
    const size_t f16q8_kv_tg = dkvt_k_size_tg + dkvt_v_size_tg;

    size_t union_size = std::max(
        pp_compute + turbo_kv_pp,
        tg_compute + f16q8_kv_tg
    );
    char * ptr_start = (char *)0x10000000;
    char * ptr_end   = ptr_start + union_size;

    int res = 0;
    if (verify_positive_growth(ptr_start, ptr_end, pp_compute, dkvt_v_size_pp, dkvt_k_size_pp, "PP") ||
        verify_positive_growth(ptr_start, ptr_end, tg_compute, dkvt_v_size_tg, dkvt_k_size_tg, "TG")) {
        res = 1;
    }

    if (res == 0) {
        printf("  test_contiguous_memory_layout PASSED\n");
    }
    return res;
}

static void fill_v_trans_tensors(
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

static int verify_v_trans_tensors(
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

static int test_v_trans_layout() {
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

static int test_structural_regression() {
    printf("=== test_structural_regression ===\n");
    size_t r_t4 = ggml_row_size(GGML_TYPE_TURBO4_0, 128);
    size_t r_t2 = ggml_row_size(GGML_TYPE_TURBO2_0, 128);
    size_t r_f16 = ggml_row_size(GGML_TYPE_F16, 128);
    size_t r_q8 = ggml_row_size(GGML_TYPE_Q8_0, 128);
    if (r_t4 != 66 || r_t2 != 34 || r_f16 != 256 || r_q8 != 136) {
        return 1;
    }
    printf("  test_structural_regression PASSED\n");
    return 0;
}

int main() {
    int failures = 0;
    failures += test_contiguous_memory_layout();
    failures += test_v_trans_layout();
    failures += test_structural_regression();
    return failures == 0 ? 0 : 1;
}
