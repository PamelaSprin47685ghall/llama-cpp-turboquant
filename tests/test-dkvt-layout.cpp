#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include "ggml.h"
#include "../src/llama-kv-cache-dkvt.h"
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

static int test_tg_target_type_mapping() {
    printf("=== test_tg_target_type_mapping ===\n");

    if (dkvt_tg_type_k(GGML_TYPE_TURBO4_0) != GGML_TYPE_F16 ||
        dkvt_tg_type_k(GGML_TYPE_TURBO2_0) != GGML_TYPE_F16) {
        fprintf(stderr, "FAIL: DKVT TG K target type must be f16\n");
        return 1;
    }

    if (dkvt_tg_type_v(GGML_TYPE_TURBO4_0) != GGML_TYPE_Q8_0 ||
        dkvt_tg_type_v(GGML_TYPE_TURBO2_0) != GGML_TYPE_Q8_0) {
        fprintf(stderr, "FAIL: DKVT TG V target type must be q8_0\n");
        return 1;
    }

    if (ggml_row_size(dkvt_tg_type_k(GGML_TYPE_TURBO4_0), 128) != 256 ||
        ggml_row_size(dkvt_tg_type_v(GGML_TYPE_TURBO2_0), 128) != 136) {
        fprintf(stderr, "FAIL: DKVT TG row sizes do not match f16/q8_0 layout\n");
        return 1;
    }

    printf("  test_tg_target_type_mapping PASSED\n");
    return 0;
}

static int test_dkvt_decode_transcode_gate() {
    printf("=== test_dkvt_decode_transcode_gate ===\n");

    // Case 1: pure single-token decode with KV data — must transcode
    if (!dkvt_should_transcode_before_graph(1, 1, true)) {
        fprintf(stderr, "FAIL: decode (1,1,kv=true) must trigger transcode\n");
        return 1;
    }

    // Case 1b: 1-token prompt (KV empty) — must NOT transcode
    if (dkvt_should_transcode_before_graph(1, 1, false)) {
        fprintf(stderr, "FAIL: 1-token prompt (1,1,kv=false) must NOT trigger transcode\n");
        return 1;
    }

    // Case 2: MTP speculative verify batch (1 accepted + 3 draft = 4 tokens, all logits)
    if (!dkvt_should_transcode_before_graph(4, 4, true)) {
        fprintf(stderr, "FAIL: speculative verify (4,4,kv=true) must trigger transcode\n");
        return 1;
    }

    // Case 3: mid-prefill multi-token (n_outputs=1 means only last token emits logit)
    if (dkvt_should_transcode_before_graph(3072, 1, true)) {
        fprintf(stderr, "FAIL: mid-prefill (3072,1) must NOT trigger transcode\n");
        return 1;
    }

    // Case 4: full prefill (all tokens output, but this is PP not TG)
    if (dkvt_should_transcode_before_graph(3072, 3072, true)) {
        fprintf(stderr, "FAIL: full prefill (3072,3072) must NOT trigger transcode\n");
        return 1;
    }

    // Case 5: after TG, full prefill must reset PP bind before graph (second request)
    if (!dkvt_should_reset_before_graph(3072, 3072, true, true)) {
        fprintf(stderr, "FAIL: post-TG full prefill must dkvt_reset\n");
        return 1;
    }
    if (dkvt_should_reset_before_graph(4, 4, true, true)) {
        fprintf(stderr, "FAIL: post-TG verify (4,4) must NOT reset (transcode instead)\n");
        return 1;
    }
    if (dkvt_should_reset_before_graph(3072, 3072, false, true)) {
        fprintf(stderr, "FAIL: reset gate must be false when not transcoded\n");
        return 1;
    }

    printf("  test_dkvt_decode_transcode_gate PASSED\n");
    return 0;
}

static int test_dkvt_transcode_row_count() {
    printf("=== test_dkvt_transcode_row_count ===\n");

    // Case 1: empty cache (used_max_p1=0) → floor at n_pad_cur=256
    if (dkvt_kv_rows_to_transcode(0, 256000, 1) != 256) {
        fprintf(stderr, "FAIL: empty cache must floor to 256\n");
        return 1;
    }

    // Case 2: small usage (100) → floor at 256
    if (dkvt_kv_rows_to_transcode(100, 256000, 1) != 256) {
        fprintf(stderr, "FAIL: 100 used must floor to 256\n");
        return 1;
    }

    // Case 3: usage crosses boundary (300) → pad up to 512
    if (dkvt_kv_rows_to_transcode(300, 256000, 1) != 512) {
        fprintf(stderr, "FAIL: 300 used must pad to 512\n");
        return 1;
    }

    // Case 4: exact multiple of 256 (512) → no over-padding
    if (dkvt_kv_rows_to_transcode(512, 256000, 1) != 512) {
        fprintf(stderr, "FAIL: 512 used must stay at 512\n");
        return 1;
    }

    // Case 5: cells_size cap (request 1000 but only 500 cells)
    if (dkvt_kv_rows_to_transcode(1000, 500, 1) != 500) {
        fprintf(stderr, "FAIL: must cap at cells_size=500\n");
        return 1;
    }

    // Case 6: large prefill (3072) → already aligned to 256
    if (dkvt_kv_rows_to_transcode(3072, 256000, 1) != 3072) {
        fprintf(stderr, "FAIL: 3072 used must stay at 3072\n");
        return 1;
    }

    // Case 6b: 3200 → pad to 3328
    if (dkvt_kv_rows_to_transcode(3200, 256000, 1) != 3328) {
        fprintf(stderr, "FAIL: 3200 used must pad to 3328\n");
        return 1;
    }

    // Case 7: n_pad=0 still floors to 256
    if (dkvt_kv_rows_to_transcode(0, 256000, 0) != 256) {
        fprintf(stderr, "FAIL: n_pad=0 must still floor to 256\n");
        return 1;
    }

    // Case 8: n_pad=512 → n_pad_cur=512, floor at 512
    if (dkvt_kv_rows_to_transcode(100, 256000, 512) != 512) {
        fprintf(stderr, "FAIL: n_pad=512 must floor to 512\n");
        return 1;
    }

    printf("  test_dkvt_transcode_row_count PASSED\n");
    return 0;
}

int main() {
    int failures = 0;
    failures += test_contiguous_memory_layout();
    failures += test_structural_regression();
    failures += test_tg_target_type_mapping();
    failures += test_dkvt_decode_transcode_gate();
    failures += test_dkvt_transcode_row_count();
    return failures == 0 ? 0 : 1;
}
