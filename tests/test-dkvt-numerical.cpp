#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include "ggml.h"
#include "../ggml/src/ggml-quants.h"

extern "C" void turbo_cpu_fwht_inverse(float * x, int group_size);

static double compute_rmse(
    const float * a, const float * b, int64_t n) {
    double sum_sq = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        double d = (double)a[i] - (double)b[i];
        sum_sq += d * d;
    }
    return sqrt(sum_sq / (double)n);
}

static double compute_max_abs(
    const float * a, const float * b, int64_t n) {
    double max_abs = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        double d = fabs((double)a[i] - (double)b[i]);
        if (d > max_abs) {
            max_abs = d;
        }
    }
    return max_abs;
}

static void fill_gaussian_signal(float * x, int64_t n) {
    unsigned int seed = 42;
    for (int64_t i = 0; i < n; i += 2) {
        float u1 = (float)(rand_r(&seed) + 1) /
                   ((float)RAND_MAX + 2.0f);
        float u2 = (float)(rand_r(&seed) + 1) /
                   ((float)RAND_MAX + 2.0f);
        float mag = sqrtf(-2.0f * logf(u1));
        x[i]     = mag * cosf(2.0f * M_PI * u2);
        x[i + 1] = mag * sinf(2.0f * M_PI * u2);
    }
}

static int test_turbo4_consistency(const float * original, int64_t k) {
    float dequantized[128];
    block_turbo4_0 quantized_block;
    quantize_row_turbo4_0_ref(original, &quantized_block, k);
    dequantize_row_turbo4_0(&quantized_block, dequantized, k);
    turbo_cpu_fwht_inverse(dequantized, k);

    double rmse = compute_rmse(original, dequantized, k);
    double max_err = compute_max_abs(original, dequantized, k);
    printf("  Turbo4 RMSE: %.6f, MaxAbsErr: %.6f\n", rmse, max_err);

    if (rmse > 0.15) {
        fprintf(stderr, "FAIL: Turbo4 RMSE %.6f > 0.15\n", rmse);
        return 1;
    }
    return 0;
}

static int test_turbo2_consistency(const float * original, int64_t k) {
    float dequantized[128];
    block_turbo2_0 quantized_block;
    quantize_row_turbo2_0_ref(original, &quantized_block, k);
    dequantize_row_turbo2_0(&quantized_block, dequantized, k);
    turbo_cpu_fwht_inverse(dequantized, k);

    double rmse = compute_rmse(original, dequantized, k);
    double max_err = compute_max_abs(original, dequantized, k);
    printf("  Turbo2 RMSE: %.6f, MaxAbsErr: %.6f\n", rmse, max_err);

    if (rmse > 0.40) {
        fprintf(stderr, "FAIL: Turbo2 RMSE %.6f > 0.40\n", rmse);
        return 1;
    }
    return 0;
}

static int test_q8_consistency(int64_t k) {
    float data_q8[128];
    unsigned int seed_q8 = 42;
    for (int64_t i = 0; i < k; ++i) {
        float u1 = (float)(rand_r(&seed_q8) + 1) /
                   ((double)RAND_MAX + 2.0);
        float u2 = (float)(rand_r(&seed_q8) + 1) /
                   ((double)RAND_MAX + 2.0);
        data_q8[i] = sqrtf(-2.0f * logf(u1)) *
                     cosf(2.0f * M_PI * u2);
    }
    block_q8_0 q8_blocks[4];  // 128 / 32 = 4 blocks
    quantize_row_q8_0_ref(data_q8, q8_blocks, k);
    float dequant_q8[128];
    dequantize_row_q8_0(q8_blocks, dequant_q8, k);

    double rmse = compute_rmse(data_q8, dequant_q8, k);
    printf("  Q8_0 RMSE: %.6f\n", rmse);
    if (rmse > 0.05) {
        fprintf(stderr, "FAIL: Q8_0 RMSE %.6f > 0.05\n", rmse);
        return 1;
    }
    return 0;
}

int main() {
    printf("=== test_turbo_numerical_consistency ===\n");
    const int64_t k = 128;
    float original[128];
    fill_gaussian_signal(original, k);

    int failures = 0;
    failures += test_turbo4_consistency(original, k);
    failures += test_turbo2_consistency(original, k);
    failures += test_q8_consistency(k);

    if (failures == 0) {
        printf("  test_turbo_numerical_consistency PASSED\n");
    }
    return failures == 0 ? 0 : 1;
}
