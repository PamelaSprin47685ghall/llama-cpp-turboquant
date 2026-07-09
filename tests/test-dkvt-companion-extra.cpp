#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include "ggml.h"
#include "ggml-backend.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "llama.h"
#include "llama-model.h"
#include "llama-hparams.h"
#include "llama-kv-cache.h"
#include "llama-memory-hybrid.h"

int test_hybrid_as_kv_cache_resolution() {
    printf("=== test_hybrid_as_kv_cache_resolution ===\n");

    llama_model_params model_params = llama_model_default_params();
    model_params.use_extra_bufts = false;
    llama_model * model = llama_model_create(LLM_ARCH_LLAMA, model_params);
    if (!model) {
        fprintf(stderr, "FAIL: could not create llama_model\n");
        return 1;
    }

    model->hparams.n_layer_all = 4;
    model->hparams.n_embd = 256;
    model->hparams.n_embd_head_k_full = 64;
    model->hparams.n_embd_head_v_full = 64;
    model->hparams.n_layer_kv_from_start = -1;
    for (uint32_t i = 0; i < model->hparams.n_layer_all; ++i) {
        model->hparams.n_head_arr[i] = 4;
        model->hparams.n_head_kv_arr[i] = 4;
    }

    const uint32_t kv_size = 256;

    llama_memory_hybrid hybrid_mem(
        *model,
        GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO2_0,
        true,
        kv_size,
        1,
        0,
        LLAMA_SWA_TYPE_NONE,
        GGML_TYPE_F16, GGML_TYPE_F16,
        128,
        1,
        1,
        false,
        true,
        nullptr, nullptr
    );

    llama_kv_cache * kv_ptr = hybrid_mem.as_kv_cache();
    if (!kv_ptr) {
        fprintf(stderr, "FAIL: as_kv_cache() returned nullptr\n");
        llama_model_free(model);
        return 1;
    }

    llama_kv_cache * attn_ptr = hybrid_mem.get_mem_attn();
    if (kv_ptr != attn_ptr) {
        fprintf(stderr, "FAIL: as_kv_cache() != get_mem_attn()\n");
        llama_model_free(model);
        return 1;
    }

    llama_memory_i * mem_base = &hybrid_mem;
    llama_kv_cache * kv_via_base = mem_base->as_kv_cache();
    if (kv_via_base != attn_ptr) {
        fprintf(stderr, "FAIL: base->as_kv_cache() != get_mem_attn()\n");
        llama_model_free(model);
        return 1;
    }

    llama_model_free(model);
    printf("  test_hybrid_as_kv_cache_resolution PASSED\n");
    return 0;
}

int test_mtp_child_inherits_parent_kv() {
    printf("=== test_mtp_child_inherits_parent_kv ===\n");

    llama_model_params model_params = llama_model_default_params();
    model_params.use_extra_bufts = false;
    llama_model * model = llama_model_create(LLM_ARCH_LLAMA, model_params);
    if (!model) {
        fprintf(stderr, "FAIL: could not create llama_model\n");
        return 1;
    }

    model->hparams.n_layer_all = 4;
    model->hparams.n_embd = 256;
    model->hparams.n_embd_head_k_full = 64;
    model->hparams.n_embd_head_v_full = 64;
    model->hparams.n_layer_kv_from_start = -1;
    for (uint32_t i = 0; i < model->hparams.n_layer_all; ++i) {
        model->hparams.n_head_arr[i] = 4;
        model->hparams.n_head_kv_arr[i] = 4;
    }

    const uint32_t kv_size = 256;

    llama_memory_hybrid hybrid_parent(
        *model,
        GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO2_0,
        true,
        kv_size,
        1,
        0,
        LLAMA_SWA_TYPE_NONE,
        GGML_TYPE_F16, GGML_TYPE_F16,
        128,
        1,
        1,
        false,
        true,
        nullptr, nullptr
    );

    llama_memory_i * parent_base = &hybrid_parent;
    llama_kv_cache * parent_kv = parent_base->as_kv_cache();
    if (!parent_kv) {
        fprintf(stderr, "FAIL: parent as_kv_cache() returned nullptr\n");
        llama_model_free(model);
        return 1;
    }

    llama_kv_cache child_cache(
        *model, model->hparams,
        GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO2_0,
        true,
        false,
        true,
        kv_size,
        1,
        1,
        0,
        LLAMA_SWA_TYPE_NONE,
        (llama_memory_t)parent_base,
        nullptr, nullptr, nullptr
    );

    if (!child_cache.other) {
        fprintf(stderr, "FAIL: child->other is nullptr\n");
        llama_model_free(model);
        return 1;
    }

    if (child_cache.other != parent_kv) {
        fprintf(stderr, "FAIL: child->other != parent_kv\n");
        llama_model_free(model);
        return 1;
    }

    if (child_cache.get_size() != parent_kv->get_size()) {
        fprintf(stderr, "FAIL: child kv_size mismatch\n");
        llama_model_free(model);
        return 1;
    }

    llama_model_free(model);
    printf("  test_mtp_child_inherits_parent_kv PASSED\n");
    return 0;
}

int test_transcoded_layout_strides() {
    printf("=== test_transcoded_layout_strides ===\n");

    llama_model_params model_params = llama_model_default_params();
    model_params.use_extra_bufts = false;
    llama_model * model = llama_model_create(LLM_ARCH_LLAMA, model_params);
    if (!model) {
        fprintf(stderr, "FAIL: could not create llama_model\n");
        return 1;
    }

    model->hparams.n_layer_all = 4;
    model->hparams.n_embd = 256;
    model->hparams.n_embd_head_k_full = 64;
    model->hparams.n_embd_head_v_full = 64;
    model->hparams.n_layer_kv_from_start = -1;
    for (uint32_t i = 0; i < model->hparams.n_layer_all; ++i) {
        model->hparams.n_head_arr[i] = 4;
        model->hparams.n_head_kv_arr[i] = 4;
    }

    const uint32_t kv_size = 256;
    const size_t mock_union_size = 4096ULL * 1024 * 1024;
    char * mock_gpu_buf = (char *)0x20000000;

    llama_kv_cache main_cache(
        *model, model->hparams,
        GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO2_0,
        true, false, true,
        kv_size, 1, 1, 0,
        LLAMA_SWA_TYPE_NONE,
        nullptr, nullptr, nullptr, nullptr
    );

    main_cache.vram_union_block = (ggml_backend_buffer_t)mock_gpu_buf;
    main_cache.ptr_start = mock_gpu_buf;
    main_cache.ptr_end = mock_gpu_buf + mock_union_size;
    main_cache.union_size = mock_union_size;

    main_cache.init_dkvt_sum_kv_sizes(1);
    main_cache.dkvt_bind_pp();

    main_cache.is_transcoded_tg = true;
    main_cache.dkvt_bind_tg();

    struct ggml_init_params ggml_params = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(ggml_params);

    llama_kv_cache::slot_info sinfo;
    sinfo.s0 = 0;
    sinfo.s1 = 0;
    sinfo.strm = {0};
    sinfo.idxs = {{0}};

    const uint32_t il = 0;
    const uint32_t n_kv = 1;
    ggml_tensor * k_view = main_cache.get_k(ctx, il, n_kv, sinfo);
    if (!k_view) {
        fprintf(stderr, "FAIL: get_k returned nullptr\n");
        ggml_free(ctx);
        llama_model_free(model);
        return 1;
    }

    ggml_tensor * k_raw = main_cache.layers[0].k;
    printf("  k_raw->type=%d, k_raw->nb[1]=%zu, k_view->nb[2]=%zu\n",
           k_raw->type, k_raw->nb[1], k_view->nb[2]);
    printf("  k_raw->ne[0]=%lld, k_view->ne[0]=%lld\n",
           (long long) k_raw->ne[0], (long long) k_view->ne[0]);

    if (k_view->nb[2] != k_raw->nb[1]) {
        fprintf(stderr, "FAIL: k_view->nb[2] (%zu) != k_raw->nb[1] (%zu) — "
                "stride collapsed after transcode, expected padded stride preserved\n",
                k_view->nb[2], k_raw->nb[1]);
        ggml_free(ctx);
        llama_model_free(model);
        return 1;
    }

    printf("  transcode stride preservation: PASSED\n");

    ggml_free(ctx);
    llama_model_free(model);
    printf("  test_transcoded_layout_strides PASSED\n");
    return 0;
}

int main() {
    int failures = 0;
    failures += test_hybrid_as_kv_cache_resolution();
    failures += test_mtp_child_inherits_parent_kv();
    failures += test_transcoded_layout_strides();
    return failures == 0 ? 0 : 1;
}
