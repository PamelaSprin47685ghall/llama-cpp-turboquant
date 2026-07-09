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

int test_dkvt_clear_resets_layout() {
    printf("=== test_dkvt_clear_resets_layout ===\n");

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

    llama_kv_cache cache(
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
        nullptr, nullptr, nullptr, nullptr
    );

    size_t union_size = 1024 * 1024 * 16;
    std::vector<uint8_t> mock_vram(union_size + 256);
    uintptr_t ptr_start = reinterpret_cast<uintptr_t>(mock_vram.data());
    ptr_start = (ptr_start + 127) & ~127;

    cache.vram_union_block = reinterpret_cast<ggml_backend_buffer_t>(ptr_start);
    cache.owns_union_block = false;
    cache.size_act_pp = 8 * 1024 * 1024;
    cache.size_act_tg = 1 * 1024 * 1024;
    cache.ptr_start = reinterpret_cast<char *>(ptr_start);
    cache.ptr_end = cache.ptr_start + union_size;
    cache.union_size = union_size;

    cache.init_dkvt_sum_kv_sizes(1);
    cache.dkvt_bind_pp();

    void * initial_k0 = cache.layers[0].k->data;
    void * initial_v0 = cache.layers[0].v->data;

    cache.is_transcoded_tg = true;
    cache.dkvt_bind_tg();
    void * tg_k0 = cache.layers[0].k->data;
    void * tg_v0 = cache.layers[0].v->data;

    if (initial_k0 == tg_k0 || initial_v0 == tg_v0) {
        fprintf(stderr, "FAIL: TG layout pointers are same as PP\n");
        llama_model_free(model);
        return 1;
    }

    cache.clear(false);

    if (cache.is_transcoded_tg) {
        fprintf(stderr, "FAIL: clear did not reset is_transcoded_tg\n");
        llama_model_free(model);
        return 1;
    }
    if (cache.layers[0].k->data != initial_k0 || cache.layers[0].v->data != initial_v0) {
        fprintf(stderr, "FAIL: clear did not re-bind PP pointers\n");
        llama_model_free(model);
        return 1;
    }

    llama_model_free(model);
    printf("  test_dkvt_clear_resets_layout PASSED\n");
    return 0;
}

int test_dkvt_companion_layout_sync_after_parent_reset() {
    printf("=== test_dkvt_companion_layout_sync_after_parent_reset ===\n");

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

    size_t union_size = 1024 * 1024 * 16;
    std::vector<uint8_t> mock_vram(union_size + 256);
    uintptr_t ptr_start = reinterpret_cast<uintptr_t>(mock_vram.data());
    ptr_start = (ptr_start + 127) & ~127;

    parent_kv->vram_union_block = reinterpret_cast<ggml_backend_buffer_t>(ptr_start);
    parent_kv->owns_union_block = false;
    parent_kv->size_act_pp = 8 * 1024 * 1024;
    parent_kv->size_act_tg = 1 * 1024 * 1024;
    parent_kv->ptr_start = reinterpret_cast<char *>(ptr_start);
    parent_kv->ptr_end = parent_kv->ptr_start + union_size;
    parent_kv->union_size = union_size;

    parent_kv->init_dkvt_sum_kv_sizes(1);
    parent_kv->dkvt_bind_pp();

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

    child_cache.init_dkvt_borrow();

    if (child_cache.is_transcoded_tg != false) {
        fprintf(stderr, "FAIL: companion should start in PP layout\n");
        llama_model_free(model);
        return 1;
    }

    parent_kv->is_transcoded_tg = true;
    parent_kv->dkvt_bind_tg();

    child_cache.transcode_to_tg(nullptr);

    if (child_cache.is_transcoded_tg != true) {
        fprintf(stderr, "FAIL: companion did not sync to TG layout\n");
        llama_model_free(model);
        return 1;
    }

    parent_kv->clear(true);

    child_cache.init_dkvt_borrow();

    if (child_cache.is_transcoded_tg != false) {
        fprintf(stderr, "FAIL: companion should reset to PP layout after parent clear/reset\n");
        llama_model_free(model);
        return 1;
    }

    if (child_cache.layers[0].k->data != parent_kv->layers[0].k->data ||
        child_cache.layers[0].v->data != parent_kv->layers[0].v->data) {
        fprintf(stderr, "FAIL: companion pointers did not match parent after parent clear/reset\n");
        llama_model_free(model);
        return 1;
    }

    llama_model_free(model);
    printf("  test_dkvt_companion_layout_sync_after_parent_reset PASSED\n");
    return 0;
}

int main() {
    int failures = 0;
    failures += test_dkvt_clear_resets_layout();
    failures += test_dkvt_companion_layout_sync_after_parent_reset();
    return failures == 0 ? 0 : 1;
}
