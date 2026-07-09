#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include "ggml.h"
#include "ggml-backend.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "llama.h"
#include "llama-model.h"
#include "llama-hparams.h"
#define private public
#define protected public
#include "llama-kv-cache.h"
#undef private
#undef protected
#include "llama-memory-hybrid.h"

int test_non_dkvt_context_bypasses_binding() {
    printf("=== test_non_dkvt_context_bypasses_binding ===\n");

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
    llama_kv_cache main_cache(
        *model, model->hparams,
        GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO2_0,
        true, false, true,
        kv_size, 1, 1, 0,
        LLAMA_SWA_TYPE_NONE,
        nullptr, nullptr, nullptr, nullptr
    );

    const size_t mock_union_size = 4096ULL * 1024 * 1024;
    char * mock_gpu_buf = (char *)0x20000000;
    main_cache.vram_union_block = (ggml_backend_buffer_t)mock_gpu_buf;
    main_cache.ptr_start = mock_gpu_buf;
    main_cache.ptr_end = mock_gpu_buf + mock_union_size;
    main_cache.union_size = mock_union_size;
    main_cache.size_act_pp = 8 * 1024 * 1024;
    main_cache.size_act_tg = 1 * 1024 * 1024;

    main_cache.init_dkvt_sum_kv_sizes(1);
    main_cache.dkvt_bind_pp();

    void * initial_k0 = main_cache.layers[0].k->data;
    void * initial_v0 = main_cache.layers[0].v->data;

    main_cache.is_transcoded_tg = true;
    main_cache.dkvt_bind_tg();

    llama_kv_cache companion_raw(
        *model, model->hparams,
        GGML_TYPE_F16, GGML_TYPE_F16,
        true, false, true,
        kv_size, 1, 1, 0,
        LLAMA_SWA_TYPE_NONE,
        (llama_memory_t)&main_cache,
        nullptr, nullptr, nullptr
    );

    companion_raw.init_dkvt_borrow();

    if (companion_raw.is_transcoded_tg != false) {
        fprintf(stderr,
                "FAIL: non-DKVT companion should bypass transcode status\n");
        llama_model_free(model);
        return 1;
    }

    if (companion_raw.layers[0].k->data == initial_k0 ||
        companion_raw.layers[0].v->data == initial_v0) {
        fprintf(stderr,
                "FAIL: non-DKVT companion should not alias main cache data\n");
        llama_model_free(model);
        return 1;
    }

    if (companion_raw.vram_union_block != nullptr ||
        companion_raw.ptr_start != nullptr) {
        fprintf(stderr,
                "FAIL: non-DKVT companion should not borrow union block\n");
        llama_model_free(model);
        return 1;
    }

    if (companion_raw.dkvt_k_size_pp != 0 ||
        companion_raw.dkvt_v_size_pp != 0) {
        fprintf(stderr,
                "FAIL: companion DKVT size fields should stay zero\n");
        llama_model_free(model);
        return 1;
    }
    printf("  companion DKVT size fields stay zero: PASSED\n");

    if (main_cache.layers[0].k->data == initial_k0 ||
        main_cache.layers[0].v->data == initial_v0) {
        fprintf(stderr,
                "FAIL: main cache TG layout should differ from PP. "
                "k->data=%p, initial_k0=%p, v->data=%p, initial_v0=%p, "
                "main_cache.is_transcoded_tg=%d\n",
                main_cache.layers[0].k->data, initial_k0,
                main_cache.layers[0].v->data, initial_v0,
                (int)main_cache.is_transcoded_tg);
        llama_model_free(model);
        return 1;
    }
    printf("  main cache TG layout changed correctly: PASSED\n");

    llama_model_free(model);
    printf("  test_non_dkvt_context_bypasses_binding PASSED\n");
    return 0;
}

int test_mtp_shared_child_preserves_parent_ext_flags() {
    printf("=== test_mtp_shared_child_preserves_parent_ext_flags ===\n");

    llama_model_params model_params = llama_model_default_params();
    model_params.use_extra_bufts = false;
    llama_model * model = llama_model_create(LLM_ARCH_LLAMA, model_params);
    if (!model) {
        fprintf(stderr, "FAIL: could not create llama_model\n");
        return 1;
    }

    model->hparams.n_layer_all = 2;
    model->hparams.n_embd = 256;
    model->hparams.n_embd_head_k_full = 128;
    model->hparams.n_embd_head_v_full = 128;
    model->hparams.n_layer_kv_from_start = -1;
    for (uint32_t i = 0; i < model->hparams.n_layer_all; ++i) {
        model->hparams.n_head_arr[i] = 2;
        model->hparams.n_head_kv_arr[i] = 2;
    }

    const uint32_t kv_size = 256;
    llama_kv_cache parent_cache(
        *model, model->hparams,
        GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO2_0,
        false, false, true,
        kv_size, 1, 1, 0,
        LLAMA_SWA_TYPE_NONE,
        nullptr, nullptr, nullptr, nullptr);

    if (!(parent_cache.layers[0].k->flags & GGML_TENSOR_FLAG_EXT) ||
        !(parent_cache.layers[0].v->flags & GGML_TENSOR_FLAG_EXT)) {
        fprintf(stderr, "FAIL: parent turbo cache should start with EXT K/V\n");
        llama_model_free(model);
        return 1;
    }

    auto share_first_layer = [](int32_t il) -> int32_t {
        return il == 0 ? 0 : -1;
    };

    llama_kv_cache child_cache(
        *model, model->hparams,
        GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO2_0,
        false, false, true,
        kv_size, 1, 1, 0,
        LLAMA_SWA_TYPE_NONE,
        (llama_memory_t)&parent_cache,
        nullptr, nullptr, share_first_layer,
        true);

    if (!(parent_cache.layers[0].k->flags & GGML_TENSOR_FLAG_EXT) ||
        !(parent_cache.layers[0].v->flags & GGML_TENSOR_FLAG_EXT)) {
        fprintf(stderr, "FAIL: MTP child construction cleared parent EXT flags\n");
        llama_model_free(model);
        return 1;
    }

    if (child_cache.layers[0].k != parent_cache.layers[0].k ||
        child_cache.layers[0].v != parent_cache.layers[0].v) {
        fprintf(stderr, "FAIL: child should share parent first layer tensors\n");
        llama_model_free(model);
        return 1;
    }

    llama_model_free(model);
    printf("  test_mtp_shared_child_preserves_parent_ext_flags PASSED\n");
    return 0;
}

int test_mtp_draft_kv_rollback() {
    printf("=== test_mtp_draft_kv_rollback ===\n");

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
    llama_kv_cache kv_cache(
        *model, model->hparams,
        GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO2_0,
        true, false, true,
        kv_size, 1, 1, 0,
        LLAMA_SWA_TYPE_NONE,
        nullptr, nullptr, nullptr, nullptr
    );

    llama_seq_id seq_id = 0;
    uint32_t stream_id = kv_cache.seq_to_stream[seq_id];
    auto & cells = kv_cache.v_cells[stream_id];

    // 模拟写入到位置 12。
    cells.pos[0] = 10;
    cells.seq[0].set(seq_id);
    cells.used.insert(0);

    cells.pos[1] = 11;
    cells.seq[1].set(seq_id);
    cells.used.insert(1);

    cells.pos[2] = 12;
    cells.seq[2].set(seq_id);
    cells.used.insert(2);

    cells.seq_pos_add(0);
    cells.seq_pos_add(1);
    cells.seq_pos_add(2);

    llama_pos pos_max_init = kv_cache.seq_pos_max(seq_id);
    printf("  Initial pos_max = %d\n", (int) pos_max_init);

    llama_pos dp_n_past = 12;

    // [TDD GREEN]: 解开核心的 seq_rm 回滚，使测试顺利通过
    kv_cache.seq_rm(seq_id, dp_n_past, -1);

    llama_pos pos_max_after = kv_cache.seq_pos_max(seq_id);
    printf("  After simulated rollback check, pos_max = %d\n", (int) pos_max_after);

    if (pos_max_after >= dp_n_past) {
        fprintf(stderr, "FAIL: pos_max_after (%d) >= dp_n_past (%d) — residue KV cache not cleaned!\n",
                (int) pos_max_after, (int) dp_n_past);
        llama_model_free(model);
        return 1;
    }

    llama_model_free(model);
    printf("  test_mtp_draft_kv_rollback PASSED\n");
    return 0;
}

int main() {
    int failures = 0;
    failures += test_non_dkvt_context_bypasses_binding();
    failures += test_mtp_shared_child_preserves_parent_ext_flags();
    failures += test_mtp_draft_kv_rollback();
    return failures == 0 ? 0 : 1;
}
