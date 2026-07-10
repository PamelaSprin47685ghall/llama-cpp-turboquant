#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include "ggml.h"
#include "ggml-backend.h"
#include "../ggml/src/ggml-backend-impl.h"

extern "C" {
struct ggml_dyn_tallocr {
    size_t alignment; size_t max_chunk_size;
    size_t cap; size_t base_offset;
    void * chunks[256]; int n_chunks;
};
struct ggml_gallocr {
    void ** bufts; void ** buffers;
    struct ggml_dyn_tallocr ** buf_tallocs;
    bool * is_borrowed; size_t * borrowed_compute_cap;
    int n_buffers;
    void * hash_set; void * hash_values;
    void * node_allocs; int n_nodes;
    void * leaf_allocs; int n_leafs;
};
struct ggml_backend_sched {
    bool is_reset;
    bool is_alloc;
    int n_backends;
    ggml_backend_t backends[16];
    ggml_backend_buffer_type_t bufts[16];
    ggml_gallocr_t galloc;
};
}

#include "llama.h"
#include "llama-model.h"
#include "llama-hparams.h"
#define private public
#define protected public
#include "llama-kv-cache.h"
#undef private
#undef protected
#include "llama-memory-hybrid.h"

static int make_model(llama_model * & model) {
    llama_model_params model_params = llama_model_default_params();
    model_params.use_extra_bufts = false;
    model = llama_model_create(LLM_ARCH_LLAMA, model_params);
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
    return 0;
}

static void setup_parent_tg(llama_kv_cache & parent,
                            llama_model * model,
                            const uint32_t kv_size) {
    const size_t mock_union_size = 4096ULL * 1024 * 1024;
    char * mock_gpu_buf = (char *)0x20000000;

    parent.vram_union_block = (ggml_backend_buffer_t)mock_gpu_buf;
    parent.ptr_start = mock_gpu_buf;
    parent.ptr_end = mock_gpu_buf + mock_union_size;
    parent.union_size = mock_union_size;
    parent.size_act_pp = 8 * 1024 * 1024;
    parent.size_act_tg = 2 * 1024 * 1024;

    parent.init_dkvt_sum_kv_sizes(1);
    parent.dkvt_bind_pp();
    parent.is_transcoded_tg = true;
    parent.dkvt_bind_tg();
}

int test_mtp_base_offset_application() {
    printf("=== test_mtp_base_offset_application ===\n");

    llama_model * model = nullptr;
    if (make_model(model) != 0) return 1;

    const uint32_t kv_size = 256;

    llama_kv_cache parent_cache(
        *model, model->hparams,
        GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO2_0,
        true, false, true,
        kv_size, 1, 1, 0,
        LLAMA_SWA_TYPE_NONE,
        nullptr, nullptr, nullptr, nullptr
    );
    setup_parent_tg(parent_cache, model, kv_size);

    size_t expected_k_tg = parent_cache.dkvt_k_size_tg;
    size_t expected_v_tg = parent_cache.dkvt_v_size_tg;
    size_t expected_base_offset = expected_k_tg + expected_v_tg;
    size_t expected_cap = expected_base_offset + parent_cache.size_act_tg;

    printf("  parent k_tg=%zu v_tg=%zu base_offset=%zu cap=%zu\n",
           expected_k_tg, expected_v_tg, expected_base_offset, expected_cap);

    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    if (!cpu_backend) {
        fprintf(stderr, "FAIL: could not create CPU backend\n");
        llama_model_free(model);
        return 1;
    }

    ggml_backend_sched_t sched = ggml_backend_sched_new(
        &cpu_backend, nullptr, 1, GGML_DEFAULT_GRAPH_SIZE, false, false);
    if (!sched) {
        fprintf(stderr, "FAIL: could not create scheduler\n");
        llama_model_free(model);
        return 1;
    }

    auto share_all = [](int32_t il) -> int32_t { return 0; };

    llama_kv_cache companion_cache(
        *model, model->hparams,
        GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO2_0,
        true, false, true,
        kv_size, 1, 1, 0,
        LLAMA_SWA_TYPE_NONE,
        (llama_memory_t)&parent_cache,
        nullptr, nullptr, share_all,
        true, true
    );

    companion_cache.init_dkvt_borrow();

    if (!companion_cache.is_transcoded_tg) {
        fprintf(stderr, "FAIL: companion is_transcoded_tg should be true\n");
        ggml_backend_sched_free(sched);
        llama_model_free(model);
        return 1;
    }
    printf("  is_transcoded_tg synced: true\n");

    if (companion_cache.dkvt_k_size_tg != expected_k_tg ||
        companion_cache.dkvt_v_size_tg != expected_v_tg) {
        fprintf(stderr, "FAIL: DKVT sizes not inherited\n");
        ggml_backend_sched_free(sched);
        llama_model_free(model);
        return 1;
    }
    printf("  DKVT sizes inherited: k=%zu v=%zu\n",
           companion_cache.dkvt_k_size_tg, companion_cache.dkvt_v_size_tg);

    companion_cache.dkvt_sched = sched;
    companion_cache.dkvt_sched_buffer_id = 0;
    companion_cache.dkvt_apply_union_compute_cap(sched);

    ggml_gallocr_t galloc = sched->galloc;
    if (!galloc || galloc->n_buffers <= 0) {
        fprintf(stderr, "FAIL: galloc has no buffers\n");
        ggml_backend_sched_free(sched);
        llama_model_free(model);
        return 1;
    }

    struct ggml_dyn_tallocr * tallocr = galloc->buf_tallocs[0];
    if (!tallocr) {
        fprintf(stderr, "FAIL: tallocr is nullptr\n");
        ggml_backend_sched_free(sched);
        llama_model_free(model);
        return 1;
    }

    if (tallocr->base_offset != expected_base_offset) {
        fprintf(stderr, "FAIL: base_offset=%zu expected=%zu\n",
                tallocr->base_offset, expected_base_offset);
        ggml_backend_sched_free(sched);
        llama_model_free(model);
        return 1;
    }
    printf("  base_offset=%zu: PASSED\n", tallocr->base_offset);

    if (tallocr->cap != expected_cap) {
        fprintf(stderr, "FAIL: cap=%zu expected=%zu\n",
                tallocr->cap, expected_cap);
        ggml_backend_sched_free(sched);
        llama_model_free(model);
        return 1;
    }
    printf("  cap=%zu: PASSED\n", tallocr->cap);

    ggml_backend_sched_free(sched);
    llama_model_free(model);
    printf("  test_mtp_base_offset_application PASSED\n");
    return 0;
}

int main() {
    return test_mtp_base_offset_application();
}
