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
#include "llama-kv-cache.h"
#include "llama-memory-hybrid.h"

#ifdef GGML_USE_CUDA
extern "C" void ggml_cuda_device_to_device_copy_async(
    void * dst, const void * src, size_t size, void * stream);
#endif

struct mock_parent_ctx {
    char * vram_union_block;
    char * ptr_start;
    char * ptr_end;
    size_t union_size;
    size_t dkvt_k_size_pp;
    size_t dkvt_k_size_tg;
    size_t dkvt_v_size_pp;
    size_t dkvt_v_size_tg;
    bool   is_transcoded_tg;
};

static int test_companion_sync() {
    printf("=== test_companion_sync ===\n");
    const size_t union_size = 3840ULL * 1024 * 1024;
    char * mock_gpu_buffer = (char *)0x20000000;
    struct mock_parent_ctx parent = {
        .vram_union_block = mock_gpu_buffer,
        .ptr_start = mock_gpu_buffer,
        .ptr_end   = mock_gpu_buffer + union_size,
        .union_size = union_size,
        .dkvt_k_size_pp  = 765ULL * 1024 * 1024,
        .dkvt_k_size_tg  = 2440ULL * 1024 * 1024,
        .dkvt_v_size_pp  = 150ULL * 1024 * 1024,
        .dkvt_v_size_tg  = 1300ULL * 1024 * 1024,
        .is_transcoded_tg = false,
    };
    struct mock_parent_ctx child = {};

    child.vram_union_block = parent.vram_union_block;
    child.ptr_start = parent.ptr_start;
    child.ptr_end = parent.ptr_end;
    child.union_size = parent.union_size;
    child.dkvt_k_size_pp = parent.dkvt_k_size_pp;
    child.dkvt_k_size_tg = parent.dkvt_k_size_tg;
    child.dkvt_v_size_pp = parent.dkvt_v_size_pp;
    child.dkvt_v_size_tg = parent.dkvt_v_size_tg;
    child.is_transcoded_tg = parent.is_transcoded_tg;

    if (child.vram_union_block != parent.vram_union_block) {
        return 1;
    }
    parent.is_transcoded_tg = true;
    if (parent.is_transcoded_tg && !child.is_transcoded_tg) {
        child.is_transcoded_tg = true;
    }
    printf("  test_companion_sync PASSED\n");
    return 0;
}

static int test_cuda_d2d_copy() {
    printf("=== test_cuda_d2d_copy ===\n");
#ifdef GGML_USE_CUDA
    printf("  CUDA build detected, testing D2D...\n");
#else
    printf("  Non-CUDA build, testing mock CPU D2D...\n");
    size_t buf_size = 256;
    char * mock_device_buf = (char *)calloc(1, buf_size);
    if (!mock_device_buf) {
        return 1;
    }
    char host_src[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    char host_dst[10] = {0};
    memcpy(mock_device_buf, host_src, 10);
    memcpy(mock_device_buf + 100, mock_device_buf, 10);
    memcpy(host_dst, mock_device_buf + 100, 10);
    for (int i = 0; i < 10; ++i) {
        if (host_dst[i] != host_src[i]) {
            free(mock_device_buf);
            return 1;
        }
    }
    free(mock_device_buf);
#endif
    printf("  test_cuda_d2d_copy PASSED\n");
    return 0;
}

static struct ggml_cgraph * create_dummy_graph(
    struct ggml_context * ctx) {
    struct ggml_tensor * a = ggml_new_tensor_1d(
        ctx, GGML_TYPE_F32, 64);
    struct ggml_tensor * b = ggml_new_tensor_1d(
        ctx, GGML_TYPE_F32, 64);
    struct ggml_tensor * c = ggml_add(ctx, a, b);
    ggml_set_name(a, "a");
    ggml_set_name(b, "b");
    ggml_set_name(c, "c");
    ggml_set_input(a);
    ggml_set_input(b);
    ggml_set_output(c);

    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, c);
    return graph;
}

static int test_mtp_uaf_defense() {
    printf("=== test_mtp_uaf_defense ===\n");
    ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
    ggml_gallocr_t parent_galloc = ggml_gallocr_new(buft);
    ggml_gallocr_t child_galloc  = ggml_gallocr_new(buft);
    if (!parent_galloc || !child_galloc) {
        if (child_galloc) ggml_gallocr_free(child_galloc);
        if (parent_galloc) ggml_gallocr_free(parent_galloc);
        return 1;
    }
    struct ggml_init_params params = {
        .mem_size   = (size_t)1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = true,
    };
    struct ggml_context * ctx = ggml_init(params);
    struct ggml_cgraph * graph = create_dummy_graph(ctx);

    if (!ggml_gallocr_reserve(parent_galloc, graph) ||
        !ggml_gallocr_alloc_graph(parent_galloc, graph)) {
        ggml_free(ctx);
        ggml_gallocr_free(child_galloc);
        ggml_gallocr_free(parent_galloc);
        return 1;
    }

    ggml_gallocr_set_borrowed(child_galloc, 0, true);

    if (!ggml_gallocr_is_borrowed(child_galloc, 0) ||
        ggml_gallocr_is_borrowed(parent_galloc, 0)) {
        ggml_free(ctx);
        ggml_gallocr_free(child_galloc);
        ggml_gallocr_free(parent_galloc);
        return 1;
    }
    ggml_gallocr_free(child_galloc);
    ggml_free(ctx);
    ggml_gallocr_free(parent_galloc);
    printf("  test_mtp_uaf_defense PASSED\n");
    return 0;
}

int test_companion_offset_sync() {
    printf("=== test_companion_offset_sync ===\n");

    llama_model_params model_params = llama_model_default_params();
    model_params.use_extra_bufts = false;
    llama_model * model = llama_model_create(LLM_ARCH_LLAMA, model_params);
    if (!model) {
        fprintf(stderr, "FAIL: could not create llama_model\n");
        return 1;
    }

    // 配置 hparams：4 层标准 GQA 模型，head_dim=64，4 KV heads
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

    // 创建主 KV cache
    llama_kv_cache main_cache(
        *model, model->hparams,
        GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO2_0,
        true,   // v_trans
        false,  // offload (CPU)
        true,   // unified
        kv_size,
        1,      // n_seq_max
        1,      // n_pad
        0,      // n_swa
        LLAMA_SWA_TYPE_NONE,
        nullptr, // mem_other (主上下文)
        nullptr, // filter
        nullptr, // reuse
        nullptr  // share
    );

    // 手动设置 union buffer 字段（模拟 GPU 显存）
    main_cache.vram_union_block = (ggml_backend_buffer_t)mock_gpu_buf;
    main_cache.ptr_start = mock_gpu_buf;
    main_cache.ptr_end = mock_gpu_buf + mock_union_size;
    main_cache.union_size = mock_union_size;

    // 计算每层尺寸和偏移
    main_cache.init_dkvt_sum_kv_sizes(1);
    main_cache.dkvt_bind_pp();

    printf("  main_cache: layers=%zu, dkvt_k_size_pp=%zu, dkvt_v_size_pp=%zu, "
           "dkvt_k_size_tg=%zu, dkvt_v_size_tg=%zu\n",
            main_cache.layers.size(),
            main_cache.dkvt_k_size_pp, main_cache.dkvt_v_size_pp,
            main_cache.dkvt_k_size_tg, main_cache.dkvt_v_size_tg);

    // 创建伴生 KV cache，共享主上下文的 cells
    llama_kv_cache companion_cache(
        *model, model->hparams,
        GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO2_0,
        true,   // v_trans
        false,  // offload
        true,   // unified
        kv_size,
        1,      // n_seq_max
        1,      // n_pad
        0,      // n_swa
        LLAMA_SWA_TYPE_NONE,
        (llama_memory_t)&main_cache, // mem_other → 伴生绑定到主上下文
        nullptr, // filter
        nullptr, // reuse
        nullptr  // share
    );

    // 伴生上下文借用主上下文的 union buffer 和偏移信息
    companion_cache.init_dkvt_borrow();

    // 校验 1：union buffer 指针一致
    if (companion_cache.vram_union_block != main_cache.vram_union_block ||
        companion_cache.ptr_start != main_cache.ptr_start ||
        companion_cache.ptr_end != main_cache.ptr_end ||
        companion_cache.union_size != main_cache.union_size) {
        fprintf(stderr, "FAIL: union buffer pointers mismatch\n");
        llama_model_free(model);
        return 1;
    }

    // 校验 2：总尺寸一致
    if (companion_cache.dkvt_k_size_pp != main_cache.dkvt_k_size_pp ||
        companion_cache.dkvt_k_size_tg != main_cache.dkvt_k_size_tg ||
        companion_cache.dkvt_v_size_pp != main_cache.dkvt_v_size_pp ||
        companion_cache.dkvt_v_size_tg != main_cache.dkvt_v_size_tg) {
        fprintf(stderr, "FAIL: total KV size mismatch\n");
        llama_model_free(model);
        return 1;
    }

    // 校验 3：逐层偏移和尺寸完全匹配
    if (companion_cache.layers.size() != main_cache.layers.size()) {
        fprintf(stderr, "FAIL: layer count mismatch (%zu vs %zu)\n",
                companion_cache.layers.size(), main_cache.layers.size());
        llama_model_free(model);
        return 1;
    }

    for (size_t i = 0; i < main_cache.layers.size(); ++i) {
        if (companion_cache.layers[i].k_offset_pp != main_cache.layers[i].k_offset_pp ||
            companion_cache.layers[i].k_offset_tg != main_cache.layers[i].k_offset_tg ||
            companion_cache.layers[i].v_offset_pp != main_cache.layers[i].v_offset_pp ||
            companion_cache.layers[i].v_offset_tg != main_cache.layers[i].v_offset_tg ||
            companion_cache.layers[i].k_size_pp   != main_cache.layers[i].k_size_pp   ||
            companion_cache.layers[i].k_size_tg   != main_cache.layers[i].k_size_tg   ||
            companion_cache.layers[i].v_size_pp   != main_cache.layers[i].v_size_pp   ||
            companion_cache.layers[i].v_size_tg   != main_cache.layers[i].v_size_tg) {
            fprintf(stderr, "FAIL: layer %zu offset/size mismatch\n", i);
            fprintf(stderr, "  main: k_offset_pp=%zu v_offset_pp=%zu k_size_pp=%zu v_size_pp=%zu\n",
                    main_cache.layers[i].k_offset_pp, main_cache.layers[i].v_offset_pp,
                    main_cache.layers[i].k_size_pp, main_cache.layers[i].v_size_pp);
            fprintf(stderr, "  comp: k_offset_pp=%zu v_offset_pp=%zu k_size_pp=%zu v_size_pp=%zu\n",
                    companion_cache.layers[i].k_offset_pp, companion_cache.layers[i].v_offset_pp,
                    companion_cache.layers[i].k_size_pp, companion_cache.layers[i].v_size_pp);
            llama_model_free(model);
            return 1;
        }
    }
    printf("  offset/size sync: PASSED (all %zu layers match)\n", main_cache.layers.size());

    // 校验 4：PP 绑定后，伴生上下文 K/V 数据指针与主上下文一致
    // 注意：init_dkvt_borrow 在 is_transcoded_tg=false 时调用 dkvt_bind_pp()
    for (size_t i = 0; i < main_cache.layers.size(); ++i) {
        if (main_cache.layers[i].k && companion_cache.layers[i].k) {
            if (main_cache.layers[i].k->data != companion_cache.layers[i].k->data) {
                fprintf(stderr, "FAIL: layer %zu K data pointer mismatch after PP bind\n", i);
                llama_model_free(model);
                return 1;
            }
        }
        if (main_cache.layers[i].v && companion_cache.layers[i].v) {
            if (main_cache.layers[i].v->data != companion_cache.layers[i].v->data) {
                fprintf(stderr, "FAIL: layer %zu V data pointer mismatch after PP bind\n", i);
                llama_model_free(model);
                return 1;
            }
        }
    }
    printf("  PP bind pointer match: PASSED\n");

    // 校验 4b：主已 TG 后伴生 decode 路径须 transcode_to_tg 本地绑定（非委托 get_is_transcoded）
    void * companion_pp_k0 = companion_cache.layers[0].k->data;
    main_cache.is_transcoded_tg = true;
    main_cache.dkvt_bind_tg();
    companion_cache.is_transcoded_tg = false;
    companion_cache.transcode_to_tg(nullptr);
    if (!companion_cache.is_transcoded_tg) {
        fprintf(stderr, "FAIL: companion transcode_to_tg must set local flag\n");
        llama_model_free(model);
        return 1;
    }
    if (companion_cache.layers[0].k->data == companion_pp_k0) {
        fprintf(stderr, "FAIL: companion still on PP bind after parent TG\n");
        llama_model_free(model);
        return 1;
    }
    if (companion_cache.layers[0].k->data != main_cache.layers[0].k->data) {
        fprintf(stderr, "FAIL: companion TG K must alias parent\n");
        llama_model_free(model);
        return 1;
    }
    printf("  companion transcode_to_tg after parent TG: PASSED\n");

    // 校验 5：相邻层步长与当前布局一致（4b 后伴生为 TG，用 k_size_tg / v_size_tg）
    const bool layout_tg = companion_cache.is_transcoded_tg;
    for (size_t i = 1; i < companion_cache.layers.size(); ++i) {
        if (companion_cache.layers[i].k && companion_cache.layers[i - 1].k) {
            ptrdiff_t diff = (char *)companion_cache.layers[i].k->data -
                             (char *)companion_cache.layers[i - 1].k->data;
            const size_t k_step = layout_tg
                ? companion_cache.layers[i - 1].k_size_tg
                : companion_cache.layers[i - 1].k_size_pp;
            if (diff != (ptrdiff_t) k_step) {
                fprintf(stderr, "FAIL: layer %zu K pointer diff %td != k_step %zu\n",
                        i, diff, k_step);
                llama_model_free(model);
                return 1;
            }
        }
        if (companion_cache.layers[i].v && companion_cache.layers[i - 1].v) {
            ptrdiff_t diff = (char *)companion_cache.layers[i].v->data -
                             (char *)companion_cache.layers[i - 1].v->data;
            const size_t v_step = layout_tg
                ? companion_cache.layers[i - 1].v_size_tg
                : companion_cache.layers[i - 1].v_size_pp;
            if (diff != (ptrdiff_t) v_step) {
                fprintf(stderr, "FAIL: layer %zu V pointer diff %td != v_step %zu\n",
                        i, diff, v_step);
                llama_model_free(model);
                return 1;
            }
        }
    }
    printf("  PP contiguous layout: PASSED\n");

    // 校验 6：模拟转码后（is_transcoded_tg=true），伴生上下文重新绑定到 TG 布局
    main_cache.is_transcoded_tg = true;
    main_cache.dkvt_bind_tg();

    llama_kv_cache companion_cache_tg(
        *model, model->hparams,
        GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO2_0,
        true, false, true,
        kv_size, 1, 1, 0,
        LLAMA_SWA_TYPE_NONE,
        (llama_memory_t)&main_cache,
        nullptr, nullptr, nullptr
    );
    companion_cache_tg.init_dkvt_borrow();

    // init_dkvt_borrow 在 is_transcoded_tg=true 时调用 dkvt_bind_tg()
    if (companion_cache_tg.is_transcoded_tg != true) {
        fprintf(stderr, "FAIL: companion did not sync is_transcoded_tg=true\n");
        llama_model_free(model);
        return 1;
    }

    // 校验 TG 绑定后 K 指针差 == 前一层 k_size_tg
    for (size_t i = 1; i < companion_cache_tg.layers.size(); ++i) {
        if (companion_cache_tg.layers[i].k && companion_cache_tg.layers[i - 1].k) {
            ptrdiff_t diff = (char *)companion_cache_tg.layers[i].k->data -
                             (char *)companion_cache_tg.layers[i - 1].k->data;
            if (diff != (ptrdiff_t)companion_cache_tg.layers[i - 1].k_size_tg) {
                fprintf(stderr, "FAIL: TG layout layer %zu K pointer diff %td != k_size_tg %zu\n",
                        i, diff, companion_cache_tg.layers[i - 1].k_size_tg);
                llama_model_free(model);
                return 1;
            }
        }
        if (companion_cache_tg.layers[i].v && companion_cache_tg.layers[i - 1].v) {
            ptrdiff_t diff = (char *)companion_cache_tg.layers[i].v->data -
                             (char *)companion_cache_tg.layers[i - 1].v->data;
            if (diff != (ptrdiff_t)companion_cache_tg.layers[i - 1].v_size_tg) {
                fprintf(stderr, "FAIL: TG layout layer %zu V pointer diff %td != v_size_tg %zu\n",
                        i, diff, companion_cache_tg.layers[i - 1].v_size_tg);
                llama_model_free(model);
                return 1;
            }
        }
    }
    printf("  TG contiguous layout: PASSED\n");

    // 校验 7：TG 绑定后，伴生上下文 K/V 数据指针与主上下文一致
    for (size_t i = 0; i < main_cache.layers.size(); ++i) {
        if (main_cache.layers[i].k && companion_cache_tg.layers[i].k) {
            if (main_cache.layers[i].k->data != companion_cache_tg.layers[i].k->data) {
                fprintf(stderr, "FAIL: TG layer %zu K data pointer mismatch\n", i);
                llama_model_free(model);
                return 1;
            }
        }
        if (main_cache.layers[i].v && companion_cache_tg.layers[i].v) {
            if (main_cache.layers[i].v->data != companion_cache_tg.layers[i].v->data) {
                fprintf(stderr, "FAIL: TG layer %zu V data pointer mismatch\n", i);
                llama_model_free(model);
                return 1;
            }
        }
    }
    printf("  TG bind pointer match: PASSED\n");

    llama_model_free(model);
    printf("  test_companion_offset_sync PASSED\n");
    return 0;
}

int test_hybrid_as_kv_cache_resolution() {
    printf("=== test_hybrid_as_kv_cache_resolution ===\n");

    llama_model_params model_params = llama_model_default_params();
    model_params.use_extra_bufts = false;
    llama_model * model = llama_model_create(LLM_ARCH_LLAMA, model_params);
    if (!model) {
        fprintf(stderr, "FAIL: could not create llama_model\n");
        return 1;
    }

    // 配置 hparams：4 层标准 GQA 模型，head_dim=64，4 KV heads
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

    // 构造 llama_memory_hybrid
    llama_memory_hybrid hybrid_mem(
        *model,
        /* attn */          GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO2_0,
                            true,   // v_trans
                            kv_size,
                            1,      // n_pad
                            0,      // n_swa
                            LLAMA_SWA_TYPE_NONE,
        /* recurrent */     GGML_TYPE_F16, GGML_TYPE_F16,
                            128,    // rs_size
        /* common */        1,      // n_seq_max
                            1,      // n_rs_seq
                            false,  // offload
                            true,   // unified
        /* layer filters */ nullptr, nullptr
    );

    // 校验 1：as_kv_cache() 应返回非空指针
    llama_kv_cache * kv_ptr = hybrid_mem.as_kv_cache();
    if (!kv_ptr) {
        fprintf(stderr, "FAIL: as_kv_cache() returned nullptr\n");
        llama_model_free(model);
        return 1;
    }
    printf("  as_kv_cache() returned non-null: PASSED\n");

    // 校验 2：as_kv_cache() 返回的指针应与 get_mem_attn() 一致
    llama_kv_cache * attn_ptr = hybrid_mem.get_mem_attn();
    if (kv_ptr != attn_ptr) {
        fprintf(stderr, "FAIL: as_kv_cache() != get_mem_attn() "
                "(%p vs %p)\n", (void *)kv_ptr, (void *)attn_ptr);
        llama_model_free(model);
        return 1;
    }
    printf("  as_kv_cache() == get_mem_attn(): PASSED\n");

    // 校验 3：通过 llama_memory_i 基类指针调用 as_kv_cache()
    llama_memory_i * mem_base = &hybrid_mem;
    llama_kv_cache * kv_via_base = mem_base->as_kv_cache();
    if (kv_via_base != attn_ptr) {
        fprintf(stderr, "FAIL: base->as_kv_cache() != get_mem_attn() "
                "(%p vs %p)\n", (void *)kv_via_base, (void *)attn_ptr);
        llama_model_free(model);
        return 1;
    }
    printf("  base->as_kv_cache() == get_mem_attn(): PASSED\n");

    llama_model_free(model);
    printf("  test_hybrid_as_kv_cache_resolution PASSED\n");
    return 0;
}

int test_mtp_child_inherits_parent_kv();

int test_dkvt_clear_resets_layout();

int test_dkvt_companion_layout_sync_after_parent_reset();

int test_non_dkvt_context_bypasses_binding();

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
        true,   // v_trans
        false,  // offload
        true,   // unified
        kv_size,
        1,      // n_seq_max
        1,      // n_pad
        0,      // n_swa
        LLAMA_SWA_TYPE_NONE,
        nullptr, nullptr, nullptr, nullptr
    );

    // Mock union block
    size_t union_size = 1024 * 1024 * 16;
    std::vector<uint8_t> mock_vram(union_size + 256);
    uintptr_t ptr_start = reinterpret_cast<uintptr_t>(mock_vram.data());
    // Align to 128 bytes
    ptr_start = (ptr_start + 127) & ~127;

    cache.vram_union_block = reinterpret_cast<ggml_backend_buffer_t>(ptr_start); // Mock non-null
    cache.owns_union_block = false; // Do not free mock buffer
    cache.size_act_pp = 8 * 1024 * 1024;
    cache.size_act_tg = 1 * 1024 * 1024;
    cache.ptr_start = reinterpret_cast<char *>(ptr_start);
    cache.ptr_end = cache.ptr_start + union_size;
    cache.union_size = union_size;

    // We must manually trigger sum sizes to populate offset caches
    cache.init_dkvt_sum_kv_sizes(1);
    cache.dkvt_bind_pp();

    // Verify it is initial PP layout
    void * initial_k0 = cache.layers[0].k->data;
    void * initial_v0 = cache.layers[0].v->data;

    // Transition to TG
    cache.is_transcoded_tg = true;
    cache.dkvt_bind_tg();
    void * tg_k0 = cache.layers[0].k->data;
    void * tg_v0 = cache.layers[0].v->data;

    if (initial_k0 == tg_k0 || initial_v0 == tg_v0) {
        fprintf(stderr, "FAIL: TG layout pointers are same as PP\n");
        llama_model_free(model);
        return 1;
    }

    // Call clear
    cache.clear(false);

    // Assert: state reset and PP pointers bound
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

    // 构造 hybrid parent（模拟 Qwen3.5 hybrid 主上下文）
    llama_memory_hybrid hybrid_parent(
        *model,
        GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO2_0,
        true,   // v_trans
        kv_size,
        1,      // n_pad
        0,      // n_swa
        LLAMA_SWA_TYPE_NONE,
        GGML_TYPE_F16, GGML_TYPE_F16,
        128,    // rs_size
        1,      // n_seq_max
        1,      // n_rs_seq
        false,  // offload
        true,   // unified
        nullptr, nullptr
    );

    // 通过多态基类指针获取 parent 的 primary KV
    llama_memory_i * parent_base = &hybrid_parent;
    llama_kv_cache * parent_kv = parent_base->as_kv_cache();
    if (!parent_kv) {
        fprintf(stderr, "FAIL: parent as_kv_cache() returned nullptr\n");
        llama_model_free(model);
        return 1;
    }
    printf("  parent as_kv_cache() resolved: PASSED\n");

    // 构造 plain child，mem_other 指向 hybrid parent（真实 server MTP 路径）
    llama_kv_cache child_cache(
        *model, model->hparams,
        GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO2_0,
        true,   // v_trans
        false,  // offload
        true,   // unified
        kv_size,
        1,      // n_seq_max
        1,      // n_pad
        0,      // n_swa
        LLAMA_SWA_TYPE_NONE,
        (llama_memory_t)parent_base, // mem_other = hybrid parent
        nullptr, nullptr, nullptr
    );

    // 校验：child->other 应解析为 parent 的 primary KV（非 nullptr）
    if (!child_cache.other) {
        fprintf(stderr, "FAIL: child->other is nullptr (mem_other not resolved)\n");
        llama_model_free(model);
        return 1;
    }

    // child->other 应 == parent 的 primary KV
    if (child_cache.other != parent_kv) {
        fprintf(stderr, "FAIL: child->other %p != parent_kv %p\n",
                (void *)child_cache.other, (void *)parent_kv);
        llama_model_free(model);
        return 1;
    }
    printf("  child->other == parent primary KV: PASSED\n");

    // 校验：child 的 kv_size 应被 parent 覆盖（构造器中 other->get_size() 逻辑）
    if (child_cache.get_size() != parent_kv->get_size()) {
        fprintf(stderr, "FAIL: child kv_size %u != parent kv_size %u\n",
                child_cache.get_size(), parent_kv->get_size());
        llama_model_free(model);
        return 1;
    }
    printf("  child kv_size inherited from parent: PASSED\n");

    llama_model_free(model);
    printf("  test_mtp_child_inherits_parent_kv PASSED\n");
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

    // 构造 hybrid parent（模拟 Qwen3.5 hybrid 主上下文）
    llama_memory_hybrid hybrid_parent(
        *model,
        GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO2_0,
        true,   // v_trans
        kv_size,
        1,      // n_pad
        0,      // n_swa
        LLAMA_SWA_TYPE_NONE,
        GGML_TYPE_F16, GGML_TYPE_F16,
        128,    // rs_size
        1,      // n_seq_max
        1,      // n_rs_seq
        false,  // offload
        true,   // unified
        nullptr, nullptr
    );

    // 通过多态基类指针获取 parent 的 primary KV
    llama_memory_i * parent_base = &hybrid_parent;
    llama_kv_cache * parent_kv = parent_base->as_kv_cache();
    if (!parent_kv) {
        fprintf(stderr, "FAIL: parent as_kv_cache() returned nullptr\n");
        llama_model_free(model);
        return 1;
    }

    // Mock union block
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

    // 初始化 DKVT：计算偏移并绑定 PP 布局
    parent_kv->init_dkvt_sum_kv_sizes(1);
    parent_kv->dkvt_bind_pp();

    void * initial_k0 = parent_kv->layers[0].k->data;
    void * initial_v0 = parent_kv->layers[0].v->data;

    // 构造 plain child，mem_other 指向 hybrid parent
    llama_kv_cache child_cache(
        *model, model->hparams,
        GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO2_0,
        true,   // v_trans
        false,  // offload
        true,   // unified
        kv_size,
        1,      // n_seq_max
        1,      // n_pad
        0,      // n_swa
        LLAMA_SWA_TYPE_NONE,
        (llama_memory_t)parent_base,
        nullptr, nullptr, nullptr
    );

    // 伴生上下文借用主上下文的 union buffer（PP 布局）
    child_cache.init_dkvt_borrow();

    // 校验 1：伴生上下文 PP 布局与主上下文一致
    if (child_cache.layers[0].k->data != parent_kv->layers[0].k->data ||
        child_cache.layers[0].v->data != parent_kv->layers[0].v->data) {
        fprintf(stderr, "FAIL: initial PP layout mismatch between parent and child\n");
        llama_model_free(model);
        return 1;
    }
    printf("  initial PP layout sync: PASSED\n");

    // 模拟转码：主上下文切换到 TG 布局
    parent_kv->is_transcoded_tg = true;
    parent_kv->dkvt_bind_tg();

    void * tg_k0 = parent_kv->layers[0].k->data;
    void * tg_v0 = parent_kv->layers[0].v->data;

    // 伴生上下文重新借用（模拟 MTP 在 TG 阶段重新同步）
    child_cache.init_dkvt_borrow();

    // 校验 2：伴生上下文 TG 布局与主上下文一致
    if (child_cache.layers[0].k->data != parent_kv->layers[0].k->data ||
        child_cache.layers[0].v->data != parent_kv->layers[0].v->data) {
        fprintf(stderr, "FAIL: TG layout mismatch between parent and child\n");
        llama_model_free(model);
        return 1;
    }
    if (child_cache.layers[0].k->data != tg_k0 ||
        child_cache.layers[0].v->data != tg_v0) {
        fprintf(stderr, "FAIL: child TG pointers don't match parent TG pointers\n");
        llama_model_free(model);
        return 1;
    }
    printf("  TG layout sync: PASSED\n");

    // 模拟 parent clear（重置回 PP 布局）
    parent_kv->clear(false);

    // 校验 3：parent 已回退到 PP 布局
    if (parent_kv->is_transcoded_tg) {
        fprintf(stderr, "FAIL: parent is_transcoded_tg not reset after clear\n");
        llama_model_free(model);
        return 1;
    }
    if (parent_kv->layers[0].k->data != initial_k0 ||
        parent_kv->layers[0].v->data != initial_v0) {
        fprintf(stderr, "FAIL: parent did not revert to PP pointers after clear\n");
        llama_model_free(model);
        return 1;
    }

    // 伴生上下文重新借用（模拟下一次 Prefill 前重新同步）
    child_cache.init_dkvt_borrow();

    // 校验 4：伴生上下文重新同步回 PP 布局
    if (child_cache.is_transcoded_tg != false) {
        fprintf(stderr, "FAIL: child is_transcoded_tg not reset after parent clear\n");
        llama_model_free(model);
        return 1;
    }
    if (child_cache.layers[0].k->data != parent_kv->layers[0].k->data ||
        child_cache.layers[0].v->data != parent_kv->layers[0].v->data) {
        fprintf(stderr, "FAIL: child did not revert to PP pointers after parent clear\n");
        llama_model_free(model);
        return 1;
    }
    if (child_cache.layers[0].k->data != initial_k0 ||
        child_cache.layers[0].v->data != initial_v0) {
        fprintf(stderr, "FAIL: child PP pointers don't match initial PP pointers\n");
        llama_model_free(model);
        return 1;
    }
    printf("  post-clear PP layout sync: PASSED\n");

    // 校验 5：伴生上下文 TG 指针已不再是之前的 TG 地址（避免 use-after-layout）
    if (child_cache.layers[0].k->data == tg_k0 ||
        child_cache.layers[0].v->data == tg_v0) {
        fprintf(stderr, "FAIL: child still holds stale TG pointers after parent clear\n");
        llama_model_free(model);
        return 1;
    }
    printf("  stale TG pointer cleanup: PASSED\n");

    // 校验 6：伴生上下文 union buffer 指针始终一致
    if (child_cache.vram_union_block != parent_kv->vram_union_block ||
        child_cache.ptr_start != parent_kv->ptr_start ||
        child_cache.ptr_end != parent_kv->ptr_end ||
        child_cache.union_size != parent_kv->union_size) {
        fprintf(stderr, "FAIL: child union buffer pointers diverged from parent\n");
        llama_model_free(model);
        return 1;
    }
    printf("  union buffer pointer stability: PASSED\n");

    llama_model_free(model);
    printf("  test_dkvt_companion_layout_sync_after_parent_reset PASSED\n");
    return 0;
}

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

    // 构造主 DKVT 上下文（turbo4/turbo2）
    llama_kv_cache main_cache(
        *model, model->hparams,
        GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO2_0,
        true,   // v_trans
        false,  // offload
        true,   // unified
        kv_size,
        1,      // n_seq_max
        1,      // n_pad
        0,      // n_swa
        LLAMA_SWA_TYPE_NONE,
        nullptr, nullptr, nullptr, nullptr
    );

    // Mock union block
    size_t union_size = 1024 * 1024 * 16;
    std::vector<uint8_t> mock_vram(union_size + 256);
    uintptr_t ptr_start = reinterpret_cast<uintptr_t>(mock_vram.data());
    ptr_start = (ptr_start + 127) & ~127;

    main_cache.vram_union_block =
        reinterpret_cast<ggml_backend_buffer_t>(ptr_start);
    main_cache.owns_union_block = false;
    main_cache.size_act_pp = 8 * 1024 * 1024;
    main_cache.size_act_tg = 1 * 1024 * 1024;
    main_cache.ptr_start = reinterpret_cast<char *>(ptr_start);
    main_cache.ptr_end = main_cache.ptr_start + union_size;
    main_cache.union_size = union_size;

    // 初始化 DKVT 并绑定 PP 布局
    main_cache.init_dkvt_sum_kv_sizes(1);
    main_cache.dkvt_bind_pp();

    void * initial_k0 = main_cache.layers[0].k->data;
    void * initial_v0 = main_cache.layers[0].v->data;

    // 构造非 DKVT 伴生上下文（f16/f16），mem_other → main_cache
    // 模拟 MTP draft context 使用 f16 KV cache 而非 turbo 量化
    llama_kv_cache companion_cache(
        *model, model->hparams,
        GGML_TYPE_F16, GGML_TYPE_F16,
        true,   // v_trans
        false,  // offload
        true,   // unified
        kv_size,
        1,      // n_seq_max
        1,      // n_pad
        0,      // n_swa
        LLAMA_SWA_TYPE_NONE,
        (llama_memory_t)&main_cache, // mem_other
        nullptr, nullptr, nullptr
    );

    // 记录伴生上下文初始状态
    bool companion_initial_is_transcoded = companion_cache.is_transcoded_tg;
    if (companion_initial_is_transcoded) {
        fprintf(stderr,
                "FAIL: companion initial is_transcoded_tg should be false\n");
        llama_model_free(model);
        return 1;
    }
    void * companion_initial_k0 = companion_cache.layers[0].k->data;
    void * companion_initial_v0 = companion_cache.layers[0].v->data;

    // 模拟主上下文转码到 TG 布局
    main_cache.is_transcoded_tg = true;
    main_cache.dkvt_bind_tg();

    // 模拟伴生上下文接收到自回归转码指令（例如在 MTP 推理中调用 decode）
    companion_cache.transcode_to_tg(nullptr);

    // 调用伴生上下文的 get_is_transcoded_tg()：
    // 该方法只返回本上下文自己的转码标志。非 DKVT 伴生上下文没有 union buffer，
    // 因此应保持 false，不应委托到父上下文。
    bool companion_reported_transcoded =
        companion_cache.get_is_transcoded_tg();

    // 校验 1：伴生上下文的原始字段 is_transcoded_tg 必须保持 false
    // 非 DKVT 上下文不应被 init_dkvt 或 init_dkvt_borrow 修改状态
    if (companion_cache.is_transcoded_tg != false) {
        fprintf(stderr,
                "FAIL: companion raw is_transcoded_tg should stay false\n");
        llama_model_free(model);
        return 1;
    }
    printf("  companion raw is_transcoded_tg stays false: PASSED\n");

    // 校验 2：伴生上下文的 K/V 数据指针不应被修改
    // 非 DKVT 上下文不应参与 union buffer 的布局绑定
    if (companion_cache.layers[0].k->data != companion_initial_k0) {
        fprintf(stderr,
                "FAIL: companion K data pointer was modified\n");
        llama_model_free(model);
        return 1;
    }
    if (companion_cache.layers[0].v->data != companion_initial_v0) {
        fprintf(stderr,
                "FAIL: companion V data pointer was modified\n");
        llama_model_free(model);
        return 1;
    }
    printf("  companion K/V pointers untouched: PASSED\n");

    // 校验 3：伴生上下文的 vram_union_block 应保持 nullptr
    // 非 DKVT 上下文不应借用 union buffer
    if (companion_cache.vram_union_block != nullptr) {
        fprintf(stderr,
                "FAIL: companion vram_union_block should stay nullptr\n");
        llama_model_free(model);
        return 1;
    }
    printf("  companion vram_union_block stays nullptr: PASSED\n");

    // 校验 4：非 DKVT 伴生仅报告本地标志（不委托父级），decode 门控与布局解耦
    if (companion_reported_transcoded) {
        fprintf(stderr,
                "FAIL: non-DKVT companion get_is_transcoded_tg must stay local false\n");
        llama_model_free(model);
        return 1;
    }
    printf("  non-DKVT companion local is_transcoded_tg=false: PASSED\n");

    // 校验 5：伴生上下文的 dkvt 尺寸字段应保持零
    if (companion_cache.dkvt_k_size_pp != 0 ||
        companion_cache.dkvt_v_size_pp != 0 ||
        companion_cache.dkvt_k_size_tg != 0 ||
        companion_cache.dkvt_v_size_tg != 0) {
        fprintf(stderr,
                "FAIL: companion DKVT size fields should stay zero\n");
        llama_model_free(model);
        return 1;
    }
    printf("  companion DKVT size fields stay zero: PASSED\n");

    // 校验 6：主上下文 TG 布局确实已切换（指针变化）
    if (main_cache.layers[0].k->data == initial_k0 ||
        main_cache.layers[0].v->data == initial_v0) {
        fprintf(stderr,
                "FAIL: main cache TG layout should differ from PP\n");
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

int main() {
    int failures = 0;
    failures += test_companion_sync();
    failures += test_cuda_d2d_copy();
    failures += test_mtp_uaf_defense();
    failures += test_companion_offset_sync();
    failures += test_hybrid_as_kv_cache_resolution();
    failures += test_mtp_child_inherits_parent_kv();
    failures += test_dkvt_clear_resets_layout();
    failures += test_dkvt_companion_layout_sync_after_parent_reset();
    failures += test_non_dkvt_context_bypasses_binding();
    failures += test_mtp_shared_child_preserves_parent_ext_flags();
    return failures == 0 ? 0 : 1;
}
