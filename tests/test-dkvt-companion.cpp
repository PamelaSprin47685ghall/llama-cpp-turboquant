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

int test_companion_offset_sync() {
    printf("=== test_companion_offset_sync ===\n");

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

    printf("  main_cache: layers=%zu, dkvt_k_size_pp=%zu, dkvt_v_size_pp=%zu, "
           "dkvt_k_size_tg=%zu, dkvt_v_size_tg=%zu\n",
            main_cache.layers.size(),
            main_cache.dkvt_k_size_pp, main_cache.dkvt_v_size_pp,
            main_cache.dkvt_k_size_tg, main_cache.dkvt_v_size_tg);

    llama_kv_cache companion_cache(
        *model, model->hparams,
        GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO2_0,
        true, false, true,
        kv_size, 1, 1, 0,
        LLAMA_SWA_TYPE_NONE,
        (llama_memory_t)&main_cache,
        nullptr, nullptr, nullptr
    );

    companion_cache.init_dkvt_borrow();

    if (companion_cache.vram_union_block != main_cache.vram_union_block ||
        companion_cache.ptr_start != main_cache.ptr_start ||
        companion_cache.ptr_end != main_cache.ptr_end ||
        companion_cache.union_size != main_cache.union_size) {
        fprintf(stderr, "FAIL: union buffer pointers mismatch\n");
        llama_model_free(model);
        return 1;
    }

    if (companion_cache.dkvt_k_size_pp != main_cache.dkvt_k_size_pp ||
        companion_cache.dkvt_k_size_tg != main_cache.dkvt_k_size_tg ||
        companion_cache.dkvt_v_size_pp != main_cache.dkvt_v_size_pp ||
        companion_cache.dkvt_v_size_tg != main_cache.dkvt_v_size_tg) {
        fprintf(stderr, "FAIL: total KV size mismatch\n");
        llama_model_free(model);
        return 1;
    }

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
            llama_model_free(model);
            return 1;
        }
    }
    printf("  offset/size sync: PASSED (all %zu layers match)\n", main_cache.layers.size());

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

    void * companion_pp_k0 = companion_cache.layers[0].k->data;
    main_cache.is_transcoded_tg = true;
    main_cache.dkvt_bind_tg();
    companion_cache.is_transcoded_tg = true;
    companion_cache.dkvt_bind_tg();
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

    if (companion_cache_tg.is_transcoded_tg != true) {
        fprintf(stderr, "FAIL: companion did not sync is_transcoded_tg=true\n");
        llama_model_free(model);
        return 1;
    }

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

int main() {
    return test_companion_offset_sync();
}
