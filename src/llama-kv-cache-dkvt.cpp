#include "llama-kv-cache.h"
#include "llama-kv-cache-dkvt.h"
#include "llama-impl.h"
#include "llama-context.h"
#include "llama-model.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "../ggml/src/ggml-quants.h"
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cstring>

#ifdef GGML_USE_CUDA
extern "C" bool ggml_cuda_get_mem_info(size_t * free, size_t * total);
#endif

void llama_kv_cache::dkvt_bind_common(size_t size_act, size_t dkvt_v_size, bool use_tg_type) {
    if (!vram_union_block || !ptr_start) return;

    char * kv_base = ptr_start + size_act;

    for (size_t i = 0; i < layers.size(); ++i) {
        ggml_type target_type_k = use_tg_type ? dkvt_tg_type_k(layers[i].orig_type_k) : layers[i].orig_type_k;
        ggml_type target_type_v = use_tg_type ? dkvt_tg_type_v(layers[i].orig_type_v) : layers[i].orig_type_v;
        size_t k_offset = use_tg_type ? layers[i].k_offset_tg : layers[i].k_offset_pp;
        size_t v_offset = use_tg_type ? layers[i].v_offset_tg : layers[i].v_offset_pp;

        if (layers[i].v && (layers[i].v->flags & GGML_TENSOR_FLAG_EXT)) {
            layers[i].v->buffer = vram_union_block;
            layers[i].v->data = kv_base + v_offset;
            dkvt_update_strides(layers[i].v, target_type_v);
        }
        if (layers[i].k && (layers[i].k->flags & GGML_TENSOR_FLAG_EXT)) {
            layers[i].k->buffer = vram_union_block;
            layers[i].k->data = kv_base + dkvt_v_size + k_offset;
            dkvt_update_strides(layers[i].k, target_type_k);
        }
    }

    for (size_t i = 0; i < layers.size(); ++i) {
        ggml_type target_type_k = use_tg_type ? dkvt_tg_type_k(layers[i].orig_type_k) : layers[i].orig_type_k;
        ggml_type target_type_v = use_tg_type ? dkvt_tg_type_v(layers[i].orig_type_v) : layers[i].orig_type_v;

        if (layers[i].k && (layers[i].k->flags & GGML_TENSOR_FLAG_EXT)) {
            for (size_t s = 0; s < layers[i].k_stream.size(); ++s) {
                if (layers[i].k_stream[s]) {
                    layers[i].k_stream[s]->buffer = vram_union_block;
                    dkvt_update_strides(layers[i].k_stream[s], target_type_k);
                    layers[i].k_stream[s]->data = (char*)layers[i].k->data + s * layers[i].k->nb[2];
                }
            }
        }
        if (layers[i].v && (layers[i].v->flags & GGML_TENSOR_FLAG_EXT)) {
            for (size_t s = 0; s < layers[i].v_stream.size(); ++s) {
                if (layers[i].v_stream[s]) {
                    layers[i].v_stream[s]->buffer = vram_union_block;
                    dkvt_update_strides(layers[i].v_stream[s], target_type_v);
                    layers[i].v_stream[s]->data = (char*)layers[i].v->data + s * layers[i].v->nb[2];
                }
            }
        }
    }

    dkvt_apply_union_compute_cap(dkvt_sched);
}

void llama_kv_cache::dkvt_bind_pp() {
    dkvt_bind_common(size_act_pp, dkvt_v_size_pp, false);
}

void llama_kv_cache::dkvt_bind_tg() {
    dkvt_bind_common(size_act_tg, dkvt_v_size_tg, true);
}

void llama_kv_cache::init_dkvt_borrow() {
    bool has_transcodable = false;
    for (size_t i = 0; i < layers.size(); ++i) {
        if ((layers[i].k && is_transcodable_type(layers[i].orig_type_k) && (layers[i].k->flags & GGML_TENSOR_FLAG_EXT)) ||
            (layers[i].v && is_transcodable_type(layers[i].orig_type_v) && (layers[i].v->flags & GGML_TENSOR_FLAG_EXT))) {
            has_transcodable = true;
            break;
        }
    }
    if (!has_transcodable) {
        return;
    }
    GGML_ASSERT(other->vram_union_block != nullptr && "Main context must be initialized before draft context (main-first).");

    vram_union_block = other->vram_union_block;
    ptr_start = other->ptr_start;
    ptr_end = other->ptr_end;
    union_size = other->union_size;
    dkvt_k_size_pp = other->dkvt_k_size_pp;
    dkvt_k_size_tg = other->dkvt_k_size_tg;
    dkvt_v_size_pp = other->dkvt_v_size_pp;
    dkvt_v_size_tg = other->dkvt_v_size_tg;
    is_transcoded_tg = other->is_transcoded_tg;
    size_act_pp = other->size_act_pp;
    size_act_tg = other->size_act_tg;
    dkvt_sched_buffer_id = other->dkvt_sched_buffer_id;

    // 逐层同步偏移和尺寸，确保伴生上下文与主上下文共享层布局完全一致。
    for (size_t i = 0; i < layers.size(); ++i) {
        int32_t il = layers[i].il;
        if (other->map_layer_ids.count(il) > 0) {
            int32_t other_i = other->map_layer_ids.at(il);
            layers[i].k_offset_pp = other->layers[other_i].k_offset_pp;
            layers[i].k_offset_tg = other->layers[other_i].k_offset_tg;
            layers[i].v_offset_pp = other->layers[other_i].v_offset_pp;
            layers[i].v_offset_tg = other->layers[other_i].v_offset_tg;
            layers[i].k_size_pp   = other->layers[other_i].k_size_pp;
            layers[i].k_size_tg   = other->layers[other_i].k_size_tg;
            layers[i].v_size_pp   = other->layers[other_i].v_size_pp;
            layers[i].v_size_tg   = other->layers[other_i].v_size_tg;
        } else {
            // Independent layer: clear EXT flag so allocator will allocate it
            if (layers[i].k) {
                layers[i].k->flags &= ~GGML_TENSOR_FLAG_EXT;
            }
            if (layers[i].v) {
                layers[i].v->flags &= ~GGML_TENSOR_FLAG_EXT;
            }
            for (size_t s = 0; s < layers[i].k_stream.size(); ++s) {
                if (layers[i].k_stream[s]) {
                    layers[i].k_stream[s]->flags &= ~GGML_TENSOR_FLAG_EXT;
                }
            }
            for (size_t s = 0; s < layers[i].v_stream.size(); ++s) {
                if (layers[i].v_stream[s]) {
                    layers[i].v_stream[s]->flags &= ~GGML_TENSOR_FLAG_EXT;
                }
            }
        }
    }

    // 根据主上下文的转码状态选择正确的绑定方式，确保伴生上下文
    // 与主上下文处于相同的布局（PP 或 TG），避免地址错位。
    // 注意：is_transcoded_tg 的同步必须在绑定之前完成，因为 clear() 会
    // 重置 is_transcoded_tg = false，若先绑定再同步则状态丢失。
    if (is_transcoded_tg) {
        dkvt_bind_tg();
    } else {
        dkvt_bind_pp();
    }

    LLAMA_LOG_INFO("llama_kv_cache: DKVT draft context borrowed union buffer from parent (%p, %.2f MiB), is_transcoded_tg=%d\n",
                   (void*)vram_union_block, (float)union_size / 1024.0f / 1024.0f, is_transcoded_tg);
}

bool llama_kv_cache::init_dkvt_alloc(ggml_backend_buffer_type_t buft) {
#ifdef GGML_USE_CUDA
    size_t free_before = 0, total_before = 0;
    ggml_cuda_get_mem_info(&free_before, &total_before);
    LLAMA_LOG_INFO("llama_kv_cache: [DKVT-ALLOC] before alloc: union_size=%.2f MiB, free=%zu MiB, total=%zu MiB\n",
                   (float)union_size / 1024.0f / 1024.0f,
                   free_before / 1024 / 1024, total_before / 1024 / 1024);
#endif

    vram_union_block = ggml_backend_buft_alloc_buffer(buft, union_size);
    if (vram_union_block) {
        owns_union_block = true;
        ptr_start = (char *)ggml_backend_buffer_get_base(vram_union_block);
        ptr_end = ptr_start + union_size;
#ifdef GGML_USE_CUDA
        size_t free_after = 0, total_after = 0;
        ggml_cuda_get_mem_info(&free_after, &total_after);
        LLAMA_LOG_INFO("llama_kv_cache: [DKVT-ALLOC] after alloc: free=%zu MiB, total=%zu MiB, consumed=%zu MiB\n",
                       free_after / 1024 / 1024, total_after / 1024 / 1024,
                       (free_before - free_after) / 1024 / 1024);
#endif
        return true;
    }
    return false;
}

void llama_kv_cache::init_dkvt(size_t n_ubatch, ggml_backend_sched_t sched) {
    if (disable_dkvt) {
        disable_dkvt_ext_flags();
        return;
    }
    LLAMA_LOG_INFO("llama_kv_cache: init_dkvt: other=%p, layers=%zu, n_seq_max=%u\n", (void*)other, layers.size(), n_seq_max);
    dkvt_sched = sched;
    if (other) {
        init_dkvt_borrow();
        dkvt_apply_union_compute_cap(sched);
        return;
    }
    if (vram_union_block) {
        dkvt_apply_union_compute_cap(sched);
        return;
    }
    if (!sched) {
        throw std::runtime_error("DKVT required but ggml backend sched is null");
    }

    // 检查是否存在可转码层且包含 EXT 标志：若无任何 EXT 标志则无需 DKVT
    bool has_transcodable = false;
    for (size_t i = 0; i < layers.size(); ++i) {
        if ((layers[i].k && is_transcodable_type(layers[i].orig_type_k) && (layers[i].k->flags & GGML_TENSOR_FLAG_EXT)) ||
            (layers[i].v && is_transcodable_type(layers[i].orig_type_v) && (layers[i].v->flags & GGML_TENSOR_FLAG_EXT))) {
            has_transcodable = true;
            break;
        }
    }
    if (!has_transcodable) return;

    // 只有真正启用 turbo KV cache 时才强制单会话独占
    if (this->n_seq_max > 1) {
        throw std::runtime_error(
            "DKVT with speculative decoding requires n_parallel/n_seq_max = 1.");
    }

    ggml_backend_buffer_type_t buft = nullptr;
    int buffer_id = -1;
    if (!init_dkvt_find_backend(sched, buft, buffer_id)) {
        throw std::runtime_error("DKVT required but no GPU backend in sched");
    }
    dkvt_gpu_alignment = ggml_backend_buft_get_alignment(buft);

    init_dkvt_compute_activation_size(n_ubatch);
    size_act_pp = dkvt_align_up(size_act_pp, dkvt_gpu_alignment);
    size_act_tg = dkvt_align_up(size_act_tg, dkvt_gpu_alignment);
    init_dkvt_sum_kv_sizes(dkvt_gpu_alignment);

    union_size = std::max(
        size_act_pp + dkvt_v_size_pp + dkvt_k_size_pp,
        size_act_tg + dkvt_v_size_tg + dkvt_k_size_tg);
    union_size = GGML_PAD(union_size, dkvt_gpu_alignment);

#ifdef GGML_USE_CUDA
    size_t free_vram = 0, total_vram = 0;
    ggml_cuda_get_mem_info(&free_vram, &total_vram);
    LLAMA_LOG_INFO("llama_kv_cache: [DKVT-DIAG] pp(act=%.2f K=%.2f V=%.2f) tg(act=%.2f K=%.2f V=%.2f) union=%.2f free=%zu total=%zu MiB\n",
                   (float)size_act_pp / 1024 / 1024, (float)dkvt_k_size_pp / 1024 / 1024, (float)dkvt_v_size_pp / 1024 / 1024,
                   (float)size_act_tg / 1024 / 1024, (float)dkvt_k_size_tg / 1024 / 1024, (float)dkvt_v_size_tg / 1024 / 1024,
                   (float)union_size / 1024 / 1024, free_vram / 1024 / 1024, total_vram / 1024 / 1024);
#endif

#ifdef GGML_USE_CUDA
    {
        size_t free_vram = 0, total_vram = 0;
        ggml_cuda_get_mem_info(&free_vram, &total_vram);
        const size_t margin = 128ULL * 1024 * 1024;
        if (free_vram < union_size + margin) {
            throw std::runtime_error(
                "DKVT union requires " + std::to_string((int) (union_size / 1024 / 1024)) +
                " MiB (+128 MiB margin) but only " + std::to_string((int) (free_vram / 1024 / 1024)) +
                " MiB VRAM free (total " + std::to_string((int) (total_vram / 1024 / 1024)) + " MiB)");
        }
    }
#endif

    if (!init_dkvt_alloc(buft)) {
        throw std::runtime_error(
            "DKVT union GPU alloc failed (" + std::to_string((int) (union_size / 1024 / 1024)) + " MiB)");
    }
    init_dkvt_bind_layers(sched, buffer_id);
}
