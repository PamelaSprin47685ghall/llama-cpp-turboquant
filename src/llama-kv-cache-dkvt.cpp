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

void llama_kv_cache::dkvt_bind_pp() {
    if (!vram_union_block || !ptr_start) return;

    // 正向连续布局：V 段在低地址，K 段在高地址，两段均从段首向段尾正向生长
    // V 段基址 = ptr_start + size_act_pp + layers[i].v_offset_pp
    // K 段基址 = ptr_start + size_act_pp + dkvt_v_size_pp + layers[i].k_offset_pp
    char * kv_base_pp = ptr_start + size_act_pp;

    for (size_t i = 0; i < layers.size(); ++i) {
        if (layers[i].v && (layers[i].v->flags & GGML_TENSOR_FLAG_EXT)) {
            layers[i].v->buffer = vram_union_block;
            layers[i].v->data = kv_base_pp + layers[i].v_offset_pp;
            dkvt_update_strides(layers[i].v, layers[i].orig_type_v);
        }
        if (layers[i].k && (layers[i].k->flags & GGML_TENSOR_FLAG_EXT)) {
            layers[i].k->buffer = vram_union_block;
            layers[i].k->data = kv_base_pp + dkvt_v_size_pp + layers[i].k_offset_pp;
            dkvt_update_strides(layers[i].k, layers[i].orig_type_k);
        }
    }

    for (size_t i = 0; i < layers.size(); ++i) {
        if (layers[i].k && (layers[i].k->flags & GGML_TENSOR_FLAG_EXT)) {
            for (size_t s = 0; s < layers[i].k_stream.size(); ++s) {
                if (layers[i].k_stream[s]) {
                    layers[i].k_stream[s]->buffer = vram_union_block;
                    dkvt_update_strides(layers[i].k_stream[s], layers[i].orig_type_k);
                    layers[i].k_stream[s]->data = (char*)layers[i].k->data + s * layers[i].k->nb[2];
                }
            }
        }
        if (layers[i].v && (layers[i].v->flags & GGML_TENSOR_FLAG_EXT)) {
            for (size_t s = 0; s < layers[i].v_stream.size(); ++s) {
                if (layers[i].v_stream[s]) {
                    layers[i].v_stream[s]->buffer = vram_union_block;
                    dkvt_update_strides(layers[i].v_stream[s], layers[i].orig_type_v);
                    layers[i].v_stream[s]->data = (char*)layers[i].v->data + s * layers[i].v->nb[2];
                }
            }
        }
    }

    // Re-apply PP compute cap (is_transcoded_tg=false → cap = size_act_pp)
    dkvt_apply_union_compute_cap(dkvt_sched);
}

void llama_kv_cache::dkvt_bind_tg() {
    if (!vram_union_block || !ptr_start) return;

    // 正向连续布局：V 段在低地址，K 段在高地址，两段均从段首向段尾正向生长
    // V 段基址 = ptr_start + size_act_tg + layers[i].v_offset_tg
    // K 段基址 = ptr_start + size_act_tg + dkvt_v_size_tg + layers[i].k_offset_tg
    char * kv_base_tg = ptr_start + size_act_tg;

    for (size_t i = 0; i < layers.size(); ++i) {
        ggml_type target_type_k = dkvt_tg_type_k(layers[i].orig_type_k);
        ggml_type target_type_v = dkvt_tg_type_v(layers[i].orig_type_v);

        if (layers[i].v && (layers[i].v->flags & GGML_TENSOR_FLAG_EXT)) {
            layers[i].v->buffer = vram_union_block;
            layers[i].v->data = kv_base_tg + layers[i].v_offset_tg;
            dkvt_update_strides(layers[i].v, target_type_v);
        }
        if (layers[i].k && (layers[i].k->flags & GGML_TENSOR_FLAG_EXT)) {
            layers[i].k->buffer = vram_union_block;
            layers[i].k->data = kv_base_tg + dkvt_v_size_tg + layers[i].k_offset_tg;
            dkvt_update_strides(layers[i].k, target_type_k);
        }
    }

    for (size_t i = 0; i < layers.size(); ++i) {
        ggml_type target_type_k = dkvt_tg_type_k(layers[i].orig_type_k);
        ggml_type target_type_v = dkvt_tg_type_v(layers[i].orig_type_v);

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

    // Re-apply TG compute cap (is_transcoded_tg=true → cap = size_act_tg)
    dkvt_apply_union_compute_cap(dkvt_sched);
}

void llama_kv_cache::init_dkvt_borrow() {
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

bool llama_kv_cache::init_dkvt_find_backend(ggml_backend_sched_t sched, ggml_backend_buffer_type_t & buft, int & buffer_id) {
    int n_backends = ggml_backend_sched_get_n_backends(sched);
    for (int i = 0; i < n_backends; ++i) {
        ggml_backend_t backend = ggml_backend_sched_get_backend(sched, i);
        if (backend) {
            ggml_backend_dev_t dev = ggml_backend_get_device(backend);
            if (dev) {
                enum ggml_backend_dev_type dev_type = ggml_backend_dev_type(dev);
                if (dev_type == GGML_BACKEND_DEVICE_TYPE_GPU || dev_type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                    buft = ggml_backend_sched_get_buffer_type(sched, backend);
                    buffer_id = i;
                    return true;
                }
            }
        }
    }
    return false;
}

void llama_kv_cache::init_dkvt_sum_kv_sizes(size_t gpu_alignment) {
    dkvt_k_size_pp = 0;
    dkvt_k_size_tg = 0;
    dkvt_v_size_pp = 0;
    dkvt_v_size_tg = 0;

    size_t v_off_pp = 0;
    size_t k_off_pp = 0;
    size_t v_off_tg = 0;
    size_t k_off_tg = 0;

    for (size_t i = 0; i < layers.size(); ++i) {
        if (layers[i].v) {
            layers[i].v_size_pp = ggml_row_size(layers[i].orig_type_v, layers[i].v->ne[0])
                               * layers[i].v->ne[1] * layers[i].v->ne[2] * layers[i].v->ne[3];
            v_off_pp = dkvt_align_up(v_off_pp, gpu_alignment);
            layers[i].v_offset_pp = v_off_pp;
            v_off_pp += layers[i].v_size_pp;

            ggml_type target_type_v = dkvt_tg_type_v(layers[i].orig_type_v);
            layers[i].v_size_tg = ggml_row_size(target_type_v, layers[i].v->ne[0])
                                * layers[i].v->ne[1] * layers[i].v->ne[2] * layers[i].v->ne[3];
            v_off_tg = dkvt_align_up(v_off_tg, gpu_alignment);
            layers[i].v_offset_tg = v_off_tg;
            v_off_tg += layers[i].v_size_tg;
        }
        if (layers[i].k) {
            layers[i].k_size_pp = ggml_row_size(layers[i].orig_type_k, layers[i].k->ne[0])
                               * layers[i].k->ne[1] * layers[i].k->ne[2] * layers[i].k->ne[3];
            k_off_pp = dkvt_align_up(k_off_pp, gpu_alignment);
            layers[i].k_offset_pp = k_off_pp;
            k_off_pp += layers[i].k_size_pp;

            ggml_type target_type_k = dkvt_tg_type_k(layers[i].orig_type_k);
            layers[i].k_size_tg = ggml_row_size(target_type_k, layers[i].k->ne[0])
                                * layers[i].k->ne[1] * layers[i].k->ne[2] * layers[i].k->ne[3];
            k_off_tg = dkvt_align_up(k_off_tg, gpu_alignment);
            layers[i].k_offset_tg = k_off_tg;
            k_off_tg += layers[i].k_size_tg;
        }
        dkvt_v_size_pp += layers[i].v_size_pp;
        dkvt_v_size_tg += layers[i].v_size_tg;
        dkvt_k_size_pp += layers[i].k_size_pp;
        dkvt_k_size_tg += layers[i].k_size_tg;
    }

    dkvt_v_size_pp = v_off_pp;
    dkvt_k_size_pp = k_off_pp;
    dkvt_v_size_tg = v_off_tg;
    dkvt_k_size_tg = k_off_tg;
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
    LLAMA_LOG_INFO("llama_kv_cache: init_dkvt: other=%p, layers=%zu, n_seq_max=%u\n", (void*)other, layers.size(), n_seq_max);
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

    // 伴生上下文直接借用主上下文的 union buffer，无需独立分配。
    // 必须使用伴生上下文自己的调度器（由调用者传入），不能复用主上下文调度器，
    // 否则 compute cap 会错误地设置到主调度器上，导致伴生图可能覆盖 union buffer 中的 KV 数据。
    dkvt_sched = sched;
    if (other) {
        init_dkvt_borrow();
        return;
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

void llama_kv_cache::init_dkvt_compute_activation_size(size_t n_ubatch) {
    // Compute activation buffer sizes from model geometry. The per-token activation
    // high water (float32) is dominated by:
    //   1. Q projection output:    n_embd
    //   2. K projection output:    n_embd_k_gqa
    //   3. V projection output:    n_embd_v_gqa
    //   4. O projection input:     n_embd
    //   5. Attention score:      n_head * n_tokens
    //   6-8. FFN gate/up/down:   3 * ffn_dim (MoE: n_expert_used * n_ff_exp + n_ff_shexp)
    //
    // Take the maximum over all layers to produce a safe upper bound for
    // heterogeneous models without hard-coding a representative layer index.

    size_t n_embd = hparams.n_embd;
    size_t n_head = 0;
    size_t n_embd_k_gqa = 0;
    size_t n_embd_v_gqa = 0;
    size_t ffn_dim = 0;
    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        n_head = std::max(n_head, (size_t) hparams.n_head(il));
        n_embd_k_gqa = std::max(n_embd_k_gqa, (size_t) hparams.n_embd_k_gqa(il));
        n_embd_v_gqa = std::max(n_embd_v_gqa, (size_t) hparams.n_embd_v_gqa(il));
        const size_t n_ff_exp_eff = hparams.n_ff_exp
            ? hparams.n_ff_exp
            : (hparams.n_expert_used > 0
                ? hparams.n_ff(il) / hparams.n_expert_used
                : hparams.n_ff(il));
        const size_t n_ff_shexp_eff = hparams.n_ff_shexp ? hparams.n_ff_shexp : hparams.n_ff(il);
        const size_t ffn_dim_l = (hparams.n_expert > 0)
            ? (hparams.n_expert_used * n_ff_exp_eff + n_ff_shexp_eff)
            : hparams.n_ff(il);
        ffn_dim = std::max(ffn_dim, ffn_dim_l);
    }

    size_t n_recr_layers = 0;
    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        if (hparams.is_recr(il)) ++n_recr_layers;
    }

    const size_t n_expert_used_eff = hparams.n_expert > 0 ? hparams.n_expert_used : 0;
    const size_t n_kv_cells = get_size();

    auto gdn_conv_output_bytes = [&](size_t n_tokens) -> size_t {
        if (n_recr_layers == 0 || hparams.ssm_d_inner == 0) {
            return 0;
        }
        const size_t conv_channels = hparams.ssm_d_inner
            + 2u * hparams.ssm_n_group * hparams.ssm_d_state;
        const size_t conv_len = n_tokens + std::max(1u, hparams.ssm_d_conv) - 1;
        return sizeof(float) * conv_len * conv_channels;
    };

    size_act_pp = dkvt_compute_activation_bytes(
        n_ubatch, n_embd, n_head, n_embd_k_gqa, n_embd_v_gqa, ffn_dim,
        n_recr_layers, n_expert_used_eff, n_kv_cells, true, gdn_conv_output_bytes(n_ubatch),
        /*n_vocab=*/ 0);

    // TG phase: decode batch is at most one token (or a small speculative-verify batch).
    // Use the same geometry-driven helper so GDN conv output and Flash-Attention
    // scratch are accounted for; do not use the full n_kv as the score length
    // because that would balloon the union buffer into multiple gigabytes.
    // Include vocab size because decode emits output logits and the verify batch
    // can output up to DKVT_SPEC_VERIFY_MAX_BATCH positions.
    const size_t n_tokens_tg = std::min<size_t>(n_ubatch, DKVT_SPEC_VERIFY_MAX_BATCH);
    size_act_tg = dkvt_compute_activation_bytes(
        n_tokens_tg, n_embd, n_head, n_embd_k_gqa, n_embd_v_gqa, ffn_dim,
        n_recr_layers, n_expert_used_eff, n_kv_cells, true, gdn_conv_output_bytes(n_tokens_tg),
        (size_t) model.vocab.n_tokens());
}

void llama_kv_cache::disable_dkvt_ext_flags() {
    for (size_t i = 0; i < layers.size(); ++i) {
        // 共享层检测：若 other 存在且当前层的 k/v 指针与 other 中任意层相同，
        // 则该层为指针别名（共享 tensor），清除 EXT 会破坏父上下文的标志位。
        bool is_shared = false;
        if (other) {
            for (size_t j = 0; j < other->layers.size(); ++j) {
                if ((layers[i].k && layers[i].k == other->layers[j].k) ||
                    (layers[i].v && layers[i].v == other->layers[j].v)) {
                    is_shared = true;
                    break;
                }
            }
        }
        if (is_shared) continue;

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
