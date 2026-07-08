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

    // 正向连续布局：计算图从 ptr_start 正向生长，KV Cache 从 ptr_start + size_act_pp 正向生长
    // V 段基址 = ptr_start + size_act_pp + layers[i].v_offset_pp
    // K 段基址 = ptr_start + size_act_pp + dkvt_v_size_pp + layers[i].k_offset_pp
    char * kv_base_pp = ptr_start + size_act_pp;

    for (size_t i = 0; i < layers.size(); ++i) {
        if (layers[i].v) {
            layers[i].v->buffer = vram_union_block;
            layers[i].v->data = kv_base_pp + layers[i].v_offset_pp;
            dkvt_update_strides(layers[i].v, layers[i].orig_type_v);
        }
        if (layers[i].k) {
            layers[i].k->buffer = vram_union_block;
            layers[i].k->data = kv_base_pp + dkvt_v_size_pp + layers[i].k_offset_pp;
            dkvt_update_strides(layers[i].k, layers[i].orig_type_k);
        }
    }

    for (size_t i = 0; i < layers.size(); ++i) {
        for (size_t s = 0; s < layers[i].k_stream.size(); ++s) {
            if (layers[i].k_stream[s]) {
                layers[i].k_stream[s]->buffer = vram_union_block;
                dkvt_update_strides(layers[i].k_stream[s], layers[i].orig_type_k);
                layers[i].k_stream[s]->data = (char*)layers[i].k->data + s * layers[i].k->nb[2];
            }
        }
        for (size_t s = 0; s < layers[i].v_stream.size(); ++s) {
            if (layers[i].v_stream[s]) {
                layers[i].v_stream[s]->buffer = vram_union_block;
                dkvt_update_strides(layers[i].v_stream[s], layers[i].orig_type_v);
                layers[i].v_stream[s]->data = (char*)layers[i].v->data + s * layers[i].v->nb[2];
            }
        }
    }
}

void llama_kv_cache::dkvt_bind_tg() {
    if (!vram_union_block || !ptr_start) return;

    // 正向连续布局：V 段在低地址，K 段在高地址，两段均从段首向段尾正向生长
    // V 段基址 = ptr_start + size_act_tg + layers[i].v_offset_tg
    // K 段基址 = ptr_start + size_act_tg + dkvt_v_size_tg + layers[i].k_offset_tg
    char * kv_base_tg = ptr_start + size_act_tg;

    for (size_t i = 0; i < layers.size(); ++i) {
        ggml_type target_type_k = is_transcodable_type(layers[i].orig_type_k) ? GGML_TYPE_F16 : layers[i].orig_type_k;
        ggml_type target_type_v = is_transcodable_type(layers[i].orig_type_v) ? GGML_TYPE_Q8_0 : layers[i].orig_type_v;

        if (layers[i].v) {
            layers[i].v->buffer = vram_union_block;
            layers[i].v->data = kv_base_tg + layers[i].v_offset_tg;
            dkvt_update_strides(layers[i].v, target_type_v);
        }
        if (layers[i].k) {
            layers[i].k->buffer = vram_union_block;
            layers[i].k->data = kv_base_tg + dkvt_v_size_tg + layers[i].k_offset_tg;
            dkvt_update_strides(layers[i].k, target_type_k);
        }
    }

    for (size_t i = 0; i < layers.size(); ++i) {
        ggml_type target_type_k = is_transcodable_type(layers[i].orig_type_k) ? GGML_TYPE_F16 : layers[i].orig_type_k;
        ggml_type target_type_v = is_transcodable_type(layers[i].orig_type_v) ? GGML_TYPE_Q8_0 : layers[i].orig_type_v;

        for (size_t s = 0; s < layers[i].k_stream.size(); ++s) {
            if (layers[i].k_stream[s]) {
                layers[i].k_stream[s]->buffer = vram_union_block;
                dkvt_update_strides(layers[i].k_stream[s], target_type_k);
                layers[i].k_stream[s]->data = (char*)layers[i].k->data + s * layers[i].k->nb[2];
            }
        }
        for (size_t s = 0; s < layers[i].v_stream.size(); ++s) {
            if (layers[i].v_stream[s]) {
                layers[i].v_stream[s]->buffer = vram_union_block;
                dkvt_update_strides(layers[i].v_stream[s], target_type_v);
                layers[i].v_stream[s]->data = (char*)layers[i].v->data + s * layers[i].v->nb[2];
            }
        }
    }
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

void llama_kv_cache::init_dkvt_sum_kv_sizes() {
    dkvt_k_size_pp = 0;
    dkvt_k_size_tg = 0;
    dkvt_v_size_pp = 0;
    dkvt_v_size_tg = 0;

    // 逐层计算尺寸和正向偏移：V 段在低地址，K 段在高地址
    // 每层偏移 = 前一层偏移 + 前一层尺寸（紧密排列，无间隙）
    for (size_t i = 0; i < layers.size(); ++i) {
        // 计算 PP 阶段尺寸（turbo4/turbo2 格式）
        if (layers[i].k) {
            layers[i].k_size_pp = ggml_row_size(layers[i].orig_type_k, layers[i].k->ne[0])
                               * layers[i].k->ne[1] * layers[i].k->ne[2] * layers[i].k->ne[3];
            layers[i].k_offset_pp = (i == 0) ? 0 : (layers[i - 1].k_offset_pp + layers[i - 1].k_size_pp);
        }
        if (layers[i].v) {
            layers[i].v_size_pp = ggml_row_size(layers[i].orig_type_v, layers[i].v->ne[0])
                               * layers[i].v->ne[1] * layers[i].v->ne[2] * layers[i].v->ne[3];
            layers[i].v_offset_pp = (i == 0) ? 0 : (layers[i - 1].v_offset_pp + layers[i - 1].v_size_pp);
        }
        // 计算 TG 阶段尺寸（f16/q8_0 格式）
        if (layers[i].k) {
            ggml_type target_type_k = is_transcodable_type(layers[i].orig_type_k) ? GGML_TYPE_F16 : layers[i].orig_type_k;
            layers[i].k_size_tg = ggml_row_size(target_type_k, layers[i].k->ne[0])
                               * layers[i].k->ne[1] * layers[i].k->ne[2] * layers[i].k->ne[3];
            layers[i].k_offset_tg = (i == 0) ? 0 : (layers[i - 1].k_offset_tg + layers[i - 1].k_size_tg);
        }
        if (layers[i].v) {
            ggml_type target_type_v = is_transcodable_type(layers[i].orig_type_v) ? GGML_TYPE_Q8_0 : layers[i].orig_type_v;
            layers[i].v_size_tg = ggml_row_size(target_type_v, layers[i].v->ne[0])
                               * layers[i].v->ne[1] * layers[i].v->ne[2] * layers[i].v->ne[3];
            layers[i].v_offset_tg = (i == 0) ? 0 : (layers[i - 1].v_offset_tg + layers[i - 1].v_size_tg);
        }
        // 累加总和
        dkvt_k_size_pp += layers[i].k_size_pp;
        dkvt_k_size_tg += layers[i].k_size_tg;
        dkvt_v_size_pp += layers[i].v_size_pp;
        dkvt_v_size_tg += layers[i].v_size_tg;
    }
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
    if (!vram_union_block) {
        LLAMA_LOG_WARN("llama_kv_cache: failed to allocate %.2f MiB on GPU. Retrying on CPU...\n",
                       (float)union_size / 1024.0f / 1024.0f);
        buft = ggml_backend_cpu_buffer_type();
        vram_union_block = ggml_backend_buft_alloc_buffer(buft, union_size);
    }
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
    if (vram_union_block) return;
    if (!sched) return;

    // 检查是否存在可转码层：若无任何 turbo 类型则无需 DKVT
    bool has_transcodable = false;
    for (size_t i = 0; i < layers.size(); ++i) {
        if ((layers[i].k && is_transcodable_type(layers[i].orig_type_k)) ||
            (layers[i].v && is_transcodable_type(layers[i].orig_type_v))) {
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

    // 伴生上下文：不参与别名分配和转码，直接返回
    if (other) {
        return;
    }

    // 主上下文：计算激活尺寸 → 缓存每层偏移/尺寸 → 计算 union_size → 分配 → 绑定
    init_dkvt_compute_activation_size(n_ubatch);
    init_dkvt_sum_kv_sizes();

    union_size = std::max(
        size_act_pp + dkvt_k_size_pp + dkvt_v_size_pp,
        size_act_tg + dkvt_k_size_tg + dkvt_v_size_tg);

#ifdef GGML_USE_CUDA
    size_t free_vram = 0, total_vram = 0;
    ggml_cuda_get_mem_info(&free_vram, &total_vram);
    LLAMA_LOG_INFO("llama_kv_cache: [DKVT-DIAG] pp(act=%.2f K=%.2f V=%.2f) tg(act=%.2f K=%.2f V=%.2f) union=%.2f free=%zu total=%zu MiB\n",
                   (float)size_act_pp / 1024 / 1024, (float)dkvt_k_size_pp / 1024 / 1024, (float)dkvt_v_size_pp / 1024 / 1024,
                   (float)size_act_tg / 1024 / 1024, (float)dkvt_k_size_tg / 1024 / 1024, (float)dkvt_v_size_tg / 1024 / 1024,
                   (float)union_size / 1024 / 1024, free_vram / 1024 / 1024, total_vram / 1024 / 1024);
#endif

    // 对齐到 buffer type 要求的 alignment，确保 CUDA 访问安全
    ggml_backend_buffer_type_t buft = nullptr;
    int buffer_id = -1;
    if (!init_dkvt_find_backend(sched, buft, buffer_id)) {
        return;
    }
    size_t alignment = ggml_backend_buft_get_alignment(buft);
    union_size = GGML_PAD(union_size, alignment);

    if (init_dkvt_alloc(buft)) {
        init_dkvt_bind_layers(sched, buffer_id);
    }
}

void llama_kv_cache::init_dkvt_compute_activation_size(size_t n_ubatch) {
    // 计算激活缓冲区尺寸（基于模型超参数数学推导，无魔法数字）：
    //
    // 每 token 的激活张量（float32）包含：
    //   1. Q 投影输出:          n_embd
    //   2. K 投影输出:          n_embd_head_k(0) * n_head_kv(0) = n_embd_k_gqa(0)
    //   3. V 投影输出:          n_embd_head_v(0) * n_head_kv(0) = n_embd_v_gqa(0)
    //   4. O 投影输入/残差:     n_embd
    //   5. Attention score:     n_head(0) * n_ubatch (PP) 或 n_head(0) * n_kv (TG)
    //   6. FFN gate 投影:       ffn_dim (MoE: n_expert_used*n_ff_exp + n_ff_shexp)
    //   7. FFN up 投影:         同 gate
    //   8. FFN down 投影输入:   同 gate
    //
    // MoE 模型：n_ff_arr[il] 存储的是 per-expert 中间层维度，
    // 每 token 实际 FFN 激活 = n_expert_used * n_ff_exp + n_ff_shexp

    const size_t layer_idx = 0; // 使用第 0 层参数作为代表（所有层共用同一激活乘数上界）
    const size_t n_embd = hparams.n_embd;
    const size_t n_head = hparams.n_head(layer_idx);
    const size_t n_embd_k_gqa = hparams.n_embd_k_gqa(layer_idx);
    const size_t n_embd_v_gqa = hparams.n_embd_v_gqa(layer_idx);

    // MoE 模型需要计算真实的 per-token FFN 维度
    const size_t ffn_dim = (hparams.n_expert > 0)
        ? (hparams.n_expert_used * hparams.n_ff_exp + hparams.n_ff_shexp)
        : hparams.n_ff(layer_idx);

    // 基础激活 = Q + K + V + O/残差 + FFN(gate + up + down)
    const size_t base_act = n_embd + n_embd_k_gqa + n_embd_v_gqa + n_embd + 3 * ffn_dim;

    // PP 阶段：attention score 矩阵 = n_head * n_ubatch * n_ubatch (causal mask 前)
    const size_t score_act_pp = n_head * n_ubatch * n_ubatch;
    size_act_pp = sizeof(float) * (n_ubatch * base_act + score_act_pp);

    // TG 阶段：attention score 矩阵 = n_head * 1 * n_kv
    // 使用 get_size() 获取实际的 KV cache 大小，而非 n_ctx_train
    const size_t kv_size = get_size();
    const size_t score_act_tg = n_head * 1 * kv_size;
    size_act_tg = sizeof(float) * (1 * base_act + score_act_tg);
}
