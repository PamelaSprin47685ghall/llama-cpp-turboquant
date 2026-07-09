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

extern "C" void turbo_cpu_fwht_inverse(float * x, int group_size);

// CUDA transcode implementations moved to src/llama-kv-cache-transcode-cuda.cpp

void llama_kv_cache::init_dkvt_bind_layers(ggml_backend_sched_t sched, int buffer_id) {
    dkvt_bind_pp();

    for (size_t i = 0; i < layers.size(); ++i) {
        if (layers[i].k && is_transcodable_type(layers[i].orig_type_k)) {
            layers[i].k->buffer = vram_union_block;
            layers[i].k->flags |= GGML_TENSOR_FLAG_EXT;
            for (size_t s = 0; s < layers[i].k_stream.size(); ++s) {
                if (layers[i].k_stream[s]) {
                    layers[i].k_stream[s]->buffer = vram_union_block;
                }
            }
        }
        if (layers[i].v && is_transcodable_type(layers[i].orig_type_v)) {
            layers[i].v->buffer = vram_union_block;
            layers[i].v->flags |= GGML_TENSOR_FLAG_EXT;
            for (size_t s = 0; s < layers[i].v_stream.size(); ++s) {
                if (layers[i].v_stream[s]) {
                    layers[i].v_stream[s]->buffer = vram_union_block;
                }
            }
        }
    }

    ggml_backend_sched_set_custom_buffer(sched, buffer_id, vram_union_block);

    // Store sched + buffer_id for compute cap control
    dkvt_sched = sched;
    dkvt_sched_buffer_id = buffer_id;

    // Set compute cap to PP size to prevent graph allocator from overwriting KV data
    dkvt_apply_union_compute_cap(sched);

    is_transcoded_tg = false;

    LLAMA_LOG_INFO("llama_kv_cache: DKVT Flat Buffer successfully allocated. Size: %.2f MiB, start: %p, end: %p\n",
                   (float)union_size / 1024.0f / 1024.0f, (void*)ptr_start, (void*)ptr_end);
}

void llama_kv_cache::dkvt_apply_union_compute_cap(ggml_backend_sched_t sched) const {
    if (other) {
        size_act_pp = other->size_act_pp;
        size_act_tg = other->size_act_tg;
    }
    if (!sched || dkvt_sched_buffer_id < 0) return;
    size_t kv_total_tg = dkvt_v_size_tg + dkvt_k_size_tg;
    size_t cap = dkvt_union_compute_cap_bytes(is_transcoded_tg, size_act_pp, size_act_tg, kv_total_tg);
    ggml_backend_sched_set_borrowed_compute_cap(sched, dkvt_sched_buffer_id, cap);
    // In TG mode (mirror layout), compute starts AFTER the KV region.
    // In PP mode, compute starts at offset 0 (KV is after compute).
    size_t base_offset = dkvt_union_compute_base_offset(is_transcoded_tg, kv_total_tg);
    ggml_backend_sched_set_borrowed_compute_base_offset(sched, dkvt_sched_buffer_id, base_offset);
}

void llama_kv_cache::dkvt_sync_pp_compute_from_sched(ggml_backend_sched_t sched, size_t measured_bytes) {
    if (!vram_union_block || !sched || measured_bytes == 0) {
        return;
    }
    const size_t margin = measured_bytes / 16 + 64 * 1024;
    const size_t new_size_act_pp = dkvt_align_up(measured_bytes + margin, std::max((size_t) 1, dkvt_gpu_alignment));
    if (new_size_act_pp <= size_act_pp) {
        return;
    }
    // Never grow the PP compute region so far that the PP KV segment no longer fits.
    const size_t kv_total_pp = dkvt_v_size_pp + dkvt_k_size_pp;
    if (new_size_act_pp > union_size || new_size_act_pp + kv_total_pp > union_size) {
        LLAMA_LOG_ERROR(
            "llama_kv_cache: DKVT measured PP compute (%.2f MiB) does not fit in union (%.2f MiB); keeping size_act_pp (%.2f MiB)\n",
            (float) new_size_act_pp / 1024 / 1024,
            (float) union_size / 1024 / 1024,
            (float) size_act_pp / 1024 / 1024);
        return;
    }
    size_act_pp = new_size_act_pp;
    LLAMA_LOG_WARN(
        "llama_kv_cache: DKVT raised size_act_pp to %.2f MiB (sched measured %.2f MiB)\n",
        (float) size_act_pp / 1024 / 1024,
        (float) measured_bytes / 1024 / 1024);
    dkvt_bind_pp();
    dkvt_apply_union_compute_cap(sched);
}

void llama_kv_cache::dkvt_sync_tg_compute_from_sched(ggml_backend_sched_t sched, size_t measured_bytes) {
    if (!vram_union_block || !sched || measured_bytes == 0) {
        return;
    }
    const size_t margin = measured_bytes / 16 + 64 * 1024;
    const size_t new_size_act_tg = dkvt_align_up(measured_bytes + margin, std::max((size_t) 1, dkvt_gpu_alignment));
    if (new_size_act_tg <= size_act_tg) {
        return;
    }
    const size_t kv_total_tg = dkvt_v_size_tg + dkvt_k_size_tg;
    if (new_size_act_tg > union_size || new_size_act_tg + kv_total_tg > union_size) {
        LLAMA_LOG_ERROR(
            "llama_kv_cache: DKVT measured TG compute (%.2f MiB) does not fit in union (%.2f MiB); keeping size_act_tg (%.2f MiB)\n",
            (float) new_size_act_tg / 1024 / 1024,
            (float) union_size / 1024 / 1024,
            (float) size_act_tg / 1024 / 1024);
        return;
    }
    size_act_tg = new_size_act_tg;
    LLAMA_LOG_WARN(
        "llama_kv_cache: DKVT raised size_act_tg to %.2f MiB (sched measured %.2f MiB)\n",
        (float) size_act_tg / 1024 / 1024,
        (float) measured_bytes / 1024 / 1024);
    dkvt_apply_union_compute_cap(sched);
}

void llama_kv_cache::transcode_to_tg_cpu() {
    // 正向连续布局：V 段在低地址，K 段在高地址，两段均正向生长
    // PP 源基址：V 起始于 ptr_start + size_act_pp, K 起始于 ptr_start + size_act_pp + dkvt_v_size_pp
    const char * v_src_base = ptr_start + size_act_pp;
    const char * k_src_base = v_src_base + dkvt_v_size_pp;

    // 重新绑定到 TG 目标地址（正向生长）
    dkvt_bind_tg();

    // 行流式转码：每次只处理一行（head_size 个元素），避免全层 buffer 造成 OOM
    // 仅转码已使用的行（get_transcode_n_kv），而非整个 ne[1] 维度
    const int64_t n_kv_rows = get_transcode_n_kv();
    for (size_t i = 0; i < layers.size(); ++i) {
        if (layers[i].k && layers[i].k_size_pp > 0) {
            const char * src_base = k_src_base + layers[i].k_offset_pp;
            if (is_transcodable_type(layers[i].orig_type_k)) {
                // 行流式：逐行反量化到 float，再量化到目标类型
                // 每行 = head_size_k 个元素，turbo block 大小为 DKVT_TURBO_BLOCK_SIZE
                int64_t head_size_k = layers[i].k->ne[0];
                int64_t blocks_per_row = head_size_k / DKVT_TURBO_BLOCK_SIZE;
                const size_t group_stride_pp_k = ggml_row_size(layers[i].orig_type_k, DKVT_TURBO_BLOCK_SIZE);
                std::vector<float> temp_row(head_size_k);
                size_t old_nb1 = ggml_row_size(layers[i].orig_type_k, head_size_k);
                size_t new_nb1 = layers[i].k->nb[1];
                int64_t n_stream = layers[i].k->ne[2];
                int64_t rows_per_stream = n_kv_rows;
                int64_t total_rows = rows_per_stream * n_stream;
                for (int64_t row = 0; row < total_rows; ++row) {
                    const char * src_row = src_base + row * old_nb1;
                    for (int64_t b = 0; b < blocks_per_row; ++b) {
                        const char * grp = src_row + b * group_stride_pp_k;
                        if (layers[i].orig_type_k == GGML_TYPE_TURBO4_0) {
                            dequantize_row_turbo4_0((const block_turbo4_0 *)grp, temp_row.data() + b * DKVT_TURBO_BLOCK_SIZE, DKVT_TURBO_BLOCK_SIZE);
                        } else {
                            dequantize_row_turbo2_0((const block_turbo2_0 *)grp, temp_row.data() + b * DKVT_TURBO_BLOCK_SIZE, DKVT_TURBO_BLOCK_SIZE);
                        }
                        turbo_cpu_fwht_inverse(temp_row.data() + b * DKVT_TURBO_BLOCK_SIZE, DKVT_TURBO_BLOCK_SIZE);
                    }
                    char * dst_row = (char*)layers[i].k->data + row * new_nb1;
                    ggml_fp32_to_fp16_row(temp_row.data(), (ggml_fp16_t *)dst_row, head_size_k);
                }
            } else {
                size_t copy_size = ggml_row_size(layers[i].orig_type_k, layers[i].k->ne[0]) * n_kv_rows;
                std::memmove(layers[i].k->data, src_base, copy_size);
            }
        }
        if (layers[i].v && layers[i].v_size_pp > 0) {
            const char * src_base = v_src_base + layers[i].v_offset_pp;
            int64_t head_size_v = v_trans ? layers[i].v->ne[1] : layers[i].v->ne[0];
            if (is_transcodable_type(layers[i].orig_type_v)) {
                int64_t blocks_per_row = head_size_v / DKVT_TURBO_BLOCK_SIZE;
                const size_t group_stride_pp_v = ggml_row_size(layers[i].orig_type_v, DKVT_TURBO_BLOCK_SIZE);
                std::vector<float> temp_row(head_size_v);
                size_t old_nb1 = ggml_row_size(layers[i].orig_type_v, head_size_v);
                size_t new_nb1 = layers[i].v->nb[1];
                int64_t n_stream = layers[i].v->ne[2];
                int64_t total_rows = n_kv_rows * n_stream;
                for (int64_t row = 0; row < total_rows; ++row) {
                    const char * src_row = src_base + row * old_nb1;
                    for (int64_t b = 0; b < blocks_per_row; ++b) {
                        const char * grp = src_row + b * group_stride_pp_v;
                        if (layers[i].orig_type_v == GGML_TYPE_TURBO4_0) {
                            dequantize_row_turbo4_0((const block_turbo4_0 *)grp, temp_row.data() + b * DKVT_TURBO_BLOCK_SIZE, DKVT_TURBO_BLOCK_SIZE);
                        } else {
                            dequantize_row_turbo2_0((const block_turbo2_0 *)grp, temp_row.data() + b * DKVT_TURBO_BLOCK_SIZE, DKVT_TURBO_BLOCK_SIZE);
                        }
                        turbo_cpu_fwht_inverse(temp_row.data() + b * DKVT_TURBO_BLOCK_SIZE, DKVT_TURBO_BLOCK_SIZE);
                    }
                    char * dst_row = (char*)layers[i].v->data + row * new_nb1;
                    ggml_quantize_chunk(GGML_TYPE_Q8_0, temp_row.data(), dst_row, 0, 1, head_size_v, nullptr);
                }
            } else {
                size_t copy_size = ggml_row_size(layers[i].orig_type_v, head_size_v) * n_kv_rows;
                std::memmove(layers[i].v->data, src_base, copy_size);
            }
        }
    }
}

void llama_kv_cache::transcode_to_tg(void * stream) {
    if (is_transcoded_tg) {
        return;
    }

    if (!vram_union_block || !ptr_start) {
        if (other) {
            return;
        }
        throw std::runtime_error("DKVT transcode required but union buffer is not initialized");
    }

    // 伴生上下文转码：伴生上下文有自己独立的层（如 MTP 层），
    // 必须执行自己的转码内核，不能仅复用主上下文的绑定。
    if (other) {
        if (other->is_transcoded_tg && !is_transcoded_tg) {
            // Check if companion has its own transcodable layers with data
            bool has_own_transcodable = false;
            for (size_t i = 0; i < layers.size(); ++i) {
                // Only transcode layers that are NOT shared with parent (no matching il in parent)
                int32_t il = layers[i].il;
                if (other->map_layer_ids.count(il) == 0) {
                    if ((layers[i].k && is_transcodable_type(layers[i].orig_type_k) && (layers[i].k->flags & GGML_TENSOR_FLAG_EXT)) ||
                        (layers[i].v && is_transcodable_type(layers[i].orig_type_v) && (layers[i].v->flags & GGML_TENSOR_FLAG_EXT))) {
                        has_own_transcodable = true;
                        break;
                    }
                }
            }
            if (has_own_transcodable && vram_union_block && ptr_start) {
                // Run transcode kernel on companion's own layers
                bool is_cuda = false;
#ifdef GGML_USE_CUDA
                ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(vram_union_block);
                if (buft) {
                    ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
                    if (dev) {
                        enum ggml_backend_dev_type dev_type = ggml_backend_dev_type(dev);
                        is_cuda = (dev_type == GGML_BACKEND_DEVICE_TYPE_GPU || dev_type == GGML_BACKEND_DEVICE_TYPE_IGPU);
                    }
                }
#endif
                if (is_cuda) {
                    if (!transcode_to_tg_cuda(stream)) {
                        throw std::runtime_error("DKVT companion CUDA transcode_to_tg failed");
                    }
                } else {
                    transcode_to_tg_cpu();
                }
            }
            this->dkvt_bind_tg();
            is_transcoded_tg = true;
            return;
        }
        throw std::runtime_error("DKVT companion transcode: parent has not completed transcode_to_tg");
    }

    // 通过 buffer type 的设备类型判断是否为 CUDA 后端
    bool is_cuda = false;
#ifdef GGML_USE_CUDA
    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(vram_union_block);
    if (buft) {
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        if (dev) {
            enum ggml_backend_dev_type dev_type = ggml_backend_dev_type(dev);
            is_cuda = (dev_type == GGML_BACKEND_DEVICE_TYPE_GPU || dev_type == GGML_BACKEND_DEVICE_TYPE_IGPU);
        }
    }
#endif

    if (is_cuda) {
        if (!transcode_to_tg_cuda(stream)) {
            throw std::runtime_error("DKVT CUDA transcode_to_tg failed (kernel or stream sync)");
        }
    } else {
        transcode_to_tg_cpu();
    }

    dkvt_bind_tg();
    is_transcoded_tg = true;
    LLAMA_LOG_WARN("%s: DKVT transcode completed (is_transcoded_tg=1)\n", __func__);
}
