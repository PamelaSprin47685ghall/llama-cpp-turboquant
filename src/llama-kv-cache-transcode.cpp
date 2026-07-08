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
extern "C" {
    typedef struct CUstream_st *cudaStream_t;
    typedef enum {
        cudaSuccess = 0
    } cudaError_t;
    typedef enum {
        cudaMemcpyDeviceToDevice = 3
    } cudaMemcpyKind;

    cudaError_t cudaMalloc(void **devPtr, size_t size);
    cudaError_t cudaFree(void *devPtr);
    cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, cudaMemcpyKind kind, cudaStream_t stream);
    cudaError_t cudaStreamSynchronize(cudaStream_t stream);
    const char * cudaGetErrorString(cudaError_t error);
}
#endif

#ifdef GGML_USE_CUDA
extern "C" void ggml_cuda_transcode_k(
    const void * src, void * dst,
    int64_t head_size_k, int64_t kv_size, int type_k, void * stream);

extern "C" void ggml_cuda_transcode_v(
    const void * src, void * dst,
    int64_t head_size_v, int64_t kv_size, int type_v, void * stream);

extern "C" void ggml_cuda_device_to_device_copy_async(void * dst, const void * src, size_t size, void * stream);

extern "C" void ggml_cuda_stream_synchronize(void * stream);
#endif

void llama_kv_cache::transcode_to_tg_cuda_k(const char * k_src_base, char * k_dst_base, void * stream) {
#ifdef GGML_USE_CUDA
    for (size_t i = 0; i < layers.size(); ++i) {
        if (layers[i].k) {
            const char * src = k_src_base + layers[i].k_offset_pp;
            char * dst = k_dst_base + layers[i].k_offset_tg;

            if (is_transcodable_type(layers[i].orig_type_k)) {
                ggml_cuda_transcode_k(
                    src, dst,
                    layers[i].k->ne[0], this->get_size(), (int)layers[i].orig_type_k, stream);
            } else {
                ggml_cuda_device_to_device_copy_async(dst, src, layers[i].k_size_pp, stream);
            }
        }
    }
#else
    GGML_UNUSED(k_src_base);
    GGML_UNUSED(k_dst_base);
    GGML_UNUSED(stream);
#endif
}

void llama_kv_cache::transcode_to_tg_cuda_v(const char * v_src_base, char * v_dst_base, void * stream) {
#ifdef GGML_USE_CUDA
    for (size_t i = 0; i < layers.size(); ++i) {
        if (layers[i].v) {
            const char * src = v_src_base + layers[i].v_offset_pp;
            char * dst = v_dst_base + layers[i].v_offset_tg;

            if (is_transcodable_type(layers[i].orig_type_v)) {
                // v_trans=true 时 V 布局为 [kv_size, head_size_v, n_head_kv, n_ctx]
                // v_trans=false 时 V 布局为 [head_size_v, kv_size, n_head_kv, n_ctx]
                int64_t head_size_v = v_trans ? layers[i].v->ne[1] : layers[i].v->ne[0];
                int64_t kv_size = v_trans ? layers[i].v->ne[0] : layers[i].v->ne[1];
                ggml_cuda_transcode_v(
                    src, dst,
                    head_size_v, kv_size, (int)layers[i].orig_type_v, stream);
            } else {
                ggml_cuda_device_to_device_copy_async(dst, src, layers[i].v_size_pp, stream);
            }
        }
    }
#else
    GGML_UNUSED(v_src_base);
    GGML_UNUSED(v_dst_base);
    GGML_UNUSED(stream);
#endif
}

void llama_kv_cache::transcode_to_tg_cuda(void * stream) {
#ifdef GGML_USE_CUDA
    size_t pp_size = dkvt_k_size_pp + dkvt_v_size_pp;
    if (pp_size == 0) return;

    void * temp_gpu_buf = nullptr;
    cudaError_t err = cudaMalloc(&temp_gpu_buf, pp_size);
    if (err != cudaSuccess) {
        LLAMA_LOG_ERROR("llama_kv_cache: failed to allocate temp GPU buffer for transcoding: %s\n", cudaGetErrorString(err));
        return;
    }

    char * src_base = ptr_start + size_act_pp;
    cudaStream_t custream = (cudaStream_t)stream;
    err = cudaMemcpyAsync(temp_gpu_buf, src_base, pp_size, cudaMemcpyDeviceToDevice, custream);
    if (err != cudaSuccess) {
        LLAMA_LOG_ERROR("llama_kv_cache: failed to copy PP KV cache to temp buffer: %s\n", cudaGetErrorString(err));
        cudaFree(temp_gpu_buf);
        return;
    }

    const char * v_src_temp = (const char *)temp_gpu_buf;
    const char * k_src_temp = v_src_temp + dkvt_v_size_pp;

    char * v_dst = ptr_start + size_act_tg;
    char * k_dst = v_dst + dkvt_v_size_tg;

    transcode_to_tg_cuda_k(k_src_temp, k_dst, stream);
    transcode_to_tg_cuda_v(v_src_temp, v_dst, stream);

    cudaStreamSynchronize(custream);
    cudaFree(temp_gpu_buf);
#else
    GGML_UNUSED(stream);
#endif
}

void llama_kv_cache::init_dkvt_bind_layers(ggml_backend_sched_t sched, int buffer_id) {
    dkvt_bind_pp();

    for (size_t i = 0; i < layers.size(); ++i) {
        if (layers[i].k) {
            layers[i].k->buffer = vram_union_block;
            if (is_transcodable_type(layers[i].orig_type_k)) {
                layers[i].k->flags |= GGML_TENSOR_FLAG_EXT;
            }
            for (size_t s = 0; s < layers[i].k_stream.size(); ++s) {
                if (layers[i].k_stream[s]) {
                    layers[i].k_stream[s]->buffer = vram_union_block;
                }
            }
        }
        if (layers[i].v) {
            layers[i].v->buffer = vram_union_block;
            if (is_transcodable_type(layers[i].orig_type_v)) {
                layers[i].v->flags |= GGML_TENSOR_FLAG_EXT;
            }
            for (size_t s = 0; s < layers[i].v_stream.size(); ++s) {
                if (layers[i].v_stream[s]) {
                    layers[i].v_stream[s]->buffer = vram_union_block;
                }
            }
        }
    }

    ggml_backend_sched_set_custom_buffer(sched, buffer_id, vram_union_block);
    is_transcoded_tg = false;

    LLAMA_LOG_INFO("llama_kv_cache: DKVT Flat Buffer successfully allocated. Size: %.2f MiB, start: %p, end: %p\n",
                   (float)union_size / 1024.0f / 1024.0f, (void*)ptr_start, (void*)ptr_end);
}

void llama_kv_cache::transcode_to_tg_cpu() {
    // 正向连续布局：V 段在低地址，K 段在高地址，两段均正向生长
    // PP 源基址：V 起始于 ptr_start + size_act_pp, K 起始于 ptr_start + size_act_pp + dkvt_v_size_pp
    const char * v_src_base = ptr_start + size_act_pp;
    const char * k_src_base = v_src_base + dkvt_v_size_pp;

    // 重新绑定到 TG 目标地址（正向生长）
    dkvt_bind_tg();

    // 行流式转码：每次只处理一行（head_size 个元素），避免全层 buffer 造成 OOM
    for (size_t i = 0; i < layers.size(); ++i) {
        if (layers[i].k && layers[i].k_size_pp > 0) {
            const char * src_base = k_src_base + layers[i].k_offset_pp;
            if (is_transcodable_type(layers[i].orig_type_k)) {
                // 行流式：逐行反量化到 float，再量化到目标类型
                // 每行 = head_size_k 个元素，turbo block 大小为 128
                int64_t head_size_k = layers[i].k->ne[0];
                int64_t n_rows = layers[i].k->ne[1] * layers[i].k->ne[2] * layers[i].k->ne[3];
                int64_t blocks_per_row = head_size_k / 128;
                std::vector<float> temp_row(head_size_k);
                // 保存旧 nb[1]（PP 格式的行 stride），因为 dkvt_bind_tg 已重写 nb[]
                size_t old_nb1 = ggml_row_size(layers[i].orig_type_k, head_size_k);
                size_t new_nb1 = layers[i].k->nb[1]; // TG 格式行 stride (F16)
                for (int64_t row = 0; row < n_rows; ++row) {
                    const char * src_row = src_base + row * old_nb1;
                    if (layers[i].orig_type_k == GGML_TYPE_TURBO4_0) {
                        for (int64_t b = 0; b < blocks_per_row; ++b) {
                            dequantize_row_turbo4_0((const block_turbo4_0 *)(src_row + b * sizeof(block_turbo4_0)), temp_row.data() + b * 128, 128);
                        }
                    } else if (layers[i].orig_type_k == GGML_TYPE_TURBO2_0) {
                        for (int64_t b = 0; b < blocks_per_row; ++b) {
                            dequantize_row_turbo2_0((const block_turbo2_0 *)(src_row + b * sizeof(block_turbo2_0)), temp_row.data() + b * 128, 128);
                        }
                    }
                    char * dst_row = (char*)layers[i].k->data + row * new_nb1;
                    ggml_fp32_to_fp16_row(temp_row.data(), (ggml_fp16_t *)dst_row, head_size_k);
                }
            } else {
                std::memmove(layers[i].k->data, src_base, layers[i].k_size_pp);
            }
        }
        if (layers[i].v && layers[i].v_size_pp > 0) {
            const char * src_base = v_src_base + layers[i].v_offset_pp;
            if (is_transcodable_type(layers[i].orig_type_v)) {
                // 行流式：逐行反量化到 float，再量化到 Q8_0
                int64_t head_size_v = layers[i].v->ne[0];
                int64_t n_rows = layers[i].v->ne[1] * layers[i].v->ne[2] * layers[i].v->ne[3];
                int64_t blocks_per_row = head_size_v / 128;
                std::vector<float> temp_row(head_size_v);
                size_t old_nb1 = ggml_row_size(layers[i].orig_type_v, head_size_v);
                size_t new_nb1 = layers[i].v->nb[1]; // TG 格式行 stride (Q8_0)
                for (int64_t row = 0; row < n_rows; ++row) {
                    const char * src_row = src_base + row * old_nb1;
                    if (layers[i].orig_type_v == GGML_TYPE_TURBO2_0) {
                        for (int64_t b = 0; b < blocks_per_row; ++b) {
                            dequantize_row_turbo2_0((const block_turbo2_0 *)(src_row + b * sizeof(block_turbo2_0)), temp_row.data() + b * 128, 128);
                        }
                    } else if (layers[i].orig_type_v == GGML_TYPE_TURBO4_0) {
                        for (int64_t b = 0; b < blocks_per_row; ++b) {
                            dequantize_row_turbo4_0((const block_turbo4_0 *)(src_row + b * sizeof(block_turbo4_0)), temp_row.data() + b * 128, 128);
                        }
                    }
                    char * dst_row = (char*)layers[i].v->data + row * new_nb1;
                    ggml_quantize_chunk(GGML_TYPE_Q8_0, temp_row.data(), dst_row, 0, head_size_v / 32, head_size_v, nullptr);
                }
            } else {
                std::memmove(layers[i].v->data, src_base, layers[i].v_size_pp);
            }
        }
    }
}

void llama_kv_cache::transcode_to_tg(void * stream) {
    if (!vram_union_block || !ptr_start) return;

    if (is_transcoded_tg) return;

    // 伴生上下文同步：若主上下文已转码，伴生上下文直接复用绑定
    if (other) {
        if (other->is_transcoded_tg && !is_transcoded_tg) {
            v_trans = true;
            this->dkvt_bind_tg();
            is_transcoded_tg = true;
        }
        return;
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
        transcode_to_tg_cuda(stream);
    } else {
        transcode_to_tg_cpu();
    }

    if (is_cuda) {
        dkvt_bind_tg();
    }

    // 确保 CUDA 流中转码操作完成后再继续
    if (is_cuda) {
        ggml_cuda_stream_synchronize(stream);
    }

    v_trans = true;
    is_transcoded_tg = true;
}
