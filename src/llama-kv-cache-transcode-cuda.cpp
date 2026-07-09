#include "llama-kv-cache.h"
#include "llama-kv-cache-dkvt.h"
#include "llama-impl.h"

#ifdef GGML_USE_CUDA
extern "C" {
    typedef struct CUstream_st *cudaStream_t;
    typedef enum {
        cudaSuccess = 0
    } cudaError_t;
    typedef enum {
        cudaMemcpyDeviceToDevice = 3
    } cudaMemcpyKind;

    cudaError_t cudaStreamSynchronize(cudaStream_t stream);
    cudaError_t cudaGetLastError(void);
    const char * cudaGetErrorString(cudaError_t error);
}

extern "C" void ggml_cuda_transcode_k_row(
    const void * src_row, void * dst_row,
    int64_t head_size_k,
    int64_t src_row_stride, int64_t dst_row_stride,
    int type_k, void * stream);

extern "C" void ggml_cuda_transcode_v_row(
    const void * src_row, void * dst_row,
    int64_t head_size_v,
    int64_t src_row_stride, int64_t dst_row_stride,
    int type_v, void * stream);

extern "C" void ggml_cuda_device_to_device_copy_async(void * dst, const void * src, size_t size, void * stream);
#endif

bool llama_kv_cache::transcode_to_tg_cuda_k(const char * k_src_base, char * k_dst_base, void * stream) {
#ifdef GGML_USE_CUDA
    const int64_t n_kv = get_transcode_n_kv();
    cudaStream_t custream = (cudaStream_t) stream;

    char * k_row_scratch = ptr_start;
    size_t scratch_cap = size_act_pp;

    for (size_t idx = 0; idx < layers.size(); ++idx) {
        size_t i = layers.size() - 1 - idx;
        if (!layers[i].k) continue;
        if (!(layers[i].k->flags & GGML_TENSOR_FLAG_EXT)) continue;

        const char * src = k_src_base + layers[i].k_offset_pp;
        char * dst = k_dst_base + layers[i].k_offset_tg;

        int64_t n_stream = layers[i].k->ne[2];
        int64_t n_ctx_seq = layers[i].k->ne[1];
        int64_t src_row_stride = layers[i].k->nb[1];
        int64_t src_stream_stride = layers[i].k->nb[2];

        if (is_transcodable_type(layers[i].orig_type_k)) {
            int64_t dst_row_stride = ggml_row_size(dkvt_tg_type_k(layers[i].orig_type_k), layers[i].k->ne[0]);
            int64_t dst_stream_stride = n_ctx_seq * dst_row_stride;

            if ((size_t) src_row_stride > scratch_cap) {
                LLAMA_LOG_ERROR("llama_kv_cache: K row scratch overflow %zu > %zu\n",
                                (size_t) src_row_stride, scratch_cap);
                return false;
            }

            for (int64_t s = 0; s < n_stream; ++s) {
                for (int64_t cell = 0; cell < n_kv; ++cell) {
                    const char * src_row = src + s * src_stream_stride + cell * src_row_stride;
                    char * dst_row = dst + s * dst_stream_stride + cell * dst_row_stride;

                    ggml_cuda_device_to_device_copy_async(
                        k_row_scratch, src_row, (size_t) src_row_stride, stream);
                    cudaError_t err = cudaGetLastError();
                    if (err != cudaSuccess) {
                        LLAMA_LOG_ERROR(
                            "llama_kv_cache: K D2D copy failed layer %zu stream %ld cell %ld: %s\n",
                            i, (long) s, (long) cell, cudaGetErrorString(err));
                        return false;
                    }
                    ggml_cuda_transcode_k_row(
                        k_row_scratch, dst_row,
                        layers[i].k->ne[0],
                        src_row_stride, dst_row_stride,
                        (int) layers[i].orig_type_k, stream);
                    err = cudaGetLastError();
                    if (err != cudaSuccess) {
                        LLAMA_LOG_ERROR(
                            "llama_kv_cache: K transcode kernel launch failed layer %zu stream %ld cell %ld: %s\n",
                            i, (long) s, (long) cell, cudaGetErrorString(err));
                        return false;
                    }
                }
            }
        } else {
            int64_t dst_row_stride = src_row_stride;
            int64_t dst_stream_stride = src_stream_stride;

            for (int64_t s = 0; s < n_stream; ++s) {
                const char * src_stream = src + s * src_stream_stride;
                char * dst_stream = dst + s * dst_stream_stride;
                size_t copy_size = (size_t) src_row_stride * n_kv;

                ggml_cuda_device_to_device_copy_async(dst_stream, src_stream, copy_size, stream);
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    LLAMA_LOG_ERROR(
                        "llama_kv_cache: K direct D2D copy failed layer %zu stream %ld: %s\n",
                        i, (long) s, cudaGetErrorString(err));
                    return false;
                }
            }
        }
    }
    return true;
#else
    GGML_UNUSED(k_src_base);
    GGML_UNUSED(k_dst_base);
    GGML_UNUSED(stream);
    return false;
#endif
}

bool llama_kv_cache::transcode_to_tg_cuda_v(const char * v_src_base, char * v_dst_base, void * stream) {
#ifdef GGML_USE_CUDA
    const int64_t n_kv = get_transcode_n_kv();
    cudaStream_t custream = (cudaStream_t) stream;
    char * v_row_scratch = ptr_start;
    size_t scratch_cap = size_act_pp;

    for (size_t i = 0; i < layers.size(); ++i) {
        if (!layers[i].v) continue;
        if (!(layers[i].v->flags & GGML_TENSOR_FLAG_EXT)) continue;

        const char * src = v_src_base + layers[i].v_offset_pp;
        char * dst = v_dst_base + layers[i].v_offset_tg;

        int64_t head_size_v = v_trans ? layers[i].v->ne[1] : layers[i].v->ne[0];
        int64_t src_row_stride = layers[i].v->nb[1];
        int64_t src_stream_stride = layers[i].v->nb[2];

        int64_t dst_row_stride = ggml_row_size(dkvt_tg_type_v(layers[i].orig_type_v), head_size_v);
        int64_t n_stream = layers[i].v->ne[2];
        int64_t n_ctx_seq = v_trans ? layers[i].v->ne[0] : layers[i].v->ne[1];
        int64_t dst_stream_stride = n_ctx_seq * dst_row_stride;

        if (is_transcodable_type(layers[i].orig_type_v)) {
            if ((size_t) src_row_stride > scratch_cap) {
                LLAMA_LOG_ERROR("llama_kv_cache: V row scratch overflow\n");
                return false;
            }
            for (int64_t s = 0; s < n_stream; ++s) {
                for (int64_t cell = 0; cell < n_kv; ++cell) {
                    const char * src_row = src + s * src_stream_stride + cell * src_row_stride;
                    char * dst_row = dst + s * dst_stream_stride + cell * dst_row_stride;

                    ggml_cuda_device_to_device_copy_async(
                        v_row_scratch, src_row, (size_t) src_row_stride, stream);
                    cudaError_t err = cudaGetLastError();
                    if (err != cudaSuccess) {
                        LLAMA_LOG_ERROR(
                            "llama_kv_cache: V D2D copy failed layer %zu stream %ld cell %ld: %s\n",
                            i, (long) s, (long) cell, cudaGetErrorString(err));
                        return false;
                    }
                    ggml_cuda_transcode_v_row(
                        v_row_scratch, dst_row, head_size_v,
                        src_row_stride, dst_row_stride,
                        (int) layers[i].orig_type_v, stream);
                    err = cudaGetLastError();
                    if (err != cudaSuccess) {
                        LLAMA_LOG_ERROR(
                            "llama_kv_cache: V transcode kernel launch failed layer %zu stream %ld cell %ld: %s\n",
                            i, (long) s, (long) cell, cudaGetErrorString(err));
                        return false;
                    }
                }
            }
        } else {
            int64_t dst_row_stride = src_row_stride;
            int64_t dst_stream_stride = src_stream_stride;

            for (int64_t s = 0; s < n_stream; ++s) {
                const char * src_stream = src + s * src_stream_stride;
                char * dst_stream = dst + s * dst_stream_stride;
                size_t copy_size = (size_t) src_row_stride * n_kv;

                ggml_cuda_device_to_device_copy_async(dst_stream, src_stream, copy_size, stream);
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    LLAMA_LOG_ERROR(
                        "llama_kv_cache: V direct D2D copy failed layer %zu stream %ld: %s\n",
                        i, (long) s, cudaGetErrorString(err));
                    return false;
                }
            }
        }
    }
    return true;
#else
    GGML_UNUSED(v_src_base);
    GGML_UNUSED(v_dst_base);
    GGML_UNUSED(stream);
    return false;
#endif
}

bool llama_kv_cache::transcode_to_tg_cuda(void * stream) {
#ifdef GGML_USE_CUDA
    if (dkvt_k_size_pp + dkvt_v_size_pp == 0) return true;

    cudaStream_t custream = (cudaStream_t)stream;

    const char * v_src = ptr_start + size_act_pp;
    const char * k_src = v_src + dkvt_v_size_pp;

    char * v_dst = ptr_start + size_act_tg;
    char * k_dst = v_dst + dkvt_v_size_tg;

    if (!transcode_to_tg_cuda_k(k_src, k_dst, stream)) {
        return false;
    }
    if (!transcode_to_tg_cuda_v(v_src, v_dst, stream)) {
        return false;
    }

    cudaError_t err = cudaStreamSynchronize(custream);
    if (err != cudaSuccess) {
        LLAMA_LOG_ERROR("llama_kv_cache: CUDA stream sync failed after transcode: %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;
#else
    GGML_UNUSED(stream);
    return false;
#endif
}
