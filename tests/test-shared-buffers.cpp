#include <cstdio>
#include <cassert>
#include "llama.h"
#include "ggml-backend.h"
#include "ggml.h"

int main() {
    printf("Testing shared compute buffers parameters...\n");
    auto params = llama_context_default_params();
    params.share_compute_buffers_with = nullptr;
    assert(params.share_compute_buffers_with == nullptr);

    // Initialize CPU backend
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    assert(backend != nullptr);

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);

    // Create src and dst schedulers
    ggml_backend_sched_t sched_src = ggml_backend_sched_new(&backend, &buft, 1, 128, false, false);
    ggml_backend_sched_t sched_dst = ggml_backend_sched_new(&backend, &buft, 1, 128, false, false);

    assert(sched_src != nullptr);
    assert(sched_dst != nullptr);

    // Create a dummy graph to trigger allocation/reservation
    struct ggml_init_params params_ggml = {
        /* .mem_size   = */ 1024 * 1024,
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ true,
    };
    struct ggml_context * ctx = ggml_init(params_ggml);
    assert(ctx != nullptr);

    // Create source graph with input and operation
    struct ggml_tensor * t_src = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10);
    t_src->flags |= GGML_TENSOR_FLAG_INPUT;
    struct ggml_tensor * node_src = ggml_add(ctx, t_src, t_src);
    struct ggml_cgraph * graph_src = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph_src, node_src);

    // Allocate graph on src
    bool ok_src = ggml_backend_sched_alloc_graph(sched_src, graph_src);
    assert(ok_src);
    assert(node_src->buffer != nullptr);

    // Create destination graph with input and operation
    struct ggml_tensor * t_dst = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10);
    t_dst->flags |= GGML_TENSOR_FLAG_INPUT;
    struct ggml_tensor * node_dst = ggml_add(ctx, t_dst, t_dst);
    struct ggml_cgraph * graph_dst = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph_dst, node_dst);

    // Allocate graph on dst (which creates a separate buffer)
    bool ok_dst = ggml_backend_sched_alloc_graph(sched_dst, graph_dst);
    assert(ok_dst);
    assert(node_dst->buffer != nullptr);
    assert(node_dst->buffer != node_src->buffer); // Different buffers initially

    // Share buffers from src to dst
    ggml_backend_sched_share_buffers(sched_dst, sched_src);

    // Reset dst scheduler allocation state before reallocating to avoid assertion failure
    ggml_backend_sched_reset(sched_dst);

    // Re-allocate graph on dst to bind to the shared buffer
    bool ok_dst_shared = ggml_backend_sched_alloc_graph(sched_dst, graph_dst);
    assert(ok_dst_shared);
    assert(node_dst->buffer == node_src->buffer); // Now they share the same buffer!

    // Clear buffers on dst before freeing to avoid double-free
    ggml_backend_sched_clear_buffers(sched_dst);

    // Free resources
    ggml_backend_sched_free(sched_src);
    ggml_backend_sched_free(sched_dst);
    ggml_free(ctx);
    ggml_backend_free(backend);

    printf("Shared compute buffers parameter is present and compiles successfully.\n");
    return 0;
}
