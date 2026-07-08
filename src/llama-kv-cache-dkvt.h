#pragma once

#include "ggml.h"

// Identify turbo quantization types that can be upconverted during TG phase
static inline bool is_transcodable_type(ggml_type type) {
    return type == GGML_TYPE_TURBO4_0 || type == GGML_TYPE_TURBO2_0;
}

// Update tensor type and recompute all strides using ggml_row_size for correctness
static inline void dkvt_update_strides(ggml_tensor * tensor, ggml_type type) {
    tensor->type = type;
    tensor->nb[0] = ggml_type_size(type);
    tensor->nb[1] = ggml_row_size(type, tensor->ne[0]);
    tensor->nb[2] = tensor->nb[1] * tensor->ne[1];
    tensor->nb[3] = tensor->nb[2] * tensor->ne[2];
}
