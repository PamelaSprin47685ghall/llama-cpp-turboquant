#pragma once

#include "ggml.h"

#include <algorithm>

// Constants for DKVT geometry-driven sizing.
//
// The speculative-verify path (draft-MTP) evaluates a small batch of draft tokens
// in one graph; 8 is the largest decode-style batch that still triggers the TG
// transcode path instead of being treated as a prefill. It is used to size the
// TG compute cap conservatively.
static constexpr uint32_t DKVT_SPEC_VERIFY_MAX_BATCH = 8;

// KV cache row padding lower bound used by get_n_kv() / dkvt_kv_rows_to_transcode.
static constexpr uint32_t DKVT_KV_ROWS_PAD_MIN = 256;

// TurboQuant uses 128-element Walsh-Hadamard groups for both packing and rotation.
static constexpr int32_t DKVT_TURBO_BLOCK_SIZE = 128;

// Heuristic Flash-Attention KV tile size used to compute the number of KV tiles
// for the softmax/scratch workspace. This value was empirically observed in the
// scheduler measurements for the target model; using a named constant makes the
// heuristic explicit and searchable.
static constexpr uint32_t DKVT_FA_KV_TILE_SIZE = 43;
static constexpr uint32_t DKVT_FA_KV_TILE_OVERHANG = DKVT_FA_KV_TILE_SIZE - 1;

// Heuristic concurrency factor for GDN (Mamba/GRIN) activation planes during PP.
// The base count covers the fixed-width recurrent overlap and the per-layer term
// covers the additional pipelined planes observed in reserve graphs.
static constexpr uint32_t DKVT_GDN_PP_CONCURRENCY_BASE = 6;
static constexpr uint32_t DKVT_GDN_PP_CONCURRENCY_PER_LAYER = 4;

// Allocation safety margin (10%) applied to the computed compute cap.
static constexpr uint32_t DKVT_ALLOC_MARGIN_NUM = 11;
static constexpr uint32_t DKVT_ALLOC_MARGIN_DENOM = 10;

// Identify turbo quantization types that can be upconverted during TG phase
static inline bool is_transcodable_type(ggml_type type) {
    return type == GGML_TYPE_TURBO4_0 || type == GGML_TYPE_TURBO2_0;
}

// TG phase target type for K: turbo types → F16, else unchanged
static inline ggml_type dkvt_tg_type_k(ggml_type orig) {
    return is_transcodable_type(orig) ? GGML_TYPE_F16 : orig;
}

// TG phase target type for V: turbo types → Q8_0, else unchanged
static inline ggml_type dkvt_tg_type_v(ggml_type orig) {
    return is_transcodable_type(orig) ? GGML_TYPE_Q8_0 : orig;
}

// Update tensor type and recompute all strides using ggml_row_size for correctness
static inline size_t dkvt_align_up(size_t n, size_t alignment) {
    if (alignment <= 1) {
        return n;
    }
    return GGML_PAD(n, alignment);
}

static inline void dkvt_update_strides(ggml_tensor * tensor, ggml_type type) {
    tensor->type = type;
    tensor->nb[0] = ggml_type_size(type);
    tensor->nb[1] = ggml_row_size(type, tensor->ne[0]);
    tensor->nb[2] = tensor->nb[1] * tensor->ne[1];
    tensor->nb[3] = tensor->nb[2] * tensor->ne[2];
}

// Compute the number of KV rows that actually need transcoding.
// Mirrors get_n_kv() padding logic: align used_max_p1 up to n_pad_cur,
// clamp to n_pad_cur floor, then cap at cells_size to avoid transcending
// empty cells beyond the allocated buffer.
//
//   n_pad_cur = max(n_pad, DKVT_KV_ROWS_PAD_MIN)
//   candidate  = GGML_PAD(used_max_p1, n_pad_cur)
//   result     = min(cells_size, max(n_pad_cur, candidate))
static inline uint32_t dkvt_kv_rows_to_transcode(
    uint32_t used_max_p1, uint32_t cells_size, uint32_t n_pad) {
    const uint32_t n_pad_cur = std::max(n_pad, DKVT_KV_ROWS_PAD_MIN);
    const uint32_t candidate  = GGML_PAD(used_max_p1, n_pad_cur);
    const uint32_t clamped    = std::max(n_pad_cur, candidate);
    return std::min(cells_size, clamped);
}

// Decide whether to trigger DKVT transcode before building the TG graph.
// Returns true only for genuine TG-style decode steps where KV cache format
// must switch from PP (turbo) to TG (high-precision).
//
// kv_has_data: true if the KV cache already contains tokens from a previous
//   prefill.  When false we are in prefill phase and must NEVER transcode —
//   even if n_tokens==1 (a 1-token prompt is PP, not TG decode).
//
// Rules:
//   - !kv_has_data → false (prefill: KV cache empty, nothing to transcode)
//   - n_outputs == 0 → false (no output, nothing to decode)
//   - n_outputs < n_tokens → false (mid-prefill: only last token outputs)
//   - n_tokens == 1 && n_outputs >= 1 → true (single-token decode)
//   - n_outputs == n_tokens && n_tokens <= DKVT_SPEC_VERIFY_MAX_BATCH → true (speculative verify)
//   - n_outputs == n_tokens && n_tokens > DKVT_SPEC_VERIFY_MAX_BATCH → false (large prefill batch)
static inline bool dkvt_should_transcode_before_graph(
        uint32_t n_tokens, uint32_t n_outputs, bool kv_has_data) {
    if (!kv_has_data) return false;
    if (n_outputs == 0) return false;
    if (n_outputs < n_tokens) return false;
    if (n_tokens == 1 && n_outputs >= 1) return true;
    if (n_outputs == n_tokens && n_tokens <= DKVT_SPEC_VERIFY_MAX_BATCH) return true;
    return false;
}

// After TG transcode, a large prefill step must re-bind PP layout
// (turbo types + PP offsets) before writing KV; otherwise graph alloc
// uses PP compute region while tensors stay TG-bound.
//
// kv_has_data: same semantics as in dkvt_should_transcode_before_graph.
static inline bool dkvt_should_reset_before_graph(
        uint32_t n_tokens, uint32_t n_outputs,
        bool is_transcoded_tg, bool kv_has_data) {
    if (!is_transcoded_tg) {
        return false;
    }
    return !dkvt_should_transcode_before_graph(n_tokens, n_outputs, kv_has_data);
}

// Return the correct compute cap for the current phase.
// PP: compute occupies [0, size_act_pp), KV occupies [size_act_pp, ...)
// TG (mirror): KV occupies [0, kv_total_tg), compute occupies [kv_total_tg, kv_total_tg + size_act_tg)
// The cap tells the graph allocator the maximum offset it may use.
// For TG, we also need a base_offset so the allocator starts after the KV region.
static inline size_t dkvt_union_compute_cap_bytes(
        bool is_transcoded_tg, size_t size_act_pp, size_t size_act_tg,
        size_t kv_total_tg = 0) {
    if (is_transcoded_tg) {
        return kv_total_tg + size_act_tg;
    }
    return size_act_pp;
}

static inline size_t dkvt_union_compute_base_offset(
        bool is_transcoded_tg, size_t kv_total_tg = 0) {
    return is_transcoded_tg ? kv_total_tg : 0;
}

// Compute activation bytes for one phase (PP or TG) from model geometry.
// n_tokens is the phase-specific token count (n_ubatch for PP, 1..DKVT_SPEC_VERIFY_MAX_BATCH for TG).
// n_kv_cells is the KV context length used for Flash-Attention mask/scratch sizing.
// gdn_conv_output_bytes is the single-recurrent-layer conv output for n_tokens.
static inline size_t dkvt_compute_activation_bytes(
        size_t n_tokens,
        size_t n_embd,
        size_t n_head,
        size_t n_embd_k_gqa,
        size_t n_embd_v_gqa,
        size_t ffn_dim,
        size_t n_recr_layers,
        size_t n_expert_used,
        size_t n_kv_cells,
        bool flash_attn,
        size_t gdn_conv_output_bytes,
        size_t n_vocab = 0) {
    // n_vocab is added only for phases that emit output logits (TG / verify).
    // It is left at 0 for PP sizing because PP logits are not always allocated on the
    // GPU backend and would otherwise bloat the union buffer unnecessarily.
    const size_t base_act = n_embd + n_embd_k_gqa + n_embd_v_gqa + n_embd + 3 * ffn_dim +
                            (n_vocab > 0 ? n_vocab : 0);
    const size_t score_len = n_head * n_tokens;
    const size_t gdn_wide = sizeof(float) * n_tokens * n_embd;
    const size_t base_est = sizeof(float) * (n_tokens * base_act + score_len);
    const size_t gdn_slabs = DKVT_GDN_PP_CONCURRENCY_BASE + n_recr_layers / DKVT_GDN_PP_CONCURRENCY_PER_LAYER;
    const size_t gdn_base = gdn_wide * gdn_slabs + sizeof(float) * score_len;
    const size_t moe_experts = n_expert_used > 0 ? n_expert_used : 1;
    const size_t moe_ffn_plane = sizeof(float) * n_tokens * ffn_dim;
    // ffn_moe_down: [n_embd, n_expert_used, n_tokens]
    const size_t moe_down_out = sizeof(float) * n_tokens * n_embd * moe_experts;
    // Peak concurrent planes: GDN base + one MoE gate/up/down layer plus a deeper down plane.
    size_t gdn_moe_peak = gdn_base + moe_ffn_plane + 2 * moe_down_out;
    if (gdn_conv_output_bytes > 0) {
        gdn_moe_peak = std::max(gdn_moe_peak, gdn_base + gdn_conv_output_bytes);
    }
    size_t size_act = std::max(base_est, gdn_moe_peak);
    if (flash_attn && n_kv_cells > 0) {
        const size_t fa_kq_mask = sizeof(uint16_t) * n_kv_cells * n_tokens;
        const size_t fa_kv_tiles = (n_kv_cells + DKVT_FA_KV_TILE_OVERHANG) / DKVT_FA_KV_TILE_SIZE;
        const size_t fa_scratch = sizeof(float) * n_head * n_tokens * fa_kv_tiles;
        const size_t fa_moe_peak = gdn_base + fa_kq_mask + fa_scratch + moe_down_out;
        // Some reserve graphs also see the MoE gate expert plane coexisting with GDN+MoE.
        const size_t moe_gate_peak = gdn_moe_peak + moe_ffn_plane;
        size_act = std::max(size_act, std::max(fa_moe_peak, moe_gate_peak));
    }
    return (size_act * DKVT_ALLOC_MARGIN_NUM) / DKVT_ALLOC_MARGIN_DENOM;
}

// Backward-compatible alias for the existing PP-only unit test.
static inline size_t dkvt_pp_compute_activation_bytes(
        size_t n_ubatch,
        size_t n_embd,
        size_t n_head,
        size_t n_embd_k_gqa,
        size_t n_embd_v_gqa,
        size_t ffn_dim,
        size_t n_recr_layers,
        size_t n_expert_used,
        size_t n_kv_cells,
        bool flash_attn,
        size_t gdn_conv_output_bytes,
        size_t n_vocab = 0) {
    return dkvt_compute_activation_bytes(
        n_ubatch, n_embd, n_head, n_embd_k_gqa, n_embd_v_gqa, ffn_dim,
        n_recr_layers, n_expert_used, n_kv_cells, flash_attn, gdn_conv_output_bytes, n_vocab);
}
