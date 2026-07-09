#include "llama-kv-cache.h"
#include "llama-kv-cache-dkvt.h"
#include "llama-impl.h"
#include "llama-model.h"
#include <algorithm>

void llama_kv_cache::init_dkvt_sum_kv_sizes(size_t gpu_alignment) {
    dkvt_k_size_pp = 0;
    dkvt_k_size_tg = 0;
    dkvt_v_size_pp = 0;
    dkvt_v_size_tg = 0;

    size_t k_off_pp = 0;
    size_t v_off_pp = 0;
    size_t k_off_tg = 0;
    size_t v_off_tg = 0;

    const uint32_t kv_size = get_size();

    for (size_t i = 0; i < layers.size(); ++i) {
        if (layers[i].k && (layers[i].k->flags & GGML_TENSOR_FLAG_EXT)) {
            layers[i].k_offset_pp = k_off_pp;
            layers[i].k_size_pp   = ggml_nbytes(layers[i].k);
            k_off_pp += GGML_PAD(layers[i].k_size_pp, gpu_alignment);

            layers[i].k_offset_tg = k_off_tg;
            ggml_type target_type_k = dkvt_tg_type_k(layers[i].orig_type_k);
            layers[i].k_size_tg = ggml_row_size(target_type_k, layers[i].k->ne[0]) *
                                  layers[i].k->ne[1] * layers[i].k->ne[2] * layers[i].k->ne[3];
            k_off_tg += GGML_PAD(layers[i].k_size_tg, gpu_alignment);
        }

        if (layers[i].v && (layers[i].v->flags & GGML_TENSOR_FLAG_EXT)) {
            layers[i].v_offset_pp = v_off_pp;
            layers[i].v_size_pp   = ggml_nbytes(layers[i].v);
            v_off_pp += GGML_PAD(layers[i].v_size_pp, gpu_alignment);

            layers[i].v_offset_tg = v_off_tg;
            ggml_type target_type_v = dkvt_tg_type_v(layers[i].orig_type_v);
            if (v_trans) {
                layers[i].v_size_tg = ggml_row_size(target_type_v, kv_size) *
                                      (layers[i].v->ne[1] / kv_size) * layers[i].v->ne[2] * layers[i].v->ne[3];
            } else {
                layers[i].v_size_tg = ggml_row_size(target_type_v, layers[i].v->ne[0]) *
                                      layers[i].v->ne[1] * layers[i].v->ne[2] * layers[i].v->ne[3];
            }
            v_off_tg += GGML_PAD(layers[i].v_size_tg, gpu_alignment);
        }

        dkvt_k_size_pp += layers[i].k_size_pp;
        dkvt_k_size_tg += layers[i].k_size_tg;
    }

    dkvt_v_size_pp = v_off_pp;
    dkvt_k_size_pp = k_off_pp;
    dkvt_v_size_tg = v_off_tg;
    dkvt_k_size_tg = k_off_tg;
}

void llama_kv_cache::init_dkvt_compute_activation_size(size_t n_ubatch) {
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

    const size_t n_tokens_tg = std::min<size_t>(n_ubatch, DKVT_SPEC_VERIFY_MAX_BATCH);
    size_act_tg = dkvt_compute_activation_bytes(
        n_tokens_tg, n_embd, n_head, n_embd_k_gqa, n_embd_v_gqa, ffn_dim,
        n_recr_layers, n_expert_used_eff, n_kv_cells, true, gdn_conv_output_bytes(n_tokens_tg),
        (size_t) model.vocab.n_tokens());
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

void llama_kv_cache::disable_dkvt_ext_flags() {
    for (size_t i = 0; i < layers.size(); ++i) {
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
