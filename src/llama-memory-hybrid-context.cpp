#include "llama-memory-hybrid.h"
#include "llama-kv-cache.h"
#include "llama-context.h"
#include <cassert>

llama_memory_hybrid_context::llama_memory_hybrid_context(llama_memory_status status) : status(status) {}

llama_memory_hybrid_context::llama_memory_hybrid_context(llama_memory_hybrid * mem) :
    ctx_attn(mem->get_mem_attn()->init_full()),
    ctx_recr(mem->get_mem_recr()->init_full()),
    status(llama_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
}

llama_memory_hybrid_context::llama_memory_hybrid_context(
        llama_memory_hybrid * mem,
              llama_context * lctx,
                       bool   optimize) :
    ctx_attn(mem->get_mem_attn()->init_update(lctx, optimize)),
    ctx_recr(mem->get_mem_recr()->init_update(lctx, optimize)),
    status(llama_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
}

llama_memory_hybrid_context::llama_memory_hybrid_context(
              llama_memory_hybrid * mem,
                  slot_info_vec_t   sinfos_attn,
        std::vector<llama_ubatch>   ubatches) :
    ubatches(std::move(ubatches)),
    ctx_attn(new llama_kv_cache_context(mem->get_mem_attn(), std::move(sinfos_attn), this->ubatches)),
    ctx_recr(new llama_memory_recurrent_context(mem->get_mem_recr(), this->ubatches)),
    status(llama_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
}

bool llama_memory_hybrid_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    ctx_attn->next();
    ctx_recr->next();

    if (++i_next >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_memory_hybrid_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    bool res = true;

    res = res & ctx_attn->apply();
    res = res & ctx_recr->apply();

    return res;
}

llama_memory_status llama_memory_hybrid_context::get_status() const {
    return status;
}

const llama_ubatch & llama_memory_hybrid_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return ubatches[i_next];
}

const llama_kv_cache_context * llama_memory_hybrid_context::get_attn() const {
    return static_cast<const llama_kv_cache_context *>(ctx_attn.get());
}

ggml_tensor * llama_memory_hybrid_context::get_turbo_rot_forward() const {
    return ctx_attn ? ctx_attn->get_turbo_rot_forward() : nullptr;
}

ggml_tensor * llama_memory_hybrid_context::get_turbo_rot_inverse() const {
    return ctx_attn ? ctx_attn->get_turbo_rot_inverse() : nullptr;
}

ggml_tensor * llama_memory_hybrid_context::get_turbo_innerq_scale_inv() const {
    return ctx_attn ? ctx_attn->get_turbo_innerq_scale_inv() : nullptr;
}

const llama_memory_recurrent_context * llama_memory_hybrid_context::get_recr() const {
    return static_cast<const llama_memory_recurrent_context *>(ctx_recr.get());
}

const llama_kv_cache_context * llama_memory_hybrid_context::as_kv_cache_context() const {
    return ctx_attn ? ctx_attn->as_kv_cache_context() : nullptr;
}
