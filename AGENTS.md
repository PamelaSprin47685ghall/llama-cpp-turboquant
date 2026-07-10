---
import:
  - 计划.md
---

# Ornith 35B 在 RTX 4060 + CPU 平台推理与调优实践指南

本指南根据最新对共享计算图缓冲区、混合内存上下文多态解耦、以及非对称层映射保护的重大修复，推翻了以往关于混合精度量化（--custom-q）、Trellis量化（IQ4_KT、IQ2_KT等）的复杂结论。经实际评估与测试，本机最优部署方案为“Q6_K一体化量化”结合“KV Cache Turbo Quant”以及一系列底层调优参数。

---

## 核心量化配置

### 1. 权重配置：混合重要性量化（IQ4_NL + Q6_K）
- **方案**：主模型的 MoE 专家层（CPU 侧计算）使用基于重要性矩阵的 **`IQ4_NL`** 量化权重；非 MoE 层 / Attention 层 / SSM 循环层（GPU 显存内计算）采用原生的 **`Q6_K`** 格式以防止数值发散。
- **依据**：MoE 专家层体积庞大（占 85% 权重），通过 IQ4_NL 可极大地缩减模型总体积（从 27.19 GiB 压缩至 18.68 GiB），有效减轻了 CPU 内存带宽瓶颈和显存消耗。而非 MoE 层保留在 Q6_K 以保证循环隐状态（recurrent state）的数值稳定性。

### 2. KV Cache配置：Turbo Quant
- **方案**：启用低比特 KV Cache，推荐参数：
  - **K Cache**：`turbo4` (4-bit 压缩)
  - **V Cache**：`turbo2` (2-bit 压缩)
- **依据**：35B 级别模型在普通 RTX 4060（8GB VRAM）下显存极度受限，在 256000 (256k) 核心大上下文下，传统的 F16 或 Q8_0 KV Cache 会瞬间导致数百 GB 的显存需求而崩溃。K: turbo4 + V: turbo2 极大幅度压缩 KV 缓存体积（K-cache 节省约75%，V-cache 节省约87%），是在 8GB 显存下稳定运行 256k 核心上下文的唯一物理底座。对于 32k 以下短上下文，可选用 K: q8_0 / f16 + V: turbo4 保守搭配。

### 3. 计算图共享：MTP 伴生计算图共享 (Share Compute Graph)
- **方案**：Ornith 没有专门 the 外部 `draft.gguf` 模型，而是集成了 MTP (Multi-Token Prediction) 预测层，通过自推测解码来实现推理加速。启用计算图共享补丁时，由主模型推理上下文与 MTP 伴生上下文（Companion Context）共享计算图与调度器内存（通过 `share_compute_buffers_with` 指向目标主模型 `ctx_tgt`）。
- **依据**：消除 MTP 自推测评估时的 VRAM 冗余分配与占用。内存分配器的预留机制自动设为 `max(main, mtp)`，防止 MTP 伴生上下文在共享后因为图节点大小多于主模型而触发 `realloc` 重分配，从而彻底杜绝由此引起的内存撕裂、主上下文缓存破坏与 use-after-free 崩溃风险。

---

## 稳定性与高可用保障（核心修复细节）

本补丁彻底攻克了 Qwen3.5-MoE 混合架构下 speculative decoding 在显存受限以及多图切换时的崩溃隐患：

1. **共享缓冲区生命周期防悬空（Borrowed Buffer Protect）**：在 `ggml_backend_sched_share_buffers` 中引入 `is_borrowed` 标志位。当伴生预测上下文的调度器（dst）尝试释放或重新分配借用的缓冲区时，跳过对主模型分配器（src）物理内存的 `ggml_vbuffer_free` 调用，改为安全释放绑定并为 dst 分配私有缓冲区，彻底杜绝了 Use-After-Free 内存撕裂。
2. **多态上下文动态解耦（Hybrid vs Plain Decoupling）**：将 `get_kv_cache_context` 中不安全的 `static_cast` 改为 `dynamic_cast`，在 RTTI 层面区分混合架构上下文 `llama_memory_hybrid_context` 与普通 KV 缓存上下文，消除了因虚表与偏移不一致产生的 SIGSEGV 段错误。
3. **初始化失败源头短路（Failed Prepare Check）**：当主模型 Attention 处于 `FAILED_PREPARE` 状态时，`get_kv_cache_context` 立即向上返回 `nullptr`，配合 downstream 算子节点的空引用保护，杜绝了下游图构建对空 `sinfos` 越界检索导致的 `std::unordered_map::at` 崩溃。
4. **非对称层级映射防护（Layer Mapping Guard）**：在 `cpy_k` / `cpy_v` 拷贝操作及注意力 `get_k` / `get_v` 计算前加入 `has_layer(il)` 哨兵检查。当 MTP 计算图检索到超出其伴生层数空间的非法层索引时，优雅退化至默认输入，避免触发层字典映射越界异常。

---

## 性能表现与评测数据

在启用以上补丁及黄金调优参数下，Ornith 35B 在本机单 GPU (RTX 4060) + CPU 混合平台取得了极佳的性能表现：

* **Prompt Prefill Speed**: **862.89 t/s** (大文本) / **50.1 t/s ~ 55.9 t/s** (16 tokens 短文本)
* **Token Generation Speed**: **22.39 t/s** (无投机基准) / **28.76 t/s ~ 38.01 t/s** (MTP 投机解码下测得，8K 短文本加速至 38.01 t/s)
* **MTP 投机接受率 (Draft Acceptance Rate)**:
  * 整体测试：**63.3%** (326 accepted / 515 generated)
* **净加速比**: 实现了 **1.214x ~ 1.30x** 的净吞吐加速（在 CPU 专家层 offload 极高延迟惩罚下取得显著性能优势）。
* **显存状况**: 双图完全共享 compute 缓冲区，无二次重分配 OOM，长文本连续解码状态稳定。

---

## 极致性能调优精华参数

### 1. 内存与绑核优化 (CPU/NUMA)
- **`numactl --interleave=all` 与 `--numa numactl`**：强制在所有可用 NUMA 节点（多通道内存）之间交叉分配内存页，最大化 CPU 侧的权重读取带宽，避免单通道饱和瓶颈。
- **`taskset -c 0-15` (P-Core 绑定)**：将线程严格限制在前 16 个逻辑核（0-15 为 P-Core 物理核心及其超线程），完全避开 Intel CPU 的慢速 E-Core（16-31），防止混合架构调度导致的 Prefill 与 Decode 性能断崖式下跌。

### 2. GPU 显存分流与算子优化
- **`-ngl 999`**：将所有能放入显存的层尽可能多地 offload 到 GPU。
- **`-ot "exps=CPU"`**：对 MoE 架构模型，将 Expert 相关的矩阵乘法（ffn_down/gate/up_exps）定向分流到 CPU 计算，而将 Attention/Shared Experts 保留在 GPU VRAM。这是在 8GB 显存下运行 35B 级别 MoE 模型不 OOM 且保持较高吞吐的关键。
- **`-fa on`**：强制启用 Flash Attention，优化自注意力机制的 VRAM 占用与计算效率。
- **`-b 3072 -ub 3072`**：增大 logical batch 和 physical batch 尺寸至 3072，配合大 Prefill 上下文，榨干 GPU INT8 MMA 的 Tensor Core 并发吞吐性能。

### 3. 过时/废弃参数的重要演进说明（对比旧 start.sh）
- **`-khad` 与 `-vhad` (已废弃/隐式集成)**：在新版中被完全开除。Walsh-Hadamard 旋转已隐式集成到 `turbo4/turbo2` 等 KV Cache 的编解码底层，只要指定该类型即可自动生效，无需也不支持显式配置。
- **`--tg-expert-used` (已废弃/自动读取)**：在新版中被完全开除。该参数直接由模型文件的 GGUF 元数据（`$arch.expert_used_count`）在引擎初始化时由 C++ 自动读取，不支持也无需通过命令行动态覆盖。
- **`-sas` 与 `-muge` (已废弃/代码移除)**：在新版中被完全开除。这两个参数已被底层执行器与编译器融合优化，强行传入会导致 `llama-cli/llama-server` 命令行解析器抛出 `invalid argument` 错误闪退。
- **投机解码复合参数格式演进**：已废弃旧的 `mtp:n_max=3...` 复合参数（会导致命令行解析报错），改用符合 `common/arg.cpp` 最新标准的 `--spec-type draft-mtp`、`--spec-draft-n-max` 等独立参数配置。

---

## 推荐启动脚本示例

### 方案 A：本地调试交互脚本 (llama-cli)
```bash
#!/usr/bin/env bash
export LD_LIBRARY_PATH=/home/kunweiz/Desktop/Ornith/llama-cpp-turboquant/build/bin

numactl --interleave=all taskset -c 0-15 ./build/bin/llama-cli \
    -m /home/kunweiz/Desktop/Ornith/weights/ornith-1.0-35b-IQ4NL-MTP-final.gguf \
    -ngl 999 \
    -ot "exps=CPU" \
    --cache-type-k turbo4 \
    --cache-type-v turbo2 \
    -fa on \
    -b 3072 \
    -ub 3072 \
    -t 8 \
    -tb 8 \
    -c 256000 \
    --numa numactl \
    --spec-type draft-mtp \
    --spec-draft-n-max 2 \
    --spec-draft-p-min 0.16 \
    --spec-draft-n-min 2 \
    --spec-draft-type-k turbo4 \
    --spec-draft-type-v turbo2 \
    -st
```

### 方案 B：生产级服务化高可用脚本 (llama-server + frpc 穿透)
该脚本利用 `trap` 捕获 `EXIT` 信号以优雅、无缝地清理在后台运行的 `frpc` 隧道守护进程，并在 `turbo4/turbo2` 的极限 KV Cache 压缩下安全运行 `256000` 核心上下文大小：
```bash
#!/usr/bin/env bash
set -euo pipefail

DIR="/home/kunweiz/Desktop/Ornith/llama-cpp-turboquant"
PORT="${PORT:-8080}"
HOST="${HOST:-127.0.0.1}"
THREADS="${THREADS:-8}"
BATCH="${BATCH:-3072}"
UBATCH="${UBATCH:-3072}"
CORES="${CORES:-0-15}"
CTX_SIZE="${CTX_SIZE:-256000}" # 256k 核心大上下文
MODEL="${MODEL:-/home/kunweiz/Desktop/Ornith/weights/ornith-1.0-35b-IQ4NL-MTP-final.gguf}"

export LD_LIBRARY_PATH="${DIR}/build/bin:${LD_LIBRARY_PATH:-}"

FRPC_BIN="${FRPC_BIN:-/usr/local/bin/frpc}"
FRPC_CONF="${FRPC_CONF:-${DIR}/build/bin/frpc_temp.toml}"

FRPC_PID=""
cleanup() {
    echo "[cleanup] 正在清理服务资源..." >&2
    if [[ -n "${FRPC_PID}" ]] && kill -0 "${FRPC_PID}" 2>/dev/null; then
        echo "[cleanup] 正在终止 frpc 守护进程 (PID: ${FRPC_PID})..." >&2
        kill "${FRPC_PID}" 2>/dev/null || true
        wait "${FRPC_PID}" 2>/dev/null || true
    fi
    if [[ -f "${FRPC_CONF}" ]]; then
        rm -f "${FRPC_CONF}"
    fi
    echo "[cleanup] 清理完成。" >&2
}
trap cleanup EXIT

if [[ -x "${FRPC_BIN}" ]]; then
    if [[ -z "${FRPC_AUTH_TOKEN:-}" ]]; then
        echo "[start] 警告: 未设置 FRPC_AUTH_TOKEN，跳过 frp 公网穿透" >&2
    else
        mkdir -p "$(dirname "${FRPC_CONF}")"
        cat > "${FRPC_CONF}" <<EOF
serverAddr = "ai.kw92.cyou"
serverPort = 7000
auth.token = "${FRPC_AUTH_TOKEN}"

[[proxies]]
name = "ornith_prod"
type = "tcp"
localIP = "127.0.0.1"
localPort = ${PORT}
remotePort = 8000
EOF
        chmod 600 "${FRPC_CONF}"
        "${FRPC_BIN}" -c "${FRPC_CONF}" &
        FRPC_PID=$!
        echo "[start] frpc 隧道已建立: 127.0.0.1:${PORT} -> ai.kw92.cyou:8000" >&2
    fi
else
    echo "[start] 警告: 未找到 ${FRPC_BIN} 可执行文件，跳过 frp 公网穿透" >&2
fi

echo "[start] 正在启动 llama-server 服务..." >&2
numactl --interleave=all taskset -c "${CORES}" "${DIR}/build/bin/llama-server" \
    -m "${MODEL}" \
    --jinja \
    --port "${PORT}" \
    --host "${HOST}" \
    --ctx-size "${CTX_SIZE}" \
    -ngl 999 \
    -ot "exps=CPU" \
    --cache-type-k turbo4 \
    --cache-type-v turbo2 \
    -fa on \
    -b "${BATCH}" \
    -ub "${UBATCH}" \
    -t "${THREADS}" \
    -tb "${THREADS}" \
    --numa numactl \
    --spec-type draft-mtp \
    --spec-draft-n-max 2 \
    --spec-draft-p-min 0.16 \
    --spec-draft-n-min 2 \
    --spec-draft-type-k turbo4 \
    --spec-draft-type-v turbo2
```
