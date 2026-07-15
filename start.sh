#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Ornith-1.0-35B 启动脚本 (llama-cpp-turboquant)
# 模型: ../weights/ornith-1.0-35b-IQ4NL-MTP-final.gguf (4bit版本)
# =============================================================================

DIR="$(cd "$(dirname "$0")" && pwd)"

PORT="${PORT:-8080}"
HOST="${HOST:-127.0.0.1}"
THREADS="${THREADS:-16}"
BATCH="${BATCH:-3072}"
UBATCH="${UBATCH:-3072}"
CORES="${CORES:-0-15}"
CTX_SIZE="${CTX_SIZE:-262144}"
MODEL="${MODEL:-/home/kunweiz/Desktop/Ornith/weights/ornith-1.0-35b-IQ4NL-MTP-final.gguf}"

export LD_LIBRARY_PATH="${DIR}/build/bin:${LD_LIBRARY_PATH:-}"

# -----------------------------------------------------------------------------
# frp 客户端: 把本地 ${PORT} 暴露到 ai.kw92.cyou:8000
# -----------------------------------------------------------------------------
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
    mkdir -p "$(dirname "${FRPC_CONF}")"
    cat > "${FRPC_CONF}" <<EOF
serverAddr = "ai.kw92.cyou"
serverPort = 7000
auth.token = "40a4c80c057d1c07e4bdcb8520f26c21"

[[proxies]]
name = "ornith"
type = "tcp"
localIP = "127.0.0.1"
localPort = ${PORT}
remotePort = 8000
EOF
    chmod 600 "${FRPC_CONF}"
    "${FRPC_BIN}" -c "${FRPC_CONF}" &
    FRPC_PID=$!
    echo "[start.sh] frpc 已启动: 127.0.0.1:${PORT} -> ai.kw92.cyou:8000" >&2
else
    echo "[start.sh] 警告: 未找到 ${FRPC_BIN}，跳过 frp 隧道" >&2
fi

echo "[start.sh] 正在启动 llama-server..." >&2
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
    --spec-draft-n-max 3 \
    --spec-draft-p-min 0.0 \
    --spec-draft-n-min 0
