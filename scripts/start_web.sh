#!/usr/bin/env bash
# DeepFashion Web demo 一键启动脚本（vLLM 服务 + Gradio 页面）。
#
# 封装全部环境准备（conda 环境 / CUDA_HOME / 禁用 flashinfer / GPU 分卡避让 /
# 健康检查 / 端口与日志管理），在仓库根任意位置运行即可把 Web demo 跑起来。
#
# 用法:
#   bash scripts/start_web.sh               # 启动 vLLM(默认双卡 TP=2 + 128k) + Gradio
#   bash scripts/start_web.sh single        # vLLM 单卡 + 32k 上下文
#   bash scripts/start_web.sh tp2           # 显式双卡 TP=2 + 128k 上下文
#   bash scripts/start_web.sh --vllm-only   # 只起 vLLM，不起 Gradio
#   bash scripts/start_web.sh --gradio-only # 只起 Gradio（要求 vLLM 已在跑）
#   bash scripts/start_web.sh --stop        # 停掉 vLLM + Gradio
#   bash scripts/start_web.sh --status      # 查看当前服务状态
#
# 说明:
# - vLLM 与 Gradio 查询模型是不同进程，本脚本自动把两者分配到不同 GPU：
#   vLLM 用显式指定的卡（默认双卡 TP=2 用 GPU0+1；single 用 GPU0），Gradio 查询模型
#   自动挑一块未被 vLLM 占用的空闲卡（避免同卡 OOM，见 docs/部署说明.md §6.6）。
# - Gradio 跑在 **vllm** 环境（gradio 6.26 + starlette 1.6 兼容），无需 llamafactory。
# - 日志分别写 logs/vllm_serve.log 与 logs/gradio.log。

set -euo pipefail

# 定位仓库根（脚本在 scripts/ 下）
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

VLLM_PORT=8000
GRADIO_PORT=7860
VLLM_LOG="$ROOT/logs/vllm_serve.log"
GRADIO_LOG="$ROOT/logs/gradio.log"

# ---- 工具函数 ----
running() { ss -tlnp 2>/dev/null | grep -q ":$1 "; }

pick_free_gpu() {
    # 找出显存占用最低的那张卡（返回其 index），避开已繁忙的卡
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' '
}

wait_port() {
    local port=$1 name=$2 wait_sec=$3
    for _ in $(seq 1 "$wait_sec"); do
        if running "$port"; then
            echo "[$name] 就绪 (端口 $port)"
            return 0
        fi
        sleep 1
    done
    echo "[$name] 启动超时（${wait_sec}s），请查日志: $VLLM_LOG / $GRADIO_LOG" >&2
    return 1
}

stop_services() {
    echo "[stop] 停止 vLLM(8000) + Gradio(7860) ..."
    pkill -f "[v]llm serve" 2>/dev/null || true
    pkill -f "[s]erve_gradio.py" 2>/dev/null || true
    sleep 2
    echo "[stop] 完成"
}

status() {
    echo "[status] vLLM(8000): $(running 8000 && echo RUNNING || echo STOPPED)"
    echo "[status] Gradio(7860): $(running 7860 && echo RUNNING || echo STOPPED)"
    echo "[status] 访问: http://127.0.0.1:7860"
}

# ---- 解析参数 ----
START_VLLM=1
START_GRADIO=1
MODE="tp2"   # 默认双卡 TP=2（与 start_vllm.sh 默认一致）
case "${1:-}" in
    ""|single|tp2) [ -n "${1:-}" ] && MODE="$1" ;;
    --vllm-only) START_GRADIO=0 ;;
    --gradio-only) START_VLLM=0 ;;
    --stop) stop_services; exit 0 ;;
    --status) status; exit 0 ;;
    *) echo "[err] 未知参数: ${1:-}（可选 single / tp2 / --vllm-only / --gradio-only / --stop / --status）" >&2; exit 1 ;;
esac

# ---- 启动 vLLM（复用 start_vllm.sh，避免重复实现启动逻辑）----
if [ "$START_VLLM" = "1" ]; then
    if running "$VLLM_PORT"; then
        echo "[vllm] 已在运行，跳过（8000 端口占用）"
    else
        # start_vllm.sh 已封装 conda / CUDA_HOME / flashinfer 禁用，默认 tp2。
        # 这里只指定其用哪几张卡（tp2=GPU0,1；single=GPU0），再后台调用并等端口就绪。
        if [ "$MODE" = "tp2" ]; then
            VLLM_GPU="0,1"
        else
            VLLM_GPU="0"
        fi
        echo "[vllm] 启动（模式=$MODE，GPU=$VLLM_GPU，复用 start_vllm.sh）..."
        CUDA_VISIBLE_DEVICES="$VLLM_GPU" nohup bash scripts/start_vllm.sh "$MODE" \
            > "$VLLM_LOG" 2>&1 &
        wait_port "$VLLM_PORT" "vllm" 240 || exit 1
    fi
fi

# ---- 启动 Gradio ----
if [ "$START_GRADIO" = "1" ]; then
    if ! running "$VLLM_PORT"; then
        echo "[gradio] 依赖的 vLLM(8000) 未运行，请先启动 vLLM（--vllm-only 或重跑）" >&2
        exit 1
    fi
    if running "$GRADIO_PORT"; then
        echo "[gradio] 已在运行，跳过（7860 端口占用）"
    else
        # 自动挑一张显存占用最低的卡给 Gradio 查询模型（避开 vLLM 所在卡）
        FREE_GPU=$(pick_free_gpu)
        echo "[gradio] 启动（查询模型放 GPU$FREE_GPU）..."
        CUDA_VISIBLE_DEVICES="$FREE_GPU" nohup conda run -n vllm \
            python scripts/serve_gradio.py > "$GRADIO_LOG" 2>&1 &
        wait_port "$GRADIO_PORT" "gradio" 60 || exit 1
    fi
fi

echo ""
echo "===== Web demo 就绪 ====="
echo "  vLLM  API : http://127.0.0.1:$VLLM_PORT/v1   (模型: deepfashion)"
echo "  Gradio 页面: http://127.0.0.1:$GRADIO_PORT"
echo "  日志      : $VLLM_LOG / $GRADIO_LOG"
echo "  停止      : bash scripts/start_web.sh --stop"
