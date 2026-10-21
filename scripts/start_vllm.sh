#!/usr/bin/env bash
# -*- coding: utf-8 -*-
"""DeepFashion vLLM 服务一键启动脚本。

封装了部署所需的所有环境准备（conda 环境 / CUDA_HOME / 禁用 flashinfer 采样器），
避免手工敲 vllm serve 时漏环境变量导致启动失败（详见 docs/部署说明.md §6.3/§6.4）。

用法（在仓库任意位置）::

    bash scripts/start_vllm.sh              # 默认：TP=2 双卡 + 128k 上下文
    bash scripts/start_vllm.sh single       # 单卡 + 32k 上下文
    bash scripts/start_vllm.sh tp2          # 显式指定双卡模式

前台运行（Ctrl+C 停止）。后台运行可自行 nohup::

    nohup bash scripts/start_vllm.sh > logs/vllm_serve.log 2>&1 &
"""

set -euo pipefail

# 定位仓库根（脚本在 scripts/ 下）
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

# ---- 环境准备（部署文档 §2/§3）----
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate vllm
export CUDA_HOME=/home/maotai/.conda/envs/vllm/lib/python3.11/site-packages/nvidia/cu13
export VLLM_USE_FLASHINFER_SAMPLER=0

MODEL="experiments/qwen3vl8b/main/weights/merged/qwen3vl8b_dfattr_merged"
MODE="${1:-tp2}"

case "$MODE" in
    single)
        ARGS="--gpu-memory-utilization 0.55 --max-model-len 32768"
        ;;
    tp2)
        ARGS="--gpu-memory-utilization 0.55 --max-model-len 131072 --tensor-parallel-size 2"
        ;;
    *)
        echo "[err] 未知模式: $MODE（可选 single / tp2）" >&2
        exit 1
        ;;
esac

echo "[vllm] 模式=$MODE 工作目录=$ROOT"
exec vllm serve "$MODEL" --served-model-name deepfashion --port 8000 $ARGS
