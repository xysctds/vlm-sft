#!/bin/bash
# 主实验 27B：Qwen3.8-27B LoRA 训练（3 卡 ZeRO-3，等效 batch = 4×3×3 = 36，与 8B 主实验一致）
# llamafactory-cli 自带 launcher：检测到多 GPU 自动用 torchrun 启动（无需 accelerate，也勿设 CUDA_VISIBLE_DEVICES）
#
# 本环境跑通 27B 的关键（详见 config/qwen38_27b_dfattr_train_cli.yaml 注释）：
#   use_reentrant_gc: false      非重入检查点（reentrant 会绕过 ZeRO-3 参数释放，显存 4.2G -> 47G OOM）
#   PYTORCH_CUDA_ALLOC_CONF      expandable_segments 抑制显存碎片
cd "$(dirname "$0")/../../../.."
export PATH=/home/maotai/.conda/envs/lf/bin:$PATH
export OMP_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p experiments/qwen38_27b/logs
LOG=experiments/qwen38_27b/logs/train_27b.log
CFG=experiments/qwen38_27b/main/config/qwen38_27b_dfattr_train_cli.yaml

echo "python: $(which python), nproc: $(nvidia-smi -L | wc -l)" >> $LOG
echo "[$(date +%F' '%H:%M:%S)] start 27B train (3GPU ZeRO-3, batch 4x3x3=36)" >> $LOG

llamafactory-cli train "$CFG" >> $LOG 2>&1
rc=$?

echo "[$(date +%F' '%H:%M:%S)] exit=$rc" >> $LOG
if [ $rc -eq 0 ]; then echo "ALL_DONE" >> $LOG; else echo "FAILED" >> $LOG; fi
