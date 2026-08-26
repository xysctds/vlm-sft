#!/bin/bash
# 数据量消融中间点：schema 指令 + 20k 子集训练（3 卡 DDP，与主训练一致；唯一差异：max_samples 20000）
# llamafactory-cli 自带 launcher：检测到多 GPU 自动用 torchrun 启动 DDP（无需 accelerate，也勿设 CUDA_VISIBLE_DEVICES）
cd "$(dirname "$0")/../../.."
export PATH=/home/maotai/.conda/envs/lf/bin:$PATH
export OMP_NUM_THREADS=16
export NPROC_PER_NODE=3
LOG=llamafactory/saves/ablation/train20k.log
echo "python: $(which python), nproc: $(nvidia-smi -L | wc -l)" >> $LOG

echo "[$(date +%H:%M:%S)] start 20k train (3GPU DDP)" >> $LOG
llamafactory-cli train llamafactory/config/ablation/qwen3vl8b_dfattr_ablate_schema20k_train.yaml \
  > llamafactory/saves/ablation/train20k_3gpu.log 2>&1
echo "[$(date +%H:%M:%S)] done 20k train" >> $LOG

echo "ALL_DONE" >> $LOG
