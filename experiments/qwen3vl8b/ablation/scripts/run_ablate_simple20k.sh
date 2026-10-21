#!/bin/bash
# 指令范式消融：simple 20k 训练（3 卡 DDP，与主训练一致；simple 指令 + 20k 子集）
cd "$(dirname "$0")/../../../.."
export PATH=/home/maotai/.conda/envs/lf/bin:$PATH
export OMP_NUM_THREADS=16
mkdir -p experiments/qwen3vl8b/ablation/weights/logs
LOG=experiments/qwen3vl8b/ablation/weights/logs/train_simple20k.log
echo "python: $(which python), nproc: $(nvidia-smi -L | wc -l)" >> $LOG

echo "[$(date +%H:%M:%S)] start simple-20k train (3GPU DDP)" >> $LOG
llamafactory-cli train experiments/qwen3vl8b/ablation/config/qwen3vl8b_dfattr_ablate_simple20k_train.yaml \
  > experiments/qwen3vl8b/ablation/weights/logs/train_simple20k_3gpu.log 2>&1
echo "[$(date +%H:%M:%S)] done simple-20k train" >> $LOG

echo "ALL_DONE" >> $LOG
