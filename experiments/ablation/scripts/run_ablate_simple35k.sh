#!/bin/bash
# 指令范式消融：simple 全量训练（3 卡 DDP，与主训练一致；simple 指令 + 全量 35,276 条）
cd "$(dirname "$0")/../../.."
export PATH=/home/maotai/.conda/envs/lf/bin:$PATH
export OMP_NUM_THREADS=16
mkdir -p experiments/ablation/weights/logs
LOG=experiments/ablation/weights/logs/train_simple35k.log
echo "python: $(which python), nproc: $(nvidia-smi -L | wc -l)" >> $LOG

echo "[$(date +%H:%M:%S)] start simple-35k train (3GPU DDP)" >> $LOG
llamafactory-cli train experiments/ablation/config/qwen3vl8b_dfattr_ablate_simple35k_train.yaml \
  > experiments/ablation/weights/logs/train_simple35k_3gpu.log 2>&1
echo "[$(date +%H:%M:%S)] done simple-35k train" >> $LOG

echo "ALL_DONE" >> $LOG
