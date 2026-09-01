#!/bin/bash
# 指令范式消融 3 卡 DDP 串行训练：先 simple-5k 后 schema-5k
# llamafactory-cli 自带 launcher：检测到多 GPU 自动用 torchrun 启动 DDP（无需 accelerate，也勿设 CUDA_VISIBLE_DEVICES）
cd "$(dirname "$0")/../../.."
export PATH=/home/maotai/.conda/envs/lf/bin:$PATH
export OMP_NUM_THREADS=16
mkdir -p experiments/ablation/weights/logs
LOG=experiments/ablation/weights/logs/schedule3.log
echo "python: $(which python), nproc: $(nvidia-smi -L | wc -l)" >> $LOG

rm -rf experiments/ablation/weights/schema_5k experiments/ablation/weights/simple_5k

echo "[$(date +%H:%M:%S)] start simple-5k (3GPU DDP)" >> $LOG
llamafactory-cli train experiments/ablation/config/qwen3vl8b_dfattr_ablate_simple5k_train.yaml \
  > experiments/ablation/weights/logs/train_simple5k_3gpu.log 2>&1
echo "[$(date +%H:%M:%S)] done simple-5k" >> $LOG

echo "[$(date +%H:%M:%S)] start schema-5k (3GPU DDP)" >> $LOG
llamafactory-cli train experiments/ablation/config/qwen3vl8b_dfattr_ablate_schema5k_train.yaml \
  > experiments/ablation/weights/logs/train_schema5k_3gpu.log 2>&1
echo "[$(date +%H:%M:%S)] done schema-5k" >> $LOG

echo "ALL_DONE" >> $LOG
