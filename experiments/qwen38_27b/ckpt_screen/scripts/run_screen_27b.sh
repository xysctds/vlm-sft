#!/bin/bash
# 27B checkpoint 快筛（15 个点 × 500 条 val）—— 纯推理，不需要训练
#
# 与 8B 的 run_screen.sh 的差异：8B 是 3 卡各跑一份 8B（每卡放得下），
# 27B（bf16 54GB）单卡放不下，所以用 infer_27b.py 的 device_map="auto" 3 卡模型并行，
# **串行**跑 15 个点（每点约 11 分钟，合计约 2.8 小时），不再三路并行。
#
# 用法：
#   bash experiments/qwen38_27b/ckpt_screen/scripts/run_screen_27b.sh          # 跑全部 15 个快筛点
#   bash experiments/qwen38_27b/ckpt_screen/scripts/run_screen_27b.sh 900 2100 # 只跑指定步数
#
# 断点续跑：infer_27b.py 会跳过已生成的记录，重跑本脚本即可。
set -u
cd "$(dirname "$0")/../../../.."
export PATH=/home/maotai/.conda/envs/lf/bin:$PATH
export OMP_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG=experiments/qwen38_27b/logs
mkdir -p "$LOG"
BATCH_SIZE="${BATCH_SIZE:-32}"
CKPTS="${*:-100 300 500 700 900 1100 1300 1500 1700 1900 2100 2300 2500 2700 2793}"

echo "[$(date +%F' '%H:%M:%S)] === 27B ckpt screen start (batch=$BATCH_SIZE): $CKPTS ===" | tee -a "$LOG/schedule_27b.log"
for c in $CKPTS; do
    echo "[$(date +%F' '%H:%M:%S)] start checkpoint-$c" | tee -a "$LOG/schedule_27b.log"
    python experiments/qwen38_27b/main/scripts/infer_27b.py \
        --config "experiments/qwen38_27b/ckpt_screen/config/ckpt27b_${c}.yaml" \
        --batch_size "$BATCH_SIZE" >> "$LOG/ckpt27b_${c}.log" 2>&1
    echo "[$(date +%F' '%H:%M:%S)] done  checkpoint-$c (exit=$?)" | tee -a "$LOG/schedule_27b.log"
done
echo "CKPT_SCREEN_27B_DONE $(date +%H:%M:%S)" | tee -a "$LOG/schedule_27b.log"
