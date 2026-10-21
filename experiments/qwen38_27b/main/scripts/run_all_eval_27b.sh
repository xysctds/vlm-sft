#!/bin/bash
# 主实验 27B：依次跑完 7 个评测变体（不含 export，导出已在 2026-09-26 05:11 完成）。
#
# 每个变体单独起一个进程：模型只加载一次即处理完该变体的全部 4410 条，
# 变体之间互不影响；任一变体失败也不阻断后续变体。
# 断点续跑：直接重跑本脚本即可，infer_27b.py 会跳过已生成的记录。
set -u
cd "$(dirname "$0")/../../../.."
export BATCH_SIZE="${BATCH_SIZE:-32}"
LOG=experiments/qwen38_27b/logs

VARIANTS="val_base val_lora test_base test_lora val_base_fewshot val_lora_simpleprompt val_lora_compress"
echo "[$(date +%F' '%H:%M:%S)] ==== 27B 全量评测开始（BATCH_SIZE=$BATCH_SIZE） ====" | tee -a "$LOG/eval_27b.log"
for v in $VARIANTS; do
    bash experiments/qwen38_27b/main/scripts/run_eval_27b.sh "$v"
done
echo "[$(date +%F' '%H:%M:%S)] ==== 27B 全量评测结束 ====" | tee -a "$LOG/eval_27b.log"
echo "ALL_EVAL_VARIANTS_DONE" | tee -a "$LOG/eval_27b.log"
