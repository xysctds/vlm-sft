#!/bin/bash
# 消融（27B）：指令范式 × 数据量 —— 5k 两格的「训练 → 评测」驱动
#
# 只做用户选定的两个训练格（schema-5k / simple-5k，各约 7.3 小时），
# 20k / simple-35k 视 5k 结果再定是否补（见 27B进度.md 的消融章节）。
#
# 用法：
#   bash experiments/qwen38_27b/ablation/scripts/run_ablate_27b.sh            # 全跑：schema5k 训练+评测，再 simple5k
#   bash experiments/qwen38_27b/ablation/scripts/run_ablate_27b.sh schema5k   # 只跑 schema5k 的训练+评测
#   bash experiments/qwen38_27b/ablation/scripts/run_ablate_27b.sh simple5k
#   bash experiments/qwen38_27b/ablation/scripts/run_ablate_27b.sh train_schema5k   # 只训练
#   bash experiments/qwen38_27b/ablation/scripts/run_ablate_27b.sh eval_schema5k    # 只评测
#
# 训练：llamafactory-cli 检测到多 GPU 自动 torchrun（勿设 CUDA_VISIBLE_DEVICES）
# 评测：由 experiments/qwen38_27b/main/scripts/infer_27b.py 执行（predict_with_generate + ZeRO-3 被 llamafactory 禁止）
# 断点续跑：训练可重跑（HF 会从 output_dir 的 checkpoint 续）；评测会自动跳过已生成的记录
set -u
cd "$(dirname "$0")/../../../.."
export PATH=/home/maotai/.conda/envs/lf/bin:$PATH
export OMP_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CFG=experiments/qwen38_27b/config
LOG=experiments/qwen38_27b/logs
mkdir -p "$LOG"
BATCH_SIZE="${BATCH_SIZE:-32}"

train_arm() {
    local arm="$1"
    echo "[$(date +%F' '%H:%M:%S)] === ablate $arm train start ===" | tee -a "$LOG/ablate_27b.log"
    llamafactory-cli train "$CFG/qwen38_27b_dfattr_ablate_${arm}_train.yaml" >> "$LOG/train_${arm}_27b.log" 2>&1
    local rc=$?
    echo "[$(date +%F' '%H:%M:%S)] === ablate $arm train exit=$rc ===" | tee -a "$LOG/ablate_27b.log"
    return $rc
}

eval_arm() {
    local arm="$1"
    echo "[$(date +%F' '%H:%M:%S)] === ablate $arm eval start (batch=$BATCH_SIZE) ===" | tee -a "$LOG/ablate_27b.log"
    python experiments/qwen38_27b/main/scripts/infer_27b.py \
        --config "$CFG/qwen38_27b_dfattr_eval_ablate_${arm}.yaml" \
        --batch_size "$BATCH_SIZE" >> "$LOG/eval_${arm}_27b.log" 2>&1
    local rc=$?
    echo "[$(date +%F' '%H:%M:%S)] === ablate $arm eval exit=$rc ===" | tee -a "$LOG/ablate_27b.log"
    return $rc
}

STAGE="${1:-all}"
case "$STAGE" in
    all)          train_arm schema5k; eval_arm schema5k; train_arm simple5k; eval_arm simple5k ;;
    schema5k)     train_arm schema5k; eval_arm schema5k ;;
    simple5k)     train_arm simple5k; eval_arm simple5k ;;
    train_schema5k) train_arm schema5k ;;
    train_simple5k) train_arm simple5k ;;
    eval_schema5k)  eval_arm schema5k ;;
    eval_simple5k)  eval_arm simple5k ;;
    *) echo "未知阶段: $STAGE"; exit 2 ;;
esac
echo "ABLATE_27B_STAGE_DONE ($STAGE)" | tee -a "$LOG/ablate_27b.log"
