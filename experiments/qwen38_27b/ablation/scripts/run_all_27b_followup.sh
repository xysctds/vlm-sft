#!/bin/bash
# 27B 消融跟进：一次串起来跑完用户选定的「分级方案」全部内容（约 22 GPU 小时）
#
# 顺序（GPU 只有 3 张，全部串行；每步都天然可断点续跑）：
#   1) B：消融 5k 两格  —— schema5k 训练+评测，再 simple5k 训练+评测（约 17.6h）
#   2) A：基座 + simple 指令 推理（约 1.5h）——补齐「模型 × 指令范式」推理侧矩阵
#   3) C：27B checkpoint 快筛 15 点 × 500 条（约 2.8h，纯推理）
#   4) 汇总：27B 消融报告 + 27B ckpt 报告 + 重新生成补充报告（含新的 base×simple 行）
#
# 用法（放进 tmux 常驻）：
#   tmux new-session -d -s followup27b "bash experiments/qwen38_27b/ablation/scripts/run_all_27b_followup.sh"
set -u
cd "$(dirname "$0")/../../../.."
export PATH=/home/maotai/.conda/envs/lf/bin:$PATH
export OMP_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG=experiments/qwen38_27b/logs/followup_27b.log
mkdir -p experiments/qwen38_27b/logs
PY=python

step() { echo "[$(date +%F' '%H:%M:%S)] ===== $* =====" | tee -a "$LOG"; }

step "1/4 B: 消融 5k 两格（训练+评测）"
bash experiments/qwen38_27b/ablation/scripts/run_ablate_27b.sh all

step "2/4 A: 27B 基座 + simple 指令 推理"
$PY experiments/qwen38_27b/main/scripts/infer_27b.py \
    --config experiments/qwen38_27b/main/config/qwen38_27b_dfattr_eval_val_base_simpleprompt.yaml \
    --batch_size 32 >> experiments/qwen38_27b/logs/val_base_simpleprompt_27b.log 2>&1
echo "[$(date +%F' '%H:%M:%S)] A exit=$?" | tee -a "$LOG"
$PY scripts/convert_preds.py \
    --preds experiments/qwen38_27b/main/predictions/val_base_simpleprompt_27b/generated_predictions.jsonl \
    --data data/data_convert/deepfashion_multi/deepfashion_attr_json_simple_val.jsonl \
    --out experiments/qwen38_27b/main/reports/preds_val_base_simpleprompt_27b.jsonl >> "$LOG" 2>&1
echo "[$(date +%F' '%H:%M:%S)] A convert exit=$?" | tee -a "$LOG"

step "3/4 C: 27B checkpoint 快筛"
bash experiments/qwen38_27b/ckpt_screen/scripts/run_screen_27b.sh

step "4/4 汇总"
$PY experiments/qwen38_27b/ablation/scripts/summarize_ablation_27b.py >> "$LOG" 2>&1
echo "[$(date +%F' '%H:%M:%S)] summarize_ablation exit=$?" | tee -a "$LOG"
$PY experiments/qwen38_27b/ckpt_screen/scripts/summarize_ckpts_27b.py >> "$LOG" 2>&1
echo "[$(date +%F' '%H:%M:%S)] summarize_ckpts exit=$?" | tee -a "$LOG"
$PY experiments/qwen38_27b/main/scripts/summarize_supplement.py >> "$LOG" 2>&1
echo "[$(date +%F' '%H:%M:%S)] summarize_supplement exit=$?" | tee -a "$LOG"

step "全部结束"
echo "FOLLOWUP_27B_DONE" | tee -a "$LOG"
