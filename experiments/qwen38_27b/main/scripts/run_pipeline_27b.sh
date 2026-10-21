#!/bin/bash
# 主实验 27B：训练结束后自动接续「导出 -> 7 个评测 -> 转换打分 -> 全部报告」
#
# 背景：27B 全流程很长（训练约 47-50 小时 + 评测约 10-20 小时），
# 本脚本让训练一结束就自动进入后续阶段，避免空等。
#
# 用法（放进 tmux 常驻）：
#   tmux new-session -d -s pipe27b "bash experiments/qwen38_27b/main/scripts/run_pipeline_27b.sh"
#
# 注意：本脚本只等待「当前已在跑的」训练，不会重复启动训练。
set -u
cd "$(dirname "$0")/../../../.."
LOG=experiments/qwen38_27b/logs/train_27b.log
PLOG=logs/pipeline_27b.log
mkdir -p logs

echo "[$(date +%F' '%H:%M:%S)] pipeline 等待训练完成 ..." >> $PLOG

while true; do
    if grep -qa "ALL_DONE" "$LOG" 2>/dev/null; then
        echo "[$(date +%F' '%H:%M:%S)] 检测到训练 ALL_DONE" >> $PLOG
        break
    fi
    if grep -qa "FAILED" "$LOG" 2>/dev/null; then
        echo "[$(date +%F' '%H:%M:%S)] 训练 FAILED，pipeline 中止" >> $PLOG
        exit 1
    fi
    sleep 60
done

echo "[$(date +%F' '%H:%M:%S)] === 阶段2: 导出 + 评测 ===" >> $PLOG
bash experiments/qwen38_27b/main/scripts/run_eval_27b.sh >> $PLOG 2>&1
echo "[$(date +%F' '%H:%M:%S)] 评测退出码=$?" >> $PLOG

echo "[$(date +%F' '%H:%M:%S)] === 阶段3: 转换 + 打分 + 报告 ===" >> $PLOG
bash experiments/qwen38_27b/main/scripts/run_summary_27b.sh >> $PLOG 2>&1
echo "[$(date +%F' '%H:%M:%S)] 汇总退出码=$?" >> $PLOG

echo "PIPELINE_DONE" >> $PLOG
