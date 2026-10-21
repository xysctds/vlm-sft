#!/bin/bash
# 主实验 27B：导出合并 + 7 个评测变体生成预测（GPU 阶段）
#
# 与 8B 主实验流程一一对应，只是基座换成 Qwen3.8-27B、产物带 _27b 后缀：
#   1) export       : LoRA adapter 合并回基座（CPU 合并，54GB 单卡放不下）
#   2) 7 个评测变体  : val/test 的 base 与 lora、few-shot、simple-prompt、压缩分辨率
#
# 用法：bash experiments/qwen38_27b/main/scripts/run_eval_27b.sh [起始阶段]
#   不带参数    = 全跑（export + 7 个评测）
#   export      = 只导出
#   <变体名>    = 只跑某个变体，如 val_lora
#
# 说明：27B 的预测不能走 `llamafactory-cli train`（parser.py 硬禁止
#       predict_with_generate + ZeRO-3，而 27B 单卡装不下必须跨卡），
#       改由 experiments/qwen38_27b/main/scripts/infer_27b.py 用 device_map="auto" 做 3 卡模型并行，
#       模板 / mm_plugin / 解码口径与 llamafactory 逐字一致，输出同格式
#       generated_predictions.jsonl，下游 convert_preds.py 无需改动。
#       每个变体脚本内支持断点续跑：重跑同一命令会跳过已生成的记录。
#
# 环境变量：BATCH_SIZE（默认 16）控制生成批大小，显存不够可调小。
set -u
cd "$(dirname "$0")/../../../.."
export PATH=/home/maotai/.conda/envs/lf/bin:$PATH
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CFG=experiments/qwen38_27b/config
LOG=experiments/qwen38_27b/logs
mkdir -p "$LOG"
STAGE="${1:-all}"
BATCH_SIZE="${BATCH_SIZE:-16}"

run_eval() {
    local name="$1"
    echo "[$(date +%F' '%H:%M:%S)] === eval $name start (batch=$BATCH_SIZE) ===" | tee -a "$LOG/eval_27b.log"
    python experiments/qwen38_27b/main/scripts/infer_27b.py \
        --config "$CFG/qwen38_27b_dfattr_eval_${name}.yaml" \
        --batch_size "$BATCH_SIZE" \
        >> "$LOG/${name}_27b.log" 2>&1
    local rc=$?
    echo "[$(date +%F' '%H:%M:%S)] === eval $name exit=$rc ===" | tee -a "$LOG/eval_27b.log"
    return $rc
}

if [ "$STAGE" = "export" ] || [ "$STAGE" = "all" ]; then
    echo "[$(date +%F' '%H:%M:%S)] === export merge start ===" | tee -a "$LOG/eval_27b.log"
    llamafactory-cli export "$CFG/qwen38_27b_dfattr_export_cli.yaml" \
        >> "$LOG/export_27b.log" 2>&1
    echo "[$(date +%F' '%H:%M:%S)] === export exit=$? ===" | tee -a "$LOG/eval_27b.log"
fi

VARIANTS="val_base val_lora test_base test_lora val_base_fewshot val_lora_simpleprompt val_lora_compress"
if [ "$STAGE" = "all" ]; then
    for v in $VARIANTS; do run_eval "$v"; done
elif [ "$STAGE" != "export" ]; then
    run_eval "$STAGE"
fi

echo "ALL_EVAL_DONE" | tee -a "$LOG/eval_27b.log"
