#!/bin/bash
# 主实验 27B：预测转换 + 打分 + 全部报告（CPU 阶段，不占 GPU）
#
# 与 8B 主实验一一对应，产物带 _27b 后缀，不覆盖 8B 的任何文件：
#   convert_preds : 7 个变体的 llamafactory 原始输出 -> eval_attr 可消费的 preds_*_27b.jsonl
#   eval_attr     : report_val_27b.md / report_test_27b.md（含逐属性 McNemar）
#   summarize_*   : report_category_27b.md / report_resolution_27b.md / report_supplement_27b.md
#                   （三个脚本已做向后兼容参数化：带 --suffix _27b 即可，不带则仍是 8B 原路径）
set -u
cd "$(dirname "$0")/../../../.."
export PATH=/home/maotai/.conda/envs/lf/bin:$PATH
PY=python
DS=data/data_convert/deepfashion_multi
P=experiments/qwen38_27b/main/predictions
R=experiments/qwen38_27b/reports
SFX=_27b

echo "=== 1/3 convert_preds ==="
declare -A MAP=(
  ["val_base"]="deepfashion_attr_json_val.jsonl"
  ["test_base"]="deepfashion_attr_json_test.jsonl"
  ["val_lora"]="deepfashion_attr_json_val.jsonl"
  ["test_lora"]="deepfashion_attr_json_test.jsonl"
  ["val_base_fewshot"]="deepfashion_attr_json_fewshot_val.jsonl"
  ["val_lora_simpleprompt"]="deepfashion_attr_json_simple_val.jsonl"
)
for v in "${!MAP[@]}"; do
  src="$P/${v}${SFX}/generated_predictions.jsonl"
  if [ ! -f "$src" ]; then echo "  [skip] 缺少 $src"; continue; fi
  $PY scripts/convert_preds.py --preds "$src" --data "$DS/${MAP[$v]}" --out "$R/preds_${v}${SFX}.jsonl" \
    && echo "  [ok] preds_${v}${SFX}.jsonl"
done

echo "=== 2/3 eval_attr 主报告 ==="
for split in val test; do
  $PY scripts/eval_attr.py stats \
      --base "$R/preds_${split}_base${SFX}.jsonl" \
      --lora "$R/preds_${split}_lora${SFX}.jsonl" \
      --out  "$R/report_${split}${SFX}.md" \
    && echo "  [ok] report_${split}${SFX}.md"
done

echo "=== 3/3 衍生报告 ==="
# 三个 summarize_* 用的是**本目录自带**的 27B 专用副本（路径内建到本命名空间），
# 不再调用 experiments/qwen3vl8b/main/scripts/ 下的 8B 版本 —— 27B 完全自足，main 不需要为它改动。
$PY experiments/qwen38_27b/main/scripts/summarize_category.py \
    && echo "  [ok] report_category_27b.md"
$PY experiments/qwen38_27b/main/scripts/summarize_resolution.py \
    && echo "  [ok] report_resolution_27b.md"
$PY experiments/qwen38_27b/main/scripts/summarize_supplement.py \
    && echo "  [ok] report_supplement_27b.md"

echo "ALL_SUMMARY_DONE"
