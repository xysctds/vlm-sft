# main：主实验（Qwen3-VL-8B-Instruct）

DeepFashion 18 维属性 JSON 抽取的第一轮主实验：LoRA 微调 → 合并导出 → 7 个评测变体。

## 做什么

- base 组（val/test）：确认微调相对基座是否有效
- lora 组（val/test）：微调后的主结果
- 三个边界对照：few-shot（基座 2-shot）、simple-prompt（换朴素指令）、压缩分辨率（768×768）

## 目录

| 子目录 | 内容 |
|---|---|
| `config/` | `train_cli`（3 卡 DDP，等效 batch 4×3×3=36，3 epoch，lr 1e-4，LoRA r16）、`train_webui` + `default_webui`（LlamaBoard 分层格式，等参，供 WebUI 导入）、`export_cli`（合并导出）、7 个评测配置 |
| `scripts/` | `data_convert_fewshot_val.py`（生成 2-shot val 数据）、`summarize_{category,resolution,supplement}.py`（生成三份衍生报告） |
| `weights/lora/` | 28 个 checkpoint（step 100~2793，`save_steps=100`），实体，git 忽略 |
| `weights/merged/` | 合并后的完整模型，软链到 `/disk16T1/scratch/maotai/models/merged` |
| `predictions/` | 7 个变体的 `generated_predictions.jsonl` + `logs/` |
| `reports/` | `preds_*.jsonl`（转换后）+ 5 份报告 + `data_stats.json` |

## 怎么跑

```bash
# 训练（多卡自动走 torchrun；勿设 CUDA_VISIBLE_DEVICES）
llamafactory-cli train experiments/qwen3vl8b/main/config/qwen3vl8b_dfattr_train_cli.yaml

# 合并导出（CPU）
llamafactory-cli export experiments/qwen3vl8b/main/config/qwen3vl8b_dfattr_export_cli.yaml

# 评测（每个变体一条，输出到 predictions/<变体>/）
llamafactory-cli train experiments/qwen3vl8b/main/config/qwen3vl8b_dfattr_eval_val_lora.yaml

# 转换预测 + 打分
python scripts/convert_preds.py \
    --preds experiments/qwen3vl8b/main/predictions/val_lora/generated_predictions.jsonl \
    --data  data/data_convert/deepfashion_multi/deepfashion_attr_json_val.jsonl \
    --out   experiments/qwen3vl8b/main/reports/preds_val_lora.jsonl
python scripts/eval_attr.py stats \
    --base experiments/qwen3vl8b/main/reports/preds_val_base.jsonl \
    --lora experiments/qwen3vl8b/main/reports/preds_val_lora.jsonl \
    --out  experiments/qwen3vl8b/main/reports/report_val.md

# 三份衍生报告
python experiments/qwen3vl8b/main/scripts/summarize_category.py
python experiments/qwen3vl8b/main/scripts/summarize_resolution.py
python experiments/qwen3vl8b/main/scripts/summarize_supplement.py
```

## 指标数值以哪份为准

`reports/` 下的 5 份 md 由脚本从 `preds_*.jsonl` 直接算出，是**指标数值的唯一来源**，不要手改、也不要在别处复述：

| 报告 | 内容 |
|---|---|
| `report_val.md` / `report_test.md` | 总览（解析率 / 全匹配 / 属性 acc / macro-F1）+ 逐属性 + McNemar |
| `report_category.md` | 按图片品类分层 |
| `report_resolution.md` | 压缩分辨率对照 |
| `report_supplement.md` | 指令鲁棒性 + few-shot 对照 |

指标定义见 `docs/技术报告.md` §6.1，实现见 `scripts/eval_attr.py`。
