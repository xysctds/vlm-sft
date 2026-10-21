# badcase_study：坏例诊断与干预实证

对主模型做样本级坏例诊断，并实测三条低成本干预路线。完整叙述（诊断过程、三条路线结果、后续选项）
见 [`docs/错误分析.md`](../../../docs/错误分析.md)，本文件只说明目录与复跑。

## 做什么

- 诊断：按样本维度看错误结构（每个失败样本错几个属性、错在漏键 / 多键 / 错值）
- 干预实证：难例续训（`hardcase_retrain`）、约束解码 / 后处理可救性、强化 prompt 重训（`prompt_retrain`）

## 目录

| 子目录 | 内容 | 入库 |
|---|---|---|
| `config/` | `hardcase_retrain` / `prompt_retrain` 的训练与评测配置（5 个） | 是 |
| `scripts/` | 坏例分析、难例集构造、prompt_retrain 数据构造（5 个） | 是 |
| `reports/` | `report_hardcase_retrain_vs_main_val.md`（逐属性评测，脚本生成） | 是 |
| `hardcase_data/` | `dataset_info.json` + 难例集 `*.jsonl`（14M，派生） | 注册是 / 数据否 |
| `prompt_retrain_data/` | `dataset_info.json` + 强化 prompt 数据（108M，派生） | 注册是 / 数据否 |
| `weights/` | `hardcase_retrain_lora/`（1.7G） | 否 |
| `predictions/` | `hardcase_retrain_val/`（53M） | 否 |

数据集不入库的理由：它们由 `scripts/` 从主数据确定性生成，与仓库根 `data/`（软链、同样不入库）策略一致。

## 怎么跑

```bash
# 派生数据集（训练前先跑；产物不入库）
python experiments/qwen3vl8b/badcase_study/scripts/build_hard_set.py            # -> hardcase_data/deepfashion_attr_json_hard_train.jsonl
python experiments/qwen3vl8b/badcase_study/scripts/build_prompt_retrain_data.py  # -> prompt_retrain_data/*.jsonl

# 坏例样本级分析（结果打到 stdout）
python experiments/qwen3vl8b/badcase_study/scripts/analyze_badcase_breakdown.py   experiments/qwen3vl8b/main/reports/preds_val_lora.jsonl
python experiments/qwen3vl8b/badcase_study/scripts/analyze_badcase_attribution.py experiments/qwen3vl8b/main/reports/preds_val_lora.jsonl

# 约束解码可救性分析
python experiments/qwen3vl8b/badcase_study/scripts/constraint_analysis.py experiments/qwen3vl8b/main/reports/preds_val_lora.jsonl

# 难例续训与评测（配置可直接交给 llamafactory-cli）
llamafactory-cli train experiments/qwen3vl8b/badcase_study/config/qwen3vl8b_dfattr_hardcase_retrain.yaml
llamafactory-cli train experiments/qwen3vl8b/badcase_study/config/qwen3vl8b_dfattr_hardcase_retrain_eval_val.yaml
```

## 指标数值以哪份为准

| 数值 | 权威来源 |
|---|---|
| 难例续训后的逐属性评测 | `reports/report_hardcase_retrain_vs_main_val.md`（`eval_attr.py stats` 生成） |
| 样本级诊断计数（错 1 处 / 多键 / 漏键等） | `scripts/analyze_badcase_*.py` 的 stdout —— 脚本未落盘，结果记录在 `docs/错误分析.md` §2 |
| 三条路线的结论与后续选项 | `docs/错误分析.md` |

主模型自身的指标（对比基线）以 `../main/reports/` 为准，本目录不复述。
