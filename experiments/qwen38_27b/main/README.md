# main：27B 主实验

在主实验基座换成 Qwen3.8-27B 之后重跑的全套主实验：训练 → 导出 → 8 个评测变体 → 5 份报告。
数值与 8B 主实验同口径（全量 4,410 条 / split），可直接对比。

## 做什么

- base 组（val / test）：确认 27B 基座本身的水平
- lora 组（val / test）：微调后的主结果
- 四个补充对照：基座 few-shot、基座 simple 指令、LoRA simple 指令、压缩分辨率

## 目录

| 子目录 | 内容 |
|---|---|
| `config/` | 12 个：`ds_z3_27b.json`（ZeRO-3 配置）、`train_cli`（训练）、`train_smoke`（冒烟）、`export_cli`（合并导出）、8 个评测配置 |
| `scripts/` | `run_train_27b.sh` / `run_eval_27b.sh` / `run_all_eval_27b.sh` / `run_pipeline_27b.sh` / `run_summary_27b.sh`；`infer_27b.py`（27B 专用推理，见下）；3 份 `summarize_*.py` |
| `weights/lora/` | 28 个 checkpoint 的 LoRA adapter（实体，git 忽略） |
| `weights/merged/` | 合并后的完整模型（软链到 `/disk16T1`） |
| `predictions/` | 8 个变体的原始预测 + 日志 |
| `reports/` | `preds_*_27b.jsonl` + 5 份报告（`report_{val,test,category,resolution,supplement}_27b.md`） |

## 怎么跑

```bash
bash experiments/qwen38_27b/main/scripts/run_train_27b.sh      # 训练（约 49h，tmux 常驻）
bash experiments/qwen38_27b/main/scripts/run_eval_27b.sh export # 只导出合并模型（CPU）
bash experiments/qwen38_27b/main/scripts/run_all_eval_27b.sh    # 8 个变体（可断点续跑，约 10~12h）
bash experiments/qwen38_27b/main/scripts/run_summary_27b.sh     # 转换 + 打分 + 5 份报告
```

## 变体与数据集对应

| 变体 | eval_dataset |
|---|---|
| `val_base` / `val_lora` | `deepfashion_attr_json_val.jsonl` |
| `test_base` / `test_lora` | `deepfashion_attr_json_test.jsonl` |
| `val_base_fewshot` | `deepfashion_attr_json_fewshot_val.jsonl` |
| `val_base_simpleprompt` / `val_lora_simpleprompt` | `deepfashion_attr_json_simple_val.jsonl` |
| `val_lora_compress` | `deepfashion_attr_json_val.jsonl`（`image_max_pixels=589824`） |

## 指标数值以哪份为准

`reports/` 下的 5 份报告由脚本从 `preds_*_27b.jsonl` 算出，是数值的唯一来源；指标定义与 8B 共用同一份
（技术报告 §5.1 + `scripts/eval_attr.py`）。

## 注意

27B 的评测**不能**走 `llamafactory-cli train <eval yaml>`：`predict_with_generate` 与 DeepSpeed ZeRO-3
互斥（框架硬禁止），而 27B 的 bf16 权重 54GB 必须跨卡。所以统一走 `infer_27b.py`——它复用 llamafactory 的
`get_infer_args` / `load_model` / 模板与 mm_plugin，用 `device_map="auto"` 做 3 卡模型并行，
输出与 8B 同格式，下游 `convert_preds.py` / `eval_attr.py` 零改动。
