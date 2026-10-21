# ablation：指令范式 × 数据量消融

训练期指令（schema / simple）× 训练数据量（5k / 20k / 35k）的二维消融。

## 做什么

- 唯一变量是训练期指令或数据量；评测期 prompt 固定为 schema 版 val（`deepfashion_attr_json_val`），保证单变量
- `schema-35k` 就是主实验本身，直接复用 `../main/weights/lora/qwen3vl8b_dfattr_lora`，没有单独训练
- 回答两个问题：schema 指令在训练期有多重要、数据量加到多少就饱和

## 目录

| 子目录 | 内容 |
|---|---|
| `config/` | 5 个训练配置（schema/simple × 5k/20k/35k；schema-35k 无训练配置）+ 5 个评测配置 |
| `scripts/` | 训练调度 `run_ablate_3gpu.sh`（3 卡 DDP 串行跑 simple-5k → schema-5k）、`run_ablate_20k.sh`、`run_ablate_simple20k.sh`、`run_ablate_simple35k.sh`；汇总 `summarize_ablation.py` |
| `weights/` | 5 个 adapter（`schema_5k` / `schema_20k` / `simple_5k` / `simple_20k` / `simple_35k`）+ `logs/`，实体，git 忽略 |
| `predictions/` | 各格评测的原始预测 |
| `reports/` | `report_ablation.md` |

## 怎么跑

```bash
# 训练（3 卡 DDP；脚本内部按顺序跑两格）
bash experiments/qwen3vl8b/ablation/scripts/run_ablate_3gpu.sh

# 评测（每格一条，加载对应 adapter，输出到 predictions/）
llamafactory-cli train experiments/qwen3vl8b/ablation/config/qwen3vl8b_dfattr_eval_ablate_schema5k.yaml

# 汇总
python experiments/qwen3vl8b/ablation/scripts/summarize_ablation.py
```

## 指标数值以哪份为准

`reports/report_ablation.md`（由 `summarize_ablation.py` 从各格预测算出，含六格总表与两组 delta）。

两点读表须知：

- 消融的等效 batch 与主实验一致（4×3×3=36），只改 `dataset` 与 `max_samples`，其余超参不动
- `simple-*` 各格的 `full-match` 为 0 是预期结果：训练期没见过 schema 指令，输出键名不对齐
