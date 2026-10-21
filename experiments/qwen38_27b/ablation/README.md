# ablation：指令范式 × 数据量消融（27B 版）

与 8B 同构的二维消融，但只补信息量最大的两格（schema-5k / simple-5k），而不是把 4 个缺格全量重跑
（约 125 GPU 小时）。`schema-35k` 就是 main 本身，不重复训练。

## 做什么

验证「schema 指令 vs 朴素指令」在 27B 上是否仍然决定性——8B 上这一项差 18~31pp，是方法论的核心证据。

## 目录

| 子目录 | 内容 |
|---|---|
| `config/` | 5 个：`schema5k_train` / `schema5k_smoke` / `simple5k_train` + 2 个评测配置 |
| `scripts/` | `run_ablate_27b.sh`（单格训练）、`run_all_27b_followup.sh`（本轮跟进调度）、`summarize_ablation_27b.py` |
| `weights/` | `qwen38_27b_schema_5k`、`qwen38_27b_simple_5k` 两个 adapter（git 忽略） |
| `predictions/` | 两格的原始预测 |
| `reports/` | `report_ablation_27b.md` |

## 怎么跑

```bash
bash experiments/qwen38_27b/ablation/scripts/run_all_27b_followup.sh   # 训练两格 + 评测 + 汇总
python experiments/qwen38_27b/ablation/scripts/summarize_ablation_27b.py
```

## 指标数值以哪份为准

`reports/report_ablation_27b.md`（脚本生成，val 全量 4,410 条）。跨基座对比见[换基座实验](../../../docs/换基座实验.md)。
