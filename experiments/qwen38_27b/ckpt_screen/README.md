# ckpt_screen：checkpoint 快筛（27B 版）

27B 训练产出 28 个 checkpoint（`save_steps=100`），本轮取 15 个点做 500 条 val 快筛，回答
「什么时候收敛、早停能省多少」。

## 做什么

训练耗时约 49 小时，若能证明后 2/3 是冗余的，后续同类实验可以直接早停。

## 目录

| 子目录 | 内容 |
|---|---|
| `config/` | 17 个 `ckpt27b_<step>.yaml`（15 个快筛 + 2 个全量确认），由 `generate_ckpt_yamls_27b.py` 自动生成，**不入库** |
| `scripts/` | `generate_ckpt_yamls_27b.py`、`run_screen_27b.sh`（3 卡轮转）、`summarize_ckpts_27b.py` |
| `predictions/` | 各 checkpoint 的原始预测（15 个目录） |
| `reports/` | `report_ckpt_screen_27b.md` |

## 怎么跑

```bash
python experiments/qwen38_27b/ckpt_screen/scripts/generate_ckpt_yamls_27b.py
bash   experiments/qwen38_27b/ckpt_screen/scripts/run_screen_27b.sh
python experiments/qwen38_27b/ckpt_screen/scripts/summarize_ckpts_27b.py
```

## 指标数值以哪份为准

`reports/report_ckpt_screen_27b.md`（每点 500 条 val，纯推理，不训练）。
