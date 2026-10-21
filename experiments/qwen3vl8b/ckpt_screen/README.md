# ckpt_screen：checkpoint 快筛与全量确认

主实验训练了 28 个 checkpoint（`save_steps=100`）。本域先取 15 个点做 500 条 val 快筛，
再对候选点做 4410 条全量确认。

## 做什么

回答两个问题：最终 checkpoint 是否最优、模型是否早饱和（即能否早停省算力）。

## 目录

| 子目录 | 内容 |
|---|---|
| `config/` | 17 个配置：15 个快筛（`ckpt_<step>.yaml`）+ 2 个全量确认（`ckpt_900_full.yaml`、`ckpt_2100_full.yaml`）。由 `scripts/generate_ckpt_yamls.py` 自动生成，`ckpt_*.yaml` 不入库 |
| `scripts/` | `generate_ckpt_yamls.py`（生成配置）、`run_screen.sh`（3 卡轮转，每卡 5 个串行）、`summarize_ckpts.py`、`summarize_confirm.py` |
| `predictions/` | 各 checkpoint 的原始预测（`ckpt_<step>/`）+ `logs/` |
| `reports/` | `report_ckpt_screen.md`（快筛）、`report_ckpt_confirm.md`（全量确认） |

## 怎么跑

```bash
# 1) 生成 ckpt_*.yaml（改选点就改脚本里的 SCREEN 列表）
python experiments/qwen3vl8b/ckpt_screen/scripts/generate_ckpt_yamls.py

# 2) 快筛：3 张卡各跑 5 个 checkpoint
bash experiments/qwen3vl8b/ckpt_screen/scripts/run_screen.sh

# 3) 汇总快筛
python experiments/qwen3vl8b/ckpt_screen/scripts/summarize_ckpts.py

# 4) 全量确认（两个候选点，各自单跑）
llamafactory-cli train experiments/qwen3vl8b/ckpt_screen/config/ckpt_900_full.yaml
llamafactory-cli train experiments/qwen3vl8b/ckpt_screen/config/ckpt_2100_full.yaml
python experiments/qwen3vl8b/ckpt_screen/scripts/summarize_confirm.py
```

## 指标数值以哪份为准

- 快筛曲线：`reports/report_ckpt_screen.md`（每点 500 条 val）
- 候选点确认：`reports/report_ckpt_confirm.md`（4410 条全量）

两点读表须知：

- 快筛只用 500 条，零点几 pt 的差异在该样本量下属于噪声，所以才有第 4 步的全量确认
- 快筛每个进程只占一张卡（8B 单卡放得下），与主实验的 DDP 训练方式不同
