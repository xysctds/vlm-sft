# 实验索引

实验按 `experiments/<基座>/<实验域>/` 两级组织，每个域自成一个目录
（`config/` `scripts/` `weights/` `predictions/` `reports/`）。

- 每个域有一个 `README.md` 作为入口（人写）：做什么、目录、怎么跑、指标数值以哪份报告为准
- 域内 `reports/` 下的 md 是脚本生成的评测报告，不要手改
- 运行一律在仓库根执行，脚本按「相对自身位置回推仓库根」定位路径，移动目录无需改脚本主体
- `weights/`、`predictions/` 与预测 / 日志类产物 git 忽略；配置与报告 md 入库

## Qwen3-VL-8B-Instruct（第一轮主实验）

| 实验域 | 内容 |
|---|---|
| [main](qwen3vl8b/main/README.md) | 主实验：训练 / 导出 / 全部评测变体（val、test、few-shot、simple-prompt、压缩分辨率） |
| [ablation](qwen3vl8b/ablation/README.md) | 指令范式消融：schema / simple × 5k / 20k / 35k |
| [ckpt_screen](qwen3vl8b/ckpt_screen/README.md) | checkpoint 快筛与全量确认（`ckpt_*.yaml` 由脚本自动生成） |
| [badcase_study](qwen3vl8b/badcase_study/README.md) | 坏例诊断与干预实证：样本级分析 → 难例续训（负结果）→ 约束解码可救性 → 强化 prompt 重训（未跑完） |
| [smoke](qwen3vl8b/smoke/README.md) | 训练可行性验证与 token / 显存基准：确认能不能跑、显存够不够、样本有多长 |

## Qwen3.8-27B（换基座复跑）

| 实验域 | 内容 |
|---|---|
| [main](qwen38_27b/main/README.md) | 主实验：训练 / 导出 / 8 个评测变体 / 5 份报告 |
| [ablation](qwen38_27b/ablation/README.md) | 指令范式 × 数据量消融（5k 两格） |
| [ckpt_screen](qwen38_27b/ckpt_screen/README.md) | checkpoint 快筛（15 点 × 500 条 val） |

换基座的决策、四处必要差异与结果对比见[换基座实验](../docs/换基座实验.md)。

- 跨域工具在 `qwen38_27b/scripts/`：`check_paths.py`（仓库结构自检）、`smoke_qwen38_load.py`（加载冒烟）
- 日志在 `qwen38_27b/logs/`（git 忽略）
- 27B 的评测不走 `llamafactory-cli train`（`predict_with_generate` 与 ZeRO-3 互斥），统一走 `qwen38_27b/main/scripts/infer_27b.py`，输出与 8B 同格式、下游零改动

`llamaboard_cache/` 是 WebUI 缓存，git 忽略。

## 指标口径只保留一份权威版本

| 内容 | 唯一权威位置 |
|---|---|
| 指标定义（全匹配 / 属性 acc / macro-F1 / McNemar） | [技术报告](../docs/技术报告.md) §5.1；实现见 `scripts/eval_attr.py` |
| 指标数值 | `experiments/**/reports/*.md`（脚本从 `preds_*.jsonl` 算出，不手工填写） |
| 各实验域做什么、怎么跑 | 各域 `README.md`（只做导航，不复述数值） |
| 叙述性结论与论证链 | `docs/` 下的主文档与专题报告 |
