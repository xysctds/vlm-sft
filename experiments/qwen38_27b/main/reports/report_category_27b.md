# 品类分层分析（test 集，LoRA 微调后模型）

- generated: 2026-10-21 13:55:03
- 预测文件: `/disk16T2/scratch/maotai/repos/vlm/experiments/qwen38_27b/main/reports/preds_test_lora_27b.jsonl`
- 维度说明：按**图片品类**（图名前缀）分层，与 eval_attr.py stats 的**属性分层**（shape/texture、二分类/多分类）维度不同。

## 1. 分品类属性准确率

| 品类 | 样本数 | 属性判定数 | 属性准确率 |
|---|---|---|---|
| WOMEN-Tees_Tanks | 1169 | 14748 | 96.4% |
| WOMEN-Blouses_Shirts | 805 | 9988 | 93.4% |
| WOMEN-Dresses | 698 | 9611 | 94.4% |
| WOMEN-Sweaters | 273 | 3558 | 91.0% |
| MEN-Tees_Tanks | 232 | 2630 | 97.1% |
| WOMEN-Jackets_Coats | 178 | 2371 | 90.8% |
| WOMEN-Rompers_Jumpsuits | 173 | 2370 | 96.1% |
| WOMEN-Cardigans | 135 | 1875 | 91.4% |
| WOMEN-Shorts | 134 | 2012 | 96.7% |
| WOMEN-Graphic_Tees | 116 | 1448 | 95.7% |
| MEN-Sweatshirts_Hoodies | 77 | 843 | 96.6% |
| WOMEN-Sweatshirts_Hoodies | 67 | 820 | 96.1% |
| MEN-Shirts_Polos | 67 | 688 | 98.4% |
| WOMEN-Skirts | 65 | 976 | 95.5% |
| MEN-Sweaters | 57 | 603 | 98.2% |
| MEN-Jackets_Vests | 48 | 622 | 92.9% |
| WOMEN-Pants | 47 | 736 | 97.1% |
| MEN-Pants | 22 | 321 | 98.8% |
| MEN-Shorts | 17 | 250 | 96.0% |
| WOMEN-Denim | 11 | 175 | 96.6% |
| WOMEN-Leggings | 10 | 160 | 95.0% |
| MEN-Denim | 8 | 121 | 98.3% |
| MEN-Suiting | 1 | 17 | 100.0% |

## 2. 性别维度汇总

| 分组 | 样本数 | 属性判定数 | 属性准确率 |
|---|---|---|---|
| MEN | 529 | 6095 | 96.9% |
| WOMEN | 3881 | 50848 | 94.6% |

## 3. 观察

- 全部 23 个品类准确率区间：90.8% ~ 100.0%，表现均衡，无灾难性品类；
- 样本量 top3 大品类：WOMEN-Tees_Tanks（1169 图）、WOMEN-Blouses_Shirts（805 图）、WOMEN-Dresses（698 图），准确率分别为 96.4% / 93.4% / 94.4%；
- 小样本品类波动大（如单样本品类），不具统计意义；
- 按品类表加权（micro-average over 已解析样本）属性准确率 94.8%，品类分层与其一致，说明模型泛化不依赖特定品类；注意该口径与 `report_test_27b.md` 的逐样本口径略有差异（后者把缺失键也计入错误）。
