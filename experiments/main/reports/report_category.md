# 品类分层分析（test 集，LoRA 微调后模型）

- generated: 2026-08-31 23:34:26
- 预测文件: `/disk16T2/scratch/maotai/repos/vlm/experiments/main/reports/preds_test_lora.jsonl`
- 维度说明：按**图片品类**（图名前缀）分层，与 eval_attr.py stats 的**属性分层**（shape/texture、二分类/多分类）维度不同。

## 1. 分品类属性准确率

| 品类 | 样本数 | 属性判定数 | 属性准确率 |
|---|---|---|---|
| WOMEN-Tees_Tanks | 1169 | 14748 | 96.4% |
| WOMEN-Blouses_Shirts | 805 | 9988 | 93.8% |
| WOMEN-Dresses | 698 | 9611 | 94.4% |
| WOMEN-Sweaters | 273 | 3558 | 90.1% |
| MEN-Tees_Tanks | 232 | 2630 | 97.4% |
| WOMEN-Jackets_Coats | 178 | 2371 | 91.4% |
| WOMEN-Rompers_Jumpsuits | 173 | 2370 | 96.4% |
| WOMEN-Cardigans | 135 | 1875 | 92.1% |
| WOMEN-Shorts | 134 | 2012 | 96.2% |
| WOMEN-Graphic_Tees | 116 | 1448 | 95.6% |
| MEN-Sweatshirts_Hoodies | 77 | 843 | 95.0% |
| WOMEN-Sweatshirts_Hoodies | 67 | 820 | 95.2% |
| MEN-Shirts_Polos | 67 | 688 | 97.8% |
| WOMEN-Skirts | 65 | 976 | 95.4% |
| MEN-Sweaters | 57 | 603 | 98.7% |
| MEN-Jackets_Vests | 48 | 622 | 91.5% |
| WOMEN-Pants | 47 | 736 | 97.7% |
| MEN-Pants | 22 | 321 | 98.1% |
| MEN-Shorts | 17 | 250 | 96.4% |
| WOMEN-Denim | 11 | 175 | 90.9% |
| WOMEN-Leggings | 10 | 160 | 90.6% |
| MEN-Denim | 8 | 121 | 97.5% |
| MEN-Suiting | 1 | 17 | 100.0% |

## 2. 性别维度汇总

| 分组 | 样本数 | 属性判定数 | 属性准确率 |
|---|---|---|---|
| MEN | 529 | 6095 | 96.6% |
| WOMEN | 3881 | 50848 | 94.6% |

## 3. 观察

- 全部 23 个品类准确率区间：90.1% ~ 100.0%，表现均衡，无灾难性品类；
- 样本量 top3 大品类：WOMEN-Tees_Tanks（1169 图）、WOMEN-Blouses_Shirts（805 图）、WOMEN-Dresses（698 图），准确率分别为 96.4% / 93.8% / 94.4%；
- 小样本品类波动大（如单样本品类），不具统计意义；
- 主结果（全品类）属性准确率 93.8%，品类分层与其一致，说明模型泛化不依赖特定品类。
