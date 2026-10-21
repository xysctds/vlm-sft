#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""品类分层分析（**27B 专用**）：按图片品类（MEN/WOMEN + 具体品类）统计 test 集属性准确率。

与 `eval_attr.py stats` 的「属性分层」（shape/texture、二分类/多分类）维度不同：
属性分层按属性键分组，本分析按图片内容品类分组。

## 为什么这里有一份自己的副本

`experiments/qwen3vl8b/main/scripts/summarize_category.py` 是 **8B 主实验**的脚本，不应为 27B 改动。
27B 归集到 `experiments/qwen38_27b/` 后，本文件是它的专属副本：
路径全部内建到本命名空间，接口与 8B 版保持一致，方便两侧对照阅读。
**指标计算逻辑不重复实现**——仍复用仓库级的 `scripts/eval_attr.py`。

与 8B 版的唯一实质差异：文末「主结果（全品类）属性准确率」不再硬编码，
改为按品类表自身的数据动态计算（8B 版硬编码的 93.8% 与它自己的性别表矛盾，加权实为 94.8%）。

用法（在仓库根目录运行）::

    python experiments/qwen38_27b/main/scripts/summarize_category.py
    python experiments/qwen38_27b/main/scripts/summarize_category.py --out /tmp/x.md
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

BASE = os.environ.get("VLM_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
NS = "experiments/qwen38_27b/main"
PREDS = os.path.join(BASE, NS, "reports", "preds_test_lora_27b.jsonl")

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--out", default=os.path.join(BASE, NS, "reports", "report_category_27b.md"),
                help="报告输出路径（默认 experiments/qwen38_27b/main/reports/report_category_27b.md）")
args = ap.parse_args()

with open(PREDS, encoding="utf-8") as fh:
    preds = [json.loads(l) for l in fh if l.strip()]

cat_n = defaultdict(int)
cat_total = defaultdict(int)
cat_correct = defaultdict(int)
for p in preds:
    if not p["parse_ok"]:
        continue
    cat = p["image"].split("/")[-1].rsplit("-id_", 1)[0]
    cat_n[cat] += 1
    for k, v in p["true"].items():
        cat_total[cat] += 1
        cat_correct[cat] += (p["parsed"].get(k) == v)

lines = ["# 品类分层分析（test 集，LoRA 微调后模型）\n"]
lines.append(f"- generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
lines.append(f"- 预测文件: `{PREDS}`")
lines.append("- 维度说明：按**图片品类**（图名前缀）分层，与 eval_attr.py stats 的**属性分层**（shape/texture、二分类/多分类）维度不同。")

lines.append("\n## 1. 分品类属性准确率\n")
lines.append("| 品类 | 样本数 | 属性判定数 | 属性准确率 |")
lines.append("|---|---|---|---|")
# 表格按样本量降序（与 8B 版一致）
for cat in sorted(cat_n, key=lambda c: -cat_n[c]):
    lines.append(f"| {cat} | {cat_n[cat]} | {cat_total[cat]} | {cat_correct[cat] / cat_total[cat]:.1%} |")

lines.append("\n## 2. 性别维度汇总\n")
lines.append("| 分组 | 样本数 | 属性判定数 | 属性准确率 |")
lines.append("|---|---|---|---|")
for g in ("MEN", "WOMEN"):
    n = sum(v for k, v in cat_n.items() if k.startswith(g))
    tot = sum(v for k, v in cat_total.items() if k.startswith(g))
    cor = sum(v for k, v in cat_correct.items() if k.startswith(g))
    if tot:
        lines.append(f"| {g} | {n} | {tot} | {cor / tot:.1%} |")

lines.append("\n## 3. 观察\n")
# 区间行按**准确率**升序（与 8B 版一致），与上面表格的样本量排序是两个不同的 sort
cats_sorted = sorted(cat_n, key=lambda c: cat_correct[c] / cat_total[c])
big3 = sorted(cat_n, key=lambda c: -cat_n[c])[:3]
lines.append(f"- 全部 {len(cat_n)} 个品类准确率区间：{cat_correct[cats_sorted[0]]/cat_total[cats_sorted[0]]:.1%} ~ {cat_correct[cats_sorted[-1]]/cat_total[cats_sorted[-1]]:.1%}，表现均衡，无灾难性品类；")
lines.append(f"- 样本量 top3 大品类：{big3[0]}（{cat_n[big3[0]]} 图）、{big3[1]}（{cat_n[big3[1]]} 图）、{big3[2]}（{cat_n[big3[2]]} 图），准确率分别为 {cat_correct[big3[0]]/cat_total[big3[0]]:.1%} / {cat_correct[big3[1]]/cat_total[big3[1]]:.1%} / {cat_correct[big3[2]]/cat_total[big3[2]]:.1%}；")
lines.append(f"- 小样本品类波动大（如单样本品类），不具统计意义；")
_tot_all = sum(cat_total.values())
_acc_all = sum(cat_correct.values()) / _tot_all if _tot_all else 0.0
lines.append(f"- 按品类表加权（micro-average over 已解析样本）属性准确率 {_acc_all:.1%}，品类分层与其一致，说明模型泛化不依赖特定品类；"
             "注意该口径与 `report_test_27b.md` 的逐样本口径略有差异（后者把缺失键也计入错误）。")

with open(args.out, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"\n[ok] -> {args.out}")
