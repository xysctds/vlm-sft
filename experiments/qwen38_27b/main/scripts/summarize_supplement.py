#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""补充实验汇总（**27B 专用**）：指令鲁棒性（LoRA × simple prompt）+ few-shot 对照（基座 2-shot）。

五组对比（val 全量 4410 条）：

  - 基座 0-shot（schema）           preds_val_base_27b.jsonl
  - 基座 0-shot（simple 指令）      preds_val_base_simpleprompt_27b.jsonl  ← 27B 独有（缺失则跳过）
  - 基座 2-shot（schema+示例）      preds_val_base_fewshot_27b.jsonl
  - LoRA 主模型（schema）           preds_val_lora_27b.jsonl
  - LoRA 主模型 + simple prompt     preds_val_lora_simpleprompt_27b.jsonl

## 为什么这里有一份自己的副本

`experiments/qwen3vl8b/main/scripts/summarize_supplement.py` 是 **8B 主实验**的脚本，不应为 27B 改动。
本文件是 27B 命名空间的专属副本，路径全部内建；指标计算仍复用仓库级 `scripts/eval_attr.py`。

与 8B 版的两处差异（都是 27B 实际数据要求的）：
  1. 多一行「基座 0-shot（simple 指令）」—— 8B 没跑这一格，文件缺失时自动跳过；
  2. few-shot 结论按 Δ 正负分支。27B 的基座 2-shot 相对 0-shot 是 **-1.7pt**（示例反而略降），
     8B 的硬编码措辞「说明 in-context 示例有帮助」在 27B 上会自相矛盾。

用法（在仓库根目录运行）::

    python experiments/qwen38_27b/main/scripts/summarize_supplement.py
"""
import argparse
import json
import os
import sys
import time

BASE = os.environ.get("VLM_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, os.path.join(BASE, "scripts"))
from eval_attr import summarize  # noqa: E402

NS = "experiments/qwen38_27b/main"
PD = os.path.join(BASE, NS, "reports")

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--out", default=os.path.join(BASE, NS, "reports", "report_supplement_27b.md"),
                help="报告输出路径（默认 experiments/qwen38_27b/main/reports/report_supplement_27b.md）")
args = ap.parse_args()


def load_preds(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


models = {
    "基座 0-shot（schema）": f"{PD}/preds_val_base_27b.jsonl",
    # 27B 独有的一格（8B 侧没有）；文件缺失时跳过，不报错
    "基座 0-shot（simple 指令）": f"{PD}/preds_val_base_simpleprompt_27b.jsonl",
    "基座 2-shot（schema+示例）": f"{PD}/preds_val_base_fewshot_27b.jsonl",
    "LoRA 主模型（schema）": f"{PD}/preds_val_lora_27b.jsonl",
    "LoRA 主模型 + simple prompt": f"{PD}/preds_val_lora_simpleprompt_27b.jsonl",
}

lines = ["# 补充实验：指令鲁棒性 + few-shot 对照（val 全量 4410 条）\n"]
lines.append(f"- generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
lines.append("- 指令鲁棒性：同一 LoRA 主模型，仅替换评测指令（schema vs simple）\n")
lines.append("| 模型×指令 | attr acc | macro-F1 | parse | full-match |")
lines.append("|---|---|---|---|---|")
sums = {}
for name, path in models.items():
    if not os.path.exists(path):
        print(f"[skip] {name}: 无预测文件 {path}")
        continue
    s = summarize(load_preds(path))
    sums[name] = s
    lines.append(f"| {name} | {s['attr_acc']:.1%} | {s['macro_f1']:.3f} | {s['parse_rate']:.1%} | {s['full_match_rate']:.1%} |")

s_main = sums["LoRA 主模型（schema）"]
s_sp = sums["LoRA 主模型 + simple prompt"]
s_fs = sums["基座 2-shot（schema+示例）"]
s_b0 = sums["基座 0-shot（schema）"]
lines.append(f"\n- 指令鲁棒性：LoRA 模型在 simple 指令下 {s_sp['attr_acc']:.1%} vs schema 指令 {s_main['attr_acc']:.1%}"
             f"（Δ {s_sp['attr_acc'] - s_main['attr_acc']:+.1%}）；")
lines.append(f"- few-shot 价值：基座 2-shot {s_fs['attr_acc']:.1%} vs 基座 0-shot {s_b0['attr_acc']:.1%}"
             f"（Δ {s_fs['attr_acc'] - s_b0['attr_acc']:+.1%}），仍远低于微调 {s_main['attr_acc']:.1%}。")

lines.append("\n## 观察\n")
lines.append("- **指令依赖度**：LoRA 模型在 simple 指令下输出**退化为自由格式 JSON**"
             "（键名如 `top.type`/`clothing[].color`，不再是属性键名），按任务 schema 对齐后属性级准确率仅 "
             f"{s_sp['attr_acc']:.1%}（parse 率 {s_sp['parse_rate']:.1%} 说明是合法 JSON 但结构不对齐）——"
             "**schema 指令是运行期必需**：业务部署必须携带 schema prompt，这也反向验证了训练期 schema 指令的价值；")
if "基座 0-shot（simple 指令）" in sums:
    s_bs = sums["基座 0-shot（simple 指令）"]
    lines.append(f"- **指令依赖是固有的，不是微调造成的**：基座在 simple 指令下也只有 {s_bs['attr_acc']:.1%}"
                 f"（schema 指令下 {s_b0['attr_acc']:.1%}，parse 率 {s_bs['parse_rate']:.1%}）——"
                 "说明模型的结构化抽取能力本就完全依赖 prompt 里的 key/value 白名单，"
                 "「推理期必须带 schema」在微调前后都成立，换基座或换训练指令都规避不了；")

d_fs = s_fs["attr_acc"] - s_b0["attr_acc"]
if d_fs >= 0:
    fs_part = (f"提升 {d_fs:+.1%}pp，说明 in-context 示例有帮助，"
               f"但**仍比微调（{s_main['attr_acc']:.1%}）低 {s_main['attr_acc'] - s_fs['attr_acc']:.1%}pp**")
else:
    fs_part = (f"反而下降 {abs(d_fs):.1%}pp，说明该基座已经足够强、示例提示不再带来收益（甚至略有干扰），"
               f"且两者都不如微调，最好者**仍比微调（{s_main['attr_acc']:.1%}）低 "
               f"{s_main['attr_acc'] - max(s_fs['attr_acc'], s_b0['attr_acc']):.1%}pp**")
lines.append(f"- **few-shot vs 微调**：基座 2-shot（{s_fs['attr_acc']:.1%}）较 0-shot（{s_b0['attr_acc']:.1%}）{fs_part}——"
             "用数据回答\"为什么微调而不 few-shot\"："
             "few-shot 无法达到结构化抽取的业务可用水平，且每次推理多 2 个示例的输入开销。")

with open(args.out, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"\n[ok] -> {args.out}")
