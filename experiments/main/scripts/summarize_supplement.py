#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""补充实验汇总：指令鲁棒性（LoRA × simple prompt）+ few-shot 对照（基座 2-shot），
生成 report_supplement.md。

四组对比（val 全量 4410 条）：
  - 基座 0-shot（schema）            preds_val_base.jsonl
  - LoRA 主模型（schema）            preds_val_lora.jsonl
  - LoRA 主模型 + simple prompt      preds_val_lora_simpleprompt.jsonl
  - 基座 2-shot（schema+示例）       preds_val_base_fewshot.jsonl
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "scripts"))
from eval_attr import summarize

BASE = os.environ.get("VLM_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def load_preds(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


models = {
    "基座 0-shot（schema）": f"{BASE}/experiments/main/reports/preds_val_base.jsonl",
    "基座 2-shot（schema+示例）": f"{BASE}/experiments/main/reports/preds_val_base_fewshot.jsonl",
    "LoRA 主模型（schema）": f"{BASE}/experiments/main/reports/preds_val_lora.jsonl",
    "LoRA 主模型 + simple prompt": f"{BASE}/experiments/main/reports/preds_val_lora_simpleprompt.jsonl",
}

lines = ["# 补充实验：指令鲁棒性 + few-shot 对照（val 全量 4410 条）\n"]
lines.append(f"- generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
lines.append("- 指令鲁棒性：同一 LoRA 主模型，仅替换评测指令（schema vs simple）\n")
lines.append("| 模型×指令 | attr acc | macro-F1 | parse | full-match |")
lines.append("|---|---|---|---|---|")
sums = {}
for name, path in models.items():
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
             "（键名如 `top.type`/`clothing[].color`，不再是 18 维属性键名），按任务 schema 对齐后属性级准确率仅 "
             f"{s_sp['attr_acc']:.1%}（parse 率 {s_sp['parse_rate']:.1%} 说明是合法 JSON 但结构不对齐）——"
             "**schema 指令是运行期必需**：业务部署必须携带 schema prompt，这也反向验证了训练期 schema 指令的价值；")
lines.append(f"- **few-shot vs 微调**：基座 2-shot（{s_fs['attr_acc']:.1%}）较 0-shot（{s_b0['attr_acc']:.1%}）提升 "
             f"{s_fs['attr_acc'] - s_b0['attr_acc']:+.1%}pp，说明 in-context 示例有帮助，但**仍比微调（{s_main['attr_acc']:.1%}）低 "
             f"{s_main['attr_acc'] - s_fs['attr_acc']:.1%}pp**——用数据回答\"为什么微调而不 few-shot\"："
             "few-shot 无法达到结构化抽取的业务可用水平，且每次推理多 2 个示例的输入开销。")

out = f"{BASE}/experiments/main/reports/report_supplement.md"
with open(out, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"\n[ok] -> {out}")
