#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""指令范式（schema/simple）× 数据量（5k/20k/35k）二维消融汇总，val 全量 4410 条。
主模型 = schema-35k（全量）；simple-35k/20k 为 2026-08-26 新补跑的两组。"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "scripts"))
from eval_attr import summarize

BASE = os.environ.get("VLM_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA = f"{BASE}/data/data_convert/deepfashion_multi/deepfashion_attr_json_val.jsonl"

with open(DATA, encoding="utf-8") as fh:
    data_rows = [json.loads(l) for l in fh if l.strip()]

# 主模型（schema-35k 全量）已有结果：eval_attr.py 格式的 preds_val_lora.jsonl
with open(f"{BASE}/experiments/main/reports/preds_val_lora.jsonl", encoding="utf-8") as fh:
    main_preds = [json.loads(l) for l in fh if l.strip()]

def load_lf_preds(pred_path: str) -> list[dict]:
    lf_preds = [json.loads(l) for l in open(pred_path) if l.strip()]
    n = min(len(lf_preds), len(data_rows))
    preds = []
    for i in range(n):
        raw = lf_preds[i].get("predict", "").strip()
        from eval_attr import parse_json_output
        parsed = parse_json_output(raw)
        preds.append({
            "image": data_rows[i]["images"][0],
            "true": json.loads(data_rows[i]["messages"][1]["content"]),
            "raw": raw, "parsed": parsed, "parse_ok": parsed is not None,
        })
    return preds

models = {
    "schema-35k": main_preds,
    "schema-20k": load_lf_preds(f"{BASE}/experiments/ablation/predictions/ablate_schema20k/generated_predictions.jsonl"),
    "schema-5k": load_lf_preds(f"{BASE}/experiments/ablation/predictions/ablate_schema5k/generated_predictions.jsonl"),
    "simple-35k": load_lf_preds(f"{BASE}/experiments/ablation/predictions/ablate_simple35k/generated_predictions.jsonl"),
    "simple-20k": load_lf_preds(f"{BASE}/experiments/ablation/predictions/ablate_simple20k/generated_predictions.jsonl"),
    "simple-5k": load_lf_preds(f"{BASE}/experiments/ablation/predictions/ablate_simple5k/generated_predictions.jsonl"),
}

# 预计算各模型汇总，供表格与对比行复用
sums = {name: summarize(preds) for name, preds in models.items()}
sc35, sc20, sc5 = sums["schema-35k"], sums["schema-20k"], sums["schema-5k"]
sm35, sm20, sm5 = sums["simple-35k"], sums["simple-20k"], sums["simple-5k"]

MAIN = "schema-35k"  # 最终主模型（全量训练），表格中整行加粗

lines = ["# 指令范式 + 数据量消融（val 全量 4410 条）\n"]
lines.append("| 模型 | n | attr acc | macro-F1 | parse | full-match |")
lines.append("|---|---|---|---|---|---|")
for name in models:
    s = sums[name]
    cells = [name, str(s["n"]), f"{s['attr_acc']:.1%}", f"{s['macro_f1']:.3f}",
             f"{s['parse_rate']:.1%}", f"{s['full_match_rate']:.1%}"]
    if name == MAIN:
        cells = [f"**{c}**" for c in cells]
    lines.append("| " + " | ".join(cells) + " |")

lines.append("\n**数据量消融（同指令，唯一变量=数据量）**")
lines.append(f"- schema：5k→20k Δ attr acc = {sc20['attr_acc'] - sc5['attr_acc']:+.1%}，20k→35k Δ = {sc35['attr_acc'] - sc20['attr_acc']:+.1%}")
lines.append(f"- simple：5k→20k Δ attr acc = {sm20['attr_acc'] - sm5['attr_acc']:+.1%}，20k→35k Δ = {sm35['attr_acc'] - sm20['attr_acc']:+.1%}")
lines.append("\n**指令范式消融（同数据量，唯一变量=指令范式）**")
lines.append(f"- 5k：schema vs simple Δ attr acc = {sc5['attr_acc'] - sm5['attr_acc']:+.1%}")
lines.append(f"- 20k：schema vs simple Δ attr acc = {sc20['attr_acc'] - sm20['attr_acc']:+.1%}")
lines.append(f"- 35k：schema vs simple Δ attr acc = {sc35['attr_acc'] - sm35['attr_acc']:+.1%}")

out = f"{BASE}/experiments/ablation/reports/report_ablation.md"
with open(out, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"\n[ok] -> {out}")
