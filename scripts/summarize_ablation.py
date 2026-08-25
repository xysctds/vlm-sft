#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""指令范式消融汇总：schema-5k vs simple-5k vs 主模型（schema-35k，val 全量）。"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_attr import summarize

BASE = os.environ.get("VLM_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = f"{BASE}/data/data_convert/deepfashion_multi/deepfashion_attr_json_val.jsonl"

with open(DATA, encoding="utf-8") as fh:
    data_rows = [json.loads(l) for l in fh if l.strip()]

# 主模型（schema-35k 全量）已有结果：eval_attr.py 格式的 preds_val_lora.jsonl
with open(f"{BASE}/llamafactory/results/preds_val_lora.jsonl", encoding="utf-8") as fh:
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
    "主模型 schema-35k（全量）": main_preds,
    "schema-5k": load_lf_preds(f"{BASE}/llamafactory/eval_outputs/ablate_schema5k/generated_predictions.jsonl"),
    "simple-5k": load_lf_preds(f"{BASE}/llamafactory/eval_outputs/ablate_simple5k/generated_predictions.jsonl"),
}

lines = ["# 指令范式消融（val 全量 4410 条）\n"]
lines.append("| 模型 | n | attr acc | macro-F1 | parse | full-match |")
lines.append("|---|---|---|---|---|---|")
main_s = summarize(main_preds)
for name, preds in models.items():
    s = summarize(preds)
    lines.append(f"| {name} | {s['n']} | {s['attr_acc']:.1%} | {s['macro_f1']:.3f} | {s['parse_rate']:.1%} | {s['full_match_rate']:.1%} |")
lines.append(f"\n- schema-5k vs simple-5k（同数据量，唯一变量=指令范式）：Δ attr acc = "
             f"{summarize(models['schema-5k'])['attr_acc'] - summarize(models['simple-5k'])['attr_acc']:+.1%}")
lines.append(f"- schema-5k vs 主模型 schema-35k（同指令，唯一变量=数据量）：Δ attr acc = "
             f"{summarize(models['schema-5k'])['attr_acc'] - main_s['attr_acc']:+.1%}")

out = f"{BASE}/llamafactory/results/report_ablation.md"
with open(out, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"\n[ok] -> {out}")
