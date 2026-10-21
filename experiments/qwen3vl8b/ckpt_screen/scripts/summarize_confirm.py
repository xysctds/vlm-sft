#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Top3 checkpoint 全量确认（4410 条 val）。

前置：快筛选出 top3 后，对选中的 checkpoint 跑全量评测（4410 条），
再运行本脚本出确认报告。ckpt-2793(final) 使用已转换的预测文件。
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))), "scripts"))
from eval_attr import parse_json_output, summarize

BASE = os.environ.get("VLM_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
DATA = f"{BASE}/data/data_convert/deepfashion_multi/deepfashion_attr_json_val.jsonl"

with open(DATA, encoding="utf-8") as fh:
    data_rows = [json.loads(l) for l in fh if l.strip()]


def load_lf_preds(pred_path: str) -> list[dict]:
    lf_preds = [json.loads(l) for l in open(pred_path) if l.strip()]
    n = min(len(lf_preds), len(data_rows))
    preds = []
    for i in range(n):
        raw = lf_preds[i].get("predict", "").strip()
        parsed = parse_json_output(raw)
        preds.append({
            "image": data_rows[i]["images"][0],
            "true": json.loads(data_rows[i]["messages"][1]["content"]),
            "raw": raw, "parsed": parsed, "parse_ok": parsed is not None,
        })
    return preds


ckpt_results = {
    "ckpt-900 (1 epoch)": load_lf_preds(
        f"{BASE}/experiments/qwen3vl8b/ckpt_screen/predictions/ckpt_900_full/generated_predictions.jsonl"),
    "ckpt-2100 (2.3 epoch)": load_lf_preds(
        f"{BASE}/experiments/qwen3vl8b/ckpt_screen/predictions/ckpt_2100_full/generated_predictions.jsonl"),
}
# 2793 最终：直接使用已转换格式的预测文件
with open(f"{BASE}/experiments/qwen3vl8b/main/reports/preds_val_lora.jsonl", encoding="utf-8") as fh:
    ckpt_results["ckpt-2793 (final)"] = [json.loads(l) for l in fh if l.strip()]

lines = ["# Top3 Checkpoint 全量确认（4410 条 val）\n"]
lines.append("| checkpoint | n | attr acc | macro-F1 | parse | full-match |")
lines.append("|---|---|---|---|---|---|")
for name, preds in ckpt_results.items():
    s = summarize(preds)
    lines.append(f"| {name} | {s['n']} | {s['attr_acc']:.1%} | {s['macro_f1']:.3f} "
                 f"| {s['parse_rate']:.1%} | {s['full_match_rate']:.1%} |")
    print(f"  {name}: n={s['n']} attr_acc={s['attr_acc']:.1%} macro_f1={s['macro_f1']:.3f}")

out = f"{BASE}/experiments/qwen3vl8b/ckpt_screen/reports/report_ckpt_confirm.md"
with open(out, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
print(f"\n[ok] -> {out}")
