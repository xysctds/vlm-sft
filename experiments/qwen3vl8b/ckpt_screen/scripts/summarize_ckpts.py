#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""汇总 checkpoint 快筛结果：llamafactory predict 输出 -> 属性准确率曲线表。"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))), "scripts"))
from eval_attr import parse_json_output, summarize

CKPTS = [100, 300, 500, 700, 900, 1100, 1300, 1500, 1700, 1900, 2100, 2300, 2500, 2700, 2793]
BASE = os.environ.get("VLM_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
DATA = f"{BASE}/data/data_convert/deepfashion_multi/deepfashion_attr_json_val.jsonl"

rows = []
with open(DATA, encoding="utf-8") as fh:
    data_rows = [json.loads(l) for l in fh if l.strip()]

for c in CKPTS:
    pred_path = f"{BASE}/experiments/qwen3vl8b/ckpt_screen/predictions/ckpt_{c}/generated_predictions.jsonl"
    if not os.path.exists(pred_path):
        print(f"[skip] checkpoint-{c}: 无预测文件")
        continue
    lf_preds = [json.loads(l) for l in open(pred_path) if l.strip()]
    n = min(len(lf_preds), len(data_rows))
    preds = []
    for i in range(n):
        raw = lf_preds[i].get("predict", "").strip()
        parsed = parse_json_output(raw)
        preds.append({
            "image": data_rows[i]["images"][0],
            "true": json.loads(data_rows[i]["messages"][1]["content"]),
            "raw": raw,
            "parsed": parsed,
            "parse_ok": parsed is not None,
        })
    s = summarize(preds)
    rows.append({"ckpt": c, "n": s["n"], "attr_acc": s["attr_acc"], "macro_f1": s["macro_f1"],
                 "parse_rate": s["parse_rate"], "full_match": s["full_match_rate"]})
    print(f"  checkpoint-{c}: n={s['n']} attr_acc={s['attr_acc']:.1%} macro_f1={s['macro_f1']:.3f} parse={s['parse_rate']:.1%}")

out_path = f"{BASE}/experiments/qwen3vl8b/ckpt_screen/reports/report_ckpt_screen.md"
with open(out_path, "w", encoding="utf-8") as fh:
    fh.write("# Checkpoint 快筛（500 条 val）\n\n")
    fh.write("| 训练步数 | n | 属性准确率 | macro-F1 | 可解析率 | 全 JSON 匹配 |\n")
    fh.write("|---|---|---|---|---|---|\n")
    for r in rows:
        fh.write(f"| {r['ckpt']} | {r['n']} | {r['attr_acc']:.1%} | {r['macro_f1']:.3f} | {r['parse_rate']:.1%} | {r['full_match']:.1%} |\n")
print(f"\n[ok] -> {out_path}")
