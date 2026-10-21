#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""汇总 27B checkpoint 快筛结果 -> report_ckpt_screen_27b.md。

与 8B 版 `summarize_ckpts.py` 完全同口径（同一批选点、同样 500 条 val 前缀、同样指标），
只是读 27B 的预测目录（qwen38_27b_ckpt_*）并写到 report_ckpt_screen_27b.md，
**不覆盖 8B 的 report_ckpt_screen.md**。
"""
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
    pred_path = f"{BASE}/experiments/qwen38_27b/ckpt_screen/predictions/qwen38_27b_ckpt_{c}/generated_predictions.jsonl"
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

out_path = f"{BASE}/experiments/qwen38_27b/ckpt_screen/reports/report_ckpt_screen_27b.md"
with open(out_path, "w", encoding="utf-8") as fh:
    fh.write("# Checkpoint 快筛（27B，500 条 val）\n\n")
    fh.write("| 训练步数 | n | 属性准确率 | macro-F1 | 可解析率 | 全 JSON 匹配 |\n")
    fh.write("|---|---|---|---|---|---|\n")
    for r in rows:
        fh.write(f"| {r['ckpt']} | {r['n']} | {r['attr_acc']:.1%} | {r['macro_f1']:.3f} | {r['parse_rate']:.1%} | {r['full_match']:.1%} |\n")

    if rows:
        best = max(rows, key=lambda r: r["attr_acc"])
        last = rows[-1]
        fh.write(f"\n- 最优 checkpoint：{best['ckpt']} 步，属性准确率 {best['attr_acc']:.1%}（macro-F1 {best['macro_f1']:.3f}）\n")
        fh.write(f"- 最终 checkpoint：{last['ckpt']} 步，属性准确率 {last['attr_acc']:.1%}"
                 f"（与最优相差 {last['attr_acc'] - best['attr_acc']:+.1%}）\n")
        half = [r for r in rows if r["ckpt"] <= 1400]
        if half:
            fh.write(f"- 前 1400 步内的最优：{max(half, key=lambda r: r['attr_acc'])['ckpt']} 步"
                     f"（{max(r['attr_acc'] for r in half):.1%}）—— 用于判断 27B 是否早饱和\n")
print(f"\n[ok] -> {out_path}")
