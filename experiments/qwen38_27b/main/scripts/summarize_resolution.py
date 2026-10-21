#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""图像分辨率对比（**27B 专用**）：不压缩（825,750）vs 压缩（589,824），LoRA 最终模型全量 val。

前置：压缩分辨率评测已完成（`predictions/main/val_lora_compress_27b/`）；
不压缩侧使用已转换好的 `reports/preds_val_lora_27b.jsonl`。

## 为什么这里有一份自己的副本

`experiments/qwen3vl8b/main/scripts/summarize_resolution.py` 是 **8B 主实验**的脚本，不应为 27B 改动。
本文件是 27B 命名空间的专属副本，路径全部内建；指标计算仍复用仓库级 `scripts/eval_attr.py`。

用法（在仓库根目录运行）::

    python experiments/qwen38_27b/main/scripts/summarize_resolution.py
"""
import argparse
import json
import os
import sys

BASE = os.environ.get("VLM_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, os.path.join(BASE, "scripts"))
from eval_attr import parse_json_output, summarize  # noqa: E402

NS = "experiments/qwen38_27b/main"
DATA = os.path.join(BASE, "data/data_convert/deepfashion_multi/deepfashion_attr_json_val.jsonl")
PREDS_825 = os.path.join(BASE, NS, "reports", "preds_val_lora_27b.jsonl")
PRED_DIR_589 = os.path.join(BASE, NS, "predictions", "val_lora_compress_27b")

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--out", default=os.path.join(BASE, NS, "reports", "report_resolution_27b.md"),
                help="报告输出路径（默认 experiments/qwen38_27b/main/reports/report_resolution_27b.md）")
args = ap.parse_args()

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


# 不压缩（825,750）：已有 eval_attr.py 格式的 preds_val_lora_27b.jsonl
with open(PREDS_825, encoding="utf-8") as fh:
    res_825 = [json.loads(l) for l in fh if l.strip()]

# 压缩（589,824）：llamafactory 原始输出
compress_path = os.path.join(PRED_DIR_589, "generated_predictions.jsonl")
if not os.path.exists(compress_path):
    sys.exit(f"[err] 无压缩评测输出: {compress_path}")
res_589 = load_lf_preds(compress_path)

s825, s589 = summarize(res_825), summarize(res_589)
lines = ["# 图像分辨率对比（推理端，LoRA 最终模型全量 val）\n"]
lines.append("| image_max_pixels | n | attr acc | macro-F1 | parse | full-match |")
lines.append("|---|---|---|---|---|---|")
lines.append(f"| 825,750（不压缩） | {s825['n']} | {s825['attr_acc']:.1%} | {s825['macro_f1']:.3f} "
             f"| {s825['parse_rate']:.1%} | {s825['full_match_rate']:.1%} |")
lines.append(f"| 589,824（压缩） | {s589['n']} | {s589['attr_acc']:.1%} | {s589['macro_f1']:.3f} "
             f"| {s589['parse_rate']:.1%} | {s589['full_match_rate']:.1%} |")
lines.append(f"\nΔ attr acc: {s589['attr_acc'] - s825['attr_acc']:+.1%}")

with open(args.out, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"\n[ok] -> {args.out}")
