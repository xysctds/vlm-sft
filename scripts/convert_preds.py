#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 llamafactory predict 输出转换为 scripts/eval_attr.py stats 可消费的预测文件。

背景：正式评测统一走 llamafactory 管线（与训练同口径：qwen3_vl_nothink 模板、
mm_plugin 图像 token 展开、同数据格式），生成 predictions 后由 eval_attr.py stats
负责打分（属性级 acc / macro-F1 / 分层 / McNemar）。

llamafactory predict 输出（output_dir/generated_predictions.jsonl）每行:
    {"prompt": ..., "predict": <模型生成文本>, "label": <真值文本>}

用法::

    python scripts/convert_preds.py \\
        --preds experiments/main/predictions/xxx/generated_predictions.jsonl \\
        --data data/data_convert/deepfashion_multi/deepfashion_attr_json_val.jsonl \\
        --out experiments/main/reports/preds_lora.jsonl

注意：--data 必须与 llamafactory 配置的 eval_dataset/max_samples 完全一致
（逐行顺序对应）。
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_attr import parse_json_output  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preds", required=True, help="llamafactory generated_predictions.jsonl")
    parser.add_argument("--data", required=True, help="原始数据集 jsonl（提供 image 字段与真值，顺序与 --preds 对应）")
    parser.add_argument("--out", required=True, help="输出 preds jsonl（eval_attr.py stats 格式）")
    args = parser.parse_args()

    rows = []
    with open(args.data, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    lf_preds = []
    with open(args.preds, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                lf_preds.append(json.loads(line))

    if len(lf_preds) != len(rows):
        print(f"[warn] 条数不一致: llamafactory={len(lf_preds)} data={len(rows)}，按较短者对齐")
    n = min(len(lf_preds), len(rows))

    out = []
    for i in range(n):
        row, lp = rows[i], lf_preds[i]
        raw = lp.get("predict", "").strip()
        parsed = parse_json_output(raw)
        true = json.loads(row["messages"][1]["content"])
        out.append({
            "image": row["images"][0],
            "true": true,
            "raw": raw,
            "parsed": parsed,
            "parse_ok": parsed is not None,
        })

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for p in out:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    ok = sum(p["parse_ok"] for p in out)
    print(f"[ok] {len(out)} predictions -> {args.out} (parse rate {ok / len(out):.1%})")


if __name__ == "__main__":
    main()
