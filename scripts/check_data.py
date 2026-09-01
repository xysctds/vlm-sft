#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DeepFashion 属性 JSON 数据集的质量关卡。

训练前必须通过的四道检查：
  1. schema 健全 : 每行是合法 sharegpt JSONL，images 字段存在
  2. 答案合法   : attr_json 答案可解析为 JSON，键/值在官方属性表内，无空答案
  3. 防泄漏     : train/val 图片零重叠（按图划分生效）
  4. 分布统计   : 每属性取值计数（类别不均衡概览）

用法::

    python scripts/check_data.py --data data/data_convert/deepfashion_multi [--out experiments/main/reports]

任一关卡失败时退出码非 0。
"""

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_convert_deepfashion_multi import (SHAPE_ATTRS, FABRIC_VALS, PATTERN_VALS,  # noqa: E402
                                            NA_VALUES)

VALID_KEYS = {k for k, _ in SHAPE_ATTRS}
VALID_KEYS |= {f"{part}_{t}" for part in ("upper", "lower", "outer")
               for t in ("fabric", "pattern")}
VALID_VALUES = {v for _, m in SHAPE_ATTRS for v in m.values() if v not in NA_VALUES}
VALID_VALUES |= {v for v in FABRIC_VALS.values() if v not in NA_VALUES}
VALID_VALUES |= {v for v in PATTERN_VALS.values() if v not in NA_VALUES}

DATA_DIR = "data/data_convert/deepfashion_multi"
SPLIT_JSONL = {
    "attr_json_train": "deepfashion_attr_json_train.jsonl",
    "attr_json_val": "deepfashion_attr_json_val.jsonl",
    "attr_json_test": "deepfashion_attr_json_test.jsonl",
    "attr_json_simple_train": "deepfashion_attr_json_simple_train.jsonl",
    "attr_json_simple_val": "deepfashion_attr_json_simple_val.jsonl",
    "attr_json_simple_test": "deepfashion_attr_json_simple_test.jsonl",
    "caption_train": "deepfashion_caption_train.jsonl",
    "caption_val": "deepfashion_caption_val.jsonl",
    "caption_test": "deepfashion_caption_test.jsonl",
}


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DATA_DIR)
    parser.add_argument("--out", default="experiments/main/reports")
    parser.add_argument("--sample-print", type=int, default=6, help="print this many rendered samples")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    problems: list[str] = []

    # ---- 加载全部 split ----------------------------------------------------- #
    data: dict[str, list[dict]] = {}
    for name, fname in SPLIT_JSONL.items():
        path = os.path.join(args.data, fname)
        if not os.path.exists(path):
            problems.append(f"[missing] {path}")
            continue
        data[name] = load_jsonl(path)
        print(f"[load] {name}: {len(data[name])} rows")

    # ---- 关卡 1: sharegpt schema ------------------------------------------- #
    print("\n== checkpoint 1: sharegpt schema ==")
    for name, rows in data.items():
        for i, row in enumerate(rows):
            msgs = row.get("messages")
            imgs = row.get("images")
            if not isinstance(msgs, list) or len(msgs) != 2:
                problems.append(f"{name}:{i} messages malformed")
                continue
            if not isinstance(imgs, list) or len(imgs) != 1:
                problems.append(f"{name}:{i} images field malformed")
                continue
            user_c = msgs[0]["content"]
            if "<image>" not in user_c:
                problems.append(f"{name}:{i} missing <image> placeholder in user content")
            if not msgs[1]["content"] or not str(msgs[1]["content"]).strip():
                problems.append(f"{name}:{i} empty assistant answer")
            if not os.path.exists(os.path.join(args.data, imgs[0])):
                problems.append(f"{name}:{i} image file not found: {imgs[0]}")
    print(f"  checked {sum(len(r) for r in data.values())} rows")

    # ---- 关卡 2: 答案合法性（仅 attr_json 类 split）-------------------------- #
    print("\n== checkpoint 2: attr answer validity ==")
    attr_splits = [n for n in data if n.startswith("attr_json")]
    bad_parse = 0
    bad_keys: Counter = Counter()
    bad_vals: Counter = Counter()
    for name in attr_splits:
        for i, row in enumerate(data[name]):
            ans = row["messages"][1]["content"]
            try:
                obj = json.loads(ans)
            except json.JSONDecodeError:
                bad_parse += 1
                problems.append(f"{name}:{i} assistant answer is not valid JSON")
                continue
            if not isinstance(obj, dict) or not obj:
                bad_parse += 1
                problems.append(f"{name}:{i} assistant answer is not a non-empty dict")
                continue
            for key, val in obj.items():
                if key not in VALID_KEYS:
                    bad_keys[key] += 1
                if val not in VALID_VALUES:
                    bad_vals[val] += 1
    print(f"  checked {len(attr_splits)} attr splits, bad_parse={bad_parse}")
    if bad_keys:
        problems.append(f"unknown keys: {dict(bad_keys)}")
    if bad_vals:
        problems.append(f"unknown values: {dict(bad_vals)}")

    # ---- 关卡 3: 防泄漏（train/val/test 三侧图片两两零重叠）------------------ #
    print("\n== checkpoint 3: train/val/test image overlap ==")
    for task in ("attr_json", "caption"):
        splits = {s: f"{task}_{s}" for s in ("train", "val", "test")}
        if all(s in data for s in splits.values()):
            img_sets = {s: {r["images"][0] for r in data[name]} for s, name in splits.items()}
            for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
                overlap = img_sets[a] & img_sets[b]
                print(f"  {task}: {a}_imgs={len(img_sets[a])} {b}_imgs={len(img_sets[b])} overlap={len(overlap)}")
                if overlap:
                    problems.append(f"{task}: {a}/{b} image overlap: {list(overlap)[:5]}")

    # ---- 关卡 4: 类别分布 ---------------------------------------------------- #
    print("\n== checkpoint 4: per-attribute class distribution (attr_json_val) ==")
    stats: dict[str, dict[str, int]] = {}
    if "attr_json_val" in data:
        for row in data["attr_json_val"]:
            obj = json.loads(row["messages"][1]["content"])
            for key, val in obj.items():
                stats.setdefault(key, Counter())[val] += 1
    stats_path = os.path.join(args.out, "data_stats.json")
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump({k: dict(v) for k, v in sorted(stats.items())}, fh, ensure_ascii=False, indent=2)
    for key, counter in sorted(stats.items()):
        total = sum(counter.values())
        top = counter.most_common(3)
        print(f"  {key}: n={total} classes={len(counter)} top={top}")
    print(f"[ok] stats -> {stats_path}")

    # ---- 样本打印 ------------------------------------------------------------ #
    print(f"\n== rendered samples (first {args.sample_print}) ==")
    shown = 0
    for name in ("attr_json_train", "caption_train"):
        for row in data.get(name, [])[: args.sample_print]:
            print(f"--- {name} | {row['images'][0]}")
            print("  user:      ", row["messages"][0]["content"][:200].replace("\n", " | "))
            print("  assistant: ", row["messages"][1]["content"][:200])
            shown += 1
            if shown >= args.sample_print:
                break
        if shown >= args.sample_print:
            break

    # ---- 结论 ---------------------------------------------------------------- #
    print("\n" + "=" * 60)
    if problems:
        print(f"[FAIL] {len(problems)} problem(s):")
        for p in problems[:50]:
            print("  -", p)
        sys.exit(1)
    print("[PASS] all data quality checkpoints passed")


if __name__ == "__main__":
    main()
