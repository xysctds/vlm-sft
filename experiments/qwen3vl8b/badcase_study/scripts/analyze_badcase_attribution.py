#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""badcase 聚焦：单错样本构成 + 多键来源 + 属性级还可提升的分布。"""
import json, sys
from collections import Counter, defaultdict

def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "experiments/qwen3vl8b/main/reports/preds_val_lora.jsonl"
    preds = [p for p in load(path) if p["parse_ok"]]
    mism = [p for p in preds if p["parsed"] != p["true"]]

    # 单错样本：只有1处不一致。分三种: 漏1键 / 多1键 / 同键错1值
    kinds = Counter(); multi_extra = Counter(); single_extra_attr = Counter()
    single_attr_err = Counter()
    for p in mism:
        parsed, true = p["parsed"], p["true"]
        keys = set(true) | set(parsed)
        miss = [k for k in true if k not in parsed]
        extra = [k for k in parsed if k not in true]
        wrng = [k for k in (set(true) & set(parsed))
                if isinstance(parsed[k], str) and parsed[k] != true[k]]
        tot = len(miss) + len(extra) + len(wrng)
        if tot == 1:
            if miss: kinds["漏1键"] += 1; single_attr_err[miss[0]] += 1
            elif extra: kinds["多1键"] += 1; single_extra_attr[extra[0]] += 1
            elif wrng: kinds["同键错1值"] += 1; single_attr_err[wrng[0]] += 1
        for k in extra:
            multi_extra[k] += 1

    print("单错样本类型构成:", dict(kinds))
    print("\n[单错样本中出错属性分布(漏键+错值)]:")
    for k, c in single_attr_err.most_common(15):
        print(f"  {k}: {c}")
    print("\n[单错样本中'多1键'多出的键分布] (模型多输出的键):")
    for k, c in single_extra_attr.most_common(15):
        print(f"  {k}: {c}")
    print("\n[全部失败样本中模型多输出的键 top] (多键来源):")
    for k, c in multi_extra.most_common(15):
        print(f"  {k}: {c}")

    # 单错"同键错1值"的那些键值对
    wrng_cnt = Counter()
    for p in mism:
        parsed, true = p["parsed"], p["true"]
        for k in set(true) & set(parsed):
            if isinstance(parsed[k], str) and parsed[k] != true[k]:
                wrng_cnt[(k, true[k], parsed[k])] += 1
    print("\n[仅看总错误次数 单属性层面 top 混淆] (含漏键=→(missing)):")
    for (k, tv, pv), c in wrng_cnt.most_common(20):
        print(f"  {k}: {tv}→{pv} x{c}")

if __name__ == "__main__":
    main()
