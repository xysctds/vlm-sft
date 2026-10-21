#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""约束解码/后处理"能救多少"的定量分析。

回答：对 val_lora 的失败样本，有多少错误是**后处理/约束解码在原理上就能修正**的？
分三类：
  1) 可救(格式层): 输出含非法值 或 键漂移(不在18维合法键) -> 约束解码/后处理可拦
  2) 可部分(可见性): 真值有该键但模型漏(漏键) / 模型输出真值没有的键(多键) ->
     需要"可见性判断"，后处理无从知道该省略/补哪个键，基本救不了
  3) 不可救(视觉): 同键错值且都在合法集内(如 chiffon->cotton) -> 纯视觉，后处理无效
用法: python experiments/qwen3vl8b/badcase_study/scripts/constraint_analysis.py
"""
import json
import sys
import os
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "scripts"))
from data_convert_deepfashion_multi import SHAPE_ATTRS, FABRIC_VALS, PATTERN_VALS, NA_VALUES

# 合法键与合法值
SHAPE_KEYS = [k for k, _ in SHAPE_ATTRS]
TEXTURE_KEYS = [f"{p}_{t}" for p in ("upper", "lower", "outer") for t in ("fabric", "pattern")]
KNOWN_KEYS = set(SHAPE_KEYS) | set(TEXTURE_KEYS)
VALID_VALS = {v for _, m in SHAPE_ATTRS for v in m.values() if v not in NA_VALUES}
VALID_VALS |= {v for v in FABRIC_VALS.values() if v not in NA_VALUES}
VALID_VALS |= {v for v in PATTERN_VALS.values() if v not in NA_VALUES}


def classify_errors(preds):
    """对失败样本统计错误类型与"可救"上限。"""
    stat = Counter()
    fixable_full = 0   # 若把格式层错误修好，有多少全匹配失败样本能被救回
    fixable_vis = 0    # 仅靠后处理能把"该样本所有键集合/值都对齐"的（几乎不可能）
    for p in preds:
        if not p["parse_ok"] or p["parsed"] == p["true"]:
            continue
        parsed, true = p["parsed"], p["true"]
        keys = set(true) | set(parsed)
        # 该样本错误类型
        has_illegal = False   # 非法值
        has_drift = False     # 漂移键
        has_wrong_val = False # 同键错值(合法)
        has_miss = False      # 漏键
        has_extra = False     # 多键
        for k in keys:
            if k in parsed and k not in true:
                has_extra = True
            if k in true and k not in parsed:
                has_miss = True
            if k in true and k in parsed:
                tv, pv = true[k], parsed[k]
                if not isinstance(pv, str) or pv not in VALID_VALS:
                    has_illegal = True
                elif pv != tv:
                    has_wrong_val = True
            if k in parsed and k not in KNOWN_KEYS:
                has_drift = True
        tag = []
        if has_illegal or has_drift: tag.append("格式层(可救)")
        if has_extra: tag.append("多键")
        if has_miss: tag.append("漏键")
        if has_wrong_val: tag.append("同键错值(视觉)")
        if not tag: tag.append("仅键集合错")
        stat["样本含类型:" + "+".join(tag)] += 1
        # 是否"格式层错误是唯一/可修正到全匹配"
        if has_illegal or has_drift:
            if not (has_wrong_val or has_miss or has_extra):
                # 仅非法值/漂移，修好即全匹配
                stat["仅格式错→可救成全匹配"] += 1
    return stat


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "experiments/qwen3vl8b/main/reports/preds_val_lora.jsonl"
    preds = [json.loads(l) for l in open(path) if l.strip()]
    n = len(preds)
    fail = [p for p in preds if p["parse_ok"] and p["parsed"] != p["true"]]
    print(f"样本 {n}, 全匹配失败 {len(fail)} ({len(fail)/n:.1%})")
    st = classify_errors(preds)
    print("\n=== 失败样本错误类型构成 ===")
    for k, v in st.most_common():
        print(f"  {k}: {v}")
    print(f"\n结论: 仅'格式层错误'且修好即全匹配的样本 = {st.get('仅格式错→可救成全匹配',0)} "
          f"({st.get('仅格式错→可救成全匹配',0)/n:.1%} 全样本)")
    print("  → 约束解码/后处理理论上最多只能救回这一小部分；其余(同键错值/多漏键/视觉)救不了。")


if __name__ == "__main__":
    main()
