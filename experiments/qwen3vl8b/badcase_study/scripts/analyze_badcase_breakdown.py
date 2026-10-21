#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""badcase 深度分析（临时脚本，仅供诊断，不入主报告链路）。

基于 val 集 LoRA 预测 preds_val_lora.jsonl 分析真实错误结构，回答：
1) 全 JSON 不匹配样本中，每样本错几个属性 / 漏几个键 / 错几个值？
2) 若只错 n 个属性，把它全改对可提升多少全匹配率（潜力上限）？
3) top 混淆对（真值→预测）与各属性漏键频次。
"""
import json, sys
from collections import Counter, defaultdict

def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "experiments/qwen3vl8b/main/reports/preds_val_lora.jsonl"
    preds = load(path)
    n = len(preds)
    parse_ok = [p for p in preds if p["parse_ok"]]
    print(f"样本总数={n}  解析成功={len(parse_ok)}")

    # 全匹配失败的样本
    mism = [p for p in parse_ok if p["parsed"] != p["true"]]
    fm = n - len(mism)
    print(f"全匹配={fm} ({fm/n:.1%})  全匹配失败={len(mism)} ({len(mism)/n:.1%})")

    # 每失败样本：错值数 / 漏键数 / 多键数
    nerr, nmiss, nextra, errdist = [], [], [], Counter()
    perattr_miss = Counter(); conf = Counter()
    for p in mism:
        parsed, true = p["parsed"], p["true"]
        keys = set(true) | set(parsed)
        miss = sum(1 for k in true if k not in parsed)          # 真值有、预测漏
        extra = sum(1 for k in parsed if k not in true)          # 预测多出、真值没有
        wv = sum(1 for k in keys
                 if k in true and k in parsed and parsed[k] != true[k])  # 同键错值
        nmiss.append(miss); nextra.append(extra); nerr.append(miss + extra + wv)
        errdist[miss + extra + wv] += 1
        for k in true:
            if k not in parsed:
                perattr_miss[k] += 1
        # 混淆对：同键且都出现但值不同
        for k in keys:
            if k in true and k in parsed and isinstance(parsed[k], str) \
                    and parsed[k] != true[k]:
                conf[(k, true[k], parsed[k])] += 1

    print("\n[错误规模分布] 每个失败样本的总错误属性数(漏键+多键+错值):")
    for c in sorted(errdist):
        print(f"  错{c}个: {errdist[c]}  累积可改善的全匹配={sum(errdist[cc] for cc in errdist if cc<=c)}")
    # 只错1/2个属性的样本，若被修好，全匹配率能到多少
    fix1 = errdist.get(1, 0) + fm
    fix2 = errdist.get(1, 0) + errdist.get(2, 0) + fm
    print(f"\n若修好全部只错1个属性的样本 -> 全匹配率≈{fix1/n:.1%}")
    print(f"若修好只错1~2个的样本         -> 全匹配率≈{fix2/n:.1%}")

    print("\n[漏键最多的属性] (真值有、预测漏):")
    for k, c in perattr_miss.most_common(12):
        print(f"  {k}: {c}")

    print("\n[Top 混淆对] 属性 | 真值→预测 | 次数:")
    for (k, tv, pv), c in conf.most_common(20):
        print(f"  {k}: {tv}→{pv}  x{c}")

    # 平均每失败样本漏键/错值
    if mism:
        print(f"\n失败样本平均漏键={sum(nmiss)/len(mism):.2f} "
              f"平均多键={sum(nextra)/len(mism):.2f} "
              f"平均总错={sum(nerr)/len(mism):.2f}")

    # 输出一批典型"只错1个属性"样本(便于人工看图归因) 到文件
    oneerr = [p for p in mism
              if sum(1 for k in set(p['true'])|set(p['parsed'])
                     if k in p['true'] and k in p['parsed'] and p['parsed'][k]!=p['true'][k]
                     or (k in p['true'] and k not in p['parsed'])
                     or (k in p['parsed'] and k not in p['true'])) == 1]
    with open("/tmp/badcase_oneerr.jsonl", "w", encoding="utf-8") as f:
        for p in oneerr:
            f.write(json.dumps({"image": p["image"], "true": p["true"],
                                "parsed": p["parsed"]}, ensure_ascii=False) + "\n")
    print(f"\n只错1个属性的样本数={len(oneerr)}  -> 已存 /tmp/badcase_oneerr.jsonl")

if __name__ == "__main__":
    main()
