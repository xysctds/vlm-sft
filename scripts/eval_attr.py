#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DeepFashion 属性 JSON 抽取评测框架（打分/报告部分）。

本脚本只负责**打分**（``stats``）：对一份或两份预测文件（Base vs LoRA）计算
属性级指标并输出对比报告（含 McNemar 显著性）。

预测生成统一走 llamafactory 管线（与训练同口径：qwen3_vl_nothink 模板、
mm_plugin 图像 token 显式展开、同数据格式），再用 ``scripts/convert_preds.py``
把 llamafactory 输出转成本脚本可消费的预测文件::

    llamafactory-cli train llamafactory/config/qwen3vl8b_dfattr_eval_val_base.yaml
    llamafactory-cli train llamafactory/config/qwen3vl8b_dfattr_eval_val_lora.yaml
    python scripts/convert_preds.py --preds <output_dir>/generated_predictions.jsonl \\
        --data data/data_convert/deepfashion_multi/deepfashion_attr_json_val.jsonl \\
        --out llamafactory/results/preds_val_lora.jsonl
    python scripts/eval_attr.py stats \\
        --base llamafactory/results/preds_val_base.jsonl \\
        --lora llamafactory/results/preds_val_lora.jsonl \\
        --out llamafactory/results/report_val.md

（历史说明：旧版含 ``infer`` 推理子命令，因 transformers 原生渲染（1 个
image_pad 占位符）与训练/llamafactory 的显式图像 token 展开（6~7 个 image_pad）
输入不一致、少数样本输出漂移（实测 30 条中 4 条不一致），已移除；
推理一律用 llamafactory 管线。）

指标（所有模型用同一套解析/对比代码——"唯一变量是是否微调"的纪律）：

  - JSON 可解析率        : 输出能被解析为 dict 的比例
  - 全 JSON 匹配率       : 解析结果 == 真值 dict（严格精确）
  - 属性级准确率         : pred[key] == true[key]；缺键算错
  - 属性级 macro-F1      : 多分类 F1，缺键并入 "_missing" 类
  - 分层统计             : 形状 vs 面料图案属性；二分类 vs 多分类属性
  - McNemar 检验         : Base vs LoRA 配对显著性（逐属性 + 总体）
"""

import argparse  # 命令行参数解析（stats 子命令与 --base/--lora/--out 选项）
import json      # 预测文件与输出内容的 JSON 解析/序列化
import os        # 路径处理、输出目录创建
import sys       # 修改模块搜索路径（sys.path）
import time      # 生成报告时间戳
from collections import Counter  # 计数器（保留导入，当前版本未直接使用）

import numpy as np  # 数值计算（各属性指标求平均）

# 把本脚本所在目录加入模块搜索路径，以便导入同目录下的
# data_convert_deepfashion_multi（提供 SHAPE_ATTRS 常量）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_convert_deepfashion_multi import SHAPE_ATTRS  # noqa: E402

# ---- 属性键与分组 ----
# 形状属性键：直接取 SHAPE_ATTRS（(键, 描述) 元组列表）中的键
SHAPE_KEYS = [k for k, _ in SHAPE_ATTRS]
# 面料/图案属性键：upper/lower/outer 三个部位 × fabric/pattern 两种属性
# 组合成 "<部位>_<属性>" 形式的键（如 upper_fabric、lower_pattern）
TEXTURE_KEYS = [f"{part}_{t}" for part in ("upper", "lower", "outer")
                for t in ("fabric", "pattern")]
# 属性 -> 分组映射：shape（形状）或 texture（面料图案），用于报告分层统计
GROUP_OF = {k: "shape" for k in SHAPE_KEYS}
GROUP_OF.update({k: "texture" for k in TEXTURE_KEYS})

MISSING = "_missing"  # 缺键折叠成的类（"预测漏了该键"）


# --------------------------------------------------------------------------- #
# 容错 JSON 解析
# --------------------------------------------------------------------------- #

def parse_json_output(text: str) -> dict | None:
    """三级容错解析：区间提取 -> 截断修复 -> None。

    模型输出往往夹带解释性文字、或 JSON 在生成中途被截断，直接 json.loads
    会失败。这里先提取首尾大括号之间的片段尝试解析；失败再逐步从右截断、
    补括号修复；全部失败返回 None（样本计入"不可解析"）。
    """
    if not text:
        return None  # 空输出直接判为不可解析
    # 第 1 级：提取第一个 '{' 到最后一个 '}' 之间的子串
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None  # 找不到成对的大括号
    candidate = text[start:end + 1]
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None  # 解析出非 dict（如列表）也判失败
    except json.JSONDecodeError:
        pass  # 解析失败，进入第 2 级
    # 第 2 级：从右截断直到前缀可解析（廉价且有界）
    # 逐字节去掉尾部，直到某个前缀是合法 JSON；若括号不平衡，先补上
    # 缺失的右括号再尝试（处理生成被截断的常见情况）。
    for cut in range(len(candidate) - 1, 0, -1):
        frag = candidate[:cut]
        if frag.count("{") > frag.count("}"):
            frag += "}" * (frag.count("{") - frag.count("}"))
        try:
            obj = json.loads(frag)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            continue  # 当前截断点仍不可解析，继续往左截
    return None  # 所有截断点均失败


# --------------------------------------------------------------------------- #
# 打分
# --------------------------------------------------------------------------- #

def load_preds(path: str) -> list[dict]:
    """逐行读取预测文件（JSONL），返回预测记录列表。

    每行是一个独立 JSON 对象（由 scripts/convert_preds.py 生成），
    典型字段：parse_ok（是否解析成功）、parsed（解析出的 dict）、
    true（真值 dict）。
    """
    preds = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:  # 跳过空行
                preds.append(json.loads(line))
    return preds


def per_attr_accuracy(preds: list[dict]) -> dict[str, dict]:
    """逐属性准确率统计。

    返回 {attr: {"n": 样本数, "correct": 正确数, "acc": 准确率}}。
    样本 = 真值 dict 中含该属性的样本；预测缺键、预测值为非字符串
    （如嵌套结构）一律按错误计。
    """
    acc: dict[str, dict] = {}
    for p in preds:
        if not p["parse_ok"]:
            continue  # 解析失败的样本不参与属性级统计
        parsed, true = p["parsed"], p["true"]
        for key, tval in true.items():
            stat = acc.setdefault(key, {"n": 0, "correct": 0})
            stat["n"] += 1
            pval = parsed.get(key)
            if not isinstance(pval, str):  # 非字符串预测值（嵌套结构等）按错误计
                pval = None
            if pval == tval:
                stat["correct"] += 1
    for s in acc.values():
        s["acc"] = s["correct"] / s["n"] if s["n"] else 0.0  # 无样本的属性记 0.0
    return acc


def per_attr_f1(preds: list[dict]) -> dict[str, float]:
    """逐属性 macro-F1；预测缺键并入 MISSING 类。

    对每个属性收集（真值标签列表, 预测标签列表），用 sklearn 的多分类
    macro-F1 计算；预测端缺键时填入 MISSING，作为独立的"漏预测"类
    参与 F1 计算。
    """
    from sklearn.metrics import f1_score
    f1s: dict[str, float] = {}
    per: dict[str, tuple[list, list]] = {}
    for p in preds:
        if not p["parse_ok"]:
            continue
        parsed, true = p["parsed"], p["true"]
        for key, tval in true.items():
            per.setdefault(key, ([], []))
            per[key][0].append(tval)               # 真值标签
            pval = parsed.get(key, MISSING)        # 预测缺键 -> MISSING
            if not isinstance(pval, str):          # 非字符串预测值（嵌套结构等）并入 MISSING 类
                pval = MISSING
            per[key][1].append(pval)               # 预测标签
    for key, (y_true, y_pred) in per.items():
        f1s[key] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return f1s


def binary_vs_multi(preds: list[dict]) -> dict[str, str]:
    """按真值中不同取值个数把属性分为二分类 / 多分类。

    统计每个属性在真值中出现的不同取值集合：个数 <= 2 记为 binary
    （二分类），否则记为 multi（多分类），供报告分层统计使用。
    """
    vals: dict[str, set] = {}
    for p in preds:
        for key, v in p["true"].items():
            vals.setdefault(key, set()).add(v)
    return {k: "binary" if len(v) <= 2 else "multi" for k, v in vals.items()}


def mcnemar(base_preds: list[dict], lora_preds: list[dict]) -> dict:
    """逐属性配对 McNemar 检验（连续校正）。

    McNemar 检验针对配对样本：只看两模型结果**不一致**的样本。
    b = Base 错而 LoRA 对的样本数；c = Base 对而 LoRA 错的样本数。
    零假设 b == c（两模型无差异）；统计量 χ² = (|b-c|-1)²/(b+c)
    （含连续性校正），p 值取自由度为 1 的卡方分布右尾。
    样本 = 两侧都解析成功且真值含该属性的样本。
    返回 {"per_attr": {attr: {...}}, "overall": b/c 累加后的总体检验}。
    """
    from scipy.stats import chi2
    res: dict[str, dict] = {}
    base_acc, lora_acc = per_attr_accuracy(base_preds), per_attr_accuracy(lora_preds)
    for key in sorted(set(base_acc) | set(lora_acc)):  # 两侧出现过的所有属性
        b = c = 0  # b: base 错且 lora 对；c: base 对且 lora 错
        for bp, lp in zip(base_preds, lora_preds):
            if not (bp["parse_ok"] and lp["parse_ok"]):
                continue  # 任一侧解析失败即排除该样本
            if key not in bp["true"]:
                continue  # 真值不含该属性的样本不参与该属性检验
            bhit = bp["parsed"].get(key) == bp["true"][key]
            lhit = lp["parsed"].get(key) == lp["true"][key]
            if (not bhit) and lhit:
                b += 1  # 只有 Base 错（LoRA 的优势样本）
            elif bhit and (not lhit):
                c += 1  # 只有 LoRA 错（Base 的优势样本）
        # 连续校正版 McNemar 统计量；b+c=0（无分歧样本）时 χ² 记 0、p 记 1
        chi2v = ((abs(b - c) - 1) ** 2) / (b + c) if b + c > 0 else 0.0
        res[key] = {
            "b": b, "c": c, "chi2": chi2v,
            "p": chi2.sf(chi2v, 1) if b + c > 0 else 1.0,
            "base_acc": base_acc.get(key, {}).get("acc", 0.0),
            "lora_acc": lora_acc.get(key, {}).get("acc", 0.0),
        }
    # 总体检验：把所有属性的 b、c 分别累加后做一次 McNemar
    b_total = sum(r["b"] for r in res.values())
    c_total = sum(r["c"] for r in res.values())
    chi2_total = ((abs(b_total - c_total) - 1) ** 2) / (b_total + c_total) if b_total + c_total > 0 else 0.0
    return {"per_attr": res, "overall": {"b": b_total, "c": c_total, "chi2": chi2_total,
                                         "p": chi2.sf(chi2_total, 1) if b_total + c_total > 0 else 1.0}}


def summarize(preds: list[dict]) -> dict:
    """汇总一份预测文件的整体指标。

    返回：样本数 n、JSON 可解析率、全 JSON 匹配率、平均属性准确率、
    平均 macro-F1，以及逐属性的准确率/样本数/F1（供报告表格使用）。
    """
    n = len(preds)
    parse_ok = sum(p["parse_ok"] for p in preds)                                    # 可解析样本数
    full_match = sum(1 for p in preds if p["parse_ok"] and p["parsed"] == p["true"])  # 全匹配样本数
    acc = per_attr_accuracy(preds)
    f1s = per_attr_f1(preds)
    attr_acc = float(np.mean([a["acc"] for a in acc.values()])) if acc else 0.0    # 各属性准确率平均
    macro_f1 = float(np.mean(list(f1s.values()))) if f1s else 0.0                  # 各属性 macro-F1 平均
    return {
        "n": n, "parse_rate": parse_ok / n if n else 0.0,
        "full_match_rate": full_match / n if n else 0.0,
        "attr_acc": attr_acc, "macro_f1": macro_f1,
        "per_attr_acc": {k: round(v["acc"], 4) for k, v in acc.items()},
        "per_attr_n": {k: v["n"] for k, v in acc.items()},
        "per_attr_f1": {k: round(v, 4) for k, v in f1s.items()},
    }


def run_stats(args) -> None:
    """stats 子命令主流程：加载预测 -> 计算指标 -> 生成 Markdown 报告。"""
    # 加载 Base 预测；--lora 可选，提供时做 Base vs LoRA 对比分析
    base_preds = load_preds(args.base)
    lora_preds = load_preds(args.lora) if args.lora else None
    # 分别汇总整体指标；有 LoRA 时额外计算 McNemar 显著性
    base_sum = summarize(base_preds)
    lora_sum = summarize(lora_preds) if lora_preds else None
    mn = mcnemar(base_preds, lora_preds) if lora_preds else None
    kinds = binary_vs_multi(base_preds)  # 属性二分类/多分类的分层信息

    lines: list[str] = []
    add = lines.append  # 局部别名，用于逐行拼装报告文本
    add("# DeepFashion 属性 JSON 评测报告")
    add("")
    add(f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    add(f"- Base 预测文件：`{args.base}`")
    if lora_preds:
        add(f"- LoRA 预测文件：`{args.lora}`")
    add("")

    # 第 1 节：总体对比表（样本数 / 解析率 / 全匹配率 / 属性准确率 / macro-F1）
    add("## 1. 总览")
    add("")
    add("| 模型 | 样本数 | 解析率 | 全匹配率 | 属性准确率 | macro-F1 |")
    add("|---|---|---|---|---|---|")
    add(f"| Base | {base_sum['n']} | {base_sum['parse_rate']:.1%} | {base_sum['full_match_rate']:.1%} "
        f"| {base_sum['attr_acc']:.1%} | {base_sum['macro_f1']:.3f} |")
    if lora_sum:
        add(f"| **LoRA** | {lora_sum['n']} | {lora_sum['parse_rate']:.1%} | {lora_sum['full_match_rate']:.1%} "
            f"| {lora_sum['attr_acc']:.1%} | {lora_sum['macro_f1']:.3f} |")
    add("")

    # 第 2 节：逐属性表格；有 LoRA 时附带 McNemar 统计（b/c/χ²/p）
    if lora_sum:
        add("## 2. 逐属性（准确率 / macro-F1 / McNemar）")
        add("")
        add("| 属性 | 分组 | 样本数 | Base 准确率 | LoRA 准确率 | Δ | b（Base✗→LoRA✓） | c（Base✓→LoRA✗） | χ² | p |")
        add("|---|---|---|---|---|---|---|---|---|---|")
    else:
        add("## 2. 逐属性（准确率 / macro-F1）")
        add("")
        add("| 属性 | 分组 | 样本数 | 准确率 | macro-F1 |")
        add("|---|---|---|---|---|")
    keys = sorted(set(base_sum["per_attr_acc"]) | (set(lora_sum["per_attr_acc"]) if lora_sum else set()))
    for key in keys:
        b_acc = base_sum["per_attr_acc"].get(key, 0.0)
        l_acc = lora_sum["per_attr_acc"].get(key, 0.0) if lora_sum else None
        n = base_sum["per_attr_n"].get(key, 0)
        delta = (l_acc - b_acc) if l_acc is not None else 0.0
        if mn:
            m = mn["per_attr"].get(key, {})
            add(f"| {key} | {GROUP_OF.get(key, '?')} | {n} | {b_acc:.1%} | {l_acc:.1%} | {delta:+.1%} "
                f"| {m.get('b', '-')} | {m.get('c', '-')} | {m.get('chi2', 0.0):.2f} | {m.get('p', 1.0):.3f} |")
        else:
            add(f"| {key} | {GROUP_OF.get(key, '?')} | {n} | {b_acc:.1%} | "
                f"{base_sum['per_attr_f1'].get(key, 0.0):.3f} |")
    if mn:
        o = mn["overall"]
        add(f"| **总体** | — | — | — | — | — | {o['b']} | {o['c']} | {o['chi2']:.2f} | {o['p']:.3f} |")
    add("")

    # 第 3 节：分层统计——按分组（shape 形状 / texture 面料图案）与
    # 类别数（binary 二分类 / multi 多分类）分别求平均准确率
    add("## 3. 分层统计（按属性分组 / 类别数）")
    add("")
    for group in ("shape", "texture"):
        ks = [k for k in keys if GROUP_OF.get(k) == group]
        b_mean = float(np.mean([base_sum["per_attr_acc"].get(k, 0.0) for k in ks])) if ks else 0.0
        l_mean = float(np.mean([lora_sum["per_attr_acc"].get(k, 0.0) for k in ks])) if (lora_sum and ks) else 0.0
        if lora_sum:
            add(f"- **{group}**（属性数={len(ks)}）：Base {b_mean:.1%} | LoRA {l_mean:.1%} "
                f"| Δ {l_mean - b_mean:+.1%}")
        else:
            add(f"- **{group}**（属性数={len(ks)}）：Base {b_mean:.1%}")
    for kind in ("binary", "multi"):
        ks = [k for k in keys if kinds.get(k) == kind]
        b_mean = float(np.mean([base_sum["per_attr_acc"].get(k, 0.0) for k in ks])) if ks else 0.0
        l_mean = float(np.mean([lora_sum["per_attr_acc"].get(k, 0.0) for k in ks])) if (lora_sum and ks) else 0.0
        if lora_sum:
            add(f"- **{kind}-class**（属性数={len(ks)}）：Base {b_mean:.1%} | LoRA {l_mean:.1%} "
                f"| Δ {l_mean - b_mean:+.1%}")
        else:
            add(f"- **{kind}-class**（属性数={len(ks)}）：Base {b_mean:.1%}")
    add("")

    # 第 4 节：指标口径说明（解析率/全匹配/属性准确率/macro-F1/McNemar）
    add("## 4. 指标口径说明")
    add("- 解析率：输出能被解析为 dict 的比例（容错解析：区间提取 + 截断修复）")
    add("- 全匹配：解析出的 dict 与真值 dict 完全一致")
    add("- 属性准确率：预测缺键按错误计")
    add("- macro-F1：预测缺键并入 `_missing` 类")
    if mn:
        add(f"- McNemar（连续校正），总体 p = {mn['overall']['p']:.3f} "
            f"（b={mn['overall']['b']}，c={mn['overall']['c']}）")

    # 确保输出目录存在后写入报告文件，并同时在终端打印
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[完成] 报告已写入 -> {args.out}")


# --------------------------------------------------------------------------- #

def main() -> None:
    """命令行入口：仅暴露 stats 子命令（推理已统一走 llamafactory 管线）。"""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    # stats：对 Base/LoRA 预测文件打分并输出 Markdown 对比报告
    p_stats = sub.add_parser("stats")
    p_stats.add_argument("--base", required=True)          # Base 预测文件（JSONL）
    p_stats.add_argument("--lora", default=None)           # LoRA 预测文件（可选）
    p_stats.add_argument("--out", default="llamafactory/results/report.md")  # 报告输出路径
    p_stats.set_defaults(func=run_stats)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
