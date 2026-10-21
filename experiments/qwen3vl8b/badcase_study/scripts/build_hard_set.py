#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""构造难例增强训练集（难例二次微调用）。

思路（与 REPORT 结论一致）：
  - 主体"同键错值"（texture 视觉本质难例）不在本次低垂果实范围；
  - 聚焦可修的"小物件存在性 + 多/漏键 + 长尾稀有类"。
策略：从全量 train 挑"难例信号"样本（含长尾稀有类值 / 小物件存在 / 关键部位属性），
加上随机普通样本凑足 N 条，作为续训子集。保持与原 train 完全相同的 sharegpt 结构
（messages/columns），只是"数据构成"变了 —— 不改 prompt、不改评测口径。

用法:
  python experiments/qwen3vl8b/badcase_study/scripts/build_hard_set.py \
      --train data/data_convert/deepfashion_multi/deepfashion_attr_json_train.jsonl \
      --out experiments/qwen3vl8b/badcase_study/hardcase_data/deepfashion_attr_json_hard_train.jsonl \
      --n-target 8000

产出 jsonl 结构 = 原 train jsonl（messages + images，images 相对 images/ 目录）。
"""
import argparse
import json
import random
from collections import Counter

# 真正的长尾稀有类取值（train 里样本很少、模型易学不足的）—— 只算这些，别把
# 超大池（denim/cotton/ring=yes 等海量样本）当成难例，否则难例集≈全量失去区分度。
RARE_FABRIC = {"furry", "leather"}          # denim/cotton/chiffon 池过大，不算稀有
RARE_PATTERN = {"lattice", "color block"}    # graphic/pure color 超大，不算
RARE_NECKLINE = {"square", "standing"}       # lapel/suspenders 池偏大
# 结构性"存在性"属性（多/漏键与可见性判断重灾区；这些在 train 里存在但模型易判错，
# 且它们在"真值含/不含"两态上都需要学会省略，故算难例信号）
STRUCTURAL = {"outer_clothing_is_cardigan", "upper_clothing_covering_navel",
              "waist_accessories", "hat", "socks", "glasses"}


def attrs_of(example):
    """取 assistant 真值 dict（解析失败返回 {}）。"""
    try:
        return json.loads(example["messages"][1]["content"])
    except Exception:
        return {}


def is_hard(attrs):
    """判断是否含难例信号：真正长尾稀有值 或 结构性存在性属性出现。"""
    if not attrs:
        return False
    for k, v in attrs.items():
        if k.endswith("_fabric") and v in RARE_FABRIC:
            return True
        if k.endswith("_pattern") and v in RARE_PATTERN:
            return True
        if k == "neckline" and v in RARE_NECKLINE:
            return True
        if k in STRUCTURAL and k in attrs:
            return True  # 该键在真值里出现（含 yes/no 任一态）即命中结构性存在性
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-target", type=int, default=8000)
    args = ap.parse_args()

    rows = []
    with open(args.train, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    hard, normal = [], []
    for ex in rows:
        (hard if is_hard(attrs_of(ex)) else normal).append(ex)
    random.seed(42)

    # 目标：hard 全留（若过多则抽样到 80% 目标），不足用 normal 补
    n_hard = min(len(hard), int(args.n_target * 0.8))
    sel_hard = random.sample(hard, n_hard)
    n_normal = max(0, args.n_target - n_hard)
    n_normal = min(n_normal, len(normal))
    sel_normal = random.sample(normal, n_normal)
    sel = sel_hard + sel_normal
    random.shuffle(sel)

    # 校验信号覆盖率
    hard_sig = Counter()
    for ex in sel:
        a = attrs_of(ex)
        for k, v in a.items():
            if k.endswith("_fabric") and v in RARE_FABRIC:
                hard_sig[f"{k}={v}"] += 1
            if k == "neckline" and v in RARE_NECKLINE:
                hard_sig[f"neckline={v}"] += 1
            if k in STRUCTURAL:
                hard_sig[f"{k} present"] += 1

    with open(args.out, "w", encoding="utf-8") as fh:
        for ex in sel:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"[ok] hard={len(hard)} normal={len(normal)} -> 选 hard {n_hard} + normal {n_normal} = {len(sel)}")
    print("难例信号 top:", dict(hard_sig.most_common(15)))


if __name__ == "__main__":
    main()
