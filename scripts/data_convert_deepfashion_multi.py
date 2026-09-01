#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将 DeepFashion-MultiModal（Text2Human, SIGGRAPH 2022）转换为 LLaMA-Factory SFT 数据集。

原始下载位于 ``data/data_origin/DeepFashion-MultiModal/googledrive/``（压缩包 + captions.json）。
本脚本读写的是解压后的工作数据集 ``data/data_convert/deepfashion_multi/``：:

    images/           44,096 张 JPEG（750x1101），命名如 MEN-Denim-id_00000080-01_7_additional.jpg
    captions.json     {img_name: 自然语言描述}  （42,544 张图）
    labels/shape/shape_anno_all.txt      <img> + 12 个形状属性 id   （42,544 行）
    labels/texture/fabric_ann.txt        <img> + 上/下/外装面料 id  （44,096 行）
    labels/texture/pattern_ann.txt       <img> + 上/下/外装图案 id  （44,096 行）
    segm/ densepose/ keypoints/          VLM 微调用不到（解析 / 姿态 / DensePose 任务）

官方属性 id -> 语义映射来自数据集 README
(https://github.com/yumingj/DeepFashion-MultiModal)。

产出 4 个 LLaMA-Factory "sharegpt" 数据集（按图片划分 train/val/test = 80/10/10，seed 42，
三侧图片两两零重叠；test 仅用于最终评测）：

1. ``deepfashion_attr_json_{train,val,test}``    图 -> 可见属性结构化 JSON
                                                 （主线任务：schema 感知指令，列出全部合法
                                                 键与取值，工业界结构化输出标准做法）
2. ``deepfashion_attr_json_simple_{train,val,test}``  图 -> 结构化 JSON（对照：朴素一句话指令）
3. ``deepfashion_caption_{train,val,test}``      图 -> 自然语言描述（captions.json）
4. ``deepfashion_attr_qa_{train,val,test}``      逐属性问答对（18 属性，剔除 NA；已弃用）

输出写在解压数据集旁边，图片路径保持相对（``images/<name>.jpg``）。
注册方式：``dataset_dir: data/data_convert/deepfashion_multi``
（参见 ``experiments/main/config/qwen3vl8b_dfattr_train.yaml``）。

用法::

    python scripts/data_convert_deepfashion_multi.py --data data/data_convert/deepfashion_multi
"""

import argparse
import json
import os
import random
from collections import Counter

# --- 官方属性定义（来自数据集 README）------------------------------------------------

# (属性键, {id: 取值}) — 12 个形状属性
SHAPE_ATTRS = [
    ("sleeve_length", {0: "sleeveless", 1: "short-sleeve", 2: "medium-sleeve", 3: "long-sleeve", 4: "not long-sleeve", 5: "NA"}),
    ("lower_clothing_length", {0: "three-point", 1: "medium short", 2: "three-quarter", 3: "long", 4: "NA"}),
    ("socks", {0: "no", 1: "socks", 2: "leggings", 3: "NA"}),
    ("hat", {0: "no", 1: "yes", 2: "NA"}),
    ("glasses", {0: "no", 1: "eyeglasses", 2: "sunglasses", 3: "glasses in hand or on clothes", 4: "NA"}),
    ("neckwear", {0: "no", 1: "yes", 2: "NA"}),
    ("wrist_wearing", {0: "no", 1: "yes", 2: "NA"}),
    ("ring", {0: "no", 1: "yes", 2: "NA"}),
    ("waist_accessories", {0: "no", 1: "belt", 2: "clothing at waist", 3: "hidden", 4: "NA"}),
    ("neckline", {0: "V-shape", 1: "square", 2: "round", 3: "standing", 4: "lapel", 5: "suspenders", 6: "NA"}),
    ("outer_clothing_is_cardigan", {0: "yes", 1: "no", 2: "NA"}),
    ("upper_clothing_covering_navel", {0: "no", 1: "yes", 2: "NA"}),
]
FABRIC_VALS = {0: "denim", 1: "cotton", 2: "leather", 3: "furry", 4: "knitted", 5: "chiffon", 6: "other", 7: "NA"}
PATTERN_VALS = {0: "floral", 1: "graphic", 2: "striped", 3: "pure color", 4: "lattice", 5: "other", 6: "color block", 7: "NA"}

NA_VALUES = {"NA", "hidden"}  # "hidden" 同样视为无有效答案

SHAPE_QUESTIONS = {
    "sleeve_length": "What is the sleeve length of the clothing?",
    "lower_clothing_length": "What is the length of the lower clothing?",
    "socks": "Is the person wearing socks or leggings?",
    "hat": "Is the person wearing a hat?",
    "glasses": "What glasses is the person wearing?",
    "neckwear": "Is the person wearing neckwear?",
    "wrist_wearing": "Is the person wearing anything on the wrist?",
    "ring": "Is the person wearing a ring?",
    "waist_accessories": "What is worn at the waist?",
    "neckline": "What is the neckline design of the clothing?",
    "outer_clothing_is_cardigan": "Is the outer clothing a cardigan?",
    "upper_clothing_covering_navel": "Does the upper clothing cover the navel?",
}
PART_QUESTIONS = {
    "upper_fabric": "What is the fabric of the upper clothing?",
    "lower_fabric": "What is the fabric of the lower clothing?",
    "outer_fabric": "What is the fabric of the outer clothing?",
    "upper_pattern": "What pattern does the upper clothing have?",
    "lower_pattern": "What pattern does the lower clothing have?",
    "outer_pattern": "What pattern does the outer clothing have?",
}

CAPTION_PROMPT = "Describe the clothing and the person in this image."

# 朴素一句话指令（"simple" 对照版本，快速上手 recipe 风格）。
SIMPLE_JSON_PROMPT = "Extract the clothing attributes visible in this image and output them as JSON."

# Schema 感知指令（主线任务）：枚举全部合法键与取值，让模型"填表"而不是自由发挥。
# 这是工业界结构化输出数据构造的标准做法（消融时与 SIMPLE_JSON_PROMPT 对比）。
SCHEMA_RULES = (
    "Rules:\n"
    "- Include only keys whose attribute is clearly visible in the image.\n"
    "- Use only the allowed values listed above.\n"
    "- Output the JSON object only, without any extra text or explanation."
)


def build_schema_prompt() -> str:
    """从官方属性表自动生成 schema 指令。"""
    lines = ["Extract the clothing attributes visible in this image and output a JSON object.",
             "", "Allowed keys and values:"]
    for key, mapping in SHAPE_ATTRS:
        vals = [v for v in mapping.values() if v not in NA_VALUES]
        lines.append(f"- {key}: {', '.join(vals)}")
    for part in ("upper", "lower", "outer"):
        lines.append(f"- {part}_fabric: {', '.join(v for v in FABRIC_VALS.values() if v not in NA_VALUES)}")
        lines.append(f"- {part}_pattern: {', '.join(v for v in PATTERN_VALS.values() if v not in NA_VALUES)}")
    lines.append("")
    lines.append(SCHEMA_RULES)
    return "\n".join(lines)

TASKS = ["caption", "attr_qa", "attr_json"]


def read_txt_ann(path: str) -> dict[str, list[int]]:
    """读取 '<img> <v1> <v2> ...' 行 -> {img: [v1, v2, ...]}。"""
    ann: dict[str, list[int]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 2:
                continue
            ann[parts[0]] = [int(v) for v in parts[1:]]
    return ann


def build_attrs(shape: dict, fabric: dict, pattern: dict) -> dict[str, dict[str, str]]:
    """每张图的可见属性字典: {img: {attr_key: value}}。"""
    attrs: dict[str, dict[str, str]] = {}
    for img, vals in shape.items():
        if len(vals) < len(SHAPE_ATTRS):
            continue
        attrs.setdefault(img, {})
        for (key, mapping), v in zip(SHAPE_ATTRS, vals):
            if v in mapping and mapping[v] not in NA_VALUES:
                attrs[img][key] = mapping[v]
    for img, vals in fabric.items():
        if len(vals) < 3:
            continue
        attrs.setdefault(img, {})
        for part, v in zip(["upper", "lower", "outer"], vals):
            if v in FABRIC_VALS and FABRIC_VALS[v] not in NA_VALUES:
                attrs[img][f"{part}_fabric"] = FABRIC_VALS[v]
    for img, vals in pattern.items():
        if len(vals) < 3:
            continue
        attrs.setdefault(img, {})
        for part, v in zip(["upper", "lower", "outer"], vals):
            if v in PATTERN_VALS and PATTERN_VALS[v] not in NA_VALUES:
                attrs[img][f"{part}_pattern"] = PATTERN_VALS[v]
    return attrs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/data_convert/deepfashion_multi", help="extracted DeepFashion-MultiModal dir")
    parser.add_argument("--tasks", nargs="+", default=TASKS, choices=TASKS, help="which SFT datasets to build")
    parser.add_argument("--val-ratio", type=float, default=0.10, help="fraction of images held out for validation")
    parser.add_argument("--test-ratio", type=float, default=0.10, help="fraction of images held out for the final test set (touched only in final eval)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # --- 加载标注 ------------------------------------------------------------
    captions = json.load(open(os.path.join(args.data, "captions.json"), encoding="utf-8"))
    shape = read_txt_ann(os.path.join(args.data, "labels", "shape", "shape_anno_all.txt"))
    fabric = read_txt_ann(os.path.join(args.data, "labels", "texture", "fabric_ann.txt"))
    pattern = read_txt_ann(os.path.join(args.data, "labels", "texture", "pattern_ann.txt"))
    attrs = build_attrs(shape, fabric, pattern)
    print(f"[load] captions={len(captions)} shape={len(shape)} fabric={len(fabric)} pattern={len(pattern)} "
          f"images_with_attrs={len(attrs)}")

    # --- 按图片划分 train/val/test（三侧图片两两零重叠）-----------------------
    image_pool = sorted(set(captions) | set(attrs))
    rng = random.Random(args.seed)
    rng.shuffle(image_pool)
    n_val = max(1, round(len(image_pool) * args.val_ratio))
    n_test = max(1, round(len(image_pool) * args.test_ratio))
    val_imgs = set(image_pool[:n_val])
    test_imgs = set(image_pool[n_val:n_val + n_test])
    train_imgs = set(image_pool[n_val + n_test:])
    print(f"[split] train={len(train_imgs)} val={len(val_imgs)} test={len(test_imgs)} "
          f"(seed={args.seed}, val={args.val_ratio:.0%} test={args.test_ratio:.0%}, by image)")

    records: dict[str, dict[str, list[dict]]] = {t: {"train": [], "val": [], "test": []} for t in args.tasks}
    if "attr_json" in args.tasks:
        records["attr_json_simple"] = {"train": [], "val": [], "test": []}
    schema_prompt = build_schema_prompt()

    for img in image_pool:
        if img in val_imgs:
            split = "val"
        elif img in test_imgs:
            split = "test"
        else:
            split = "train"
        img_rel = os.path.join("images", img)
        if "caption" in args.tasks and img in captions and str(captions[img]).strip():
            # 跳过空描述（原始 captions.json 中有 7 条）
            records["caption"][split].append(
                {"messages": [{"role": "user", "content": f"<image>{CAPTION_PROMPT}"},
                              {"role": "assistant", "content": captions[img]}],
                 "images": [img_rel]}
            )
        img_attrs = attrs.get(img, {})
        if "attr_json" in args.tasks and img_attrs:
            answer = json.dumps(img_attrs, ensure_ascii=False)
            records["attr_json"][split].append(
                {"messages": [{"role": "user", "content": f"<image>{schema_prompt}"},
                              {"role": "assistant", "content": answer}],
                 "images": [img_rel]}
            )
            records["attr_json_simple"][split].append(
                {"messages": [{"role": "user", "content": f"<image>{SIMPLE_JSON_PROMPT}"},
                              {"role": "assistant", "content": answer}],
                 "images": [img_rel]}
            )
        if "attr_qa" in args.tasks:
            for key, value in img_attrs.items():
                question = SHAPE_QUESTIONS.get(key) or PART_QUESTIONS.get(key)
                if question is None:
                    continue
                records["attr_qa"][split].append(
                    {"messages": [{"role": "user", "content": f"<image>{question}"},
                                  {"role": "assistant", "content": value}],
                     "images": [img_rel]}
                )

    # --- 写出 jsonl + dataset_info.json ---------------------------------------
    dataset_info: dict[str, dict] = {}
    counts: Counter = Counter()
    for task in records:  # 包含派生的 "attr_json_simple" 变体
        for split in ["train", "val", "test"]:
            name = f"deepfashion_{task}_{split}"
            out_path = os.path.join(args.data, f"{name}.jsonl")
            with open(out_path, "w", encoding="utf-8") as fh:
                for rec in records[task][split]:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            dataset_info[name] = {
                "file_name": f"{name}.jsonl",
                "formatting": "sharegpt",
                "columns": {"messages": "messages", "images": "images"},
                "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"},
            }
            counts[name] = len(records[task][split])
            print(f"[ok] {name}: {counts[name]} rows -> {out_path}")

    with open(os.path.join(args.data, "dataset_info.json"), "w", encoding="utf-8") as fh:
        json.dump(dataset_info, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(f"[ok] registry -> {os.path.join(args.data, 'dataset_info.json')}")
    print("summary:", dict(counts))


if __name__ == "__main__":
    main()
