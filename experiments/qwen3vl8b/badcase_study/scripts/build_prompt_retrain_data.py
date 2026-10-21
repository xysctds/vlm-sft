#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""prompt_retrain：构造"强化可见性 prompt"版 attr_json 数据集（训练+评测同步改口径）。

背景：主文档 §7.3.1 实证——同一模型换评测 prompt 会崩（输出键名漂移），因此改 prompt
必须"训练数据 + 评测数据"用同一份新 prompt，作为**独立新口径**从头训一个新模型，
不与旧口径混比。本脚本生成该新口径数据。

改动（针对 badcase 主体缺陷，保持 18 维键/值枚举**不变**，只强化 Rules 的可见性纪律）：
  1. 小物件/配饰（ring/neckwear/wrist/waist/hat/glasses/socks）——放大观察、
     明确可见才判 yes，不可见应判 no（多键幻觉=把真值没有的键硬输出，故强调"仅在
     明确可见时输出该键的非 no 判断；无法确认就是 no，不要杜撰新键"）。
  2. 多/漏键——重申"只输出真值确实可见的属性键，不要臆测"，并提示对每件可穿戴物
     逐项做 yes/no 核对，避免漏判。

用法（不占 GPU；train/val/test 三个分片各生成一份）:
  python experiments/qwen3vl8b/badcase_study/scripts/build_prompt_retrain_data.py
产出写到 experiments/qwen3vl8b/badcase_study/prompt_retrain_data/（含 dataset_info.json + images 软链由外部保证）
"""
import json
import os
import sys

SRC_DIR = "data/data_convert/deepfashion_multi"
OUT_DIR = "experiments/qwen3vl8b/badcase_study/prompt_retrain_data"

# 原 schema prompt（== 原 jsonl 里 user 的 <image> 之后内容），用于定位替换。
# 不依赖源文件内嵌字符串，而是"读第一条样本取出旧 body 作为待替换基准"更稳，
# 但为可读，直接按与 data_convert 相同的构造写死（键值枚举部分逐字一致）。
_KEYS = [
    "sleeve_length: sleeveless, short-sleeve, medium-sleeve, long-sleeve, not long-sleeve",
    "lower_clothing_length: three-point, medium short, three-quarter, long",
    "socks: no, socks, leggings",
    "hat: no, yes",
    "glasses: no, eyeglasses, sunglasses, glasses in hand or on clothes",
    "neckwear: no, yes",
    "wrist_wearing: no, yes",
    "ring: no, yes",
    "waist_accessories: no, belt, clothing at waist",
    "neckline: V-shape, square, round, standing, lapel, suspenders",
    "outer_clothing_is_cardigan: yes, no",
    "upper_clothing_covering_navel: no, yes",
]
_FAB = "denim, cotton, leather, furry, knitted, chiffon, other"
_PAT = "floral, graphic, striped, pure color, lattice, other, color block"

# 强化版 Rules：保持键/值枚举不变，仅改写并扩充 Rules 的"可见性纪律"。
# 修正(2026-09-07)：本数据集中某键在"该部位/衣物未出现或不可见"时是被**省略**的
# （如没有外衣则无 outer_*、看不清该配饰部位则省略该键），因此纪律是"仅在有对应
# 衣物且能看清时输出该键；否则省略"，绝不能鼓励把不可见项臆造为 no。
ENHANCED_RULES = (
    "Rules:\n"
    "- Only output a key if the corresponding clothing part or item actually appears in "
    "the image AND is clearly visible enough for you to decide its value.\n"
    "- If the relevant clothing part is absent (e.g. no lower garment, no outer garment) "
    "or is not visible, omit that key entirely.\n"
    "- For each small accessory and item (ring, neckwear, wrist_wearing, waist_accessories, "
    "hat, glasses, socks): inspect the corresponding part of the body one by one. Only "
    "answer \"yes\" (or a specific non-\"no\" value) if you can actually see it. If the "
    "part is clearly visible but the item is not worn, output \"no\". If the part cannot be "
    "seen well enough, omit the key.\n"
    "- Do NOT guess or invent a value for a key whose clothing part is absent or unclear "
    "in the image.\n"
    "- Use only the allowed values listed above.\n"
    "- Output the JSON object only, without any extra text or explanation."
)


def build_enhanced_prompt() -> str:
    lines = ["Extract the clothing attributes visible in this image and output a JSON object.",
             "", "Allowed keys and values:"]
    lines += [f"- {k}" for k in _KEYS]
    for part in ("upper", "lower", "outer"):
        lines.append(f"- {part}_fabric: {_FAB}")
        lines.append(f"- {part}_pattern: {_PAT}")
    lines.append("")
    lines.append(ENHANCED_RULES)
    return "\n".join(lines)


def main() -> None:
    # 取一份旧 body 作为替换基准（val 首条）
    with open(os.path.join(SRC_DIR, "deepfashion_attr_json_val.jsonl"), encoding="utf-8") as fh:
        old_body = json.loads(fh.readline())["messages"][0]["content"][len("<image>"):]
    new_prompt = build_enhanced_prompt()

    os.makedirs(OUT_DIR, exist_ok=True)
    # images 软链由外层保证（experiments/qwen3vl8b/badcase_study/prompt_retrain_data/images -> 原 images）
    dataset_info = {}
    for split in ("train", "val", "test"):
        src = os.path.join(SRC_DIR, f"deepfashion_attr_json_{split}.jsonl")
        dst = os.path.join(OUT_DIR, f"deepfashion_attr_json_p1_{split}.jsonl")
        replaced = 0
        with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
            for line in fin:
                if not line.strip():
                    continue
                rec = json.loads(line)
                body = rec["messages"][0]["content"][len("<image>"):]
                if body == old_body:
                    rec["messages"][0]["content"] = "<image>" + new_prompt
                    replaced += 1
                else:
                    # 兜底：非 schema 体（理论上不会发生），跳过不替换但保留
                    pass
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        name = f"deepfashion_attr_json_p1_{split}"
        dataset_info[name] = {
            "file_name": f"{name}.jsonl", "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {"role_tag": "role", "content_tag": "content",
                     "user_tag": "user", "assistant_tag": "assistant"},
        }
        print(f"[ok] {split}: 替换 {replaced}/{sum(1 for _ in open(dst, encoding='utf-8'))} 行 -> {dst}")

    with open(os.path.join(OUT_DIR, "dataset_info.json"), "w", encoding="utf-8") as fh:
        json.dump(dataset_info, fh, ensure_ascii=False, indent=2)
    # 打印新 prompt 供人工核对
    print("\n===== 新 user prompt（供核对）=====")
    print("<image>" + new_prompt)


if __name__ == "__main__":
    main()
