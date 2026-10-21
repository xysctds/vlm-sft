#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 few-shot 评测数据：在 schema 指令基础上嵌入 2 个文本 JSON 示例（不带图），
用于评测"基座 few-shot vs 微调"（回答为什么需要微调而非 in-context 示例）。

输出: data/data_convert/deepfashion_multi/deepfashion_attr_json_fewshot_val.jsonl
并在 dataset_info.json 注册 deepfashion_attr_json_fewshot_val。
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))), "scripts"))
from data_convert_deepfashion_multi import build_schema_prompt

BASE = os.environ.get("VLM_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
DATA_DIR = f"{BASE}/data/data_convert/deepfashion_multi"

# 1) 从 train 取 2 个示例（固定取前 2 条，JSON 文本，不带图）
with open(f"{DATA_DIR}/deepfashion_attr_json_train.jsonl", encoding="utf-8") as fh:
    train_rows = [json.loads(l) for l in fh if l.strip()]
examples = [r["messages"][1]["content"] for r in train_rows[:2]]
print("示例 1:", examples[0][:80], "...")
print("示例 2:", examples[1][:80], "...")

schema = build_schema_prompt()
fewshot_instruction = (
    schema
    + "\n\nExample 1:\n" + examples[0]
    + "\n\nExample 2:\n" + examples[1]
    + "\n\nNow extract the attributes for the image below:"
)

# 2) 为 val 每条构造 few-shot user 消息（<image> 在原位）
with open(f"{DATA_DIR}/deepfashion_attr_json_val.jsonl", encoding="utf-8") as fh:
    val_rows = [json.loads(l) for l in fh if l.strip()]

out_rows = []
for r in val_rows:
    user_content = r["messages"][0]["content"]  # "<image>Extract the clothing..."
    # 把原指令替换为 few-shot 指令（保留 <image> 前缀）
    new_user = user_content.replace(user_content.split(">", 1)[1], "") + fewshot_instruction
    out_rows.append({
        "messages": [
            {"role": "user", "content": new_user},
            {"role": "assistant", "content": r["messages"][1]["content"]},
        ],
        "images": r["images"],
    })

out_path = f"{DATA_DIR}/deepfashion_attr_json_fewshot_val.jsonl"
with open(out_path, "w", encoding="utf-8") as fh:
    for r in out_rows:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"[ok] {len(out_rows)} 条 -> {out_path}")
print("示例 user 开头:", out_rows[0]["messages"][0]["content"][:120].replace("\n", " "), "...")

# 3) 注册 dataset_info.json
ds_path = f"{DATA_DIR}/dataset_info.json"
with open(ds_path, encoding="utf-8") as fh:
    ds = json.load(fh)
ds["deepfashion_attr_json_fewshot_val"] = {
    "file_name": "deepfashion_attr_json_fewshot_val.jsonl",
    "formatting": "sharegpt",
    "columns": {"messages": "messages", "images": "images"},
    "tags": {"role_tag": "role", "content_tag": "content",
             "user_tag": "user", "assistant_tag": "assistant"},
}
with open(ds_path, "w", encoding="utf-8") as fh:
    json.dump(ds, fh, ensure_ascii=False, indent=2)
print("[ok] dataset_info.json 已注册 deepfashion_attr_json_fewshot_val")
