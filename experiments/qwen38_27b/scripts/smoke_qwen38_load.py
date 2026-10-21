#!/usr/bin/env python
"""冒烟验证：Qwen3.8-27B 能否在 lf 环境加载并完成一次「图像 -> JSON」前向。

用途：换基座前的可行性确认（架构/权重/processor/多卡切分/生成）。
不写任何实验产物，只打印日志。
"""
import json
import os
import time

import torch
from PIL import Image
from transformers import AutoProcessor
from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration

REPO = "/disk16T2/scratch/maotai/repos/vlm"
MODEL = os.path.join(REPO, "models/Qwen3.8-27B")
DS = os.path.join(REPO, "data/data_convert/deepfashion_multi")

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


log("loading processor ...")
proc = AutoProcessor.from_pretrained(MODEL)
log(f"processor OK: {type(proc).__name__}")

log("loading model (bf16, device_map=auto) ...")
model = Qwen3_5ForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto")
model.eval()
n_param = sum(p.numel() for p in model.parameters()) / 1e9
log(f"model OK: {n_param:.2f}B params, model_type={model.config.model_type}")

for i in range(torch.cuda.device_count()):
    log(f"  cuda:{i} allocated = {torch.cuda.memory_allocated(i) / 2**30:.1f} GiB")

with open(os.path.join(DS, "deepfashion_attr_json_val.jsonl")) as f:
    sample = json.loads(f.readline())
img_path = os.path.join(DS, sample["images"][0])
image = Image.open(img_path).convert("RGB")
user_text = sample["messages"][0]["content"].replace("<image>", "").strip()
log(f"sample image: {sample['images'][0]} size={image.size}")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": user_text},
        ],
    }
]
inputs = proc.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
)
inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
log(f"input_ids shape: {tuple(inputs['input_ids'].shape)}")

with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=256, do_sample=False)

gen = out[0][inputs["input_ids"].shape[1]:]
text = proc.decode(gen, skip_special_tokens=True)
log("=== GENERATED ===")
print(text, flush=True)
log("=== REFERENCE (ground truth) ===")
print(sample["messages"][1]["content"], flush=True)
log("=== DONE ===")
