#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 27B 的 checkpoint 快筛评测配置（ckpt27b_*.yaml，自动生成、勿手改）。

与 8B 版 `generate_ckpt_yamls.py` 的关系：**同一套选点与口径**，只是：
  - 基座          Qwen3-VL-8B-Instruct -> Qwen3.8-27B
  - LoRA 目录     qwen3vl8b_dfattr_lora -> qwen38_27b_dfattr_lora
  - 模板/思考开关  qwen3_vl_nothink -> qwen3_5 + enable_thinking: false
  - 新增          use_reentrant_gc: false 与 deepspeed（仅记录原始设计；实际由 infer_27b.py 执行）
  - 文件名前缀     ckpt_ -> ckpt27b_（**不碰 8B 的 ckpt_*.yaml**）
其余（选点 100~2793 共 15 个、max_samples=500、eval_dataset、解码、seed、分辨率）逐字一致。

用法（在仓库根目录运行）：
    python experiments/qwen38_27b/ckpt_screen/scripts/generate_ckpt_yamls_27b.py

生成的配置由 experiments/qwen38_27b/main/scripts/infer_27b.py 消费（原因：llamafactory 禁止
predict_with_generate + ZeRO-3，而 27B 单卡装不下）：
    python experiments/qwen38_27b/main/scripts/infer_27b.py \
        --config experiments/qwen38_27b/ckpt_screen/config/ckpt27b_500.yaml
"""
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config")
VLM = "."
MODEL = "models/Qwen3.8-27B"
LORA = f"{VLM}/experiments/qwen38_27b/main/weights/lora/qwen38_27b_dfattr_lora"
DEEPSPEED = f"{VLM}/experiments/qwen38_27b/main/config/ds_z3_27b.json"

SCREEN = [100, 300, 500, 700, 900, 1100, 1300, 1500, 1700, 1900, 2100, 2300, 2500, 2700, 2793]
FULL = [900, 2100]  # 全量确认的候选点（与 8B 相同）

TPL = """# checkpoint 快筛配置（自动生成，勿手改）：27B / checkpoint-{ckpt}
# 由 infer_27b.py 执行：python experiments/qwen38_27b/main/scripts/infer_27b.py --config <本文件>
stage: sft
model_name_or_path: {model}
adapter_name_or_path: {lora}/checkpoint-{ckpt}
preprocessing_num_workers: 4
finetuning_type: lora
quantization_method: bnb
template: qwen3_5
enable_thinking: false
use_reentrant_gc: false
flash_attn: auto
deepspeed: {deepspeed}
dataset_dir: {vlm}/data/data_convert/deepfashion_multi
eval_dataset: deepfashion_attr_json_val
cutoff_len: 2048
max_samples: {max_samples}
per_device_eval_batch_size: 4
predict_with_generate: true
report_to: none
max_new_tokens: 512
do_sample: false
seed: 42
image_max_pixels: 825750
output_dir: {vlm}/experiments/qwen38_27b/ckpt_screen/predictions/qwen38_27b_ckpt_{ckpt}{suffix}
trust_remote_code: true
ddp_timeout: 180000000
do_predict: true
"""


def main() -> None:
    n = 0
    for c in SCREEN:
        with open(os.path.join(ROOT, f"ckpt27b_{c}.yaml"), "w", encoding="utf-8") as f:
            f.write(TPL.format(ckpt=c, model=MODEL, lora=LORA, vlm=VLM, deepspeed=DEEPSPEED,
                               max_samples=500, suffix=""))
        n += 1
    for c in FULL:
        with open(os.path.join(ROOT, f"ckpt27b_{c}_full.yaml"), "w", encoding="utf-8") as f:
            f.write(TPL.format(ckpt=c, model=MODEL, lora=LORA, vlm=VLM, deepspeed=DEEPSPEED,
                               max_samples=100000, suffix="_full"))
        n += 1
    print(f"[ok] 生成 {n} 个 27B 配置 -> {ROOT}")


if __name__ == "__main__":
    main()
