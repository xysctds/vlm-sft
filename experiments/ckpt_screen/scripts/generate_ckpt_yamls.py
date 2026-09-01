#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 checkpoint 快筛/确认评测配置（ckpt_*.yaml 自动生成、勿手改、已 gitignore）。

用法（在仓库根目录运行）：
    python experiments/ckpt_screen/scripts/generate_ckpt_yamls.py

生成（写入 `experiments/ckpt_screen/config/`）：
  - ckpt_<N>.yaml        快筛（max_samples=500）：100~2793 每 200 步一个点（15 个）
  - ckpt_<N>_full.yaml   全量确认（max_samples=100000）：900、2100
之后用 run_screen.sh 3 卡轮转跑快筛；全量确认单独跑 ckpt_<N>_full.yaml。
"""
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config")
VLM = "."  # 相对路径：配置在仓库根运行，直接写仓库内相对路径
MODEL = "models/Qwen3-VL-8B-Instruct"
LORA = f"{VLM}/experiments/main/weights/lora/qwen3vl8b_dfattr_lora"

SCREEN = [100, 300, 500, 700, 900, 1100, 1300, 1500, 1700, 1900, 2100, 2300, 2500, 2700, 2793]
FULL = [900, 2100]  # 全量确认的候选点

TPL = """# checkpoint 快筛配置（自动生成，勿手改）：checkpoint-{ckpt}
stage: sft
model_name_or_path: {model}
adapter_name_or_path: {lora}/checkpoint-{ckpt}
preprocessing_num_workers: 4
finetuning_type: lora
quantization_method: bnb
template: qwen3_vl_nothink
flash_attn: auto
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
output_dir: {vlm}/experiments/ckpt_screen/predictions/ckpt_{ckpt}{suffix}
trust_remote_code: true
ddp_timeout: 180000000
do_predict: true
"""


def main() -> None:
    n = 0
    for c in SCREEN:
        with open(os.path.join(ROOT, f"ckpt_{c}.yaml"), "w", encoding="utf-8") as f:
            f.write(TPL.format(ckpt=c, model=MODEL, lora=LORA, vlm=VLM, max_samples=500, suffix=""))
        n += 1
    for c in FULL:
        with open(os.path.join(ROOT, f"ckpt_{c}_full.yaml"), "w", encoding="utf-8") as f:
            f.write(TPL.format(ckpt=c, model=MODEL, lora=LORA, vlm=VLM, max_samples=100000, suffix="_full"))
        n += 1
    print(f"[ok] 生成 {n} 个配置 -> {ROOT}")


if __name__ == "__main__":
    main()
