#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""量化单条训练样本进 LLM 的各部分 token，写日志（纯 CPU，不加载模型权重、不占 GPU）。

口径：完全沿 llamafactory 训练管线（qwen3_vl_nothink 模板 + Qwen3VLPlugin 占位展开）：
  - <image> 展开为 <|vision_start|><|image_pad|>×N<|vision_end|>（N 由 grid_thw 决定）
  - processor 再按 grid_thw 把 <|image_pad|> 展开成 N 个，最终 input_ids 与训练逐 token 一致
  - 进 LLM 视觉 token = <|image_pad|> 占位 N 个 + <|vision_start|>/<|vision_end|> 2 个标记

用法（仓库根，跑在 lf 环境）::
    conda activate lf
    python experiments/smoke/scripts/bench_tokens_mem.py                 # 默认取 3 条
    python experiments/smoke/scripts/bench_tokens_mem.py --samples 5     # 取 5 条

日志写 `experiments/smoke/bench_tokens_mem.log`。
"""

import argparse
import json
import os
import random
from typing import Any

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
LOG = os.path.join(REPO, "experiments", "smoke", "bench_tokens_mem.log")
MODEL = os.path.join(REPO, "models", "Qwen3-VL-8B-Instruct")
TRAIN_JSONL = os.path.join(REPO, "data", "data_convert", "deepfashion_multi",
                           "deepfashion_attr_json_train.jsonl")
IMAGE_DIR = os.path.join(REPO, "data", "data_convert", "deepfashion_multi", "images")

# 与 llamafactory Qwen3VLPlugin（alpha/llamafactory/src/llamafactory/data/mm_plugin.py）一致：
# <image> -> <|vision_start|> + <|image_pad|>×N + <|vision_end|>
VISION_BOS = "<|vision_start|>"
VISION_EOS = "<|vision_end|>"
IMAGE_PAD = "<|image_pad|>"
IMAGE_PLACEHOLDER = "<image>"

# 采样范围：随机从训练 jsonl 前多少条里取材（供可复现的随机抽样）。
SAMPLE_POOL = 200
SEED = 42

SEP = "=" * 70


def build_prompt(user: str, assistant: str) -> str:
    """按 qwen3_vl_nothink 模板渲染对话，并把 <image> 展开为 vision 标记 + image_pad。"""
    image_expand = f"{VISION_BOS}{IMAGE_PAD}{VISION_EOS}"
    user = user.replace(IMAGE_PLACEHOLDER, image_expand)
    return (f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant}<|im_end|>\n")


def load_samples(samples: int) -> list[tuple[str, str]]:
    """随机取 `samples` 条真实训练样本，返回 [(文件名, prompt 字符串)]。

    一次遍历 jsonl 收集前 SAMPLE_POOL 条，按固定 SEED 抽样，保证可复现。
    """
    entries: list[tuple[str, str, str]] = []
    with open(TRAIN_JSONL, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            entries.append(
                (os.path.basename(d["images"][0]),
                 d["messages"][0]["content"], d["messages"][1]["content"]))
            if len(entries) >= SAMPLE_POOL:
                break

    random.seed(SEED)
    return [(fname, build_prompt(user, asst))
            for fname, user, asst in random.sample(entries, samples)]


def count_tokens(proc: Any, prompt: str, image: Any, cutoff_len: int) -> dict[str, int]:
    """对一条已渲染的 prompt + 图像统计各部分 token 数。"""
    inp = proc(text=[prompt], images=[image], return_tensors="pt")
    input_ids = inp["input_ids"]

    thw = inp["image_grid_thw"][0].tolist()
    tok = proc.tokenizer
    n_total = input_ids.shape[1]
    n_vis_pad = (input_ids == tok.convert_tokens_to_ids(IMAGE_PAD)).sum().item()
    n_vis_mark = ((input_ids == tok.convert_tokens_to_ids(VISION_BOS)).sum().item()
                  + (input_ids == tok.convert_tokens_to_ids(VISION_EOS)).sum().item())
    n_vis_llm = n_vis_pad + n_vis_mark  # 进 LLM 视觉 token = 占位 + 2 标记
    n_text = n_total - n_vis_llm        # 文本部分（去掉全部视觉 token）
    n_patch = thw[0] * thw[1] * thw[2]  # vision 未 merge 的 patch 数

    return dict(grid_thw=thw, total=n_total, vis_pad=n_vis_pad, vis_mark=n_vis_mark,
                vis_llm=n_vis_llm, text=n_text, patch=n_patch,
                trunc=n_total > cutoff_len)


def run(samples: int, min_pixels: int, max_pixels: int, cutoff_len: int, out) -> None:
    """主流程：加载模型 → 逐条统计 → 打印 + 写日志。"""
    from PIL import Image
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    proc.image_processor.min_pixels = min_pixels
    proc.image_processor.max_pixels = max_pixels

    def emit(line: str) -> None:
        print(line)
        out.write(line + "\n")

    emit(SEP)
    emit(f"[token] 模型={os.path.basename(MODEL)}  min_pixels={min_pixels}  "
         f"max_pixels={max_pixels}  cutoff_len={cutoff_len}")
    emit(SEP)

    sample_paths = load_samples(samples)
    results: list[dict[str, int]] = []
    for i, (fname, prompt) in enumerate(sample_paths, 1):
        image = Image.open(os.path.join(IMAGE_DIR, fname)).convert("RGB")
        r = count_tokens(proc, prompt, image, cutoff_len)
        results.append(dict(file=fname, **r))

        emit(f"[样本{i}] {fname}")
        emit(f"  grid_thw={r['grid_thw']}  完整序列(input_ids)={r['total']}  "
             f"其中视觉={r['vis_llm']}(占位{r['vis_pad']}+标记{r['vis_mark']})  "
             f"文本(去视觉)={r['text']}")
        emit(f"  视觉patch(未merge)={r['patch']}  进LLM视觉token={r['vis_llm']}")
        emit(f"  cutoff_len={cutoff_len}  => " + ("截断!" if r["trunc"] else "不截断"))

    # 汇总
    n = len(results)
    avg = lambda k: sum(r[k] for r in results) / n
    rng = lambda k: (min(r[k] for r in results), max(r[k] for r in results))
    marks = {r["vis_mark"] for r in results}

    emit(SEP)
    emit(f"[token 汇总] 平均文本={avg('text'):.0f} (区间 {rng('text')[0]}~{rng('text')[1]})  "
         f"平均视觉(占位+标记)={avg('vis_llm'):.0f} (占位平均={avg('vis_pad'):.0f}, "
         f"标记恒为 {list(marks)[0]} / 一致={len(marks)==1})  "
         f"平均总序列={avg('total'):.0f} (区间 {rng('total')[0]}~{rng('total')[1]})  "
         f"cutoff={cutoff_len} => 全不截断={all(not r['trunc'] for r in results)}")
    emit("口径: 与 llamafactory qwen3_vl_nothink 训练逐 token 一致(含 vision_start/end 2 标记); "
         "llamafactory 超 cutoff 时截断续训而非丢弃, 见 processor/supervised.py")
    emit("参数总量校验: 8.81B / LoRA 43.6M / trainable% 0.4954% 见训练日志(此处不重复加载)")
    emit(SEP)
    emit(f"[done] 日志已写: {LOG}")
    out.flush()


def main() -> None:
    ap = argparse.ArgumentParser(description="量化单条训练样本进 LLM 的各部分 token（纯 CPU）。")
    ap.add_argument("--samples", type=int, default=100, help="取样多少条真实训练图")
    ap.add_argument("--min-pixels", type=int, default=1024)
    ap.add_argument("--max-pixels", type=int, default=825750)
    ap.add_argument("--cutoff-len", type=int, default=2048, help="训练配置的 cutoff_len")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "w", encoding="utf-8") as out:
        run(args.samples, args.min_pixels, args.max_pixels, args.cutoff_len, out)


if __name__ == "__main__":
    main()
