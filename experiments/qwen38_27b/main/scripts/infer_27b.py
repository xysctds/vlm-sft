#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""27B 主实验：批量生成预测，替代 `llamafactory-cli train` 的 predict 管线。

## 为什么需要这个脚本

LLaMA-Factory 在 `hparams/parser.py` 硬禁止 `predict_with_generate` 与
DeepSpeed ZeRO-3 组合（`predict_with_generate is incompatible with DeepSpeed
ZeRO-3.`）；而 Qwen3.8-27B 的 bf16 权重约 54GB，单张 48GB 的 4090 装不下，
必须跨卡放置，所以原 8B 的评测配置（deepspeed 分片 + predict_with_generate）
在 27B 上从设计上就跑不通。

本脚本绕过该管线，改用 transformers 的 `device_map="auto"` 做 3 卡模型并行，
但**完整复用 LLaMA-Factory 自己的组件**，保证与训练、与 8B 主评测同口径：

    get_infer_args            # 解析同一份评测 yaml（device_map 置为 "auto"）
    load_tokenizer            # tokenizer + processor，并 patch 上 image_max_pixels
    get_template_and_fix_tokenizer
    load_model                # 含 LoRA adapter 注入（finetuning_type: lora）

prompt 构造与解码逐字复刻 `llamafactory/chat/hf_engine.py::HuggingfaceEngine._process_args`：

    mm_plugin.process_messages -> encode_oneturn -> mm_plugin.process_token_ids
    mm_plugin.get_mm_inputs    -> model.generate(greedy)

输出 `generated_predictions.jsonl` 每行 `{"prompt", "predict", "label"}`，与
`llamafactory` 的 `save_predictions` 完全同格式，后续 `scripts/convert_preds.py`
可直接消费，无需改动下游脚本。

## 用法

    python experiments/qwen38_27b/main/scripts/infer_27b.py \
        --config experiments/qwen38_27b/main/config/qwen38_27b_dfattr_eval_val_base.yaml \
        --batch_size 8

支持断点续跑：`generated_predictions.jsonl` 已有多少条合法记录就跳过多少条样本，
崩溃后重跑即可从断点继续（7 个变体每个约 1 小时，续跑能力很重要）。
"""

import argparse
import json
import os
import sys
import time
from copy import deepcopy

# 评测 yaml 里带有 deepspeed / predict_with_generate 等推理管线不认识的键，
# 允许多余键后由 HfArgumentParser 自行忽略（必须在本模块 import llamafactory 之前设好）。
os.environ.setdefault("ALLOW_EXTRA_ARGS", "1")

# 保证 `import llamafactory` 指向**仓库内**的 LLaMA-Factory，而不是环境里可能存在的旧版本。
# 仓库在 2026-10-20 做过一次目录重构：alpha/llamafactory -> vendor/llamafactory，
# 两个位置都探一下，谁在用都能跑（新结构优先）。
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
_LF_SRC_CANDIDATES = [
    os.path.join(_REPO_ROOT, "vendor", "llamafactory", "src"),
    os.path.join(_REPO_ROOT, "alpha", "llamafactory", "src"),
]
for _cand in _LF_SRC_CANDIDATES:
    if os.path.isdir(_cand):
        if _cand not in sys.path:
            sys.path.insert(0, _cand)
        break
else:
    print(
        "[warn] 未在仓库里找到 llamafactory 源码目录（vendor/llamafactory/src 或 alpha/llamafactory/src），"
        "将回退到环境里已安装的 llamafactory。",
        flush=True,
    )

import torch  # noqa: E402
import yaml  # noqa: E402
from llamafactory.data import get_template_and_fix_tokenizer  # noqa: E402
from llamafactory.extras.constants import IMAGE_PLACEHOLDER  # noqa: E402
from llamafactory.hparams import get_infer_args  # noqa: E402
from llamafactory.model import load_model, load_tokenizer  # noqa: E402
from transformers import GenerationConfig  # noqa: E402

PROGRESS_EVERY = 20


def log(msg: str) -> None:
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def count_done(path: str) -> int:
    """返回输出文件里已完整写好的记录数（用于断点续跑）。"""
    if not os.path.isfile(path):
        return 0
    done = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError:
                break  # 最后一行写坏了，从这条重来
            done += 1
    return done


def resolve_media(path: str, media_dir: str | None) -> str:
    """数据集中图片是 `images/xxx.jpg` 这样的相对路径，按媒体目录补全。"""
    if os.path.isabs(path) or os.path.isfile(path):
        return path
    if media_dir:
        cand = os.path.join(media_dir, path)
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError(f"找不到图片: {path} (media_dir={media_dir})")


def build_prompt_ids(template, tokenizer, processor, messages, images):
    """复刻 hf_engine._process_args 的 prompt 构造。"""
    messages = deepcopy(messages)
    if images and not any(IMAGE_PLACEHOLDER in m["content"] for m in messages):
        messages[0]["content"] = IMAGE_PLACEHOLDER * len(images) + messages[0]["content"]

    messages = template.mm_plugin.process_messages(messages, images, [], [], processor)
    paired_messages = messages + [{"role": "assistant", "content": ""}]
    prompt_ids, _ = template.encode_oneturn(tokenizer, paired_messages, None, None)
    prompt_ids, _ = template.mm_plugin.process_token_ids(prompt_ids, None, images, [], [], tokenizer, processor)
    return prompt_ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="评测 yaml（与训练/8B 同口径的那份）")
    parser.add_argument("--batch_size", type=int, default=8, help="生成批大小（越大越快，显存不够就调小）")
    parser.add_argument("--max_samples", type=int, default=0, help="只用前 N 条做冒烟测试，0=全量")
    parser.add_argument("--max_new_tokens", type=int, default=0, help="覆盖 yaml 里的 max_new_tokens，0=用 yaml")
    parser.add_argument("--out_dir", default=None, help="覆盖输出目录（冒烟测试用，避免污染正式产物）")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    model_args, data_args, finetuning_args, generating_args = get_infer_args(cfg)
    log(f"config={args.config}")
    log(f"model={model_args.model_name_or_path} adapter={model_args.adapter_name_or_path}")
    log(f"template={data_args.template} enable_thinking={data_args.enable_thinking} "
        f"image_max_pixels={model_args.image_max_pixels}")
    log(f"dataset_dir={data_args.dataset_dir} eval_dataset={data_args.eval_dataset}")

    # ---- 数据（顺序必须与 llamafactory 一致，convert_preds.py 依赖逐行对应）----
    eval_dataset = data_args.eval_dataset
    if isinstance(eval_dataset, (list, tuple)):
        if len(eval_dataset) != 1:
            raise ValueError(f"本脚本一次只处理一个数据集，收到: {eval_dataset}")
        eval_dataset = eval_dataset[0]
    data_path = os.path.join(data_args.dataset_dir, f"{eval_dataset}.jsonl")
    rows = read_jsonl(data_path)
    # max_samples 必须与 llamafactory 一致地消费（ckpt 快筛靠它只取前 500 条）；
    # 命令行 --max_samples 优先于配置，配置 100000 这类大值等价于不截断。
    cap = args.max_samples or int(getattr(data_args, "max_samples", 0) or 0)
    if cap and cap < len(rows):
        rows = rows[:cap]
    log(f"样本数: {len(rows)} <- {data_path}（max_samples={cap or '不限'}）")

    # ---- 模型 / tokenizer / 模板 ----
    tokenizer_module = load_tokenizer(model_args)
    tokenizer, processor = tokenizer_module["tokenizer"], tokenizer_module["processor"]
    tokenizer.padding_side = "left"  # decoder-only 生成必须左 padding
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    model = load_model(tokenizer, model_args, finetuning_args, is_trainable=False)
    model.eval()
    log(f"模型就绪，device={model.device}, dtype={next(model.parameters()).dtype}")

    media_dir = data_args.media_dir or data_args.dataset_dir
    max_new_tokens = args.max_new_tokens or cfg.get("max_new_tokens", 512)
    cutoff_len = cfg.get("cutoff_len", 2048)

    # 与 hf_engine._process_args 完全相同地组织生成参数（greedy）
    gen_cfg = generating_args.to_dict()
    gen_cfg.update(
        dict(
            do_sample=False,
            temperature=None,
            top_p=None,
            num_return_sequences=1,
            max_new_tokens=max_new_tokens,
            eos_token_id=template.get_stop_token_ids(tokenizer),
            pad_token_id=tokenizer.pad_token_id,
        )
    )
    gen_cfg.pop("temperature", None)
    gen_cfg.pop("top_p", None)
    generation_config = GenerationConfig(**gen_cfg)
    log(f"生成参数: {generation_config.to_diff_dict()}")

    out_dir = args.out_dir or cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "generated_predictions.jsonl")

    done = count_done(out_path)
    if done:
        log(f"断点续跑：已有 {done} 条，从第 {done} 条继续")
    if done >= len(rows):
        log("全部样本已生成，无需运行")
        return 0

    prompt_lens: list[int] = []
    n_done = done
    t_start = time.time()
    t_last = t_start
    n_last = done
    out_fh = open(out_path, "a", encoding="utf-8")

    try:
        for start in range(done, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            prompt_ids_list, labels, prompts_text, all_images, imglens = [], [], [], [], []
            for row in batch:
                images = [resolve_media(p, media_dir) for p in (row.get("images") or [])]
                msgs = row["messages"][:-1]  # 最后一条 assistant 是标签，不作为输入
                ids = build_prompt_ids(template, tokenizer, processor, msgs, images)
                if len(ids) > cutoff_len:
                    log(f"[warn] 第 {n_done} 条 prompt {len(ids)} > cutoff_len {cutoff_len}，训练侧会被截断")
                prompt_ids_list.append(ids)
                prompt_lens.append(len(ids))
                all_images.extend(images)
                imglens.append(len(images))
                labels.append(row["messages"][-1]["content"])
                prompts_text.append(tokenizer.decode(ids, skip_special_tokens=False))

            # 左 padding 对齐成一个 batch
            max_len = max(len(x) for x in prompt_ids_list)
            input_ids = torch.full((len(batch), max_len), tokenizer.pad_token_id, dtype=torch.long)
            attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
            for i, ids in enumerate(prompt_ids_list):
                input_ids[i, max_len - len(ids) :] = torch.tensor(ids, dtype=torch.long)
                attention_mask[i, max_len - len(ids) :] = 1

            gen_kwargs = {
                "input_ids": input_ids.to(model.device),
                "attention_mask": attention_mask.to(model.device),
                "generation_config": deepcopy(generation_config),
            }
            mm_inputs = template.mm_plugin.get_mm_inputs(
                images=all_images,
                videos=[],
                audios=[],
                imglens=imglens,
                vidlens=[0] * len(batch),
                audlens=[0] * len(batch),
                batch_ids=prompt_ids_list,
                processor=processor,
            )
            for key, value in mm_inputs.items():
                if not isinstance(value, torch.Tensor):
                    value = torch.tensor(value)
                if torch.is_floating_point(value):
                    value = value.to(model.dtype)
                gen_kwargs[key] = value.to(model.device)

            with torch.inference_mode():
                output = model.generate(**gen_kwargs)
            new_ids = output[:, input_ids.shape[1] :]
            texts = tokenizer.batch_decode(
                new_ids,
                skip_special_tokens=getattr(generation_config, "skip_special_tokens", True),
                clean_up_tokenization_spaces=True,
            )

            for i in range(len(batch)):
                rec = {"prompt": prompts_text[i], "predict": texts[i], "label": labels[i]}
                out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_fh.flush()
            n_done += len(batch)

            if (start // args.batch_size) % PROGRESS_EVERY == 0:
                now = time.time()
                speed = (n_done - n_last) / max(now - t_last, 1e-6)
                eta = (len(rows) - n_done) / max(speed, 1e-6)
                pct = n_done / len(rows)
                log(f"进度 {n_done}/{len(rows)} ({pct:.1%}) | {speed:.2f} 样本/s | "
                    f"ETA {eta / 60:.1f} min | prompt_len p50={sorted(prompt_lens)[len(prompt_lens) // 2]}")
                t_last, n_last = now, n_done
    finally:
        out_fh.close()

    elapsed = time.time() - t_start
    log(f"完成 {n_done - done} 条，用时 {elapsed / 60:.1f} min -> {out_path}")
    if prompt_lens:
        s = sorted(prompt_lens)
        log(f"prompt token 长度 min={s[0]} p50={s[len(s) // 2]} max={s[-1]}（训练侧 8B 参考 min=1170 p50=1268 max=1316）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
