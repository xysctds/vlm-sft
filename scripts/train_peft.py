#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PEFT 手写 LoRA 微调脚本 —— 复刻 llamafactory/config/qwen3vl8b_dfattr_train_cli.yaml
目标：Qwen3-VL-8B-Instruct 在 DeepFashion 属性 JSON 抽取任务上的 LoRA SFT。

====================================================================
【与 LLaMA-Factory YAML 的等价性对照】
  字段(YAML)                     本脚本实现
  ----------------------------------------------------------------
  finetuning_type=lora           LoraConfig(r=16, alpha=32, dropout=0.05)
  lora_target=all                target_modules 正则只打 language_model 的
                                 attention(q/k/v/o)+MLP(gate/up/down)
                                 （约等于，需对 print_trainable_parameters 核对）
  freeze_vision_tower=true       requires_grad=False 冻结 model.visual
  freeze_multi_modal_projector   requires_grad=False 冻结 model.merger
  template=qwen3_vl_nothink      该模型默认 chat_template 无 think 块==nothink ✅
  train_on_prompt=false          把 prompt 部分 label 置 -100（只训 assistant）
  image_min/max_pixels           1024 / 825750（=750×1101 原图，不压缩）
  cutoff_len=2048                processor 里 truncation max_length=2048
  batch=4 × grad_accum=3 × 3卡   等效 batch=36（单卡改 grad_accum=9）
  val_size=0.05                  从训练集随机切 5% 画 val loss
  eval_strategy=steps/100        eval_steps=100, per_device_eval_batch_size=4
  max_samples=100000             你的数据~3.5w，实为 no-op，仍补上
  optim=adamw_torch              TrainingArguments 默认即 adamw_torch
  bf16 / lr=1e-4 / cosine / 等   一一对应

运行（与 llamafactory 同环境：transformers>=4.57 / peft / datasets / torch）：
  单卡：   python scripts/train_peft.py
  3 卡：   torchrun --nproc_per_node=3 scripts/train_peft.py
  单卡请把 TRAIN_ARGS 的 gradient_accumulation_steps 改成 9。
====================================================================
"""
import os
import torch
from PIL import Image
from datasets import load_dataset           # 读 jsonl 并做 map 预处理
from peft import LoraConfig, get_peft_model # LoRA 本体
from transformers import (
    AutoModelForImageTextToText,  # 新版多模态入口类（Vision2Seq 已弃用）
    AutoProcessor,                # 同时加载 tokenizer + image processor
    Trainer,                      # transformers 训练器（DDP 用 torchrun 启动即多卡）
    TrainingArguments,
)

# ------------------------- 配置（对应你的 YAML） -------------------------
MODEL_PATH = "models/Qwen3-VL-8B-Instruct"
DATASET_DIR = "data/data_convert/deepfashion_multi"
TRAIN_FILE = os.path.join(DATASET_DIR, "deepfashion_attr_json_train.jsonl")
IMG_BASE = os.path.join(DATASET_DIR, "images")  # jsonl 里 images 是相对此目录的路径

MAX_SAMPLES = 100000   # 与 YAML 一致（数据量小，实为 no-op）
CUTOFF = 2048          # 截断长度：超长样本直接截掉，节省显存
VAL_SIZE = 0.05        # 从训练集切 5% 当验证集（只画 val loss，不与手动 test 重叠）

# LoRA 超参（与 YAML 完全一致）
LORA_R = 16
LORA_ALPHA = 32        # 实际缩放系数 = alpha / r = 2
LORA_DROPOUT = 0.05
# lora_target=all 的口径：只打 language_model 的 attention + MLP 线性层。
# vision tower(visual) 和投影层(merger) 已被冻结，peft 不会打它们。
LORA_TARGET_MODULES = r"model\.language_model\..*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"

# Trainer 训练参数（逐条对应 YAML）
TRAIN_ARGS = dict(
    output_dir="peft_saves/dfattr/lora/qwen3vl8b_dfattr_lora",  # 保存 LoRA adapter
    learning_rate=1e-4,
    num_train_epochs=3.0,
    per_device_train_batch_size=4,   # 每卡 batch
    gradient_accumulation_steps=3,   # 3卡DDP: 4×3×3=36；单卡改 9
    lr_scheduler_type="cosine",
    warmup_steps=150,                # 对应 YAML 的 warmup
    max_grad_norm=1.0,
    logging_steps=10,
    save_steps=100,
    eval_strategy="steps",           # 按步 eval，配合 eval_steps
    eval_steps=100,
    per_device_eval_batch_size=4,
    bf16=True,                       # 半精度训练，省显存
    seed=42,
    dataloader_num_workers=4,        # YAML 是 preprocessing_num_workers=16，无语义差异
    remove_unused_columns=False,     # 关键！否则 Trainer 会删掉 pxl 等自定义列
    report_to="none",                # 不传 wandb 等
)


def load_img(rel_path: str) -> Image.Image:
    """按 jsonl 里的相对路径加载图片，统一转 RGB。"""
    return Image.open(os.path.join(IMG_BASE, rel_path)).convert("RGB")


def build_conversation(messages):
    """
    把 jsonl 的 messages 转成 processor 认识的"结构化对话"（含图片占位）。
    jsonl 里 content 是纯字符串 + <image>，这里把图片抽出来作为 type:image 块，
    文字里残留的 <image> 去掉，避免模板里出现第二个占位符。
    同时返回只含 user(prompt) 的对话，供计算 label mask 的长度。
    """
    user, asst = messages[0], messages[1]
    user_text = user["content"].replace("<image>", "").strip()

    # 完整对话（user 带图 + assistant 回答）
    convo = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]},
        {"role": "assistant", "content": [{"type": "text", "text": asst["content"]}]},
    ]
    # 仅 prompt（含图），add_generation_prompt=True 让它在 <|im_start|>assistant\n 处停住
    prompt_convo = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]},
    ]
    return convo, prompt_convo


def main():
    # ---------- 第 1 步：加载 processor（tokenizer + 图像处理器）----------
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    # 对齐 image_min/max_pixels：1024(=32×32) / 825750(=750×1101 原图，不压缩)。
    # max_pixels 是"上限"，原图小于它就不压缩，保住细节；过大则爆显存。
    processor.image_processor.min_pixels = 1024
    processor.image_processor.max_pixels = 825750

    # ---------- 第 2 步：读数据并切验证集 ----------
    raw = load_dataset("json", data_files=TRAIN_FILE)["train"]
    if MAX_SAMPLES is not None:
        raw = raw.select(range(min(MAX_SAMPLES, len(raw))))  # max_samples=100000
    if VAL_SIZE > 0:
        # train_test_split 按 seed 随机切，得到画 val loss 的验证集（与手动 test 零重叠）
        split = raw.train_test_split(test_size=VAL_SIZE, seed=42)
        train_ds, eval_ds = split["train"], split["test"]
    else:
        train_ds, eval_ds = raw, None

    # ---------- 第 3 步：预处理函数（这是 VLM 最核心、最容易错的一步）----------
    def tokenize(example):
        """
        每个样本做：
          1) 套 chat template，得到含 <|image_pad|> 的文本；
          2) 把图片 + 文本交给 processor，得到 input_ids / attention_mask /
             pixel_values / image_grid_thw；
          3) 计算"纯 prompt"的 token 数，把 prompt 部分的 label 置 -100，
             实现 train_on_prompt=false（只对 assistant 回答算 loss）。
        """
        convo, prompt_convo = build_conversation(example["messages"])
        image = load_img(example["images"][0])   # 本任务一图一样本，取第一张

        # 文本化：把结构化对话渲染成 token 串（含 <|image_pad|> 占位，此时还是文本）
        full_text = processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
        prompt_text = processor.apply_chat_template(prompt_convo, tokenize=False, add_generation_prompt=True)

        # 完整编码：processor 会把 <|image_pad|> 替换成真正的 vision token 序列
        enc = processor(
            images=[image], text=full_text, return_tensors="pt",
            padding=False, truncation=True, max_length=CUTOFF,
        )
        # prompt 单独编码——【必须也传图像】！否则 <|image_pad|> 只算 1 个占位符，
        # 而带图会展开成几十个 vision token，两者长度不同会导致 mask 错位。
        # 同一张图 + 相同 min/max_pixels => 展开的 vision token 数相同，可放心对齐。
        penc = processor(
            images=[image], text=prompt_text, return_tensors="pt",
            padding=False, truncation=True, max_length=CUTOFF,
        )
        # 用 prompt 的 token 数做边界；再和 full 长度取 min，防截断导致越界
        prompt_len = min(penc["input_ids"].shape[1], enc["input_ids"].shape[1])

        input_ids = enc["input_ids"][0].tolist()
        # label：prompt 部分 -100（不参与 loss），assistant 部分 = 原文 token
        labels = [-100] * prompt_len + input_ids[prompt_len:]
        attention_mask = enc["attention_mask"][0].tolist()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": enc["pixel_values"][0],   # 图像 tensor (1,3,H,W)
            "image_grid_thw": enc["image_grid_thw"][0],  # 图像分块信息，位置编码用
        }

    cols = ["messages", "images"]   # map 后删除原始列，只留 token 特征
    train_ds = train_ds.map(tokenize, remove_columns=cols)
    if eval_ds is not None:
        eval_ds = eval_ds.map(tokenize, remove_columns=cols)

    # ---------- 第 4 步：加载模型 + 冻结 + 套 LoRA ----------
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",  # 没装 flash-attn 就改成 "auto"
    )
    model.config.use_cache = False  # 训练必须关 KV cache 才省显存/正确

    # freeze_vision_tower + freeze_multi_modal_projector：
    # vision encoder 已训好、投影层参数少，都冻结；只微调 language_model。
    for name, p in model.named_parameters():
        if name.startswith("model.visual") or name.startswith("model.merger"):
            p.requires_grad = False

    # LoRA 配置：目标模块用正则（可训练参数只在这批线性层的低秩分支上）
    lora_cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES, task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()  # 打印可训练参数量，用于和 YAML 的 lora_target=all 核对

    # ---------- 第 5 步：自定义 collator（不同尺寸图像不能直接 stack）----------
    class VLDataCollator:
        """
        默认 collator 会要求 batch 内所有 tensor 形状一致，但每张图 H×W 不同，
        pixel_values 形状不同，直接 stack 会崩。这里手动处理：
          - input_ids / attention_mask / labels：右 padding 到 batch 内最长
          - pixel_values：零填充到 batch 内最大 H×W
          - image_grid_thw 直接 cat（每图一行，无需对齐）
        """
        def __init__(self, proc):
            self.pad_id = proc.tokenizer.pad_token_id

        def __call__(self, features):
            max_len = max(len(f["input_ids"]) for f in features)
            input_ids, attention_mask, labels = [], [], []
            for f in features:
                pl = len(f["input_ids"])
                input_ids.append(f["input_ids"] + [self.pad_id] * (max_len - pl))
                attention_mask.append(f["attention_mask"] + [0] * (max_len - pl))
                labels.append(f["labels"] + [-100] * (max_len - pl))  # padding 部分不算 loss
            max_h = max(f["pixel_values"].shape[-2] for f in features)
            max_w = max(f["pixel_values"].shape[-1] for f in features)
            pvs = []
            for f in features:
                pv = f["pixel_values"]
                h, w = pv.shape[-2], pv.shape[-1]
                if (h, w) != (max_h, max_w):
                    pad = torch.zeros(1, 3, max_h, max_w, dtype=pv.dtype)
                    pad[:, :, :h, :w] = pv  # 原图放左上角，其余零填充
                    pv = pad
                pvs.append(pv)
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "pixel_values": torch.cat(pvs, dim=0),
                "image_grid_thw": torch.cat([f["image_grid_thw"] for f in features], dim=0),
            }

    # ---------- 第 6 步：组装 Trainer 并训练 ----------
    trainer = Trainer(
        model=model,
        args=TrainingArguments(**TRAIN_ARGS),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=VLDataCollator(processor),
    )
    trainer.train()
    trainer.save_model()  # 保存的是 LoRA adapter（peft 格式，与 llamafactory 可互相加载）
    processor.save_pretrained(TRAIN_ARGS["output_dir"])


if __name__ == "__main__":
    main()
