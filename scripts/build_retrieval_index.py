#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""构建 DeepFashion 检索索引（B 方案：属性 + 视觉向量）。

用途
----
用冻结的 Qwen3-VL 视觉塔（``model.visual``）对库内每条"可展示的穿搭图"抽
embedding（mean-pool 视觉 patch 特征），叠加官方标注的属性约束，产出检索库：
    1. ``index_meta.json``   每条穿搭图的元数据（list）：{id, sub, view, angle, image, attrs, caption}
    2. ``index_vectors.npy`` 与 meta 同序的归一化视觉向量（[N, 1152] bf16）

口径
----
- 索引粒度：**全量图都进索引**（每条 = 一张穿搭图，含 ``id``/``sub``/``view``/``angle``/``image``/``attrs``/``caption``）。
  建索引不做角度预过滤；是否屏蔽某类角度（如 ``side``/``back`` 或只留 ``front``）由**检索侧开关**决定，
  以后换策略无需重抽索引。
- 属性约束用官方 ``SHAPE_ATTRS / FABRIC_VALS / PATTERN_VALS``（与训练数据同源）。

运行（``vllm`` 环境，模型在 ``models/Qwen3-VL-8B-Instruct``）::

    conda activate vllm
    python scripts/build_retrieval_index.py --limit 200   # 小样本冒烟
    python scripts/build_retrieval_index.py               # 全量（约 4.4 万条穿搭图）

产物默认写到 ``data/data_convert/deepfashion_multi/retrieval/``（git 忽略）。
"""

import argparse
import json
import os
import random
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_convert_deepfashion_multi import (  # noqa: E402
    FABRIC_VALS,
    PATTERN_VALS,
    SHAPE_ATTRS,
    build_attrs,
    read_txt_ann,
)

MODEL_PATH = "models/Qwen3-VL-8B-Instruct"
IMG_DIR = "data/data_convert/deepfashion_multi/images"

# 文件名 -> (商品 id, 子序号, 视角号, 角度)
_FN_RE = re.compile(r"^.*-id_(\d+)-(\d+)_(\d+)_([a-z_]+)\.jpg$")


def parse_filename(fn: str) -> tuple[str, str, str, str] | None:
    """解析文件名，返回 (商品id, 子序号, 视角号, 角度)。"""
    m = _FN_RE.match(fn)
    if not m:
        return None
    pid, sub, view, angle = m.groups()
    return pid, sub, view, angle


def build_item_index(data_dir: str, img_dir: str = IMG_DIR) -> list[dict]:
    """全量聚合官方标注 -> 每条穿搭图一条 metadata（list）。

    粒度：全库每张图一个条目，不做角度过滤。每个条目：
    {id, sub, view, angle, image, attrs, caption}。attr 缺失（无官方标注）时为空 dict。
    """
    captions = json.load(open(os.path.join(data_dir, "captions.json"), encoding="utf-8"))
    shape = read_txt_ann(os.path.join(data_dir, "labels", "shape", "shape_anno_all.txt"))
    fabric = read_txt_ann(os.path.join(data_dir, "labels", "texture", "fabric_ann.txt"))
    pattern = read_txt_ann(os.path.join(data_dir, "labels", "texture", "pattern_ann.txt"))
    attrs = build_attrs(shape, fabric, pattern)  # {img: {attr: value}}

    items: list[dict] = []
    for fn in sorted(os.listdir(img_dir)):
        parsed = parse_filename(fn)
        if parsed is None:
            continue
        pid, sub, view, angle = parsed
        items.append({
            "id": pid,
            "sub": sub,
            "view": view,
            "angle": angle,
            "image": fn,
            "attrs": attrs.get(fn, {}),
            "caption": str(captions.get(fn, "")),
        })
    return items


def load_model(device: str | None = None):
    """加载冻结的视觉塔并移到目标设备。

    注意：必须显式 ``model.to(device)`` 并确保输入 tensor 在相同设备，
    否则 ``model.model.visual`` 会在 CPU 上逐图推理，导致全量抽取极慢
    （8B 视觉塔 CPU 推理会卡住）。默认取 CUDA 可用设备。
    """
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    processor.image_processor.min_pixels = 1024
    processor.image_processor.max_pixels = 825750
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH, trust_remote_code=True, dtype=torch.bfloat16,
    )
    model.to(device).eval()
    return processor, model, device


def embed_image(processor, model, path: str, device: str) -> np.ndarray:
    """返回单张图的 mean-pool 视觉向量（[1152] bf16），推理在指定设备上。"""
    import torch
    from PIL import Image

    img = Image.open(path).convert("RGB")
    # text='' 空占位：兼容 transformers 5.8（只传 images 会因 image_token in None 崩）
    out = processor(images=[img], text="", return_tensors="pt")
    pv = out["pixel_values"].to(device)
    grid_thw = out["image_grid_thw"].to(device)
    with torch.no_grad():
        feats = model.model.visual(pv, grid_thw=grid_thw)
        vec = feats.last_hidden_state.mean(dim=0)  # [num_patches,1152] -> [1152]
    return vec.detach().cpu().to(torch.float32).numpy()


def build_shard(out_dir: str, items: list[dict], processor, model, device,
                shard: int = -1, nshards: int = 1) -> None:
    """抽取 embedding 并落盘。shard>=0 时分片写独立文件；shard<0 写最终文件。"""
    print(f"[3/4] 抽取 embedding（分片 {shard + 1}/{nshards}，共 {len(items)} 条）...")
    order, vecs = [], []
    for i, it in enumerate(items):
        path = os.path.join(IMG_DIR, it["image"])
        if not os.path.exists(path):
            print(f"     跳过缺失图 {it['image']}")
            continue
        vec = embed_image(processor, model, path, device)
        vec /= np.linalg.norm(vec) + 1e-9  # 归一化，检索直接用点积
        order.append(it)
        vecs.append(vec)
        if (i + 1) % 200 == 0:
            print(f"     {i + 1}/{len(items)} done")

    if not vecs:
        print("ERROR: 没有抽到任何向量", file=sys.stderr)
        sys.exit(1)

    meta = order
    arr = np.stack(vecs).astype(np.float32)
    if shard >= 0:
        vec_path = os.path.join(out_dir, f"index_vectors.shard{shard}.npy")
        meta_path = os.path.join(out_dir, f"index_meta.shard{shard}.json")
    else:
        vec_path = os.path.join(out_dir, "index_vectors.npy")
        meta_path = os.path.join(out_dir, "index_meta.json")
    np.save(vec_path, arr)
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)
    print(f"     [{shard + 1}/{nshards}] vectors: {arr.shape} items: {len(meta)} -> {vec_path}")

    # 每片内部自检：vector[0] 检索自身应 top-1
    self_cos = arr @ arr[0]
    self_top = int(np.argmax(self_cos))
    print(f"     自检：vector[0] top-1 = {meta[self_top]['id']} (正确应为 {meta[0]['id']})")


def merge_shards(out_dir: str) -> None:
    """合并所有 index_*.shard*.npy/json 为最终 index_vectors.npy + index_meta.json。"""
    files = sorted(f for f in os.listdir(out_dir)
                   if f.startswith("index_vectors.shard") and f.endswith(".npy"))
    if not files:
        print("ERROR: 没有找到分片文件", file=sys.stderr)
        sys.exit(1)
    all_vecs, all_meta = [], []
    for f in files:
        all_vecs.append(np.load(os.path.join(out_dir, f)))
        shard = f.replace("index_vectors.shard", "").replace(".npy", "")
        mf = os.path.join(out_dir, f"index_meta.shard{shard}.json")
        all_meta.extend(json.load(open(mf, encoding="utf-8")))
    arr = np.concatenate(all_vecs, axis=0).astype(np.float32)
    np.save(os.path.join(out_dir, "index_vectors.npy"), arr)
    with open(os.path.join(out_dir, "index_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(all_meta, fh, ensure_ascii=False, indent=1)
    # 清理分片文件
    for f in files:
        os.remove(os.path.join(out_dir, f))
        os.remove(os.path.join(out_dir, f.replace(".npy", ".json").replace("vectors", "meta")))
    print(f"[merge] 合并完成：{len(all_meta)} 件 -> index_vectors.npy {arr.shape} + index_meta.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/data_convert/deepfashion_multi",
                        help="解压后的 DeepFashion-MultiModal 工作目录")
    parser.add_argument("--limit", type=int, default=0,
                        help="只处理前 N 个商品 id（冒烟用，0=全量）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default=None,
                        help="产物目录（默认 <data>/retrieval）")
    parser.add_argument("--shard", type=int, default=-1,
                        help="并行分片号（0-based）；与 --nshards 搭配")
    parser.add_argument("--nshards", type=int, default=1,
                        help="总分片数")
    parser.add_argument("--merge", action="store_true",
                        help="合并所有分片为最终索引")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.data, "retrieval")
    os.makedirs(out_dir, exist_ok=True)

    if args.merge:
        merge_shards(out_dir)
        return

    print("[1/4] 聚合官方标注 -> 穿搭图候选索引 ...")
    items = build_item_index(args.data, os.path.join(args.data, "images"))
    if args.limit:
        rng = random.Random(args.seed)
        rng.shuffle(items)
        items = items[: args.limit]
    print(f"     候选条目总数 {len(items)}，本次处理 {len(items)}")

    # 分片切分：均匀切成 nshards 段，每卡处理一段
    if args.nshards > 1 and args.shard >= 0:
        chunk = (len(items) + args.nshards - 1) // args.nshards
        items = items[args.shard * chunk:(args.shard + 1) * chunk]
        print(f"     [shard {args.shard}/{args.nshards}] 处理 {len(items)} 条")
        assert args.shard < args.nshards, "--shard 必须 < --nshards"

    print("[2/4] 加载视觉塔 + processor ...")
    processor, model, device = load_model()

    build_shard(out_dir, items, processor, model, device,
                shard=args.shard, nshards=args.nshards)


if __name__ == "__main__":
    main()
