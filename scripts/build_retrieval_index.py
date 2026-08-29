#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""构建 DeepFashion 检索索引（B 方案：属性 + 视觉向量）。

用途
----
用冻结的 Qwen3-VL 视觉塔（``model.visual``）对库内每个商品 id 的一张代表图抽
embedding（mean-pool 视觉 patch 特征），叠加官方标注的属性约束，产出检索库：
    1. ``index_meta.json``   商品 id -> {代表图, 属性字典, caption, 角度, 子序号}
    2. ``index_vectors.npy`` 与 meta 同序的归一化视觉向量（[N, 1152] bf16）

口径
----
- 商品 id：文件名 ``<大类>-<小类>-id_<商品id>-<子序号>_<视角号>_<角度>.jpg`` 中
  ``id_`` 之后到 ``-`` 之前的部分（全库 7975 个唯一 id；有 shape/caption 标注并入索引的 7972 个，3 个商品无标注被过滤）。
- 代表图优先 ``1_front`` / ``2_side``，其次任意一张；shape 有标注的优先，保证属性约束可用。
- 属性约束用官方 ``SHAPE_ATTRS / FABRIC_VALS / PATTERN_VALS``（与训练数据同源）。

运行（``peft`` 环境，模型在 ``models/Qwen3-VL-8B-Instruct``）::

    conda activate peft
    python scripts/build_retrieval_index.py --limit 200   # 小样本冒烟
    python scripts/build_retrieval_index.py               # 全量 7972 件

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

# 代表图角度优先级（front 最佳，side 次之，back/additional/full 兜底）
_ANGLE_PRIORITY = {"1_front": 0, "2_side": 1, "4_full": 2, "3_back": 3, "7_additional": 4}

# 文件名 -> (商品 id, 子序号, 视角号, 角度)
_FN_RE = re.compile(r"^.*-id_(\d+)-(\d+)_(\d+)_([a-z_]+)\.jpg$")


def parse_filename(fn: str) -> tuple[str, str, str, str] | None:
    """解析文件名，返回 (商品id, 子序号, 视角号, 角度)。"""
    m = _FN_RE.match(fn)
    if not m:
        return None
    pid, sub, view, angle = m.groups()
    return pid, sub, view, angle


def pick_representative(items: list[dict]) -> dict:
    """从同一商品的多张图中选代表图：角度优先，shape 有标注的优先。"""
    def key(it):
        angle = it["angle"]
        prio = _ANGLE_PRIORITY.get(angle, 99)
        # (是否少 shape 优先=0, 角度优先级, 子序号) —— 有 shape 标注的排前面
        return (0 if it["has_shape"] else 1, prio, int(it["sub"]))

    return min(items, key=key)


def build_item_index(data_dir: str) -> dict[str, dict]:
    """聚合官方标注 -> {商品id: {代表图, 属性, caption, angle}}。"""
    captions = json.load(open(os.path.join(data_dir, "captions.json"), encoding="utf-8"))
    shape = read_txt_ann(os.path.join(data_dir, "labels", "shape", "shape_anno_all.txt"))
    fabric = read_txt_ann(os.path.join(data_dir, "labels", "texture", "fabric_ann.txt"))
    pattern = read_txt_ann(os.path.join(data_dir, "labels", "texture", "pattern_ann.txt"))
    attrs = build_attrs(shape, fabric, pattern)  # {img: {attr: value}}

    # 图片集合：优先 shape 有标注的，否则 captions 有描述的
    image_pool = sorted(set(shape) | set(captions))
    groups: dict[str, list[dict]] = {}
    for fn in image_pool:
        parsed = parse_filename(fn)
        if parsed is None:
            continue
        pid, sub, view, angle = parsed
        groups.setdefault(pid, []).append({
            "img": fn,
            "sub": sub,
            "view": view,
            "angle": angle,
            "has_shape": fn in shape,
            "attrs": attrs.get(fn, {}),
            "caption": str(captions.get(fn, "")),
        })

    items: dict[str, dict] = {}
    for pid, group in groups.items():
        rep = pick_representative(group)
        # 若代表图缺 shape，尽量补充同 id 里有 shape 的视角属性（代表图优先，其次聚合）
        merged = dict(rep["attrs"])
        for g in group:  # 补全：同 id 其它视角可见的属性并入（真值检索约束）
            for k, v in g["attrs"].items():
                merged.setdefault(k, v)
        items[pid] = {
            "id": pid,
            "image": rep["img"],
            "angle": rep["angle"],
            "caption": rep["caption"],
            "attrs": merged,
            "n_views": len(group),
        }
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


def build_shard(out_dir: str, items: dict, pids: list[str], processor, model, device,
                shard: int = -1, nshards: int = 1) -> None:
    """抽取 embedding 并落盘。shard>=0 时分片写独立文件；shard<0 写最终文件。"""
    print(f"[3/4] 抽取 embedding（分片 {shard + 1}/{nshards}，共 {len(pids)} 件）...")
    order, vecs = [], []
    for i, pid in enumerate(pids):
        it = items[pid]
        path = os.path.join(IMG_DIR, it["image"])
        if not os.path.exists(path):
            print(f"     跳过缺失图 {it['image']}")
            continue
        vec = embed_image(processor, model, path, device)
        vec /= np.linalg.norm(vec) + 1e-9  # 归一化，检索直接用点积
        order.append(pid)
        vecs.append(vec)
        if (i + 1) % 200 == 0:
            print(f"     {i + 1}/{len(pids)} done")

    if not vecs:
        print("ERROR: 没有抽到任何向量", file=sys.stderr)
        sys.exit(1)

    meta = [items[pid] for pid in order]
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

    print("[1/4] 聚合官方标注 -> 商品索引 ...")
    items = build_item_index(args.data)
    pids = sorted(items)
    if args.limit:
        rng = random.Random(args.seed)
        rng.shuffle(pids)
        pids = pids[: args.limit]
    print(f"     商品总数 {len(items)}，本次处理 {len(pids)}")

    # 分片切分：按 id 排序后均匀切成 nshards 段，每卡处理一段
    if args.nshards > 1 and args.shard >= 0:
        chunk = (len(pids) + args.nshards - 1) // args.nshards
        pids = pids[args.shard * chunk:(args.shard + 1) * chunk]
        print(f"     [shard {args.shard}/{args.nshards}] 处理 {len(pids)} 件")
        assert args.shard < args.nshards, "--shard 必须 < --nshards"

    print("[2/4] 加载视觉塔 + processor ...")
    processor, model, device = load_model()

    build_shard(out_dir, items, pids, processor, model, device,
                shard=args.shard, nshards=args.nshards)


if __name__ == "__main__":
    main()
