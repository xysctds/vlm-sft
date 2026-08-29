#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DeepFashion 相似款检索（B 方案：属性 + 视觉向量）。

加载 ``build_retrieval_index.py`` 产出的索引，输入「属性 JSON + 查询图 embedding」，
输出 top-k 相似商品。支持两种加权模式：
  - vis       纯视觉余弦（w_vis=1）
  - hybrid    视觉 0.7 + 官方属性命中率 0.3（默认）

用法（库内自检 + 任意图）::

    python scripts/retrieval.py --mode vis            # 库内自检，应 top-1 命中自身
    python scripts/retrieval.py --mode hybrid --query <某张库内图> --json <属性json文件>

依赖：numpy；向量检索用归一化点积，不引入 faiss（7972×1152 一次全量毫秒级）。
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_convert_deepfashion_multi import SHAPE_ATTRS  # noqa: E402

DEFAULT_INDEX = "data/data_convert/deepfashion_multi/retrieval"

# 属性字段标准顺序（与 schema Prompt 一致：shape 在前，再按上/下/外衣的 fabric/pattern）。
ATTR_ORDER = [k for k, _ in SHAPE_ATTRS] + [f"{p}_{t}" for p in ("upper", "lower", "outer")
                                            for t in ("fabric", "pattern")]


def _ordered_attrs(attrs):
    """按 ATTR_ORDER 重排属性 dict 的键顺序，未知键追加到末尾。"""
    if not isinstance(attrs, dict):
        return attrs
    ordered = {k: attrs[k] for k in ATTR_ORDER if k in attrs}
    for k in attrs:
        if k not in ordered:
            ordered[k] = attrs[k]
    return ordered


class RetrievalIndex:
    def __init__(self, index_dir: str = DEFAULT_INDEX):
        self.meta = json.load(open(os.path.join(index_dir, "index_meta.json"), encoding="utf-8"))
        self.vecs = np.load(os.path.join(index_dir, "index_vectors.npy"))
        assert len(self.meta) == self.vecs.shape[0], "meta 与 vectors 数量不一致"
        self.ids = [m["id"] for m in self.meta]
        # 索引按 id 建的，保证同 id 排重
        self._n = self.vecs.shape[0]

    # ---- 属性命中率 ------------------------------------------------------
    def attr_score(self, query: dict) -> np.ndarray:
        """查询 JSON（可见键值）对每个商品的属性命中率 = 命中维数 / 查询维数。"""
        qitems = [(k, str(v)) for k, v in query.items()]
        if not qitems:
            return np.ones(self._n, dtype=np.float32)
        scores = np.zeros(self._n, dtype=np.float32)
        for i, m in enumerate(self.meta):
            cand = m["attrs"]
            hit = sum(1 for k, v in qitems if cand.get(k) == v)
            scores[i] = hit / len(qitems)
        return scores

    # ---- 检索 ------------------------------------------------------------
    def search(self, query_vec: np.ndarray, query_json: dict | None = None,
               mode: str = "hybrid", w_vis: float = 0.7, topk: int = 5,
               min_attr: float = 0.0, exclude_same_id: str | None = None):
        """返回 top-k 商品列表。

        query_vec   查询图向量（归一化后）
        query_json  模型抽出的属性 JSON（可为 None 则纯视觉）
        mode        vis / hybrid
        min_attr    属性硬过滤阈值（默认 0，即不过滤）
        exclude_same_id  排除某个商品 id（用于同款排重时排除查询自身）
        """
        vis = self.vecs @ (query_vec / (np.linalg.norm(query_vec) + 1e-9))
        if mode == "vis":
            scores = vis.copy()
        else:
            attr = self.attr_score(query_json or {})
            scores = w_vis * vis + (1 - w_vis) * attr

        order = np.argsort(-scores)
        results = []
        seen_ids = set()
        for idx in order:
            m = self.meta[idx]
            if exclude_same_id and m["id"] == exclude_same_id:
                continue
            if min_attr > 0 and self.attr_score(query_json or {})[idx] < min_attr:
                continue
            if m["id"] in seen_ids:  # 同款只留一个
                continue
            seen_ids.add(m["id"])
            results.append({
                "id": m["id"],
                "image": m["image"],
                "angle": m["angle"],
                "caption": m["caption"],
                "vis_cos": float(vis[idx]),
                "attr_score": float(self.attr_score(query_json or {})[idx]),
                "final": float(scores[idx]),
                "attrs": _ordered_attrs(m["attrs"]),
            })
            if len(results) >= topk:
                break
        return results

    def selfcheck(self) -> bool:
        """库内自检：index[0] 检索应 top-1 命中自身。"""
        res = self.search(self.vecs[0], mode="vis", topk=1)
        return res and res[0]["id"] == self.ids[0]


_query_model = None  # (vision, processor, device) 单例缓存，避免每次检索重复加载视觉塔


def load_vision_tower(model_path: str):
    """只加载 Qwen3-VL 视觉塔（约 576M 参数 / 1.2GB bf16），跳过 8B 语言模型。

    比 from_pretrained(整个模型, 17GB) 快一个数量级：单次检索首次加载
    从 ~48s 降到 ~2-4s。返回 (vision_model, processor, device)。
    视觉塔权重取 checkpoint 中 model.visual.*（位于最后一个 safetensors shard）。
    """
    import os
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig, AutoProcessor
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    processor.image_processor.min_pixels = 1024
    processor.image_processor.max_pixels = 825750

    vision = Qwen3VLVisionModel(cfg.vision_config)
    state_dict = {}
    shards = sorted(f for f in os.listdir(model_path) if f.endswith(".safetensors"))
    for fn in shards:
        with safe_open(os.path.join(model_path, fn), framework="pt") as f:
            for k in f.keys():
                if k.startswith("model.visual."):
                    state_dict[k.replace("model.visual.", "")] = f.get_tensor(k)
    missing, unexpected = vision.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"视觉塔权重加载不完整: missing={missing} unexpected={unexpected}")
    vision.to(device).eval()
    return vision, processor, device


def preload_vision_tower(model_path: str = "models/Qwen3-VL-8B-Instruct"):
    """预热视觉塔（启动后台调用，避免用户第一次点击检索时等待模型加载）。"""
    global _query_model
    if _query_model is None:
        _query_model = load_vision_tower(model_path)
    return _query_model


def embed_query(model_path: str, image_path: str) -> np.ndarray:
    """对任意查询图抽归一化视觉向量（只加载视觉塔，口径与索引构建一致）。

    关键优化：只加载 Qwen3-VL 视觉塔（576M / 1.2GB）而非整个 8B 模型（17GB），
    首次加载从几十秒降到数秒；模型缓存在模块级单例，之后每次仅做 ~0.2s 推理。

    建议通过 CUDA_VISIBLE_DEVICES 指定查询模型所在卡（如与 vLLM 服务分卡），
    避免不同进程挤在同一 GPU 导致 OOM。
    """
    from PIL import Image
    import torch

    global _query_model
    if _query_model is None:
        _query_model = load_vision_tower(model_path)

    vision, processor, device = _query_model
    img = Image.open(image_path).convert("RGB")
    # text='' 空占位：兼容 transformers 5.8（只传 images 会因 image_token in None 崩）
    out = processor(images=[img], text="", return_tensors="pt")
    pv = out["pixel_values"].to(device)
    grid_thw = out["image_grid_thw"].to(device)
    with torch.no_grad():
        feats = vision(pv, grid_thw=grid_thw)  # 视觉塔直接前向
        vec = feats.last_hidden_state.mean(dim=0).detach().cpu().to(torch.float32).numpy()
    vec /= np.linalg.norm(vec) + 1e-9
    return vec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", default=DEFAULT_INDEX)
    ap.add_argument("--mode", choices=["vis", "hybrid"], default="hybrid")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--min-attr", type=float, default=0.0)
    ap.add_argument("--query", default=None, help="查询图（缺省则库内自检）")
    ap.add_argument("--json", default=None, help="查询属性 JSON 文件（可选）")
    ap.add_argument("--model", default="models/Qwen3-VL-8B-Instruct")
    args = ap.parse_args()

    idx = RetrievalIndex(args.index)
    if args.query is None:
        ok = idx.selfcheck()
        print(f"库内自检 top-1 命中自身：{'OK' if ok else 'FAIL'}")
        return

    qv = embed_query(args.model, args.query)
    qj = json.load(open(args.json, encoding="utf-8")) if args.json else None
    print(f"查询图: {args.query}")
    print(f"属性 JSON: {qj}")
    print(f"模式: {args.mode}  topk={args.topk}")
    for i, r in enumerate(idx.search(qv, qj, mode=args.mode, topk=args.topk,
                                    min_attr=args.min_attr)):
        print(f"\n[{i+1}] id={r['id']}  {r['image']}  (vis={r['vis_cos']:.3f} "
              f"attr={r['attr_score']:.2f} final={r['final']:.3f})")
        if r["attrs"]:
            print("    attr:", json.dumps(r["attrs"], ensure_ascii=False))
        if r["caption"]:
            print("    cap :", r["caption"])


if __name__ == "__main__":
    main()
