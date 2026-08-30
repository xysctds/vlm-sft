#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统一基准：只加载视觉塔 + 预热的 效果对比（一个脚本测三档，输出三条"一次点击"时间线）。

背景：此前三套脚本各测各口径，数字对不齐——`bench_model_load` 只测加载、`bench_cold_request`
只测 embedding、`bench_end_to_end` 测"冷启动→抽→检索"却混淆加载与推理。本脚本把三档统一成
**同一个动作、同一个单位（秒）：用户第一次点击到出结果**，一次运行给出可对比的三条时间线：

   基准A（原本）   : 加载完整模型 8.8B 抽视觉 + 属性抽取(vLLM) + 检索
   基准B（视觉塔冷）: 冷加载视觉塔 576M 抽视觉 + 属性抽取(vLLM) + 检索
   优化后（预热）  : 视觉塔已预热，点击只做 推理 + 属性抽取 + 检索（不含加载）

每档用**独立子进程**测 N 次取均值，避免 OS cache 美化冷值。属性抽取统一走真实 vLLM 服务（与
demo 一致），因此需要 vLLM 已在 8000 端口就绪。

注：为聚焦"点击链"，本脚本的视觉 retrieval 用真实索引（单例）；基准A 的视觉用完整模型
`model.model.visual`，基准B/优化后用视觉塔。输出并给出"加载那刀 / 预热那刀"各自省了多少。

用法（仓库根，跑在 **vllm** 环境；vLLM 服务请先 `bash scripts/start_web.sh --vllm-only`）::

    CUDA_VISIBLE_DEVICES=2 python scripts/bench_schemes.py <查询图>          # 三档各测 3 次
    N=5 CUDA_VISIBLE_DEVICES=2 python scripts/bench_schemes.py <查询图>
结果写 `logs/bench_schemes.log`。
"""

import os
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import retrieval as re  # noqa: E402
import serve_gradio as sg  # noqa: E402

MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "models", "Qwen3-VL-8B-Instruct")
LOG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "logs", "bench_schemes.log")

_idx_cache = None  # 视觉检索索引单例（与真实服务一致，只建一次）

# worker 子进程（--worker）只算耗时、用 stdout 回传结果，**不写日志**；
# 日志文件只由 driver 打开并写入（避免多进程抢占 "w" 截断导致乱码/空文件）。
_log = None


def _emit(*args):
    line = " ".join(str(a) for a in args)
    print(line)
    global _log
    if _log is None:
        os.makedirs(os.path.dirname(LOG), exist_ok=True)
        _log = open(LOG, "a", encoding="utf-8")
    _log.write(line + "\n")
    _log.flush()


def _retrieve(qv: np.ndarray, attrs: dict) -> None:
    """真实检索（属性加权 + 按套聚合，top-5），索引单例缓存。"""
    global _idx_cache
    if _idx_cache is None:
        _idx_cache = re.RetrievalIndex("data/data_convert/deepfashion_multi/retrieval")
    _idx_cache.search(qv, attrs, mode="hybrid", topk=5, group=True)


def _embed_full_model(path: str) -> np.ndarray:
    """基准A：用完整模型(8.8B) model.model.visual 抽 query embedding（含加载成本）。"""
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, trust_remote_code=True, dtype=torch.bfloat16)
    model.to("cuda").eval()
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    processor.image_processor.min_pixels = 1024
    processor.image_processor.max_pixels = 825750
    img = Image.open(path).convert("RGB")
    out = processor(images=[img], text="", return_tensors="pt")
    pv = out["pixel_values"].to("cuda")
    grid_thw = out["image_grid_thw"].to("cuda")
    with torch.no_grad():
        feats = model.model.visual(pv, grid_thw=grid_thw)
        vec = feats.last_hidden_state.mean(dim=0).detach().cpu().to(torch.float32).numpy()
    vec /= np.linalg.norm(vec) + 1e-9
    return vec


def _measure(scheme: str, path: str) -> dict:
    """单次测量某方案，返回各段耗时 dict。

    三档的统一动作 = 进程冷启动 →（视觉模型就绪）→ 抽一次 query embedding(视觉)
                        → 属性抽取(vLLM) → 检索。差别只在"视觉模型"如何就绪：
        - base_a   ：完整模型冷加载，加载计入视觉段；
        - base_b   ：视觉塔冷加载，加载计入视觉段；
        - optimized：视觉塔已预热（预加载不计时），点击仅推理。
    属性抽取三档都走真实 vLLM；检索统一用单例索引。
    """
    times = {}

    if scheme == "base_a":
        t0 = time.perf_counter()
        qv = _embed_full_model(path)          # 加载整模型 + 抽视觉（加载计入）
        times["vis"] = time.perf_counter() - t0
    elif scheme == "base_b":
        t0 = time.perf_counter()
        qv = re.embed_query(MODEL, path)      # 冷加载视觉塔 + 抽视觉（加载计入）
        times["vis"] = time.perf_counter() - t0
    elif scheme == "optimized":
        re.preload_vision_tower(MODEL)        # 预热（加载成本不计入本次点击）
        t0 = time.perf_counter()
        qv = re.embed_query(MODEL, path)      # 仅推理 + 抽视觉
        times["vis"] = time.perf_counter() - t0
    else:
        raise ValueError(f"未知方案: {scheme}")

    t1 = time.perf_counter()
    _, attrs = sg.predict(path, sg.build_schema_prompt())   # 属性抽取（真实 vLLM）
    times["attr"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    _retrieve(qv, attrs)                     # 检索
    times["search"] = time.perf_counter() - t2
    torch.cuda.synchronize()

    times["total"] = times["vis"] + times["attr"] + times["search"]
    return times


def main() -> None:
    if "--worker" in sys.argv:
        scheme = sys.argv[sys.argv.index("--worker") + 1]
        t = _measure(scheme, os.environ['QIMG'])
        print(f"{t['vis']:.3f} {t['attr']:.3f} {t['search']:.3f} {t['total']:.3f}")
        return

    # driver：先清空并只在此分支打开日志（worker 不写、不截断）
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    global _log
    _log = open(LOG, "w", encoding="utf-8")

    path = os.path.abspath(sys.argv[1])
    os.environ["QIMG"] = path
    n = int(os.environ.get("N", "3"))
    schemes = [("base_a", "基准A 原本(整模型)"), ("base_b", "基准B 视觉塔·冷"),
               ("optimized", "优化后(视觉塔·预热)")]

    _emit(f"==== 统一基准：一次点击出结果（各 {n} 次独立进程，查询图={os.path.basename(path)}） ====")
    results = {}
    for key, label in schemes:
        rows = []
        _emit(f"— {label} —")
        for i in range(n):
            out = subprocess.check_output(
                [sys.executable, os.path.abspath(__file__), "--worker", key],
                stderr=subprocess.DEVNULL).decode().strip().split()
            v, a, s, t = map(float, out)
            rows.append({"vis": v, "attr": a, "search": s, "total": t})
            _emit(f"  第{i+1}次: 视觉={v:5.2f}s 属性={a:5.2f}s 检索={s:5.2f}s → 总={t:5.2f}s")
        results[key] = rows
        mv = statistics.mean(r["vis"] for r in rows)
        ma = statistics.mean(r["attr"] for r in rows)
        ms = statistics.mean(r["search"] for r in rows)
        mt = statistics.mean(r["total"] for r in rows)
        _emit(f"  均值: 视觉={mv:5.2f}s 属性={ma:5.2f}s 检索={ms:5.2f}s → 总={mt:5.2f}s")

    _emit("=" * 68)
    fa = statistics.mean(r["total"] for r in results["base_a"])
    fb = statistics.mean(r["total"] for r in results["base_b"])
    fo = statistics.mean(r["total"] for r in results["optimized"])
    _emit(f"基准A→优化后 : {fa:.2f}s → {fo:.2f}s   {fa / fo:.1f}x")
    _emit(f"基准B→优化后 : {fb:.2f}s → {fo:.2f}s   {fb / fo:.1f}x")
    _emit(f"基准A→基准B : {fa:.2f}s → {fb:.2f}s   {fa / fb:.1f}x  (这一刀=只加载视觉塔)")
    _emit(f"[log] 已保存: {LOG}")


if __name__ == "__main__":
    main()
