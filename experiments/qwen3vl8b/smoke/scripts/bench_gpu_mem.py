#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU 显存占用采样脚本：反复读取 nvidia-smi 的各卡显存，写入 CSV。

用途：配合真实训练/推理测量 GPU 显存占用曲线（峰值）。
  - 训练显存请用 `llamafactory-cli train` + 本采样器并行跑，得到每卡真实峰值。
  - 之前用脚本自测训练显存会虚高（单进程无法复现 flash_attn/3卡DDP/官方封装），
    因此显存以「真实任务 + nvidia-smi 采样」为准。

用法（仓库根，无需 conda 或任一本机 python 即可）::
    1) 先启动本采样器(后台):
       python experiments/qwen3vl8b/smoke/scripts/bench_gpu_mem.py --interval 0.3 --out experiments/qwen3vl8b/smoke/gpu_mem_sample.csv
       (或用 `nohup ... &` 后台跑)
    2) 另开终端跑真实任务(如 3 卡 DDP 训练):
       llamafactory-cli train experiments/qwen3vl8b/smoke/train_3gpu_verify.yaml
    3) 任务结束后, 停止采样器(Ctrl+C / kill), 读 CSV 峰值:
       python experiments/qwen3vl8b/smoke/scripts/bench_gpu_mem.py --report --out experiments/qwen3vl8b/smoke/gpu_mem_sample.csv

输出 CSV 每行: `<gpu_index>, <memory_used_MiB>`（逐帧，多卡循环），
report 模式打印每卡峰值与时间覆盖。
"""

import argparse
import csv
import os
import subprocess
import sys
import time


def _sample_once() -> list[tuple[int, int]]:
    """返回一次采样: [(gpu_index, memory_used_MiB), ...]（全部卡）。"""
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used",
         "--format=csv,noheader,nounits"], text=True)
    rows = []
    for line in out.strip().splitlines():
        if not line:
            continue
        idx, mem = line.split(",")
        rows.append((int(idx.strip()), int(mem.strip())))
    return rows


def _record(out_path: str, interval: float, gpus: set[int] | None):
    """持续采样写入 CSV（等价于 while 循环 nvidia-smi）。"""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    started = time.time()
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        t0 = None
        while True:
            rows = _sample_once()
            if gpus:
                rows = [r for r in rows if r[0] in gpus]
            if t0 is None:
                t0 = time.time()
                w.writerow(["t_sec"] + [f"gpu{g}" for g, _ in rows])
            w.writerow([round(time.time() - t0, 2)] + [m for _, m in rows])
            f.flush()
            # 每 ~100 秒在 stderr 提示一次
            if int(time.time() - started) % 100 < interval:
                sys.stderr.write(f"[sampler] t={time.time()-started:.0f}s 仍在采样...\n")
                sys.stderr.flush()
            time.sleep(interval)


def _report(out_path: str):
    """读取 CSV，打印每卡峰值（最大 used）与时间覆盖。"""
    if not os.path.exists(out_path):
        print(f"[report] 文件不存在: {out_path}")
        return
    with open(out_path, newline="", encoding="utf-8") as f:
        rd = csv.reader(f)
        header = next(rd)
        cols = header[1:]  # 每卡列
        totals = {c: 0 for c in cols}
        n = 0
        last_t = 0.0
        for row in rd:
            if not row:
                continue
            try:
                last_t = float(row[0])
            except ValueError:
                continue
            for c, v in zip(cols, row[1:]):
                try:
                    totals[c] = max(totals[c], int(v))
                except ValueError:
                    pass
            n += 1
    print(f"[report] 采样点数(x 行): {n}")
    print(f"[report] 时间覆盖: {last_t:.1f}s")
    for c, peak in totals.items():
        print(f"[report] {c} 峰值 = {peak} MiB ≈ {peak/1024:.1f} GB")
    print(f"[report] 若为 3 卡 DDP 训练, 峰值即每卡显存占用。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=0.3, help="采样间隔秒")
    ap.add_argument("--out", default="experiments/qwen3vl8b/smoke/gpu_mem_sample.csv",
                    help="输出 CSV 路径")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--record", action="store_true", help="持续采样(默认)")
    grp.add_argument("--report", action="store_true", help="只读 CSV 打印峰值")
    grp.add_argument("--gpus", type=str, default=None,
                     help="只采样哪些卡，如 0,1,2(默认全部)")
    args = ap.parse_args()

    if args.report:
        _report(args.out)
        return
    gpus = {int(x) for x in args.gpus.split(",")} if args.gpus else None
    print(f"[sampler] 开始采样 → {args.out} (间隔 {args.interval}s, "
          f"卡={gpus if gpus else '全部'})；Ctrl+C 停止")
    try:
        _record(args.out, args.interval, gpus)
    except KeyboardInterrupt:
        print("\n[sampler] 已停止；运行 --report 查看峰值。")


if __name__ == "__main__":
    main()
