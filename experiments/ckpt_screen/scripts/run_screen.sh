#!/bin/bash
# checkpoint 快筛调度：3 卡轮转（每卡 5 个 checkpoint 串行）
cd "$(dirname "$0")/../../.."
mkdir -p experiments/ckpt_screen/predictions/logs
LF=/home/maotai/.conda/envs/lf/bin/llamafactory-cli

run_queue() {
  local gpu=$1; shift
  for ckpt in "$@"; do
    echo "[GPU$gpu] start checkpoint-$ckpt $(date +%H:%M:%S)" >> experiments/ckpt_screen/predictions/logs/schedule.log
    OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=$gpu $LF train experiments/ckpt_screen/config/ckpt_${ckpt}.yaml \
      > experiments/ckpt_screen/predictions/logs/ckpt_${ckpt}.log 2>&1
    echo "[GPU$gpu] done  checkpoint-$ckpt $(date +%H:%M:%S)" >> experiments/ckpt_screen/predictions/logs/schedule.log
  done
}

run_queue 0 100 300 500 700 900 &
run_queue 1 1100 1300 1500 1700 1900 &
run_queue 2 2100 2300 2500 2700 2793 &
wait
echo "ALL DONE $(date +%H:%M:%S)" >> experiments/ckpt_screen/predictions/logs/schedule.log
