# smoke：训练可行性与基准验证

本域回答三个工程问题：训练能不能跑通、显存够不够、一条样本有多长。产出的 token 规模与显存峰值
被技术报告 §4.3 与 §6.6 引用为工程数据，但不构成实验结论——这里没有唯一变量，也不参与效果 / 归因 / 边界 / 过程的论证。

## 做什么

| 内容 | 说明 |
|---|---|
| 训练可行性验证 | `train_3gpu_verify.yaml`：3 卡 DDP 训练验证，参数与 `../main` 的训练配置等价 |
| 样本长度基准 | `scripts/bench_tokens_mem.py`：纯 CPU，把一条训练样本拆成 token 段（视觉 token / 指令 / JSON），用于估算序列长度与显存 |
| 显存基准 | `scripts/bench_gpu_mem.py`：反复采样 `nvidia-smi` 的各卡显存并写入 CSV，用于验证 batch 大小是否安全 |

## 目录

| 路径 | 内容 | 入库 |
|---|---|---|
| `train_3gpu_verify.yaml` | 3 卡训练验证配置 | 是 |
| `scripts/bench_tokens_mem.py`、`scripts/bench_gpu_mem.py` | 两个基准脚本 | 是 |
| `weights/dfattr_lora_verify_30g/` | 验证训练产出的 adapter | 否 |
| `bench_tokens_mem.log`、`gpu_mem_sample.csv` | 两个基准脚本的输出 | 否 |

## 怎么跑

```bash
# 纯 CPU，不加载权重
python experiments/qwen3vl8b/smoke/scripts/bench_tokens_mem.py
python experiments/qwen3vl8b/smoke/scripts/bench_gpu_mem.py

# 3 卡训练验证
llamafactory-cli train experiments/qwen3vl8b/smoke/train_3gpu_verify.yaml
```

## 指标数值以哪份为准

本目录不产生指标。主实验的指标以 `../main/reports/` 下脚本生成的报告为准。
