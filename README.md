# vlm-sft —— Qwen3-VL 服装属性抽取 LoRA 微调

> 基于 Qwen3-VL-8B-Instruct，对 DeepFashion-MultiModal（4.4 万图、18 维官方标注属性）做 LoRA 微调，
> 实现「服装全身图 → 结构化属性 JSON」抽取，并在抽取结果之上落地全库相似款检索。
>
> 核心结果（test 集 4,410 张未见图）：属性级准确率 59.8% → 93.8%（+34.0pp），macro-F1 0.417 → 0.677，
> 全 JSON 匹配率 47.3%，18 个属性全部通过 McNemar 检验（p<0.001）。

📄 **入口**：[技术报告](docs/技术报告.md)（主文档）｜ [文档索引](docs/README.md) ｜ [脚本索引](scripts/README.md) ｜ [实验索引](experiments/README.md) ｜ [复现说明](docs/复现说明.md)

---

## 仓库结构

```
vlm/
├── data/                     # 数据集：原始数据 + 转换产物（软链，不入库）
├── models/                   # 模型：基座 + merged（软链，不入库）
├── docs/                     # 项目文档：主文档 + 专题说明 + 复现/面试文档 + 错误样本
├── experiments/              # 实验工作台：<基座>/<实验域>/（config/scripts/weights/predictions/reports）
├── scripts/                  # 通用脚本：数据转换 / 评测打分 / 检索 / 部署与 demo
├── logs/                     # 运行日志（不入库）
├── requirements_lf_full.txt  # lf 环境依赖（pip freeze 全量精确版）
├── vendor/                   # 第三方代码与资料（不入库）
└── scratch/                  # 早期实验与资料（不入库）
```

---

## 各目录

| 目录 | 内容 | 入口 |
|---|---|---|
| `data/` | 数据集：原始数据 + 转换产物（**软链**，git 忽略） | [数据说明](docs/数据说明.md) |
| `docs/` | 项目文档：主文档 + 5 份专题说明 + 复现 / 面试文档 | [文档索引](docs/README.md) |
| `experiments/` | 实验工作台：`<基座>/<实验域>/`，每个域一个 README | [实验索引](experiments/README.md) |
| `scripts/` | 通用脚本：数据转换 / 评测打分 / 检索 / 部署与 demo | [脚本索引](scripts/README.md) |
| `models/` | 基座模型（**软链**，git 忽略） | — |

### 软链清单（外部资产）

| 仓库内路径 | 目标（/disk16T1） | 说明 |
|---|---|---|
| `data/` | `scratch/maotai/data/vlm_data` | 数据集（原始+转换+检索索引） |
| `models/` | `scratch/maotai/models` | 基座与合并模型：`Qwen3-VL-8B-Instruct` / `Qwen3.8-27B` / `merged/` 等 |
| `experiments/qwen3vl8b/main/weights/merged/` | `scratch/maotai/models/merged` | 合并后完整模型（17G） |

**换机/迁移**：保留相对路径约定，只重建以上 3 条软链即可（`data/`、`models/`、`experiments/qwen3vl8b/main/weights/merged/`）。

---

> 环境（`lf` / `vllm` 两套 conda 环境）、模型放置与相对路径约定见[复现说明 §4](docs/复现说明.md)；
> 从零到部署的完整命令序列也在该文。
