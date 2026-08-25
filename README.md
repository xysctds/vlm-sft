# vlm-sft —— Qwen3-VL 服装属性抽取 LoRA 微调

> **主线**：基于 Qwen3-VL-8B-Instruct，对 DeepFashion-MultiModal（4.4 万图、18 维官方标注属性）做 LoRA 微调，实现「服装全身图 → 结构化属性 JSON」抽取（schema 感知指令 + 自建评测体系）。

**核心成果**（test 集 4,410 张未见图，基座 0-shot vs LoRA 微调）：

| 指标 | 基座 | LoRA | 提升 |
|---|---|---|---|
| 属性级准确率 | 59.8% | **93.8%** | **+34.0pp** |
| macro-F1 | 0.417 | **0.677** | +0.260 |
| 全 JSON 精确匹配率 | 0.0% | **47.3%** | +47.3pp |
| JSON 可解析率 | 100% | **100%** | — |

- 18 个属性全部通过 McNemar 显著性检验（p<0.001）；
- 消融证明 **schema 感知指令是决定性方法**（同数据下比朴素指令 +31.5pp）、训练无过拟合、推理对分辨率鲁棒；

📄 **文档**：[项目策划案](docs/DeepFashion微调项目方案.md) ｜ [技术报告](docs/DeepFashion微调技术报告.md) ｜ [数据说明](docs/数据说明.md) ｜ [调参说明](docs/调参说明.md)

---

## 仓库结构

```
vlm/
├── data/ -> /disk16T1/scratch/maotai/data/vlm_data   # 数据集（软链）
├── docs/                   # 方案策划案 + 技术报告 + 数据/调参说明
├── llamafactory/           # 工作文件夹（训练配置、评测产物、模型权重）
├── requirements_lf_full.txt # lf 环境依赖（pip freeze 全量精确版，可直接复现）
└── scripts/                # 数据转换 / 评测打分 / 报告生成 / 部署脚本
```

---

## 各文件夹说明

### data/
数据集目录，**软链**到 `/disk16T1/scratch/maotai/data/vlm_data`，**已被 git 忽略**。内部按「原始数据 / 转换后数据」组织，详细说明见 docs/ 下的《数据说明》文档。

### docs/
项目文档（两件套 + 说明文档）：

- `DeepFashion微调项目方案.md`：**主线项目策划案**（可单独拿出去投递/展示）——完整给出数据、任务、方法、评测、消融、部署的规划与设计依据。
- `DeepFashion微调技术报告.md`：**主线成果技术报告**——实际执行结果、实验数据与论证链（效果/归因/边界/过程四环），与策划案成对。
- `DeepFashion属性对照表.md`：18 维属性体系与取值对照（数据口径速查）。
- `数据说明.md`：数据转换与 LLaMA-Factory 接入说明。
- `调参说明.md`：训练超参与调参经验记录。

### llamafactory/（工作文件夹）
日常训练工作都在这：训练配置、模型权重、评测产物。

- `config/`：训练/评测/导出配置——`qwen3vl8b_dfattr_train*.yaml`（DeepFashion 主线训练）、`qwen3vl8b_dfattr_eval_val_*.yaml`（val 评测，含 few-shot / simple-prompt / 压缩分辨率变体）、`qwen3vl8b_dfattr_eval_test_*.yaml`（test 最终评测）、`qwen3vl8b_dfattr_export_cli.yaml`（导出合并）；子目录 `ablation/`（指令范式消融）、`ckpt_screen/`（checkpoint 快筛，配置由脚本生成）。
- `saves/`：模型权重与训练/调度日志（**git 忽略**）——主线 `dfattr/`（LoRA + merged）、早期实验 `v1/ v2/`、消融 `ablation/`、checkpoint 快筛 `ckpt_screen/`。
- `eval_outputs/`：评测生成预测（`generated_predictions.jsonl` 等），按实验分目录（`val_base/`、`val_lora/`、`test_*`、`ablate_*`、`ckpt_screen/`、few-shot/simple-prompt/压缩分辨率变体）。
- `results/`：评测汇总产物（`preds_*.jsonl`、`report_*.md` 评测报告、`data_stats.json`）与评测日志（`eval_*.log`）；`report_*.md` 由 `scripts/summarize_*.py` 生成，`report_val.md`/`report_test.md` 由 `eval_attr.py stats` 生成。

### scripts/
数据转换 / 评测打分 / 报告生成 / 部署脚本：

- `data_convert_deepfashion_multi.py`：DeepFashion-MultiModal → LLaMA-Factory 训练格式（主线 schema 版 + simple 对照 + 描述 + 属性 QA）。
- `data_convert_fewshot_val.py`：生成 few-shot 评测数据（schema 指令 + 2 个文本示例）。
- `check_data.py`：DeepFashion 数据质量四关卡（结构/答案/防泄漏/分布）。
- `eval_attr.py`：评测打分 `stats`（属性级 acc / macro-F1 / 分层 / McNemar，容错解析）；预测生成走 llamafactory 管线。
- `convert_preds.py`：把 llamafactory 预测输出（`generated_predictions.jsonl`）转为 `eval_attr.py stats` 可消费格式。
- `summarize_ablation.py` / `summarize_category.py` / `summarize_ckpts.py` / `summarize_confirm.py` / `summarize_resolution.py` / `summarize_supplement.py`：报告生成（分别对应消融、品类分层、checkpoint 快筛/确认、分辨率、补充实验报告）。
- `serve_demo.py`：DeepFashion 部署 demo 客户端（vLLM OpenAI 兼容）。

### 根目录文件
- `requirements_lf_full.txt`：lf 环境全量依赖（pip freeze 精确版本，`pip install -r` 可直接复现环境）。

---

## 环境

项目用 conda 管理两套隔离环境，对应根目录两个依赖文件：

| 环境名 | 用途 | 依赖文件 |
|---|---|---|
| `lf` | LLaMA-Factory 微调（主线） | `requirements_lf_full.txt`（pip freeze 全量精确版） |

### 模型与路径约定

仓库内路径均为**相对路径**（在仓库根运行即可），不含任何本机绝对路径：

| 项 | 约定 |
|---|---|
| 仓库内数据/产物路径 | 相对仓库根（`data/`、`llamafactory/eval_outputs/` 等），直接可用 |
| 基座模型 | 放到 `models/Qwen3-VL-8B-Instruct`（全部训练/评测/导出配置中 `model_name_or_path` 均写此相对路径，在仓库根运行即按本地目录加载） |

> **注意**：本仓库 llamafactory 的 CLI 用 `OmegaConf.to_container` 读配置，**不解析 `${oc.env:...}` 插值**（OmegaConf 2.3.x 的 `to_container` 默认不 resolve 插值，`${oc.env:LLM_MODEL_PATH, ...}` 会原样传给 HF 导致启动失败），因此配置中**直接写相对路径**、不要用插值写法；换机/发布时把模型软链到仓库 `models/Qwen3-VL-8B-Instruct` 即可（`grep -rn "model_name_or_path" llamafactory/config` 可找到全部位置）。
