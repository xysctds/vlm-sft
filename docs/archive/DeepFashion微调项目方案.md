# DeepFashion 服装属性提取与相似款检索项目方案

> **文档性质**：本方案为**独立成文的项目技术方案书**，内容自包含（数据、代码、配置、命令均全文给出，无需依赖其他文档），可直接用于岗位投递附件、面试前置材料或作品集展示。
>
> **一句话项目定位**：基于 Qwen3-VL-8B-Instruct，用 LLaMA-Factory 对 DeepFashion-MultiModal 时装数据集（SIGGRAPH 2022，4.4 万图、18 维官方标注属性）做 LoRA 微调，实现「服装全身图 → 结构化属性 JSON」抽取，配套自建评测体系（属性级指标 + 分层统计 + McNemar 显著性 + 容错解析）与 vLLM 部署 demo；并在抽取结果之上落地**相似款检索推荐**（属性 + 视觉混合加权，全库 top-k 相似款），升级为「电商标注 + 相似款推荐」完整闭环，对标电商货架属性 / 商品知识图谱 / 相似款推荐场景。

---

## 目录

1. [项目摘要](#1-项目摘要)
2. [背景与动机](#2-背景与动机)
3. [技术方案总览](#3-技术方案总览)
4. [数据与任务](#4-数据与任务)
5. [训练方案](#5-训练方案)
6. [评测方案](#6-评测方案)
7. [badcase 分析与改进闭环](#7-badcase-分析与改进闭环)
8. [相似款检索推荐](#8-相似款检索推荐)
9. [部署与 Demo](#9-部署与-demo)
10. [步骤规划](#10-步骤规划)
11. [配套实现与仓库结构](#11-配套实现与仓库结构)
12. [附录：简历表述与面试准备](#12-附录简历表述与面试准备)
13. [附录：关键命令速查](#13-附录关键命令速查)

---

## 1. 项目摘要

| 项 | 内容 |
|---|---|
| 任务 | 服装全身图 → 结构化属性 JSON（袖长/领型/面料/图案等 18 维属性）+ 全库相似款检索推荐 |
| 模型 | Qwen3-VL-8B-Instruct（约 8.5B，单卡 48G） |
| 框架 | LLaMA-Factory（CLI），LoRA 微调（可训练参数 <1%；秩/α 等超参以训练配置为准） |
| 数据 | DeepFashion-MultiModal（Text2Human, SIGGRAPH 2022）：44,096 图，官方人工标注 |
| 数据构造 | **schema 感知指令**（枚举全部合法键与取值，工业界结构化输出标准做法）+ 朴素指令对照 |
| 评测 | 自建：属性级准确率 / macro-F1 / 混淆矩阵 / 分层统计 / McNemar 显著性 / JSON 容错解析 |
| 检索 | 属性约束 + 视觉 embedding 混合加权，检索库 7972 件 top-k 相似款，Gradio 一键演示 |
| 部署 | LLaMA-Factory 导出合并 + vLLM OpenAI 兼容接口 + CLI/Web demo |
| 硬件/周期 | 3×48GB 4090 服务器；14 天（每天 2~4 小时） |
| 面向岗位 | 电商算法（商品属性/货架/知识图谱/相似款推荐）、多模态内容理解、大模型应用算法实习 |

**三个核心卖点**：

1. **业务贴合**——「结构化属性抽取 + 相似款检索推荐」覆盖电商核心链路（货架属性、搜索召回、商品知识图谱、相似款推荐），价值一句话讲清；
2. **区分度**——ChartQA 微调是烂大街模板；"DeepFashion + VLM 微调 + 结构化属性抽取 + 相似款检索"完整闭环几乎无人做；
3. **严谨评测**——数据集无官方指标 → 自建完整评测体系，反而展示方法论能力（对比组纪律、分层、显著性检验）。

**文档定位**：本文件为项目**策划案**——完整给出数据、任务、方法、评测、消融、检索、部署的规划与设计依据，可作为岗位投递附件/面试前置材料。

---

## 2. 背景与动机

电商平台商品属性（袖长、面料、图案、领型等）是货架展示、搜索召回、商品知识图谱的基础数据，目前大量依赖人工标注，存在成本高、标准不一、长尾商品覆盖差的问题。通用多模态大模型（VLM）虽具备基础视觉理解能力，但对服装的**细粒度属性**（如 medium-sleeve vs long-sleeve、cotton vs knitted）输出不可控：格式不稳定、属性张冠李戴、幻觉图上不存在的属性。本项目用领域数据对 VLM 做轻量 LoRA 微调，把"通用理解"变成"业务可用的结构化抽取能力"，并进一步把抽取结果用于**相似款检索**，形成「标注 + 推荐」的完整电商闭环。

---

## 3. 技术方案总览

```
用户上传服装全身图
        │
        ▼
┌─ 结构化属性抽取（主线）─────────────────────────────┐
│  数据：DeepFashion-MultiModal 4.4 万图 / 18 维属性     │
│  方法：Qwen3-VL-8B + LoRA 微调（冻结视觉塔，schema 指令）│
│  评测：自建指标 + McNemar + 消融（93.8% acc）         │
└───────────────┬──────────────────────────────────┘
                │  属性 JSON + 视觉 embedding
                ▼
┌─ 相似款检索推荐（应用闭环）───────────────────────────┐
│  属性约束过滤 + 向量余弦混合加权 → 全库 top-k 相似款     │
│  Gradio Gallery 展示（图上标匹配分）                  │
└───────────────┬──────────────────────────────────┘
                ▼
        vLLM 部署 demo（CLI / Web）
```

技术主线为**结构化属性抽取**（数据 → 微调 → 评测，证据链完整）；检索推荐是抽取结果之上的**应用出口**，把项目串成「电商标注 + 相似款推荐」的完整电商场景。

---

## 4. 数据与任务

### 4.1 为什么用 DeepFashion-MultiModal

| 维度 | ChartQA（备选） | DeepFashion-MultiModal（选定） |
|---|---|---|
| 区分度 | 微调模板烂大街 | 用的人极少，组合几乎空白 |
| 业务匹配 | BI/报表问答，岗位少 | 电商/内容平台岗位多，价值秒懂 |
| 微调收益 | base 模型已不差，空间有限 | 细粒度属性 base 做不好，容易出大提升 |
| 评测基准 | 官方 ground truth | 无官方指标 → 自建体系（展示方法论） |
| 数据形态 | 合成图表 | 真实人像/衣物图，demo 冲击力强 |
| 成本 | 子集小、快 | 4.4 万图，8B 单卡可控（`max_samples` 限流） |

**主线数据 = DeepFashion-MultiModal（attr_json 任务）**；ChartQA 仅作 pipeline 练手（两套数据脚本同构，切换成本低）。风险对策：数据为 2022 年 → 叙事强调"任务不变、方法新"（用 Qwen3-VL 刷新旧 benchmark 基线）；标签含 NA/hidden 噪声 → 转换时剔除不可见属性，评测与训练口径一致。

### 4.2 数据集与属性体系

- **数据集**：DeepFashion-MultiModal（[yumingj/DeepFashion-MultiModal](https://github.com/yumingj/DeepFashion-MultiModal)，Text2Human，SIGGRAPH 2022），官方人工标注，无版权风险，可公开复现。
- **规模**：44,096 张 750×1101 全身人像图；`captions.json` 42,544 条描述；labels 含 12 个形状属性 + 面料/图案标注。
- **原始结构**：`images.zip` / `labels.zip` / `captions.json` 即可（segm/densepose/keypoints 与 VLM 微调无关，可不下）。

**属性体系（18 维 = 12 形状 + 3 部位×面料 + 3 部位×图案）**：

| 分组 | 属性键 | 取值（官方口径，NA 已剔除） |
|---|---|---|
| 形状 | `sleeve_length` | sleeveless / short-sleeve / medium-sleeve / long-sleeve |
| | `lower_clothing_length` | three-point / medium short / three-quarter / long |
| | `neckline` | V-shape / square / round / standing / lapel / suspenders |
| | `hat` / `socks` / `glasses` | no / yes（或 2~4 类） |
| | `neckwear` / `wrist_wearing` / `ring` | no / yes |
| | `waist_accessories` | no / belt / clothing at waist |
| | `outer_clothing_is_cardigan` / `upper_clothing_covering_navel` | yes / no |
| 面料 | `{upper,lower,outer}_fabric` | denim / cotton / leather / furry / knitted / chiffon / other |
| 图案 | `{upper,lower,outer}_pattern` | floral / graphic / striped / pure color / lattice / other / color block |

要点：**不可见（NA）与 `hidden` 属性在转换时剔除**，训练/评测只针对可见属性；`outer_*` 仅在图中存在外衣时出现（每图实际输出 10~16 个键）。完整英中对照见 `docs/数据说明.md` 附录。

### 4.3 任务形态与数据格式

**任务形态**：

| 数据集（dataset_info.json 注册名） | train / val / test | 输入 → 输出 | 定位 |
|---|---|---|---|
| `deepfashion_attr_json_*` | 35,276 / 4,410 / 4,410 | 图 + **schema 枚举指令** → 完整属性 JSON | **主线任务** |
| `deepfashion_attr_json_simple_*` | 35,276 / 4,410 / 4,410 | 图 + 朴素一句话指令 → 完整属性 JSON | 对照实验 |
| `deepfashion_caption_*` | 34,006 / 4,258 / 4,273 | 图 → 自然语言描述 | 辅助（多任务组合暂不纳入，见 5.3） |
| `deepfashion_attr_qa_*` | 451,744 / 56,860 / 56,943 | 逐属性单问 → 单值 | 弃用（模板重复易过拟合） |

**为什么 attr_json 做主任务**：一次推理输出完整结构化属性，贴合真实电商抽取场景；输出为 JSON，与"图像 → 结构化输出"主线一致；评测可按属性键解构，严谨且细粒度。

**schema 感知指令**（行业标准结构化输出的数据侧做法；`build_schema_prompt()` 由属性表自动生成，训练/评测共用）：

```
Extract the clothing attributes visible in this image and output a JSON object.

Allowed keys and values:
- sleeve_length: sleeveless, short-sleeve, medium-sleeve, long-sleeve
- ...（全部 18 个键的取值枚举）...

Rules:
- Include only keys whose attribute is clearly visible in the image.
- Use only the allowed values listed above.
- Output the JSON object only, without any extra text or explanation.
```

朴素指令对照版（simple）：`Extract the clothing attributes visible in this image and output them as JSON.`

**训练样本示例**（sharegpt 格式，LLaMA-Factory 直接消费）：

```json
{"messages": [{"role": "user", "content": "<image>Extract the clothing attributes visible in this image and output a JSON object.\n\nAllowed keys and values:\n- sleeve_length: sleeveless, short-sleeve, medium-sleeve, long-sleeve\n...\nRules:\n- Include only keys whose attribute is clearly visible in the image.\n- Use only the allowed values listed above.\n- Output the JSON object only, without any extra text or explanation."}, {"role": "assistant", "content": "{\"sleeve_length\": \"short-sleeve\", \"lower_clothing_length\": \"three-point\", \"socks\": \"no\", \"hat\": \"no\", \"glasses\": \"sunglasses\", \"neckwear\": \"no\", \"wrist_wearing\": \"no\", \"ring\": \"yes\", \"waist_accessories\": \"no\", \"neckline\": \"round\", \"outer_clothing_is_cardigan\": \"no\", \"upper_clothing_covering_navel\": \"no\", \"upper_fabric\": \"cotton\", \"lower_fabric\": \"cotton\", \"upper_pattern\": \"pure color\", \"lower_pattern\": \"lattice\"}"}], "images": ["images/WOMEN-Shorts-id_00005838-04_4_full.jpg"]}
```

### 4.4 数据准备与质量保障

**下载与解压**（一次性）：

```bash
# 官方 GitHub → Google Drive 下载 images.zip / labels.zip / captions.json
unzip images.zip -d data/data_origin/DeepFashion-MultiModal/   # 44,096 张图
unzip labels.zip -d data/data_origin/DeepFashion-MultiModal/
cp captions.json data/data_origin/DeepFashion-MultiModal/
```

**转换**（schema 主线 + simple 对照 + caption + attr_qa）：

```bash
conda activate lf
python scripts/data_convert_deepfashion_multi.py   # 无参运行，幂等；解压产物默认路径
```

转换规则（脚本内实现）：
- 按**图片粒度**划分 train/val/test = **80/10/10**（seed=42，三侧图片两两零重叠）——同一张图的所有样本同侧，防泄漏；**test 仅用于最终评测，全程只碰一次**；
- 只保留可见（非 NA/non-hidden）属性；清洗空 caption（原始数据有 7 条空描述，已剔除）；
- 图片按内容哈希去重，`images/` 字段为相对 `dataset_dir` 路径；
- 输出目录自带 `dataset_info.json` 注册表，LLaMA-Factory 直接读取。

**数据质量四关卡**（`python scripts/check_data.py`）：

| 关卡 | 检查内容 | 当前状态 |
|---|---|---|
| 1. schema 健全 | sharegpt 结构、`<image>` 占位符、图片文件存在 | ✅ |
| 2. 答案合法 | assistant JSON 可解析、键/值在官方表内、无空答案 | ✅ |
| 3. 防泄漏 | train/val/test 三侧图片两两零重叠 | ✅ |
| 4. 分布统计 | 每属性类别分布（不均衡预览）→ `llamafactory/results/data_stats.json` | ✅ |

已知不均衡点（影响评测设计）：多分类属性（sleeve_length/neckline/fabric/pattern）长尾明显（如 `suspenders`、`furry` 样本少）；评测用 macro-F1 + 分层，避免被 majority 类掩盖。

---

## 5. 训练方案

### 5.1 训练配置与关键做法

训练超参数以仓库内配置文件为**唯一事实来源**（配置会随调参迭代而调整，本文档不内嵌参数副本，避免与配置脱节过时）：

- `llamafactory/config/qwen3vl8b_dfattr_train_cli.yaml`：训练配置（扁平 CLI 格式；仓库另有 WebUI 分层版 `qwen3vl8b_dfattr_train.yaml` 与之等价），用 `llamafactory-cli train` 运行。

关键做法（与具体取值无关，稳定不变）：

- 模板 `qwen3_vl_nothink`：训练-推理一致、关 thinking；
- 冻结视觉塔与多模态投影器（`freeze_vision_tower` / `freeze_multi_modal_projector`，固定冻结；解冻对照暂不纳入，见 5.3）；
- 训练中监控：通过 `val_size: 0.05` 从训练集内部随机切 5% 看 eval loss 曲线（与手动 val/test 零重叠，不污染评测）；**最终结论仍以同一套 `scripts/eval_attr.py` 在 test 上的属性级指标为准**，保证口径一致；
- 消融通过修改配置文件中的 `dataset` / `max_samples` / `lora_rank` 进行（视觉塔冻结/解冻、多任务组合成本高，暂不纳入，见 5.3）。

### 5.2 训练步骤与调参

```bash
# 训练（单机多卡自动走 DDP，默认用全部可见 GPU；本机 3 卡，等效 batch = 4×3×3 = 36）
llamafactory-cli train llamafactory/config/qwen3vl8b_dfattr_train_cli.yaml
# 单卡：CUDA_VISIBLE_DEVICES=0 限制，并把配置文件里 gradient_accumulation_steps 改为 9
```

| 现象 | 动作 |
|---|---|
| 欠拟合（val 不涨） | 升 lr / 加 epoch / 解冻视觉塔 |
| 过拟合（val 先升后降） | 降 lr / 加 dropout / 减数据 / 冻结视觉塔 |
| JSON 输出乱 | 确认 `qwen3_vl_nothink` 生效；训练前打印渲染样本 |
| 显存溢出 | 降 batch；确认 gradient_checkpointing 开启 |

### 5.3 消融与对照实验设计（面试提分项）

> 组织方式按**论证链**分四环：**效果**（微调是否有效）→ **归因**（提升来自哪里）→ **边界**（效果依赖与鲁棒性）→ **过程**（训练稳定性）。所有对照在 **val 集**评测（test 只留最终报告），口径：llamafactory 管线生成预测 + `eval_attr.py stats` 打分；能免训练的先做，重训按"展示价值/成本"比排序。

| 论证环节 | 实验 | 变量 | 变更方式 |
|---|---|---|---|
| **效果** | 主结果 | 是否微调 | Base vs LoRA，test 4,410 条；McNemar 显著性 + 属性分层 |
| **效果** | 品类分层 | 图片品类 | 按图名前缀（MEN/WOMEN + 品类）统计准确率，检验泛化是否依赖品类 |
| **归因** | 指令范式 × 数据量 | 训练指令 / 数据规模 | schema vs simple × 5k/20k/35.3k，同一组 6 配置双轴解读 |
| **归因** | LoRA 秩 | r ∈ {8,16,32} | `lora_rank: 8`（可选） |
| **边界** | 指令鲁棒性 | 推理期指令 | 同一模型换 simple prompt 评测（与归因"指令范式"互补：归因换训练指令=方法收益，此处换推理指令=运行依赖） |
| **边界** | few-shot 对照 | 示例数量 | 基座 + 2 个 in-context 示例，回答"为什么微调不 few-shot" |
| **边界** | 图像分辨率 | 评测输入分辨率 | `image_max_pixels` 589,824 vs 825,750（推理端） |
| **过程** | Checkpoint 选择 | 训练步数 | 28 个 checkpoint 抽样快筛（500 条）+ 关键点全量确认，选最优定稿 test |

**暂不纳入**：

- **视觉塔冻结 vs 解冻**：解冻训练慢、结果方向不可控；
- **任务组合 attr_json vs +caption 多任务**：全量成本最高、收益叙事一般；
- **未见品类泛化**：需按品类整体划分数据并重训（当前按图随机划分），成本高。

---

## 6. 评测方案

### 6.1 指标体系（自建，针对 attr_json）

| 指标 | 口径 | 说明 |
|---|---|---|
| 属性级准确率 | 每个属性键 pred==true 比例 | 主指标；**预测缺键算错**（暴露漏检） |
| 属性级 macro-F1 | 多分类 F1 平均，缺键并入 `_missing` 类 | 对抗类别不均衡 |
| 全 JSON 匹配率 | parsed == true 精确一致 | 严指标 |
| **JSON 可解析率** | 输出能解析为 dict 的比例 | ≥98% 达标；格式错误单独统计 |
| caption（辅助） | CIDEr + LLM-as-judge 粗评 | 不进主线结论 |

**容错解析三级策略**（`scripts/eval_attr.py` 实现）：① 提取首个 `{...}` 区间 → `json.loads`；② 失败则从右截断至最后一个完整键值对；③ 仍失败记「格式错误」。

### 6.2 对比组设计（统一 baseline 纪律）

| 组 | 说明 |
|---|---|
| Base-0shot | 基座 + 同一 schema 指令（3 个候选 prompt 取最优后固定） |
| **LoRA 微调后** | 与 Base 用完全相同的评测代码与指令 |

> 核心纪律：**评测代码、数据集、prompt 三者在各组间完全一致，唯一变量是"是否微调"**。

### 6.3 评测执行

> **评测口径**：预测生成**统一走 llamafactory 管线**（与训练 100% 同口径：`qwen3_vl_nothink` 模板、mm_plugin 图像 token 显式展开、同数据格式），`scripts/eval_attr.py` 仅承担 `stats` 打分。原因：transformers 原生渲染（1 个 `<image_pad>` 占位符）与训练/llamafactory 的显式图像 token 展开（6~7 个 `<image_pad>`）输入不一致，30 条实测 4 条输出漂移（属性级 99% 一致、文本 87% 一致）——评测必须与训练同口径。

```bash
# ===== 开发阶段（模型选择/验证）：val 集，可反复跑 =====
llamafactory-cli train llamafactory/config/qwen3vl8b_dfattr_eval_val_base.yaml   # base
llamafactory-cli train llamafactory/config/qwen3vl8b_dfattr_eval_val_lora.yaml   # lora（两份配置除 adapter 外完全一致）
python scripts/convert_preds.py --preds llamafactory/eval_outputs/val_base/generated_predictions.jsonl \
    --data data/data_convert/deepfashion_multi/deepfashion_attr_json_val.jsonl --out llamafactory/results/preds_val_base.jsonl
python scripts/convert_preds.py --preds llamafactory/eval_outputs/val_lora/generated_predictions.jsonl \
    --data data/data_convert/deepfashion_multi/deepfashion_attr_json_val.jsonl --out llamafactory/results/preds_val_lora.jsonl
python scripts/eval_attr.py stats --base llamafactory/results/preds_val_base.jsonl \
    --lora llamafactory/results/preds_val_lora.jsonl --out llamafactory/results/report_val.md

# ===== 最终报告：test 集，全程只碰一次 =====
# 复制 val 两份配置为 test 版（eval_dataset 改 deepfashion_attr_json_test），重复上面步骤，
# 产出 llamafactory/results/report_test.md。
```

### 6.4 分层统计与显著性

- **按属性组分层**：形状 / 面料 / 图案 分别报指标；
- **按类别数分层**：二分类 vs 多分类（多分类是细粒度难点，重点报告）；
- **按品类泛化（加分项）**：按图名前缀（MEN/WOMEN + 品类）统计准确率；⚠️ 真正"未见品类泛化"需按品类整体划分数据并重训（当前按图随机划分下，test 品类在 train 均出现过）；
- **McNemar 检验**（配对二分类，连续校正）：每属性 + 总体 p<0.05 才下结论；或 3 seed 报均值 ± std。

---

## 7. badcase 分析与改进闭环

1. 从评测集挑 20~30 个答错样本截图，归类打标签：

| badcase 类别 | 典型表现 | 归因方向 |
|---|---|---|
| 类别混淆 | short↔medium-sleeve、cotton↔knitted、pure color↔color block | 细粒度边界样本不足 |
| 遮挡/姿势 | 手挡腰部、侧身误判领型 | 视角多样性不足 |
| 多件套误判 | 外搭开衫遮内搭，属性张冠李戴 | prompt 明确"只看可见层" |
| NA/隐藏误报 | 输出图上不可见属性 | prompt 强调只输出可见属性 |
| JSON 格式错误 | thinking 污染 / 截断 | 模板确认 + 容错解析 + few-shot |
| 背景干扰 | 复杂背景误判图案 | 视觉塔消融对照 |

2. 选 1~2 个高性价比对策（补样本 / 改 prompt / 模板化输出）→ 复训 → 同口径再评测，「改→训→评」闭环 1~2 轮，指标再提升若干 pp。

---

## 8. 相似款检索推荐

把主线 demo 的终点（属性 JSON）变成起点：用抽出的 18 维属性 + 视觉特征，在 DeepFashion-MultiModal 全库（约 4.4 万图 / 7975 件商品）检索最相似的「相似款」，让 demo 升级成「电商标注 + 相似款推荐」完整场景。

相似款检索采用**属性约束 + 视觉向量**混合方式：用冻结的 Qwen3-VL 视觉塔（`model.visual`）抽库内图 embedding 走向量检索，叠加官方标注的属性约束做过滤/加权——复用现有 `model.visual` 与现有数据，零新增标注、不动训练。

```
用户上传图 ──▶ 属性抽取（主线）──▶ 属性 JSON
                                        │
                                        ▼
        retrieval.py：属性约束过滤 + 视觉 cosine + 同款排重
                                        │
                                        ▼
                Gradio 相似款 Gallery（top-k，图上标匹配分）
```

**落地产物**：

| 文件 | 内容 |
|---|---|
| `scripts/build_retrieval_index.py` | 解析官方标注 + captions → 按商品 id 聚合 → 视觉塔抽 embedding → 存索引；支持多卡分片并行 + 合并 |
| `scripts/retrieval.py` | 加载索引；属性过滤/加权 + 向量余弦 → top-k 商品；只加载视觉塔 + 模型单例缓存 + 预热 |
| `scripts/serve_gradio.py`（改造） | 属性抽取基础上加「相似款」Gallery |
| `docs/检索推荐方案.md` | 检索/推荐闭环方案（检索管线、工程纪律、工作量） |

**工程纪律**：索引用 `lf` 环境构建（与训练/评测同 transformers 5.8），检索/Gradio 跑在 `vllm` 环境（embedding 口径一致，已验证 cosine=1.0）；只用视觉塔（576M）不加载 8B 语言模型；向量检索用 numpy 归一化点积（不引入 faiss）；检索约束一律用官方属性映射，与训练数据同源。

> 检索管线、索引构建参数、属性约束口径与工程纪律详见 `docs/检索推荐方案.md`。

---

## 9. 部署与 Demo

```bash
# 1) 合并 adapter（llamafactory/config/qwen3vl8b_dfattr_export_cli.yaml）
llamafactory-cli export llamafactory/config/qwen3vl8b_dfattr_export_cli.yaml

# 2) vLLM 服务（推荐一键脚本 bash scripts/start_vllm.sh；手动须在仓库根目录且设环境变量）
conda activate vllm
export CUDA_HOME=/home/maotai/.conda/envs/vllm/lib/python3.11/site-packages/nvidia/cu13
export VLLM_USE_FLASHINFER_SAMPLER=0   # 必须禁用 flashinfer 采样器（见部署说明 §5.4）
vllm serve llamafactory/saves/dfattr/merged/qwen3vl8b_dfattr_merged \
    --served-model-name deepfashion --port 8000 --gpu-memory-utilization 0.55 \
    --max-model-len 131072 --tensor-parallel-size 2   # 多卡 TP=2 + 128k；单卡去掉 TP 并把 max-model-len 改 32768

# 3) demo（另开终端）：传一张没见过的图 → 输出结构化 JSON
python scripts/serve_demo.py --image /path/to/never_seen.jpg   # CLI
python scripts/serve_gradio.py                                 # Web 页面 → http://127.0.0.1:7860
```

演示重点：**微调前（输出乱/属性错）vs 微调后（稳定 JSON、属性准）同图对比**——最有冲击力；Web 页面支持 schema/simple 模板切换与 prompt 手动编辑，可现场演示「朴素指令 vs schema 指令」的差异（消融中 +31.5pp 的关键方法），并一键检索相似款（见 §8）。

> 环境安装、单卡/多卡参数、API 对接示例（OpenAI 兼容）与全部踩坑记录见 `docs/部署说明.md`。

---

## 10. 步骤规划

> 按流程环节组织，每环节可独立迭代（无固定时间表，量力而行）。各环节细节见对应章节。

| 步骤 | 环节 | 关键动作 | 产出 |
|---|---|---|---|
| 1 | 环境与基线 | `lf` 环境确认 llamafactory；拉取 Qwen3-VL-8B-Instruct；关 thinking 推理一张图；基座 0-shot 小样本可学性验证 | 推理稳定 + 基线 |
| 2 | 数据准备（§4.4） | 下载解压 → 转换（四任务）→ 四道质量关卡 → 冻结评测集 | 训练/评测 jsonl + 质量报告 |
| 3 | 小步验证 | 500 条训 1~2 步；评测脚本骨架定型（llamafactory 管线生成 + `eval_attr.py stats` 打分） | loss 下降 + 评测管线通 |
| 4 | 主训练与导出（§5） | 5k 子集跑通 → 全量 35.3k（3 卡 DDP）→ `export` 合并 adapter | adapter + merged 模型 |
| 5 | 正式评测（§6） | val：Base vs LoRA 全对比；test 最终报告（全程只碰一次）；McNemar + 属性分层 | `report_val.md` / `report_test.md` |
| 6 | 消融与对照（§5.3） | 效果 / 归因 / 边界 / 过程 四环 | 各 `report_*.md` |
| 7 | badcase 分析（§7） | 错例归类归因；选 1~2 个高性价比对策 | badcase 报告 |
| 8 | 改进闭环（可选） | 补数据 / 改 prompt / 模板化输出 → 复训 → 同口径再评测 | 指标再提升 |
| 9 | 检索推荐（§8） | 建索引 → retrieval → Gradio 集成相似款 Gallery | 检索索引 + 相似款 demo |
| 10 | 部署 demo（§9） | vLLM serve + CLI/Web；微调前后同图对比 | 部署 demo（CLI + Web） |
| 11 | 简历与开源化 | 简历定稿（§12 模板）；15 分钟叙事演练；GitHub/HF + 技术博客 | 项目交付 |

---

## 11. 配套实现与仓库结构

| 路径 | 内容 |
|---|---|
| `scripts/data_convert_deepfashion_multi.py` | 数据转换（schema 主线 + simple 对照 + caption + attr_qa，属性表内嵌） |
| `scripts/check_data.py` | 数据质量四关卡 |
| `scripts/eval_attr.py` | 评测打分：`stats`（属性级 acc / macro-F1 / 分层 / McNemar）；预测生成走 llamafactory 管线（见 6.3） |
| `scripts/convert_preds.py` | 把 llamafactory predict 输出（generated_predictions.jsonl）转为 `eval_attr.py stats` 可消费的预测文件 |
| `scripts/serve_demo.py` | vLLM demo 客户端（CLI） |
| `scripts/serve_gradio.py` | Gradio Web demo（模板切换 + prompt 手动编辑；含相似款检索 Gallery） |
| `scripts/build_retrieval_index.py` | 检索索引构建（官方标注聚合 + 视觉塔抽 embedding + 多卡分片） |
| `scripts/retrieval.py` | 相似款检索（视觉+属性混合加权；只加载视觉塔 + 缓存 + 预热） |
| `scripts/start_vllm.sh` | vLLM 一键启动脚本（封装 CUDA_HOME / flashinfer 禁用等环境准备，单卡/多卡切换） |
| `scripts/start_web.sh` | Web demo 一键启动（vLLM + Gradio 全自动：GPU 分卡避让 / 健康检查） |
| `llamafactory/config/` | 训练/评测/导出配置（`ablation/`、`ckpt_screen/` 子目录） |
| `llamafactory/saves/` | 模型权重与训练/调度日志（LoRA adapter / merged；软链至 /disk16T1） |
| `llamafactory/eval_outputs/` | 评测生成预测（`generated_predictions.jsonl` 等，按实验分目录） |
| `llamafactory/results/` | 评测汇总与评测日志（`preds_*.jsonl`、报告、数据统计、`eval_*.log`） |
| `data/data_convert/deepfashion_multi/` | 转换产物（jsonl + `dataset_info.json` + `images/`） |
| `docs/部署说明.md` | 部署文档：环境准备、单卡/多卡启动、API 示例、踩坑记录 |
| `docs/检索推荐方案.md` | 检索/推荐闭环方案（相似款推荐） |
| `docs/数据说明.md` | 数据组织、原始格式、转换脚本、属性对照表 |
| `DeepFashion微调项目方案.md`（archive/ 下） | 本文档 |

**工程纪律**：seed 固定（划分/训练/推理全链路）；每个实验一行 CSV + 元数据（模型、r、数据量、seed、指标）；requirements 精确版本；转换/评测脚本与报告发布 GitHub/HuggingFace（公开可复现）。

---

## 12. 附录：简历表述与面试准备

### 12.1 简历项目描述

> 「基于 Qwen3-VL-8B-Instruct 对 DeepFashion-MultiModal 时装数据集（4.4 万图、18 维服装属性体系）做 LoRA 微调，实现服装图 → 结构化属性 JSON 抽取，并在此基础上落地全库相似款检索推荐；采用 schema 感知指令构造训练数据，属性级准确率从 XX% → YY%（+Z.Zpp），macro-F1 同步提升并通过 McNemar 检验；自建评测体系（属性级 accuracy / 混淆矩阵 / 分层统计 / 容错解析）；badcase 归因驱动补数据 + 模板化 prompt 二次迭代；产出 vLLM 部署 demo 与一键复现脚本。」

### 12.2 15 分钟叙事

动机（电商属性标注靠人工、通用 VLM 细粒度不可控）→ 方法（LoRA 数学 ΔW≈BA + 为什么冻结视觉塔）→ 数据（官方标注、NA/hidden 清洗、按图划分防泄漏、类别不均衡）→ 结果（量化提升 + 显著性 + 消融）→ 失败分析（short vs medium-sleeve 混淆、遮挡、多件套误判的归因与闭环）→ 工程（schema 感知数据构造、容错解析、约束解码、相似款检索、vLLM 部署）。

### 12.3 追问应对要点

| 追问 | 应答要点 |
|---|---|
| 为什么 DeepFashion 不用 ChartQA？ | 业务匹配 + 区分度 + 细粒度任务收益空间大 |
| 属性体系哪来的？可信吗？ | 官方人工标注（Text2Human/SIGGRAPH 2022），id→语义官方口径 |
| 为什么 attr_json 而不是逐属性 QA？ | 一次推理完整结构化、贴真实场景；attr_qa 模板重复易过拟合 |
| 类别不均衡怎么处理？ | macro-F1 + 混淆矩阵 + 分层报告；badcase 驱动补样本 |
| 为什么冻结视觉塔？ | 小数据防过拟合 + 节省训练资源；解冻作对照暂不纳入（成本高、结果方向不可控） |
| JSON 解析失败怎么办？ | 三级容错解析；格式错误单列统计（≥98% 达标） |
| 怎么防泄漏？ | 按图划分 80/10/10（seed 42）；test 只在最终评测碰一次 |
| LoRA rank 怎么定？ | 经验区间 16-64；当前用 16，计划补 r=8/32 消融（5k 数据，可选） |
| schema 指令比一句话好在哪？ | 键名/取值不漂移、格式错误率下降（对照实验量化） |
| 相似款检索怎么做？ | 属性约束 + 视觉 embedding 余弦混合加权，官方标注约束、只加载视觉塔（详见 §8） |

---

## 13. 附录：关键命令速查

```bash
# ===== 环境 =====
conda activate lf
pip install "llamafactory[torch]"            # 若未装
export USE_MODELSCOPE_HUB=1

# ===== 数据（解压产物放到 data/data_convert/deepfashion_multi/ 后无参运行，幂等）=====
python scripts/data_convert_deepfashion_multi.py
python scripts/check_data.py                  # 四关卡，退出码 0 才继续

# ===== 训练（单机多卡自动 DDP，默认全部可见 GPU；本机 3 卡）=====
llamafactory-cli train llamafactory/config/qwen3vl8b_dfattr_train_cli.yaml

# ===== 评测（正式口径：llamafactory 管线生成预测 + eval_attr.py stats 打分，详见 6.3）=====
llamafactory-cli train llamafactory/config/qwen3vl8b_dfattr_eval_val_base.yaml   # 基座 0-shot（val）
llamafactory-cli train llamafactory/config/qwen3vl8b_dfattr_eval_val_lora.yaml   # LoRA 微调后（val）
python scripts/convert_preds.py --preds llamafactory/eval_outputs/val_base/generated_predictions.jsonl \
    --data data/data_convert/deepfashion_multi/deepfashion_attr_json_val.jsonl --out llamafactory/results/preds_val_base.jsonl
python scripts/convert_preds.py --preds llamafactory/eval_outputs/val_lora/generated_predictions.jsonl \
    --data data/data_convert/deepfashion_multi/deepfashion_attr_json_val.jsonl --out llamafactory/results/preds_val_lora.jsonl
python scripts/eval_attr.py stats --base llamafactory/results/preds_val_base.jsonl \
    --lora llamafactory/results/preds_val_lora.jsonl --out llamafactory/results/report_val.md

# ===== 检索（见 §8 与 docs/检索推荐方案.md）=====
python scripts/build_retrieval_index.py       # 建检索索引（官方标注聚合 + 视觉塔抽 embedding）

# ===== 部署（完整命令与环境准备见 §9 与 docs/部署说明.md）=====
llamafactory-cli export llamafactory/config/qwen3vl8b_dfattr_export_cli.yaml
conda activate vllm && export CUDA_HOME=/home/maotai/.conda/envs/vllm/lib/python3.11/site-packages/nvidia/cu13
VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve llamafactory/saves/dfattr/merged/qwen3vl8b_dfattr_merged \
    --served-model-name deepfashion --port 8000 --max-model-len 32768
python scripts/serve_demo.py --image <path/to/test.jpg>
```
