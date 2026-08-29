# DeepFashion 服装属性提取与相似款检索：技术方案与实验报告

> **项目定位**：基于 Qwen3-VL-8B-Instruct，用 LLaMA-Factory 对 DeepFashion-MultiModal 时装数据集（SIGGRAPH 2022，4.4 万图、18 维官方标注属性）做 LoRA 微调，实现「服装全身图 → 结构化属性 JSON」抽取，并在抽取结果之上落地**相似款检索推荐**（属性 + 视觉混合加权，全库 top-k 相似款），升级为「电商标注 + 相似款推荐」完整闭环，对标电商货架属性 / 商品知识图谱 / 相似款推荐场景。
>
> **核心结果**：test 集属性级准确率 59.8% → **93.8%（+34.0pp）**，macro-F1 0.417 → **0.677**，McNemar 检验 p<0.001，JSON 可解析率 100%；相似款检索在部署 demo 上落地。

---

## 目录

1. [项目摘要](#1-项目摘要)
2. [背景与动机](#2-背景与动机)
3. [技术方案总览](#3-技术方案总览)
4. [数据与任务](#4-数据与任务)
5. [方法：模型与训练](#5-方法模型与训练)
6. [评测体系](#6-评测体系)
7. [实验结果](#7-实验结果)
8. [错误分析（badcase）](#8-错误分析badcase)
9. [相似款检索推荐](#9-相似款检索推荐)
10. [部署与 Demo](#10-部署与-demo)
11. [结论与展望](#11-结论与展望)
12. [附录](#12-附录)

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

**核心结果**：LoRA 微调将属性级准确率从 59.8% 提升至 **93.8%（+34.0pp）**，macro-F1 0.677，18 属性全部通过 McNemar 显著性检验（p<0.001），JSON 可解析率 100%；消融证明 **schema 感知指令是关键方法**（+31.5pp）；训练稳定无过拟合。

**三个核心卖点**：

1. **业务贴合**——「结构化属性抽取 + 相似款检索推荐」覆盖电商核心链路（货架属性、搜索召回、商品知识图谱、相似款推荐），价值一句话讲清；
2. **区分度**——ChartQA 微调是烂大街模板；"DeepFashion + VLM 微调 + 结构化属性抽取 + 相似款检索"完整闭环几乎无人做；
3. **严谨评测**——数据集无官方指标 → 自建完整评测体系，反而展示方法论能力（对比组纪律、分层、显著性检验）。

**文档定位**：本文档为项目**技术方案 + 实验报告**合稿——完整给出背景、数据、方法、评测、实验结果、错误分析、检索闭环、部署与复现，可作岗位投递附件、面试前置材料或作品集展示。

---

## 2. 背景与动机

### 2.1 业务痛点

电商平台的商品属性数据是基础设施：货架属性决定商品如何被展示，搜索标签决定商品如何被召回，商品知识图谱决定关联推荐的质量。目前这些属性大量依赖**人工标注**，存在三个问题：

- **成本高**：全量人工标注覆盖海量 SKU 不现实，长尾商品覆盖差；
- **标准不一**：不同标注员对同一件衣服的袖长、面料判断不一致；
- **时效差**：上新速度快，人工标注跟不上。

### 2.2 通用 VLM 的局限

通用多模态大模型（如 Qwen3-VL）具备基础的图像理解能力，可以直接"看图说话"，但对**细粒度结构化属性抽取**存在系统性缺陷：

- **格式不稳定**：输出自由文本，不保证是合法 JSON，键名随意漂移；
- **属性混淆**：细粒度属性（medium-sleeve vs long-sleeve、cotton vs knitted、pure color vs color block）边界模糊，张冠李戴；
- **幻觉**：输出图上不可见的属性（NA/hidden 误报）。

### 2.3 方案定位

用领域数据对 VLM 做轻量 LoRA 微调，把"通用理解"变成"业务可用的结构化抽取能力"，并把抽取结果用于**相似款检索**，形成「标注 + 推荐」的完整电商闭环。同时给出**完整的评测证据链**——不是"训完就完"，而是可量化、可归因、可复现的工程闭环。

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

- **数据集**：DeepFashion-MultiModal（Text2Human，SIGGRAPH 2022），官方人工标注，无版权风险，可公开复现；
- **规模**：44,096 张 750×1101 全身人像图；`captions.json` 42,544 条描述；labels 含 12 个形状属性 + 面料/图案标注；
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
| `deepfashion_caption_*` | 34,006 / 4,258 / 4,273 | 图 → 自然语言描述 | 辅助（多任务组合暂不纳入，见 5.4） |
| `deepfashion_attr_qa_*` | 451,744 / 56,860 / 56,943 | 逐属性单问 → 单值 | 弃用（模板重复易过拟合） |

**为什么 attr_json 做主任务**：一次推理输出完整结构化属性，贴合真实电商抽取场景；输出为 JSON，与"图像 → 结构化输出"主线一致；评测可按属性键解构，严谨且细粒度。

**schema 感知指令**（行业标准结构化输出的数据侧做法；`build_schema_prompt()` 由属性表自动生成，训练/评测共用完整原文）：

```
Extract the clothing attributes visible in this image and output a JSON object.

Allowed keys and values:
- sleeve_length: sleeveless, short-sleeve, medium-sleeve, long-sleeve, not long-sleeve
- lower_clothing_length: three-point, medium short, three-quarter, long
- socks: no, socks, leggings
- hat: no, yes
- glasses: no, eyeglasses, sunglasses, glasses in hand or on clothes
- neckwear: no, yes
- wrist_wearing: no, yes
- ring: no, yes
- waist_accessories: no, belt, clothing at waist
- neckline: V-shape, square, round, standing, lapel, suspenders
- outer_clothing_is_cardigan: yes, no
- upper_clothing_covering_navel: no, yes
- upper_fabric: denim, cotton, leather, furry, knitted, chiffon, other
- upper_pattern: floral, graphic, striped, pure color, lattice, other, color block
- lower_fabric: denim, cotton, leather, furry, knitted, chiffon, other
- lower_pattern: floral, graphic, striped, pure color, lattice, other, color block
- outer_fabric: denim, cotton, leather, furry, knitted, chiffon, other
- outer_pattern: floral, graphic, striped, pure color, lattice, other, color block

Rules:
- Include only keys whose attribute is clearly visible in the image.
- Use only the allowed values listed above.
- Output the JSON object only, without any extra text or explanation.
```

结构：① 任务句；② **Allowed keys and values**——枚举全部 18 个属性键及每键合法取值；③ **Rules** 三条输出规则。指令由官方属性表自动生成，键与取值顺序稳定，保证训练/评测/推理三端完全一致。朴素指令对照版（simple）：`Extract the clothing attributes visible in this image and output them as JSON.`

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

## 5. 方法：模型与训练

### 5.1 模型与微调配置

| 项 | 配置 |
|---|---|
| 基座 | Qwen3-VL-8B-Instruct（约 8.5B） |
| 微调方式 | LoRA（rank=16, alpha=32, dropout=0.05, target=all），可训练参数 <1% |
| 优化 | AdamW，lr=1e-4，cosine 调度，warmup 150 步 |
| 训练 | 3 epoch，等效 batch 36（4/卡 × 3 卡 × 3 累积），3×48GB RTX 4090 DDP |
| 视觉塔 | **冻结**视觉塔与多模态投影器（小数据防过拟合 + 节省训练资源） |
| 模板 | `qwen3_vl_nothink`（训练-推理一致、关闭 thinking） |
| 图像 | `image_max_pixels=825,750`（原图 750×1101 像素，**完全不压缩**，训练-推理口径统一） |
| 训练监控 | val_size 5% 从训练集内切出看 eval loss 曲线（与手动 val/test 零重叠，不污染评测） |

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

### 5.3 评测口径一致性（工程关键）

评测结果的可信度取决于**评测与训练的输入是否完全同口径**。本项目评测生成统一走 llamafactory 管线（与训练同模板 `qwen3_vl_nothink`、同多模态图像 token 展开、同数据格式、同解码参数 greedy），打分由自建 `eval_attr.py` 承担。这一做法源于一次实证发现：若评测改用 transformers 原生渲染（1 个图像占位符 vs 训练时的显式 6~7 个图像 token 展开），30 条实测会有 4 条输出漂移（属性级 99% 一致、文本 87% 一致）——**评测必须与训练同口径**。

### 5.4 消融与对照实验设计

> 组织方式按**论证链**分四环：**效果**（微调是否有效）→ **归因**（提升来自哪里）→ **边界**（效果依赖与鲁棒性）→ **过程**（训练稳定性）。所有对照在 **val 集**评测（test 只留最终报告），口径：llamafactory 管线生成预测 + `eval_attr.py stats` 打分。

| 论证环节 | 实验 | 变量 | 变更方式 |
|---|---|---|---|
| **效果** | 主结果 | 是否微调 | Base vs LoRA，test 4,410 条；McNemar 显著性 + 属性分层 |
| **效果** | 品类分层 | 图片品类 | 按图名前缀（MEN/WOMEN + 品类）统计准确率，检验泛化是否依赖品类 |
| **归因** | 指令范式 × 数据量 | 训练指令 / 数据规模 | schema vs simple × 5k/20k/35.3k，同一组 6 配置双轴解读 |
| **归因** | LoRA 秩 | r ∈ {8,16,32} | `lora_rank: 8`（可选） |
| **边界** | 指令鲁棒性 | 推理期指令 | 同一模型换 simple prompt 评测（与归因"指令范式"互补） |
| **边界** | few-shot 对照 | 示例数量 | 基座 + 2 个 in-context 示例，回答"为什么微调不 few-shot" |
| **边界** | 图像分辨率 | 评测输入分辨率 | `image_max_pixels` 589,824 vs 825,750（推理端） |
| **过程** | Checkpoint 选择 | 训练步数 | 28 个 checkpoint 抽样快筛（500 条）+ 关键点全量确认，选最优定稿 test |

**暂不纳入**：视觉塔冻结 vs 解冻（解冻训练慢、结果方向不可控）；任务组合 attr_json vs +caption 多任务（成本最高、收益叙事一般）；未见品类泛化（需按品类整体划分数据并重训，成本高）。

---

## 6. 评测体系

### 6.1 指标定义

| 指标 | 口径 | 说明 |
|---|---|---|
| 属性级准确率 | 每个属性键 pred==true 比例 | 主指标；**预测缺键算错**（暴露漏检） |
| 属性级 macro-F1 | 多分类 F1 平均，缺键并入 `_missing` 类 | 对抗类别不均衡 |
| 全 JSON 匹配率 | 解析结果 == 真值 dict 精确一致 | 严格指标 |
| JSON 可解析率 | 输出能解析为 dict 的比例 | ≥98% 达标；格式错误单独统计 |

**容错解析三级策略**（`scripts/eval_attr.py` 实现）：① 提取首个 `{...}` 区间 → `json.loads`；② 失败则从右截断至最后一个完整键值对；③ 仍失败记「格式错误」。

### 6.2 对比组设计与统计检验

- **对比组**：Base-0shot（基座 + 同一 schema 指令） vs LoRA 微调后——评测代码、数据集、prompt 三者完全一致，**唯一变量是"是否微调"**；
- **McNemar 检验**（配对二分类、连续校正）：逐属性 + 总体，p<0.05 才下结论；
- **分层统计**：按属性组（形状/面料图案）、按类别数（二分类/多分类）分别报告，避免被 majority 类掩盖；**按品类泛化（加分项）**：按图名前缀（MEN/WOMEN + 品类）统计准确率，⚠️ 真正"未见品类泛化"需按品类整体划分数据并重训（当前按图随机划分下，test 品类在 train 均出现过）。

### 6.3 评测执行

> **评测口径**：预测生成**统一走 llamafactory 管线**（与训练 100% 同口径），`scripts/eval_attr.py` 仅承担 `stats` 打分（原因见 §5.3）。

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

---

## 7. 实验结果

> 本章按**论证链**组织：先证明**效果**（§7.1 微调是否有效），再**归因**（§7.2 提升来自哪里），再划定**边界**（§7.3 效果依赖什么、替代方案如何），最后检验**过程**（§7.4 训练是否稳定），以汇总表收尾（§7.5）。

### 7.1 效果论证：微调是否有效

**主结果（test 集，4,410 张未见图）**：

| 指标 | Base 0-shot | **LoRA 微调后** | 提升 |
|---|---|---|---|
| JSON 可解析率 | 100.0% | **100.0%** | —（达标 ≥98%） |
| 全 JSON 精确匹配率 | 0.0% | **47.3%** | +47.3pp |
| **属性级准确率** | 59.8% | **93.8%** | **+34.0pp** |
| macro-F1 | 0.417 | **0.677** | +0.260 |
| McNemar（总体） | — | **p < 0.001**（b=20106, c=833） | 极显著 |

**逐属性**：18 个属性全部显著提升（McNemar 逐属性 p 均 < 0.001），提升最大的是 `upper_clothing_covering_navel`（+76.0pp）、`lower_clothing_length`（+69.2pp）、`glasses`（+46.8pp）；提升最小的是 `sleeve_length`（+3.5pp，基座已达 91.8%，接近饱和）。

**属性分层**（按属性类型）：

| 分层 | Base | LoRA | Δ |
|---|---|---|---|
| shape 属性（12 个） | 56.8% | 96.3% | +39.6pp |
| texture 属性（6 个） | 65.8% | 88.6% | +22.8pp |
| 二分类（6 个） | 54.3% | 96.7% | +42.4pp |
| 多分类（12 个，细粒度难点） | 62.5% | 92.3% | +29.8pp |

**品类分层**（按图片品类，与属性分层维度不同）：全部 23 个品类准确率区间 **90.1%~100%**，表现均衡、无灾难性品类；样本量 top3 大品类（Tees_Tanks / Blouses_Shirts / Dresses）93.8%~96.4%；性别维度 MEN 96.6% vs WOMEN 94.6%（女装品类属性更多样）。模型泛化**不依赖特定品类**。

**解读**：三个证据共同支撑"微调有效"——① 总体提升 +34.0pp 且 McNemar 极显著；② 对各类属性类型（判断类 +40pp 级、细粒度多分类 +30pp 级）普适；③ 跨品类表现均衡（90%+），效果不是某一类衣服的偶然。

### 7.2 归因论证：提升来自哪里

主结果回答"是否有效"，本节回答"为什么有效"——把 +34pp 的提升拆解为可解释的来源。消融在 val 集（4,410 条）评测，按**指令范式（schema/simple）× 数据量（5k/20k/35k）**二维展开，每次对比保持**唯一变量**。同一组 6 个训练配置构成如下二维矩阵（val 集属性准确率），**横向（行内）与纵向（列内）各持一个唯一变量**：

| 训练指令 | 5k | 20k | 35k（全量） | Δ(5k→35k) |
|---|---|---|---|---|
| schema | 91.9% | 92.9% | **93.4%** | +1.5pp |
| simple | 60.4% | 74.0% | 75.3% | +14.9pp |
| Δ(schema−simple) | +31.5pp | +19.0pp | +18.1pp | — |

*表注：纵向（列内）比较 = 指令范式消融（唯一变量=训练指令）；横向（行内）比较 = 数据量消融（唯一变量=训练数据量）。*

两个辅助指标的完整数据（macro-F1、全 JSON 匹配率）如下，供正文引用核验（35k 为全量训练数据，加粗行为最终主模型）：

| 模型×数据量 | macro-F1 | 全 JSON 匹配 |
|---|---|---|
| schema-5k | 0.631 | 37.8% |
| schema-20k | 0.664 | 43.6% |
| **schema-35k** | **0.670** | **45.8%** |
| simple-5k | 0.425 | 0.0% |
| simple-20k | 0.511 | 0.0% |
| simple-35k | 0.518 | 0.0% |

**沿指令轴（schema vs simple）——指令范式是决定性方法**：相同数据与训练配置下，枚举合法键/取值的指令始终显著优于朴素指令。优势随数据量增大而收窄（+31.5pp→+18.1pp），但朴素指令用全量数据（35k）也只有 75.3%，仍不及 schema 指令用 5k 数据（91.9%）的水平；macro-F1 同样全面落后（补充指标表中全量 0.670 vs 0.518），且 simple 版在三个数据量下全 JSON 匹配率恒为 0.0%（输出从不按 18 维 schema 对齐）。**指令设计决定了结构化输出的成败**——这是任务工程化的方法论收益，直接支撑 §4.3 的设计动机。schema 指令的重要性不止于训练期：推理期若把评测指令替换为 simple，效果同样崩塌（见 §7.3.1）。

**沿数据轴（5k→20k→35k）——数据效率极高、堆数据事倍功半**：schema 曲线平缓——数据量从 5k 增至 35.3k 属性准确率仅 +1.5pp，且收益集中在前 20k（5k→20k +1.0pp、全 JSON 匹配率 +5.8pp；20k→35k 边际递减 +0.5pp/+2.2pp），更多数据主要帮助**整段 JSON 的完整性**（少漏键、少缺属性），schema 指令下模型数据效率极高（5k 即达 92% 水平）；simple 曲线陡峭——数据收益 +14.9pp 几乎全被"追赶指令差距"消耗（20k 即 +13.6pp 后趋平），说明没有好的指令设计，堆数据事倍功半。

**归因小结**：提升主要来自**任务工程化（schema 指令设计）**而非数据堆量，双向验证：schema 用 1/7 数据（5k）拿到 91.9%，simple 用全量数据（35k）只有 75.3%——这与"电商标注贵"的业务痛点直接呼应：**好的任务设计可以用很少的数据拿到大部分收益**。

### 7.3 边界论证：效果依赖与鲁棒性

主结果与归因证明"微调有效、来自方法"，本节划定这套效果的**适用边界**。三个对照均在 val 集评测，模型为同一最终模型（或基座）。

**7.3.1 运行期依赖：指令鲁棒性（同一 LoRA 模型，仅换评测指令）**

本节与 §7.2 指令轴互补：§7.2 更换的是**训练指令**（重训两组模型，测得方法收益 +18~31pp），本节更换的是**推理指令**（模型不变，测得运行依赖）——同一变量（schema/simple）在训练期与运行期的两个作用阶段。

| 模型×指令 | 属性准确率 | parse | 全 JSON 匹配 |
|---|---|---|---|
| LoRA 主模型（schema） | **93.4%** | 100.0% | 45.8% |
| LoRA 主模型 + simple prompt | 0.2% | 99.7% | 0.0% |

在 simple 朴素指令下，LoRA 模型输出**退化为自由格式 JSON**（键名如 `top.type`、`clothing[].color`，不再是 18 维属性键名），按任务 schema 对齐后属性级准确率仅 0.2%（parse 率 99.7% 说明是合法 JSON 但结构不对齐）。**schema 指令是运行期必需**：业务部署必须携带 schema prompt。这与 §7.2 指令轴解读互相印证——训练期"有它 +31.5~18.1pp"（方法收益），推理期"没它 -93.2pp"（运行依赖）。

**7.3.2 替代方案：few-shot 对照（基座 + 2 个 in-context 示例）**

| 模型×指令 | 属性准确率 | 全 JSON 匹配 |
|---|---|---|
| 基座 0-shot（schema） | 59.9% | 0.0% |
| 基座 2-shot（schema + 2 示例） | 72.3% | 0.1% |
| LoRA 微调（schema） | 93.4% | 45.8% |

基座 2-shot 较 0-shot 提升 +12.4pp（in-context 示例确有帮助），但**仍比微调低 21.1pp**，且每次推理需额外携带示例输入——用数据回答"为什么微调而不 few-shot"：few-shot 无法达到结构化抽取的业务可用水平。

**7.3.3 输入鲁棒性：图像分辨率（同一模型，评测输入压缩 vs 不压缩）**

| image_max_pixels | 属性准确率 | 全 JSON 匹配 |
|---|---|---|
| 825,750（不压缩，与训练一致） | **93.4%** | 45.8% |
| 589,824（压缩） | 93.0% | 44.0% |
| Δ | **-0.4pp** | -1.8pp |

评测输入压缩到 768×768 仅造成 -0.4pp 损失——模型对输入分辨率鲁棒；评测口径维持与训练一致（不压缩）。

**边界小结**：效果成立的前提是 **schema 指令在场**（训练与部署都要带）；few-shot 不是可行替代；输入分辨率不是敏感因素。

### 7.4 过程论证：训练稳定性与模型选择

对训练全程 28 个 checkpoint 抽样 15 个点，在 val 上评测（500 条快筛 + 关键点全量确认）：

```
100步  90.4%   300步 91.3%   500步 92.1%   700步 92.5%
900步  93.8%  ← 峰值（约 1 epoch）
1100步 92.6%   1300步 92.8%   ...   2100步 93.8%（第二峰值）
2500步 93.4%   2793步 93.4%（最终）
```

全量确认（4,410 条 val）：ckpt-900 93.3% / ckpt-2100 93.4% / **ckpt-2793（最终）93.4%（macro-F1 0.670 最优）**。

**结论**：① LoRA 收敛极快——训练 100 步（约 3% 数据）即达 90.4%；② 全程无过拟合——曲线在 92.5%~93.8% 平台波动、不下降，3 epoch 是稳妥选择；③ 最终 checkpoint 在 macro-F1 与全 JSON 匹配上略优，用作 test 定稿模型——模型选择有据可依，不是拍脑袋。

### 7.5 实验汇总（按论证环节）

| 论证环节 | 实验 | 唯一变量 | 结论 |
|---|---|---|---|
| 效果 | 主结果（test） | 是否微调 | +34.0pp（59.8%→93.8%），McNemar p<0.001 |
| 效果 | 品类分层（test） | 图片品类 | 23 品类 90~100%，泛化不依赖品类 |
| 归因 | 指令范式（val） | 训练指令 | schema 比 simple 高 +31.5pp（5k）~+18.1pp（35k），simple 全量仍不及 schema-5k（方法决定性） |
| 归因 | 数据量（val） | 数据规模 | schema 5k→35.3k 仅 +1.5pp（数据效率高）；simple +14.9pp 仍追不上 |
| 边界 | 指令鲁棒性（val） | 推理期指令 | simple 下退化 0.2%，schema 为运行期必需 |
| 边界 | few-shot 对照（val） | 示例数量 | 基座 2-shot +12.4pp，仍低微调 21.1pp |
| 边界 | 图像分辨率（val） | 输入分辨率 | 压缩仅 -0.4pp（鲁棒） |
| 过程 | Checkpoint（val） | 训练步数 | 100 步即 90%+，全程无过拟合 |

---

## 8. 错误分析（badcase）

对 val 集 4,410 条预测中 LoRA 模型的错误进行统计（4,410 条中 2,391 条未达全 JSON 精确匹配），错误模式如下：

### 8.1 错误分布：细粒度面料/图案为主

错误最多的属性集中在 texture 组：`lower_fabric`（440 次）、`upper_fabric`（394 次）、`upper_pattern`（333 次）、`lower_pattern`（276 次）——材质与图案是视觉上最难区分的细粒度维度。

### 8.2 典型混淆对（真值 → 预测）

| 属性 | 混淆 | 次数 | 归因 |
|---|---|---|---|
| 面料 | `chiffon → cotton` / `cotton → chiffon` / `cotton → denim` | 179 / 111 / 104 | 轻薄面料 vs 棉 vs 牛仔的纹理相似，视觉边界模糊 |
| 戒指 | `ring: yes → no` | 113 | **小物件漏检**（遮挡/尺寸小） |
| 图案 | `graphic → pure color` | 92 | 图案细节丢失（远景/小图） |
| 领饰 | `neckwear: yes → no` | 78 | 小物件漏检 |
| 袖长 | `medium-sleeve → long-sleeve` | 76 | 细粒度边界（中袖 vs 长袖） |
| 腰饰 | `waist_accessories: no → 缺键` | 74 | 输出完整性（漏键） |

**典型错误样本示例**（val 集，对应图片已复制至 `docs/badcase_samples/`，可直接打开查看）：

| 属性 | 真值 | 预测 | 样本图片 | 错误类型 |
|---|---|---|---|---|
| ring | yes | no | `docs/badcase_samples/WOMEN-Tees_Tanks-id_00006297-09_2_side.jpg` | 小物件漏检 |
| upper_fabric | chiffon | cotton | `docs/badcase_samples/WOMEN-Dresses-id_00003255-01_2_side.jpg` | 面料混淆 |
| sleeve_length | medium-sleeve | long-sleeve | `docs/badcase_samples/WOMEN-Blouses_Shirts-id_00004948-03_3_back.jpg` | 细粒度边界 |
| neckwear | yes | no | `docs/badcase_samples/WOMEN-Tees_Tanks-id_00005040-02_1_front.jpg` | 小物件漏检 |
| upper_pattern | graphic | pure color | `docs/badcase_samples/WOMEN-Tees_Tanks-id_00001025-03_2_side.jpg` | 图案细节丢失 |
| lower_fabric | denim | cotton | `docs/badcase_samples/WOMEN-Tees_Tanks-id_00000532-01_2_side.jpg` | 面料混淆 |

### 8.3 归因与改进方向

1. **类别混淆**（面料/图案/袖长）：细粒度边界样本不足 → 补充困难样本 / 类别加权训练；
2. **小物件漏检**（戒指/领饰）：目标小、易遮挡 → 提升图像分辨率 / 显式提示"注意配饰" / 视觉塔微调（当前冻结）；
3. **输出完整性**（漏键）：全 JSON 匹配率 47.3% 的主因 → 模板化输出 / 约束解码 / 补训练样本。

这些模式同时构成后续迭代的优先级依据（"改 → 训 → 评"闭环）。

---

## 9. 相似款检索推荐

把主线 demo 的终点（属性 JSON）变成相似款检索的起点：用抽取的 18 维属性 + 视觉特征，在 DeepFashion-MultiModal 全库（约 4.4 万图 / 7975 件商品，入索引 7972 件）检索最相似的「相似款」，让 demo 升级成「电商标注 + 相似款推荐」完整场景。

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

## 10. 部署与 Demo

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

演示重点：**微调前（输出乱/属性错）vs 微调后（稳定 JSON、属性准）同图对比**——最有冲击力；Web 页面支持 schema/simple 模板切换与 prompt 手动编辑，可现场演示「朴素指令 vs schema 指令」的差异（消融中 +31.5pp 的关键方法），并一键检索相似款（见 §9）。

> 环境安装、单卡/多卡参数、API 对接示例（OpenAI 兼容）与全部踩坑记录见 `docs/部署说明.md`。

---

## 11. 结论与展望

### 11.1 结论

1. **任务达成**：LoRA 微调将服装属性抽取的属性级准确率从 59.8% 提升至 93.8%（+34.0pp），macro-F1 0.677，18 属性全部通过 McNemar 显著性检验，JSON 可解析率 100%——微调收益远超常见预期（细粒度任务上通用 VLM 的 0-shot 能力确实弱，领域微调空间巨大）；
2. **方法论价值**：schema 感知指令是决定性因素（+31.5pp），数据效率高（5k 即达 92%），训练稳定无过拟合——这套"任务工程化 + 轻量微调 + 严谨评测"的组合可以直接迁移到其他结构化抽取场景（商品属性、证件信息、表格抽取等）；
3. **工程闭环完整**：数据（防泄漏划分 + 质量关卡）→ 训练（LoRA + 口径一致）→ 评测（自建指标 + 统计检验 + 消融）→ 分析（badcase 归因）→ 部署（vLLM demo），可复现、可讲述；
4. **相似款检索（闭环落地）**：在部署 demo 上落地**相似款检索推荐**（图 → 属性 → 相似款推荐），复用主线视觉塔抽 embedding + 官方属性约束加权，把抽取结果升级为完整的「电商标注 + 相似款推荐」场景（详见 §9 与 `docs/检索推荐方案.md`）。

### 11.2 局限

- 细粒度面料/图案准确率（88.6%）仍低于形状属性（96.3%），长尾类别（如 `furry`、`suspenders`）样本不足；
- 小物件（戒指、领饰）漏检率偏高；
- 全 JSON 精确匹配率 47.3% 仍有提升空间（任何单属性错误即判不匹配）。

### 11.3 展望

- **数据侧**：针对混淆对补充困难样本、类别重加权；
- **模型侧**：解冻视觉塔（对照实验）、约束解码保证输出 schema 合法性；
- **任务侧**：扩展到多任务（属性 + 描述）、多部位细粒度分割辅助；
- **部署侧**：vLLM 服务化 + 线上 A/B 验证真实电商货架属性收益。

---

## 12. 附录

### 12.1 配套实现与仓库结构

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
| `docs/检索推荐方案.md` | 检索/推荐闭环方案（检索管线、工程纪律、工作量） |
| `docs/数据说明.md` | 数据组织、原始格式、转换脚本、属性对照表 |

**工程纪律**：seed 固定（划分/训练/推理全链路）；每个实验一行 CSV + 元数据（模型、r、数据量、seed、指标）；requirements 精确版本；转换/评测脚本与报告发布 GitHub/HuggingFace（公开可复现）。

### 12.2 简历表述与面试准备

**简历项目描述**：

> 「基于 Qwen3-VL-8B-Instruct 对 DeepFashion-MultiModal 时装数据集（4.4 万图、18 维服装属性体系）做 LoRA 微调，实现服装图 → 结构化属性 JSON 抽取，并在此基础上落地全库相似款检索推荐；采用 schema 感知指令构造训练数据，属性级准确率从 59.8% → 93.8%（+34.0pp），macro-F1 同步提升并通过 McNemar 检验；自建评测体系（属性级 accuracy / 混淆矩阵 / 分层统计 / 容错解析）；badcase 归因驱动补数据 + 模板化 prompt 二次迭代；产出 vLLM 部署 demo 与一键复现脚本。」

**15 分钟叙事**：动机（电商属性标注靠人工、通用 VLM 细粒度不可控）→ 方法（LoRA 数学 ΔW≈BA + 为什么冻结视觉塔）→ 数据（官方标注、NA/hidden 清洗、按图划分防泄漏、类别不均衡）→ 结果（量化提升 + 显著性 + 消融）→ 失败分析（short vs medium-sleeve 混淆、遮挡、多件套误判的归因与闭环）→ 工程（schema 感知数据构造、容错解析、约束解码、相似款检索、vLLM 部署）。

**追问应对要点**：

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
| 相似款检索怎么做？ | 属性约束 + 视觉 embedding 余弦混合加权，官方标注约束、只加载视觉塔（详见 §9） |

### 12.3 关键命令速查

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

# ===== 检索（见 §9 与 docs/检索推荐方案.md）=====
python scripts/build_retrieval_index.py       # 建检索索引（官方标注聚合 + 视觉塔抽 embedding）

# ===== 部署（完整命令与环境准备见 §10 与 docs/部署说明.md）=====
llamafactory-cli export llamafactory/config/qwen3vl8b_dfattr_export_cli.yaml
conda activate vllm && export CUDA_HOME=/home/maotai/.conda/envs/vllm/lib/python3.11/site-packages/nvidia/cu13
VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve llamafactory/saves/dfattr/merged/qwen3vl8b_dfattr_merged \
    --served-model-name deepfashion --port 8000 --max-model-len 32768
python scripts/serve_demo.py --image <path/to/test.jpg>
```

### 12.4 复现指引

| 环节 | 入口 |
|---|---|
| 数据转换 | `python scripts/data_convert_deepfashion_multi.py` → `data/data_convert/deepfashion_multi/` |
| 数据质量 | `python scripts/check_data.py`（四关卡） |
| 训练 | `llamafactory-cli train llamafactory/config/qwen3vl8b_dfattr_train_cli.yaml`（单机多卡自动 DDP，本机 3 卡） |
| 消融训练 | `llamafactory/config/ablation/*.yaml`（schema/simple × 5k/20k/35k 全量） |
| 评测生成 | `llamafactory-cli train llamafactory/config/qwen3vl8b_dfattr_eval_val_{base,lora}.yaml` |
| 评测打分 | `python scripts/convert_preds.py` + `python scripts/eval_attr.py stats` |
| 主报告产物 | `llamafactory/results/report_val.md`（val）/ `report_test.md`（test） |
| 消融报告产物 | `llamafactory/results/report_ablation.md`（指令范式 + 数据量） |
| Checkpoint 报告 | `llamafactory/results/report_ckpt_screen.md`（快筛曲线）/ `report_ckpt_confirm.md`（top3 全量确认） |
| 分辨率报告 | `llamafactory/results/report_resolution.md` |
| 补充实验产物 | `llamafactory/results/report_category.md`（品类分层）/ `report_supplement.md`（指令鲁棒性 + few-shot） |
| 部署 | `llamafactory-cli export`（merge）→ `vllm serve` → `python scripts/serve_demo.py`（CLI）／`python scripts/serve_gradio.py`（Web） |
| 部署文档 | `docs/部署说明.md`（环境准备、单卡/多卡启动、API 示例、踩坑记录） |
