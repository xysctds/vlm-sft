# 基于 Qwen3-VL-8B-Instruct 的服装属性抽取：LoRA 微调技术报告

> 任务：服装全身图 → 结构化属性 JSON（18 维服装属性）
> 模型：Qwen3-VL-8B-Instruct + LoRA（可训练参数 <1%）
> 数据：DeepFashion-MultiModal（SIGGRAPH 2022，44,096 图，官方人工标注）
> 核心结果：test 集属性级准确率 59.8% → **93.8%（+34.0pp）**，macro-F1 0.417 → **0.677**，McNemar 检验 p<0.001；JSON 可解析率 100%

---

## 摘要

电商平台的商品属性（袖长、领型、面料、图案等）是货架展示、搜索召回与商品知识图谱的基础数据，当前高度依赖人工标注，成本高且标准不一。通用多模态大模型（VLM）具备基础视觉理解能力，但对服装细粒度属性的输出**不可控**：格式不稳定、属性张冠李戴、幻觉图上不存在的属性。

本报告给出完整的领域微调方案：基于 Qwen3-VL-8B-Instruct，对 DeepFashion-MultiModal 时装数据集（4.4 万图、18 维官方标注属性体系）做 LoRA 微调，配合 **schema 感知指令**（枚举全部合法键与取值，结构化输出的工业标准做法），实现"服装全身图 → 结构化属性 JSON"的稳定抽取。配套自建评测体系（属性级准确率 / macro-F1 / 分层统计 / McNemar 显著性 / 容错解析）与严格的口径纪律（评测与训练同输入表示、唯一变量为是否微调）。

在 4,410 张 test 图上，微调后模型属性级准确率 93.8%（基座 0-shot 仅 59.8%，+34.0pp），全 JSON 精确匹配率 47.3%（基座 0.0%），18 个属性全部通过 McNemar 显著性检验（p<0.001）。消融实验证明：**schema 感知指令是关键方法**——5k 数据下比朴素指令高 +31.5pp，全量 35k 下仍高 +18.1pp（朴素指令用全量数据 35k 也只有 75.3%，仍不及 schema 指令用 5k 数据的 91.9%），且模型数据效率高（5k 数据即可达全量 91.9/93.4 的水平）；Checkpoint 分析显示训练无过拟合；推理对输入分辨率鲁棒（压缩仅 -0.4pp）。

---

## 1. 问题背景与动机

### 1.1 业务痛点

电商平台的商品属性数据是基础设施：货架属性决定商品如何被展示，搜索标签决定商品如何被召回，商品知识图谱决定关联推荐的质量。目前这些属性大量依赖**人工标注**，存在三个问题：

- **成本高**：全量人工标注覆盖海量 SKU 不现实，长尾商品覆盖差；
- **标准不一**：不同标注员对同一件衣服的袖长、面料判断不一致；
- **时效差**：上新速度快，人工标注跟不上。

### 1.2 通用 VLM 的局限

通用多模态大模型（如 Qwen3-VL）具备基础的图像理解能力，可以直接"看图说话"，但对**细粒度结构化属性抽取**存在系统性缺陷：

- **格式不稳定**：输出自由文本，不保证是合法 JSON，键名随意漂移；
- **属性混淆**：细粒度属性（medium-sleeve vs long-sleeve、cotton vs knitted、pure color vs color block）边界模糊，张冠李戴；
- **幻觉**：输出图上不可见的属性（NA/hidden 误报）。

### 1.3 方案定位

用领域数据对 VLM 做轻量 LoRA 微调，把"通用理解"变成"业务可用的结构化抽取能力"，并给出**完整的评测证据链**——不是"训完就完"，而是可量化、可归因、可复现的工程闭环。

---

## 2. 数据集与任务设计

### 2.1 数据集：DeepFashion-MultiModal

- **来源**：DeepFashion-MultiModal（Text2Human，SIGGRAPH 2022，[GitHub](https://github.com/yumingj/DeepFashion-MultiModal)），官方人工标注，无版权风险，可公开复现；
- **规模**：44,096 张 750×1101 全身人像图；
- **标注**：labels 含 12 个形状属性 + 面料/图案标注，构成 18 维属性体系。

### 2.2 属性体系（18 维 = 12 形状 + 3 部位×面料 + 3 部位×图案）

| 分组 | 属性键（示例） | 取值（官方口径，NA 已剔除） |
|---|---|---|
| 形状（12） | `sleeve_length` | sleeveless / short-sleeve / medium-sleeve / long-sleeve |
| | `neckline` | V-shape / square / round / standing / lapel / suspenders |
| | `hat` / `socks` / `glasses` / `neckwear` / `wrist_wearing` / `ring` 等 | no / yes（或 2~4 类） |
| 面料（3） | `{upper,lower,outer}_fabric` | denim / cotton / leather / furry / knitted / chiffon / other |
| 图案（3） | `{upper,lower,outer}_pattern` | floral / graphic / striped / pure color / lattice / other / color block |

要点：**不可见（NA）与 hidden 属性在数据转换时剔除**，训练/评测只针对可见属性；`outer_*` 仅在图中存在外衣时出现（每图实际输出 10~16 个键）。

### 2.3 任务形态：图 → 结构化属性 JSON

一次推理输出完整结构化属性，贴合真实电商抽取场景；输出为 JSON，与"图像 → 结构化输出"主线一致；评测可按属性键解构，严谨且细粒度。

### 2.4 schema 感知指令（方法核心）

训练/评测共用同一指令模板（`scripts/data_convert_deepfashion_multi.py::build_schema_prompt()` 自动生成，完整原文）：

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

结构：① 任务句（一句指令）；② **Allowed keys and values**——枚举全部 18 个属性键及每键合法取值（shape 12 键 + upper/lower/outer 三部位 × fabric/pattern 6 键）；③ **Rules** 三条输出规则（只输出可见属性 / 只用合法取值 / 只输出 JSON 不加解释）。指令由官方属性表自动生成，键与取值顺序稳定，保证训练/评测/推理三端完全一致。

设计动机：**键名/取值不漂移、格式错误率下降**——这是结构化输出任务中行业标准的工程做法，其收益将在 §5.2 由消融实验量化验证。对照版（simple）为朴素一句话指令："Extract the clothing attributes visible in this image and output them as JSON."

### 2.5 数据划分与防泄漏

- 按**图片粒度**划分 train/val/test = **80/10/10**（seed=42，三侧图片两两零重叠）——同一张图的所有样本同侧，防止泄漏；**test 仅用于最终评测，全程只碰一次**；
- 只保留可见（非 NA/non-hidden）属性；清洗空描述；图片按内容哈希去重；
- 数据质量四道关卡（结构健全 / 答案合法 / 防泄漏 / 分布统计）全部通过。

---

## 3. 方法

### 3.1 模型与微调配置

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

### 3.2 评测口径一致性（工程关键）

评测结果的可信度取决于**评测与训练的输入是否完全同口径**。本项目评测生成统一走 llamafactory 管线（与训练同模板 `qwen3_vl_nothink`、同多模态图像 token 展开、同数据格式、同解码参数 greedy），打分由自建 `eval_attr.py` 承担。这一决策源于一次实证发现：若评测改用 transformers 原生渲染（1 个图像占位符 vs 训练时的显式 6~7 个图像 token 展开），30 条实测会有 4 条输出漂移（属性级 99% 一致、文本 87% 一致）——**评测必须与训练同口径**。

---

## 4. 评测体系

### 4.1 指标定义

| 指标 | 口径 | 说明 |
|---|---|---|
| 属性级准确率 | 每个属性键 pred==true 比例 | 主指标；**预测缺键算错**（暴露漏检） |
| 属性级 macro-F1 | 多分类 F1 平均，缺键并入 `_missing` 类 | 对抗类别不均衡 |
| 全 JSON 匹配率 | 解析结果 == 真值 dict 精确一致 | 严格指标 |
| JSON 可解析率 | 输出能解析为 dict 的比例 | ≥98% 达标；格式错误单独统计 |

**容错解析三级策略**：① 提取首个 `{...}` 区间 → `json.loads`；② 失败则从右截断至最后一个完整键值对；③ 仍失败记「格式错误」。

### 4.2 对比纪律与统计检验

- **对比组**：Base-0shot（基座 + 同一 schema 指令） vs LoRA 微调后——评测代码、数据集、prompt 三者完全一致，**唯一变量是"是否微调"**；
- **McNemar 检验**（配对二分类、连续校正）：逐属性 + 总体，p<0.05 才下结论；
- **分层统计**：按属性组（形状/面料图案）、按类别数（二分类/多分类）分别报告，避免被 majority 类掩盖。

---

## 5. 实验结果

> 本章按**论证链**组织实验：先证明**效果**（§5.1 微调是否有效），再**归因**（§5.2 提升来自哪里），再划定**边界**（§5.3 效果依赖什么、替代方案如何），最后检验**过程**（§5.4 训练是否稳定），以汇总表收尾（§5.5）。

### 5.1 效果论证：微调是否有效

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

### 5.2 归因论证：提升来自哪里

主结果回答"是否有效"，本节回答"为什么有效"——把 +34pp 的提升拆解为可解释的来源。消融在 val 集（4,410 条）评测，按**指令范式（schema/simple）× 数据量（5k/20k/35k）**二维展开，每次对比保持**唯一变量**。

同一组 6 个训练配置构成如下二维矩阵（val 集属性准确率），**横向（行内）与纵向（列内）各持一个唯一变量**：

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

**沿指令轴（schema vs simple）——指令范式是决定性方法**：相同数据与训练配置下，枚举合法键/取值的指令始终显著优于朴素指令。优势随数据量增大而收窄（+31.5pp→+18.1pp），但朴素指令用全量数据（35k）也只有 75.3%，仍不及 schema 指令用 5k 数据（91.9%）的水平；macro-F1 同样全面落后（补充指标表中全量 0.670 vs 0.518），且 simple 版在三个数据量下全 JSON 匹配率恒为 0.0%（输出从不按 18 维 schema 对齐）。**指令设计决定了结构化输出的成败**——这是任务工程化的方法论收益，直接支撑 §2.4 的设计动机。schema 指令的重要性不止于训练期：推理期若把评测指令替换为 simple，效果同样崩塌（见 §5.3.1）。

**沿数据轴（5k→20k→35k）——数据效率极高、堆数据事倍功半**：schema 曲线平缓——数据量从 5k 增至 35.3k 属性准确率仅 +1.5pp，且收益集中在前 20k（5k→20k +1.0pp、全 JSON 匹配率 +5.8pp；20k→35k 边际递减 +0.5pp/+2.2pp，全 JSON 匹配率见补充指标表），更多数据主要帮助**整段 JSON 的完整性**（少漏键、少缺属性），schema 指令下模型数据效率极高（5k 即达 92% 水平）；simple 曲线陡峭——数据收益 +14.9pp 几乎全被"追赶指令差距"消耗（20k 即 +13.6pp 后趋平），说明没有好的指令设计，堆数据事倍功半。

**归因小结**：提升主要来自**任务工程化（schema 指令设计）**而非数据堆量，双向验证：schema 用 1/7 数据（5k）拿到 91.9%，simple 用全量数据（35k）只有 75.3%——这与"电商标注贵"的业务痛点直接呼应：**好的任务设计可以用很少的数据拿到大部分收益**。

### 5.3 边界论证：效果依赖与鲁棒性

主结果与归因证明"微调有效、来自方法"，本节划定这套效果的**适用边界**——什么条件下成立、什么替代方案不行、对输入是否敏感。三个对照均在 val 集评测，模型为同一最终模型（或基座）。

**5.3.1 运行期依赖：指令鲁棒性（同一 LoRA 模型，仅换评测指令）**

本节与 §5.2 指令轴互补：§5.2 更换的是**训练指令**（重训两组模型，测得方法收益 +18~31pp），本节更换的是**推理指令**（模型不变，测得运行依赖）——同一变量（schema/simple）在训练期与运行期的两个作用阶段。

| 模型×指令 | 属性准确率 | parse | 全 JSON 匹配 |
|---|---|---|---|
| LoRA 主模型（schema） | **93.4%** | 100.0% | 45.8% |
| LoRA 主模型 + simple prompt | 0.2% | 99.7% | 0.0% |

在 simple 朴素指令下，LoRA 模型输出**退化为自由格式 JSON**（键名如 `top.type`、`clothing[].color`，不再是 18 维属性键名），按任务 schema 对齐后属性级准确率仅 0.2%（parse 率 99.7% 说明是合法 JSON 但结构不对齐）。**schema 指令是运行期必需**：业务部署必须携带 schema prompt。这与 §5.2 指令轴解读互相印证——训练期"有它 +31.5~18.1pp"（方法收益，随数据量收窄但始终显著），推理期"没它 -93.2pp"（运行依赖）。

**5.3.2 替代方案：few-shot 对照（基座 + 2 个 in-context 示例）**

| 模型×指令 | 属性准确率 | 全 JSON 匹配 |
|---|---|---|
| 基座 0-shot（schema） | 59.9% | 0.0% |
| 基座 2-shot（schema + 2 示例） | 72.3% | 0.1% |
| LoRA 微调（schema） | 93.4% | 45.8% |

基座 2-shot 较 0-shot 提升 +12.4pp（in-context 示例确有帮助），但**仍比微调低 21.1pp**，且每次推理需额外携带示例输入——用数据回答"为什么微调而不 few-shot"：few-shot 无法达到结构化抽取的业务可用水平。

**5.3.3 输入鲁棒性：图像分辨率（同一模型，评测输入压缩 vs 不压缩）**

| image_max_pixels | 属性准确率 | 全 JSON 匹配 |
|---|---|---|
| 825,750（不压缩，与训练一致） | **93.4%** | 45.8% |
| 589,824（压缩） | 93.0% | 44.0% |
| Δ | **-0.4pp** | -1.8pp |

评测输入压缩到 768×768 仅造成 -0.4pp 损失——模型对输入分辨率鲁棒；评测口径维持与训练一致（不压缩）。

**边界小结**：效果成立的前提是 **schema 指令在场**（训练与部署都要带）；few-shot 不是可行替代；输入分辨率不是敏感因素。

### 5.4 过程论证：训练稳定性与模型选择

对训练全程 28 个 checkpoint 抽样 15 个点，在 val 上评测（500 条快筛 + 关键点全量确认）：

```
100步  90.4%   300步 91.3%   500步 92.1%   700步 92.5%
900步  93.8%  ← 峰值（约 1 epoch）
1100步 92.6%   1300步 92.8%   ...   2100步 93.8%（第二峰值）
2500步 93.4%   2793步 93.4%（最终）
```

全量确认（4,410 条 val）：ckpt-900 93.3% / ckpt-2100 93.4% / **ckpt-2793（最终）93.4%（macro-F1 0.670 最优）**。

**结论**：① LoRA 收敛极快——训练 100 步（约 3% 数据）即达 90.4%；② 全程无过拟合——曲线在 92.5%~93.8% 平台波动、不下降，3 epoch 是稳妥选择；③ 最终 checkpoint 在 macro-F1 与全 JSON 匹配上略优，用作 test 定稿模型——模型选择有据可依，不是拍脑袋。

### 5.5 实验汇总（按论证环节）

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


## 6. 错误分析（badcase）

对 val 集 4,410 条预测中 LoRA 模型的错误进行统计（4,410 条中 2,391 条未达全 JSON 精确匹配），错误模式如下：

### 6.1 错误分布：细粒度面料/图案为主

错误最多的属性集中在 texture 组：`lower_fabric`（440 次）、`upper_fabric`（394 次）、`upper_pattern`（333 次）、`lower_pattern`（276 次）——材质与图案是视觉上最难区分的细粒度维度。

### 6.2 典型混淆对（真值 → 预测）

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

### 6.3 归因与改进方向

1. **类别混淆**（面料/图案/袖长）：细粒度边界样本不足 → 补充困难样本 / 类别加权训练；
2. **小物件漏检**（戒指/领饰）：目标小、易遮挡 → 提升图像分辨率 / 显式提示"注意配饰" / 视觉塔微调（当前冻结）；
3. **输出完整性**（漏键）：全 JSON 匹配率 47.3% 的主因 → 模板化输出 / 约束解码 / 补训练样本。

这些模式同时构成后续迭代的优先级依据（"改 → 训 → 评"闭环）。

---

## 7. 结论与展望

### 7.1 结论

1. **任务达成**：LoRA 微调将服装属性抽取的属性级准确率从 59.8% 提升至 93.8%（+34.0pp），macro-F1 0.677，18 属性全部通过 McNemar 显著性检验，JSON 可解析率 100%——微调收益远超常见预期（细粒度任务上通用 VLM 的 0-shot 能力确实弱，领域微调空间巨大）；
2. **方法论价值**：schema 感知指令是决定性因素（+31.5pp），数据效率高（5k 即达 92%），训练稳定无过拟合——这套"任务工程化 + 轻量微调 + 严谨评测"的组合可以直接迁移到其他结构化抽取场景（商品属性、证件信息、表格抽取等）；
3. **工程闭环完整**：数据（防泄漏划分 + 质量关卡）→ 训练（LoRA + 口径一致）→ 评测（自建指标 + 统计检验 + 消融）→ 分析（badcase 归因）→ 部署（vLLM demo），可复现、可讲述。

### 7.2 局限

- 细粒度面料/图案准确率（88.6%）仍低于形状属性（96.3%），长尾类别（如 `furry`、`suspenders`）样本不足；
- 小物件（戒指、领饰）漏检率偏高；
- 全 JSON 精确匹配率 47.3% 仍有提升空间（任何单属性错误即判不匹配）。

### 7.3 展望

- **数据侧**：针对混淆对补充困难样本、类别重加权；
- **模型侧**：解冻视觉塔（对照实验）、约束解码保证输出 schema 合法性；
- **任务侧**：扩展到多任务（属性 + 描述）、多部位细粒度分割辅助；
- **部署侧**：vLLM 服务化 + 线上 A/B 验证真实电商货架属性收益。

---

## 附录：复现指引

| 环节 | 入口 |
|---|---|
| 数据转换 | `python scripts/data_convert_deepfashion_multi.py` → `data/data_convert/deepfashion_multi/` |
| 数据质量 | `python scripts/check_data.py`（四关卡） |
| 训练 | LlamaBoard 导入 `llamafactory/config/qwen3vl8b_dfattr_train.yaml`（3 卡 DDP） |
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
