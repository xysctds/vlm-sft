# DeepFashion-MultiModal 属性对照表（英中）

> 来源：DeepFashion-MultiModal（Text2Human, SIGGRAPH 2022）官方 README
> （https://github.com/yumingj/DeepFashion-MultiModal）
> 属性定义内嵌于 `scripts/data_convert_deepfashion_multi.py`（`SHAPE_ATTRS` / `FABRIC_VALS` / `PATTERN_VALS`）。
>
> 共 **18 维属性 = 12 个形状属性 + 3 个部位（上装/下装/外装）× 面料 + 3 个部位 × 图案**。

## 一、形状属性（12 个，来自 `labels/shape/shape_anno_all.txt`）

| 属性键（英文） | 属性键（中文） | 取值（官方英文） | 取值（中文） |
|---|---|---|---|
| `sleeve_length` | 袖长 | sleeveless / short-sleeve / medium-sleeve / long-sleeve / not long-sleeve / NA | 无袖 / 短袖 / 中袖 / 长袖 / 非长袖 / 不可见 |
| `lower_clothing_length` | 下装长度 | three-point / medium short / three-quarter / long / NA | 三分长度（短款）/ 中短 / 七分 / 长 / 不可见 |
| `socks` | 袜子 | no / socks / leggings / NA | 无 / 袜子 / 打底裤（紧身裤）/ 不可见 |
| `hat` | 帽子 | no / yes / NA | 无 / 有 / 不可见 |
| `glasses` | 眼镜 | no / eyeglasses / sunglasses / glasses in hand or on clothes / NA | 无 / 框架眼镜 / 太阳镜 / 拿在手中或挂在衣服上 / 不可见 |
| `neckwear` | 颈饰（围巾/项链等） | no / yes / NA | 无 / 有 / 不可见 |
| `wrist_wearing` | 手腕佩戴物 | no / yes / NA | 无 / 有 / 不可见 |
| `ring` | 戒指 | no / yes / NA | 无 / 有 / 不可见 |
| `waist_accessories` | 腰部配饰 | no / belt / clothing at waist / hidden / NA | 无 / 腰带 / 系在腰间的衣物 / 被遮挡 / 不可见 |
| `neckline` | 领型 | V-shape / square / round / standing / lapel / suspenders / NA | V 领 / 方领 / 圆领 / 立领 / 翻领 / 背带（吊带）/ 不可见 |
| `outer_clothing_is_cardigan` | 外衣是否为开衫 | yes / no / NA | 是 / 否 / 不可见 |
| `upper_clothing_covering_navel` | 上衣是否遮住肚脐 | no / yes / NA | 否（露脐）/ 是 / 不可见 |

## 二、面料属性（`labels/texture/fabric_ann.txt`，上/下/外装各一列）

| 属性键（英文） | 属性键（中文） | 取值（官方英文） | 取值（中文） |
|---|---|---|---|
| `upper_fabric` | 上衣面料 | | |
| `lower_fabric` | 下装面料 | denim / cotton / leather / furry / knitted / chiffon / other / NA | 牛仔 / 棉 / 皮革 / 毛绒 / 针织 / 雪纺 / 其他 / 不可见 |
| `outer_fabric` | 外衣面料 | | |

## 三、图案属性（`labels/texture/pattern_ann.txt`，上/下/外装各一列）

| 属性键（英文） | 属性键（中文） | 取值（官方英文） | 取值（中文） |
|---|---|---|---|
| `upper_pattern` | 上衣图案 | | |
| `lower_pattern` | 下装图案 | floral / graphic / striped / pure color / lattice / other / color block / NA | 碎花 / 图形 / 条纹 / 纯色 / 格纹 / 其他 / 色块 / 不可见 |
| `outer_pattern` | 外衣图案 | | |

## 补充说明

- `NA` 表示该属性在图中不可见（如没穿外衣时 `outer_*` 为 NA）；`waist_accessories` 的 `hidden` 表示腰部被遮挡、无法判断。
- 转换脚本将 **`NA` 和 `hidden` 均视为无效值剔除**（`NA_VALUES = {"NA", "hidden"}`），训练/评测只针对可见属性，因此每张图实际输出的键约 10~16 个。
- 每个属性键对应一个官方问句模板（如 `sleeve_length` → "What is the sleeve length of the clothing?"），见脚本 `SHAPE_QUESTIONS` / `PART_QUESTIONS`。
