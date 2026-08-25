# 指令范式消融（val 全量 4410 条）

| 模型 | n | attr acc | macro-F1 | parse | full-match |
|---|---|---|---|---|---|
| 主模型 schema-35k（全量） | 4410 | 93.4% | 0.670 | 100.0% | 45.8% |
| schema-5k | 4410 | 91.9% | 0.631 | 100.0% | 37.8% |
| simple-5k | 4410 | 60.4% | 0.425 | 100.0% | 0.0% |

- schema-5k vs simple-5k（同数据量，唯一变量=指令范式）：Δ attr acc = +31.5%
- schema-5k vs 主模型 schema-35k（同指令，唯一变量=数据量）：Δ attr acc = -1.5%
