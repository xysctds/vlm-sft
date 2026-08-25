# 补充实验：指令鲁棒性 + few-shot 对照（val 全量 4410 条）

- generated: 2026-08-25 17:12:52
- 指令鲁棒性：同一 LoRA 主模型，仅替换评测指令（schema vs simple）

| 模型×指令 | attr acc | macro-F1 | parse | full-match |
|---|---|---|---|---|
| 基座 0-shot（schema） | 59.9% | 0.412 | 100.0% | 0.0% |
| 基座 2-shot（schema+示例） | 72.3% | 0.445 | 100.0% | 0.1% |
| LoRA 主模型（schema） | 93.4% | 0.670 | 100.0% | 45.8% |
| LoRA 主模型 + simple prompt | 0.2% | 0.000 | 99.7% | 0.0% |

- 指令鲁棒性：LoRA 模型在 simple 指令下 0.2% vs schema 指令 93.4%（Δ -93.2%）；
- few-shot 价值：基座 2-shot 72.3% vs 基座 0-shot 59.9%（Δ +12.4%），仍远低于微调 93.4%。

## 观察

- **指令依赖度**：LoRA 模型在 simple 指令下输出**退化为自由格式 JSON**（键名如 `top.type`/`clothing[].color`，不再是 18 维属性键名），按任务 schema 对齐后属性级准确率仅 0.2%（parse 率 99.7% 说明是合法 JSON 但结构不对齐）——**schema 指令是运行期必需**：业务部署必须携带 schema prompt，这也反向验证了训练期 schema 指令的价值；
- **few-shot vs 微调**：基座 2-shot（72.3%）较 0-shot（59.9%）提升 +12.4%pp，说明 in-context 示例有帮助，但**仍比微调（93.4%）低 21.1%pp**——用数据回答"为什么微调而不 few-shot"：few-shot 无法达到结构化抽取的业务可用水平，且每次推理多 2 个示例的输入开销。
