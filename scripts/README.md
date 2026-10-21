# 脚本索引

跨实验共用的脚本。实验专属脚本各随其域，见 `experiments/*/*/scripts/`。
所有命令都在仓库根目录执行。

## 数据

| 脚本 | 作用 |
|---|---|
| `data_convert_deepfashion_multi.py` | DeepFashion-MultiModal → LLaMA-Factory 训练格式（schema 主线 + simple 对照 + 描述 + 属性 QA） |
| `check_data.py` | 数据质量四关卡：结构 / 答案合法 / 防泄漏 / 分布统计 |

## 评测

| 脚本 | 作用 |
|---|---|
| `eval_attr.py` | 打分 `stats`：属性级 acc / macro-F1 / 分层 / McNemar，容错解析 |
| `convert_preds.py` | 把 llamafactory 的 `generated_predictions.jsonl` 转成 `eval_attr.py stats` 可消费的格式 |

## 检索

| 脚本 | 作用 |
|---|---|
| `build_retrieval_index.py` | 建检索索引（官方标注聚合 + 冻结视觉塔抽 embedding）→ `index_vectors.npy` + `index_meta.json` |
| `retrieval.py` | 相似款检索（视觉余弦 + 官方属性命中率混合加权，按套聚合 + 同款排重），含库内自检 |

## 部署与 Demo

| 脚本 | 作用 |
|---|---|
| `start_vllm.sh` | vLLM 一键启动（封装 CUDA_HOME / flashinfer 禁用等环境准备，单卡 / 多卡切换） |
| `start_web.sh` | Web demo 一键启动（vLLM + Gradio 全自动：GPU 分卡避让、健康检查、`--stop` / `--status`） |
| `serve_demo.py` | demo 客户端（CLI，走 vLLM OpenAI 兼容接口） |
| `serve_gradio.py` | Web 页面（Gradio，模板切换 + prompt 手动编辑，浏览器传图看结构化 JSON + 相似款检索） |

参数与完整流程见[复现说明](../docs/复现说明.md)与[部署说明](../docs/部署说明.md)。
