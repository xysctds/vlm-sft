# vendor/ 说明

> **类型**：第三方代码与资料清单　|　**读者**：要重新获取上游代码或查资料的人　|　**更新**：2026-10-20
>
> 本目录只放**别人写的、我们只读**的东西。内容不入库（见 `../.gitignore`），删了可按本文件重新获取。

## 清单

| 目录 | 内容 | 来源 | 版本 |
|---|---|---|---|
| `llamafactory/` | LLaMA-Factory 训练框架（`config/` 下的 yaml 直接喂给它） | <https://github.com/hiyouga/LlamaFactory.git> | 自带 `.git`，见该目录 |
| `minimind-v/` | MiniMind-V 多模态模型与数据（12G：`checkpoints/`、`dataset/`） | <https://github.com/jingyaogong/minimind-v> | 自带 `.git`，见该目录 |
| `ms-swift/` | ms-swift 相关资料：`Qwen3+VL 微调.pdf` 与配套脚本摘录 | 第三方教程资料 | — |

## 使用方式

- **训练/导出**：`../experiments/*/scripts/` 直接调用 `llamafactory` 的 CLI，yaml 在 `../experiments/*/config/`。
- **minimind-v**：早期对照实验使用，见 `../scratch/` 的说明。
- **ms-swift**：只作阅读参考，未接入主线流程。

## 约定

1. **不在这里改代码**。需要改动时，在 `../experiments/` 或 `../scratch/` 里包装。
2. `llamafactory/` 与 `minimind-v/` 各自的 `.git` 保留原样，不要在本项目里对它们提交。
3. 新增上游依赖时，在本表登记来源与版本。
