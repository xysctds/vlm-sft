#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DeepFashion 属性抽取 Gradio Web demo（浏览器传图 → 结构化属性 JSON）。

依赖 vLLM 服务已启动（完整命令见 docs/部署说明.md）::

    vllm serve llamafactory/saves/dfattr/merged/qwen3vl8b_dfattr_merged \\
        --served-model-name deepfashion --port 8000 --max-model-len 32768

启动本页面（跑在 **lf** 环境：gradio 5.50 与 starlette 0.52 兼容，vllm 环境里两者版本冲突）::

    conda activate lf
    python scripts/serve_gradio.py          # 浏览器打开 http://127.0.0.1:7860

本脚本只是 vLLM 的 HTTP 客户端，不需要 vllm 包。
"""

import base64
import json
import mimetypes
import os
import sys

import gradio as gr
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_convert_deepfashion_multi import SIMPLE_JSON_PROMPT, build_schema_prompt  # noqa: E402

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "deepfashion"
MAX_TOKENS = 512


def _parse_json(content: str) -> dict:
    """从模型输出中容错提取 JSON（与 eval_attr.py 同思路）。"""
    start, end = content.find("{"), content.rfind("}")
    return json.loads(content[start:end + 1])


def predict(image_path: str, prompt: str) -> tuple[str, dict]:
    """传一张服装图 + prompt，返回（模型原始输出, 解析后的属性 JSON）。"""
    with open(image_path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode()
    mime = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            {"type": "text", "text": prompt},
        ]}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
    }
    resp = requests.post(URL, json=payload, timeout=120)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    try:
        return content, _parse_json(content)
    except (ValueError, json.JSONDecodeError):
        return content, {"error": "模型输出无法解析为 JSON", "raw": content}


def on_style_change(style: str) -> str:
    """切换 schema/simple 模板时，把对应 prompt 填进可编辑框。"""
    return build_schema_prompt() if style == "schema" else SIMPLE_JSON_PROMPT


def main() -> None:
    with gr.Blocks(title="DeepFashion 服装属性抽取 Demo") as demo:
        with gr.Row():
            with gr.Column(scale=1):
                img = gr.Image(type="filepath", label="上传服装全身图", height=300)
                with gr.Row():
                    style = gr.Radio(["schema", "simple"], value="schema", label="模板")
                    btn = gr.Button("抽取属性", variant="primary")
                prompt_box = gr.Textbox(
                    value=build_schema_prompt(),
                    label="Prompt（可手动编辑）", lines=5, max_lines=10,
                )
            with gr.Column(scale=1):
                raw = gr.Textbox(label="模型原始输出", lines=5, max_lines=12)
                parsed = gr.JSON(label="解析后的结构化属性", max_height=None)
        style.change(fn=on_style_change, inputs=style, outputs=prompt_box)
        btn.click(fn=predict, inputs=[img, prompt_box], outputs=[raw, parsed])
    demo.launch(server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    main()
