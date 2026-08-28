#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DeepFashion 属性抽取 Gradio Web demo（浏览器传图 → 结构化属性 JSON）。

依赖 vLLM 服务已启动（完整命令见 docs/部署说明.md）::

    vllm serve llamafactory/saves/dfattr/merged/qwen3vl8b_dfattr_merged \\
        --served-model-name deepfashion --port 8000 --max-model-len 32768

启动本页面（跑在 **vllm** 环境：部署环境已重建为纯部署用途，gradio 6 与 starlette 1.6 完全兼容）::

    conda activate vllm
    python scripts/serve_gradio.py          # 浏览器打开 http://127.0.0.1:7860

本脚本只是 vLLM 的 HTTP 客户端，不依赖 llamafactory（vllm 环境已卸载之）；页面检索所用 embedding 由本环境用与 lf 同版本的 transformers 抽取，口径一致。
"""

import base64
import json
import mimetypes
import os
import sys

import gradio as gr
import requests

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPTS_DIR)  # 仓库根
sys.path.insert(0, SCRIPTS_DIR)
os.chdir(BASE_DIR)  # 以仓库根为工作目录（脚本内 data/、models/ 等相对路径均依赖此，任意目录启动均可）
from data_convert_deepfashion_multi import SIMPLE_JSON_PROMPT, build_schema_prompt  # noqa: E402

try:
    from retrieval import RetrievalIndex  # noqa: E402
    _RETRIEVAL_AVAILABLE = True
except Exception:
    _RETRIEVAL_AVAILABLE = False  # 未建索引时 demo 仍可单独跑属性抽取

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "deepfashion"
MAX_TOKENS = 512
RETRIEVAL_INDEX = "data/data_convert/deepfashion_multi/retrieval"
IMAGES_DIR = os.path.join(BASE_DIR, "data", "data_convert", "deepfashion_multi", "images")

_retrieval_idx = None


def _get_retrieval_index():
    global _retrieval_idx
    if _retrieval_idx is None and _RETRIEVAL_AVAILABLE:
        _retrieval_idx = RetrievalIndex(RETRIEVAL_INDEX)
    return _retrieval_idx


def embed_query_image(image_path: str):
    """对查询图抽归一化视觉向量。

    逻辑收敛在 retrieval.embed_query（内部含模型单例缓存，首次加载后复用，
    避免每次检索重复加载 17GB 模型）。embedding 口径与索引构建一致。
    建议启动时用 CUDA_VISIBLE_DEVICES 指定查询模型所在卡，避免与 vLLM 撞 GPU。
    """
    from retrieval import embed_query
    return embed_query("models/Qwen3-VL-8B-Instruct", image_path)


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
        attrs = _parse_json(content)
    except (ValueError, json.JSONDecodeError):
        attrs = {"error": "模型输出无法解析为 JSON", "raw": content}
    return content, attrs


def predict_with_retrieval(image_path: str, prompt: str, topk: int = 5):
    """抽取属性 + 检索相似款，返回（原始输出, 属性JSON, 相似款图列表, 相似款详情）。"""
    content, attrs = predict(image_path, prompt)
    sim_imgs, sim_info = [], []
    try:
        idx = _get_retrieval_index()
    except Exception as exc:  # 索引加载失败（如未构建/路径问题）时优雅降级，不影响属性抽取
        idx = None
        sim_info = [{"error": f"检索索引加载失败：{exc}"}]
    if idx is not None and isinstance(attrs, dict) and "error" not in attrs:
        try:
            qv = embed_query_image(image_path)
            results = idx.search(qv, attrs, mode="hybrid", topk=topk)
            for r in results:
                p = os.path.join("data/data_convert/deepfashion_multi/images", r["image"])
                # Gallery 用 (图, caption) 元组，把综合匹配分直接标在图上
                caption = f"id={r['id']} · 匹配 {r['final']:.2f}"
                sim_imgs.append((p, caption))
                sim_info.append({
                    "id": r["id"], "image": r["image"], "angle": r["angle"],
                    "vis": round(r["vis_cos"], 3), "attr": round(r["attr_score"], 2),
                    "final": round(r["final"], 3), "attrs": r["attrs"],
                })
        except Exception as exc:  # 检索失败不影响属性抽取结果
            sim_info = [{"error": str(exc)}]
    return content, attrs, sim_imgs, sim_info


def on_style_change(style: str) -> str:
    """切换 schema/simple 模板时，把对应 prompt 填进可编辑框。"""
    return build_schema_prompt() if style == "schema" else SIMPLE_JSON_PROMPT


def _preload_query_model():
    """后台预热视觉塔，让用户第一次点检索时不用等模型加载（方案2）。"""
    import threading
    from retrieval import preload_vision_tower

    def _load():
        try:
            preload_vision_tower()
        except Exception as exc:  # 预热失败不影响 demo 启动，首次点击时再加载
            print(f"[warn] 视觉塔预热失败：{exc}")

    t = threading.Thread(target=_load, daemon=True)
    t.start()


def main() -> None:
    _preload_query_model()  # 启动即后台预热查询模型
    with gr.Blocks(title="DeepFashion 服装属性抽取 + 相似款检索 Demo") as demo:
        with gr.Row():
            with gr.Column(scale=1):
                img = gr.Image(type="filepath", label="上传服装全身图", height=300)
                with gr.Row():
                    style = gr.Radio(["schema", "simple"], value="schema", label="模板")
                    btn = gr.Button("抽取属性 + 检索相似款", variant="primary")
                topk = gr.Slider(3, 9, value=5, step=1, label="相似款数量")
                prompt_box = gr.Textbox(
                    value=build_schema_prompt(),
                    label="Prompt（可手动编辑）", lines=5, max_lines=10,
                )
            with gr.Column(scale=1):
                raw = gr.Textbox(label="模型原始输出", lines=5, max_lines=12)
                parsed = gr.JSON(label="解析后的结构化属性", max_height=None)
                sim_gallery = gr.Gallery(
                    label="相似款（视觉+属性混合检索）", columns=5, height=260,
                    object_fit="cover",
                )
                sim_info = gr.JSON(label="相似款详情（id / 匹配分 / 官方属性）")
        style.change(fn=on_style_change, inputs=style, outputs=prompt_box)
        btn.click(
            fn=predict_with_retrieval,
            inputs=[img, prompt_box, topk],
            outputs=[raw, parsed, sim_gallery, sim_info],
        )
    demo.launch(server_name="127.0.0.1", server_port=7860, allowed_paths=[IMAGES_DIR],
                show_error=True)


if __name__ == "__main__":
    main()
