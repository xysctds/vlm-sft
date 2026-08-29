#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DeepFashion 服装属性抽取 + 相似款检索 Gradio Web demo。

依赖 vLLM 服务已启动（完整命令见 docs/部署说明.md）::

    vllm serve llamafactory/saves/dfattr/merged/qwen3vl8b_dfattr_merged \\
        --served-model-name deepfashion --port 8000 --max-model-len 32768

启动本页面（跑在 **vllm** 环境：部署环境已重建为纯部署用途，gradio 6 与 starlette 1.6 完全兼容）::

    conda activate vllm
    python scripts/serve_gradio.py          # 浏览器打开 http://127.0.0.1:7860

本脚本只是 vLLM 的 HTTP 客户端，不依赖 llamafactory（vllm 环境已卸载之）；页面检索所用 embedding
由本环境用与 lf 同版本的 transformers 抽取，口径一致。检索索引可用性、Embedding 预热状态会在顶部
信息栏展示，方便现场演示时交代系统构成。
"""

import base64
import json
import mimetypes
import os
import random
import sys
import time

import gradio as gr
import requests

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPTS_DIR)  # 仓库根
sys.path.insert(0, SCRIPTS_DIR)
os.chdir(BASE_DIR)  # 以仓库根为工作目录（脚本内 data/、models/ 等相对路径均依赖此，任意目录启动均可）
from data_convert_deepfashion_multi import SIMPLE_JSON_PROMPT, build_schema_prompt  # noqa: E402

try:
    from retrieval import RetrievalIndex  # noqa: E402
    from retrieval import _ordered_attrs  # noqa: E402
    _RETRIEVAL_AVAILABLE = True
except Exception:
    _RETRIEVAL_AVAILABLE = False  # 未建索引时 demo 仍可单独跑属性抽取

    def _ordered_attrs(attrs):  # 无检索时兜底：原样返回
        return attrs

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "deepfashion"
MODEL_DISPLAY = "Qwen3-VL-8B (vLLM)"
MAX_TOKENS = 512
RETRIEVAL_INDEX = "data/data_convert/deepfashion_multi/retrieval"
IMAGES_DIR = os.path.join(BASE_DIR, "data", "data_convert", "deepfashion_multi", "images")

# 供 gr.Examples 快速演示用的示例图：每次启动从图片库随机抽几张，方便现场换着演示。
def _sample_examples(n: int = 6) -> list[str]:
    """随机抽 n 张图片作为示例图（绝对路径，任意目录启动均可访问）。"""
    try:
        files = [f for f in os.listdir(IMAGES_DIR)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    except Exception:
        return []
    if not files:
        return []
    picks = random.sample(files, min(n, len(files)))
    return [os.path.join(IMAGES_DIR, f) for f in picks]

# 「该款信息」要展示的字段（含三个相似度 + 属性）。attrs 顺序由 retrieval._ordered_attrs 统一规范化。
INFO_KEYS = ["id", "image", "angle", "vis", "attr", "final", "attrs"]

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


def _search_similar(image_path: str, attrs: dict, topk: int = 5):
    """用查询图视觉特征 + 抽取的属性做混合检索，返回（相似款图列表, 相似款详情记录）。

    Gallery 用 (图, caption) 元组，caption 为「#短id · 匹配分」，随图显示；记录用于下方该款信息。
    """
    sim_imgs, records = [], []
    try:
        idx = _get_retrieval_index()
    except Exception as exc:  # 索引加载失败（如未构建/路径问题）时优雅降级，不影响属性抽取
        idx = None
        records = [{"error": f"检索索引加载失败：{exc}"}]
    if idx is not None and isinstance(attrs, dict) and "error" not in attrs:
        try:
            qv = embed_query_image(image_path)
            results = idx.search(qv, attrs, mode="hybrid", topk=topk)
            for r in results:
                p = os.path.join("data/data_convert/deepfashion_multi/images", r["image"])
                # r['id'] 是纯 8 位数字（如 00000001），去掉前导零后更短：00000001 -> 1
                raw = str(r["id"])
                short = str(int(raw)) if raw.isdigit() else raw
                caption = f"#{short}\n匹配 {r['final']:.2f}"
                sim_imgs.append((p, caption))
                records.append({
                    "id": short, "image": r["image"], "angle": r["angle"],
                    "vis": round(r["vis_cos"], 3), "attr": round(r["attr_score"], 2),
                    "final": round(r["final"], 3), "attrs": r["attrs"],
                })
        except Exception as exc:  # 检索失败不影响属性抽取结果
            records = [{"error": str(exc)}]
    return sim_imgs, records


def _info_of(rec: dict) -> dict:
    """把一条记录转成「该款信息」JSON（含三个相似度 + 属性，attrs 按标准顺序重排）。"""
    if not isinstance(rec, dict):
        return {}
    info = {k: rec.get(k) for k in INFO_KEYS}
    info["attrs"] = rec.get("attrs")  # 已在 retrieval 层按标准顺序排好
    return info


def _first_record(records):
    """取第一条有效记录（跳过 error 记录），无则返回空 dict。"""
    if not records:
        return {}
    for r in records:
        if isinstance(r, dict) and "error" not in r:
            return r
    return {}


def extract_attrs(image_path: str, prompt: str) -> tuple[dict, dict]:
    """只抽取属性，返回（属性 JSON, 耗时指标）。attrs 按标准顺序重排，与相似款一致。"""
    t0 = time.perf_counter()
    _, attrs = predict(image_path, prompt)
    elapsed = time.perf_counter() - t0
    perf = {"模型调用(s)": round(elapsed, 3), "总耗时(s)": round(elapsed, 3)}
    return _ordered_attrs(attrs), perf


def retrieve_similar(image_path: str, prompt: str, topk: int = 5):
    """只检索相似款，返回（画廊, 记录, 该款信息, 耗时指标）。"""
    t0 = time.perf_counter()
    _, attrs = predict(image_path, prompt)
    t1 = time.perf_counter()
    sim_imgs, records = _search_similar(image_path, attrs, topk)
    t2 = time.perf_counter()
    perf = {
        "模型调用(s)": round(t1 - t0, 3),
        "检索(s)": round(t2 - t1, 3),
        "总耗时(s)": round(t2 - t0, 3),
    }
    return gr.update(value=sim_imgs), records, _info_of(_first_record(records)), perf


def predict_with_retrieval(image_path: str, prompt: str, topk: int = 5):
    """一键：先抽取属性，再检索相似款，返回（属性JSON, 画廊, 记录, 该款信息, 耗时指标）。"""
    t0 = time.perf_counter()
    _, attrs = predict(image_path, prompt)
    t1 = time.perf_counter()
    sim_imgs, records = _search_similar(image_path, attrs, topk)
    t2 = time.perf_counter()
    perf = {
        "模型调用(s)": round(t1 - t0, 3),
        "检索(s)": round(t2 - t1, 3),
        "总耗时(s)": round(t2 - t0, 3),
    }
    return _ordered_attrs(attrs), gr.update(value=sim_imgs), records, _info_of(_first_record(records)), perf


def on_gallery_select(evt: gr.SelectData, records: list):
    """点击某张相似款缩略图（含预览选中时），更新下方「该款信息」。"""
    if not records:
        return {}
    i = evt.index if evt is not None else 0
    if isinstance(records, list) and 0 <= i < len(records):
        rec = records[i]
    else:
        rec = records[0]
    if not isinstance(rec, dict) or "error" in rec:
        rec = {}
    return _info_of(rec)


def on_style_change(style: str) -> str:
    """切换 schema/simple 模板时，把对应 prompt 填进可编辑框。"""
    return build_schema_prompt() if style == "schema" else SIMPLE_JSON_PROMPT


def _build_info() -> str:
    """构造顶部信息栏 Markdown：最简洁地交代「模型 + 检索可用性 + 预热」，方便现场一句话带过。"""
    model = MODEL_DISPLAY
    idx_status = "✅ 就绪" if _RETRIEVAL_AVAILABLE else "⚠ 未就绪（仅属性抽取）"
    return (f"**模型** {model} &nbsp;·&nbsp; **检索索引** {idx_status} "
            f"&nbsp;·&nbsp; **视觉 Embedding** 自动预热")


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


# 主题：Soft 基调 + 少量自定义样式，告别默认灰。
THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="indigo",
    neutral_hue="slate",
)
CSS = """
.gradio-container { width: min(1500px, 96vw) !important; margin: 0 auto !important;
                    overflow-x: hidden !important; }
.gradio-container .wrap, .gradio-container .block { min-width: 0 !important; }
.sim-gallery .grid-wrap { padding-top: 56px !important; }
.sim-gallery .caption-label { white-space: pre-line !important; font-size: 12px !important; }
.gallery-container, .grid-wrap { width: 100% !important; }
#app-title { text-align: center; margin-bottom: 2px; }
#info-bar { background: rgba(255,255,255,.7); border: 1px solid #e5e7eb;
            border-radius: 8px; padding: 6px 14px; margin-bottom: 10px;
            font-size: .95em; text-align: center; }
"""


def main() -> None:
    _preload_query_model()  # 启动即后台预热查询模型
    with gr.Blocks(title="DeepFashion 服装属性抽取 + 相似款检索 Demo",
                   theme=THEME, css=CSS) as demo:
        gr.Markdown("# DeepFashion · 服装属性抽取与相似款检索", elem_id="app-title")
        gr.Markdown(_build_info(), elem_id="info-bar")
        with gr.Row():
            with gr.Column(scale=1):  # 左：输入
                img = gr.Image(type="filepath", label="上传服装全身照", height=320)
                gr.Examples(examples=[[p] for p in _sample_examples()], inputs=[img],
                            label="示例图（点击加载，启动时随机更换）", examples_per_page=6)
                with gr.Row():
                    style = gr.Radio(["schema", "simple"], value="schema", label="Prompt 模板")
                    topk = gr.Slider(1, 10, value=5, step=1, label="相似款数量（Top-K）")
                prompt_box = gr.Textbox(
                    value=build_schema_prompt(),
                    label="提示词（可编辑）", lines=20, max_lines=20,
                )
                with gr.Row():
                    btn_extract = gr.Button("抽取属性", variant="secondary")
                    btn_retrieve = gr.Button("检索相似款", variant="secondary")
                btn_both = gr.Button("一键：抽取 + 检索", variant="primary")
                perf = gr.JSON(label="性能指标（推理与检索耗时）")
            with gr.Column(scale=1):  # 右：输出
                parsed = gr.JSON(label="结构化属性抽取结果", max_height=None)
                # gradio 原生 Gallery 预览：点图会在图框内部放大（大图 + 待选缩略图条一起显示全），
                # caption「#短id · 匹配分」随图显示。allow_preview=True 保持原生行为。
                sim_gallery = gr.Gallery(
                    label="相似款（视觉 + 属性混合检索）", columns=5, height=300,
                    object_fit="contain", fit_columns=False, allow_preview=True,
                    elem_classes="sim-gallery",
                )
                preview_info = gr.JSON(label="相似款详情（视觉 / 属性 / 综合相似度 & 属性）", max_height=None)
                sim_records = gr.State([])  # 检索结果记录，供点击缩略图更新该款信息
        style.change(fn=on_style_change, inputs=style, outputs=prompt_box)
        btn_extract.click(
            fn=extract_attrs, inputs=[img, prompt_box], outputs=[parsed, perf],
            show_progress="full",
        )
        btn_retrieve.click(
            fn=retrieve_similar, inputs=[img, prompt_box, topk],
            outputs=[sim_gallery, sim_records, preview_info, perf],
            show_progress="full",
        )
        btn_both.click(
            fn=predict_with_retrieval, inputs=[img, prompt_box, topk],
            outputs=[parsed, sim_gallery, sim_records, preview_info, perf],
            show_progress="full",
        )
        sim_gallery.select(
            fn=on_gallery_select, inputs=[sim_records], outputs=[preview_info],
        )
    demo.queue(default_concurrency_limit=1)  # 串行排队，避免撞 GPU
    demo.launch(server_name="127.0.0.1", server_port=7860, allowed_paths=[IMAGES_DIR],
                show_error=True)


if __name__ == "__main__":
    main()
