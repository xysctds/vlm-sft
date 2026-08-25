#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""已部署的 DeepFashion 属性抽取模型 demo 客户端。

先启动 vLLM 服务（在仓库根目录）::

    vllm serve llamafactory/saves/dfattr/merged/qwen3vl8b_dfattr_merged --served-model-name deepfashion --port 8000

然后运行::

    python scripts/serve_demo.py --image /path/to/never_seen.jpg

同时展示模型原始输出与解析后的结构化 JSON。
"""

import argparse
import base64
import json
import os
import sys

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_convert_deepfashion_multi import SIMPLE_JSON_PROMPT, build_schema_prompt  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="服装图片路径（jpg/png）")
    parser.add_argument("--url", default="http://localhost:8000/v1/chat/completions")
    parser.add_argument("--model", default="deepfashion", help="vLLM 的 --served-model-name")
    parser.add_argument("--prompt-style", choices=["schema", "simple"], default="schema")
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    if not os.path.exists(args.image):
        sys.exit(f"[err] image not found: {args.image}")

    with open(args.image, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode()
    data_uri = f"data:image/jpeg;base64,{b64}"

    prompt = build_schema_prompt() if args.prompt_style == "schema" else SIMPLE_JSON_PROMPT
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_uri}},
            {"type": "text", "text": prompt},
        ]}],
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
    }

    print(f"[demo] image={args.image} prompt_style={args.prompt_style}")
    resp = requests.post(args.url, json=payload, timeout=120)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]

    print("\n--- raw model output ---")
    print(content)

    print("\n--- parsed JSON ---")
    try:
        start, end = content.find("{"), content.rfind("}")
        obj = json.loads(content[start:end + 1])
        print(json.dumps(obj, indent=2, ensure_ascii=False))
    except (ValueError, json.JSONDecodeError):
        print("(could not parse output as JSON)")


if __name__ == "__main__":
    main()
