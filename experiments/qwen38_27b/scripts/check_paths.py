#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验目录的静态自检（**不需要 GPU**，纯路径/配置校验）。

## 目录规则（本文件就是这条规则的机器化表达）

```
experiments/
├── <基座>/                       qwen3vl8b | qwen38_27b
│   └── <实验域>/
│       ├── config/               yaml / json（进 git）
│       ├── scripts/              shell / python（进 git）
│       ├── weights/              权重（gitignore）
│       ├── predictions/          原始预测（gitignore）
│       └── reports/              preds_* 与 report_*.md（报告进 git）
├── qwen38_27b/scripts/           跨域工具（check_paths / 加载冒烟）
├── qwen38_27b/logs/              27B 日志集中处
└── llamaboard_cache/             工具缓存
```

## 检查六件事

1. **配置可解析**：所有 `<基座>/<域>/config/*` 都能被 yaml/json 解析，且
   `model_name_or_path` / `adapter_name_or_path` / `deepspeed` / `dataset_dir`+`dataset|eval_dataset`
   这些**输入**路径真实存在（`output_dir` / `export_dir` 属产物，缺失只告警）。
2. **脚本引用完整**：脚本里出现的 `experiments/...` 路径字面量都能落到真实文件。
3. **旧结构残留**：不应再出现 `experiments/{main,ablation,ckpt_screen,smoke}/`
   （这些已收进 `experiments/qwen3vl8b/`）。
4. **8B 不依赖 27B**：`experiments/qwen3vl8b/**` 不得引用任何 27B 内容。
5. **无历史残留**：不应再有 `alpha/llamafactory` 引用（已改名 `vendor/llamafactory`）。
6. **软链完好**：`<基座>/<域>/weights/merged` 必须是软链（实体在 `/disk16T1`），不是实体目录。

用法（在仓库根目录运行）::

    python experiments/qwen38_27b/scripts/check_paths.py
"""

import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
BACKBONES = ("qwen3vl8b", "qwen38_27b")
DOMAINS = ("main", "ablation", "ckpt_screen", "badcase_study", "smoke")
SCAN_EXT = (".py", ".sh", ".yaml", ".yml", ".json", ".md")
SKIP_SCAN = {".git", "vendor", "data", "models", "scratch", "node_modules"}

# 分级方案里刻意未跑的消融格子：`summarize_ablation_27b.py` 会 exists 判断后跳过
OPTIONAL = {
    "experiments/qwen38_27b/ablation/predictions/qwen38_27b_ablate_schema20k/generated_predictions.jsonl",
    "experiments/qwen38_27b/ablation/predictions/qwen38_27b_ablate_simple20k/generated_predictions.jsonl",
    "experiments/qwen38_27b/ablation/predictions/qwen38_27b_ablate_simple35k/generated_predictions.jsonl",
    # 8B 的 badcase P1（难例续训第二阶段）从未执行：报告目录只有 P0，配置为当时预留。
    # 属既有的"未跑"状态，不是目录整理造成的断链。
    "experiments/qwen3vl8b/badcase_study/weights/prompt_retrain_lora",
}
# 这些文件里出现旧目录名是刻意的（记录重命名历史 / 新旧路径兼容回退）
OLD_NAME_OK = {
    "experiments/qwen38_27b/scripts/check_paths.py",
    "experiments/qwen38_27b/main/scripts/infer_27b.py",
}

OLD_PREFIX_PAT = re.compile(r"experiments/(main|ablation|ckpt_screen|smoke)/")
PAT_27B = re.compile(r"qwen38_27b|ckpt27b|_\s*27b|27B")
PATH_LITERAL = re.compile(r"experiments/[A-Za-z0-9_.\-/]+")

errors: list[str] = []
warns: list[str] = []


def rel(p: str) -> str:
    return os.path.normpath(os.path.join(ROOT, p))


def exists(path: str) -> bool:
    return os.path.exists(rel(path))


def check_exists(label: str, path: str, hard: bool = True) -> None:
    if exists(path):
        return
    if os.path.normpath(path) in OPTIONAL:
        return
    (errors if hard else warns).append(f"{label}: 不存在 -> {path}")


def iter_text_files():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_SCAN]
        for fn in filenames:
            if fn.endswith(SCAN_EXT):
                yield os.path.join(dirpath, fn)


# ---------------------------------------------------------------- 1. 配置
def load_yaml(path: str):
    import yaml

    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def check_configs() -> int:
    n = 0
    for bb in BACKBONES:
        for dom in DOMAINS:
            for f in sorted(glob.glob(rel(f"experiments/{bb}/{dom}/config/*"))):
                name = os.path.relpath(f, ROOT)
                try:
                    cfg = load_yaml(f)
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{name}: 解析失败 {exc}")
                    continue
                if not isinstance(cfg, dict):
                    errors.append(f"{name}: 顶层不是映射")
                    continue
                n += 1
                for key in ("model_name_or_path", "adapter_name_or_path", "deepspeed"):
                    val = cfg.get(key)
                    if isinstance(val, str) and val.strip():
                        check_exists(f"{name} 的 {key}", val.strip())
                for key in ("output_dir", "export_dir"):
                    val = cfg.get(key)
                    if isinstance(val, str) and val.strip():
                        check_exists(f"{name} 的 {key}（产物路径，仅告警）", val.strip(), hard=False)
                ds_dir = cfg.get("dataset_dir")
                if isinstance(ds_dir, str) and ds_dir.strip():
                    check_exists(f"{name} 的 dataset_dir", ds_dir.strip())
                    for key in ("dataset", "eval_dataset"):
                        val = cfg.get(key)
                        vals = val if isinstance(val, list) else ([val] if isinstance(val, str) else [])
                        for one in vals:
                            if one.strip():
                                check_exists(f"{name} 的 {key}", os.path.join(ds_dir.strip(), f"{one.strip()}.jsonl"))
    return n


# ---------------------------------------------------------------- 2. 脚本引用
def check_script_refs() -> int:
    n = 0
    for bb in BACKBONES:
        files = glob.glob(rel(f"experiments/{bb}/*/scripts/*")) + glob.glob(rel(f"experiments/{bb}/scripts/*"))
        for f in sorted(files):
            if not f.endswith((".sh", ".py")):
                continue
            n += 1
            with open(f, encoding="utf-8") as fh:
                text = fh.read()
            for raw in set(PATH_LITERAL.findall(text)):
                token = raw.rstrip(".,)`;:\"'")
                if any(ch in token for ch in "{}*$") or token.endswith("/"):
                    continue
                if not os.path.splitext(token)[1] and not os.path.isdir(rel(token)):
                    continue
                check_exists(f"{os.path.relpath(f, ROOT)} 引用的路径", token)
    return n


# ---------------------------------------------------------------- 3~5. 文本级检查
def scan_text_issues() -> tuple[list[str], list[str], list[str]]:
    old_prefix, b8_leak, alpha = [], [], []
    for path in iter_text_files():
        r = os.path.relpath(path, ROOT)
        if r == ".gitignore":
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                lines = fh.readlines()
        except (UnicodeDecodeError, OSError):
            continue
        in_8b = r.startswith("experiments/qwen3vl8b/")
        for i, line in enumerate(lines, 1):
            if OLD_PREFIX_PAT.search(line):
                old_prefix.append(f"{r}:{i}")
            if in_8b and PAT_27B.search(line):
                b8_leak.append(f"{r}:{i}: {line.strip()[:80]}")
            if r not in OLD_NAME_OK and "alpha/llamafactory" in line:
                alpha.append(f"{r}:{i}")
    return old_prefix, b8_leak, alpha


# ---------------------------------------------------------------- 6. 软链
def check_symlinks() -> list[str]:
    bad = []
    for p in sorted(glob.glob(rel("experiments/*/*/weights/merged"))):
        if not os.path.islink(p):
            bad.append(f"{os.path.relpath(p, ROOT)} 应为软链，实际是实体目录")
        else:
            target = os.readlink(p)
            if "/disk16T1" not in target:
                bad.append(f"{os.path.relpath(p, ROOT)} 指向 {target}，预期在 /disk16T1")
    return bad


# ---------------------------------------------------------------- 7. 回推仓库根的层数
# 目录调整过层级后，脚本里「相对自身位置回推仓库根」的层数必须同步，否则会 cd 到仓库的上一级。
# 这类错误静态路径检查看不出来（路径字面量本身是对的），只能靠层数核对。
SH_CD = re.compile(r'cd "\$\(dirname "\$0"\)((?:/\.\.)+)"')
PY_CHAIN = re.compile(r"(?:os\.path\.dirname\()+(?:os\.path\.abspath\(__file__\))\)+")


def check_root_depth() -> list[str]:
    bad = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_SCAN]
        need_dir = len(os.path.relpath(dirpath, ROOT).split(os.sep))      # 目录相对仓库根有几层
        for fn in filenames:
            path = os.path.join(dirpath, fn)
            r = os.path.relpath(path, ROOT)
            if fn.endswith(".sh"):
                try:
                    text = open(path, encoding="utf-8").read()
                except (UnicodeDecodeError, OSError):
                    continue
                for m in SH_CD.finditer(text):
                    got = m.group(1).count("/..")
                    if got != need_dir:
                        bad.append(f"{r}: cd 层数 {got}，应为 {need_dir}（会 cd 到仓库的上一级）")
            elif fn.endswith(".py"):
                try:
                    text = open(path, encoding="utf-8").read()
                except (UnicodeDecodeError, OSError):
                    continue
                for c in set(PY_CHAIN.findall(text)):
                    pass
                for m in PY_CHAIN.finditer(text):
                    n = m.group(0).count("os.path.dirname")
                    if n == 1:
                        continue                                            # 脚本自身目录，语义不同
                    if n != need_dir + 1:
                        bad.append(f"{r}: dirname×{n}，应为 {need_dir + 1}（会解析到仓库的上一级）")
    return bad


def main() -> int:
    print(f"仓库根: {ROOT}\n")

    n_cfg = check_configs()
    print(f"[1] 配置检查：{n_cfg} 个配置已解析")

    n_scripts = check_script_refs()
    print(f"[2] 脚本引用检查：{n_scripts} 个脚本已扫描")

    old_prefix, b8_leak, alpha = scan_text_issues()
    print(f"[3] 旧结构残留：{len(old_prefix)} 处")
    for h in old_prefix:
        errors.append(f"仍使用旧结构路径: {h}")
    print(f"[4] 8B 依赖 27B：{len(b8_leak)} 处")
    for h in b8_leak:
        errors.append(f"8B 目录引用 27B 内容: {h}")
    print(f"[5] alpha/ 残留：{len(alpha)} 处")
    for h in alpha:
        errors.append(f"仍引用 alpha/llamafactory: {h}")

    bad_links = check_symlinks()
    print(f"[6] 软链完好：{len(bad_links)} 处异常")
    errors.extend(bad_links)

    depth_bad = check_root_depth()
    print(f"[7] 脚本回推仓库根的层数：{len(depth_bad)} 处不匹配")
    errors.extend(depth_bad)

    print()
    for w in warns:
        print(f"  [warn] {w}")
    if errors:
        print(f"\n[FAIL] {len(errors)} 个问题：")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("[OK] 七项全通过：配置可解析、引用路径存在、无旧结构残留、8B 不依赖 27B、"
          "无 alpha 引用、软链完好、脚本深度正确")
    return 0


if __name__ == "__main__":
    sys.exit(main())
