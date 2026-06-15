#!/usr/bin/env bash
set -euo pipefail

HF_HOME="${HF_HOME:-/data/yinghaoliu/cache/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HUGGINGFACE_HUB_CACHE:-${HF_HOME}}}"
HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HUB_CACHE}}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
http_proxy="${http_proxy:-http://127.0.0.1:7897}"
https_proxy="${https_proxy:-http://127.0.0.1:7897}"
HTTP_PROXY="${HTTP_PROXY:-${http_proxy}}"
HTTPS_PROXY="${HTTPS_PROXY:-${https_proxy}}"

export HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE HF_DATASETS_CACHE HF_ENDPOINT REPO_ROOT
export http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

echo "[env] HF_ENDPOINT=${HF_ENDPOINT:-}"
echo "[env] HF_HOME=${HF_HOME}"
echo "[env] http_proxy=${http_proxy}"

python - <<'PY'
import os
import re
from pathlib import Path

from datasets import load_dataset

repo_root = Path(os.environ["REPO_ROOT"])
task_root = repo_root / "lm-evaluation-harness" / "lm_eval" / "tasks"

datasets_to_fetch = [
    # Core LM eval tasks used by scripts/eval_blagpt_lm_eval.sh.
    ("allenai/ai2_arc", "ARC-Challenge"),
    ("Rowan/hellaswag", None),
    ("truthfulqa/truthful_qa", "multiple_choice"),
    ("allenai/winogrande", "winogrande_xl"),
    ("openai/gsm8k", "main"),
    ("baber/piqa", None),
    ("allenai/ai2_arc", "ARC-Easy"),
    ("Xnhyacinth/LongBench", "triviaqa"),
    ("Xnhyacinth/LongBench", "qasper"),
    ("xinrongzhang2022/InfiniteBench", None),
    ("recursal/longbench-v2", "government_single"),
]

def add(path, name=None):
    item = (path, name)
    if item not in datasets_to_fetch:
        datasets_to_fetch.append(item)

def yaml_value(text, key):
    match = re.search(rf"(?m)^\s*[\"']?{re.escape(key)}[\"']?\s*:\s*(.+?)\s*$", text)
    if not match:
        return None
    value = match.group(1).strip()
    if value in {"", '""', "''", "null", "None"}:
        return None
    return value.strip("\"'")

# The mmlu group expands to all subject configs under tasks/mmlu/default.
for yaml_path in sorted((task_root / "mmlu" / "default").glob("mmlu_*.yaml")):
    name = yaml_value(yaml_path.read_text(), "dataset_name")
    if name:
        add("cais/mmlu", name)

for path, name in datasets_to_fetch:
    label = path if name is None else f"{path}/{name}"
    print(f"[fetch] {label}")
    if name is None:
        load_dataset(path)
    else:
        load_dataset(path, name)

print("[done] lm-eval task datasets are cached")
PY
