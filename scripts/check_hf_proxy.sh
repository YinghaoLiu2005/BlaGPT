#!/usr/bin/env bash
set -euo pipefail

PROXY="${PROXY:-http://127.0.0.1:7897}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

echo "[check] proxy=${PROXY}"
echo "[check] endpoint=${HF_ENDPOINT}"

echo "[curl] endpoint through proxy"
curl -I -L --max-time 20 -x "${PROXY}" "${HF_ENDPOINT}/datasets/EleutherAI/lambada_openai/resolve/main/README.md"

echo "[python] huggingface_hub metadata"
http_proxy="${PROXY}" https_proxy="${PROXY}" HTTP_PROXY="${PROXY}" HTTPS_PROXY="${PROXY}" HF_ENDPOINT="${HF_ENDPOINT}" \
python - <<'PY'
from huggingface_hub import HfApi

api = HfApi()
info = api.dataset_info("EleutherAI/lambada_openai")
print(info.id)
PY
