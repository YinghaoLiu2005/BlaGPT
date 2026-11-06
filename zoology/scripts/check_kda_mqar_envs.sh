#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PDEP="/data/yinghaoliu/cache/uv/archive-v0/XJyYec6VabYR1Q40:/data/yinghaoliu/cache/uv/archive-v0/kVu9aRc1YqjvqZlk:/data/yinghaoliu/cache/uv/archive-v0/QhR_OFg--t0l0bhx:/data/yinghaoliu/cache/uv/archive-v0/NvfKYmlMU5DwnAIH"
export PYTHONPATH="$PDEP:$ROOT:$ROOT/3rdparty/flash-linear-attention:${PYTHONPATH:-}"

echo "blagpt: Mamba2/Mamba2-GQA imports"
/data/yinghaoliu/conda/conda_envs/blagpt/bin/python - <<'PY'
from zoology.experiments.kda_mqar.mixers import Mamba2, Mamba2GQA
Mamba2(256); Mamba2GQA(256)
print("ok")
PY

echo "fla: KDA/GDN import (requires a CUDA driver)"
/data/yinghaoliu/conda/conda_envs/fla/bin/python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable: FLA KDA/GDN cannot run on this host")
from zoology.experiments.kda_mqar.mixers import KDA, GDN
KDA(256); GDN(256)
print("ok")
PY
