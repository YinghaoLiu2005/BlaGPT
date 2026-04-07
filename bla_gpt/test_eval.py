import sys
from pathlib import Path

import torch

# Support running this file as: python bla_gpt/test_eval.py
REPO_ROOT = Path(__file__).resolve().parents[1]
BLA_GPT_DIR = REPO_ROOT / "bla_gpt"
MAMBA_DIR = REPO_ROOT / "mamba"
for p in (str(BLA_GPT_DIR), str(MAMBA_DIR), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from train import DistributedDataLoader
from utils import get_model

ckpt_path = "logs/my_experiment_7/state_step025500.pt"
model_name = "blagpt"

ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)

model_config, model_ctor = get_model(model_name)
for k, v in ckpt["model_config"].items():
    if hasattr(model_config, k):
        setattr(model_config, k, v)

train_cfg = ckpt["train_config"]

model = model_ctor(model_config).cuda().eval()

state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
model.load_state_dict(state_dict, strict=True)

B = train_cfg["device_batch_size"]
T = train_cfg["sequence_length"]
val_tokens = train_cfg["val_tokens"]
input_val_bin = train_cfg["input_val_bin"]

val_loader = DistributedDataLoader(
    input_val_bin, B, T, process_rank=0, num_processes=1
)

val_steps = 100
val_loss = 0.0

with torch.no_grad():
    for _ in range(val_steps):
        x, y = val_loader.next_batch()
        _, loss = model(x, y)
        if isinstance(loss, dict):
            loss = loss["total"]
        val_loss += loss.item()

val_loss /= val_steps
print("val_loss =", val_loss)
