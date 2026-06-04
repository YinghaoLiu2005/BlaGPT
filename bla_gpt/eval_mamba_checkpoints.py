import argparse
import csv
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


REPO_ROOT = Path(__file__).resolve().parents[1]
BLA_GPT_DIR = REPO_ROOT / "bla_gpt"
MAMBA_DIR = REPO_ROOT / "mamba"
for path in (str(BLA_GPT_DIR), str(MAMBA_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from train import DistributedDataLoader
from utils import get_model


DEFAULT_CKPTS = {
    #"mamba_baseline": "logs/mamba2_0/state_step025000.pt",
    #"regular": "logs/regular_1/state_step025000.pt",
    "mamba2_GQA": "logs/mamba2_GQA_w512_2/state_step078242.pt",
    #"mamba2_MHA": "logs/mamba2_MHA_w512_1/state_step025000.pt",
}


def parse_ckpt_arg(value):
    if ":" not in value:
        raise argparse.ArgumentTypeError("--ckpt must use name:path format")
    name, path = value.split(":", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("--ckpt must use name:path format")
    return name, path


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def resolve_pattern(pattern):
    path = Path(pattern)
    if path.is_absolute():
        return str(path)
    return str(REPO_ROOT / path)


def load_state_dict(model, state_dict):
    try:
        model.load_state_dict(state_dict, strict=True)
        return
    except RuntimeError:
        pass

    stripped = {
        key.replace("_orig_mod.", "", 1): value
        for key, value in state_dict.items()
    }
    model.load_state_dict(stripped, strict=True)


def scalar_loss(loss):
    if isinstance(loss, dict):
        loss = loss["total"]
    return loss


def build_model(model_name, ckpt, sequence_length, compile_model):
    model_config, model_ctor = get_model(model_name)
    for key, value in ckpt["model_config"].items():
        if hasattr(model_config, key):
            setattr(model_config, key, value)
    model_config.block_size = sequence_length

    model = model_ctor(model_config).cuda().eval()
    if compile_model:
        model = torch.compile(model)
    load_state_dict(model, ckpt["model"])
    return model, model_config


def evaluate_one(model, val_loader, val_steps, ctx):
    val_loader.reset()
    val_loss = torch.tensor(0.0, device="cuda")

    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(val_steps):
            x_val, y_val = val_loader.next_batch()
            with ctx:
                _, loss = model(x_val, y_val)
                val_loss += scalar_loss(loss).detach()
                del loss
    torch.cuda.synchronize()
    elapsed = time.time() - start

    dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
    val_loss /= val_steps
    return val_loss.item(), elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="blagpt")
    parser.add_argument("--lengths", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    parser.add_argument("--ckpt", action="append", type=parse_ckpt_arg, default=None)
    parser.add_argument("--input_val_bin", type=str, default=None)
    parser.add_argument("--val_tokens", type=int, default=None)
    parser.add_argument("--device_batch_size", type=int, default=None)
    parser.add_argument("--output_csv", type=str, default="logs/mamba_length_eval.csv")
    parser.add_argument("--compile_model", action="store_true")
    args = parser.parse_args()

    assert torch.cuda.is_available()
    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    ckpts = dict(args.ckpt) if args.ckpt else DEFAULT_CKPTS
    rows = []

    if master_process:
        print(f"using {ddp_world_size} process(es)")

    for ckpt_name, ckpt_path in ckpts.items():
        ckpt_path = resolve_path(ckpt_path)
        if not ckpt_path.exists():
            if master_process:
                print(f"skip missing checkpoint: {ckpt_name} -> {ckpt_path}")
            continue

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        train_cfg = ckpt.get("train_config", {})
        input_val_bin = args.input_val_bin or train_cfg.get(
            "input_val_bin", "data/fineweb10B/fineweb_val_*.bin"
        )
        input_val_bin = resolve_pattern(input_val_bin)
        val_tokens = args.val_tokens or int(train_cfg.get("val_tokens", 10485760))
        device_batch_size = args.device_batch_size or int(
            train_cfg.get("device_batch_size", 8)
        )

        for sequence_length in args.lengths:
            tokens_per_step = device_batch_size * sequence_length * ddp_world_size
            if val_tokens % tokens_per_step != 0:
                raise ValueError(
                    f"val_tokens={val_tokens} must be divisible by "
                    f"device_batch_size * sequence_length * world_size = {tokens_per_step}"
                )
            val_steps = val_tokens // tokens_per_step

            model, model_config = build_model(
                args.model_name, ckpt, sequence_length, args.compile_model
            )
            model = DDP(model, device_ids=[ddp_local_rank], broadcast_buffers=False)

            val_loader = DistributedDataLoader(
                input_val_bin,
                device_batch_size,
                sequence_length,
                ddp_rank,
                ddp_world_size,
            )
            if master_process:
                print(
                    f"eval {ckpt_name} len={sequence_length} "
                    f"steps={val_steps} attention={model_config.attention}"
                )

            val_loss, elapsed = evaluate_one(model, val_loader, val_steps, ctx)
            tokens_per_sec = val_tokens / elapsed
            row = {
                "name": ckpt_name,
                "checkpoint": str(ckpt_path),
                "attention": model_config.attention,
                "mamba_window_size": model_config.mamba_window_size,
                "sequence_length": sequence_length,
                "device_batch_size": device_batch_size,
                "val_tokens": val_tokens,
                "val_steps": val_steps,
                "val_loss": val_loss,
                "elapsed_sec": elapsed,
                "tokens_per_sec": tokens_per_sec,
            }
            rows.append(row)
            if master_process:
                print(
                    f"{ckpt_name}, len={sequence_length}, "
                    f"val_loss={val_loss:.4f}, "
                    f"tokens/s={tokens_per_sec:.1f}, elapsed={elapsed:.2f}s"
                )

            del model, val_loader
            torch.cuda.empty_cache()

        del ckpt
        

    if master_process and rows:
        output_csv = resolve_path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {output_csv}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
