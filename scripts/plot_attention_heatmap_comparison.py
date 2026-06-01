#!/usr/bin/env python
"""Plot local-mixing heatmaps for Mamba2, regular attention, GQA-SWA, and MHA-SWA.

Mamba2 does not expose an attention matrix, so its panel visualizes a fixed
causal-conv proxy: the average absolute conv1d kernel strength placed on the
corresponding recent-token diagonals.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/data/yinghaoliu/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "bla_gpt"))
sys.path.insert(0, str(REPO_ROOT / "mamba"))

from bla_gpt import GPT, GPTConfig  # noqa: E402


RUNS = {
    "Mamba2 Conv": "logs/mamba2_0/best_model_25000.pt",
    "Full Attention": "logs/regular_1/best_model_25000.pt",
    "Ours: GQA-SWA": "logs/mamba2_new_v3_w512_0/best_model_25000.pt",
    "MHA-SWA": "logs/mamba2_new_v5_w512_1/best_model_25000.pt",
}

COLORS = {
    "Mamba2 Conv": "#0072B2",
    "Full Attention": "#D55E00",
    "Ours: GQA-SWA": "#009E73",
    "MHA-SWA": "#CC79A7",
}


def load_tokens(val_bin: Path, seq_len: int, offset: int) -> torch.Tensor:
    with val_bin.open("rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        if header[0] != 20240520:
            raise ValueError(f"{val_bin} has unexpected magic number {header[0]}")
        tokens = np.frombuffer(f.read(), dtype=np.uint16).astype(np.int64)
    if offset + seq_len > len(tokens):
        raise ValueError(f"offset + seq_len exceeds validation shard length: {len(tokens)}")
    return torch.from_numpy(tokens[offset : offset + seq_len]).long()[None, :]


def clean_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key.removeprefix("_orig_mod."): value
        for key, value in state_dict.items()
    }


def load_model(ckpt_path: Path, device: torch.device) -> GPT:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    config = GPTConfig()
    for key, value in ckpt["model_config"].items():
        if hasattr(config, key):
            setattr(config, key, value)
    model = GPT(config)
    model.load_state_dict(clean_state_dict(ckpt["model"]), strict=True)
    model.eval()
    model.to(device)
    return model


@torch.no_grad()
def layer_input(model: GPT, idx: torch.Tensor, layer_idx: int) -> torch.Tensor:
    device = idx.device
    _, t = idx.shape
    pos = torch.arange(0, t, dtype=torch.long, device=device)
    x = model.transformer.wte(idx)
    if model.transformer.wpe is not None:
        x = x + model.transformer.wpe(pos)
    x = model.transformer.drop(x)

    if layer_idx != 0:
        raise NotImplementedError(
            "This script intentionally defaults to layer 0 so it can compare "
            "checkpoints without executing Mamba kernels. Use --layer 0."
        )
    return model.transformer.h[layer_idx].ln_1(x)


def causal_window_mask(seq_len: int, window_size: int | None, device: torch.device) -> torch.Tensor:
    q_pos = torch.arange(seq_len, device=device)[:, None]
    k_pos = torch.arange(seq_len, device=device)[None, :]
    mask = k_pos <= q_pos
    if window_size is not None and window_size > 0:
        mask = mask & (k_pos >= q_pos - window_size + 1)
    return mask


def softmax_heatmap(q: torch.Tensor, k: torch.Tensor, window_size: int | None) -> torch.Tensor:
    # q, k: (T, D)
    logits = q.float() @ k.float().transpose(0, 1)
    logits = logits / math.sqrt(k.shape[-1])
    mask = causal_window_mask(q.shape[0], window_size, q.device)
    logits = logits.masked_fill(~mask, float("-inf"))
    return torch.softmax(logits, dim=-1).detach().cpu().to(torch.float16)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    y = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    return (y * weight.float()).to(dtype=x.dtype)


def apply_extended_rotary(x: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
    with torch.autocast(device_type=x.device.type, enabled=False):
        seq_len = x.shape[1]
        pos = torch.arange(seq_len, device=x.device, dtype=inv_freq.dtype)
        freqs = torch.outer(pos, inv_freq.to(x.device))
        cos = freqs.cos()[None, :, None, :]
        sin = freqs.sin()[None, :, None, :]
        half = x.shape[-1] // 2
        x1 = x[..., :half].float()
        x2 = x[..., half:].float()
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], dim=-1).type_as(x)


@torch.no_grad()
def regular_attention_heatmap(model: GPT, idx: torch.Tensor, layer_idx: int, head_idx: int) -> torch.Tensor:
    x = layer_input(model, idx, layer_idx)
    attn = model.transformer.h[layer_idx].attn
    b, t, _ = x.shape
    q = attn._project_query(x, b, t)
    k, _ = attn._project_kv(x, b, t)
    if hasattr(attn, "q_norm"):
        q, k = attn._apply_norm(q, k)
    if hasattr(attn, "rotary"):
        q = apply_extended_rotary(q, attn.rotary.inv_freq)
        k = apply_extended_rotary(k, attn.rotary.inv_freq)
    kv_group = max(1, attn.n_head // attn.n_kv_head)
    kv_head_idx = min(head_idx // kv_group, attn.n_kv_head - 1)
    return softmax_heatmap(q[0, :, head_idx], k[0, :, kv_head_idx], window_size=None)


@torch.no_grad()
def mamba_swa_heatmap(model: GPT, idx: torch.Tensor, layer_idx: int, head_idx: int) -> torch.Tensor:
    x = layer_input(model, idx, layer_idx)
    mamba = model.transformer.h[layer_idx].attn.mamba
    _, t, _ = x.shape
    attn_input = rmsnorm(x, mamba.input_norm.weight, mamba.input_norm.eps) if hasattr(mamba, "input_norm") else x
    q = mamba.input_q_proj(attn_input).view(1, t, mamba.attn_nheads, mamba.headdim)

    kv_nheads = getattr(mamba, "attn_kv_nheads", mamba.attn_nheads)
    k = mamba.input_k_proj(attn_input).view(1, t, kv_nheads, mamba.headdim)
    cos, sin = mamba._rotary_cos_sin(t, x.device, q.dtype)
    q = mamba._apply_rotary(q, cos, sin)
    k = mamba._apply_rotary(k, cos, sin)
    kv_group = max(1, mamba.attn_nheads // kv_nheads)
    kv_head_idx = min(head_idx // kv_group, kv_nheads - 1)
    return softmax_heatmap(q[0, :, head_idx], k[0, :, kv_head_idx], mamba.window_size)


@torch.no_grad()
def mamba_conv_proxy_heatmap(model: GPT, idx: torch.Tensor, layer_idx: int) -> torch.Tensor:
    t = idx.shape[1]
    conv = model.transformer.h[layer_idx].attn.mamba.conv1d
    kernel = conv.weight.detach().abs().mean(dim=(0, 1)).float().cpu()
    kernel = kernel / kernel.sum().clamp_min(1e-12)
    heatmap = torch.zeros(t, t)
    # Put the fixed kernel on the causal recent-token diagonals.
    for lag, strength in enumerate(reversed(kernel)):
        rows = torch.arange(lag, t)
        cols = rows - lag
        heatmap[rows, cols] = strength
    heatmap = heatmap / heatmap.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return heatmap.to(torch.float16)


def heatmap_view(heatmap: torch.Tensor, view_size: int) -> np.ndarray:
    view_size = min(view_size, heatmap.shape[0], heatmap.shape[1])
    return heatmap[-view_size:, -view_size:].float().numpy()


def robust_norm(values: np.ndarray) -> PowerNorm:
    nonzero = values[values > 0]
    if nonzero.size == 0:
        vmax = 1.0
    else:
        vmax = float(np.quantile(nonzero, 0.997))
    return PowerNorm(gamma=0.42, vmin=0.0, vmax=max(vmax, 1e-8))


def distance_profile(heatmap: torch.Tensor, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    h = heatmap.float()
    seq_len = h.shape[0]
    max_lag = min(max_lag, seq_len - 1)
    lags = torch.arange(0, max_lag + 1)
    vals = []
    for lag in lags.tolist():
        diag = torch.diagonal(h, offset=-lag)
        vals.append(diag.mean() if diag.numel() else torch.tensor(0.0))
    return lags.numpy(), torch.stack(vals).numpy()


def plot_heatmaps(heatmaps: dict[str, torch.Tensor], output: Path, view_size: int) -> None:
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.4), constrained_layout=True)
    for ax, (title, heatmap) in zip(axes.flat, heatmaps.items()):
        view = heatmap_view(heatmap, view_size)
        norm = robust_norm(view)
        im = ax.imshow(view, origin="lower", aspect="equal", cmap="cividis", norm=norm, interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("key position in displayed block")
        ax.set_ylabel("query position in displayed block")
        ax.set_xticks([0, view.shape[1] // 2, view.shape[1] - 1])
        ax.set_yticks([0, view.shape[0] // 2, view.shape[0] - 1])
        ax.text(
            0.02,
            0.96,
            f"last {view.shape[0]} tokens",
            transform=ax.transAxes,
            va="top",
            ha="left",
            color="white",
            fontsize=8,
            bbox=dict(facecolor="black", edgecolor="none", alpha=0.45, pad=2),
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.ax.tick_params(labelsize=7)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle("Local mixing patterns computed from an 8192-token context", y=1.02, fontsize=12)
    fig.savefig(output, dpi=320, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_distance_profiles(heatmaps: dict[str, torch.Tensor], output: Path, max_lag: int) -> None:
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(6.8, 3.4), constrained_layout=True)
    for title, heatmap in heatmaps.items():
        lags, vals = distance_profile(heatmap, max_lag)
        ax.plot(lags, vals + 1e-12, label=title, color=COLORS.get(title), linewidth=1.8)
    ax.set_yscale("log")
    ax.set_xlabel("relative distance to current token")
    ax.set_ylabel("mean mixing / attention weight")
    ax.set_title("Average weight assigned to previous tokens")
    ax.grid(True, which="both", linewidth=0.4, alpha=0.35)
    ax.legend(frameon=False, ncol=2)
    ax.set_xlim(0, max_lag)
    fig.savefig(output, dpi=320, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--offset", type=int, default=4096)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--head", type=int, default=0)
    parser.add_argument("--view-size", type=int, default=1024)
    parser.add_argument("--profile-max-lag", type=int, default=2048)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "logs" / "attention_heatmap_comparison.png")
    parser.add_argument("--profile-output", type=Path, default=REPO_ROOT / "logs" / "attention_distance_profile.png")
    parser.add_argument("--val-bin", type=Path, default=REPO_ROOT / "data" / "fineweb10B" / "fineweb_val_000000.bin")
    args = parser.parse_args()

    if args.layer != 0:
        raise SystemExit("Only --layer 0 is supported by this lightweight visualization script.")

    device = torch.device(args.device)
    idx_cpu = load_tokens(args.val_bin, args.seq_len, args.offset)
    idx = idx_cpu.to(device)

    heatmaps: dict[str, torch.Tensor] = {}
    for title, rel_ckpt in RUNS.items():
        ckpt_path = REPO_ROOT / rel_ckpt
        print(f"Loading {ckpt_path}")
        model = load_model(ckpt_path, device)
        attention_type = model.config.attention
        if attention_type == "mamba2":
            heatmaps[title] = mamba_conv_proxy_heatmap(model, idx, args.layer)
        elif attention_type == "regular":
            heatmaps[title] = regular_attention_heatmap(model, idx, args.layer, args.head)
        elif attention_type in {"mamba2_new_v3", "mamba2_new_v5"}:
            heatmaps[title] = mamba_swa_heatmap(model, idx, args.layer, args.head)
        else:
            raise ValueError(f"Unsupported attention type: {attention_type}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    plot_heatmaps(heatmaps, args.output, args.view_size)
    plot_distance_profiles(heatmaps, args.profile_output, args.profile_max_lag)
    print(f"Saved {args.output}")
    print(f"Saved {args.profile_output}")


if __name__ == "__main__":
    main()
