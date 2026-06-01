#!/usr/bin/env python
"""Mechanistic visualizations for Mamba2 vs GQA-SWA-Mamba2.

The figures are designed to separate three mechanisms that are otherwise easy
to conflate:

1. local token mixing: fixed causal conv in Mamba2 vs dynamic GQA-SWA;
2. SSM retention: exp(sum dt * A), the learned exponential decay route;
3. effective SSM mixing: |C_t B_s| * dt_s * decay(t, s).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/data/yinghaoliu/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import PowerNorm, TwoSlopeNorm


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "bla_gpt"))
sys.path.insert(0, str(REPO_ROOT / "mamba"))

from bla_gpt import GPT, GPTConfig  # noqa: E402


RUNS = {
    "baseline": {
        "label": "Mamba2 baseline",
        "path": REPO_ROOT / "logs" / "mamba2_0" / "best_model_25000.pt",
        "color": "#0072B2",
    },
    "ours": {
        "label": "Ours: GQA-SWA + Mamba2",
        "path": REPO_ROOT / "logs" / "mamba2_new_v3_w512_0" / "best_model_25000.pt",
        "color": "#009E73",
    },
    "mha_swa": {
        "label": "MHA-SWA + Mamba2",
        "path": REPO_ROOT / "logs" / "mamba2_new_v5_w512_1" / "best_model_25000.pt",
        "color": "#CC79A7",
    },
    "regular": {
        "label": "Full attention",
        "path": REPO_ROOT / "logs" / "regular_1" / "best_model_25000.pt",
        "color": "#D55E00",
    },
}


@dataclass
class MambaDiagnostics:
    label: str
    color: str
    A: torch.Tensor
    dt: torch.Tensor
    B: torch.Tensor
    C: torch.Tensor
    dA_cumsum: torch.Tensor
    intrinsic_half_life: torch.Tensor


def setup_style() -> None:
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 12,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def savefig(fig: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=320, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


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
    return {key.removeprefix("_orig_mod."): value for key, value in state_dict.items()}


def load_model(ckpt_path: Path, device: torch.device) -> GPT:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    config = GPTConfig()
    for key, value in ckpt["model_config"].items():
        if hasattr(config, key):
            setattr(config, key, value)
    model = GPT(config)
    model.load_state_dict(clean_state_dict(ckpt["model"]), strict=True)
    model.eval().to(device)
    return model


@torch.no_grad()
def layer0_input(model: GPT, idx: torch.Tensor, abs_start: int) -> torch.Tensor:
    device = idx.device
    t = idx.shape[1]
    x = model.transformer.wte(idx)
    if model.transformer.wpe is not None:
        pos = torch.arange(abs_start, abs_start + t, dtype=torch.long, device=device)
        x = x + model.transformer.wpe(pos)
    x = model.transformer.drop(x)
    return model.transformer.h[0].ln_1(x)


@torch.no_grad()
def layer0_input_zero_pos(model: GPT, idx: torch.Tensor) -> torch.Tensor:
    device = idx.device
    t = idx.shape[1]
    x = model.transformer.wte(idx)
    if model.transformer.wpe is not None:
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        x = x + model.transformer.wpe(pos)
    x = model.transformer.drop(x)
    return model.transformer.h[0].ln_1(x)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    y = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    return (y * weight.float()).to(dtype=x.dtype)


def apply_rotary_abs(x: torch.Tensor, inv_freq: torch.Tensor, abs_start: int) -> torch.Tensor:
    with torch.autocast(device_type=x.device.type, enabled=False):
        seq_len = x.shape[1]
        pos = torch.arange(abs_start, abs_start + seq_len, device=x.device, dtype=inv_freq.dtype)
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
def apply_input_swa(mamba, u: torch.Tensor, abs_start: int, block_size: int = 128) -> torch.Tensor:
    """Exact causal sliding-window attention for the first Mamba block.

    This is a transparent PyTorch implementation for visualization. It avoids
    FlashAttention so we can inspect the same quantities on CPU.
    """
    if not hasattr(mamba, "input_q_proj"):
        return u

    attn_input = rmsnorm(u, mamba.input_norm.weight, mamba.input_norm.eps) if hasattr(mamba, "input_norm") else u
    _, seq_len, _ = u.shape
    q = mamba.input_q_proj(attn_input).view(1, seq_len, mamba.attn_nheads, mamba.headdim)
    kv_nheads = getattr(mamba, "attn_kv_nheads", mamba.attn_nheads)
    k = mamba.input_k_proj(attn_input).view(1, seq_len, kv_nheads, mamba.headdim)
    v = mamba.input_v_proj(attn_input).view(1, seq_len, kv_nheads, mamba.headdim)
    q = apply_rotary_abs(q, mamba.rotary_inv_freq, abs_start)
    k = apply_rotary_abs(k, mamba.rotary_inv_freq, abs_start)

    repeat = max(1, mamba.attn_nheads // kv_nheads)
    k = k.repeat_interleave(repeat, dim=2)[0]
    v = v.repeat_interleave(repeat, dim=2)[0]
    q = q[0]

    out = torch.empty_like(q)
    scale = 1.0 / math.sqrt(mamba.headdim)
    window = int(mamba.window_size)
    key_pos_all = torch.arange(seq_len, device=u.device)
    for q0 in range(0, seq_len, block_size):
        q1 = min(q0 + block_size, seq_len)
        k0 = max(0, q0 - window + 1)
        k1 = q1
        q_block = q[q0:q1].transpose(0, 1).float()  # (H, Q, D)
        k_ctx = k[k0:k1].transpose(0, 1).float()  # (H, K, D)
        scores = torch.matmul(q_block, k_ctx.transpose(-1, -2)) * scale
        q_pos = key_pos_all[q0:q1][:, None]
        k_pos = key_pos_all[k0:k1][None, :]
        mask = (k_pos <= q_pos) & (k_pos >= q_pos - window + 1)
        scores = scores.masked_fill(~mask[None, :, :], float("-inf"))
        probs = torch.softmax(scores, dim=-1).to(v.dtype)
        v_ctx = v[k0:k1].transpose(0, 1)
        out[q0:q1] = torch.matmul(probs, v_ctx).transpose(0, 1)

    attn_out = out.reshape(1, seq_len, mamba.d_model).to(dtype=u.dtype)
    return u + mamba.input_out_proj(attn_out)


def split_mamba_projection(mamba, zxbcdt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    d_mlp = (zxbcdt.shape[-1] - 2 * mamba.d_ssm - 2 * mamba.ngroups * mamba.d_state - mamba.nheads) // 2
    z0, x0, z, xBC, dt = torch.split(
        zxbcdt,
        [d_mlp, d_mlp, mamba.d_ssm, mamba.d_ssm + 2 * mamba.ngroups * mamba.d_state, mamba.nheads],
        dim=-1,
    )
    return z0, x0, z, xBC, dt


@torch.no_grad()
def extract_mamba_diagnostics(
    model: GPT,
    idx: torch.Tensor,
    abs_start: int,
    keep_start: int,
    label: str,
    color: str,
) -> MambaDiagnostics:
    x = layer0_input(model, idx, abs_start)
    mamba = model.transformer.h[0].attn.mamba
    u = apply_input_swa(mamba, x, abs_start)
    zxbcdt = mamba.in_proj(u)
    _, _, _, xBC, dt_pre = split_mamba_projection(mamba, zxbcdt)

    if hasattr(mamba, "conv1d"):
        xBC = F.conv1d(
            xBC.transpose(1, 2),
            mamba.conv1d.weight,
            bias=mamba.conv1d.bias,
            padding=mamba.d_conv - 1,
            groups=mamba.conv1d.groups,
        ).transpose(1, 2)[:, : xBC.shape[1]]
        xBC = F.silu(xBC)
    else:
        xBC = F.silu(xBC)

    _, B, C = torch.split(
        xBC,
        [mamba.d_ssm, mamba.ngroups * mamba.d_state, mamba.ngroups * mamba.d_state],
        dim=-1,
    )
    B = B.view(B.shape[0], B.shape[1], mamba.ngroups, mamba.d_state)
    C = C.view(C.shape[0], C.shape[1], mamba.ngroups, mamba.d_state)
    dt = F.softplus(dt_pre + mamba.dt_bias.to(dtype=dt_pre.dtype))
    A = -torch.exp(mamba.A_log.float())

    A = A.detach().cpu().float()
    dt = dt[0, keep_start:].detach().cpu().float()
    B = B[0, keep_start:, 0].detach().cpu().float()
    C = C[0, keep_start:, 0].detach().cpu().float()
    dA_cumsum = torch.cumsum(dt * A[None, :], dim=0)
    half_life = torch.log(torch.tensor(0.5)).neg() / (F.softplus(mamba.dt_bias.detach().cpu().float()) * torch.exp(mamba.A_log.detach().cpu().float()))
    return MambaDiagnostics(label, color, A, dt, B, C, dA_cumsum, half_life)


def sampled_positions(n: int, token_span: int, heatmap_bins: int) -> torch.Tensor:
    token_span = min(token_span, n)
    bins = min(token_span, heatmap_bins)
    start = n - token_span
    if bins == token_span:
        return torch.arange(start, n)
    return torch.linspace(start, n - 1, steps=bins).round().long()


def get_decay_matrix(
    diag: MambaDiagnostics,
    token_span: int,
    heatmap_bins: int,
    head_average: bool = True,
) -> torch.Tensor:
    n = diag.dt.shape[0]
    rows = sampled_positions(n, token_span, heatmap_bins)
    cols = rows
    log_decay = diag.dA_cumsum[rows][:, None, :] - diag.dA_cumsum[cols][None, :, :]
    mask = cols[None, :] <= rows[:, None]
    decay = torch.exp(log_decay.clamp(min=-80.0, max=0.0)).masked_fill(~mask[:, :, None], 0.0)
    return decay.mean(dim=-1) if head_average else decay[:, :, 0]


def get_effective_matrix(
    diag: MambaDiagnostics,
    token_span: int,
    heatmap_bins: int,
    normalize_rows: bool = True,
) -> torch.Tensor:
    n = diag.dt.shape[0]
    rows = sampled_positions(n, token_span, heatmap_bins)
    cols = rows
    log_decay = diag.dA_cumsum[rows][:, None, :] - diag.dA_cumsum[cols][None, :, :]
    mask = cols[None, :] <= rows[:, None]
    decay = torch.exp(log_decay.clamp(min=-80.0, max=0.0)).masked_fill(~mask[:, :, None], 0.0)
    cb = torch.matmul(diag.C[rows], diag.B[cols].transpose(0, 1)).abs()
    dt_keys = diag.dt[cols]  # (K, H)
    eff = (decay * dt_keys[None, :, :]).mean(dim=-1) * cb
    eff = eff.masked_fill(~mask, 0.0)
    if normalize_rows:
        eff = eff / eff.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return eff


@torch.no_grad()
def regular_attention_matrix(
    model: GPT,
    idx: torch.Tensor,
    token_span: int,
    heatmap_bins: int,
    head_idx: int = 0,
    row_block: int = 32,
) -> torch.Tensor:
    x = layer0_input_zero_pos(model, idx)
    attn = model.transformer.h[0].attn
    batch, seq_len, _ = x.shape
    q = attn._project_query(x, batch, seq_len)
    k, _ = attn._project_kv(x, batch, seq_len)
    if hasattr(attn, "q_norm"):
        q, k = attn._apply_norm(q, k)
    if hasattr(attn, "rotary"):
        q = apply_rotary_abs(q, attn.rotary.inv_freq, abs_start=0)
        k = apply_rotary_abs(k, attn.rotary.inv_freq, abs_start=0)

    kv_group = max(1, attn.n_head // attn.n_kv_head)
    if head_idx >= 0:
        q = q[0, :, head_idx].float()
        k = k[0, :, min(head_idx // kv_group, attn.n_kv_head - 1)].float()
    else:
        k = k.repeat_interleave(kv_group, dim=2)
        q = q[0].float()
        k = k[0].float()

    rows = sampled_positions(seq_len, token_span, heatmap_bins).to(idx.device)
    cols = rows
    key_pos = torch.arange(seq_len, device=idx.device)
    out = torch.empty((rows.numel(), cols.numel()), dtype=torch.float32)
    scale = 1.0 / math.sqrt(k.shape[-1])
    soft_cap_value = float(getattr(attn, "soft_cap", 0.0))

    for start in range(0, rows.numel(), row_block):
        end = min(start + row_block, rows.numel())
        row_idx = rows[start:end]
        if head_idx >= 0:
            scores = torch.matmul(q[row_idx], k.transpose(0, 1)) * scale
            scores = scores[:, None, :]
        else:
            scores = torch.einsum("qhd,khd->qhk", q[row_idx], k) * scale
        if soft_cap_value > 0:
            scores = torch.tanh(scores / soft_cap_value) * soft_cap_value
        scores = scores.masked_fill(key_pos[None, None, :] > row_idx[:, None, None], float("-inf"))
        probs = torch.softmax(scores, dim=-1).mean(dim=1)
        view = probs[:, cols]
        view = view.masked_fill(cols[None, :] > row_idx[:, None], 0.0)
        view = view / view.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        out[start:end] = view.detach().cpu()
    return out


def lag_profiles(diag: MambaDiagnostics, max_lag: int, query_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = diag.dt.shape[0]
    query_rows = torch.arange(n - query_count, n)
    lags = torch.arange(0, min(max_lag, n - 1) + 1)
    decay_vals = []
    eff_vals = []
    for lag in lags.tolist():
        rows = query_rows[query_rows >= lag]
        cols = rows - lag
        log_decay = diag.dA_cumsum[rows] - diag.dA_cumsum[cols]
        decay_h = torch.exp(log_decay.clamp(min=-80.0, max=0.0))
        decay_vals.append(decay_h.mean())
        cb = (diag.C[rows] * diag.B[cols]).sum(dim=-1).abs()
        eff = cb * (decay_h * diag.dt[cols]).mean(dim=-1)
        eff_vals.append(eff.mean())
    return lags.numpy(), torch.stack(decay_vals).numpy(), torch.stack(eff_vals).numpy()


def intrinsic_half_life_by_layer(model: GPT) -> tuple[np.ndarray, np.ndarray]:
    layers = []
    half_lives = []
    for layer_idx, block in enumerate(model.transformer.h):
        if not hasattr(block.attn, "mamba"):
            continue
        mamba = block.attn.mamba
        dt0 = F.softplus(mamba.dt_bias.detach().cpu().float())
        A_abs = torch.exp(mamba.A_log.detach().cpu().float())
        hl = torch.log(torch.tensor(0.5)).neg() / (dt0 * A_abs)
        layers.extend([layer_idx] * hl.numel())
        half_lives.extend(hl.tolist())
    return np.asarray(layers), np.asarray(half_lives)


@torch.no_grad()
def local_attention_lag_profile(model: GPT, idx: torch.Tensor, abs_start: int, max_lag: int, query_count: int) -> tuple[np.ndarray, np.ndarray]:
    x = layer0_input(model, idx, abs_start)
    mamba = model.transformer.h[0].attn.mamba
    if not hasattr(mamba, "input_q_proj"):
        raise ValueError("Model has no input SWA projections")
    attn_input = rmsnorm(x, mamba.input_norm.weight, mamba.input_norm.eps) if hasattr(mamba, "input_norm") else x
    _, seq_len, _ = x.shape
    q = mamba.input_q_proj(attn_input).view(1, seq_len, mamba.attn_nheads, mamba.headdim)
    kv_nheads = getattr(mamba, "attn_kv_nheads", mamba.attn_nheads)
    k = mamba.input_k_proj(attn_input).view(1, seq_len, kv_nheads, mamba.headdim)
    q = apply_rotary_abs(q, mamba.rotary_inv_freq, abs_start)[0].float()
    k = apply_rotary_abs(k, mamba.rotary_inv_freq, abs_start)[0].float()
    k = k.repeat_interleave(max(1, mamba.attn_nheads // kv_nheads), dim=1)
    window = min(int(mamba.window_size), max_lag + 1)
    rows = torch.arange(seq_len - query_count, seq_len, device=x.device)
    lags = torch.arange(0, max_lag + 1, device=x.device)
    vals = torch.zeros(max_lag + 1, device=x.device)
    counts = torch.zeros(max_lag + 1, device=x.device)
    scale = 1.0 / math.sqrt(mamba.headdim)
    for row in rows.tolist():
        k0 = max(0, row - window + 1)
        k1 = row + 1
        scores = torch.einsum("hd,khd->hk", q[row], k[k0:k1]) * scale
        probs = torch.softmax(scores, dim=-1).mean(dim=0)
        lag_vec = row - torch.arange(k0, k1, device=x.device)
        keep = lag_vec <= max_lag
        vals[lag_vec[keep]] += probs[keep]
        counts[lag_vec[keep]] += 1
    profile = vals / counts.clamp_min(1)
    return lags.cpu().numpy(), profile.cpu().numpy()


def plot_mechanism_overview(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 3.1), constrained_layout=True)
    ax.axis("off")
    rows = [
        ("Mamba2\nbaseline", ["token\nx", "depthwise\nconv1d", "x, B, C,\ndt", "SSM decay\nscan", "output"]),
        ("Ours\nGQA-SWA", ["token\nx", "GQA-SWA\nlocal mixer", "x, B, C,\ndt", "SSM decay\nscan", "output"]),
    ]
    colors = ["#0072B2", "#009E73"]
    for r, (label, boxes) in enumerate(rows):
        y = 0.72 - r * 0.44
        ax.text(0.025, y, label, va="center", ha="left", fontsize=12, weight="bold", color=colors[r])
        x0 = 0.24
        for i, box in enumerate(boxes):
            x = x0 + i * 0.145
            ax.add_patch(plt.Rectangle((x, y - 0.09), 0.115, 0.18, facecolor="#F7F7F7", edgecolor=colors[r], lw=1.3))
            ax.text(x + 0.0575, y, box, va="center", ha="center", fontsize=8.5)
            if i < len(boxes) - 1:
                ax.annotate("", xy=(x + 0.139, y), xytext=(x + 0.117, y), arrowprops=dict(arrowstyle="->", color="#333333", lw=1.0))
    ax.text(
        0.50,
        0.05,
        "Diagnostic target: separate local mixing from the learned exponential SSM memory path.",
        ha="center",
        va="center",
        fontsize=9,
    )
    savefig(fig, output_dir / "01_mechanism_overview.png")


def add_lag_guides(ax: plt.Axes, axis_size: int, token_span: int, color: str, text_color: str) -> None:
    for lag in [512, 1024, 2048, 4096]:
        if lag >= token_span:
            continue
        lag_px = lag * (axis_size - 1) / max(1, token_span - 1)
        xs = np.asarray([0, axis_size - lag_px - 1])
        ys = xs + lag_px
        ax.plot(xs, ys, color=color, lw=0.8, ls=(0, (4, 3)), alpha=0.78)
        label_x = max(8, axis_size - lag_px - 1 - int(axis_size * 0.10))
        label_y = label_x + lag_px
        ax.text(
            label_x,
            label_y,
            f"lag {lag}",
            color=text_color,
            fontsize=7,
            ha="right",
            va="bottom",
            rotation=38,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.62, pad=0.8),
        )


def set_block_ticks(ax: plt.Axes, axis_size: int, token_span: int) -> None:
    if token_span >= 8192:
        tick_tokens = [0, 2048, 4096, 6144, token_span - 1]
    elif token_span >= 2048:
        tick_tokens = [0, 512, 1024, 1536, token_span - 1]
    else:
        tick_tokens = [0, token_span // 2, token_span - 1]
    ticks = [round(token * (axis_size - 1) / max(1, token_span - 1)) for token in tick_tokens]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([str(token) for token in tick_tokens])
    ax.set_yticklabels([str(token) for token in tick_tokens])


def plot_decay_heatmaps(
    diags: dict[str, MambaDiagnostics],
    output_dir: Path,
    token_span: int,
    heatmap_bins: int,
    regular_map: torch.Tensor | None = None,
    regular_head: int = 0,
) -> None:
    mats = {key: get_decay_matrix(diags[key], token_span, heatmap_bins).numpy() for key in ["baseline", "ours"]}
    axis_size = mats["baseline"].shape[0]
    mask = np.tril(np.ones((axis_size, axis_size), dtype=bool))

    log_mats = {}
    for key, mat in mats.items():
        log_mat = np.log10(np.clip(mat, 1e-30, 1.0))
        log_mats[key] = np.ma.masked_where(~mask, log_mat)

    ratio = np.log10((mats["ours"] + 1e-30) / (mats["baseline"] + 1e-30))
    ratio = np.ma.masked_where(~mask, np.clip(ratio, -2.0, 8.0))

    retention_cmap = plt.get_cmap("magma").copy()
    retention_cmap.set_bad("#F4F4F4")
    ratio_cmap = plt.get_cmap("RdBu_r").copy()
    ratio_cmap.set_bad("#F4F4F4")
    if regular_map is None:
        fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.55), constrained_layout=True)
        retention_axes = axes[:2]
        shared_value_axes = retention_axes
        ratio_ax = axes[2]
    else:
        fig, axes = plt.subplots(1, 4, figsize=(14.2, 3.55), constrained_layout=True)
        regular_log = np.log10(np.clip(regular_map.numpy(), 1e-8, 1.0))
        regular_log = np.ma.masked_where(~mask, regular_log)
        regular_title = "Regular full attention" if regular_head < 0 else f"Regular full attention head {regular_head}"
        im_regular = axes[0].imshow(regular_log, origin="lower", cmap=retention_cmap, vmin=-14.0, vmax=0.0, aspect="equal")
        axes[0].set_title(regular_title)
        axes[0].set_xlabel(f"key offset in final {token_span} tokens")
        axes[0].set_ylabel("query position")
        set_block_ticks(axes[0], axis_size, token_span)
        add_lag_guides(axes[0], axis_size, token_span, color="white", text_color="#1A1A1A")
        retention_axes = axes[1:3]
        shared_value_axes = axes[:3]
        ratio_ax = axes[3]

    for ax, key in zip(retention_axes, ["baseline", "ours"]):
        im_ret = ax.imshow(log_mats[key], origin="lower", cmap=retention_cmap, vmin=-14.0, vmax=0.0, aspect="equal")
        ax.set_title(diags[key].label)
        ax.set_xlabel(f"key offset in final {token_span} tokens")
        ax.set_ylabel("query position")
        set_block_ticks(ax, axis_size, token_span)
        add_lag_guides(ax, axis_size, token_span, color="white", text_color="#1A1A1A")

    im_ratio = ratio_ax.imshow(
        ratio,
        origin="lower",
        cmap=ratio_cmap,
        norm=TwoSlopeNorm(vmin=-2.0, vcenter=0.0, vmax=8.0),
        aspect="equal",
    )
    ratio_ax.set_title("Ours / baseline")
    ratio_ax.set_xlabel(f"key offset in final {token_span} tokens")
    ratio_ax.set_ylabel("query position")
    set_block_ticks(ratio_ax, axis_size, token_span)
    add_lag_guides(ratio_ax, axis_size, token_span, color="#222222", text_color="#1A1A1A")

    cbar_ret = fig.colorbar(im_ret, ax=shared_value_axes, fraction=0.035, pad=0.018)
    cbar_ret.set_label("shared log10 scale")
    cbar_ratio = fig.colorbar(im_ratio, ax=ratio_ax, fraction=0.046, pad=0.02)
    cbar_ratio.set_label("log10 ratio")
    title = f"SSM retention over the final {token_span} tokens ({axis_size} sampled positions)"
    if regular_map is not None:
        title = "Regular attention and SSM retention over the final " + f"{token_span} tokens ({axis_size} sampled positions)"
    fig.suptitle(title)
    savefig(fig, output_dir / "02_ssm_decay_heatmaps.png")


def plot_effective_heatmaps(diags: dict[str, MambaDiagnostics], output_dir: Path, token_span: int, heatmap_bins: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), constrained_layout=True)
    for ax, key in zip(axes, ["baseline", "ours"]):
        mat = get_effective_matrix(diags[key], token_span, heatmap_bins).numpy()
        axis_size = mat.shape[0]
        nonzero = mat[mat > 0]
        vmax = float(np.quantile(nonzero, 0.997)) if nonzero.size else 1.0
        im = ax.imshow(mat, origin="lower", cmap="cividis", norm=PowerNorm(gamma=0.45, vmin=0.0, vmax=vmax), aspect="equal")
        ax.set_title(diags[key].label)
        ax.set_xlabel(f"key offset in final {token_span} tokens")
        ax.set_ylabel("query position in final block")
        set_block_ticks(ax, axis_size, token_span)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(f"Effective SSM mixing over the final {token_span} tokens")
    savefig(fig, output_dir / "03_ssm_effective_heatmaps.png")


def plot_regular_mamba_ours_history_map(
    regular_map: torch.Tensor,
    diags: dict[str, MambaDiagnostics],
    output_dir: Path,
    token_span: int,
    heatmap_bins: int,
    regular_head: int,
) -> None:
    regular_title = "Regular\nfull attention" if regular_head < 0 else f"Regular\nfull attention head {regular_head}"
    mats = {
        regular_title: regular_map.numpy(),
        "Mamba2 baseline\nSSM effective": get_effective_matrix(diags["baseline"], token_span, heatmap_bins).numpy(),
        "Ours: GQA-SWA + Mamba2\nSSM effective": get_effective_matrix(diags["ours"], token_span, heatmap_bins).numpy(),
    }
    axis_size = next(iter(mats.values())).shape[0]
    mask = np.tril(np.ones((axis_size, axis_size), dtype=bool))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#F4F4F4")

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.65), constrained_layout=True)
    im = None
    for ax, (title, mat) in zip(axes, mats.items()):
        log_mat = np.log10(np.clip(mat, 1e-8, 1.0))
        log_mat = np.ma.masked_where(~mask, log_mat)
        im = ax.imshow(log_mat, origin="lower", cmap=cmap, vmin=-6.0, vmax=0.0, aspect="equal")
        ax.set_title(title)
        ax.set_xlabel(f"key offset in final {token_span} tokens")
        ax.set_ylabel("query offset")
        set_block_ticks(ax, axis_size, token_span)
        add_lag_guides(ax, axis_size, token_span, color="white", text_color="#1A1A1A")
    cbar = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.018)
    cbar.set_label("log10 row-normalized displayed contribution")
    fig.suptitle(f"Regular attention vs Mamba SSM routing over the final {token_span} tokens")
    savefig(fig, output_dir / "09_regular_mamba_ours_history_map.png")


def plot_profiles(diags: dict[str, MambaDiagnostics], attn_profiles: dict[str, tuple[np.ndarray, np.ndarray]], output_dir: Path, max_lag: int, query_count: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), constrained_layout=True)
    for key in ["baseline", "ours"]:
        lags, decay, eff = lag_profiles(diags[key], max_lag, query_count)
        axes[0].plot(lags, decay + 1e-12, label=diags[key].label, color=diags[key].color, lw=1.8)
        axes[1].plot(lags, eff + 1e-12, label=diags[key].label, color=diags[key].color, lw=1.8)
    for ax, title, ylabel in [
        (axes[0], "SSM retention by lag", "mean decay"),
        (axes[1], "Effective SSM contribution by lag", "mean |CB| dt decay"),
    ]:
        ax.set_yscale("log")
        ax.set_xlim(0, max_lag)
        ax.set_xlabel("relative distance t-s")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which="both", lw=0.4, alpha=0.35)
        ax.legend(frameon=False)
    savefig(fig, output_dir / "04_ssm_lag_profiles.png")

    fig, ax = plt.subplots(figsize=(6.5, 3.1), constrained_layout=True)
    for key, (lags, vals) in attn_profiles.items():
        meta = RUNS[key]
        ax.plot(lags, vals + 1e-12, label=meta["label"], color=meta["color"], lw=1.8)
    ax.set_yscale("log")
    ax.set_xlim(0, max_lag)
    ax.set_xlabel("relative distance t-s")
    ax.set_ylabel("mean local-attention weight")
    ax.set_title("SWA local mixer profile")
    ax.grid(True, which="both", lw=0.4, alpha=0.35)
    ax.legend(frameon=False)
    savefig(fig, output_dir / "05_swa_lag_profiles.png")


def plot_half_life(models: dict[str, GPT], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 3.4), constrained_layout=True)
    offsets = {"baseline": -0.12, "ours": 0.12}
    for key in ["baseline", "ours"]:
        layers, hl = intrinsic_half_life_by_layer(models[key])
        jitter = np.random.default_rng(0).normal(0, 0.025, size=layers.shape)
        ax.scatter(layers + offsets[key] + jitter, hl, s=11, alpha=0.65, color=RUNS[key]["color"], label=RUNS[key]["label"], edgecolor="none")
        med = [np.median(hl[layers == l]) for l in sorted(set(layers.tolist()))]
        ax.plot(sorted(set(layers.tolist())), med, color=RUNS[key]["color"], lw=2.0)
    ax.set_yscale("log")
    ax.set_xlabel("layer")
    ax.set_ylabel("intrinsic SSM half-life (tokens)")
    ax.set_title("Learned SSM timescales from A and dt_bias")
    ax.grid(True, which="both", lw=0.4, alpha=0.35)
    ax.legend(frameon=False)
    savefig(fig, output_dir / "06_intrinsic_half_life.png")


def value_at_lag(lags: np.ndarray, vals: np.ndarray, lag: int) -> float:
    idx = int(np.argmin(np.abs(lags - lag)))
    return float(vals[idx])


def write_summary_metrics(
    diags: dict[str, MambaDiagnostics],
    models: dict[str, GPT],
    attn_profiles: dict[str, tuple[np.ndarray, np.ndarray]],
    output_dir: Path,
    max_lag: int,
    query_count: int,
) -> None:
    selected_lags = [0, 16, 64, 128, 256, 512, 1024, 2048, 4096, 8191, max_lag]
    selected_lags = sorted({lag for lag in selected_lags if 0 <= lag <= max_lag})
    summary: dict[str, object] = {
        "notes": {
            "decay": "mean_h exp(sum_{i=s+1..t} dt_i,h A_h), averaged over the final query block",
            "effective_ssm": "mean |C_t B_s| * mean_h(dt_s,h decay_h(t,s)), not row-normalized in the lag profile",
            "swa": "mean local attention probability by relative distance",
        },
        "lags": selected_lags,
        "mamba": {},
        "swa": {},
    }
    for key in ["baseline", "ours"]:
        lags, decay, eff = lag_profiles(diags[key], max_lag, query_count)
        layers, hl = intrinsic_half_life_by_layer(models[key])
        summary["mamba"][key] = {
            "label": RUNS[key]["label"],
            "half_life_median": float(np.median(hl)),
            "half_life_q25": float(np.quantile(hl, 0.25)),
            "half_life_q75": float(np.quantile(hl, 0.75)),
            "decay_by_lag": {str(lag): value_at_lag(lags, decay, lag) for lag in selected_lags},
            "effective_by_lag": {str(lag): value_at_lag(lags, eff, lag) for lag in selected_lags},
        }
    for key, (lags, vals) in attn_profiles.items():
        summary["swa"][key] = {
            "label": RUNS[key]["label"],
            "attention_by_lag": {str(lag): value_at_lag(lags, vals, lag) for lag in selected_lags},
        }

    def ratio(num: float, den: float) -> float | None:
        return None if den == 0 else num / den

    ratios = {}
    for lag in selected_lags:
        lag_s = str(lag)
        ratios[lag_s] = {
            "ours_over_baseline_decay": ratio(
                summary["mamba"]["ours"]["decay_by_lag"][lag_s],
                summary["mamba"]["baseline"]["decay_by_lag"][lag_s],
            ),
            "ours_over_baseline_effective": ratio(
                summary["mamba"]["ours"]["effective_by_lag"][lag_s],
                summary["mamba"]["baseline"]["effective_by_lag"][lag_s],
            ),
        }
    summary["ratios"] = ratios

    output_json = output_dir / "summary_metrics.json"
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "| lag | baseline decay | ours decay | decay ratio | baseline effective | ours effective | effective ratio |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for lag in selected_lags:
        lag_s = str(lag)
        bd = summary["mamba"]["baseline"]["decay_by_lag"][lag_s]
        od = summary["mamba"]["ours"]["decay_by_lag"][lag_s]
        be = summary["mamba"]["baseline"]["effective_by_lag"][lag_s]
        oe = summary["mamba"]["ours"]["effective_by_lag"][lag_s]
        rd = ratios[lag_s]["ours_over_baseline_decay"]
        re = ratios[lag_s]["ours_over_baseline_effective"]
        lines.append(f"| {lag} | {bd:.3e} | {od:.3e} | {rd:.3e} | {be:.3e} | {oe:.3e} | {re:.3e} |")
    (output_dir / "summary_metrics_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--offset", type=int, default=4096)
    parser.add_argument("--view-size", type=int, default=8192, help="Token span covered by SSM heatmaps.")
    parser.add_argument("--heatmap-bins", type=int, default=1024, help="Sampled positions per heatmap axis.")
    parser.add_argument("--profile-max-lag", type=int, default=8191)
    parser.add_argument("--query-count", type=int, default=512)
    parser.add_argument("--regular-head", type=int, default=0, help="Regular attention head for the unified map; use -1 to average all heads.")
    parser.add_argument("--only-decay-heatmap", action="store_true", help="Only render 02_ssm_decay_heatmaps and skip the remaining figures.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "logs" / "visualization")
    parser.add_argument("--val-bin", type=Path, default=REPO_ROOT / "data" / "fineweb10B" / "fineweb_val_000000.bin")
    args = parser.parse_args()

    setup_style()
    device = torch.device(args.device)
    full_idx = load_tokens(args.val_bin, args.seq_len, args.offset)

    # We analyze the last query_count tokens and lags up to profile_max_lag.
    # For SWA models, include a warmup context equal to the local attention window.
    profile_max_lag = min(args.profile_max_lag, args.seq_len - 1)
    warmup = 512
    needed = profile_max_lag + args.query_count + warmup
    sub_start = max(0, args.seq_len - needed)
    keep_start = max(0, args.seq_len - (profile_max_lag + args.query_count) - sub_start)
    idx = full_idx[:, sub_start:].to(device)
    abs_start = args.offset + sub_start

    models: dict[str, GPT] = {}
    model_keys = ["baseline", "ours", "regular"] if args.only_decay_heatmap else ["baseline", "ours", "mha_swa", "regular"]
    for key in model_keys:
        print(f"Loading {RUNS[key]['path']}")
        models[key] = load_model(RUNS[key]["path"], device)

    diags = {
        key: extract_mamba_diagnostics(
            models[key],
            idx,
            abs_start,
            keep_start,
            RUNS[key]["label"],
            RUNS[key]["color"],
        )
        for key in ["baseline", "ours"]
    }

    regular_map = regular_attention_matrix(models["regular"], idx, args.view_size, args.heatmap_bins, args.regular_head)
    if args.only_decay_heatmap:
        plot_decay_heatmaps(diags, args.output_dir, args.view_size, args.heatmap_bins, regular_map, args.regular_head)
        print(f"Saved decay heatmap to {args.output_dir}")
        return

    attn_profiles = {}
    for key in ["ours", "mha_swa"]:
        lags, vals = local_attention_lag_profile(models[key], idx, abs_start, profile_max_lag, args.query_count)
        attn_profiles[key] = (lags, vals)

    plot_mechanism_overview(args.output_dir)
    plot_decay_heatmaps(diags, args.output_dir, args.view_size, args.heatmap_bins, regular_map, args.regular_head)
    plot_effective_heatmaps(diags, args.output_dir, args.view_size, args.heatmap_bins)
    plot_regular_mamba_ours_history_map(regular_map, diags, args.output_dir, args.view_size, args.heatmap_bins, args.regular_head)
    plot_profiles(diags, attn_profiles, args.output_dir, profile_max_lag, args.query_count)
    plot_half_life(models, args.output_dir)
    write_summary_metrics(diags, models, attn_profiles, args.output_dir, profile_max_lag, args.query_count)

    print(f"Saved figures to {args.output_dir}")


if __name__ == "__main__":
    main()
