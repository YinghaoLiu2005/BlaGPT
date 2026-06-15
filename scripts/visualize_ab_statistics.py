#!/usr/bin/env python
"""Statistics and visualisation of Mamba2 A / B / dt / C parameter ranges.

Usage:
    python scripts/visualize_ab_statistics.py
    python scripts/visualize_ab_statistics.py --only-layer 0
    python scripts/visualize_ab_statistics.py --seq-len 4096 --output-dir logs/ab_stats
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/data/yinghaoliu/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "bla_gpt"))
sys.path.insert(0, str(REPO_ROOT / "mamba"))

from bla_gpt import GPT, GPTConfig  # noqa: E402


# ---------------------------------------------------------------------------
# Model registry  – same paths as visualize_mamba_mechanisms.py
# ---------------------------------------------------------------------------

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
}

LEGACY_ATTENTION_NAMES = {
    "mamba2_new_v3": "mamba2_GQA",
    "mamba2_new_v5": "mamba2_MHA",
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

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
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def clean_state_dict(state_dict: dict) -> dict:
    return {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}


def load_model(ckpt_path: Path, device: torch.device) -> GPT:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    config = GPTConfig()
    for key, value in ckpt["model_config"].items():
        if hasattr(config, key):
            if key == "attention":
                value = LEGACY_ATTENTION_NAMES.get(value, value)
            setattr(config, key, value)
    model = GPT(config)
    model.load_state_dict(clean_state_dict(ckpt["model"]), strict=True)
    model.eval().to(device)
    return model


def load_tokens(val_bin: Path, seq_len: int, offset: int) -> torch.Tensor:
    with val_bin.open("rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        tokens = np.frombuffer(f.read(), dtype=np.uint16).astype(np.int64)
    return torch.from_numpy(tokens[offset : offset + seq_len]).long()[None, :]


def rmsnorm(x, weight, eps):
    y = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return (y * weight.float()).to(x.dtype)


def apply_rotary(x, inv_freq, abs_start):
    seq_len = x.shape[1]
    pos = torch.arange(abs_start, abs_start + seq_len, device=x.device, dtype=inv_freq.dtype)
    freqs = torch.outer(pos, inv_freq.to(x.device))
    cos, sin = freqs.cos()[None, :, None, :], freqs.sin()[None, :, None, :]
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half].float(), x[..., half:].float()
    return torch.cat([x1 * cos + x2 * sin, -x1 * sin + x2 * cos], -1).type_as(x)


# ---------------------------------------------------------------------------
# Extract A, B, C, dt across ALL layers
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_all_layers(
    model: GPT,
    idx: torch.Tensor,
    abs_start: int,
    only_layer: int | None = None,
) -> dict:
    """Return dict with A, B, C, dt, dt_bias, A_log for each mamba layer.

    Shapes:
        A:        (n_heads, d_state)          – static parameter
        B:        (T, n_groups, d_state)       – input-dependent, one forward pass
        C:        (T, n_groups, d_state)       – input-dependent
        dt:       (T, n_heads)                 – input-dependent
        dt_bias:  (n_heads,)                   – static parameter
        A_log:    (n_heads, d_state)           – static parameter
    """
    device = idx.device
    results = {}

    # Forward through embedding + first layernorm
    x = model.transformer.wte(idx)
    if model.transformer.wpe is not None:
        pos = torch.arange(abs_start, abs_start + idx.shape[1], device=device, dtype=torch.long)
        x = x + model.transformer.wpe(pos)
    x = model.transformer.drop(x)

    for layer_idx, block in enumerate(model.transformer.h):
        if not hasattr(block.attn, "mamba"):
            # Skip non-mamba layers (e.g. regular attention)
            # Still need to forward through them for correct input to next layer
            residual = x
            h = block.ln_1(x)
            if hasattr(block.attn, '_project_query'):
                # regular attention – just pass through
                h = block.attn(h, None)
            else:
                h = block.attn(h)
            x = residual + h
            if hasattr(block, 'ln_2'):
                residual = x
                x = residual + block.mlp(block.ln_2(x))
            continue

        mamba = block.attn.mamba

        # layernorm
        x_norm = block.ln_1(x)

        # SWA front-end (for GQA / MHA variants)
        if hasattr(mamba, 'input_q_proj'):
            attn_in = rmsnorm(x_norm, mamba.input_norm.weight, mamba.input_norm.eps) if hasattr(mamba, 'input_norm') else x_norm
            _, T, _ = attn_in.shape
            q = mamba.input_q_proj(attn_in).view(1, T, mamba.attn_nheads, mamba.headdim)
            kv_nheads = getattr(mamba, 'attn_kv_nheads', mamba.attn_nheads)
            k = mamba.input_k_proj(attn_in).view(1, T, kv_nheads, mamba.headdim)
            v = mamba.input_v_proj(attn_in).view(1, T, kv_nheads, mamba.headdim)
            q = apply_rotary(q, mamba.rotary_inv_freq, abs_start)
            k = apply_rotary(k, mamba.rotary_inv_freq, abs_start)
            repeat = max(1, mamba.attn_nheads // kv_nheads)
            k = k.repeat_interleave(repeat, dim=2)
            v = v.repeat_interleave(repeat, dim=2)
            # Use FlashAttention or manual SWA
            try:
                from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
            except ImportError:
                pass
            # Manual causal SWA for extraction
            scale = 1.0 / (mamba.headdim ** 0.5)
            window = int(mamba.window_size)
            scores = torch.einsum("bthd,bshd->bhts", q.float(), k.float()) * scale
            key_pos = torch.arange(T, device=device)
            causal_mask = key_pos[None, None, :] <= key_pos[None, None, :, None].transpose(-1, -2)
            # Actually build proper mask: (1, 1, T, T) where [t, s] = (s <= t) and (s >= t - window + 1)
            q_pos = key_pos[:, None]
            k_pos = key_pos[None, :]
            mask_2d = (k_pos <= q_pos) & (k_pos >= q_pos - window + 1)
            scores = scores.masked_fill(~mask_2d[None, None], float("-inf"))
            probs = torch.softmax(scores, -1).to(v.dtype)
            attn_out = torch.einsum("bhts,bshd->bthd", probs, v)
            attn_out = attn_out.reshape(1, T, mamba.d_model)
            u = x_norm + mamba.input_out_proj(attn_out)
        else:
            u = x_norm

        # in_proj
        zxbcdt = mamba.in_proj(u)
        d_mlp = (zxbcdt.shape[-1] - 2 * mamba.d_ssm - 2 * mamba.ngroups * mamba.d_state - mamba.nheads) // 2
        z0, x0, z, xBC, dt_pre = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, mamba.d_ssm, mamba.d_ssm + 2 * mamba.ngroups * mamba.d_state, mamba.nheads],
            dim=-1,
        )

        # conv1d or SiLU
        if hasattr(mamba, "conv1d"):
            xBC = F.conv1d(
                xBC.transpose(1, 2), mamba.conv1d.weight, bias=mamba.conv1d.bias,
                padding=mamba.d_conv - 1, groups=mamba.conv1d.groups,
            ).transpose(1, 2)[:, :xBC.shape[1]]
        xBC = F.silu(xBC)

        _, B, C = torch.split(
            xBC,
            [mamba.d_ssm, mamba.ngroups * mamba.d_state, mamba.ngroups * mamba.d_state],
            dim=-1,
        )
        B = B.view(1, -1, mamba.ngroups, mamba.d_state)[0].float().cpu()
        C = C.view(1, -1, mamba.ngroups, mamba.d_state)[0].float().cpu()
        dt = F.softplus(dt_pre + mamba.dt_bias.to(dtype=dt_pre.dtype))[0].float().cpu()
        A = -torch.exp(mamba.A_log.float()).cpu()
        A_log = mamba.A_log.detach().cpu().float()
        dt_bias = mamba.dt_bias.detach().cpu().float()

        # Ensure A and A_log are 2D: (n_heads, d_state)
        # Some model variants store A_log as 1D (n_heads,) with per-head scalar A
        if A.dim() == 1:
            A = A.unsqueeze(-1)
            A_log = A_log.unsqueeze(-1)

        results[layer_idx] = {
            "A": A,              # (n_heads, d_state)
            "A_log": A_log,      # (n_heads, d_state)
            "B": B,              # (T, n_groups, d_state)
            "C": C,              # (T, n_groups, d_state)
            "dt": dt,            # (T, n_heads)
            "dt_bias": dt_bias,  # (n_heads,)
            "|A|": A.abs(),      # (n_heads, d_state)  -- derived, for plotting
            "|B|": B.abs(),      # (T, n_groups, d_state)  -- derived, for plotting
        }

        if only_layer is not None and layer_idx >= only_layer:
            break

        # Continue forward through the rest of the block
        # We need to apply out_proj, gate, and residual + MLP
        # Simplified: just run the full block forward
        # Actually let's just run the full block to get correct hidden states
        # Re-do the forward properly
        residual = x
        # Re-run the full attention block
        h = block.ln_1(x)
        h = block.attn(h)
        x = residual + h
        if hasattr(block, 'ln_2'):
            residual = x
            x = residual + block.mlp(block.ln_2(x))

    return results


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_statistics(all_data: dict) -> dict:
    """Compute per-layer statistics for A, B, C, dt."""
    stats = {}
    for layer_idx, data in all_data.items():
        A = data["A"]           # (H, d_state)
        B = data["B"]           # (T, G, d_state)
        C = data["C"]           # (T, G, d_state)
        dt = data["dt"]         # (T, H)
        dt_bias = data["dt_bias"]  # (H,)
        A_log = data["A_log"]   # (H, d_state)

        stats[layer_idx] = {
            "A": {
                "mean": float(A.mean()),
                "std": float(A.std()),
                "min": float(A.min()),
                "max": float(A.max()),
                "median": float(A.median()),
                "q05": float(A.quantile(0.05)),
                "q95": float(A.quantile(0.95)),
            },
            "A_log": {
                "mean": float(A_log.mean()),
                "std": float(A_log.std()),
                "min": float(A_log.min()),
                "max": float(A_log.max()),
            },
            "|A|": {
                "mean": float(A.abs().mean()),
                "std": float(A.abs().std()),
                "min": float(A.abs().min()),
                "max": float(A.abs().max()),
                "median": float(A.abs().median()),
            },
            "B": {
                "mean": float(B.mean()),
                "std": float(B.std()),
                "min": float(B.min()),
                "max": float(B.max()),
                "median": float(B.median()),
                "q05": float(B.quantile(0.05)),
                "q95": float(B.quantile(0.95)),
            },
            "|B|": {
                "mean": float(B.abs().mean()),
                "std": float(B.abs().std()),
                "min": float(B.abs().min()),
                "max": float(B.abs().max()),
            },
            "C": {
                "mean": float(C.mean()),
                "std": float(C.std()),
                "min": float(C.min()),
                "max": float(C.max()),
                "q05": float(C.quantile(0.05)),
                "q95": float(C.quantile(0.95)),
            },
            "dt": {
                "mean": float(dt.mean()),
                "std": float(dt.std()),
                "min": float(dt.min()),
                "max": float(dt.max()),
                "median": float(dt.median()),
            },
            "dt_bias": {
                "mean": float(dt_bias.mean()),
                "std": float(dt_bias.std()),
                "min": float(dt_bias.min()),
                "max": float(dt_bias.max()),
            },
            "dt*A": {
                "mean": float((dt[:, :, None] * A[None, :, :]).mean()),
                "std": float((dt[:, :, None] * A[None, :, :]).std()),
                "min": float((dt[:, :, None] * A[None, :, :]).min()),
                "max": float((dt[:, :, None] * A[None, :, :]).max()),
            },
        }
    return stats


def print_statistics(stats: dict, model_name: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {model_name}")
    print(f"{'='*70}")
    for layer_idx in sorted(stats.keys()):
        s = stats[layer_idx]
        print(f"\n--- Layer {layer_idx} ---")
        for param in ["A", "A_log", "|A|", "B", "|B|", "C", "dt", "dt_bias", "dt*A"]:
            vals = s[param]
            parts = [f"{k}={v:.4f}" for k, v in vals.items()]
            print(f"  {param:>8s}:  {'  '.join(parts)}")


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_A_distribution(all_data: dict, output_dir: Path, model_name: str, color: str) -> None:
    """Histogram of A values across all layers and heads."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)

    # Collect A across layers
    A_all = []
    A_log_all = []
    A_abs_all = []
    layer_labels = []
    for layer_idx in sorted(all_data.keys()):
        A = all_data[layer_idx]["A"]
        A_all.append(A.flatten().numpy())
        A_log_all.append(all_data[layer_idx]["A_log"].flatten().numpy())
        A_abs_all.append(A.abs().flatten().numpy())

    A_cat = np.concatenate(A_all)
    A_log_cat = np.concatenate(A_log_all)
    A_abs_cat = np.concatenate(A_abs_all)

    # A values
    axes[0].hist(A_cat, bins=100, color=color, alpha=0.8, edgecolor="none")
    axes[0].axvline(0, color="red", ls="--", lw=0.8)
    axes[0].set_xlabel("A = -exp(A_log)")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"A distribution (mean={A_cat.mean():.3f}, std={A_cat.std():.3f})")

    # A_log values
    axes[1].hist(A_log_cat, bins=100, color=color, alpha=0.8, edgecolor="none")
    axes[1].set_xlabel("A_log")
    axes[1].set_ylabel("count")
    axes[1].set_title(f"A_log distribution (mean={A_log_cat.mean():.3f}, std={A_log_cat.std():.3f})")

    # |A| values
    axes[2].hist(A_abs_cat, bins=100, color=color, alpha=0.8, edgecolor="none")
    axes[2].set_xlabel("|A|")
    axes[2].set_ylabel("count")
    axes[2].set_title(f"|A| distribution (mean={A_abs_cat.mean():.3f}, median={np.median(A_abs_cat):.3f})")

    fig.suptitle(f"{model_name}: A parameter distribution across all layers")
    savefig(fig, output_dir / f"A_distribution_{model_name.replace(' ', '_')}.png")


def plot_B_distribution(all_data: dict, output_dir: Path, model_name: str, color: str) -> None:
    """Histogram of B values across all layers (sampled from one forward pass)."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)

    B_all = []
    B_abs_all = []
    for layer_idx in sorted(all_data.keys()):
        B = all_data[layer_idx]["B"]
        # Sample to avoid huge arrays
        B_flat = B.reshape(-1)
        if B_flat.numel() > 100000:
            idx = torch.randperm(B_flat.numel())[:100000]
            B_flat = B_flat[idx]
        B_all.append(B_flat.numpy())
        B_abs_all.append(B_flat.abs().numpy())

    B_cat = np.concatenate(B_all)
    B_abs_cat = np.concatenate(B_abs_all)

    axes[0].hist(B_cat, bins=100, color=color, alpha=0.8, edgecolor="none")
    axes[0].axvline(0, color="red", ls="--", lw=0.8)
    axes[0].set_xlabel("B value")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"B distribution (mean={B_cat.mean():.4f}, std={B_cat.std():.4f})")

    axes[1].hist(B_abs_cat, bins=100, color=color, alpha=0.8, edgecolor="none")
    axes[1].set_xlabel("|B|")
    axes[1].set_ylabel("count")
    axes[1].set_title(f"|B| distribution (mean={B_abs_cat.mean():.4f}, median={np.median(B_abs_cat):.4f})")

    fig.suptitle(f"{model_name}: B distribution (input-dependent, one forward pass)")
    savefig(fig, output_dir / f"B_distribution_{model_name.replace(' ', '_')}.png")


def plot_dt_distribution(all_data: dict, output_dir: Path, model_name: str, color: str) -> None:
    """Histogram of dt and dt_bias."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)

    dt_all = []
    dt_bias_all = []
    dtA_all = []
    for layer_idx in sorted(all_data.keys()):
        dt = all_data[layer_idx]["dt"]
        dt_flat = dt.reshape(-1)
        if dt_flat.numel() > 100000:
            idx = torch.randperm(dt_flat.numel())[:100000]
            dt_flat = dt_flat[idx]
        dt_all.append(dt_flat.numpy())
        dt_bias_all.append(all_data[layer_idx]["dt_bias"].numpy())
        # dt * A averaged over d_state
        A = all_data[layer_idx]["A"]  # (H, d_state)
        dtA = (dt[:, :, None] * A[None, :, :]).mean(dim=-1)  # (T, H)
        dtA_flat = dtA.reshape(-1)
        if dtA_flat.numel() > 100000:
            idx = torch.randperm(dtA_flat.numel())[:100000]
            dtA_flat = dtA_flat[idx]
        dtA_all.append(dtA_flat.numpy())

    dt_cat = np.concatenate(dt_all)
    dt_bias_cat = np.concatenate(dt_bias_all)
    dtA_cat = np.concatenate(dtA_all)

    axes[0].hist(dt_cat, bins=100, color=color, alpha=0.8, edgecolor="none")
    axes[0].set_xlabel("dt (discretization step)")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"dt distribution (mean={dt_cat.mean():.4f}, std={dt_cat.std():.4f})")

    axes[1].hist(dt_bias_cat, bins=50, color=color, alpha=0.8, edgecolor="none")
    axes[1].set_xlabel("dt_bias (learned)")
    axes[1].set_ylabel("count")
    axes[1].set_title(f"dt_bias distribution (mean={dt_bias_cat.mean():.4f})")

    axes[2].hist(dtA_cat, bins=100, color=color, alpha=0.8, edgecolor="none")
    axes[2].axvline(0, color="red", ls="--", lw=0.8)
    axes[2].set_xlabel("dt * A (mean over d_state)")
    axes[2].set_ylabel("count")
    axes[2].set_title(f"dt*A distribution (mean={dtA_cat.mean():.4f}, std={dtA_cat.std():.4f})")

    fig.suptitle(f"{model_name}: dt and dt*A distribution")
    savefig(fig, output_dir / f"dt_distribution_{model_name.replace(' ', '_')}.png")


def plot_per_layer_box(
    all_data: dict,
    param_name: str,
    output_dir: Path,
    model_name: str,
    color: str,
    ylabel: str | None = None,
) -> None:
    """Box plot of a parameter across layers."""
    fig, ax = plt.subplots(figsize=(max(6, len(all_data) * 0.6), 3.5), constrained_layout=True)

    data_list = []
    labels = []
    for layer_idx in sorted(all_data.keys()):
        val = all_data[layer_idx][param_name]
        flat = val.reshape(-1).numpy()
        if flat.size > 50000:
            flat = np.random.default_rng(0).choice(flat, 50000, replace=False)
        data_list.append(flat)
        labels.append(str(layer_idx))

    bp = ax.boxplot(data_list, labels=labels, patch_artist=True, showfliers=False,
                    boxprops=dict(facecolor=color, alpha=0.6),
                    medianprops=dict(color="black", lw=1.5))
    ax.set_xlabel("layer")
    ax.set_ylabel(ylabel or param_name)
    ax.set_title(f"{model_name}: {param_name} per layer")
    ax.grid(True, axis="y", lw=0.3, alpha=0.4)
    savefig(fig, output_dir / f"{param_name}_per_layer_{model_name.replace(' ', '_')}.png")


def plot_A_heatmap(all_data: dict, output_dir: Path, model_name: str) -> None:
    """Heatmap: |A| for each (layer, head), averaged over d_state."""
    n_layers = len(all_data)
    n_heads = all_data[0]["A"].shape[0]

    mat = np.zeros((n_layers, n_heads))
    for layer_idx, data in all_data.items():
        mat[layer_idx] = data["A"].abs().mean(dim=-1).numpy()

    fig, ax = plt.subplots(figsize=(max(6, n_heads * 0.4), max(3, n_layers * 0.4)), constrained_layout=True)
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    ax.set_xticks(range(n_heads))
    ax.set_yticks(range(n_layers))
    ax.set_title(f"{model_name}: mean |A| per (layer, head)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    savefig(fig, output_dir / f"A_heatmap_{model_name.replace(' ', '_')}.png")


def plot_dtA_heatmap(all_data: dict, output_dir: Path, model_name: str) -> None:
    """Heatmap: mean dt*A per (layer, head), averaged over d_state and tokens."""
    n_layers = len(all_data)
    n_heads = all_data[0]["A"].shape[0]

    mat = np.zeros((n_layers, n_heads))
    for layer_idx, data in all_data.items():
        A = data["A"]       # (H, d_state)
        dt = data["dt"]     # (T, H)
        dtA = (dt[:, :, None] * A[None, :, :]).mean(dim=(0, -1))  # (H,)
        mat[layer_idx] = dtA.numpy()

    fig, ax = plt.subplots(figsize=(max(6, n_heads * 0.4), max(3, n_layers * 0.4)), constrained_layout=True)
    # TwoSlopeNorm requires vmin < vcenter < vmax; fall back to normalisation when 0 is outside range
    if mat.min() < 0 < mat.max():
        norm = plt.matplotlib.colors.TwoSlopeNorm(vmin=mat.min(), vcenter=0, vmax=mat.max())
    else:
        norm = plt.matplotlib.colors.Normalize(vmin=mat.min(), vmax=mat.max())
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", norm=norm)
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    ax.set_xticks(range(n_heads))
    ax.set_yticks(range(n_layers))
    ax.set_title(f"{model_name}: mean dt*A per (layer, head) — negative = memory, positive = forget")
    fig.colorbar(im, ax=ax, fraction=0.046)
    savefig(fig, output_dir / f"dtA_heatmap_{model_name.replace(' ', '_')}.png")


def plot_half_life_heatmap(all_data: dict, output_dir: Path, model_name: str) -> None:
    """Heatmap: intrinsic half-life per (layer, head).

    half_life = ln(2) / (dt_bias * |A|)
    """
    n_layers = len(all_data)
    n_heads = all_data[0]["A"].shape[0]

    mat = np.zeros((n_layers, n_heads))
    for layer_idx, data in all_data.items():
        dt_bias = data["dt_bias"]    # (H,)
        A_abs = data["A"].abs()      # (H, d_state)
        A_mean = A_abs.mean(dim=-1)  # (H,)
        hl = np.log(2) / (dt_bias.numpy() * A_mean.numpy() + 1e-12)
        mat[layer_idx] = hl

    fig, ax = plt.subplots(figsize=(max(6, n_heads * 0.4), max(3, n_layers * 0.4)), constrained_layout=True)
    im = ax.imshow(mat, aspect="auto", cmap="magma_r")
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    ax.set_xticks(range(n_heads))
    ax.set_yticks(range(n_layers))
    ax.set_title(f"{model_name}: intrinsic half-life (tokens) per (layer, head)")
    fig.colorbar(im, ax=ax, fraction=0.046, label="tokens")
    savefig(fig, output_dir / f"half_life_heatmap_{model_name.replace(' ', '_')}.png")


# ---------------------------------------------------------------------------
# Comparison plot across models
# ---------------------------------------------------------------------------

def plot_comparison_box(
    all_models_data: dict[str, dict],
    param_name: str,
    output_dir: Path,
    ylabel: str | None = None,
) -> None:
    """Box plot comparing a parameter across models and layers."""
    model_keys = list(all_models_data.keys())
    n_models = len(model_keys)
    n_layers = max(len(d) for d in all_models_data.values())

    fig, ax = plt.subplots(figsize=(max(8, n_layers * n_models * 0.35), 4), constrained_layout=True)

    positions = []
    data_list = []
    colors = []
    tick_positions = []
    tick_labels = []

    for li in range(n_layers):
        base_pos = li * (n_models + 1)
        tick_positions.append(base_pos + n_models / 2)
        tick_labels.append(str(li))
        for mi, key in enumerate(model_keys):
            if li not in all_models_data[key]:
                continue
            pos = base_pos + mi
            positions.append(pos)
            val = all_models_data[key][li][param_name]
            flat = val.reshape(-1).numpy()
            if flat.size > 30000:
                flat = np.random.default_rng(0).choice(flat, 30000, replace=False)
            data_list.append(flat)
            colors.append(RUNS[key]["color"])

    bp = ax.boxplot(data_list, positions=positions, widths=0.7, patch_artist=True,
                    showfliers=False, medianprops=dict(color="black", lw=1.2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("layer")
    ax.set_ylabel(ylabel or param_name)
    ax.set_title(f"{param_name} per layer — comparison")
    ax.grid(True, axis="y", lw=0.3, alpha=0.4)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=RUNS[k]["color"], alpha=0.7, label=RUNS[k]["label"]) for k in model_keys]
    ax.legend(handles=legend_elements, loc="best", frameon=False)

    savefig(fig, output_dir / f"{param_name}_comparison.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Mamba2 A/B/dt/C parameter statistics")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--only-layer", type=int, default=None, help="Only extract up to this layer (for speed)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "logs" / "ab_statistics")
    parser.add_argument("--val-bin", type=Path, default=REPO_ROOT / "data" / "fineweb10B" / "fineweb_val_000000.bin")
    args = parser.parse_args()

    setup_style()
    device = torch.device(args.device)
    idx = load_tokens(args.val_bin, args.seq_len, args.offset).to(device)

    all_models_data: dict[str, dict] = {}

    for key in ["baseline", "ours", "mha_swa"]:
        print(f"\nLoading {RUNS[key]['label']} ...")
        model = load_model(RUNS[key]["path"], device)
        data = extract_all_layers(model, idx, args.offset, only_layer=args.only_layer)
        all_models_data[key] = data

        # Statistics
        stats = compute_statistics(data)
        print_statistics(stats, RUNS[key]["label"])

        # Individual plots
        model_name = RUNS[key]["label"]
        color = RUNS[key]["color"]
        plot_A_distribution(data, args.output_dir, model_name, color)
        plot_B_distribution(data, args.output_dir, model_name, color)
        plot_dt_distribution(data, args.output_dir, model_name, color)
        plot_per_layer_box(data, "|A|", args.output_dir, model_name, color, ylabel="mean |A| over d_state")
        plot_per_layer_box(data, "B", args.output_dir, model_name, color)
        plot_per_layer_box(data, "dt", args.output_dir, model_name, color)
        plot_A_heatmap(data, args.output_dir, model_name)
        plot_dtA_heatmap(data, args.output_dir, model_name)
        plot_half_life_heatmap(data, args.output_dir, model_name)

        # Free memory
        del model
        torch.cuda.empty_cache()

    # Comparison across models
    print("\nGenerating comparison plots ...")
    for param in ["A", "B", "dt"]:
        plot_comparison_box(all_models_data, param, args.output_dir)
    plot_comparison_box(all_models_data, "A_log", args.output_dir, ylabel="A_log")
    plot_comparison_box(all_models_data, "dt_bias", args.output_dir, ylabel="dt_bias")

    print(f"\nAll figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
