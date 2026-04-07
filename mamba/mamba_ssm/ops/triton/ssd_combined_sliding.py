# Copyright (c) 2024, Tri Dao, Albert Gu.

from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange

from mamba_ssm.ops.triton.ssd_combined import ssd_selective_scan
from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn

try:
    from transformers.modeling_flash_attention_utils import (
        _flash_attention_forward,
        is_flash_attn_available,
    )
except ImportError:
    _flash_attention_forward = None

    def is_flash_attn_available() -> bool:
        return False

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:
    SDPBackend, sdpa_kernel = None, None


def _sdpa_math_context():
    if sdpa_kernel is None or SDPBackend is None:
        return nullcontext()
    return sdpa_kernel(backends=[SDPBackend.MATH])


def _build_sliding_window_mask(
    seqlen: int,
    window_size: int,
    device: torch.device,
    seq_idx: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if seq_idx is not None:
        assert seq_idx.dim() == 2, "seq_idx must have shape (batch, seqlen)"
        assert seq_idx.shape[1] == seqlen, "seq_idx.shape[1] must match input seqlen"
    q_pos = torch.arange(seqlen, device=device)
    kv_pos = torch.arange(seqlen, device=device)
    rel = q_pos[:, None] - kv_pos[None, :]
    base_mask = (rel >= 0) & (rel < window_size)
    if seq_idx is None:
        return base_mask.unsqueeze(0).unsqueeze(0)
    same_seq = seq_idx[:, :, None] == seq_idx[:, None, :]
    return (same_seq & base_mask.unsqueeze(0)).unsqueeze(1)


def _reshape_projected(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    return rearrange(x, "b s (h d) -> b h s d", h=num_heads)


def _normalize_attn_input(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMS normalization with gradient stability for fp16."""
    # For fp16, clamp to safe range before normalization
    if x.dtype == torch.float16:
        # Use smaller range for fp16 safety
        x = torch.clamp(x, min=-65000.0, max=65000.0)
    
    # Use in-place operations to save memory
    norm = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return x / norm


def _index_first_axis(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(-1, *tensor.shape[2:])[indices]


def _flatten_seq_idx(seq_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    assert seq_idx.dim() == 2, "seq_idx must have shape (batch, seqlen)"
    batch, seqlen = seq_idx.shape
    flat_seq = seq_idx.reshape(-1)
    valid = flat_seq >= 0
    indices = valid.nonzero(as_tuple=False).flatten()
    if indices.numel() == 0:
        return indices, torch.zeros(1, dtype=torch.int32, device=seq_idx.device), 0
    batch_ids = torch.arange(batch, device=seq_idx.device, dtype=seq_idx.dtype)
    flat_batch = batch_ids[:, None].expand(batch, seqlen).reshape(-1)[indices]
    flat_seq = flat_seq[indices]
    seq_starts = torch.ones(indices.shape[0], dtype=torch.bool, device=seq_idx.device)
    seq_starts[1:] = (flat_seq[1:] != flat_seq[:-1]) | (flat_batch[1:] != flat_batch[:-1])
    start_idx = seq_starts.nonzero(as_tuple=False).flatten()
    lengths = torch.diff(
        torch.cat([start_idx, start_idx.new_tensor([indices.shape[0]])])
    ).to(torch.int32)
    cu_seqlens = F.pad(torch.cumsum(lengths, dim=0), (1, 0))
    return indices, cu_seqlens, int(lengths.max().item())


def _require_flash_attention() -> None:
    if _flash_attention_forward is None or not is_flash_attn_available():
        raise RuntimeError(
            "sliding_window_attention requires flash-attn via "
            "transformers.modeling_flash_attention_utils, but flash-attn is not available."
        )


def sliding_window_attention_flash(
    x_bsd: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    out_proj_weight: torch.Tensor,
    q_proj_bias: Optional[torch.Tensor] = None,
    k_proj_bias: Optional[torch.Tensor] = None,
    v_proj_bias: Optional[torch.Tensor] = None,
    out_proj_bias: Optional[torch.Tensor] = None,
    num_heads: int = 1,
    num_kv_heads: int = 1,
    window_size: int = 512,
    seq_idx: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """
    Sliding-window causal attention backed by the flash-attn kernel.

    Argument:
        x_bsd: (batch, seqlen, dim)
        seq_idx: (batch, seqlen), int32. Tokens with seq_idx < 0 are treated as padding.
    Return:
        out: (batch, seqlen, dim)
    """
    _require_flash_attention()
    batch, seqlen, dim = x_bsd.shape
    assert dim % num_heads == 0, "x_bsd.shape[-1] must be divisible by num_heads"
    assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
    head_dim = dim // num_heads
    kv_dim = num_kv_heads * head_dim
    assert k_proj_weight.shape[0] == kv_dim, "k_proj_weight output dim must match num_kv_heads * head_dim"
    assert v_proj_weight.shape[0] == kv_dim, "v_proj_weight output dim must match num_kv_heads * head_dim"

    # Normalize input with stability checks
    x_bsd = _normalize_attn_input(x_bsd)
    
    # Project Q, K, V
    q = F.linear(x_bsd, q_proj_weight, q_proj_bias).view(batch, seqlen, num_heads, head_dim)
    k = F.linear(x_bsd, k_proj_weight, k_proj_bias).view(batch, seqlen, num_kv_heads, head_dim)
    v = F.linear(x_bsd, v_proj_weight, v_proj_bias).view(batch, seqlen, num_kv_heads, head_dim)
    
    # CRITICAL: Delete x_bsd to free memory before attention
    del x_bsd

    flash_kwargs = dict(
        attention_mask=None,
        query_length=seqlen,
        is_causal=True,
        dropout=dropout_p if training else 0.0,
        softmax_scale=head_dim**-0.5,
        sliding_window=window_size,
        attn_implementation="flash_attention_2",
    )
    if seq_idx is None:
        y = _flash_attention_forward(q, k, v, **flash_kwargs)
    else:
        indices, cu_seqlens, max_seqlen = _flatten_seq_idx(seq_idx)
        if indices.numel() == 0:
            y = torch.zeros_like(q)  # Better for gradients
        else:
            q_unpad = _index_first_axis(q, indices).unsqueeze(0)
            k_unpad = _index_first_axis(k, indices).unsqueeze(0)
            v_unpad = _index_first_axis(v, indices).unsqueeze(0)
            
            # Free memory before flash attention
            del indices
            
            y_unpad = _flash_attention_forward(
                q_unpad,
                k_unpad,
                v_unpad,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                **flash_kwargs,
            )
            
            # Free unpadded tensors
            del q_unpad, k_unpad, v_unpad
            
            y = torch.zeros(batch * seqlen, num_heads, head_dim, device=q.device, dtype=q.dtype)
            # Create indices again (small memory cost)
            indices, _, _ = _flatten_seq_idx(seq_idx)
            y[indices] = y_unpad.squeeze(0)
            del y_unpad, indices
            y = y.view(batch, seqlen, num_heads, head_dim)
    
    # Free q, k, v before output projection
    del q, k, v

    y = rearrange(y, "b s h d -> b s (h d)")
    out = F.linear(y, out_proj_weight, out_proj_bias)
    
    # Free y before return
    del y
    
    return out


def sliding_window_attention_ref(
    x_bsd: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    out_proj_weight: torch.Tensor,
    q_proj_bias: Optional[torch.Tensor] = None,
    k_proj_bias: Optional[torch.Tensor] = None,
    v_proj_bias: Optional[torch.Tensor] = None,
    out_proj_bias: Optional[torch.Tensor] = None,
    num_heads: int = 1,
    num_kv_heads: int = 1,
    window_size: int = 4,
    seq_idx: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """
    Argument:
        x_bsd: (batch, seqlen, dim)
        q_proj_weight: (dim, dim)
        k_proj_weight: (kv_dim, dim)
        v_proj_weight: (kv_dim, dim)
        out_proj_weight: (dim, dim)
        seq_idx: (batch, seqlen), int32
    Return:
        out: (batch, seqlen, dim)
    """
    batch, seqlen, dim = x_bsd.shape
    assert dim % num_heads == 0, "x_bsd.shape[-1] must be divisible by num_heads"
    assert dim % num_kv_heads == 0, "x_bsd.shape[-1] must be divisible by num_kv_heads"
    assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen), "seq_idx must have shape (batch, seqlen)"

    x_bsd = _normalize_attn_input(x_bsd)
    q = _reshape_projected(F.linear(x_bsd, q_proj_weight, q_proj_bias), num_heads)
    k = _reshape_projected(F.linear(x_bsd, k_proj_weight, k_proj_bias), num_kv_heads)
    v = _reshape_projected(F.linear(x_bsd, v_proj_weight, v_proj_bias), num_kv_heads)
    mask = _build_sliding_window_mask(seqlen, window_size, x_bsd.device, seq_idx=seq_idx)
    # Masked local attention + GQA can dispatch to a different SDPA backend in eval/no_grad,
    # which led to validation-only numerical regressions. Force the stable math backend here.
    ctx = _sdpa_math_context() if not training else nullcontext()
    with ctx:
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=dropout_p if training else 0.0,
            is_causal=False,
            enable_gqa=num_heads != num_kv_heads,
        )
    y = rearrange(y, "b h s d -> b s (h d)")
    return F.linear(y, out_proj_weight, out_proj_bias)


# 对于序列长度 <= 1500，使用 PyTorch 原生实现更省内存
# 对于序列长度 > 1500，使用 Flash Attention 更快
def _get_sliding_window_attention_impl():
    """根据可用性和序列长度自动选择最佳实现"""
    import os
    
    # 可通过环境变量强制选择实现
    force_impl = os.environ.get('SLIDING_ATTN_IMPL', 'auto').lower()
    
    if force_impl == 'flash':
        if is_flash_attn_available():
            print("🚀 使用 Flash Attention 实现")
            return sliding_window_attention_flash
        else:
            print("⚠️  Flash Attention 不可用，回退到参考实现")
            return sliding_window_attention_ref
    elif force_impl == 'ref':
        print("📘 使用 PyTorch 参考实现（更省内存）")
        return sliding_window_attention_ref
    else:  # auto
        if is_flash_attn_available():
            print("🔄 自动模式：将根据序列长度选择实现")
            # 返回一个自适应包装器
            def adaptive_attention(x_bsd, *args, window_size=None, **kwargs):
                seqlen = x_bsd.shape[1]
                # 对于 <= 1500 的序列，用参考实现更省内存
                if seqlen <= 1500:
                    return sliding_window_attention_ref(x_bsd, *args, window_size=window_size, **kwargs)
                else:
                    return sliding_window_attention_flash(x_bsd, *args, window_size=window_size, **kwargs)
            return adaptive_attention
        else:
            print("📘 使用 PyTorch 参考实现（Flash Attention 不可用）")
            return sliding_window_attention_ref

sliding_window_attention = _get_sliding_window_attention_impl()


def sliding_window_attention_step(
    x_bd: torch.Tensor,
    attn_state: dict,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    out_proj_weight: torch.Tensor,
    q_proj_bias: Optional[torch.Tensor] = None,
    k_proj_bias: Optional[torch.Tensor] = None,
    v_proj_bias: Optional[torch.Tensor] = None,
    out_proj_bias: Optional[torch.Tensor] = None,
    num_heads: int = 1,
    num_kv_heads: int = 1,
) -> torch.Tensor:
    """Single-step sliding window attention for inference."""
    _, dim = x_bd.shape
    assert dim % num_heads == 0, "x_bd.shape[-1] must be divisible by num_heads"
    assert dim % num_kv_heads == 0, "x_bd.shape[-1] must be divisible by num_kv_heads"
    assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

    # Normalize with stability
    x_bd = _normalize_attn_input(x_bd)
    
    q = rearrange(F.linear(x_bd, q_proj_weight, q_proj_bias), "b (h d) -> b h 1 d", h=num_heads)
    k_new = rearrange(F.linear(x_bd, k_proj_weight, k_proj_bias), "b (h d) -> b h 1 d", h=num_kv_heads)
    v_new = rearrange(F.linear(x_bd, v_proj_weight, v_proj_bias), "b (h d) -> b h 1 d", h=num_kv_heads)
    
    # Free x_bd
    del x_bd

    k_cache = attn_state["k_cache"]
    v_cache = attn_state["v_cache"]
    filled = int(attn_state["filled"].item())
    window_size = k_cache.shape[2]
    if filled < window_size:
        k_cache[:, :, filled:filled + 1, :].copy_(k_new)
        v_cache[:, :, filled:filled + 1, :].copy_(v_new)
        filled += 1
    else:
        k_cache[:, :, :-1, :].copy_(k_cache[:, :, 1:, :])
        v_cache[:, :, :-1, :].copy_(v_cache[:, :, 1:, :])
        k_cache[:, :, -1:, :].copy_(k_new)
        v_cache[:, :, -1:, :].copy_(v_new)
    attn_state["filled"].fill_(filled)

    k = k_cache[:, :, :filled, :]
    v = v_cache[:, :, :filled, :]
    y = F.scaled_dot_product_attention(
        q,
        k,
        v,
        dropout_p=0.0,
        is_causal=False,
        enable_gqa=num_heads != num_kv_heads,
    )
    y = rearrange(y, "b h s d -> b s (h d)").squeeze(1)
    return F.linear(y, out_proj_weight, out_proj_bias)


def prefill_sliding_window_cache(
    x_bsd: torch.Tensor,
    attn_state: dict,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    k_proj_bias: Optional[torch.Tensor] = None,
    v_proj_bias: Optional[torch.Tensor] = None,
    num_kv_heads: int = 1,
) -> None:
    x_bsd = _normalize_attn_input(x_bsd)
    k = _reshape_projected(F.linear(x_bsd, k_proj_weight, k_proj_bias), num_kv_heads)
    v = _reshape_projected(F.linear(x_bsd, v_proj_weight, v_proj_bias), num_kv_heads)
    window_size = attn_state["k_cache"].shape[2]
    keep = min(window_size, x_bsd.shape[1])
    attn_state["k_cache"].zero_()
    attn_state["v_cache"].zero_()
    attn_state["k_cache"][:, :, :keep, :].copy_(k[:, :, -keep:, :])
    attn_state["v_cache"][:, :, :keep, :].copy_(v[:, :, -keep:, :])
    attn_state["filled"].fill_(keep)


def mamba_sliding_scan_ref(
    xBC,
    dt,
    A,
    chunk_size,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    out_proj_weight_attn: torch.Tensor,
    D=None,
    z=None,
    dt_bias=None,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
    q_proj_bias: Optional[torch.Tensor] = None,
    k_proj_bias: Optional[torch.Tensor] = None,
    v_proj_bias: Optional[torch.Tensor] = None,
    out_proj_bias_attn: Optional[torch.Tensor] = None,
    headdim=None,
    ngroups: int = 1,
    window_size: int = 4,
    num_heads: int = 1,
    num_kv_heads: int = 1,
    seq_idx: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Argument:
        xBC: (batch, seqlen, dim + 2 * ngroups * dstate) where dim == nheads * headdim
        dt: (batch, seqlen, nheads) or (batch, seqlen, nheads, headdim)
        A: (nheads)
        D: (nheads, headdim) or (nheads,)
        z: (batch, seqlen, dim)
        dt_bias: (nheads) or (nheads, headdim)
        seq_idx: (batch, seqlen), int32
        headdim: if D is 1D and z is None, headdim must be passed in
    Return:
        out: (batch, seqlen, dim)
    """
    del chunk_size
    batch, seqlen, nheads = dt.shape[:3]
    assert nheads % ngroups == 0
    if z is not None:
        dim = z.shape[-1]
        assert dim % nheads == 0
        headdim = dim // nheads
    else:
        if D.dim() == 1:
            assert headdim is not None
        else:
            headdim = D.shape[1]
        dim = nheads * headdim
    xBC = sliding_window_attention_ref(
        xBC,
        q_proj_weight=q_proj_weight,
        k_proj_weight=k_proj_weight,
        v_proj_weight=v_proj_weight,
        out_proj_weight=out_proj_weight_attn,
        q_proj_bias=q_proj_bias,
        k_proj_bias=k_proj_bias,
        v_proj_bias=v_proj_bias,
        out_proj_bias=out_proj_bias_attn,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        window_size=window_size,
        seq_idx=seq_idx,
    )
    dstate = (xBC.shape[-1] - dim) // ngroups // 2
    x, B, C = torch.split(xBC, [dim, ngroups * dstate, ngroups * dstate], dim=-1)
    x = rearrange(x, "b l (h p) -> b l h p", h=nheads)
    B = rearrange(B, "b l (g n) -> b l g n", g=ngroups)
    C = rearrange(C, "b l (g n) -> b l g n", g=ngroups)
    z = rearrange(z, "b l (h p) -> b l h p", h=nheads) if z is not None else None
    out = ssd_selective_scan(
        x, dt.to(x.dtype), A, B, C, D=D.float(), z=z,
        dt_bias=dt_bias, dt_softplus=dt_softplus, dt_limit=dt_limit
    )
    return rearrange(out, "b s h p -> b s (h p)")


def mamba_split_sliding_scan_ref(
    zxbcdt: torch.Tensor,
    dt_bias: torch.Tensor,
    A: torch.Tensor,
    D: torch.Tensor,
    chunk_size: int,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    out_proj_weight_attn: torch.Tensor,
    q_proj_bias: Optional[torch.Tensor] = None,
    k_proj_bias: Optional[torch.Tensor] = None,
    v_proj_bias: Optional[torch.Tensor] = None,
    out_proj_bias_attn: Optional[torch.Tensor] = None,
    dt_limit=(0.0, float("inf")),
    rmsnorm_weight: Optional[torch.Tensor] = None,
    rmsnorm_eps: float = 1e-6,
    outproj_weight: Optional[torch.Tensor] = None,
    outproj_bias: Optional[torch.Tensor] = None,
    headdim: Optional[int] = None,
    ngroups: int = 1,
    norm_before_gate: bool = True,
    window_size: int = 4,
    num_heads: int = 1,
    num_kv_heads: int = 1,
    seq_idx: Optional[torch.Tensor] = None,
):
    """
    Argument:
        zxbcdt: (batch, seqlen, 2 * dim + 2 * ngroups * dstate + nheads) where dim == nheads * headdim
        dt_bias: (nheads,)
        A: (nheads)
        D: (nheads, headdim) or (nheads,)
        rmsnorm_weight: (dim,)
        outproj_weight: (out_dim, dim)
        outproj_bias: (out_dim,)
        seq_idx: (batch, seqlen), int32
        headdim: if D is 1D, headdim must be passed in
        norm_before_gate: if True, we do RMSNorm(x) * F.silu(z). If False, we do RMSNorm(x * F.silu(z))
    Return:
        out: (batch, seqlen, dim)
    """
    if D.dim() == 1:
        assert headdim is not None
        nheads, = D.shape
    else:
        nheads, headdim = D.shape
    assert nheads % ngroups == 0
    batch, seqlen, _ = zxbcdt.shape
    dim = nheads * headdim
    dstate = (zxbcdt.shape[-1] - 2 * dim - nheads) // ngroups // 2
    assert zxbcdt.shape == (batch, seqlen, 2 * dim + 2 * ngroups * dstate + nheads)
    assert dt_bias.shape == (nheads,)
    assert A.shape == (nheads,)
    if rmsnorm_weight is not None:
        assert rmsnorm_weight.shape == (dim,)
    z, xBC, dt = torch.split(zxbcdt, [dim, dim + 2 * ngroups * dstate, nheads], dim=-1)
    xBC = sliding_window_attention_ref(
        xBC,
        q_proj_weight=q_proj_weight,
        k_proj_weight=k_proj_weight,
        v_proj_weight=v_proj_weight,
        out_proj_weight=out_proj_weight_attn,
        q_proj_bias=q_proj_bias,
        k_proj_bias=k_proj_bias,
        v_proj_bias=v_proj_bias,
        out_proj_bias=out_proj_bias_attn,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        window_size=window_size,
        seq_idx=seq_idx,
    )
    x, B, C = torch.split(xBC, [dim, ngroups * dstate, ngroups * dstate], dim=-1)
    x = rearrange(x, "b l (h p) -> b l h p", h=nheads)
    B = rearrange(B, "b l (g n) -> b l g n", g=ngroups)
    C = rearrange(C, "b l (g n) -> b l g n", g=ngroups)
    z = rearrange(z, "b l (h p) -> b l h p", h=nheads)
    out = ssd_selective_scan(x, dt.to(x.dtype), A, B, C, D=D.float(),
                             z=z if rmsnorm_weight is None else None, dt_bias=dt_bias, dt_softplus=True, dt_limit=dt_limit)
    out = rearrange(out, "b s h p -> b s (h p)")
    if rmsnorm_weight is not None:
        out = rmsnorm_fn(out, rmsnorm_weight, None, z=rearrange(z, "b l h p -> b l (h p)"), eps=rmsnorm_eps,
                         norm_before_gate=norm_before_gate)
    if outproj_weight is not None:
        out = F.linear(out, outproj_weight, outproj_bias)
    return out
