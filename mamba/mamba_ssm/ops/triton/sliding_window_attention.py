# Copyright (c) 2024, Tri Dao, Albert Gu.

import inspect
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange

try:
    from transformers.modeling_flash_attention_utils import (
        _flash_attention_forward,
        is_flash_attn_available,
    )
except ImportError:
    _flash_attention_forward = None

    def is_flash_attn_available() -> bool:
        return False


_FLASH_FORWARD_SIGNATURE = (
    inspect.signature(_flash_attention_forward).parameters if _flash_attention_forward is not None else {}
)
_FLASH_SUPPORTS_SLIDING_WINDOW = "sliding_window" in _FLASH_FORWARD_SIGNATURE
_FLASH_SUPPORTS_VARLEN = "cu_seq_lens_q" in _FLASH_FORWARD_SIGNATURE


def _require_flash_sliding_window(x: torch.Tensor) -> None:
    if _flash_attention_forward is None or not is_flash_attn_available():
        raise RuntimeError(
            "Sliding Mamba requires Flash Attention 2, but transformers Flash Attention utilities are unavailable."
        )
    if not _FLASH_SUPPORTS_SLIDING_WINDOW:
        raise RuntimeError(
            "Sliding Mamba requires a Flash Attention implementation with `sliding_window` support."
        )
    if x.device.type != "cuda":
        raise RuntimeError("Sliding Mamba requires CUDA tensors for flash-native sparse sliding-window attention.")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError(
            "Sliding Mamba requires float16 or bfloat16 inputs for flash-native sparse sliding-window attention."
        )


def _require_flash_varlen() -> None:
    if not _FLASH_SUPPORTS_VARLEN:
        raise RuntimeError(
            "Sliding Mamba varlen attention requires Flash Attention support for `cu_seq_lens_q` / `cu_seq_lens_k`."
        )


def _reshape_projected(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    return rearrange(x, "b s (h d) -> b h s d", h=num_heads)


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
    lengths = torch.diff(torch.cat([start_idx, start_idx.new_tensor([indices.shape[0]])])).to(torch.int32)
    cu_seqlens = F.pad(torch.cumsum(lengths, dim=0, dtype=torch.int32), (1, 0))
    return indices, cu_seqlens, int(lengths.max().item())


def _validate_even_head_dim(head_dim: int) -> None:
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE requires even head_dim, but got head_dim={head_dim}.")


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # q/k: (b, s, h, d)
    # cos/sin: (b, s, d)
    cos = cos.unsqueeze(2)  # (b, s, 1, d)
    sin = sin.unsqueeze(2)  # (b, s, 1, d)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def gather_cos_sin_by_position_ids(
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Accept cos/sin in shapes:
    # - (max_seq_len, head_dim)
    # - (1, max_seq_len, head_dim)
    if cos.dim() == 3:
        if cos.shape[0] != 1:
            raise ValueError(f"Expected cos shape (1, S, D) when 3D, got {tuple(cos.shape)}")
        cos = cos.squeeze(0)
    if sin.dim() == 3:
        if sin.shape[0] != 1:
            raise ValueError(f"Expected sin shape (1, S, D) when 3D, got {tuple(sin.shape)}")
        sin = sin.squeeze(0)

    if cos.dim() != 2 or sin.dim() != 2:
        raise ValueError(f"Expected cos/sin 2D after squeeze, got cos={cos.dim()}D, sin={sin.dim()}D")
    if cos.shape != sin.shape:
        raise ValueError(f"cos/sin shape mismatch: {tuple(cos.shape)} vs {tuple(sin.shape)}")

    # position_ids expected shape: (b, s) or (s,)
    if position_ids.dim() == 1:
        position_ids = position_ids.unsqueeze(0)
    if position_ids.dim() != 2:
        raise ValueError(f"position_ids must be 1D or 2D, got shape={tuple(position_ids.shape)}")

    # Guard invalid negative ids (e.g. -1 mask values)
    if (position_ids < 0).any():
        raise ValueError("position_ids contains negative values (e.g. -1), which are invalid for RoPE gather.")

    max_seq_len = cos.shape[0]
    if (position_ids >= max_seq_len).any():
        raise ValueError(
            f"position_ids contains values outside [0, {max_seq_len - 1}] for provided cos/sin cache."
        )

    return cos[position_ids], sin[position_ids]


def _build_flash_kwargs(
    *,
    query_length: int,
    head_dim: int,
    window_size: int,
    dropout_p: float,
    training: bool,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
) -> dict:
    kwargs = dict(
        attention_mask=None,
        query_length=query_length,
        is_causal=True,
        dropout=dropout_p if training else 0.0,
        softmax_scale=head_dim**-0.5,
        sliding_window=window_size,
        attn_implementation="flash_attention_2",
    )
    if cu_seqlens is not None:
        _require_flash_varlen()
        kwargs.update(
            cu_seq_lens_q=cu_seqlens,
            cu_seq_lens_k=cu_seqlens,
            max_length_q=max_seqlen,
            max_length_k=max_seqlen,
        )
    return kwargs


def sliding_window_attention(
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
    cu_seqlens: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    cos: Optional[torch.Tensor] = None,
    sin: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    _require_flash_sliding_window(x_bsd)
    if seq_idx is not None and cu_seqlens is not None:
        raise ValueError("Pass only one of seq_idx or cu_seqlens to sliding_window_attention.")

    batch, seqlen, dim = x_bsd.shape
    assert dim % num_heads == 0, "x_bsd.shape[-1] must be divisible by num_heads"
    assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
    head_dim = dim // num_heads
    _validate_even_head_dim(head_dim)

    kv_dim = num_kv_heads * head_dim
    assert k_proj_weight.shape[0] == kv_dim
    assert v_proj_weight.shape[0] == kv_dim

    q = F.linear(x_bsd, q_proj_weight, q_proj_bias).view(batch, seqlen, num_heads, head_dim)
    k = F.linear(x_bsd, k_proj_weight, k_proj_bias).view(batch, seqlen, num_kv_heads, head_dim)
    v = F.linear(x_bsd, v_proj_weight, v_proj_bias).view(batch, seqlen, num_kv_heads, head_dim)

    if cos is not None or sin is not None:
        if cos is None or sin is None:
            raise ValueError("Pass both cos and sin for RoPE.")
        if position_ids is None:
            raise ValueError("position_ids is required when applying RoPE.")
        cos_g, sin_g = gather_cos_sin_by_position_ids(cos, sin, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos_g, sin_g)

    if cu_seqlens is not None:
        if batch != 1:
            raise ValueError("Packed cu_seqlens attention expects input shaped as a single flattened batch.")
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        max_seqlen = int(lengths.max().item()) if lengths.numel() > 0 else 0
        y = _flash_attention_forward(
            q,
            k,
            v,
            **_build_flash_kwargs(
                query_length=seqlen,
                head_dim=head_dim,
                window_size=window_size,
                dropout_p=dropout_p,
                training=training,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            ),
        )
    elif seq_idx is None:
        y = _flash_attention_forward(
            q,
            k,
            v,
            **_build_flash_kwargs(
                query_length=seqlen,
                head_dim=head_dim,
                window_size=window_size,
                dropout_p=dropout_p,
                training=training,
            ),
        )
    else:
        indices, packed_cu_seqlens, max_seqlen = _flatten_seq_idx(seq_idx)
        if indices.numel() == 0:
            y = x_bsd.new_zeros(batch, seqlen, num_heads, head_dim)
        else:
            q_packed = _index_first_axis(q, indices).unsqueeze(0)
            k_packed = _index_first_axis(k, indices).unsqueeze(0)
            v_packed = _index_first_axis(v, indices).unsqueeze(0)
            y_packed = _flash_attention_forward(
                q_packed,
                k_packed,
                v_packed,
                **_build_flash_kwargs(
                    query_length=q_packed.shape[1],
                    head_dim=head_dim,
                    window_size=window_size,
                    dropout_p=dropout_p,
                    training=training,
                    cu_seqlens=packed_cu_seqlens,
                    max_seqlen=max_seqlen,
                ),
            )
            y = x_bsd.new_zeros(batch * seqlen, num_heads, head_dim)
            y[indices] = y_packed.squeeze(0)
            y = y.view(batch, seqlen, num_heads, head_dim)

    y = rearrange(y, "b s h d -> b s (h d)")
    return F.linear(y, out_proj_weight, out_proj_bias)


def prefill_sliding_window_cache(
    x_bsd: torch.Tensor,
    attn_state: dict,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    k_proj_bias: Optional[torch.Tensor] = None,
    v_proj_bias: Optional[torch.Tensor] = None,
    num_kv_heads: int = 1,
    cu_seqlens: Optional[torch.Tensor] = None,
    seq_idx: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    cos: Optional[torch.Tensor] = None,
    sin: Optional[torch.Tensor] = None,
) -> None:
    window_size = attn_state["k_cache"].shape[2]
    attn_state["k_cache"].zero_()
    attn_state["v_cache"].zero_()
    attn_state["filled"].zero_()

    head_dim = k_proj_weight.shape[0] // num_kv_heads
    _validate_even_head_dim(head_dim)

    if cu_seqlens is None and seq_idx is None:
        # Explicitly use b,s,h,d to avoid layout ambiguity
        k = F.linear(x_bsd, k_proj_weight, k_proj_bias).view(x_bsd.shape[0], x_bsd.shape[1], num_kv_heads, head_dim)
        v = F.linear(x_bsd, v_proj_weight, v_proj_bias).view(x_bsd.shape[0], x_bsd.shape[1], num_kv_heads, head_dim)

        if cos is not None or sin is not None:
            if cos is None or sin is None:
                raise ValueError("Pass both cos and sin for RoPE.")
            if position_ids is None:
                raise ValueError("position_ids is required when applying RoPE.")
            cos_g, sin_g = gather_cos_sin_by_position_ids(cos, sin, position_ids)
            # Only K uses RoPE in cache prefill
            k, _ = apply_rotary_pos_emb(k, k, cos_g, sin_g)

        k = rearrange(k, "b s h d -> b h s d")
        v = rearrange(v, "b s h d -> b h s d")

        keep = min(window_size, x_bsd.shape[1])
        attn_state["k_cache"][:, :, :keep, :].copy_(k[:, :, -keep:, :])
        attn_state["v_cache"][:, :, :keep, :].copy_(v[:, :, -keep:, :])
        attn_state["filled"].fill_(keep)
        return

    if cu_seqlens is None:
        indices, cu_seqlens, _ = _flatten_seq_idx(seq_idx)
        x_flat = x_bsd.reshape(-1, x_bsd.shape[-1])[indices]
        if position_ids is not None:
            flat_pos = position_ids.reshape(-1)[indices]
            pos_for_packed = flat_pos.unsqueeze(0)  # (1, S)
        else:
            pos_for_packed = None
    else:
        assert x_bsd.shape[0] == 1, "Packed sliding-window cache prefill expects a single flattened batch"
        x_flat = x_bsd.squeeze(0)
        if position_ids is not None:
            pos_for_packed = position_ids
            if pos_for_packed.dim() == 1:
                pos_for_packed = pos_for_packed.unsqueeze(0)  # Ensure (1, S)
        else:
            pos_for_packed = None

    if x_flat.numel() == 0:
        return

    # k: (1, s, h, d), v: (h, s, d)
    k = rearrange(F.linear(x_flat, k_proj_weight, k_proj_bias), "s (h d) -> 1 s h d", h=num_kv_heads)
    v = rearrange(F.linear(x_flat, v_proj_weight, v_proj_bias), "s (h d) -> h s d", h=num_kv_heads)

    if cos is not None or sin is not None:
        if cos is None or sin is None:
            raise ValueError("Pass both cos and sin for RoPE.")
        if pos_for_packed is None:
            raise ValueError("position_ids is required when applying RoPE.")
        cos_g, sin_g = gather_cos_sin_by_position_ids(cos, sin, pos_for_packed)  # (1, s, d)
        k, _ = apply_rotary_pos_emb(k, k, cos_g, sin_g)

    k = rearrange(k, "1 s h d -> h s d")

    num_sequences = cu_seqlens.shape[0] - 1
    assert num_sequences <= attn_state["k_cache"].shape[0], "Cache batch size is smaller than packed sequence count"
    for seq in range(num_sequences):
        start, end = cu_seqlens[seq].item(), cu_seqlens[seq + 1].item()
        keep = min(window_size, end - start)
        if keep <= 0:
            continue
        attn_state["k_cache"][seq, :, :keep, :].copy_(k[:, end - keep:end, :])
        attn_state["v_cache"][seq, :, :keep, :].copy_(v[:, end - keep:end, :])
        attn_state["filled"][seq] = keep


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
    position_ids: Optional[torch.Tensor] = None,
    cos: Optional[torch.Tensor] = None,
    sin: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    batch, dim = x_bd.shape
    assert dim % num_heads == 0, "x_bd.shape[-1] must be divisible by num_heads"
    assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
    head_dim = dim // num_heads
    _validate_even_head_dim(head_dim)

    q = rearrange(F.linear(x_bd, q_proj_weight, q_proj_bias), "b (h d) -> b h 1 d", h=num_heads)
    k_new = rearrange(F.linear(x_bd, k_proj_weight, k_proj_bias), "b (h d) -> b h 1 d", h=num_kv_heads)
    v_new = rearrange(F.linear(x_bd, v_proj_weight, v_proj_bias), "b (h d) -> b h 1 d", h=num_kv_heads)

    if cos is not None or sin is not None:
        if cos is None or sin is None:
            raise ValueError("Pass both cos and sin for RoPE.")
        if position_ids is None:
            raise ValueError("position_ids is required when applying RoPE.")
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(1)  # (b, 1)
        cos_g, sin_g = gather_cos_sin_by_position_ids(cos, sin, position_ids)  # (b, 1, d)
        q_bt = q.transpose(1, 2)      # (b, 1, h, d)
        k_bt = k_new.transpose(1, 2)  # (b, 1, h_kv, d)
        q_bt, k_bt = apply_rotary_pos_emb(q_bt, k_bt, cos_g, sin_g)
        q = q_bt.transpose(1, 2)      # (b, h, 1, d)
        k_new = k_bt.transpose(1, 2)  # (b, h_kv, 1, d)

    k_cache = attn_state["k_cache"]
    v_cache = attn_state["v_cache"]
    filled = attn_state["filled"]
    window_size = k_cache.shape[2]

    outputs = []
    for b in range(batch):
        filled_b = int(filled[b].item())
        if filled_b < window_size:
            k_cache[b, :, filled_b : filled_b + 1, :].copy_(k_new[b])
            v_cache[b, :, filled_b : filled_b + 1, :].copy_(v_new[b])
            filled_b += 1
        else:
            k_cache[b, :, :-1, :].copy_(k_cache[b, :, 1:, :].clone())
            v_cache[b, :, :-1, :].copy_(v_cache[b, :, 1:, :].clone())
            k_cache[b, :, -1:, :].copy_(k_new[b])
            v_cache[b, :, -1:, :].copy_(v_new[b])
        filled[b] = filled_b

        k = k_cache[b : b + 1, :, :filled_b, :]
        v = v_cache[b : b + 1, :, :filled_b, :]
        y = F.scaled_dot_product_attention(
            q[b : b + 1],
            k,
            v,
            dropout_p=0.0,
            is_causal=False,
            enable_gqa=num_heads != num_kv_heads,
        )
        outputs.append(rearrange(y, "b h s d -> b s (h d)"))

    y = torch.cat(outputs, dim=0).reshape(batch, num_heads * head_dim)
    return F.linear(y, out_proj_weight, out_proj_bias)


def mamba_split_sliding_window_scan_combined(*args, **kwargs):
    raise RuntimeError(
        "mamba_split_sliding_window_scan_combined is intentionally disabled. "
        "Use Mamba2Sliding with the split `mamba_chunk_scan_combined` path instead."
    )