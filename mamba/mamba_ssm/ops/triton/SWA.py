from __future__ import annotations

import inspect
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    from transformers.cache_utils import Cache, CacheLayerMixin
except ImportError:
    Cache = object
    CacheLayerMixin = object

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


def _get_decoder_cfg(config):
    if hasattr(config, "get_text_config"):
        return config.get_text_config(decoder=True)
    return config


def _can_use_flash(x: torch.Tensor) -> bool:
    return (
        _flash_attention_forward is not None
        and is_flash_attn_available()
        and _FLASH_SUPPORTS_SLIDING_WINDOW
        and x.device.type == "cuda"
        and x.dtype in (torch.float16, torch.bfloat16)
    )


def _validate_even_head_dim(head_dim: int) -> None:
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE requires even head_dim, but got head_dim={head_dim}.")


def _rmsnorm_last_dim(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    if weight.dim() != 1 or weight.shape[0] != x.shape[-1]:
        raise ValueError(
            f"RMSNorm weight must have shape ({x.shape[-1]},), got {tuple(weight.shape)}"
        )
    x_float = x.float()
    rms = torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + eps)
    return (x_float * rms).to(dtype=x.dtype) * weight.to(dtype=x.dtype)


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
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def gather_cos_sin_by_position_ids(
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cos.dim() == 3:
        if cos.shape[0] != 1:
            raise ValueError(f"Expected cos shape (1, S, D) when 3D, got {tuple(cos.shape)}")
        cos = cos.squeeze(0)
    if sin.dim() == 3:
        if sin.shape[0] != 1:
            raise ValueError(f"Expected sin shape (1, S, D) when 3D, got {tuple(sin.shape)}")
        sin = sin.squeeze(0)
    if cos.dim() != 2 or sin.dim() != 2:
        raise ValueError(f"Expected 2D cos/sin after squeeze, got {cos.dim()}D and {sin.dim()}D")
    if cos.shape != sin.shape:
        raise ValueError(f"cos/sin shape mismatch: {tuple(cos.shape)} vs {tuple(sin.shape)}")
    if position_ids.dim() == 1:
        position_ids = position_ids.unsqueeze(0)
    if position_ids.dim() != 2:
        raise ValueError(f"position_ids must be 1D or 2D, got {tuple(position_ids.shape)}")
    if (position_ids < 0).any():
        raise ValueError("position_ids contains negative values.")
    max_seq_len = cos.shape[0]
    if (position_ids >= max_seq_len).any():
        raise ValueError(f"position_ids contains values outside [0, {max_seq_len - 1}]")
    return cos[position_ids], sin[position_ids]


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _repeat_kv_seq_major(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, seqlen, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, :, None, :].expand(
        batch, seqlen, num_key_value_heads, n_rep, head_dim
    )
    return hidden_states.reshape(batch, seqlen, num_key_value_heads * n_rep, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def _index_first_axis(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(-1, *tensor.shape[2:])[indices]


def _flatten_seq_idx(seq_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    if seq_idx.dim() != 2:
        raise ValueError(f"seq_idx must have shape (batch, seqlen), got {tuple(seq_idx.shape)}")
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
        if not _FLASH_SUPPORTS_VARLEN:
            raise RuntimeError("Flash Attention varlen support is unavailable in this environment.")
        kwargs.update(
            cu_seq_lens_q=cu_seqlens,
            cu_seq_lens_k=cu_seqlens,
            max_length_q=max_seqlen,
            max_length_k=max_seqlen,
        )
    return kwargs


def _build_local_causal_mask(
    seqlen_q: int,
    seqlen_k: int,
    window_size: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    q_pos = torch.arange(seqlen_q, device=device)[:, None]
    k_pos = torch.arange(seqlen_k, device=device)[None, :]
    visible = (k_pos <= q_pos) & (k_pos >= (q_pos - window_size + 1))
    mask = torch.full((seqlen_q, seqlen_k), torch.finfo(torch.float32).min, device=device)
    mask = torch.where(visible, torch.zeros_like(mask), mask)
    return mask


def _chunked_local_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    window_size: int,
    dropout_p: float,
    training: bool,
    chunk_size: int = 128,
) -> torch.Tensor:
    batch, num_heads, seqlen_q, head_dim = q.shape
    seqlen_k = k.shape[-2]
    out = torch.empty_like(q)
    scale = head_dim ** -0.5
    for q_start in range(0, seqlen_q, chunk_size):
        q_end = min(seqlen_q, q_start + chunk_size)
        k_start = max(0, q_start - window_size + 1)
        k_end = min(seqlen_k, q_end)

        q_chunk = q[:, :, q_start:q_end, :]
        k_chunk = k[:, :, k_start:k_end, :]
        v_chunk = v[:, :, k_start:k_end, :]

        q_abs = torch.arange(q_start, q_end, device=q.device)[:, None]
        k_abs = torch.arange(k_start, k_end, device=q.device)[None, :]
        visible = (k_abs <= q_abs) & (k_abs >= (q_abs - window_size + 1))
        mask = torch.full(
            (q_end - q_start, k_end - k_start),
            torch.finfo(torch.float32).min,
            device=q.device,
        )
        mask = torch.where(visible, torch.zeros_like(mask), mask)

        scores = torch.matmul(q_chunk.float(), k_chunk.float().transpose(-2, -1)) * scale
        scores = scores + mask.unsqueeze(0).unsqueeze(0)
        attn = torch.softmax(scores, dim=-1, dtype=torch.float32)
        if training and dropout_p > 0.0:
            attn = F.dropout(attn, p=dropout_p, training=True)
        out[:, :, q_start:q_end, :] = torch.matmul(attn, v_chunk.float()).to(dtype=q.dtype)
    return out


def _attention_fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    window_size: int,
    dropout_p: float,
    training: bool,
) -> torch.Tensor:
    n_rep = num_heads // num_kv_heads
    if n_rep != 1:
        k = _repeat_kv_seq_major(k, n_rep)
        v = _repeat_kv_seq_major(v, n_rep)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    y = _chunked_local_attention(
        q,
        k,
        v,
        window_size=window_size,
        dropout_p=dropout_p,
        training=training,
    )
    return y.transpose(1, 2)


def _run_dense_or_packed_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    window_size: int,
    cu_seqlens: Optional[torch.Tensor],
    dropout_p: float,
    training: bool,
) -> torch.Tensor:
    n_rep = num_heads // num_kv_heads
    if n_rep != 1:
        k = _repeat_kv_seq_major(k, n_rep)
        v = _repeat_kv_seq_major(v, n_rep)
        num_kv_heads = num_heads
    if cu_seqlens is None:
        if _can_use_flash(q):
            return _flash_attention_forward(
                q,
                k,
                v,
                **_build_flash_kwargs(
                    query_length=q.shape[1],
                    head_dim=q.shape[-1],
                    window_size=window_size,
                    dropout_p=dropout_p,
                    training=training,
                ),
            )
        return _attention_fallback(
            q,
            k,
            v,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            window_size=window_size,
            dropout_p=dropout_p,
            training=training,
        )

    _require_packed_length(q, cu_seqlens)
    if _can_use_flash(q):
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        max_seqlen = int(lengths.max().item()) if lengths.numel() > 0 else 0
        return _flash_attention_forward(
            q,
            k,
            v,
            **_build_flash_kwargs(
                query_length=q.shape[1],
                head_dim=q.shape[-1],
                window_size=window_size,
                dropout_p=dropout_p,
                training=training,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            ),
        )

    if q.shape[0] != 1:
        raise ValueError("Packed attention fallback expects input shaped as a single flattened batch.")
    outputs = []
    for seq in range(cu_seqlens.shape[0] - 1):
        start = int(cu_seqlens[seq].item())
        end = int(cu_seqlens[seq + 1].item())
        if end <= start:
            continue
        y_seq = _attention_fallback(
            q[:, start:end],
            k[:, start:end],
            v[:, start:end],
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            window_size=window_size,
            dropout_p=dropout_p,
            training=training,
        )
        outputs.append(y_seq)
    if not outputs:
        return q.new_zeros(q.shape[0], q.shape[1], q.shape[2], q.shape[3])
    return torch.cat(outputs, dim=1)


class StaticSlidingWindowLayerPrealloc(CacheLayerMixin):
    is_sliding = True

    def __init__(
        self,
        *,
        config,
        batch_size: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        zero_init: bool = False,
    ):
        super().__init__()
        cfg = _get_decoder_cfg(config)
        num_kv_heads = int(getattr(cfg, "num_key_value_heads", getattr(cfg, "num_attention_heads")))
        head_dim = int(getattr(cfg, "head_dim"))
        window = (
            getattr(cfg, "sliding_window", None)
            or getattr(cfg, "attention_chunk_size", None)
            or getattr(cfg, "max_position_embeddings", None)
        )
        if window is None or int(window) <= 0:
            raise ValueError("SWA requires a positive sliding_window / attention_chunk_size / max_position_embeddings")

        self.sliding_window = int(window)
        self.capacity = max(self.sliding_window - 1, 0)
        self.is_initialized = True
        self.dtype = dtype
        self.device = device
        self.batch_size = int(batch_size)
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.size = 0
        self.cumulative_length = 0

        alloc = torch.zeros if zero_init else torch.empty
        self._buf_keys = alloc(
            (self.batch_size, self.num_kv_heads, max(self.capacity, 1), self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self._buf_values = alloc(
            (self.batch_size, self.num_kv_heads, max(self.capacity, 1), self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self.keys = self._buf_keys[:, :, :0, :]
        self.values = self._buf_values[:, :, :0, :]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        conv_state: Optional[tuple] = None,
        recurrent_state: Optional[torch.Tensor] = None,
        cache_kwargs: Optional[dict] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if key_states.shape != value_states.shape:
            raise ValueError(f"K/V shapes must match, got {tuple(key_states.shape)} vs {tuple(value_states.shape)}")
        batch, num_kv_heads, q_len, head_dim = key_states.shape
        if batch != self.batch_size:
            raise ValueError(f"SWA cache batch_size={self.batch_size}, but got batch={batch}")
        if num_kv_heads != self.num_kv_heads or head_dim != self.head_dim:
            raise ValueError(
                f"SWA cache head mismatch, got heads={num_kv_heads}, dim={head_dim}, "
                f"expected heads={self.num_kv_heads}, dim={self.head_dim}"
            )

        prev_k = self.keys
        prev_v = self.values
        full_k = torch.cat([prev_k, key_states], dim=-2)
        full_v = torch.cat([prev_v, value_states], dim=-2)

        new_size = min(self.capacity, self.size + q_len)
        if new_size > 0:
            k_tail = full_k[:, :, -new_size:, :]
            v_tail = full_v[:, :, -new_size:, :]
            self._buf_keys[:, :, :new_size, :].copy_(k_tail)
            self._buf_values[:, :, :new_size, :].copy_(v_tail)
            self.keys = self._buf_keys[:, :, :new_size, :]
            self.values = self._buf_values[:, :, :new_size, :]
        else:
            self.keys = self._buf_keys[:, :, :0, :]
            self.values = self._buf_values[:, :, :0, :]

        self.size = int(new_size)
        self.cumulative_length += q_len
        return full_k, full_v

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        q_len = int(cache_position.shape[0])
        pre_cum = max(int(self.cumulative_length) - q_len, 0)
        kv_offset = max(pre_cum - self.sliding_window + 1, 0)
        if pre_cum >= self.sliding_window:
            kv_len = (self.sliding_window - 1) + q_len
        else:
            kv_len = pre_cum + q_len
        return kv_len, kv_offset

    def get_seq_length(self) -> int:
        return int(self.cumulative_length)

    def get_max_cache_shape(self) -> int:
        return int(self.sliding_window)

    def crop(self, max_length: int) -> None:
        if self.get_seq_length() >= self.sliding_window:
            raise ValueError("Cropping is forbidden after filling the sliding window.")
        if max_length < 0:
            new_size = max(0, self.size - abs(max_length))
        else:
            new_size = min(self.size, max_length)
        if new_size > 0:
            self._buf_keys[:, :, :new_size, :].copy_(self._buf_keys[:, :, self.size - new_size : self.size, :])
            self._buf_values[:, :, :new_size, :].copy_(self._buf_values[:, :, self.size - new_size : self.size, :])
        self.keys = self._buf_keys[:, :, :new_size, :]
        self.values = self._buf_values[:, :, :new_size, :]
        self.size = int(new_size)
        self.cumulative_length = int(self.size)

    def batch_repeat_interleave(self, repeats: int) -> None:
        if repeats != 1:
            raise RuntimeError("Static cache forbids changing batch size (repeat_interleave)")

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        if indices.numel() != self.batch_size:
            raise RuntimeError("Static cache forbids changing batch size (select_indices)")

    def lazy_initialization(self, *args, **kwargs):
        return


class StaticCachePrealloc(Cache):
    def __init__(
        self,
        *,
        config,
        batch_size: int = 1,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        zero_init: bool = False,
        offloading: bool = False,
        offload_only_non_sliding: bool = False,
    ):
        cfg = _get_decoder_cfg(config)
        num_hidden_layers = int(getattr(cfg, "num_hidden_layers"))
        layers = [
            StaticSlidingWindowLayerPrealloc(
                config=cfg,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
                zero_init=zero_init,
            )
            for _ in range(num_hidden_layers)
        ]
        super().__init__(layers=layers, offloading=offloading, offload_only_non_sliding=offload_only_non_sliding)

    def update(
        self,
        layer_idx: int,
        key_states: torch.Tensor = None,
        value_states: torch.Tensor = None,
        conv_state=None,
        recurrent_state=None,
        cache_kwargs=None,
    ):
        return self.layers[layer_idx].update(key_states, value_states, conv_state, recurrent_state, cache_kwargs)


def _sync_tail_to_state(
    attn_state: dict,
    keys: torch.Tensor,
    values: torch.Tensor,
    filled: torch.Tensor,
) -> None:
    attn_state["k_cache"].zero_()
    attn_state["v_cache"].zero_()
    batch = keys.shape[0]
    for b in range(batch):
        keep = int(filled[b].item())
        if keep > 0:
            attn_state["k_cache"][b : b + 1, :, :keep, :].copy_(keys[b : b + 1, :, -keep:, :])
            attn_state["v_cache"][b : b + 1, :, :keep, :].copy_(values[b : b + 1, :, -keep:, :])
    attn_state["filled"].copy_(filled.to(dtype=attn_state["filled"].dtype, device=attn_state["filled"].device))


def _reset_state(attn_state: dict) -> None:
    attn_state["k_cache"].zero_()
    attn_state["v_cache"].zero_()
    attn_state["filled"].zero_()


def _get_past_capacity(attn_state: dict) -> int:
    # The external Mamba cache is allocated with logical window size W.
    # Following the InfiniteVL update semantics, the persistent state stores only W - 1 past tokens.
    return max(attn_state["k_cache"].shape[2] - 1, 0)


def _require_packed_length(x_bshd: torch.Tensor, cu_seqlens: torch.Tensor) -> None:
    packed_length = int(cu_seqlens[-1].item()) if cu_seqlens.numel() > 0 else 0
    if x_bshd.shape[1] != packed_length:
        raise ValueError(
            f"Packed cu_seqlens expects flattened length {packed_length}, but got sequence length {x_bshd.shape[1]}"
        )


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
    q_norm_weight: Optional[torch.Tensor] = None,
    k_norm_weight: Optional[torch.Tensor] = None,
    norm_eps: float = 1e-5,
    dropout_p: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    batch, seqlen, dim = x_bsd.shape
    if dim % num_heads != 0:
        raise ValueError("x_bsd.shape[-1] must be divisible by num_heads")
    if num_heads % num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    head_dim = dim // num_heads
    _validate_even_head_dim(head_dim)

    kv_dim = num_kv_heads * head_dim
    if k_proj_weight.shape[0] != kv_dim or v_proj_weight.shape[0] != kv_dim:
        raise ValueError("K/V projection output dims do not match num_kv_heads * head_dim")

    q = F.linear(x_bsd, q_proj_weight, q_proj_bias).view(batch, seqlen, num_heads, head_dim)
    k = F.linear(x_bsd, k_proj_weight, k_proj_bias).view(batch, seqlen, num_kv_heads, head_dim)
    v = F.linear(x_bsd, v_proj_weight, v_proj_bias).view(batch, seqlen, num_kv_heads, head_dim)
    if q_norm_weight is not None:
        q = _rmsnorm_last_dim(q, q_norm_weight, norm_eps)
    if k_norm_weight is not None:
        k = _rmsnorm_last_dim(k, k_norm_weight, norm_eps)

    if cos is not None or sin is not None:
        if cos is None or sin is None:
            raise ValueError("Pass both cos and sin for RoPE.")
        if position_ids is None:
            raise ValueError("position_ids is required when applying RoPE.")
        cos_g, sin_g = gather_cos_sin_by_position_ids(cos, sin, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos_g, sin_g)

    # Prefer seq_idx when available because it preserves valid-token layout and padding semantics.
    if seq_idx is not None:
        indices, packed_cu_seqlens, _ = _flatten_seq_idx(seq_idx)
        if indices.numel() == 0:
            y = x_bsd.new_zeros(batch, seqlen, num_heads, head_dim)
        else:
            q_packed = _index_first_axis(q, indices).unsqueeze(0)
            k_packed = _index_first_axis(k, indices).unsqueeze(0)
            v_packed = _index_first_axis(v, indices).unsqueeze(0)
            y_packed = _run_dense_or_packed_attention(
                q_packed,
                k_packed,
                v_packed,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                window_size=window_size,
                cu_seqlens=packed_cu_seqlens,
                dropout_p=dropout_p,
                training=training,
            )
            y = x_bsd.new_zeros(batch * seqlen, num_heads, head_dim)
            y[indices] = y_packed.squeeze(0)
            y = y.view(batch, seqlen, num_heads, head_dim)
    elif cu_seqlens is not None:
        if batch != 1:
            raise ValueError("Packed cu_seqlens attention expects input shaped as a single flattened batch.")
        y = _run_dense_or_packed_attention(
            q,
            k,
            v,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            window_size=window_size,
            cu_seqlens=cu_seqlens,
            dropout_p=dropout_p,
            training=training,
        )
    else:
        y = _run_dense_or_packed_attention(
            q,
            k,
            v,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            window_size=window_size,
            cu_seqlens=None,
            dropout_p=dropout_p,
            training=training,
        )
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
    k_norm_weight: Optional[torch.Tensor] = None,
    norm_eps: float = 1e-5,
) -> None:
    past_capacity = _get_past_capacity(attn_state)
    head_dim = k_proj_weight.shape[0] // num_kv_heads
    _validate_even_head_dim(head_dim)
    _reset_state(attn_state)

    if cu_seqlens is None and seq_idx is None:
        k = F.linear(x_bsd, k_proj_weight, k_proj_bias).view(x_bsd.shape[0], x_bsd.shape[1], num_kv_heads, head_dim)
        v = F.linear(x_bsd, v_proj_weight, v_proj_bias).view(x_bsd.shape[0], x_bsd.shape[1], num_kv_heads, head_dim)
        if k_norm_weight is not None:
            k = _rmsnorm_last_dim(k, k_norm_weight, norm_eps)
        if cos is not None or sin is not None:
            if cos is None or sin is None:
                raise ValueError("Pass both cos and sin for RoPE.")
            if position_ids is None:
                raise ValueError("position_ids is required when applying RoPE.")
            cos_g, sin_g = gather_cos_sin_by_position_ids(cos, sin, position_ids)
            k, _ = apply_rotary_pos_emb(k, k, cos_g, sin_g)
        k = rearrange(k, "b s h d -> b h s d")
        v = rearrange(v, "b s h d -> b h s d")
        filled = torch.full(
            (k.shape[0],),
            min(past_capacity, k.shape[2]),
            dtype=torch.int32,
            device=attn_state["filled"].device,
        )
        _sync_tail_to_state(attn_state, k, v, filled)
        return

    if seq_idx is not None:
        indices, cu_seqlens, _ = _flatten_seq_idx(seq_idx)
        x_flat = x_bsd.reshape(-1, x_bsd.shape[-1])[indices]
        if position_ids is not None:
            flat_pos = position_ids.reshape(-1)[indices].unsqueeze(0)
        else:
            flat_pos = None
    else:
        if x_bsd.shape[0] != 1:
            raise ValueError("Packed sliding-window cache prefill expects a single flattened batch")
        _require_packed_length(x_bsd, cu_seqlens)
        x_flat = x_bsd.squeeze(0)
        flat_pos = position_ids.unsqueeze(0) if position_ids is not None and position_ids.dim() == 1 else position_ids

    if x_flat.numel() == 0:
        return

    k = rearrange(F.linear(x_flat, k_proj_weight, k_proj_bias), "s (h d) -> 1 s h d", h=num_kv_heads)
    v = rearrange(F.linear(x_flat, v_proj_weight, v_proj_bias), "s (h d) -> 1 s h d", h=num_kv_heads)
    if k_norm_weight is not None:
        k = _rmsnorm_last_dim(k, k_norm_weight, norm_eps)
    if cos is not None or sin is not None:
        if cos is None or sin is None:
            raise ValueError("Pass both cos and sin for RoPE.")
        if flat_pos is None:
            raise ValueError("position_ids is required when applying RoPE.")
        cos_g, sin_g = gather_cos_sin_by_position_ids(cos, sin, flat_pos)
        k, _ = apply_rotary_pos_emb(k, k, cos_g, sin_g)

    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    for seq in range(cu_seqlens.shape[0] - 1):
        start = int(cu_seqlens[seq].item())
        end = int(cu_seqlens[seq + 1].item())
        if end <= start:
            continue
        keep = min(past_capacity, end - start)
        if keep > 0:
            attn_state["k_cache"][seq : seq + 1, :, :keep, :].copy_(k[:, :, end - keep : end, :])
            attn_state["v_cache"][seq : seq + 1, :, :keep, :].copy_(v[:, :, end - keep : end, :])
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
    q_norm_weight: Optional[torch.Tensor] = None,
    k_norm_weight: Optional[torch.Tensor] = None,
    norm_eps: float = 1e-5,
) -> torch.Tensor:
    batch, dim = x_bd.shape
    if dim % num_heads != 0:
        raise ValueError("x_bd.shape[-1] must be divisible by num_heads")
    if num_heads % num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    head_dim = dim // num_heads
    _validate_even_head_dim(head_dim)

    q = rearrange(F.linear(x_bd, q_proj_weight, q_proj_bias), "b (h d) -> b h 1 d", h=num_heads)
    k_new = rearrange(F.linear(x_bd, k_proj_weight, k_proj_bias), "b (h d) -> b h 1 d", h=num_kv_heads)
    v_new = rearrange(F.linear(x_bd, v_proj_weight, v_proj_bias), "b (h d) -> b h 1 d", h=num_kv_heads)
    if q_norm_weight is not None:
        q = _rmsnorm_last_dim(q, q_norm_weight, norm_eps)
    if k_norm_weight is not None:
        k_new = _rmsnorm_last_dim(k_new, k_norm_weight, norm_eps)

    if cos is not None or sin is not None:
        if cos is None or sin is None:
            raise ValueError("Pass both cos and sin for RoPE.")
        if position_ids is None:
            raise ValueError("position_ids is required when applying RoPE.")
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(1)
        cos_g, sin_g = gather_cos_sin_by_position_ids(cos, sin, position_ids)
        q_bt = q.transpose(1, 2)
        k_bt = k_new.transpose(1, 2)
        q_bt, k_bt = apply_rotary_pos_emb(q_bt, k_bt, cos_g, sin_g)
        q = q_bt.transpose(1, 2)
        k_new = k_bt.transpose(1, 2)

    outputs = []
    past_capacity = _get_past_capacity(attn_state)
    for b in range(batch):
        filled_b = int(attn_state["filled"][b].item())
        prev_k = attn_state["k_cache"][b : b + 1, :, :filled_b, :]
        prev_v = attn_state["v_cache"][b : b + 1, :, :filled_b, :]
        full_k = torch.cat([prev_k, k_new[b : b + 1]], dim=-2)
        full_v = torch.cat([prev_v, v_new[b : b + 1]], dim=-2)
        y = F.scaled_dot_product_attention(
            q[b : b + 1],
            repeat_kv(full_k, num_heads // num_kv_heads),
            repeat_kv(full_v, num_heads // num_kv_heads),
            dropout_p=0.0,
            is_causal=False,
            enable_gqa=False,
        )
        outputs.append(rearrange(y, "b h s d -> b s (h d)"))

        new_size = min(past_capacity, full_k.shape[-2])
        attn_state["k_cache"][b].zero_()
        attn_state["v_cache"][b].zero_()
        if new_size > 0:
            attn_state["k_cache"][b : b + 1, :, :new_size, :].copy_(full_k[:, :, -new_size:, :])
            attn_state["v_cache"][b : b + 1, :, :new_size, :].copy_(full_v[:, :, -new_size:, :])
        attn_state["filled"][b] = new_size

    y = torch.cat(outputs, dim=0).reshape(batch, num_heads * head_dim)
    return F.linear(y, out_proj_weight, out_proj_bias)


def mamba_split_sliding_window_scan_combined(*args, **kwargs):
    raise RuntimeError(
        "mamba_split_sliding_window_scan_combined is intentionally disabled. "
        "Use Mamba2Sliding with the split `mamba_chunk_scan_combined` path instead."
    )
