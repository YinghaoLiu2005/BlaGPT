# Copyright (c) 2024, Tri Dao, Albert Gu.

import inspect

import torch
import torch.nn as nn

from einops import rearrange
from mamba_ssm.distributed.distributed_utils import all_reduce, reduce_scatter
from mamba_ssm.ops.triton.sliding_window_attention import (
    prefill_sliding_window_cache,
    sliding_window_attention_step,
    sliding_window_attention,
)

from huggingface_hub import PyTorchModelHubMixin


_SWA_SIGNATURES = {
    "sliding_window_attention": inspect.signature(sliding_window_attention).parameters,
    "prefill_sliding_window_cache": inspect.signature(prefill_sliding_window_cache).parameters,
    "sliding_window_attention_step": inspect.signature(sliding_window_attention_step).parameters,
}


def _filter_supported_kwargs(fn_name: str, kwargs: dict) -> dict:
    supported = _SWA_SIGNATURES[fn_name]
    return {key: value for key, value in kwargs.items() if key in supported}


def _maybe_cast(tensor, dtype):
    return None if tensor is None else tensor.to(dtype=dtype)


class Mamba2SlidingTest(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        d_model,
        d_state=128,
        d_conv=4,
        conv_init=None,
        expand=2,
        headdim=64,
        d_ssm=None,
        ngroups=1,
        A_init_range=(1, 16),
        D_has_hdim=False,
        rmsnorm=True,
        norm_before_gate=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        bias=False,
        conv_bias=True,
        chunk_size=256,
        use_mem_eff_path=True,
        layer_idx=None,
        process_group=None,
        sequence_parallel=True,
        device=None,
        dtype=None,
        window_size=512,
        num_kv_heads=None,
        num_heads=None,
        dropout_p=0.0,
        rmsnorm_before_qk=False,
        pos_encoding=None,
        rope_theta=10000,
        block_size=1024,
        rope_variant="standard",
        use_qkv_bias=False,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel
        self.world_size = 1 if process_group is None else process_group.size()
        self.layer_idx = layer_idx
        self.window_size = self.d_conv if window_size is None else window_size

        if num_heads is None:
            if headdim is None:
                raise ValueError("Pass either num_heads or headdim for transformer-style SWA.")
            if d_model % headdim != 0:
                raise ValueError(f"d_model={d_model} must be divisible by headdim={headdim}.")
            num_heads = d_model // headdim
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}.")
        self.attn_num_heads = num_heads
        self.attn_head_dim = d_model // num_heads
        self.num_kv_heads = self.attn_num_heads if num_kv_heads is None else num_kv_heads
        if self.attn_num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads={self.attn_num_heads} must be divisible by num_kv_heads={self.num_kv_heads}."
            )
        if self.attn_head_dim % 2 != 0:
            raise ValueError("SWA RoPE/cache helpers require an even attention head dimension.")

        self.dropout_p = dropout_p
        self.pos_encoding = pos_encoding
        self.rope_variant = rope_variant
        self.rmsnorm_before_qk = rmsnorm_before_qk
        self.norm_eps = 1e-5

        qkv_bias = bias or use_qkv_bias
        kv_dim = self.num_kv_heads * self.attn_head_dim
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=qkv_bias, **factory_kwargs)
        self.k_proj = nn.Linear(self.d_model, kv_dim, bias=qkv_bias, **factory_kwargs)
        self.v_proj = nn.Linear(self.d_model, kv_dim, bias=qkv_bias, **factory_kwargs)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=bias, **factory_kwargs)
        if self.conv_init is not None:
            nn.init.uniform_(self.q_proj.weight, -self.conv_init, self.conv_init)
            nn.init.uniform_(self.k_proj.weight, -self.conv_init, self.conv_init)
            nn.init.uniform_(self.v_proj.weight, -self.conv_init, self.conv_init)
            nn.init.uniform_(self.out_proj.weight, -self.conv_init, self.conv_init)

        if self.rmsnorm_before_qk:
            self.q_norm_weight = nn.Parameter(torch.ones(self.attn_head_dim, **factory_kwargs))
            self.k_norm_weight = nn.Parameter(torch.ones(self.attn_head_dim, **factory_kwargs))
        else:
            self.q_norm_weight = None
            self.k_norm_weight = None

        if self.pos_encoding == "rotary":
            if self.rope_variant != "standard":
                raise ValueError("Mamba2SlidingTest SWA baseline only supports standard RoPE.")
            inv_freq = 1.0 / (
                rope_theta
                ** (torch.arange(0, self.attn_head_dim, 2, device=device, dtype=torch.float32) / self.attn_head_dim)
            )
            self.register_buffer("rotary_inv_freq", inv_freq, persistent=False)
            self.rotary_block_size = block_size
        elif self.pos_encoding in ("none", None):
            self.rotary_inv_freq = None
        else:
            raise ValueError(f"Mamba2SlidingTest SWA baseline does not support pos_encoding={self.pos_encoding}.")

    def _position_ids(self, batch, seqlen, device, cu_seqlens=None, seq_idx=None, offset=0):
        if self.pos_encoding != "rotary":
            return None
        if cu_seqlens is not None:
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            return torch.cat([torch.arange(int(length.item()), device=device) for length in lengths]).to(torch.long)
        if seq_idx is not None:
            return torch.arange(seqlen, device=device).unsqueeze(0).expand(batch, seqlen)
        return torch.arange(offset, offset + seqlen, device=device).unsqueeze(0).expand(batch, seqlen)

    def _attention_dtype(self, x):
        if (
            x.device.type == "cuda"
            and x.dtype not in (torch.float16, torch.bfloat16)
        ):
            return torch.bfloat16
        return x.dtype

    def _cos_sin(self, position_ids, dtype):
        if position_ids is None:
            return None, None
        max_pos = int(position_ids.max().item()) + 1 if position_ids.numel() > 0 else 0
        t = torch.arange(max_pos, device=position_ids.device, dtype=self.rotary_inv_freq.dtype)
        freqs = torch.outer(t, self.rotary_inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)

    def _swa_common_kwargs(self, position_ids, dtype):
        cos, sin = self._cos_sin(position_ids, dtype)
        return dict(
            position_ids=position_ids,
            cos=cos,
            sin=sin,
            q_norm_weight=_maybe_cast(self.q_norm_weight, dtype),
            k_norm_weight=_maybe_cast(self.k_norm_weight, dtype),
            norm_eps=self.norm_eps,
        )

    def _linear_kwargs(self, dtype):
        return dict(
            q_proj_weight=self.q_proj.weight.to(dtype=dtype),
            k_proj_weight=self.k_proj.weight.to(dtype=dtype),
            v_proj_weight=self.v_proj.weight.to(dtype=dtype),
            out_proj_weight=self.out_proj.weight.to(dtype=dtype),
            q_proj_bias=_maybe_cast(self.q_proj.bias, dtype),
            k_proj_bias=_maybe_cast(self.k_proj.bias, dtype),
            v_proj_bias=_maybe_cast(self.v_proj.bias, dtype),
            out_proj_bias=_maybe_cast(self.out_proj.bias, dtype),
        )

    def forward(self, u, seqlen=None, seq_idx=None, cu_seqlens=None, inference_params=None):
        """
        u: (batch, seqlen, hidden_dim) if seqlen=None.
            If seqlen is not None, u is (batch * seqlen, hidden_dim). This is so that when we
            split u during sequence parallel, we split the batch * seqlen dimension
            (in case batch is small).
        Returns: same shape as u
        """
        seqlen_og = seqlen
        if seqlen is None:
            batch, seqlen, dim = u.shape
        else:
            batch_seqlen, dim = u.shape
            batch = batch_seqlen // seqlen

        conv_state, ssm_state = None, None
        if inference_params is not None:
            inference_batch = cu_seqlens.shape[0] - 1 if cu_seqlens is not None else batch
            conv_state, _ = self._get_states_from_cache(inference_params, inference_batch)
            if inference_params.seqlen_offset > 0:
                out, _, _ = self.step(
                    u,
                    conv_state,
                    None,
                    position_offset=inference_params.seqlen_offset,
                )
                return out

        if seqlen_og is not None:
            u = rearrange(u, "(b l) d -> b l d", l=seqlen)

        orig_dtype = u.dtype
        attn_dtype = self._attention_dtype(u)
        u_attn = u.to(dtype=attn_dtype)
        linear_kwargs = self._linear_kwargs(attn_dtype)
        position_ids = self._position_ids(batch, seqlen, u.device, cu_seqlens=cu_seqlens, seq_idx=seq_idx)
        common_kwargs = self._swa_common_kwargs(position_ids, attn_dtype)
        if conv_state is not None:
            prefill_sliding_window_cache(
                u_attn,
                conv_state,
                k_proj_weight=linear_kwargs["k_proj_weight"],
                v_proj_weight=linear_kwargs["v_proj_weight"],
                k_proj_bias=linear_kwargs["k_proj_bias"],
                v_proj_bias=linear_kwargs["v_proj_bias"],
                num_kv_heads=self.num_kv_heads,
                cu_seqlens=cu_seqlens,
                seq_idx=seq_idx,
                **_filter_supported_kwargs("prefill_sliding_window_cache", common_kwargs),
            )

        y = sliding_window_attention(
            u_attn,
            **linear_kwargs,
            num_heads=self.attn_num_heads,
            num_kv_heads=self.num_kv_heads,
            window_size=self.window_size,
            seq_idx=seq_idx,
            cu_seqlens=cu_seqlens,
            dropout_p=self.dropout_p,
            training=self.training,
            **_filter_supported_kwargs("sliding_window_attention", common_kwargs),
        )
        y = y.to(dtype=orig_dtype)
        if seqlen_og is not None:
            y = rearrange(y, "b l d -> (b l) d")
        if self.process_group is not None:
            reduce_fn = reduce_scatter if self.sequence_parallel else all_reduce
            y = reduce_fn(y, self.process_group)
        return y

    def step(self, hidden_states, conv_state, ssm_state=None, position_offset=0):
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        x = hidden_states.squeeze(1)
        orig_dtype = x.dtype
        attn_dtype = self._attention_dtype(x)
        x_attn = x.to(dtype=attn_dtype)
        linear_kwargs = self._linear_kwargs(attn_dtype)
        position_ids = self._position_ids(
            x.shape[0],
            1,
            x.device,
            offset=position_offset,
        )
        common_kwargs = self._swa_common_kwargs(position_ids, attn_dtype)
        y = sliding_window_attention_step(
            x_attn,
            conv_state,
            **linear_kwargs,
            num_heads=self.attn_num_heads,
            num_kv_heads=self.num_kv_heads,
            **_filter_supported_kwargs("sliding_window_attention_step", common_kwargs),
        )
        return y.to(dtype=orig_dtype).unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.q_proj.weight.dtype if dtype is None else dtype
        if (
            device.type == "cuda"
            and conv_dtype not in (torch.float16, torch.bfloat16)
        ):
            conv_dtype = torch.bfloat16
        conv_state = {
            "k_cache": torch.zeros(
                batch_size,
                self.num_kv_heads,
                self.window_size,
                self.attn_head_dim,
                device=device,
                dtype=conv_dtype,
            ),
            "v_cache": torch.zeros(
                batch_size,
                self.num_kv_heads,
                self.window_size,
                self.attn_head_dim,
                device=device,
                dtype=conv_dtype,
            ),
            "filled": torch.zeros(batch_size, device=device, dtype=torch.int32),
        }
        return conv_state, None

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            inference_params.key_value_memory_dict[self.layer_idx] = self.allocate_inference_cache(batch_size, 0)
        else:
            conv_state, _ = inference_params.key_value_memory_dict[self.layer_idx]
            if initialize_states:
                conv_state["k_cache"].zero_()
                conv_state["v_cache"].zero_()
                conv_state["filled"].zero_()
        return inference_params.key_value_memory_dict[self.layer_idx]
