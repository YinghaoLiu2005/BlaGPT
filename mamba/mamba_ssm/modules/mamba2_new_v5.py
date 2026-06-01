# Copyright (c) 2024, Tri Dao, Albert Gu.
# Based on v2, apply input sliding-window attention as a pre-norm residual block.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None

from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated

from mamba_ssm.distributed.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from mamba_ssm.distributed.distributed_utils import all_reduce, reduce_scatter

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

from huggingface_hub import PyTorchModelHubMixin


class Mamba2(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        d_model,
        d_state=128,
        d_conv=4,
        window_size=None,
        use_flash_swa=True,
        rope_theta=500000.0,
        dropout_p=0.0,
        conv_init=None,
        expand=2,
        headdim=64,
        d_ssm=None,  # If not None, we only apply SSM on this many dimensions, the rest uses gated MLP
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
        # Fused kernel and sharding options
        chunk_size=256,
        use_mem_eff_path=True,
        layer_idx=None,  # Absorb kwarg for general module
        process_group=None,
        sequence_parallel=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.window_size = d_conv if window_size is None else window_size
        self.use_flash_swa = use_flash_swa
        self.rope_theta = rope_theta
        self.dropout_p = dropout_p
        self.conv_init = conv_init
        self.expand = expand
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel
        self.world_size = 1 if process_group is None else process_group.size()
        self.local_rank = 0 if process_group is None else process_group.rank()
        self.d_inner = (self.expand * self.d_model) // self.world_size
        assert self.d_inner * self.world_size == self.expand * self.d_model
        self.headdim = headdim
        self.d_ssm = self.d_inner if d_ssm is None else d_ssm // self.world_size
        assert ngroups % self.world_size == 0
        self.ngroups = ngroups // self.world_size
        assert self.d_ssm % self.headdim == 0
        self.nheads = self.d_ssm // self.headdim
        self.D_has_hdim = D_has_hdim
        self.rmsnorm = rmsnorm
        self.norm_before_gate = norm_before_gate
        self.dt_limit = dt_limit
        self.activation = "silu"
        self.chunk_size = chunk_size
        self.use_mem_eff_path = False
        self.layer_idx = layer_idx

        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        if self.process_group is None:
            self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)
        else:
            self.in_proj = ColumnParallelLinear(self.d_model, d_in_proj * self.world_size, bias=bias,
                                                process_group=self.process_group, sequence_parallel=self.sequence_parallel,
                                                **factory_kwargs)

        assert self.d_model % self.headdim == 0, "Flash SWA input dim must be divisible by headdim"
        assert self.headdim % 2 == 0, "RoPE requires an even attention head dimension"
        self.attn_nheads = self.d_model // self.headdim
        self.input_q_proj = nn.Linear(self.d_model, self.d_model, bias=conv_bias, **factory_kwargs)
        self.input_k_proj = nn.Linear(self.d_model, self.d_model, bias=conv_bias, **factory_kwargs)
        self.input_v_proj = nn.Linear(self.d_model, self.d_model, bias=conv_bias, **factory_kwargs)
        self.input_out_proj = nn.Linear(self.d_model, self.d_model, bias=conv_bias, **factory_kwargs)
        self.input_norm = RMSNormGated(self.d_model, eps=1e-5, norm_before_gate=True, **factory_kwargs)
        rotary_inv_freq = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, self.headdim, 2, device=device, dtype=torch.float32) / self.headdim)
        )
        self.register_buffer("rotary_inv_freq", rotary_inv_freq, persistent=False)

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_ssm if self.D_has_hdim else self.nheads, device=device))
        self.D._no_weight_decay = True

        if self.rmsnorm:
            assert RMSNormGated is not None
            self.norm = RMSNormGated(self.d_ssm, eps=1e-5, norm_before_gate=self.norm_before_gate,
                                     group_size=self.d_ssm // ngroups, **factory_kwargs)

        if self.process_group is None:
            self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        else:
            self.out_proj = RowParallelLinear(self.d_inner * self.world_size, self.d_model, bias=bias,
                                              process_group=self.process_group, sequence_parallel=self.sequence_parallel,
                                              **factory_kwargs)

    def _rotary_cos_sin(self, seqlen, device, dtype):
        position_ids = torch.arange(seqlen, device=device, dtype=self.rotary_inv_freq.dtype)
        freqs = torch.outer(position_ids, self.rotary_inv_freq)
        return (
            freqs.cos()[None, :, None, :].to(dtype=dtype),
            freqs.sin()[None, :, None, :].to(dtype=dtype),
        )

    def _apply_rotary(self, x, cos, sin):
        with torch.autocast(device_type=x.device.type, enabled=False):
            half = x.shape[-1] // 2
            x1 = x[..., :half].float()
            x2 = x[..., half:].float()
            y1 = x1 * cos + x2 * sin
            y2 = x1 * (-sin) + x2 * cos
        return torch.cat([y1, y2], dim=-1).type_as(x)

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

        if inference_params is not None:
            raise NotImplementedError("Mamba2 v5 input sliding-window attention only supports full-sequence training for now.")

        if seqlen_og is not None:
            u = rearrange(u, "(b l) d -> b l d", l=seqlen)
        if self.use_flash_swa:
            if flash_attn_func is None:
                raise RuntimeError("use_flash_swa=True requires flash-attn to be installed.")
            if u.device.type != "cuda":
                raise RuntimeError("flash-attn sliding-window attention requires CUDA tensors.")
            attn_input = self.input_norm(u)
            q = rearrange(self.input_q_proj(attn_input), "b l (h p) -> b l h p", h=self.attn_nheads)
            k = rearrange(self.input_k_proj(attn_input), "b l (h p) -> b l h p", h=self.attn_nheads)
            v = rearrange(self.input_v_proj(attn_input), "b l (h p) -> b l h p", h=self.attn_nheads)
            if q.dtype not in (torch.float16, torch.bfloat16):
                q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)
            cos, sin = self._rotary_cos_sin(seqlen, u.device, q.dtype)
            q = self._apply_rotary(q, cos, sin)
            k = self._apply_rotary(k, cos, sin)
            attn_out = flash_attn_func(
                q, k, v,
                dropout_p=self.dropout_p if self.training else 0.0,
                causal=True,
                window_size=(self.window_size - 1, 0),
            )
            attn_out = rearrange(attn_out, "b l h p -> b l (h p)").to(dtype=u.dtype)
            u = u + self.input_out_proj(attn_out)

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj)
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        A = -torch.exp(self.A_log.float())  # (nheads) or (d_inner, d_state)
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)
        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1
        )
        xBC = self.act(xBC)
        x, B, C = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        y = mamba_chunk_scan_combined(
            rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
            dt,
            A,
            rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
            rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
            chunk_size=self.chunk_size,
            D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
            z=rearrange(z, "b l (h p) -> b l h p", p=self.headdim) if not self.rmsnorm else None,
            dt_bias=self.dt_bias,
            dt_softplus=True,
            seq_idx=seq_idx,
            cu_seqlens=cu_seqlens,
            **dt_limit_kwargs,
        )
        y = rearrange(y, "b l h p -> b l (h p)")
        if self.rmsnorm:
            y = self.norm(y, z)
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        if seqlen_og is not None:
            y = rearrange(y, "b l d -> (b l) d")
        out = self.out_proj(y)
        if self.process_group is not None:
            reduce_fn = reduce_scatter if self.sequence_parallel else all_reduce
            out = reduce_fn(out, self.process_group)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        raise NotImplementedError("Mamba2 v5 input sliding-window attention does not support step decoding yet.")

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        raise NotImplementedError("Mamba2 v5 input sliding-window attention does not support inference cache yet.")

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        raise NotImplementedError("Mamba2 v5 input sliding-window attention does not support inference cache yet.")
