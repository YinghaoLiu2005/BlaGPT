# Copyright (c) 2024, Tri Dao, Albert Gu.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
from mamba_ssm.distributed.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from mamba_ssm.distributed.distributed_utils import all_reduce, reduce_scatter
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from mamba_ssm.ops.triton.ssd_combined_sliding import (
    prefill_sliding_window_cache,
    sliding_window_attention,
    sliding_window_attention_step,
)

from huggingface_hub import PyTorchModelHubMixin


def _resolve_num_heads(total_dim: int, preferred_heads: int) -> int:
    preferred_heads = min(total_dim, max(1, preferred_heads))
    for heads in range(preferred_heads, 0, -1):
        if total_dim % heads == 0:
            return heads
    return 1


class Mamba2Sliding(nn.Module, PyTorchModelHubMixin):
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
        use_mem_eff_path=False,
        layer_idx=None,
        process_group=None,
        sequence_parallel=True,
        device=None,
        dtype=None,
        window_size=512,
        num_kv_heads=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
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
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx
        # Use reasonable window size, not d_conv (which is typically only 4)
        self.window_size = window_size if window_size is not None else 512
        if self.window_size < self.d_conv:
            import warnings
            warnings.warn(f"window_size ({self.window_size}) is smaller than d_conv ({self.d_conv}). "
                         f"This may limit the model's ability to capture local context.")

        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        if self.process_group is None:
            self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)
        else:
            self.in_proj = ColumnParallelLinear(
                self.d_model,
                d_in_proj * self.world_size,
                bias=bias,
                process_group=self.process_group,
                sequence_parallel=self.sequence_parallel,
                **factory_kwargs,
            )

        self.conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state
        preferred_q_heads = self.conv_dim // self.headdim if self.conv_dim % self.headdim == 0 else self.nheads
        self.attn_num_heads = _resolve_num_heads(self.conv_dim, preferred_q_heads)
        preferred_kv_heads = self.ngroups if num_kv_heads is None else num_kv_heads
        preferred_kv_heads = min(self.attn_num_heads, preferred_kv_heads)
        self.num_kv_heads = _resolve_num_heads(self.conv_dim, preferred_kv_heads)
        while self.attn_num_heads % self.num_kv_heads != 0:
            self.num_kv_heads = _resolve_num_heads(self.conv_dim, self.num_kv_heads - 1)
        self.attn_head_dim = self.conv_dim // self.attn_num_heads

        self.q_proj = nn.Linear(self.conv_dim, self.conv_dim, bias=bias, **factory_kwargs)
        kv_dim = self.num_kv_heads * self.attn_head_dim
        self.k_proj = nn.Linear(self.conv_dim, kv_dim, bias=bias, **factory_kwargs)
        self.v_proj = nn.Linear(self.conv_dim, kv_dim, bias=bias, **factory_kwargs)
        self.xbc_out_proj = nn.Linear(self.conv_dim, self.conv_dim, bias=bias, **factory_kwargs)
        
        # Initialize projection weights with smaller scale to prevent initial NaN
        with torch.no_grad():
            nn.init.xavier_uniform_(self.q_proj.weight, gain=0.5)
            nn.init.xavier_uniform_(self.k_proj.weight, gain=0.5)
            nn.init.xavier_uniform_(self.v_proj.weight, gain=0.5)
            nn.init.xavier_uniform_(self.xbc_out_proj.weight, gain=0.5)
            if self.q_proj.bias is not None:
                nn.init.zeros_(self.q_proj.bias)
            if self.k_proj.bias is not None:
                nn.init.zeros_(self.k_proj.bias)
            if self.v_proj.bias is not None:
                nn.init.zeros_(self.v_proj.bias)
            if self.xbc_out_proj.bias is not None:
                nn.init.zeros_(self.xbc_out_proj.bias)
        self.act = nn.SiLU()

        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.d_ssm if self.D_has_hdim else self.nheads, device=device))
        self.D._no_weight_decay = True

        if self.rmsnorm:
            self.norm = RMSNormGated(
                self.d_ssm,
                eps=1e-5,
                norm_before_gate=self.norm_before_gate,
                group_size=self.d_ssm // ngroups,
                **factory_kwargs,
            )

        if self.process_group is None:
            self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        else:
            self.out_proj = RowParallelLinear(
                self.d_inner * self.world_size,
                self.d_model,
                bias=bias,
                process_group=self.process_group,
                sequence_parallel=self.sequence_parallel,
                **factory_kwargs,
            )

    def _mix_xbc(self, xBC, seq_idx=None):
        """Apply sliding window attention to xBC.
        
        Memory-efficient implementation that:
        1. Uses Flash Attention 2 for O(N) memory complexity
        2. Frees intermediate tensors aggressively
        3. Avoids NaN checks in forward pass for speed (rely on autograd)
        """
        xBC = sliding_window_attention(
            xBC,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            out_proj_weight=self.xbc_out_proj.weight,
            q_proj_bias=self.q_proj.bias,
            k_proj_bias=self.k_proj.bias,
            v_proj_bias=self.v_proj.bias,
            out_proj_bias=self.xbc_out_proj.bias,
            num_heads=self.attn_num_heads,
            num_kv_heads=self.num_kv_heads,
            window_size=self.window_size,
            seq_idx=seq_idx,
            training=self.training,
        )
        return self.act(xBC)

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
            batch, seqlen, _ = u.shape
        else:
            batch_seqlen, _ = u.shape
            batch = batch_seqlen // seqlen

        conv_state, ssm_state = None, None
        if inference_params is not None:
            inference_batch = cu_seqlens.shape[0] - 1 if cu_seqlens is not None else batch
            conv_state, ssm_state = self._get_states_from_cache(inference_params, inference_batch)
            if inference_params.seqlen_offset > 0:
                out, _, _ = self.step(u, conv_state, ssm_state)
                return out

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj) or (B * L, d_in_proj)
        if seqlen_og is not None:
            zxbcdt = rearrange(zxbcdt, "(b l) d -> b l d", l=seqlen)
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        A = -torch.exp(self.A_log.float())  # (nheads) or (d_inner, d_state)
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)

        if self.use_mem_eff_path and inference_params is None:
            raise NotImplementedError("Mamba2Sliding does not support the fused mem-efficient path.")
        else:
            if cu_seqlens is not None:
                raise NotImplementedError("Mamba2Sliding does not support cu_seqlens/varlen packing.")
            d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
            z0, x0, z, xBC, dt = torch.split(
                zxbcdt,
                [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
                dim=-1
            )
            if conv_state is not None:
                prefill_sliding_window_cache(
                    xBC,
                    conv_state,
                    k_proj_weight=self.k_proj.weight,
                    v_proj_weight=self.v_proj.weight,
                    k_proj_bias=self.k_proj.bias,
                    v_proj_bias=self.v_proj.bias,
                    num_kv_heads=self.num_kv_heads,
                )
            xBC = self._mix_xbc(xBC, seq_idx=seq_idx)
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
                **dt_limit_kwargs,
                return_final_states=ssm_state is not None,
                return_varlen_states=False,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
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
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        zxbcdt = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1
        )

        # Sliding-attention step
        xBC = sliding_window_attention_step(
            xBC,
            conv_state,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            out_proj_weight=self.xbc_out_proj.weight,
            q_proj_bias=self.q_proj.bias,
            k_proj_bias=self.k_proj.bias,
            v_proj_bias=self.v_proj.bias,
            out_proj_bias=self.xbc_out_proj.bias,
            num_heads=self.attn_num_heads,
            num_kv_heads=self.num_kv_heads,
        )
        xBC = self.act(xBC).to(dtype=dtype)

        x, B, C = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        A = -torch.exp(self.A_log.float())  # (nheads,)

        # SSM step
        if selective_state_update is None:
            assert self.ngroups == 1, "Only support ngroups=1 for this inference code path"
            dt = F.softplus(dt + self.dt_bias.to(dtype=dt.dtype))
            dA = torch.exp(dt * A)
            x = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            dBx = torch.einsum("bh,bn,bhp->bhpn", dt, B, x)
            ssm_state.copy_(ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)
            y = torch.einsum("bhpn,bn->bhp", ssm_state.to(dtype), C)
            y = y + rearrange(self.D.to(dtype), "h -> h 1") * x
            y = rearrange(y, "b h p -> b (h p)")
            if not self.rmsnorm:
                y = y * self.act(z)
        else:
            A = repeat(A, "h -> h p n", p=self.headdim, n=self.d_state).to(dtype=torch.float32)
            dt = repeat(dt, "b h -> b h p", p=self.headdim)
            dt_bias = repeat(self.dt_bias, "h -> h p", p=self.headdim)
            D = repeat(self.D, "h -> h p", p=self.headdim)
            B = rearrange(B, "b (g n) -> b g n", g=self.ngroups)
            C = rearrange(C, "b (g n) -> b g n", g=self.ngroups)
            x_reshaped = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            if not self.rmsnorm:
                z = rearrange(z, "b (h p) -> b h p", p=self.headdim)
            y = selective_state_update(
                ssm_state,
                x_reshaped,
                dt,
                A,
                B,
                C,
                D,
                z=z if not self.rmsnorm else None,
                dt_bias=dt_bias,
                dt_softplus=True,
            )
            y = rearrange(y, "b h p -> b (h p)")

        if self.rmsnorm:
            y = self.norm(y, z)
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.q_proj.weight.dtype if dtype is None else dtype
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
            "filled": torch.zeros((), device=device, dtype=torch.int32),
        }
        ssm_dtype = self.in_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size, self.nheads, self.headdim, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            inference_params.key_value_memory_dict[self.layer_idx] = self.allocate_inference_cache(batch_size, 0)
        conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
        if initialize_states:
            conv_state["k_cache"].zero_()
            conv_state["v_cache"].zero_()
            conv_state["filled"].zero_()
            ssm_state.zero_()
        return conv_state, ssm_state
