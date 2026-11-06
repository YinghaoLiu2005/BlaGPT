"""Adapters so the paper mixers can be instantiated by Zoology's TransformerBlock."""
from __future__ import annotations

import sys
import importlib.util
from pathlib import Path

import torch
from torch import nn


def _add_fla():
    root = Path(__file__).resolve().parents[3] / "3rdparty" / "flash-linear-attention"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


class KDA(nn.Module):
    def __init__(self, d_model: int, layer_idx: int = None, **kwargs):
        super().__init__()
        _add_fla()
        from fla.layers import KimiDeltaAttention
        self.layer = KimiDeltaAttention(
            hidden_size=d_model, expand_v=1.0, head_dim=128,
            num_heads=2, num_v_heads=2, mode="chunk",
            use_short_conv=True, conv_size=4, layer_idx=layer_idx,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.layer(x)
        return y[0] if isinstance(y, tuple) else y


class GDN(nn.Module):
    def __init__(self, d_model: int, layer_idx: int = None, **kwargs):
        super().__init__()
        _add_fla()
        from fla.layers import GatedDeltaNet
        self.layer = GatedDeltaNet(
            hidden_size=d_model, expand_v=1.0, head_dim=128,
            num_heads=2, mode="chunk", use_gate=True,
            use_short_conv=True, conv_size=4, layer_idx=layer_idx,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.layer(x)
        return y[0] if isinstance(y, tuple) else y


class Mamba2(nn.Module):
    def __init__(self, d_model: int, layer_idx: int = None, **kwargs):
        super().__init__()
        # Use the blagpt environment's official state-spaces implementation.
        from mamba_ssm.modules.mamba2 import Mamba2 as _Mamba2
        self.layer = _Mamba2(
            d_model=d_model, d_state=128, d_conv=4, expand=1,
            headdim=128, ngroups=1, rmsnorm=True, chunk_size=256,
            use_mem_eff_path=True, layer_idx=layer_idx,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class Mamba2GQA(nn.Module):
    """The user's local GQA-SWA Mamba2 implementation, isolated from Zoology."""
    def __init__(self, d_model: int, layer_idx: int = None, **kwargs):
        super().__init__()
        source = Path(__file__).resolve().parents[3] / "third_party" / "mamba_ssm" / "modules" / "mamba2_GQA.py"
        spec = importlib.util.spec_from_file_location("mamba_ssm.modules.mamba2_GQA_local", source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _Mamba2GQA = module.Mamba2
        self.layer = _Mamba2GQA(
            d_model=d_model, d_state=128, d_conv=4, window_size=128,
            use_flash_swa=True, gqa_num_kv_heads=1, expand=1,
            headdim=128, ngroups=1, rmsnorm=True, chunk_size=256,
            use_mem_eff_path=False, layer_idx=layer_idx,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)
