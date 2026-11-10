# Adapted from KLA lit_gpt/model.py (Modified by Songlin Yang & Ali Hatamizadeh).
# Copyright (c) Microsoft Corporation. Licensed under the MIT license.
# Copyright Lightning AI. Licensed under the Apache License 2.0;
# see https://github.com/Lightning-AI/litgpt/blob/main/LICENSE.
# KLA repository license is retained at upstream/kla/LICENSE.
"""Five mixers in KLA's sequential pre-norm, tied-embedding GPT shell.

Task configs are copied from KLA; only the model-name registry and mixers change.
RMSNorm/SwiGLU use PyTorch operations for portability across the two CUDA stacks.
"""
import importlib
import math
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

from .config import Config, name_to_config

ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURES = ("KDA", "GDN", "Mamba2", "Mamba2GQA", "Mamba2GQASink")
for task in ("MQAR", "Palindrome", "Stack"):
    base = name_to_config[f"GatedDeltaNet_{task}"]
    for architecture in ARCHITECTURES:
        name = f"{architecture}_{task}"
        name_to_config[name] = dict(base, name=name)


class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.w2 = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.w3 = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class LLaMAMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.swiglu = SwiGLU(config)

    def forward(self, x):
        return self.swiglu(x)


def make_mixer(config, layer_idx):
    architecture = config.name.rsplit("_", 1)[0]
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unsupported architecture: {architecture}")
    head_dim = config.n_embd // config.n_head
    if architecture in ("KDA", "GDN"):
        sys.path.insert(0, str(ROOT / "vendor/fla"))
        module = importlib.import_module("fla.layers." + (
            "kda" if architecture == "KDA" else "gated_deltanet"))
        cls = getattr(module, "KimiDeltaAttention" if architecture == "KDA" else "GatedDeltaNet")
        return cls(hidden_size=config.n_embd, num_heads=config.n_head,
                   head_dim=head_dim, expand_v=1, conv_size=4,
                   layer_idx=layer_idx, allow_neg_eigval=False)
    sys.path.insert(0, str(ROOT / "vendor/mamba"))
    filename = {"Mamba2": "mamba2", "Mamba2GQA": "mamba2_GQA",
                "Mamba2GQASink": "mamba-gqa-sink"}[architecture]
    cls = importlib.import_module("mamba_ssm.modules." + filename).Mamba2
    kwargs = dict(d_model=config.n_embd, d_state=128, expand=1,
                  headdim=head_dim, ngroups=1, d_conv=4, chunk_size=256,
                  layer_idx=layer_idx)
    if architecture != "Mamba2":
        kwargs.update(window_size=512, gqa_num_kv_heads=1,
                      rope_theta=500000., dropout_p=0.)
    if architecture == "Mamba2GQASink":
        kwargs.update(use_attn_sink=True, sink_init=-2.)
    return cls(**kwargs)


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.norm_1 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.attn = make_mixer(config, layer_idx)
        self.norm_2 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.mlp = LLaMAMLP(config)

    def forward(self, x):
        output = self.attn(self.norm_1(x))
        if isinstance(output, tuple):
            output = output[0]
        x = x + output
        return x + self.mlp(self.norm_2(x))


# Imported (but unused) by the original MQAR trainer.
MBlock = Block


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=False)
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
            h=nn.ModuleList(Block(config, i) for i in range(config.n_layer)),
            ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
        ))
        self.mamba_init = config.mamba_init
        if self.mamba_init:
            self.lm_head.weight = self.transformer.wte.weight

    def _init_weights(self, module, n_layer):
        # KLA GPT._init_weights for the mamba_init=True common task configs.
        if not self.mamba_init:
            raise ValueError("These KLA task presets require mamba_init=True")
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.Linear):
            if module.bias is not None and not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
        for name, p in module.named_parameters():
            if name in ("out_proj.weight", "fc2.weight") or (
                name == "w3.weight" and isinstance(module, SwiGLU)
            ):
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(2 * n_layer)

    def forward(self, idx):
        x = self.transformer.wte(idx)
        for block in self.transformer.h:
            x = block(x)
        return self.lm_head(self.transformer.ln_f(x))
