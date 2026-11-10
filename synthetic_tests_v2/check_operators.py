"""Check portable operations and Sink gradients against independent references."""
import ast
import copy
from functools import partial
import importlib
import importlib.util
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from lit_gpt.model import Config, GPT, LLaMAMLP, SwiGLU
from lit_gpt.rmsnorm import RMSNorm

ROOT = Path(__file__).resolve().parent


def check():
    torch.set_num_threads(1)
    torch.manual_seed(123)
    # Use the actual upstream norm implementation as the CPU oracle.
    spec = importlib.util.spec_from_file_location("kla_norm", ROOT / "upstream/kla/lit_gpt/rmsnorm.py")
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    upstream.dropout_layer_norm = None  # Exercise KLA's own portable CPU branch.
    x = torch.randn(2, 16, 256, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    norm, norm_ref = RMSNorm(256), upstream.FusedRMSNorm(256)
    y, y_ref = norm(x), norm_ref(x_ref)
    torch.testing.assert_close(y, y_ref, atol=1e-6, rtol=1e-6)
    grad = torch.randn_like(y)
    y.backward(grad)
    y_ref.backward(grad)
    torch.testing.assert_close(x.grad, x_ref.grad, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(norm.weight.grad, norm_ref.weight.grad, atol=1e-5, rtol=1e-5)

    # Execute the unchanged upstream initializer on an identical model copy.
    tree = ast.parse((ROOT / "upstream/kla/lit_gpt/model.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GPT")
    init = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_init_weights")
    namespace = dict(nn=nn, torch=torch, math=math, LLaMAMLP=LLaMAMLP,
                     SwiGLU=SwiGLU, CausalSelfAttention=type("UnusedAttention", (), {}))
    exec(compile(ast.Module(body=[init], type_ignores=[]), "upstream_initializer", "exec"), namespace)
    model = GPT(Config.from_name("Mamba2GQASink_MQAR"))
    reference = copy.deepcopy(model)
    torch.manual_seed(42)
    model.apply(partial(model._init_weights, n_layer=2))
    torch.manual_seed(42)
    reference.apply(lambda module: namespace["_init_weights"](reference, module, 2))
    for name, parameter in model.state_dict().items():
        torch.testing.assert_close(parameter, reference.state_dict()[name], rtol=0, atol=0)

    assert torch.cuda.is_available(), "GPU required for fused-loss and Sink reference checks"
    from flash_attn.losses.cross_entropy import CrossEntropyLoss
    logits = torch.randn(35, 128, device="cuda", requires_grad=True)
    logits_ref = logits.detach().clone().requires_grad_(True)
    labels = torch.randint(0, 128, (35,), device="cuda")
    labels[::3] = -100
    loss = F.cross_entropy(logits, labels)
    loss_ref = CrossEntropyLoss()(logits_ref, labels)
    loss.backward()
    loss_ref.backward()
    torch.testing.assert_close(loss, loss_ref, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(logits.grad, logits_ref.grad, atol=1e-6, rtol=1e-5)

    sink_module = importlib.import_module("mamba_ssm.modules.mamba-gqa-sink")
    errors = []
    for heads, dim in ((4, 64), (2, 128)):
        # Independent dense softmax over a local causal window plus zero-value sink.
        q = torch.randn(2, 79, heads, dim, device="cuda", dtype=torch.float16, requires_grad=True)
        k = torch.randn(2, 79, 1, dim, device="cuda", dtype=torch.float16, requires_grad=True)
        v = torch.randn_like(k, requires_grad=True)
        s = torch.full((heads,), -2., device="cuda", requires_grad=True)
        refs = [p.detach().float().clone().requires_grad_(True) for p in (q, k, v, s)]
        qr, kr, vr, sr = refs
        scores = torch.einsum("blhd,bshd->bhls", qr, kr.expand(-1, -1, heads, -1)) / math.sqrt(dim)
        pos = torch.arange(79, device="cuda")
        allowed = (pos[:, None] >= pos[None, :]) & (pos[:, None] - pos[None, :] < 33)
        scores = scores.masked_fill(~allowed, -torch.inf)
        scores = torch.cat((scores, sr[None, :, None, None].expand(2, -1, 79, 1)), dim=-1)
        probabilities = scores.softmax(-1)[..., :-1]
        expected = torch.einsum("bhls,bshd->blhd", probabilities, vr.expand(-1, -1, heads, -1))
        actual = sink_module.flash_attn_sink_func(q, k, v, s, 33)
        gradient = torch.randn_like(actual)
        actual.backward(gradient)
        expected.backward(gradient.float())
        torch.testing.assert_close(actual.float(), expected, atol=2e-3, rtol=1e-2)
        for native, ref in zip((q, k, v, s), refs):
            torch.testing.assert_close(native.grad.float(), ref.grad, atol=1e-2, rtol=2e-2)
        errors.append(dict(heads=heads, head_dim=dim,
                           max_output_error=float((actual.float() - expected).abs().max()),
                           max_sink_gradient_error=float((s.grad - sr.grad).abs().max())))
    return dict(rmsnorm_forward_backward=True, kla_initialization_exact=True,
                cross_entropy_forward_backward=True, sink_dense_reference=errors)


if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
