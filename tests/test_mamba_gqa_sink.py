"""Sink equivalence tests; run directly with the CUDA/Mamba Python environment."""

import importlib.util
from pathlib import Path
import unittest
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_module(relative_path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULES = [
    load_module("mamba/mamba_ssm/modules/mamba-gqa-sink.py", "main_gqa"),
    load_module("zoology/third_party/mamba_ssm/modules/mamba-gqa-sink.py", "zoology_gqa"),
]


def dense_attention(q, k, v, window, sinks=None):
    """Independent reference: append a sink logit and a zero value explicitly."""
    group = q.shape[2] // k.shape[2]
    k = k.repeat_interleave(group, dim=2)
    v = v.repeat_interleave(group, dim=2)
    logits = torch.einsum("blhd,bshd->bhls", q, k) * q.shape[-1] ** -0.5
    pos = torch.arange(q.shape[1], device=q.device)
    distance = pos[:, None] - pos[None, :]
    logits = logits.masked_fill(~((distance >= 0) & (distance < window)), float("-inf"))
    if sinks is not None:
        sink_logits = sinks[None, :, None, None].expand(*logits.shape[:-1], 1)
        logits = torch.cat((logits, sink_logits), dim=-1)
        v = torch.cat((v, torch.zeros_like(v[:, :1])), dim=1)
    return torch.einsum("bhls,bshd->blhd", logits.softmax(dim=-1), v)


class TestMambaGQASink(unittest.TestCase):
    def make_model(self, module, **kwargs):
        return module.Mamba2(
            d_model=128, d_state=16, headdim=32, expand=1, chunk_size=32,
            **kwargs,
        )

    def test_output_and_all_attention_gradients(self):
        # Includes MQA, GQA, MHA, sequence start, and query-block boundaries.
        cases = [(11, w, h) for w in (1, 4, 32) for h in (1, 2, 4)]
        cases += [(137, w, 2) for w in (4, 128, 200)]
        for module in MODULES:
            for length, window, kv_heads in cases:
                with self.subTest(module=module.__name__, length=length, window=window, kv_heads=kv_heads):
                    torch.manual_seed(12)
                    model = self.make_model(module, use_attn_sink=True, window_size=window,
                                            gqa_num_kv_heads=kv_heads)
                    with torch.no_grad():
                        model.attn_sinks.copy_(torch.tensor([-2.0, 0.0, 2.0, 4.0]))
                    q = torch.randn(2, length, 4, 32, requires_grad=True)
                    k = torch.randn(2, length, kv_heads, 32, requires_grad=True)
                    v = torch.randn_like(k, requires_grad=True)
                    tensors = (q, k, v, model.attn_sinks)
                    refs = tuple(t.detach().double().requires_grad_() for t in tensors)
                    out = dense_attention(q, k, v, window) * model._compute_sink_gate(q, k)[..., None]
                    ref = dense_attention(*refs[:3], window, sinks=refs[3])
                    grad_out = torch.randn_like(out)
                    grads = torch.autograd.grad(out, tensors, grad_out)
                    ref_grads = torch.autograd.grad(ref, refs, grad_out.double())
                    torch.testing.assert_close(out.double(), ref, atol=2e-6, rtol=2e-5)
                    for grad, ref_grad in zip(grads, ref_grads):
                        torch.testing.assert_close(grad.double(), ref_grad, atol=5e-6, rtol=3e-5)

    def test_fp16_dot_product_does_not_overflow(self):
        for module in MODULES:
            model = self.make_model(module, use_attn_sink=True, window_size=1)
            q = torch.full((1, 3, 4, 32), 128.0, dtype=torch.float16, requires_grad=True)
            k = torch.full((1, 3, 1, 32), 128.0, dtype=torch.float16, requires_grad=True)
            with torch.no_grad():
                model.attn_sinks.fill_(128.0 ** 2 * 32 ** 0.5)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                gate = model._compute_sink_gate(q, k)
            torch.testing.assert_close(gate, torch.full_like(gate, 0.5))
            gate.sum().backward()
            for grad in (q.grad, k.grad, model.attn_sinks.grad):
                self.assertTrue(torch.isfinite(grad).all())
                self.assertGreater(grad.abs().sum().item(), 0)

    def test_opt_in_initialization_and_invalid_settings(self):
        for module in MODULES:
            baseline = self.make_model(module)
            self.assertNotIn("attn_sinks", baseline.state_dict())
            self.make_model(module, use_attn_sink=False).load_state_dict(baseline.state_dict(), strict=True)
            sink_model = self.make_model(module, use_attn_sink=True)
            torch.testing.assert_close(sink_model.attn_sinks, torch.full((4,), -2.0))
            self.assertTrue(sink_model.attn_sinks._no_weight_decay)
            for settings in ({"use_flash_swa": False}, {"window_size": 0},
                             {"window_size": -1}, {"sink_init": float("nan")}):
                with self.assertRaises(ValueError):
                    self.make_model(module, use_attn_sink=True, **settings)

    @unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA and FlashAttention-2")
    def test_flash_attention_output_and_gradients(self):
        for module in MODULES:
            for dtype in (torch.float16, torch.bfloat16):
                with self.subTest(module=module.__name__, dtype=dtype):
                    torch.manual_seed(27)
                    model = self.make_model(module, use_attn_sink=True, window_size=4, device="cuda")
                    q = torch.randn(2, 137, 4, 32, device="cuda", dtype=dtype, requires_grad=True)
                    k = torch.randn(2, 137, 1, 32, device="cuda", dtype=dtype, requires_grad=True)
                    v = torch.randn_like(k, requires_grad=True)
                    cos, sin = model._rotary_cos_sin(137, q.device, q.dtype)
                    qr, kr = model._apply_rotary(q, cos, sin), model._apply_rotary(k, cos, sin)
                    tensors = (qr, kr, v, model.attn_sinks)
                    refs = tuple(t.detach().float().requires_grad_() for t in tensors)
                    out = model._attention(qr, kr, v)
                    ref = dense_attention(*refs[:3], 4, sinks=refs[3])
                    grad_out = torch.randn_like(out)
                    grads = torch.autograd.grad(out, tensors, grad_out)
                    ref_grads = torch.autograd.grad(ref, refs, grad_out.float())
                    atol = 0.004 if dtype == torch.float16 else 0.035
                    torch.testing.assert_close(out.float(), ref, atol=atol, rtol=atol)
                    for grad, ref_grad in zip(grads, ref_grads):
                        torch.testing.assert_close(grad.float(), ref_grad, atol=atol, rtol=atol)

    @unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA and FlashAttention-2")
    def test_full_model_backward_and_baseline_limit(self):
        for module in MODULES:
            for dropout in (0.0, 0.2):
                torch.manual_seed(31)
                model = self.make_model(module, use_attn_sink=True, window_size=4,
                                        dropout_p=dropout, device="cuda")
                x = torch.randn(2, 17, 128, device="cuda", requires_grad=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(x)
                self.assertEqual(out.shape, x.shape)
                out.float().square().mean().backward()
                for grad in (x.grad, model.attn_sinks.grad, model.input_q_proj.weight.grad,
                             model.input_k_proj.weight.grad, model.input_v_proj.weight.grad):
                    self.assertIsNotNone(grad)
                    self.assertTrue(torch.isfinite(grad).all())
                    self.assertGreater(grad.abs().sum().item(), 0)
                # A very negative sink must recover the unmodified baseline.
                model.eval()
                baseline = self.make_model(module, window_size=4, device="cuda").eval()
                baseline.load_state_dict({k: v for k, v in model.state_dict().items() if k != "attn_sinks"})
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    model.attn_sinks.fill_(-1000)
                    torch.testing.assert_close(model(x), baseline(x), atol=0, rtol=0)
                    flat_out = model(x.reshape(-1, 128), seqlen=17)
                    torch.testing.assert_close(flat_out, baseline(x).reshape(-1, 128), atol=0, rtol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA and FlashAttention-2")
    def test_optimized_matches_recompute_with_dropout(self):
        for module in MODULES:
            for dtype in (torch.float16, torch.bfloat16):
                for dropout in (0.0, 0.2):
                    for window, kv_heads, dim in ((1, 4, 32), (4, 1, 10), (128, 2, 64)):
                        with self.subTest(module=module.__name__, dtype=dtype, dropout=dropout,
                                          window=window, kv_heads=kv_heads, dim=dim):
                            torch.manual_seed(34)
                            model = self.make_model(module, use_attn_sink=True, window_size=window,
                                                    dropout_p=dropout, device="cuda")
                            with torch.no_grad():
                                model.attn_sinks.copy_(torch.tensor([-2., 0., 2., 4.], device="cuda"))
                            q = torch.randn(1, 73, 4, dim, device="cuda", dtype=dtype, requires_grad=True)
                            k = torch.randn(1, 73, kv_heads, dim, device="cuda", dtype=dtype, requires_grad=True)
                            v = torch.randn_like(k, requires_grad=True)
                            tensors = (q, k, v, model.attn_sinks)
                            rng = torch.cuda.get_rng_state()
                            out = model._attention(q, k, v)
                            # Replay the identical FA2 dropout mask for the old implementation.
                            torch.cuda.set_rng_state(rng)
                            ref = module.flash_attn_func(q, k, v, dropout_p=dropout,
                                                         causal=True, window_size=(window - 1, 0))
                            ref = ref * model._compute_sink_gate(q, k)[..., None].to(dtype)
                            upstream = torch.randn_like(out)
                            grads = torch.autograd.grad(out, tensors, upstream)
                            ref_grads = torch.autograd.grad(ref, tensors, upstream)
                            tol = 0.006 if dtype == torch.float16 else 0.06
                            torch.testing.assert_close(out, ref, atol=tol, rtol=tol)
                            for grad, ref_grad in zip(grads, ref_grads):
                                torch.testing.assert_close(grad, ref_grad, atol=tol, rtol=tol)

    def test_packed_metadata_is_rejected_instead_of_ignored(self):
        for module in MODULES:
            for sink in (False, True):
                model = self.make_model(module, use_attn_sink=sink)
                x = torch.zeros(1, 17, 128)
                for metadata in ({"cu_seqlens": torch.tensor([0, 8, 17], dtype=torch.int32)},
                                 {"seq_idx": torch.tensor([[0] * 8 + [1] * 9], dtype=torch.int32)}):
                    with self.assertRaisesRegex(NotImplementedError, "packed sequence"):
                        model(x, **metadata)
        sys.path.insert(0, str(ROOT / "bla_gpt"))
        core = load_module("bla_gpt/bla_gpt.py", "gqa_integration")
        cfg = core.GPTConfig(vocab_size=64, n_layer=1, n_embd=128, n_head=4,
                             attention="mamba2_GQA", mamba_headdim=32, mamba_use_attn_sink=True)
        model = core.GPT(cfg)
        with self.assertRaisesRegex(NotImplementedError, "packed sequence"):
            model(torch.zeros(1, 17, dtype=torch.long), cu_seqlens=torch.tensor([0, 8, 17]))

    @unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA and FlashAttention-2")
    def test_batch_rows_remain_independent(self):
        for module in MODULES:
            for sink in (False, True):
                model = self.make_model(module, use_attn_sink=sink, device="cuda").eval()
                x = torch.randn(2, 17, 128, device="cuda", requires_grad=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(x)
                grad = torch.autograd.grad(out[1].float().sum(), x)[0]
                self.assertEqual(grad[0].abs().max().item(), 0.0)
                changed = x.detach().clone()
                changed[0] *= 5
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    changed_out = model(changed)
                torch.testing.assert_close(out[1], changed_out[1], atol=0, rtol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA and FlashAttention-2")
    def test_dense_attention_matches_sink_for_strong_and_weak_sinks(self):
        for module in MODULES:
            for window in (1, 4, 512):
                for kv_heads in (1, 2, 4):
                    torch.manual_seed(55)
                    q = torch.randn(1, 19, 4, 32, device="cuda", dtype=torch.float16, requires_grad=True)
                    k = torch.randn(1, 19, kv_heads, 32, device="cuda", dtype=torch.float16, requires_grad=True)
                    v = torch.randn_like(k, requires_grad=True)
                    s = torch.tensor([-100., -2., 3., 100.], device="cuda", requires_grad=True)
                    tensors = (q, k, v, s)
                    refs = tuple(t.detach().double().requires_grad_() for t in tensors)
                    out = module.flash_attn_sink_func(q, k, v, s, window)
                    ref = dense_attention(*refs[:3], window, refs[3])
                    upstream = torch.randn_like(out)
                    grads = torch.autograd.grad(out, tensors, upstream)
                    ref_grads = torch.autograd.grad(ref, refs, upstream.double())
                    torch.testing.assert_close(out.double(), ref, atol=0.004, rtol=0.004)
                    for grad, ref_grad in zip(grads, ref_grads):
                        torch.testing.assert_close(grad.double(), ref_grad, atol=0.006, rtol=0.006)



if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
