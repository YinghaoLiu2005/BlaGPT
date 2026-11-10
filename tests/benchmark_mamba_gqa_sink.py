"""Compare no sink, the original LSE recomputation, and the optimized FA2 sink.

Run with the blagpt environment and CUDA_VISIBLE_DEVICES set to an idle GPU.
Includes backward; timings exclude warmup/compilation and memory excludes
already-resident inputs/weights. The original implementation is retained as
Mamba2._compute_sink_gate so this comparison remains reproducible.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import time
import types

import torch
import flash_attn


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("benchmark_gqa", ROOT / "mamba/mamba_ssm/modules/mamba-gqa-sink.py")
gqa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gqa)


def old_attention(self, q, k, v, cu_seqlens=None, max_seqlen=None):
    assert cu_seqlens is None
    out = gqa.flash_attn_func(q, k, v, causal=True, window_size=(self.window_size - 1, 0))
    return out * self._compute_sink_gate(q, k).unsqueeze(-1).to(out.dtype)


def measure(forward, leaves, upstream, warmup, iterations, repeats):
    def clear():
        for value in leaves:
            value.grad = None

    def step():
        clear()
        out = forward()
        out.backward(upstream)

    for _ in range(warmup):
        step()
    clear()
    torch.cuda.synchronize()
    # One step measures live tensor memory independently of timing repeats.
    torch.cuda.reset_peak_memory_stats()
    starting = torch.cuda.memory_allocated()
    step()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    clear()
    wall_times, gpu_times = [], []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        wall_start = time.perf_counter()
        for _ in range(iterations):
            step()
        end.record()
        torch.cuda.synchronize()
        wall_times.append((time.perf_counter() - wall_start) * 1000 / iterations)
        gpu_times.append(start.elapsed_time(end) / iterations)
    clear()
    return dict(
        forward_backward_ms=statistics.median(wall_times),
        cuda_event_ms=statistics.median(gpu_times),
        repeat_wall_ms=wall_times,
        incremental_peak_mib=(peak - starting) / 2**20,
        total_peak_mib=peak / 2**20,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scopes", nargs="+", choices=("attention", "layer"), default=["attention", "layer"])
    parser.add_argument("--windows", nargs="+", type=int, default=[4, 128, 512])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(20261110)
    config = {**vars(args), "output": str(args.output)}
    result = dict(
        gpu=torch.cuda.get_device_name(), torch=torch.__version__, flash_attn=flash_attn.__version__,
        dtype="bfloat16", config=config,
        measurement="median wall time per forward+backward, warmup excluded; incremental allocated peak includes gradients",
        rows=[],
    )
    batch, length, heads, kv_heads, dim = args.batch_size, args.sequence_length, args.heads, args.kv_heads, args.head_dim
    for scope in args.scopes:
        model = gqa.Mamba2(heads * dim, headdim=dim, gqa_num_kv_heads=kv_heads,
                          d_state=128, expand=1, chunk_size=256, use_attn_sink=True, device="cuda")
        q = torch.randn(batch, length, heads, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(batch, length, kv_heads, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn_like(k, requires_grad=True)
        x = torch.randn(batch, length, heads * dim, device="cuda", requires_grad=True)
        shape = q.shape if scope == "attention" else x.shape
        upstream = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        leaves = (q, k, v, model.attn_sinks) if scope == "attention" else (x, *model.parameters())

        def forward():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return model._attention(q, k, v) if scope == "attention" else model(x)

        for window in args.windows:
            model.window_size = window
            for mode in ("no_sink", "recompute", "optimized"):
                model.use_attn_sink = mode != "no_sink"
                fn = old_attention if mode == "recompute" else gqa.Mamba2._attention
                model._attention = types.MethodType(fn, model)
                row = dict(scope=scope, window=window, mode=mode, **measure(
                    forward, leaves, upstream, args.warmup, args.iterations, args.repeats,
                ))
                result["rows"].append(row)
                print(json.dumps(row), flush=True)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2) + "\n")
        del model, q, k, v, x, leaves, upstream
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
