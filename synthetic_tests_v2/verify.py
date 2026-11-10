"""Source/config tests plus native GPU and unchanged-trainer integration tests.

python verify.py --cpu-only
FLA_PYTHON=... MAMBA_PYTHON=... CUDA_VISIBLE_DEVICES=0 python verify.py
"""
import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent
ARCHS = ("KDA", "GDN", "Mamba2", "Mamba2GQA", "Mamba2GQASink")
TASKS = {"mqar": "MQAR", "palindrome": "Palindrome", "stack": "Stack"}


def integrity():
    manifest = json.loads((ROOT / "source_manifest.json").read_text())
    for name, expected in manifest["files"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    for task in TASKS:
        for name in (f"{task}_data.py", f"train_{task}.py", f"{task}_analysis.py"):
            assert (ROOT / name).read_bytes() == (ROOT / "upstream/kla" / name).read_bytes(), name
        # Shell defaults, flags and loop structure must be identical apart from
        # model names and interpreter dispatch. This includes optional LR sweep.
        original = (ROOT / f"upstream/kla/run_{task}_experiment.sh").read_text()
        current = (ROOT / f"run_{task}_experiment.sh").read_text()
        declared = re.search(r'MODELS=\((.*?)\)', current, re.S).group(1)
        assert re.findall(r'"([^"]+)"', declared) == [f"{a}_{TASKS[task]}" for a in ARCHS]
        def normalize(text):
            text = re.sub(r'MODELS=\(.*?\)', 'MODELS=()', text, count=1, flags=re.S)
            text = re.sub(r'^PYTHON_BIN=.*\n', '', text, flags=re.M)
            text = text.replace('"${PYTHON_BIN}"', 'python').replace('${PYTHON_BIN}', 'python')
            return '\n'.join(line for line in text.splitlines() if line.strip())
        assert normalize(original) == normalize(current), f"Launcher config drift: {task}"
    assert (ROOT / "lit_gpt/config.py").read_bytes() == (ROOT / "upstream/kla/lit_gpt/config.py").read_bytes()
    return {"hashed_files": len(manifest["files"]), "unchanged_task_files": 9,
            "launcher_config_parity": 3}


def data_tests():
    import numpy as np
    from lit_gpt.model import Config
    checks = []
    for task, suffix in TASKS.items():
        module = importlib.import_module(f"{task}_data")
        spec = importlib.util.spec_from_file_location(f"upstream_{task}", ROOT / f"upstream/kla/{task}_data.py")
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        options = dict(num_examples=8, seq_len=128, seed=42)
        if task == "palindrome":
            options["predict_first_token"] = True
        actual = getattr(module, f"generate_{task}_data")(**options)
        expected = getattr(original, f"generate_{task}_data")(**options)
        for key in actual:
            np.testing.assert_array_equal(actual[key], expected[key])
        x, y, mask = (actual[key] for key in ("input_ids", "labels", "masks"))
        np.testing.assert_array_equal(mask, y != -100)
        assert mask.any(axis=1).all()
        if task == "palindrome":
            for xx, yy in zip(x, y):
                np.testing.assert_array_equal(yy[yy != -100], xx[:128][::-1])
        elif task == "stack":
            for xx, yy in zip(x, y):
                stacks = [[] for _ in range(64)]
                expected_y = np.full_like(yy, -100)
                for pos in range(0, len(xx) - 2, 3):
                    op, sid, value = xx[pos:pos + 3]
                    if op == 90:
                        stacks[sid].append(value)
                    else:
                        assert op == 91 and value == 92
                        expected_y[pos + 1] = stacks[sid].pop()
                np.testing.assert_array_equal(yy, expected_y)
        # MQAR deliberately keeps upstream's replacement sampling and layout.
        for arch in ARCHS:
            config = Config.from_name(f"{arch}_{suffix}")
            assert config.n_layer == 2 and config.n_embd == 256
            assert config.n_head == (4 if task == "mqar" else 2)
            assert config.vocab_size == (8192 if task == "mqar" else 128)
            assert config.mamba_init and not config.parallel_residual
        checks.append(task)
    from analysis_plot_style import normalize_model_name
    assert len({normalize_model_name(f"{a}_MQAR") for a in ARCHS}) == 5
    return {"data_parity": checks, "independent_oracles": ["palindrome", "stack"],
            "model_task_configs": 15, "plot_labels_distinct": True}


def native_tests(architecture):
    import functools
    import torch
    from torch.nn import functional as F
    from lit_gpt.model import GPT, Config

    torch.set_num_threads(1)
    assert torch.cuda.is_available(), "GPU tests require CUDA; no CPU substitution"
    results = []
    for task, suffix in TASKS.items():
        torch.manual_seed(42)
        config = Config.from_name(f"{architecture}_{suffix}")
        model = GPT(config).cuda()
        model.apply(functools.partial(model._init_weights, n_layer=2))
        assert model.lm_head.weight is model.transformer.wte.weight
        mixer = model.transformer.h[0].attn
        source = Path(sys.modules[type(mixer).__module__].__file__).resolve()
        assert source.is_relative_to(ROOT / "vendor"), source
        if architecture.startswith("Mamba"):
            assert mixer.nheads == config.n_head and mixer.d_state == 128 and mixer.expand == 1
            if architecture != "Mamba2":
                assert mixer.attn_kv_nheads == 1 and mixer.window_size == 512
        else:
            assert mixer.num_heads == config.n_head and mixer.head_k_dim == config.head_size
        generator = getattr(importlib.import_module(f"{task}_data"), f"generate_{task}_data")
        options = dict(num_examples=2, seq_len=256 if task == "mqar" else 1024, seed=42)
        if task == "palindrome":
            options["predict_first_token"] = True
        data = generator(**options)
        x = torch.as_tensor(data["input_ids"], device="cuda")
        labels = torch.as_tensor(data["labels"], device="cuda")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=.1, betas=(.9, .95))
        before = model.transformer.wte.weight.detach().clone()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        loss.backward()
        assert torch.isfinite(loss)
        for name, param in model.named_parameters():
            assert param.grad is not None and torch.isfinite(param.grad).all(), name
        if architecture == "Mamba2GQASink":
            assert mixer.use_attn_sink and (mixer.attn_sinks == -2).all()
            assert mixer.attn_sinks.grad.abs().sum() > 0
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        assert not torch.equal(before, model.transformer.wte.weight)
        del before, logits
        # Change only a suffix; all preceding logits must be unchanged.
        model.eval()
        tokens = x[:, :128].clone()
        changed = tokens.clone()
        changed[:, 64:] = (changed[:, 64:] + 1) % config.vocab_size
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            reference = model(tokens)
            alternate = model(changed)
        torch.testing.assert_close(reference[:, :64], alternate[:, :64], atol=2e-3, rtol=2e-3)
        # Actual serialization roundtrip, including tied embeddings.
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "state.pt"
            torch.save(model.state_dict(), checkpoint)
            restored = GPT(config).cuda().eval()
            restored.load_state_dict(torch.load(checkpoint, weights_only=True))
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                roundtrip = restored(tokens)
            torch.testing.assert_close(reference, roundtrip, rtol=0, atol=0)
            assert restored.lm_head.weight is restored.transformer.wte.weight
        # Largest upstream evaluation length (Palindrome uses 2*source+1 tokens).
        max_length = 4097 if task == "palindrome" else 2048
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            extrapolated = model(torch.zeros((1, max_length), dtype=torch.long, device="cuda"))
        assert torch.isfinite(extrapolated).all()
        results.append(dict(task=task, architecture=architecture, loss=float(loss.detach()),
                            train_tokens=x.shape[1], eval_tokens=max_length,
                            finite_gradients=True, optimizer_update=True, causal=True,
                            checkpoint_roundtrip=True, source=str(source.relative_to(ROOT))))
        print(f"PASS native {architecture}/{task}", flush=True)
        del model, restored, optimizer, mixer, reference, alternate, roundtrip, extrapolated, loss
        torch.cuda.empty_cache()
    return dict(torch=torch.__version__, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(), checks=results)


def integration_tests(architecture, output):
    import numpy as np
    results = []
    for task, suffix in TASKS.items():
        # Test overrides are isolated under test_outputs, never formal data/out.
        folder = output / "integration" / architecture / task
        data_dir = folder / "seq128_key1_seed42"
        data_dir.mkdir(parents=True, exist_ok=True)
        generator = getattr(importlib.import_module(f"{task}_data"), f"generate_{task}_data")
        for index, split in enumerate(("train", "val", "test")):
            kwargs = dict(num_examples=4, seq_len=128, seed=42 + index)
            if task == "palindrome":
                kwargs["predict_first_token"] = True
            np.savez_compressed(data_dir / f"{split}.npz", **generator(**kwargs))
        run_dir = folder / "out"
        if run_dir.exists():
            raise FileExistsError(f"Use a fresh --output directory: {run_dir}")
        command = [sys.executable, str(ROOT / f"train_{task}.py"),
                   "--model_name", f"{architecture}_{suffix}", "--data_dir", str(data_dir),
                   "--out_dir", str(run_dir), "--wandb_dir", str(folder / "wandb"),
                   "--max_steps", "2", "--batch_size", "2", "--val_interval", "1",
                   "--save_interval", "2", "--log_interval", "1", "--num_workers", "0", "--debug"]
        if task == "mqar":
            command += ["--extrapolation_factors", "1,2", "--extrapol_num_val", "4"]
        else:
            command += ["--eval_seq_lens", "128,256", "--eval_num_val", "4", "--n_head", "2"]
            if task == "palindrome":
                command += ["--predict_first_token"]
        with (folder / "console.log").open("w") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = json.loads((run_dir / "results.json").read_text())
        assert result["model_name"] == f"{architecture}_{suffix}"
        assert 0 <= result["test_acc"] <= 1 and 0 <= result["best_val_acc"] <= 1
        assert (run_dir / "ckpt_step_2.pth").exists()
        evaluations = result.get("final_val_acc", result.get("eval_acc_by_seq_len"))
        assert len(evaluations) == 2 and all(0 <= value <= 1 for value in evaluations.values())
        console = (folder / "console.log").read_text()
        assert "Loading best model" in console
        losses = re.findall(r"Step \d+: Loss (\S+), Acc", console)
        assert len(losses) == 2 and all(math.isfinite(float(value)) for value in losses)
        assert "non-finite loss detected" not in console
        results.append(dict(task=task, architecture=architecture, steps=2,
                            test_acc=result["test_acc"], evaluation=evaluations))
        print(f"PASS original trainer {architecture}/{task}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--worker", choices=("cpu",) + ARCHS)
    parser.add_argument("--output", type=Path, default=ROOT / "test_outputs" / time.strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        if args.worker == "cpu":
            report = dict(integrity=integrity(), data=data_tests())
        else:
            report = dict(native=native_tests(args.worker),
                          integration=integration_tests(args.worker, output))
        (output / f"{args.worker}.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"PASS {args.worker}", flush=True)
        return
    workers = ("cpu",) if args.cpu_only else ("cpu",) + ARCHS
    report = {}
    for worker in workers:
        python = os.environ.get("FLA_PYTHON" if worker in ("KDA", "GDN") else "MAMBA_PYTHON", sys.executable)
        print(f"Testing {worker} with {python}; logs: {output / (worker + '.log')}", flush=True)
        with (output / f"{worker}.log").open("w") as log:
            subprocess.run([python, str(Path(__file__).resolve()), "--worker", worker,
                            "--output", str(output)], cwd=ROOT, stdout=log,
                           stderr=subprocess.STDOUT, check=True,
                           env=dict(os.environ, WANDB_MODE="disabled", OMP_NUM_THREADS="1"))
        report[worker] = json.loads((output / f"{worker}.json").read_text())
        print(f"PASS {worker}", flush=True)
    if not args.cpu_only:
        python = os.environ.get("MAMBA_PYTHON", sys.executable)
        print("Testing operator equivalence and Sink gradients", flush=True)
        with (output / "operators.json").open("w") as log, (output / "operators.log").open("w") as errors:
            subprocess.run([python, str(ROOT / "check_operators.py")], cwd=ROOT,
                           stdout=log, stderr=errors, check=True)
        report["operators"] = json.loads((output / "operators.json").read_text())
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"All checks passed: {output / 'report.json'}")


if __name__ == "__main__":
    main()
