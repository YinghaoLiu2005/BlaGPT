# KDA paper MQAR reproduction

This is a clean Zoology-native sweep. It uses `MQARConfig`, `DataConfig`,
`ModelConfig`, `TrainConfig`, and `zoology.launch`; no independent data
generator or model shell is used.

The primary protocol has four models: KDA, GDN, official Zoology Mamba2, and
the local Mamba2-GQA implementation copied under `third_party/mamba_ssm`.
Every length uses `num_kv_pairs = length / 4`: 64, 128, 256, and 512. The
paper's 2-layer, 2-head, 128-head-dimension, 20k-step and four-learning-rate
settings are represented here. Zoology's current trainer expresses the
20,000-step budget as 64 epochs over 20,000 examples at batch size 64; the
resulting step count is recorded in each run log.

Run after installing the repository and FLA submodule. The two model families
must use their separate environments:

```bash
bash scripts/run_kda_mqar_split.sh
```

`configs_fla.py` runs KDA/GDN with `/data/yinghaoliu/conda/conda_envs/fla/bin/python`.
`configs_blagpt.py` runs Mamba2/Mamba2-GQA with
`/data/yinghaoliu/conda/conda_envs/blagpt/bin/python`.

The primary run intentionally uses native Zoology MQAR, which has no explicit
separator token. A separate paper-format variant should be added only after
the native control reproduces Zoology's own MQAR behavior.

### Mamba2-GQA attention sink ablation

The adapter accepts an optional virtual, zero-value attention sink per query
head. Existing configurations keep it disabled. To enable it in a separate
experiment, set the sequence mixer explicitly:

```python
sequence_mixer=ModuleConfig(
    name="zoology.experiments.kda_mqar.mixers.Mamba2GQA",
    kwargs={"use_attn_sink": True, "sink_init": -2.0, "window_size": 128},
)
```

The implementations are separate files: `mamba2_GQA.py` is restored to the
original version before sinks were added; `mamba-gqa-sink.py` contains the
optimized sink implementation. The adapter and BlaGPT wrapper select the
appropriate file using the existing sink flag. The sink file's constructor
still accepts `use_attn_sink`; the wrappers pass `True` when selecting it.

Use the same window, data, seed, and training budget for the baseline. This
adapter defaults to a 128-token window; the standalone module defaults to 4,
and BlaGPT's `GPTConfig` defaults to 512. In BlaGPT, enable the same feature
with `mamba_use_attn_sink=True` and `mamba_sink_init=-2.0` in the model config,
or pass `--mamba_use_attn_sink --mamba_sink_init -2.0` to `bla_gpt/train.py`.

The gate is `sigmoid(logsumexp(local_logits) - sink_logit)`, applied before
the attention output projection. The optimized path reuses FA2's LSE and
supplies the sink-normalized LSE/output to its backward kernel, preserving
the full Q/K/V/sink gradients without recomputing QK. This uses the FA2 2.6.x
internal interface, validated with 2.6.3; incompatible interfaces fail with
an explicit error. The former differentiable recomputation remains as
`_compute_sink_gate` only for tests and benchmarks. No real token is inserted,
and a near-zero gate still leaves the output projection bias. Incremental
decoding and higher-order derivatives remain unsupported.

Independent samples in separate batch rows are already isolated; current MQAR
uses this layout. This implementation keeps the original fixed-length input
interface and does not add packed-sequence support. Passing `seq_idx` or
`cu_seqlens` to the sink implementation (or GPT with sinks enabled) raises an
explicit error rather than silently ignoring boundaries. The restored baseline
retains its original behavior, including ignoring those boundaries in attention.
For independent samples, use one sample per batch row.

The pretraining loader slices a continuous token stream and does not supply
document boundaries. Within a row, later documents can therefore use earlier
context; EOS tokens do not create a mask. This is cross-document context, not
by itself evidence of future-token or train/validation leakage. The SSM also
carries state across a row, so isolating attention alone would be insufficient.

Run the dense-reference output/gradient tests and CUDA smoke tests with:

```bash
cd /data/yinghaoliu/BlaGPT
CUDA_VISIBLE_DEVICES=0 /data/yinghaoliu/conda/conda_envs/blagpt/bin/python tests/test_mamba_gqa_sink.py
```

Use an idle GPU for reproducible speed/memory comparisons:

```bash
CUDA_VISIBLE_DEVICES=0 /data/yinghaoliu/conda/conda_envs/blagpt/bin/python tests/benchmark_mamba_gqa_sink.py --output logs/mamba_gqa_sink_benchmark.json
```
