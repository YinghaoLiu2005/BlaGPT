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
