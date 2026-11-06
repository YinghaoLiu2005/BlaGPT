# KDA MQAR clean rebuild

This directory is a fresh clone of HazyResearch/Zoology at commit
`1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb`. Its FLA submodule is checked out
at `a400aff6ec0eb81cc88721aedba782cd7d535357` (the upstream main revision
that contains `KimiDeltaAttention`; the pinned Zoology submodule revision does
not contain KDA).
The previous `output/`, custom training scripts, and old `kda_mqar` directory
were removed before this recipe was added.

The recipe uses Zoology's `MQARConfig` and model/training configuration. The
primary sweep compares KDA, GDN, Mamba2, and the local Mamba2-GQA mixer. It
uses lengths 256/512/1024/2048, `num_kv_pairs=L/4`, two layers, two heads,
head dimension 128, the four paper learning rates, and seed 1337.

The current Zoology trainer is epoch based. The recipe uses 20,000 examples,
batch size 64, and 64 epochs, which is approximately 20,000 optimizer steps;
the exact step count must be recorded in the run metadata before using the
results as a paper comparison. Native MQAR has no separator token; the
separator variant is intentionally a follow-up ablation.

The host has four RTX 3090 GPUs and driver 570.133.20. The Codex session's
ordinary namespace does not expose `/dev/nvidia*`; the training/check scripts
must run in the GPU-enabled namespace. The blagpt environment's
`mamba_ssm/__init__.py` had optional sliding imports that referenced a missing
module; it is patched to expose the installed official Mamba2 implementation.
