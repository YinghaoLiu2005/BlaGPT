"""Zoology-native KDA Figure 4 reconstruction sweep.

Each length is trained with L/4 key-value pairs, matching Zoology's harder
MQAR scaling. The four learning rates and 20k-step budget come from KDA §5.1.
"""
import uuid
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig
from zoology.data.multiquery_ar import MQARConfig

VOCAB = 8192
LENGTHS = (256, 512, 1024, 2048)
LRS = (5e-5, 1e-4, 5e-4, 1e-3)
MODELS = {
    "kda": "zoology.experiments.kda_mqar.mixers.KDA",
    "gdn": "zoology.experiments.kda_mqar.mixers.GDN",
    "mamba2": "zoology.experiments.kda_mqar.mixers.Mamba2",
    "mamba2_gqa": "zoology.experiments.kda_mqar.mixers.Mamba2GQA",
}

configs = []
for name, mixer_name in MODELS.items():
    for length in LENGTHS:
        kv = length // 4
        data = DataConfig(
            train_configs=[MQARConfig(vocab_size=VOCAB, input_seq_len=length,
                                      num_examples=20_000, num_kv_pairs=kv)],
            test_configs=[MQARConfig(vocab_size=VOCAB, input_seq_len=length,
                                     num_examples=1_000, num_kv_pairs=kv)],
            batch_size=(64, 64), seed=1337,
        )
        for lr in LRS:
            model = ModelConfig(
                name=f"{name}_L{length}", vocab_size=VOCAB, d_model=256,
                n_layers=2, max_position_embeddings=0, embed_dropout=0.0,
                resid_dropout=0.0, block_type="TransformerBlock",
                sequence_mixer=ModuleConfig(name=mixer_name),
            )
            configs.append(TrainConfig(
                model=model, data=data, learning_rate=lr,
                max_epochs=64, weight_decay=0.0, scheduler="constant",
                early_stopping_metric=None, slice_keys=["num_kv_pairs"],
                seed=1337, run_id=f"{name}-L{length}-lr{lr:.0e}-{uuid.uuid4().hex[:6]}",
            ))
