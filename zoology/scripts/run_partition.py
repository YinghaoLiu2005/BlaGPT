"""Run one deterministic partition of a Zoology config module."""
import importlib
import sys

from zoology.train import train

module_name, partition, partitions = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
module = importlib.import_module(module_name)
configs = module.configs
selected = configs[partition::partitions]
print(f"partition {partition}/{partitions}: {len(selected)} configs", flush=True)
for config in selected:
    train(config)
