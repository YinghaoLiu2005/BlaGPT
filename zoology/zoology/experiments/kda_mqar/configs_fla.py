"""KDA/GDN half of the sweep; run this with the fla environment."""
from zoology.experiments.kda_mqar.configs import configs as _configs

configs = [c for c in _configs if c.model.name.startswith(("kda_", "gdn_"))]
