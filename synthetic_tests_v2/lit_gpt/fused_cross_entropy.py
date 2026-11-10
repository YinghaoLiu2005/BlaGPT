"""Same mean, ignore_index=-100 objective without a FlashAttention dependency."""
from torch.nn import CrossEntropyLoss as FusedCrossEntropyLoss
