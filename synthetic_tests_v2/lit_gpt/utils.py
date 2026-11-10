# Adapted from KLA lit_gpt/utils.py.
# Copyright (c) Microsoft Corporation. Licensed under the MIT license.
# Copyright Lightning AI. Licensed under the Apache License 2.0;
# see https://github.com/Lightning-AI/litgpt/blob/main/LICENSE.
def find_multiple(n, k):
    assert k > 0
    return ((n + k - 1) // k) * k


def num_parameters(module, requires_grad=None):
    return sum(p.numel() for p in module.parameters()
               if requires_grad is None or p.requires_grad == requires_grad)
