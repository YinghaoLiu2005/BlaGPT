"""Minimal package shim for the local Mamba2-GQA mixer.

The original project-level package imports optional MHA modules at package
import time. Zoology only needs ``modules.mamba2_GQA`` here, so keep this shim
small and avoid pulling unrelated optional kernels into the experiment.
"""
__version__ = "2.3.1-local"
