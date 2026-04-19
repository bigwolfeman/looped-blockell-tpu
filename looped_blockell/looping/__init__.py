"""Looping infrastructure: diagonal injection, depth sampling, scan-based model."""

from .diagonal_injection import DiagonalInjection
from .depth_sampler import DepthPlan, sample_depth, sample_fixed
from .looped_model import LoopedTransformer, create_looped_transformer, model_fwd

__all__ = [
    "DiagonalInjection",
    "DepthPlan",
    "sample_depth",
    "sample_fixed",
    "LoopedTransformer",
    "create_looped_transformer",
    "model_fwd",
]
