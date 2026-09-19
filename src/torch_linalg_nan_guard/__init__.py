"""torch-linalg-nan-guard: detect and safely guard against
torch.linalg.svdvals()/eigvalsh() silently swallowing NaN input, and
torch.linalg.vector_norm()/torch.norm() silently losing precision or
overflowing to inf for float16/bfloat16/float32 input.
"""
from .core import (
    TorchUnavailableError,
    diagnose,
    diagnose_norm_precision,
    safe_eigvalsh,
    safe_norm,
    safe_svdvals,
    safe_vector_norm,
)

__all__ = [
    "TorchUnavailableError",
    "diagnose",
    "diagnose_norm_precision",
    "safe_eigvalsh",
    "safe_norm",
    "safe_svdvals",
    "safe_vector_norm",
]

__version__ = "0.2.0"
