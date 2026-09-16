"""torch-linalg-nan-guard: detect and safely guard against
torch.linalg.svdvals()/eigvalsh() silently swallowing NaN input.
"""
from .core import (
    TorchUnavailableError,
    diagnose,
    safe_eigvalsh,
    safe_svdvals,
)

__all__ = [
    "TorchUnavailableError",
    "diagnose",
    "safe_eigvalsh",
    "safe_svdvals",
]

__version__ = "0.1.0"
