# torch-linalg-nan-guard

[![English](https://img.shields.io/badge/English-555555?style=flat)](README.md) [![简体中文](https://img.shields.io/badge/简体中文-555555?style=flat)](README.zh-CN.md)

Two independent guards against silent PyTorch `torch.linalg` correctness bugs:

1. Reject NaN and Inf before calling `torch.linalg.svdvals()` or `torch.linalg.eigvalsh()`, which can silently return finite-looking values for invalid input instead of raising or propagating NaN.
2. Compute `torch.linalg.vector_norm()` / `torch.norm()` (L2/Frobenius) without PyTorch's CPU float32 accumulator bug, which silently loses precision or overflows to `inf` (with a resulting silent all-zero gradient) for ordinary float16/bfloat16/float32 input.

Both guards are explicit call-site wrappers with a CLI that checks how your installed PyTorch build actually behaves. Neither patches PyTorch globally.

## Install and try

Requires Python 3.9+ and a compatible PyTorch 2.0+ installation. Actual Python support also depends on your PyTorch wheel.

```bash
git clone https://github.com/zhuhroscar-tech/torch-linalg-nan-guard.git
cd torch-linalg-nan-guard
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[torch]"
torch-linalg-nan-guard --json
```

If you already manage a platform-specific PyTorch installation, install the package with `pip install -e .` in that environment instead.

```python
import torch
from torch_linalg_nan_guard import safe_svdvals, safe_eigvalsh, safe_vector_norm, safe_norm

matrix = torch.eye(3, dtype=torch.float64)
singular_values = safe_svdvals(matrix)
eigenvalues = safe_eigvalsh(matrix, UPLO="L")

x = torch.randn(50_000_000, dtype=torch.float32)
norm = safe_vector_norm(x, ord=2)   # or safe_norm(x)
```

For finite inputs, `safe_svdvals`/`safe_eigvalsh` delegate to PyTorch unchanged. `safe_vector_norm`/`safe_norm` delegate unchanged for `float64`/`complex` input and for any non-L2 `ord`/`p`; for the affected dtypes and L2 norm, they accumulate the sum-of-squares in `float64` internally and cast back to the input dtype. PyTorch's shape, dtype, device, and symmetry requirements still apply throughout.

## What the diagnostics mean

`torch-linalg-nan-guard` (no flags) runs both diagnoses. `--skip-norm-check` skips the norm precision/overflow diagnosis (it runs a 16M-element CPU reduction by default, which takes real time and memory on constrained hosts).

- **NaN-swallowing diagnostic**: compares values-only operations with their full-decomposition counterparts across small CPU matrices. JSON distinguishes an unguarded exception from a silently finite result and records whether the guard raises.
- **Norm precision diagnostic**: compares `torch.linalg.vector_norm()` at several element counts, and one deliberately large-magnitude case, against an independent `float64` reference computed from the exact same input. Reports the unguarded and guarded relative error at each size, and whether either overflows to `inf`.

Results depend on the installed PyTorch build/backend; a reported clean case is not a guarantee about every input, dtype, or device. Both diagnoses were reproduced from scratch on this project's own macOS CPU host (see the tests for the exact repro) and verified on `ubuntu-latest` CI; GPU/MPS-specific accumulator behavior is not covered by either diagnostic.

Related reports:
- NaN-swallowing: [PyTorch #187759](https://github.com/pytorch/pytorch/issues/187759), [NumPy #20280](https://github.com/numpy/numpy/issues/20280).
- Norm precision/overflow: [PyTorch #169237](https://github.com/pytorch/pytorch/issues/169237) (precision loss; a proposed fix, [PR #169996](https://github.com/pytorch/pytorch/pull/169996), was closed without merging after review), [PyTorch #193006](https://github.com/pytorch/pytorch/issues/193006) (overflow/silent-zero-gradient; a proposed fix, [PR #194326](https://github.com/pytorch/pytorch/pull/194326), remains an open, unmerged draft). Both issues are open as of this writing.

## Limits and development

- Only `svdvals`/`eigvalsh` (NaN-swallowing) and `vector_norm`/`norm` L2/Frobenius (accumulator precision/overflow) are guarded; this is not a general numerical-correctness checker.
- NaN-swallowing guard: the whole tensor is scanned. One invalid element rejects an entire batch; no per-matrix attribution is provided.
- Norm guard: the `float64` intermediate buffer roughly doubles peak memory for the reduction versus the native (buggy) path, and is itself still subject to `float64`'s own (far larger, but nonzero) precision and range limits at truly extreme scales.
- Scanning/casting adds work, and `.item()` may synchronize accelerators. GPU-specific accumulator behavior is not covered by either CPU diagnostic -- both bugs were reproduced and are guarded here only on CPU float16/bfloat16/float32 tensors.

```bash
pip install -e ".[dev,torch]"
pytest -v --cov=torch_linalg_nan_guard --cov-report=term-missing
```

[MIT license](LICENSE).
