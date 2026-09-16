# torch-linalg-nan-guard

[![English](https://img.shields.io/badge/English-555555?style=flat)](README.md) [![简体中文](https://img.shields.io/badge/简体中文-555555?style=flat)](README.zh-CN.md)

Reject NaN and Inf before calling PyTorch's `torch.linalg.svdvals()` or `torch.linalg.eigvalsh()`. The package provides two explicit wrappers and a CLI that checks how your installed PyTorch build handles non-finite inputs.

Some numerical backends can return finite-looking values for invalid inputs; others raise an error. This guard makes the policy explicit: invalid input raises `ValueError` before decomposition. It does not patch PyTorch globally.

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
from torch_linalg_nan_guard import safe_svdvals, safe_eigvalsh

matrix = torch.eye(3, dtype=torch.float64)
singular_values = safe_svdvals(matrix)
eigenvalues = safe_eigvalsh(matrix, UPLO="L")
```

For finite inputs, the wrappers delegate to PyTorch. `safe_svdvals` accepts `driver`; `safe_eigvalsh` accepts `UPLO`. PyTorch's shape, dtype, device, and symmetry requirements still apply.

## What the diagnostic means

The CLI compares values-only operations with their full-decomposition counterparts across small CPU matrices. JSON distinguishes an unguarded exception from a silently finite result and records whether the guard raises. Results depend on the installed backend; a reported clean case is not a guarantee about every input or device.

Related reports: [PyTorch #187759](https://github.com/pytorch/pytorch/issues/187759) and [NumPy #20280](https://github.com/numpy/numpy/issues/20280).

## Limits and development

- Only these two operations are guarded; this is not a general numerical-correctness checker.
- The whole tensor is scanned. One invalid element rejects an entire batch; no per-matrix attribution is provided.
- Scanning adds work, and `.item()` may synchronize accelerators. GPU-specific behavior is not covered by the CPU diagnostic.

```bash
pip install -e ".[dev,torch]"
pytest -v --cov=torch_linalg_nan_guard --cov-report=term-missing
```

[MIT license](LICENSE).
