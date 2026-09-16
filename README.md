# torch-linalg-nan-guard

Detects and safely guards against a real, currently-open bug:
`torch.linalg.svdvals()` and `torch.linalg.eigvalsh()` silently return
**finite, plausible-looking** singular values / eigenvalues when the
input matrix contains NaN or Inf, instead of propagating NaN or
raising -- unlike their sibling full-decomposition functions
(`torch.linalg.svd()`, `torch.linalg.eigh()`), which mostly do
propagate NaN on the same input.

Tracking issues:

- [pytorch/pytorch#187759](https://github.com/pytorch/pytorch/issues/187759)
  (open, filed 2026-07-04) -- `svdvals` vs `svd` disagree on NaN input;
  root-caused in the issue thread to LAPACK's `dgesdd` routine not
  propagating NaN when called with `jobz='N'` (the values-only path
  `svdvals` uses), while the `jobz='S'`/`'A'` paths (`svd`'s
  `compute_uv=True`) happen to propagate it via the U/Vh computation.
  A fix PR ([#187765](https://github.com/pytorch/pytorch/pull/187765))
  was closed unmerged 2026-06-22 (maintainers asked for issue
  "actionable" triage before a PR, per PyTorch's contribution process);
  a second PR ([#188053](https://github.com/pytorch/pytorch/pull/188053),
  test-only) remains open as of this writing.
- [numpy/numpy#20280](https://github.com/numpy/numpy/issues/20280)
  (open since 2021) -- the identical inconsistency between
  `eigvalsh`/`eigh`, for NumPy. This package independently confirmed
  (see below) that PyTorch's `eigvalsh`/`eigh` pair reproduces the same
  pattern, since both share the same values-only-vs-full LAPACK
  dispatch design.

This package does not wait for or depend on either upstream fix; it
works around both bugs entirely at the call site with a simple,
independently-testable guard: check the input for NaN/Inf before
calling the real function, and raise a clear `ValueError` immediately
if found, rather than either silently returning a wrong finite answer
(the bug) or relying on the *sibling* function's inconsistent partial
NaN propagation. This mirrors the policy SciPy already applies by
default (`check_finite=True`), which raises `ValueError: array must
not contain infs or NaNs` for exactly this input.

## The bug, reproduced

```python
import torch

M = torch.tensor([[float("nan"), 0.0], [0.0, 1.0]], dtype=torch.float64)

print(torch.linalg.svdvals(M))     # tensor([1., 1.])   <- NaN silently swallowed
print(torch.linalg.svd(M).S)       # tensor([1., nan])  <- NaN correctly propagated

E = torch.diag(torch.tensor([1.0, 2.0], dtype=torch.float64))
E[0, 0] = float("nan")
print(torch.linalg.eigvalsh(E))    # tensor([0., -0.])  <- NaN silently swallowed
print(torch.linalg.eigh(E)[0])     # tensor([nan, 1.])  <- NaN correctly propagated
```

`svdvals`/`eigvalsh` and their full-decomposition siblings are
documented to compute the same singular values / eigenvalues, but here
they disagree -- and the values-only path is the one silently wrong.
A NaN anywhere in an embedding matrix, covariance matrix, or Gram
matrix (entirely plausible after a divide-by-zero, an unmasked padding
row, or an unstable upstream op) produces a finite, superficially
reasonable singular value or eigenvalue with **no error, warning, or
visible symptom** -- so downstream code (`matrix_rank`, spectral
normalization, condition-number checks, PCA/whitening) keeps running
on silently corrupted results.

## Usage

```python
from torch_linalg_nan_guard import safe_svdvals, safe_eigvalsh

safe_svdvals(some_matrix)     # raises ValueError if NaN/Inf present,
                              # otherwise identical to torch.linalg.svdvals
safe_eigvalsh(some_symmetric_matrix)  # same guard for eigvalsh
```

Or run the CLI to check whether the bug reproduces on your installed
torch build and confirm the guard catches every tested case:

```bash
torch-linalg-nan-guard
torch-linalg-nan-guard --json
```

## Independent verification (per this project's evidence-before-acceptance policy)

Before writing this package, the exact repro from pytorch/pytorch#187759
was re-run from scratch on the currently installed torch build (not
just cited from the issue tracker):

```
torch 2.14.0 (macOS, Accelerate/vecLib LAPACK backend)
svdvals(M): tensor([1., 1.], dtype=torch.float64)      <- bug reproduces
svd(M).S:   tensor([1., nan], dtype=torch.float64)
```

The `eigvalsh`/`eigh` inconsistency (mirroring numpy/numpy#20280) was
also independently reproduced on this host's torch build across matrix
sizes 2-4 and every diagonal NaN position, confirmed via the
`diagnose()` function this package ships (`tests/test_core.py` runs the
same checks as a regression suite). SciPy's own `check_finite=True`
default was checked as the intended-behavior baseline: SciPy raises
`ValueError: array must not contain infs or NaNs` for the same input,
confirming NaN-refusal (not silent computation) is the established
correct behavior this package's guard replicates for torch.

**LAPACK backend matters, confirmed by this project's own CI (a real
finding, not assumed):** the same torch version's `svdvals`/`eigvalsh`
NaN behavior is backend-dependent. On this development host (macOS,
Accelerate/vecLib), the silent-finite-result bug reproduces exactly as
pytorch/pytorch#187759 describes. On `ubuntu-latest` CI (OpenBLAS),
`torch.linalg.svd()` itself raises `_LinAlgError: ... input matrix
contained non-finite values` for the identical input, rather than
silently succeeding -- a stricter, already-loud failure rather than
the silent one. The `eigvalsh`/`eigh` pair still exhibits the silent
bug on Linux/OpenBLAS for several diagonal-NaN placements (confirmed
in CI logs). Either way, **the guard's job is unconditional**: raise a
clear `ValueError` before ever calling the underlying LAPACK routine,
regardless of which backend-specific behavior the unguarded call would
otherwise exhibit. The test suite and `diagnose()` output both record
this distinction explicitly (`buggy_raised` / `reference_raised`
fields) rather than assuming one platform's behavior universally.

## Reproducible build & test

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,torch]"
pytest -v --cov=torch_linalg_nan_guard --cov-report=term-missing
torch-linalg-nan-guard
```

CI runs this on both `ubuntu-latest` and `macos-latest` across Python
3.10 and 3.12, then builds a wheel and sdist, computes
`SHA256SUMS.txt`, and smoke-tests the built wheel (installed into a
clean venv) on the Linux runner before publishing a release.

## Limitations

- This guards two specific, verified functions (`svdvals`, `eigvalsh`);
  it does not audit every `torch.linalg` function for the same
  values-only-vs-full inconsistency pattern, though the same technique
  (check `isnan`/`isinf` before delegating) generalizes trivially to
  any other affected function if one is found.
- The guard checks the *whole* input tensor for NaN/Inf up front (an
  O(n^2) scan before an O(n^3) decomposition), so it adds negligible
  overhead but is not free; this is the same tradeoff SciPy's
  `check_finite=True` default makes.
- Batched inputs (a leading batch dimension) are checked as a whole:
  if any matrix in the batch contains NaN/Inf, the guard raises for
  the entire batched call rather than identifying which batch entry
  was the culprit. Fine-grained per-batch-entry diagnosis is not
  implemented.
- No GPU-specific behavior (CUDA/MPS numerical differences) is tested
  here; diagnosis and tests run on CPU. The underlying LAPACK
  dispatch bug is not expected to be device-specific (it's a routine-
  selection issue, not a numerical-precision one), but this has not
  been independently verified on GPU hardware.

## License

MIT
