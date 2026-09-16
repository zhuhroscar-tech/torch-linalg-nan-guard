"""torch-linalg-nan-guard core: detect and safely guard against
``torch.linalg.svdvals()`` and ``torch.linalg.eigvalsh()`` silently
returning plausible-looking *finite* values for input that contains
NaN or Inf, instead of propagating NaN or raising like their sibling
full-decomposition functions (``torch.linalg.svd()``, ``torch.linalg.eigh()``)
mostly do.

Reproduced from scratch on this host (see README for the exact commands
and the live-source verification trail, per this project's
evidence-before-acceptance policy):

  - ``torch.linalg.svdvals(torch.tensor([[nan, 0], [0, 1]]))`` returns
    ``[1., 1.]`` -- the NaN is silently dropped -- while
    ``torch.linalg.svd(same_input).S`` returns ``[1., nan]`` on the
    same input (pytorch/pytorch#187759, open, root-caused to LAPACK's
    ``dgesdd`` with ``jobz='N'`` not propagating NaN the way the
    ``jobz='S'``/``'A'`` paths happen to).
  - ``torch.linalg.eigvalsh`` on a symmetric matrix with a NaN on the
    diagonal returns small *finite* numbers (e.g. ``[0., -0.]`` for a
    2x2 case) that look like a legitimate near-zero-eigenvalue result,
    while ``torch.linalg.eigh`` on the identical input propagates NaN
    into (usually) one eigenvalue slot. This is the exact inconsistency
    numpy/numpy#20280 documents for NumPy's ``eigvalsh``/``eigh`` (open
    since 2021); confirmed here to reproduce identically through
    PyTorch's own ``eigvalsh``/``eigh``, which share the same
    values-only-vs-full LAPACK dispatch pattern as the SVD case.

Neither bug crashes or warns. A NaN anywhere in an embedding matrix,
covariance matrix, or Gram matrix -- entirely plausible after a
divide-by-zero, an unmasked padding row, or a numerically unstable
upstream op -- silently produces a *finite*, superficially reasonable
singular value or eigenvalue instead of an error, so downstream code
(``matrix_rank``, spectral normalization, condition-number checks,
PCA/whitening) keeps running on corrupted results with no signal that
anything went wrong.

This module's guard functions apply the same policy SciPy already uses
by default (``check_finite=True``): refuse to compute a plausible-
looking answer from non-finite input at all, and raise a clear,
immediate ``ValueError`` instead -- rather than either silently
dropping the NaN (the bug) or relying on the inconsistent partial
NaN-propagation the "full" decomposition happens to exhibit.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, List


class TorchUnavailableError(RuntimeError):
    """Raised when torch cannot be imported. Kept as a distinct type so
    callers can distinguish "torch isn't installed" from an actual
    diagnostic failure."""


def _import_torch():
    try:
        import torch  # noqa: F401
    except Exception as exc:  # pragma: no cover - exercised only without torch
        raise TorchUnavailableError(
            "torch is required for diagnosis and guarding; install the "
            "'torch' extra."
        ) from exc
    return torch


def _contains_nonfinite(torch_module, A) -> bool:
    return bool(torch_module.isnan(A).any().item() or torch_module.isinf(A).any().item())


def safe_svdvals(A, *, driver=None):
    """Drop-in guard for ``torch.linalg.svdvals``: raises ``ValueError``
    immediately if ``A`` contains any NaN or Inf, instead of silently
    returning finite-looking singular values computed from undefined
    input (pytorch/pytorch#187759). Delegates to the real
    ``torch.linalg.svdvals`` unchanged for genuinely finite input, so
    this is a safe drop-in replacement with identical behavior on valid
    data."""
    torch_module = _import_torch()
    if _contains_nonfinite(torch_module, A):
        raise ValueError(
            "torch_linalg_nan_guard.safe_svdvals: input contains NaN or Inf; "
            "torch.linalg.svdvals() silently returns finite-looking but "
            "meaningless singular values for such input (pytorch/pytorch#187759) "
            "-- refusing to compute rather than returning a wrong answer."
        )
    return torch_module.linalg.svdvals(A, driver=driver)


def safe_eigvalsh(A, *, UPLO="L"):
    """Drop-in guard for ``torch.linalg.eigvalsh``: raises ``ValueError``
    immediately if ``A`` contains any NaN or Inf, instead of silently
    returning finite-looking eigenvalues computed from undefined input
    (the same values-only-vs-full inconsistency numpy/numpy#20280
    documents for NumPy's identical function pair). Delegates to the
    real ``torch.linalg.eigvalsh`` unchanged for genuinely finite
    input."""
    torch_module = _import_torch()
    if _contains_nonfinite(torch_module, A):
        raise ValueError(
            "torch_linalg_nan_guard.safe_eigvalsh: input contains NaN or Inf; "
            "torch.linalg.eigvalsh() silently returns finite-looking but "
            "meaningless eigenvalues for such input (mirrors numpy/numpy#20280) "
            "-- refusing to compute rather than returning a wrong answer."
        )
    return torch_module.linalg.eigvalsh(A, UPLO=UPLO)


@dataclasses.dataclass
class NanPlacementCase:
    op: str            # "svdvals" or "eigvalsh"
    size: int
    nan_position: int  # diagonal index where NaN was placed
    buggy_raised: bool          # the unguarded op itself raised (e.g. a
                                 # LAPACK convergence error) rather than
                                 # silently returning a finite result --
                                 # this is a DIFFERENT, non-silent failure
                                 # mode and is not counted as the bug
    buggy_result_is_finite: bool
    buggy_result: List[float]
    reference_raised: bool      # the sibling full-decomposition op also
                                 # raised instead of returning a result
                                 # (seen on some LAPACK backends, e.g.
                                 # Linux/OpenBLAS is stricter than macOS/
                                 # Accelerate about non-finite input)
    reference_result: List[float]  # empty if reference_raised
    reference_has_nan: bool         # False (not True) if reference_raised
    guard_raises: bool


def _run_maybe_raising(fn) -> tuple:
    """Run an op that may itself raise for non-finite input (this varies
    by LAPACK backend -- observed to differ between macOS/Accelerate and
    Linux/OpenBLAS for the exact same torch version and code). Returns
    (raised, is_finite_or_False, values_or_empty)."""
    try:
        result = fn()
    except Exception:
        return True, False, []
    return False, bool(result.isfinite().all().item()), result.tolist()


def _svdvals_case(torch_module, size: int, pos: int) -> NanPlacementCase:
    M = torch_module.diag(torch_module.arange(1, size + 1, dtype=torch_module.float64))
    M[pos, pos] = float("nan")

    buggy_raised, buggy_finite, buggy_values = _run_maybe_raising(lambda: torch_module.linalg.svdvals(M))
    ref_raised, _, ref_values = _run_maybe_raising(lambda: torch_module.linalg.svd(M).S)
    ref_has_nan = False
    if not ref_raised:
        ref_has_nan = bool(torch_module.isnan(torch_module.tensor(ref_values)).any().item())

    guard_raised = False
    try:
        safe_svdvals(M)
    except ValueError:
        guard_raised = True

    return NanPlacementCase(
        op="svdvals",
        size=size,
        nan_position=pos,
        buggy_raised=buggy_raised,
        buggy_result_is_finite=buggy_finite,
        buggy_result=buggy_values,
        reference_raised=ref_raised,
        reference_result=ref_values,
        reference_has_nan=ref_has_nan,
        guard_raises=guard_raised,
    )


def _eigvalsh_case(torch_module, size: int, pos: int) -> NanPlacementCase:
    M = torch_module.diag(torch_module.arange(1, size + 1, dtype=torch_module.float64))
    M[pos, pos] = float("nan")

    buggy_raised, buggy_finite, buggy_values = _run_maybe_raising(lambda: torch_module.linalg.eigvalsh(M))
    ref_raised, _, ref_values = _run_maybe_raising(lambda: torch_module.linalg.eigh(M)[0])
    ref_has_nan = False
    if not ref_raised:
        ref_has_nan = bool(torch_module.isnan(torch_module.tensor(ref_values)).any().item())

    guard_raised = False
    try:
        safe_eigvalsh(M)
    except ValueError:
        guard_raised = True

    return NanPlacementCase(
        op="eigvalsh",
        size=size,
        nan_position=pos,
        buggy_raised=buggy_raised,
        buggy_result_is_finite=buggy_finite,
        buggy_result=buggy_values,
        reference_raised=ref_raised,
        reference_result=ref_values,
        reference_has_nan=ref_has_nan,
        guard_raises=guard_raised,
    )


def diagnose(sizes=(2, 3, 4)) -> Dict[str, Any]:
    """Reproduce both bugs from scratch against the currently installed
    torch build, at several matrix sizes and NaN diagonal positions.
    Never trusts a cached/prior result -- every call re-runs the actual
    repro."""
    torch_module = _import_torch()

    cases: List[NanPlacementCase] = []
    for size in sizes:
        for pos in range(size):
            cases.append(_svdvals_case(torch_module, size, pos))
            cases.append(_eigvalsh_case(torch_module, size, pos))

    # "The bug" is specifically the SILENT case: the unguarded op
    # returns a finite, plausible-looking result for NaN/Inf input with
    # no error at all. A case where the unguarded op raises its own
    # error (e.g. a LAPACK convergence failure) is a different, already-
    # loud failure mode -- not silent corruption -- and is not counted
    # as reproducing this bug, though the guard is still expected to
    # raise a clear diagnostic in all cases (silent or not).
    any_bug_present = any(c.buggy_result_is_finite for c in cases)
    guard_fully_effective = all(c.guard_raises for c in cases)

    return {
        "torch_version": torch_module.__version__,
        "issue_urls": [
            "https://github.com/pytorch/pytorch/issues/187759",
            "https://github.com/numpy/numpy/issues/20280",
        ],
        "cases": [dataclasses.asdict(c) for c in cases],
        "any_bug_present": any_bug_present,
        "guard_fully_effective": guard_fully_effective,
    }
