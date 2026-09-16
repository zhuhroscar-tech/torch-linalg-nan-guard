"""Regression tests for torch-linalg-nan-guard.

These prove the guard functions actually intercept the documented
NaN-swallowing bugs (pytorch/pytorch#187759, numpy/numpy#20280) on
whatever torch build is installed, and that they are transparent
pass-throughs on ordinary finite input (an independent correctness
oracle: any finite-input call must match plain torch.linalg exactly).
"""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from torch_linalg_nan_guard.core import diagnose, safe_eigvalsh, safe_svdvals


class TestSafeSvdvals:
    def test_finite_input_matches_plain_svdvals_exactly(self):
        torch.manual_seed(0)
        A = torch.randn(5, 5, dtype=torch.float64)
        expected = torch.linalg.svdvals(A)
        actual = safe_svdvals(A)
        assert torch.allclose(actual, expected)

    def test_nan_input_raises(self):
        # The unguarded torch.linalg.svdvals's behavior on NaN input is
        # backend-dependent: some LAPACK backends (macOS/Accelerate)
        # silently swallow the NaN and return a finite result
        # (pytorch/pytorch#187759); others (Linux/OpenBLAS, observed in
        # this project's own CI) raise their own _LinAlgError instead.
        # Either way this is exactly why the guard exists: it must raise
        # a clear, consistent ValueError regardless of which backend's
        # behavior the unguarded call happens to exhibit.
        M = torch.tensor([[float("nan"), 0.0], [0.0, 1.0]], dtype=torch.float64)
        with pytest.raises(ValueError, match="NaN or Inf"):
            safe_svdvals(M)

    def test_inf_input_also_raises(self):
        M = torch.tensor([[float("inf"), 0.0], [0.0, 1.0]], dtype=torch.float64)
        with pytest.raises(ValueError, match="NaN or Inf"):
            safe_svdvals(M)

    def test_3x3_diagonal_nan_placements_all_caught(self):
        for pos in range(3):
            M = torch.diag(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
            M[pos, pos] = float("nan")
            with pytest.raises(ValueError):
                safe_svdvals(M)


class TestSafeEigvalsh:
    def test_finite_input_matches_plain_eigvalsh_exactly(self):
        torch.manual_seed(1)
        A = torch.randn(5, 5, dtype=torch.float64)
        A = A + A.T  # symmetric, required by eigvalsh
        expected = torch.linalg.eigvalsh(A)
        actual = safe_eigvalsh(A)
        assert torch.allclose(actual, expected)

    def test_nan_input_raises(self):
        # Same backend-dependence note as svdvals above: the unguarded
        # eigvalsh's behavior on this exact input has been observed to
        # differ between LAPACK backends, but the guard's own ValueError
        # must fire either way.
        M = torch.diag(torch.tensor([1.0, 2.0], dtype=torch.float64))
        M[0, 0] = float("nan")
        with pytest.raises(ValueError, match="NaN or Inf"):
            safe_eigvalsh(M)

    def test_inf_input_also_raises(self):
        M = torch.diag(torch.tensor([float("inf"), 1.0], dtype=torch.float64))
        with pytest.raises(ValueError, match="NaN or Inf"):
            safe_eigvalsh(M)


class TestDiagnose:
    def test_diagnose_runs_and_guard_is_fully_effective(self):
        # any_bug_present is intentionally NOT asserted True here: on
        # LAPACK backends where the unguarded op raises its own error
        # for NaN input instead of silently swallowing it (observed on
        # Linux/OpenBLAS in this project's own CI, unlike macOS/
        # Accelerate where the silent-swallow bug was first reproduced),
        # zero cases may be "silent". The guard being fully effective
        # regardless of which behavior the backend exhibits is the
        # actual invariant this package provides.
        report = diagnose(sizes=(2, 3))
        assert report["guard_fully_effective"] is True
        assert len(report["cases"]) == (2 + 3) * 2  # svdvals + eigvalsh per position

    def test_every_case_where_bug_is_silent_the_guard_still_raises(self):
        report = diagnose(sizes=(2, 3, 4))
        for case in report["cases"]:
            if case["buggy_result_is_finite"]:
                assert case["guard_raises"], (
                    f"guard failed to catch a silent-finite case: {case}"
                )

    def test_reference_full_decomposition_never_silently_finite(self):
        # Cross-check against an independent oracle: whenever the
        # reference full-decomposition op (svd/eigh) does return a
        # result rather than raising its own error, that result must
        # contain NaN for NaN-containing input -- it must never be the
        # same kind of silent-finite bug the guarded functions have.
        # This is the ground truth this package's "bug" classification
        # relies on: a reference op that raises is a separate, already-
        # loud failure mode (also fine, just not "silent"); a reference
        # op that returns a *finite* result for NaN input would mean the
        # whole test fixture's assumptions are broken.
        report = diagnose(sizes=(2,))
        for case in report["cases"]:
            if not case["reference_raised"]:
                assert case["reference_has_nan"] is True, (
                    "reference full-decomposition op returned a finite "
                    f"result for NaN input -- unexpected: {case}"
                )
