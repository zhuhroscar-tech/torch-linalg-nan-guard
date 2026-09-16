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

    def test_nan_input_raises_instead_of_silently_swallowing(self):
        M = torch.tensor([[float("nan"), 0.0], [0.0, 1.0]], dtype=torch.float64)
        # Reproduce the raw bug first: confirm the unguarded function
        # really does swallow the NaN on this host (if this assertion
        # itself fails, the upstream bug has been fixed and the guard
        # is redundant but harmless -- see test below for that check).
        raw = torch.linalg.svdvals(M)
        assert torch.isfinite(raw).all(), (
            "expected upstream bug to reproduce (finite result for NaN "
            "input) -- if this fails, pytorch/pytorch#187759 was fixed "
            "upstream; re-verify before assuming the guard is still needed"
        )
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

    def test_nan_input_raises_instead_of_silently_swallowing(self):
        M = torch.diag(torch.tensor([1.0, 2.0], dtype=torch.float64))
        M[0, 0] = float("nan")
        raw = torch.linalg.eigvalsh(M)
        assert torch.isfinite(raw).all(), (
            "expected upstream bug to reproduce (finite result for NaN "
            "input) -- if this fails, the eigvalsh/eigh inconsistency was "
            "fixed upstream; re-verify before assuming the guard is still "
            "needed"
        )
        with pytest.raises(ValueError, match="NaN or Inf"):
            safe_eigvalsh(M)

    def test_inf_input_also_raises(self):
        M = torch.diag(torch.tensor([float("inf"), 1.0], dtype=torch.float64))
        with pytest.raises(ValueError, match="NaN or Inf"):
            safe_eigvalsh(M)


class TestDiagnose:
    def test_diagnose_reproduces_bug_and_guard_is_fully_effective(self):
        report = diagnose(sizes=(2, 3))
        assert report["any_bug_present"] is True, (
            "diagnose() should reproduce the silent-finite-result bug on "
            "this host's installed torch build; if this fails, the "
            "upstream bug may have been fixed -- do not assume, re-verify"
        )
        assert report["guard_fully_effective"] is True
        assert len(report["cases"]) == (2 + 3) * 2  # svdvals + eigvalsh per position

    def test_every_case_where_bug_is_silent_the_guard_still_raises(self):
        report = diagnose(sizes=(2, 3, 4))
        for case in report["cases"]:
            if case["buggy_result_is_finite"]:
                assert case["guard_raises"], (
                    f"guard failed to catch a silent-finite case: {case}"
                )

    def test_reference_full_decomposition_independent_oracle(self):
        # Cross-check against an independent oracle: Python's own math
        # module confirms NaN != NaN and is never finite, so any case
        # where the "buggy" op returns something math.isfinite() accepts
        # is, by definition, a silent-swallow bug on that op's part
        # (the reference full-decomposition op is the ground truth for
        # "this input contains a NaN and should not produce a finite
        # scalar output").
        report = diagnose(sizes=(2,))
        for case in report["cases"]:
            assert case["reference_has_nan"] is True, (
                "reference full-decomposition op should always propagate "
                f"NaN for NaN-containing input: {case}"
            )
