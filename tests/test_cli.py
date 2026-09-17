"""Tests for the CLI entry point: argument parsing, --version, --json,
--no-color, and exit codes -- independent of whether torch is installed.
Mirrors the test_cli.py pattern already used across the fleet (e.g.
rng-leak-audit, causality-audit) for a repo that had none.

The branch-coverage tests below mock ``core.diagnose`` so every CLI
message path (torch-unavailable, no-bug "info" line, and a
guard-mismatch "fail" line) is exercised deterministically, regardless
of whether this host's installed torch build happens to reproduce the
underlying bug. Real fleet-wide gap found by pytest-cov inspection:
cli.py sat at 81% coverage with the negative branches of each status
line (lines 40-46, 63, 74) and the ``__main__`` guard (line 98) never
hit by any existing test."""
from __future__ import annotations

import json
import runpy
import sys

import pytest

from torch_linalg_nan_guard import core
from torch_linalg_nan_guard.cli import main


def _fake_report(**overrides):
    report = {
        "torch_version": "9.9.9-fake",
        "issue_urls": ["https://github.com/pytorch/pytorch/issues/187759"],
        "cases": [],
        "any_bug_present": False,
        "guard_fully_effective": True,
    }
    report.update(overrides)
    return report


def test_version_flag(capsys):
    code = main(["--version"])
    out = capsys.readouterr().out
    assert code == 0
    assert "torch-linalg-nan-guard" in out


def test_json_output_is_valid_json_and_reports_guard_status(capsys):
    torch = pytest.importorskip("torch")
    code = main(["--json"])
    out = capsys.readouterr().out
    report = json.loads(out)
    assert "torch_version" in report
    assert report["torch_version"] == torch.__version__
    assert "guard_fully_effective" in report
    assert code in (0, 1)


def test_json_exit_code_matches_guard_fully_effective(capsys):
    pytest.importorskip("torch")
    code = main(["--json"])
    out = capsys.readouterr().out
    report = json.loads(out)
    assert code == (0 if report["guard_fully_effective"] else 1)


def test_text_output_no_color_has_no_ansi_escapes(capsys):
    pytest.importorskip("torch")
    main(["--no-color"])
    out = capsys.readouterr().out
    assert "\x1b[" not in out


def test_text_output_reports_per_case_results(capsys):
    pytest.importorskip("torch")
    main(["--no-color"])
    out = capsys.readouterr().out
    assert "torch version" in out
    assert "per-case results" in out


def test_torch_unavailable_json_mode_reports_error_and_exit_2(monkeypatch, capsys):
    """cli.py lines 41-42: TorchUnavailableError + --json emits a JSON
    error object and exits 2, regardless of torch install state."""

    def _raise(*args, **kwargs):
        raise core.TorchUnavailableError("torch is required for diagnosis")

    monkeypatch.setattr(core, "diagnose", _raise)
    code = main(["--json"])
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload == {"error": "torch is required for diagnosis"}
    assert code == 2


def test_torch_unavailable_text_mode_reports_fail_headline_and_exit_2(monkeypatch, capsys):
    """cli.py lines 43-46: TorchUnavailableError in text mode prints a
    'fail' status headline (not the JSON branch) and exits 2."""

    def _raise(*args, **kwargs):
        raise core.TorchUnavailableError("torch is required for diagnosis")

    monkeypatch.setattr(core, "diagnose", _raise)
    code = main(["--no-color"])
    out = capsys.readouterr().out
    assert "torch unavailable: torch is required for diagnosis" in out
    assert "[X]" in out
    assert code == 2


def test_no_bug_present_prints_info_line(monkeypatch, capsys):
    """cli.py line 63: the 'info' (not 'warn') branch when the host's
    torch build does NOT reproduce the NaN-swallowing bug."""
    monkeypatch.setattr(core, "diagnose", lambda: _fake_report())
    main(["--no-color"])
    out = capsys.readouterr().out
    assert "bug NOT reproduced on this host's installed torch build (fixed upstream)" in out
    assert "bug reproduced on this host's installed torch build" not in out


def test_bug_present_prints_warn_line(monkeypatch, capsys):
    """Positive-branch companion: when the bug IS reproduced, the
    'warn' headline fires instead (line 61)."""
    monkeypatch.setattr(core, "diagnose", lambda: _fake_report(any_bug_present=True))
    main(["--no-color"])
    out = capsys.readouterr().out
    assert "bug reproduced on this host's installed torch build" in out


def test_guard_mismatch_prints_fail_line_and_exit_1(monkeypatch, capsys):
    """cli.py line 74 + 94: guard_fully_effective=False prints the
    'fail' headline (not the 'ok' one) and the process exits 1."""
    monkeypatch.setattr(core, "diagnose", lambda: _fake_report(guard_fully_effective=False))
    code = main(["--no-color"])
    out = capsys.readouterr().out
    assert "guard did NOT catch at least one NaN case" in out
    assert "catch every tested NaN case" not in out
    assert code == 1


def test_module_entry_point_runs_main_and_exits_with_its_code(monkeypatch):
    """cli.py line 98 (``if __name__ == "__main__": sys.exit(main())``):
    running the module as a script must invoke main() and propagate its
    return code via SystemExit, not just be dead code."""
    monkeypatch.setattr(core, "diagnose", lambda: _fake_report(guard_fully_effective=False))
    monkeypatch.setattr(sys, "argv", ["torch-linalg-nan-guard", "--no-color"])
    monkeypatch.delitem(sys.modules, "torch_linalg_nan_guard.cli", raising=False)
    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("torch_linalg_nan_guard.cli", run_name="__main__")
    assert exc_info.value.code == 1
