"""Command-line interface: run the from-scratch diagnosis of
torch.linalg.svdvals()/eigvalsh() NaN-swallowing against the currently
installed torch build, using the shared semantic-color design system."""
from __future__ import annotations

import argparse
import json
import sys

from .style import print_fields, resolve_style, section, status_headline


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="torch-linalg-nan-guard",
        description=(
            "Diagnose whether the currently installed torch build's "
            "torch.linalg.svdvals()/eigvalsh() silently return finite-"
            "looking values for NaN/Inf input (pytorch/pytorch#187759, "
            "numpy/numpy#20280) and verify the guard functions catch it, "
            "on THIS host's actual installed torch version -- never trusts "
            "the upstream issue's reported version alone."
        ),
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of text")
    parser.add_argument("--no-color", action="store_true", help="disable ANSI color even on a TTY")
    parser.add_argument("--version", action="store_true", help="print version and exit")
    args = parser.parse_args(argv)

    if args.version:
        from . import __version__

        print(f"torch-linalg-nan-guard {__version__}")
        return 0

    from .core import TorchUnavailableError, diagnose

    try:
        report = diagnose()
    except TorchUnavailableError as exc:
        if args.json:
            print(json.dumps({"error": str(exc)}, indent=2))
        else:
            style = resolve_style(no_color_flag=args.no_color)
            print(status_headline(style, "fail", f"torch unavailable: {exc}"))
        return 2

    if args.json:
        print(json.dumps(report, indent=2))
        return 0 if report["guard_fully_effective"] else 1

    style = resolve_style(no_color_flag=args.no_color)
    print_fields(
        [
            ("torch version", report["torch_version"]),
            ("tracking issues", ", ".join(report["issue_urls"])),
        ]
    )

    if report["any_bug_present"]:
        print(status_headline(style, "warn", "bug reproduced on this host's installed torch build"))
    else:
        print(
            status_headline(
                style,
                "info",
                "bug NOT reproduced on this host's installed torch build (fixed upstream)",
            )
        )

    if report["guard_fully_effective"]:
        print(status_headline(style, "ok", "safe_svdvals()/safe_eigvalsh() catch every tested NaN case"))
    else:
        print(status_headline(style, "fail", "guard did NOT catch at least one NaN case"))

    section("per-case results (matrix size x NaN diagonal position)")
    for c in report["cases"]:
        if c["buggy_result_is_finite"]:
            flag = "SILENT-BUG"
        elif c["buggy_raised"]:
            flag = "raised-error"
        else:
            flag = "propagates-nan"
        guard_flag = "guard-ok" if c["guard_raises"] else "GUARD-FAILED"
        print_fields(
            [
                (
                    f"{c['op']} n={c['size']} pos={c['nan_position']}",
                    f"{flag:15s}  ref_nan={c['reference_has_nan']!s:5s}  ref_raised={c['reference_raised']!s:5s}  {guard_flag}",
                )
            ]
        )

    return 0 if report["guard_fully_effective"] else 1


if __name__ == "__main__":
    sys.exit(main())
