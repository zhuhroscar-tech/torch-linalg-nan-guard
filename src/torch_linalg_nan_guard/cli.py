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
    parser.add_argument(
        "--skip-norm-check",
        action="store_true",
        help="skip the torch.linalg.vector_norm/torch.norm precision-and-overflow diagnosis (it runs a 16M-element reduction by default)",
    )
    args = parser.parse_args(argv)

    if args.version:
        from . import __version__

        print(f"torch-linalg-nan-guard {__version__}")
        return 0

    from .core import TorchUnavailableError, diagnose, diagnose_norm_precision

    try:
        report = diagnose()
        norm_report = None if args.skip_norm_check else diagnose_norm_precision()
    except TorchUnavailableError as exc:
        if args.json:
            print(json.dumps({"error": str(exc)}, indent=2))
        else:
            style = resolve_style(no_color_flag=args.no_color)
            print(status_headline(style, "fail", f"torch unavailable: {exc}"))
        return 2

    if args.json:
        combined = {"nan_swallowing": report}
        if norm_report is not None:
            combined["norm_precision"] = norm_report
        print(json.dumps(combined, indent=2))
        overall_ok = report["guard_fully_effective"] and (
            norm_report is None or norm_report["guard_fully_effective"]
        )
        return 0 if overall_ok else 1

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

    overall_ok = report["guard_fully_effective"]

    if norm_report is not None:
        section("torch.linalg.vector_norm / torch.norm precision-and-overflow check")
        print_fields(
            [
                ("tracking issues", ", ".join(norm_report["issue_urls"])),
            ]
        )
        if norm_report["any_precision_loss_present"] or norm_report["any_overflow_present"]:
            print(status_headline(style, "warn", "float32 accumulator precision/overflow bug reproduced on this host's installed torch build"))
        else:
            print(status_headline(style, "info", "float32 accumulator bug NOT reproduced on this host's installed torch build (fixed upstream)"))

        if norm_report["guard_fully_effective"]:
            print(status_headline(style, "ok", "safe_vector_norm()/safe_norm() match the float64 reference at every tested size"))
        else:
            print(status_headline(style, "fail", "guard did NOT match the float64 reference at every tested size"))

        for c in norm_report["precision_cases"]:
            print_fields(
                [
                    (
                        f"vector_norm numel={c['numel']}",
                        f"unguarded_rel_err={c['unguarded_rel_err']:.3e}  guarded_rel_err={c['guarded_rel_err']:.3e}",
                    )
                ]
            )
        oc = norm_report["overflow_case"]
        print_fields(
            [
                (
                    "vector_norm large-magnitude overflow case",
                    f"unguarded_is_inf={oc['unguarded_is_inf']!s:5s}  guarded_result={oc['guarded_result']:.6e}",
                )
            ]
        )

        overall_ok = overall_ok and norm_report["guard_fully_effective"]

    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
