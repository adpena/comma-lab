#!/usr/bin/env python3
"""ddm_cd1 - price a corrector port against a MEASURED token-stage decomposition.

The question this answers, and the one it refuses to answer.

``ddm_rr7`` retired the full native token-decode port on the shipping axis: byte-perfect
and 15.3% SLOWER on T4.  The float64 ddm_rr2 corrector is the only decode-wall candidate
left.  ``ddm_rr6`` 6 prices it at 59.2% of a 326.2 s split run -- but that is an M5 Max
number on a DIFFERENT decoder configuration, and transferring it is exactly the
cross-regime error rr7 measured at 2.02x.  So this tool takes a breakdown MEASURED on the
axis being decided and does the only arithmetic that is honest there.

THE CEILING IS SUBTRACTION, NOT A RATIO.  A port replaces the corrector; it cannot make
the rest of the loop faster.  So

    T_token(k)  =  T_token_measured  -  T_port_scope * (1 - 1/k)

for a port that is ``k`` times faster on its own scope, and the k -> infinity floor is
``T_token_measured - T_port_scope``.  If THAT floor is still outside the CI window, the
port is CLOSED BY ARITHMETIC -- no build, no benchmark, no argument.  A ceiling is worth
more than an estimate precisely because it cannot be beaten by optimism.

WHAT IT WILL NOT DO.  It never converts a local ratio into a shipping one, it never treats
the family sums as separable from the axis they were measured on, and it prices the port
against ``port_scope_seconds`` only -- ``group_state + coding_row + observe``.  Folding in
the probability table, the digests, the RC64 ctypes calls or the transfers is the error
``ddm_rr6`` 6 named when it refused to quote 62.8%.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from tac.contest_budget import (  # noqa: E402
    CONTEST_CUDA,
    evaluate_budget,
    normalize_axis,
    residual_window,
)

INFLATE_REPORT_SCHEMA = "ddm_f26p_inflate_report.v1"
#: ``inflate.py`` prints ``json.dumps(report, sort_keys=True)``, so the report always opens
#: on its alphabetically-first key.  That is what makes it findable inside a captured
#: stdout string without guessing at brace balance.
REPORT_OPENER = '{"archive_bytes":'

#: Default speedups to table.  ``inf`` is the load-bearing row: it is the CEILING, and it
#: is the only row a build cannot argue with.
DEFAULT_SPEEDUPS = (2.0, 3.0, 5.0, 10.0, math.inf)


class CeilingError(RuntimeError):
    """A receipt did not contain the measurement this arithmetic requires."""


def _iter_inflate_reports(node: Any):
    """Yield every inflate report reachable from a parsed receipt.

    Two shapes occur in practice: the report as a nested dict (a local receipt) and the
    report as text inside a captured stdout string (every Modal row).  Both are searched
    because a tool that handled only one would silently return nothing for the other --
    and a silent nothing reads exactly like "no measurement", which is the vacuity genus.
    """
    if isinstance(node, dict):
        if node.get("schema") == INFLATE_REPORT_SCHEMA:
            yield node
        for value in node.values():
            yield from _iter_inflate_reports(value)
    elif isinstance(node, list):
        for value in node:
            yield from _iter_inflate_reports(value)
    elif isinstance(node, str):
        decoder = json.JSONDecoder()
        start = node.find(REPORT_OPENER)
        while start != -1:
            try:
                candidate, _ = decoder.raw_decode(node, start)
            except ValueError:
                candidate = None
            if isinstance(candidate, dict) and candidate.get("schema") == INFLATE_REPORT_SCHEMA:
                yield candidate
            start = node.find(REPORT_OPENER, start + 1)


def load_inflate_report(path: Path) -> dict[str, Any]:
    """Return the single inflate report in ``path``, refusing on 0 or >1 distinct ones."""
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        parsed: Any = json.loads(text)
    except ValueError:
        parsed = text  # a raw log; the string branch still finds the report
    found: list[dict[str, Any]] = []
    seen: set[str] = set()
    for report in _iter_inflate_reports(parsed):
        key = json.dumps(report, sort_keys=True)
        if key not in seen:
            seen.add(key)
            found.append(report)
    if not found:
        raise CeilingError(f"no {INFLATE_REPORT_SCHEMA} in {path}")
    if len(found) > 1:
        raise CeilingError(
            f"{len(found)} DISTINCT inflate reports in {path}; a decomposition must name "
            "which run it prices, so this refuses rather than picking one"
        )
    return found[0]


def _require(mapping: dict[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise CeilingError(f"{where} lacks {key!r}")
    return mapping[key]


def price(
    report: dict[str, Any],
    *,
    axis: str = CONTEST_CUDA,
    baseline_token_seconds: float | None = None,
    speedups: tuple[float, ...] = DEFAULT_SPEEDUPS,
    evaluate_seconds: float | None = None,
    inflate_elapsed_seconds: float | None = None,
) -> dict[str, Any]:
    """Turn one measured breakdown into a ceiling, a break-even, and graded rows."""
    stages = _require(report, "stage_seconds", "inflate report")
    token_seconds = float(_require(stages, "token_decode_or_checkpoint_load", "stage_seconds"))
    # The report's own total EXCLUDES the process the harness timed around it: interpreter
    # start, the torch import and the archive verification.  jg5 MEASURED 1415.024 s inside
    # and 1419.904 s outside -- 4.880 s the budget window is graded against and the report
    # cannot see.  A port does not touch that either, so when the harness figure is supplied
    # it is authoritative and the difference lands in non_token_inflate where it belongs.
    internal_total = float(_require(stages, "total_including_raw_sha256", "stage_seconds"))
    inflate_seconds = (
        internal_total if inflate_elapsed_seconds is None else float(inflate_elapsed_seconds)
    )
    decoder = _require(report, "token_decoder", "inflate report")
    breakdown = decoder.get("token_stage_breakdown")
    if not isinstance(breakdown, dict):
        raise CeilingError(
            "this receipt carries no token_stage_breakdown -- it came from an "
            "UNINSTRUMENTED tree, and the split cannot be recovered from the total"
        )
    port_scope = float(_require(breakdown, "port_scope_seconds", "token_stage_breakdown"))
    families = dict(_require(breakdown, "family_seconds", "token_stage_breakdown"))

    # Everything outside the token stage is untouched by a token-stage port.
    non_token_inflate = inflate_seconds - token_seconds
    window = residual_window(normalize_axis(axis))

    # FRAME B, DERIVED rather than quoted.  The canonical window already netted out an
    # ESTIMATED evaluate, so charging a MEASURED one against it double-charges by that
    # estimate.  Adding the estimate back and removing the measurement recovers the frame the
    # #1111 packet publishes.  ``ddm_rr7`` 2 gives [890.6, 1430.6] for jg5's 51.428 s
    # evaluate; this reproduces it from ``window.ci_steps`` so the number is never typed.
    estimate = next(
        (s for s in window.ci_steps if s.step.startswith("evaluate_py_600_pairs")), None
    )
    frame_b: dict[str, Any] | None = None
    if estimate is not None and evaluate_seconds is not None:
        frame_b = {
            "narrow_end_seconds": window.narrow_end_seconds
            + (estimate.typical_seconds - evaluate_seconds),
            "wide_end_seconds": window.wide_end_seconds
            + (estimate.worst_seconds - evaluate_seconds),
            "derivation": (
                f"[{window.narrow_end_seconds}, {window.wide_end_seconds}] + "
                f"(evaluate_estimate {estimate.typical_seconds:.0f}..{estimate.worst_seconds:.0f} s "
                f"- evaluate_measured {evaluate_seconds:.3f} s)"
            ),
            "graded_against": "inflate ALONE (the measured evaluate is already netted out)",
            "note": (
                "Frames A and B are ONE measurement seen twice, related by exactly this "
                "correction. Quoting either without the other hides the correction, not a "
                "disagreement -- and B is the friendlier of the two."
            ),
        }

    rows = []
    for speedup in speedups:
        remaining = 0.0 if math.isinf(speedup) else port_scope / speedup
        token_after = token_seconds - port_scope + remaining
        inflate_after = token_after + non_token_inflate
        verdict = evaluate_budget(axis, inflate_after, evaluate_seconds)
        rows.append(
            {
                "port_speedup": "inf" if math.isinf(speedup) else speedup,
                "token_seconds": token_after,
                "inflate_seconds": inflate_after,
                "charged_seconds": verdict.charged_seconds,
                "frame_a_verdict": verdict.verdict,
                "frame_a_margin_vs_narrow_end_seconds": verdict.margin_vs_narrow_end_seconds,
                "frame_a_margin_vs_wide_end_seconds": verdict.margin_vs_wide_end_seconds,
                "frame_b_margin_vs_narrow_end_seconds": (
                    None if frame_b is None else frame_b["narrow_end_seconds"] - inflate_after
                ),
                "frame_b_margin_vs_wide_end_seconds": (
                    None if frame_b is None else frame_b["wide_end_seconds"] - inflate_after
                ),
            }
        )

    floor_inflate = inflate_seconds - port_scope
    # Break-even: the speedup at which inflate ALONE reaches a target.  It exists only when
    # the k -> inf floor already clears that target, so an unreachable one returns None
    # rather than a large finite number that reads as hard-but-possible.
    def break_even(target: float) -> float | None:
        need = inflate_seconds - target  # seconds the port must remove
        if need <= 0.0:
            return 1.0  # already there without the port
        if need >= port_scope:
            return None  # unreachable: the whole scope is not enough
        return port_scope / (port_scope - need)

    return {
        "schema": "ddm_cd1_corrector_ceiling.v1",
        "axis": normalize_axis(axis),
        "measured": {
            "token_stage_seconds": token_seconds,
            "inflate_seconds": inflate_seconds,
            "inflate_seconds_source": (
                "inflate report total_including_raw_sha256 (harness figure NOT supplied; "
                "this UNDERSTATES the graded inflate by the process/import prelude)"
                if inflate_elapsed_seconds is None
                else "harness inflate_elapsed_seconds (authoritative for budget grading)"
            ),
            "inflate_report_internal_total_seconds": internal_total,
            "harness_minus_report_seconds": inflate_seconds - internal_total,
            "non_token_inflate_seconds": non_token_inflate,
            "evaluate_seconds": evaluate_seconds,
            "family_seconds": families,
            "port_scope_seconds": port_scope,
            "port_scope_share_of_token_stage": port_scope / token_seconds,
            "unattributed_in_loop_seconds": breakdown.get("unattributed_in_loop_seconds"),
            "prelude_seconds": breakdown.get("prelude_seconds"),
        },
        "instrumentation_overhead": (
            None
            if baseline_token_seconds is None
            else {
                "baseline_token_stage_seconds": float(baseline_token_seconds),
                "excess_seconds": token_seconds - float(baseline_token_seconds),
                "excess_fraction": token_seconds / float(baseline_token_seconds) - 1.0,
                "note": (
                    "UPPER BOUND on the timing calls: it also carries run-to-run noise, "
                    "which on a shared cloud runner is not small. It is never subtracted "
                    "from the families -- an overhead bound is not an overhead measurement."
                ),
            }
        ),
        "ceiling": {
            "token_seconds_at_infinite_speedup": token_seconds - port_scope,
            "inflate_seconds_at_infinite_speedup": floor_inflate,
            "seconds_a_perfect_port_can_remove": port_scope,
            "note": (
                "A port replaces the corrector; it cannot speed up the rest of the loop. "
                "If this floor is outside the window, the port is CLOSED BY ARITHMETIC."
            ),
        },
        "frame_b_window": frame_b,
        "break_even_port_speedup": {
            "frame_a_narrow_end": break_even(window.narrow_end_seconds - (evaluate_seconds or 0.0)),
            "frame_a_wide_end": break_even(window.wide_end_seconds - (evaluate_seconds or 0.0)),
            "frame_b_narrow_end": (
                None if frame_b is None else break_even(frame_b["narrow_end_seconds"])
            ),
            "frame_b_wide_end": (
                None if frame_b is None else break_even(frame_b["wide_end_seconds"])
            ),
            "frame_a_window": [window.narrow_end_seconds, window.wide_end_seconds],
            "note": (
                "null means UNREACHABLE: even an infinitely fast corrector leaves inflate "
                "above that end. 1.0 means the target is already met without the port. The "
                "frame-A targets subtract the measured evaluate because frame A charges "
                "inflate+evaluate while the port only shortens inflate."
            ),
        },
        "rows": rows,
        "is_score_claim": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path, help="a receipt containing the inflate report")
    parser.add_argument("--axis", default=CONTEST_CUDA)
    parser.add_argument(
        "--baseline-token-seconds",
        type=float,
        default=None,
        help="the UNINSTRUMENTED token stage on the same axis; bounds the timing overhead",
    )
    parser.add_argument(
        "--evaluate-seconds",
        type=float,
        default=None,
        help="measured evaluate wall clock; omitted makes the charge a lower bound",
    )
    parser.add_argument(
        "--inflate-elapsed-seconds",
        type=float,
        default=None,
        help=(
            "the HARNESS inflate wall clock the budget window is graded against; it exceeds "
            "the report's internal total by the interpreter/import prelude (jg5: 4.880 s)"
        ),
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        report = load_inflate_report(args.receipt)
        priced = price(
            report,
            axis=args.axis,
            baseline_token_seconds=args.baseline_token_seconds,
            evaluate_seconds=args.evaluate_seconds,
            inflate_elapsed_seconds=args.inflate_elapsed_seconds,
        )
    except CeilingError as error:
        print(f"REFUSE: {error}", file=sys.stderr)
        return 2
    text = json.dumps(priced, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
