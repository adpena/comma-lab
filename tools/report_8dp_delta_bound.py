#!/usr/bin/env python3
"""Compose the report-8dp bound sentence for a DELTA between two measured rows.

Use this instead of typing a bound into a seal, a memo, or a falsifier. Three
landed defects came from typing it (rv13 F2/F3/F9, round-12 F1); zero came from
computing it.

    .venv/bin/python tools/report_8dp_delta_bound.py \\
        --base /Volumes/APDataStore/pact/ddm_ck2/t4_row_r2/MODAL_REMOTE_RESULT.json \\
        --candidate /Volumes/APDataStore/pact/ddm_to1/t4_row_r1/MODAL_REMOTE_RESULT.json \\
        --net-ds -6.991519007781832e-05

Both receipts may be the Modal summary wrapper or the inner contest_auth_eval
result; the nested artifact is opened for you. There is no flag to supply a
bound by hand, and there will not be one.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from tac.report_8dp_bounds import BoundContractError, delta_bound  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # Same widening as make_candidate_seal.py: a net dS is a small NEGATIVE
    # number in scientific notation, which argparse's default matcher reads as
    # an unknown option.
    ap._negative_number_matcher = re.compile(r"^-\d+$|^-\d*\.\d+$|^-\d*\.?\d+[eE][+-]?\d+$")
    ap.add_argument("--base", required=True, help="the BASE row's auth-eval receipt")
    ap.add_argument("--candidate", required=True, help="the CANDIDATE row's auth-eval receipt")
    ap.add_argument("--net-ds", type=float, default=None, help="measured net dS, to state the multiple")
    ap.add_argument("--base-label", default="base")
    ap.add_argument("--candidate-label", default="candidate")
    ap.add_argument("--json", action="store_true", help="emit a machine-readable block")
    return ap


def _load(path_str: str) -> dict:
    path = Path(path_str)
    if not path.is_file():
        raise SystemExit(f"FATAL: not a file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"FATAL: {path} is not readable JSON: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        bound = delta_bound(
            _load(args.base),
            _load(args.candidate),
            base_label=args.base_label,
            candidate_label=args.candidate_label,
        )
    except BoundContractError as exc:
        print(f"FATAL: cannot compute the bound: {exc}", file=sys.stderr)
        return 3

    if args.json:
        payload = {
            "schema": "report_8dp_delta_bound.v1",
            "total": bound.total,
            "addends": {
                args.base_label: {"seg": bound.base.seg, "pose": bound.base.pose, "total": bound.base.total,
                                  "d_pose": bound.base.d_pose, "source": bound.base.source},
                args.candidate_label: {"seg": bound.candidate.seg, "pose": bound.candidate.pose,
                                       "total": bound.candidate.total, "d_pose": bound.candidate.d_pose,
                                       "source": bound.candidate.source},
            },
            "rows_are_equal": bound.rows_are_equal,
            "net_ds": args.net_ds,
            "multiple": bound.multiple_of(args.net_ds) if args.net_ds is not None else None,
            "sentence": bound.describe(net_ds=args.net_ds),
        }
        print(json.dumps(payload, indent=2))
        return 0

    print(bound.describe(net_ds=args.net_ds))
    if bound.base.source == "derived" or bound.candidate.source == "derived":
        print(
            "\nNOTE: at least one bound was DERIVED (the receipt published none). "
            "The derivation uses the exact endpoint form and matches the harness "
            "digit-for-digit where both exist."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
