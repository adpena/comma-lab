#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a non-promotional L5-v2 paired measurement dispatch plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.optimization.l5_v2_measurement_schedule import (  # noqa: E402
    L5V2_MEASUREMENT_SCHEDULE_ARTIFACT_PATH,
)
from tac.optimization.l5_v2_paired_measurement_dispatch_plan import (  # noqa: E402
    L5V2_PAIRED_MEASUREMENT_DISPATCH_PLAN_ARTIFACT_PATH,
    L5V2_PAIRED_MEASUREMENT_DISPATCH_PLAN_REPORT_PATH,
    build_l5_v2_paired_measurement_dispatch_plan,
    dispatch_plan_json,
    render_l5_v2_paired_measurement_dispatch_plan_markdown,
)


def _refuse_tmp(path: Path) -> None:
    text = str(path)
    if text.startswith("/tmp/") or "/private/tmp/" in text or "/var/tmp/" in text:
        raise ValueError(f"refusing to write L5 v2 dispatch plan to tmp: {text!r}")


def _write(path: Path, text: str) -> None:
    _refuse_tmp(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"schedule JSON must be an object: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--schedule-json",
        type=Path,
        default=Path(L5V2_MEASUREMENT_SCHEDULE_ARTIFACT_PATH),
        help="Input L5-v2 measurement schedule JSON.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(L5V2_PAIRED_MEASUREMENT_DISPATCH_PLAN_ARTIFACT_PATH),
        help="Output dispatch-plan JSON path.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(L5V2_PAIRED_MEASUREMENT_DISPATCH_PLAN_REPORT_PATH),
        help="Output dispatch-plan markdown path.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        schedule = _read_json(args.schedule_json)
        schedule_sha = _sha256_file(args.schedule_json)
        plan = build_l5_v2_paired_measurement_dispatch_plan(
            schedule=schedule,
            schedule_path=str(args.schedule_json),
            schedule_sha256=schedule_sha,
        )
        _write(args.output_json, dispatch_plan_json(plan))
        _write(
            args.output_md,
            render_l5_v2_paired_measurement_dispatch_plan_markdown(plan),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[l5-v2-dispatch-plan] FATAL: {exc}", file=sys.stderr)
        return 2
    print(
        "[l5-v2-dispatch-plan] "
        f"plan_id={plan['plan_id']} "
        f"work_unit_count={plan['work_unit_count']} "
        "score_claim=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
