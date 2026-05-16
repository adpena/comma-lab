#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build the L5 v2 first-match lattice measurement schedule."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.optimization.l5_v2_measurement_schedule import (  # noqa: E402
    L5V2_MEASUREMENT_SCHEDULE_ARTIFACT_PATH,
    L5V2_MEASUREMENT_SCHEDULE_REPORT_PATH,
    build_l5_v2_lattice_measurement_schedule,
    render_l5_v2_lattice_measurement_schedule_markdown,
    schedule_json,
)


def _read_json(path: Path | None, *, label: str) -> dict[str, object] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} JSON must be an object: {path}")
    return payload


def _refuse_tmp(path: Path) -> None:
    text = str(path)
    if text.startswith("/tmp/") or "/private/tmp/" in text or "/var/tmp/" in text:
        raise ValueError(f"refusing to write L5 v2 measurement schedule to tmp: {text!r}")


def _write(path: Path, text: str) -> None:
    _refuse_tmp(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe-intake-json",
        type=Path,
        default=None,
        help="Optional l5_v2_probe_observation_intake JSON artifact.",
    )
    parser.add_argument(
        "--sideinfo-effect-curve-json",
        type=Path,
        default=None,
        help="Optional paired TT5L side-info effect-curve summary JSON artifact.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(L5V2_MEASUREMENT_SCHEDULE_ARTIFACT_PATH),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(L5V2_MEASUREMENT_SCHEDULE_REPORT_PATH),
        help="Output markdown report path.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        intake = _read_json(args.probe_intake_json, label="probe intake")
        sideinfo_effect_curve = _read_json(
            args.sideinfo_effect_curve_json,
            label="side-info effect curve",
        )
        schedule = build_l5_v2_lattice_measurement_schedule(
            probe_intake=intake,
            sideinfo_effect_curve=sideinfo_effect_curve,
        )
        _write(args.output_json, schedule_json(schedule))
        _write(args.output_md, render_l5_v2_lattice_measurement_schedule_markdown(schedule))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[l5-v2-measurement-schedule] FATAL: {exc}", file=sys.stderr)
        return 2
    print(
        "[l5-v2-measurement-schedule] "
        f"active_rule_id={schedule['active_rule_id']} "
        f"active_measurement_ids={schedule['active_measurement_ids']} "
        "score_claim=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
