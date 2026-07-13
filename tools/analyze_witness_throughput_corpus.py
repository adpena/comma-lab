#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Read-only throughput accounting over witness JSONL/log artifacts.

The tool never imports MLX, launches training, loads scorer weights, or writes
inside an input run directory.  It measures verdict completion cadence and D-A
rows while preserving two fail-closed distinctions:

* zero ``verdict_skip`` rows does not establish zero contention; and
* D-A backward fields are inclusive and therefore not additive components.

Outputs are MEANS-only, non-promotable, and never move the score pointer.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from tac.canonical_equations.async_overlap_throughput_20260713 import (  # noqa: E402
    EQUATION_ID,
    async_only_identifiability,
    derive_incremental_vjp_s,
)

COMPONENT_FIELDS = (
    "teacher_forward_s",
    "teacher_backward_s",
    "witness_forward_s",
    "witness_backward_s",
    "realized_R_s",
    "verdict_s",
    "checkpoint_io_s",
    "epoch_total_s",
)


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _median(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def load_rows(path: Path) -> tuple[list[dict[str, Any]], int, int]:
    rows: list[dict[str, Any]] = []
    malformed = 0
    total = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            total += 1
            line = raw.strip()
            if not line.startswith("{"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows, malformed, total


def _component_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    components = [row for row in rows if row.get("stage") == "witness_component_wallclock"]
    complete = [row for row in components if row.get("complete") is True]
    medians = {
        field: _median([
            float(row[field])
            for row in complete
            if isinstance(row.get(field), (int, float)) and math.isfinite(float(row[field]))
        ])
        for field in COMPONENT_FIELDS
    }
    incremental: list[float] = []
    unresolved = 0
    for row in complete:
        try:
            incremental.append(derive_incremental_vjp_s(
                forward_s=float(row["teacher_forward_s"]),
                backward_inclusive_s=float(row["teacher_backward_s"]),
            ))
        except (KeyError, TypeError, ValueError):
            unresolved += 1
    return {
        "row_count": len(components),
        "complete_row_count": len(complete),
        "median_seconds": medians,
        "teacher_incremental_vjp_median_s": _median(incremental),
        "teacher_incremental_vjp_unresolved_rows": unresolved,
        "measurement_scope": sorted({
            str(row.get("measurement_scope")) for row in components if row.get("measurement_scope")
        }),
        "additivity_refused": True,
        "reason": (
            "teacher_backward_s and witness_backward_s include their required forward; "
            "observed fields cannot be summed into epoch_total_s"
        ),
    }


def analyze_log(path: Path) -> dict[str, Any]:
    rows, malformed, total = load_rows(path)
    verdicts = [row for row in rows if row.get("stage") == "verdict"]
    timed = sorted(
        (
            (int(row["epoch"]), ts, row)
            for row in verdicts
            if isinstance(row.get("epoch"), (int, float)) and (ts := _timestamp(row.get("ts")))
        ),
        key=lambda item: item[0],
    )
    done = {
        int(row["epoch"]): float(row["secs"])
        for row in rows
        if row.get("stage") == "verdict_async_done"
        and isinstance(row.get("epoch"), (int, float))
        and isinstance(row.get("secs"), (int, float))
    }
    windows: list[dict[str, Any]] = []
    for (ep0, ts0, _), (ep1, ts1, _) in pairwise(timed):
        if ep1 <= ep0:
            continue
        elapsed = (ts1 - ts0).total_seconds()
        if elapsed <= 0:
            continue
        worker = done.get(ep0)
        windows.append({
            "start_epoch": ep0,
            "end_epoch": ep1,
            "epoch_delta": ep1 - ep0,
            "completion_interval_s": elapsed,
            "completion_interval_s_per_epoch": elapsed / (ep1 - ep0),
            "async_service_s": worker,
            "service_over_completion_interval": (worker / elapsed if worker is not None else None),
            "denominator_scope": (
                "DERIVED completion-to-completion interval; not a dispatch-to-dispatch train window"
            ),
        })
    skips = [row for row in rows if row.get("stage") == "verdict_skip"]
    services = list(done.values())
    intervals = [float(row["completion_interval_s_per_epoch"]) for row in windows]
    duties = [
        float(row["service_over_completion_interval"])
        for row in windows
        if row["service_over_completion_interval"] is not None
    ]
    gt_rows = [row for row in rows if row.get("stage") == "gt"]
    provenance = next((row for row in rows if row.get("stage") == "provenance"), {})
    profile = [row for row in rows if row.get("stage") == "profile_timing"]
    return {
        "path": str(path),
        "n_pairs": (gt_rows[-1].get("n_pairs") if gt_rows else None),
        "provenance": provenance,
        "line_count": total,
        "json_row_count": len(rows),
        "malformed_json_line_count": malformed,
        "verdict_count": len(verdicts),
        "async_done_count": len(done),
        "verdict_skip_count": len(skips),
        "async_identifiability": async_only_identifiability(cadence_miss_count=len(skips)),
        "completion_windows": windows,
        "completion_interval_s_per_epoch_median": _median(intervals),
        "async_service_s_median": _median(services),
        "service_over_completion_interval_median": _median(duties),
        "contention_penalty_fraction": None,
        "contention_status": "UNMEASURED_NO_MATCHED_SOLO_ARM",
        "profile_timing_rows": profile,
        "component_timing": _component_summary(rows),
    }


def _microbatch_summary(runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    arms: list[dict[str, Any]] = []
    for run in runs:
        parent = Path(str(run["path"])).parent.name
        if not (parent.startswith("B") and parent[1:].isdigit()):
            continue
        profiles = run.get("profile_timing_rows") or []
        if not profiles:
            continue
        batch = int(parent[1:])
        arm = {
            "micro_batch_pairs": batch,
            "path": run["path"],
            "git_sha": (run.get("provenance") or {}).get("git_sha"),
            "profile_row_count": len(profiles),
            "median_epoch_s": _median([
                float(row["t_epoch_s"]) for row in profiles
                if isinstance(row.get("t_epoch_s"), (int, float))
            ]),
            "median_step_s": _median([
                float(row["t_step_fwd_bwd_opt_ema_s"]) for row in profiles
                if isinstance(row.get("t_step_fwd_bwd_opt_ema_s"), (int, float))
            ]),
            "median_verdict_s": _median([
                float(row["t_verdict_s"]) for row in profiles
                if isinstance(row.get("t_verdict_s"), (int, float))
            ]),
        }
        arms.append(arm)
    baseline = next((arm for arm in arms if arm["micro_batch_pairs"] == 1), None)
    if baseline is None:
        return None
    for arm in arms:
        arm["epoch_speedup_vs_B1"] = (
            float(baseline["median_epoch_s"]) / float(arm["median_epoch_s"])
            if baseline["median_epoch_s"] and arm["median_epoch_s"] else None
        )
        arm["step_speedup_vs_B1"] = (
            float(baseline["median_step_s"]) / float(arm["median_step_s"])
            if baseline["median_step_s"] and arm["median_step_s"] else None
        )
    return {
        "status": "MEASURED_HISTORICAL_N24_DIAGNOSTIC__NOT_CURRENT_V9_AUTHORITY",
        "arms": sorted(arms, key=lambda arm: int(arm["micro_batch_pairs"])),
        "pairing_caveat": (
            "two profile rows per separately launched arm; git SHAs may differ; no ABBA thermal control"
        ),
        "current_v9_in_loop_transfer_established": False,
        "score_claim": False,
    }


def build_report(paths: list[Path]) -> dict[str, Any]:
    runs = [analyze_log(path) for path in paths]
    return {
        "schema": "witness_throughput_corpus_analysis.v1",
        "generated_at_utc": _utc_now(),
        "equation_id": EQUATION_ID,
        "authority": {
            "evidence": "MEASURED artifact fields plus explicitly DERIVED arithmetic",
            "axis": "[macOS local wall-clock advisory] NON-PROMOTABLE",
            "score_claim": False,
            "pointer_moved": False,
            "means_caveat": "throughput is MEANS; only a byte-closed exact row moves score",
        },
        "runs": runs,
        "microbatch_comparison": _microbatch_summary(runs),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    for path in args.logs:
        if not path.is_file():
            parser.error(f"input log does not exist: {path}")
    if args.output is not None:
        for path in args.logs:
            if _is_within(args.output, path.parent):
                parser.error(
                    f"refusing output inside sacred input run directory: {args.output}"
                )
    report = build_report(args.logs)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.output.with_name(f".{args.output.name}.tmp")
        tmp.write_text(payload)
        tmp.replace(args.output)
        print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
