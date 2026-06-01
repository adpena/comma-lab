#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a queue-ready Z8 detail entropy-delta campaign plan.

This chains the reusable Z8 headroom profiler into the per-subband RD schedule
and then into the exact materializer work order. It does not execute the
materializer; the queue runner owns execution, receiver proof, replay gates, and
auth dispatch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from tac.substrates.z8_hierarchical_predictive_coding.detail_entropy_headroom import (
    _parse_quant_steps,
    _parse_workers,
    build_report,
)
from tac.substrates.z8_hierarchical_predictive_coding.entropy_delta_schedule import (
    build_entropy_delta_campaign_plan,
)


def _parse_num_pairs(raw: str) -> int:
    text = str(raw).strip().lower()
    if text == "all":
        return 1_000_000_000
    value = int(text)
    if value < 1:
        raise ValueError("--num-pairs must be >= 1 or 'all'")
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    raw = path.read_bytes()
    return {
        "path": path.as_posix(),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-bin", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--materializer-output-dir",
        type=Path,
        default=None,
        help="Candidate output dir for the materializer command. Defaults to output-dir/materialized.",
    )
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument(
        "--num-pairs",
        default="all",
        help="Number of pairs to measure, or 'all' for strict full-archive coverage.",
    )
    parser.add_argument(
        "--quant-steps",
        default="0.00390625,0.0078125,0.015625,0.03125,0.0625,0.125,0.25",
        help="Comma-separated detail quantization steps.",
    )
    parser.add_argument("--max-subband-mse", type=float, default=1.0e-5)
    parser.add_argument("--measure-static-range", action="store_true")
    parser.add_argument("--static-range-sample-cap", type=int, default=20000)
    parser.add_argument("--workers", default="auto")
    parser.add_argument(
        "--allow-partial-headroom-coverage",
        action="store_true",
        help="Allow advisory partial coverage. Strict queue execution keeps this off.",
    )
    parser.add_argument("--emit-receiver-proof", action="store_true")
    parser.add_argument("--run-inflate-runtime-benchmark", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing report/schedule/work_order/manifest JSON files in output-dir.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    archive_bin = args.archive_bin.resolve()
    if not archive_bin.is_file():
        raise SystemExit(f"archive not found: {archive_bin}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "headroom_report.json"
    schedule_path = output_dir / "entropy_delta_schedule.json"
    work_order_path = output_dir / "materializer_work_order.json"
    manifest_path = output_dir / "manifest.json"
    if not args.overwrite:
        for path in (report_path, schedule_path, work_order_path, manifest_path):
            if path.exists():
                raise SystemExit(f"refusing to overwrite existing artifact: {path}")

    report = build_report(
        archive_path=archive_bin,
        num_pairs=_parse_num_pairs(args.num_pairs),
        quant_steps=_parse_quant_steps(args.quant_steps),
        measure_static_range=bool(args.measure_static_range),
        static_range_sample_cap=int(args.static_range_sample_cap),
        workers=_parse_workers(args.workers),
    )
    materializer_output_dir = (
        args.materializer_output_dir.resolve()
        if args.materializer_output_dir is not None
        else output_dir / "materialized"
    )
    plan = build_entropy_delta_campaign_plan(
        report,
        max_subband_mse=float(args.max_subband_mse),
        schedule_json_path=schedule_path,
        materializer_output_dir=materializer_output_dir,
        archive_bin=archive_bin,
        repo_root=args.repo_root,
        require_full_archive_coverage=not bool(args.allow_partial_headroom_coverage),
        emit_receiver_proof=bool(args.emit_receiver_proof),
        run_inflate_runtime_benchmark=bool(args.run_inflate_runtime_benchmark),
    )

    report_artifact = _write_json(report_path, report)
    schedule_artifact = _write_json(schedule_path, plan["schedule"])
    work_order_artifact = _write_json(work_order_path, plan["materializer_work_order"])
    manifest = {
        "schema": "z8_entropy_delta_campaign_plan_artifacts.v1",
        "purpose": (
            "Durable artifacts for queue-owned Z8 headroom -> schedule -> "
            "materializer work-order planning."
        ),
        "score_claim": False,
        "promotion_eligible": False,
        "archive_bin": archive_bin.as_posix(),
        "archive_bytes": archive_bin.stat().st_size,
        "archive_sha256": hashlib.sha256(archive_bin.read_bytes()).hexdigest(),
        "output_dir": output_dir.as_posix(),
        "report": report_artifact,
        "schedule": schedule_artifact,
        "work_order": work_order_artifact,
        "plan": plan,
    }
    manifest_artifact = _write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "schema": plan["schema"],
                "manifest": manifest_artifact,
                "ready_for_queue_execution": plan["ready_for_queue_execution"],
                "blockers": plan["blockers"],
                "materializer_command": plan["materializer_work_order"].get("materializer_command"),
                "score_claim": False,
                "promotion_eligible": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
