#!/usr/bin/env python3
"""Operator "learn and analyze" reader for the B1 HiNeRV full-curriculum run.

This is the queryable analysis surface per CLAUDE.md "Max observability" facet
4 (queryable post-hoc): it reads the canonical telemetry artifacts a B1 run
emits (the canonical ``telemetry.jsonl`` per-epoch JSONL + best-checkpoint
``.meta.json``) and the canonical PR95 8-stage curriculum factory, and prints
a human-readable summary (``--summary``) or machine-readable JSON (``--json``)
that the operator + orchestrator drive the learn/analyze/iterate loop with.

It delegates ALL projection logic to
``tac.substrates.hi_nerv.training_telemetry`` (no duplicated parsing); this is
a thin CLI per the AGENTS.md "thin CLIs delegate to tac" rule.

Every number printed is ``[macOS-MLX research-signal]`` per CLAUDE.md "MPS auth
eval is NOISE" — never a contest score.

Usage:
    # Human-readable summary of a live or finished run directory:
    .venv/bin/python tools/read_b1_telemetry.py \\
        --run-dir /Volumes/VertigoDataTier/pact/<run_id>/out/pr95_hnerv_mlx_training

    # Machine-readable JSON (for the orchestrator / dashboards):
    .venv/bin/python tools/read_b1_telemetry.py --run-dir <dir> --json

    # Just the canonical PR95 8-stage L14 curriculum table for a budget:
    .venv/bin/python tools/read_b1_telemetry.py --stage-table --total-epochs 29650

    # Point directly at a telemetry.jsonl (instead of a run dir):
    .venv/bin/python tools/read_b1_telemetry.py --telemetry <path>/telemetry.jsonl --summary
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Ensure repo src is importable when run as a script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tac.substrates.hi_nerv.training_telemetry import (  # noqa: E402
    MLX_RESEARCH_SIGNAL_AXIS,
    build_pr95_curriculum_stage_records,
    read_decomposed_epoch_rows,
    summarize_run,
)


def _find_telemetry(run_dir: Path) -> Path | None:
    """Locate the canonical telemetry.jsonl under a run directory."""
    direct = run_dir / "telemetry.jsonl"
    if direct.is_file():
        return direct
    matches = sorted(run_dir.rglob("telemetry.jsonl"))
    return matches[0] if matches else None


def _find_best_meta(run_dir: Path) -> Path | None:
    """Locate the most-recent best-checkpoint .meta.json under a run directory."""
    matches = sorted(run_dir.rglob("best_epoch*.meta.json"))
    return matches[-1] if matches else None


def _print_stage_table(records: tuple[Any, ...]) -> None:
    print(f"PR95 8-stage L14/L15 curriculum  [{MLX_RESEARCH_SIGNAL_AXIS}]")
    print(
        f"{'stage':>5}  {'epochs':>13}  {'n':>6}  {'loss_family':<26}  "
        f"{'qat':>3}  {'lambda':>6}  {'sigma':>5}  {'muon':>4}  descriptor"
    )
    for r in records:
        print(
            f"{r.stage_index:>5}  "
            f"{f'[{r.start_epoch},{r.end_epoch})':>13}  "
            f"{r.epochs_in_stage:>6}  "
            f"{r.loss_family:<26}  "
            f"{('Y' if r.qat_active else '-'):>3}  "
            f"{r.c1a_lambda:>6.3f}  "
            f"{r.sigma:>5.2f}  "
            f"{('Y' if r.muon_active else '-'):>4}  "
            f"{r.descriptor_id}"
        )


def _print_summary(summary: Any, telemetry_path: Path) -> None:
    print(f"B1 HiNeRV run telemetry  [{MLX_RESEARCH_SIGNAL_AXIS}]")
    print(f"  telemetry: {telemetry_path}")
    print(f"  epochs observed:        {summary.epochs_observed}")
    print(f"  loss finite:            {summary.loss_is_finite}")
    print(
        f"  total proxy loss:       first={summary.first_total:.5f}  "
        f"last={summary.last_total:.5f}  best={summary.best_total:.5f}"
    )
    print(
        f"  pose axis:              first={summary.first_pose:.5f}  "
        f"last={summary.last_pose:.5f}  best={summary.best_pose:.5f}"
    )
    print(
        f"  seg axis:               first={summary.first_seg:.5f}  "
        f"last={summary.last_seg:.5f}  best={summary.best_seg:.5f}"
    )
    print(f"  rate axis (last):       {summary.last_rate:.5f}")
    print(
        f"  wall-clock:             {summary.total_wall_clock_seconds:.1f}s  "
        f"({summary.mean_seconds_per_epoch:.3f}s/epoch)"
    )
    print(f"  muon epochs observed:   {summary.muon_epochs_observed}")
    print(f"  stages observed:        {len(summary.stages_observed)}")
    print(
        "  NOTE: all values are research signal; exact contest score is "
        "measured later on contest hardware (CPU x86_64 + CUDA T4), never MLX."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read + analyze B1 HiNeRV full-curriculum telemetry."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="run directory containing telemetry.jsonl + checkpoints/.",
    )
    parser.add_argument(
        "--telemetry",
        type=Path,
        default=None,
        help="explicit path to a canonical telemetry.jsonl.",
    )
    parser.add_argument(
        "--stage-table",
        action="store_true",
        help="print only the canonical PR95 8-stage L14 curriculum table.",
    )
    parser.add_argument(
        "--total-epochs",
        type=int,
        default=29650,
        help="curriculum total epoch budget for --stage-table (default 29650).",
    )
    parser.add_argument(
        "--muon-policy",
        type=str,
        default="faithful_stage8_only",
        choices=["faithful_stage8_only", "every_stage"],
        help="Muon policy for --stage-table (default faithful_stage8_only).",
    )
    parser.add_argument("--summary", action="store_true", help="human-readable summary.")
    parser.add_argument("--json", action="store_true", help="machine-readable JSON.")
    args = parser.parse_args(argv)

    # Stage-table-only mode: no run needed.
    if args.stage_table and args.run_dir is None and args.telemetry is None:
        records = build_pr95_curriculum_stage_records(
            total_epoch_budget=int(args.total_epochs),
            muon_policy=str(args.muon_policy),
        )
        if args.json:
            print(
                json.dumps(
                    {
                        "schema": "b1_hinerv_stage_table.v1",
                        "total_epoch_budget": int(args.total_epochs),
                        "muon_policy": str(args.muon_policy),
                        "measurement_axis": MLX_RESEARCH_SIGNAL_AXIS,
                        "stages": [r.as_dict() for r in records],
                    },
                    indent=2,
                )
            )
        else:
            _print_stage_table(records)
        return 0

    # Resolve the telemetry path.
    telemetry_path: Path | None = args.telemetry
    if telemetry_path is None and args.run_dir is not None:
        telemetry_path = _find_telemetry(args.run_dir)
    if telemetry_path is None or not telemetry_path.is_file():
        print(
            "ERROR: no telemetry.jsonl found. Pass --telemetry <path> or "
            "--run-dir <dir>, or use --stage-table for the curriculum table.",
            file=sys.stderr,
        )
        return 2

    rows = read_decomposed_epoch_rows(telemetry_path)
    if not rows:
        print(
            f"WARN: telemetry has no epoch rows yet: {telemetry_path}",
            file=sys.stderr,
        )
        return 0

    summary = summarize_run(rows)

    payload: dict[str, Any] = {
        "schema": "b1_hinerv_telemetry_read.v1",
        "telemetry_path": str(telemetry_path),
        "summary": summary.as_dict(),
        "epoch_rows": [r.as_dict() for r in rows],
    }

    # Best checkpoint manifest if available (informational; build manifest only
    # when a meta is present so the reader works on a live run mid-training).
    if args.run_dir is not None:
        best_meta = _find_best_meta(args.run_dir)
        if best_meta is not None:
            payload["best_checkpoint_meta_path"] = str(best_meta)

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    # Default + --summary: human-readable.
    _print_summary(summary, telemetry_path)
    if args.stage_table:
        print()
        records = build_pr95_curriculum_stage_records(
            total_epoch_budget=int(args.total_epochs),
            muon_policy=str(args.muon_policy),
        )
        _print_stage_table(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
