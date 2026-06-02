#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Admit vetted NeRV long-training rows into a local-MLX scheduler queue."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis.nerv_long_training_campaign_admission import (  # noqa: E402
    DEFAULT_LOCAL_MLX_LONG_TRAINING_TIMEOUT_SECONDS,
    DEFAULT_QUEUE_ID,
    DEFAULT_STORAGE_EXPECTED_BYTES_PER_ROW,
    build_nerv_long_training_campaign_execution_admission,
    render_nerv_long_training_campaign_execution_admission_markdown,
)
from tac.repo_io import read_json, write_json_artifact, write_text_artifact  # noqa: E402


def _default_output_json() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return (
        REPO_ROOT
        / f".omx/research/nerv_long_training_campaign_execution_admission_{stamp}_codex.json"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consumer-verdict", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--output-queue", type=Path)
    parser.add_argument("--queue-id", default=DEFAULT_QUEUE_ID)
    parser.add_argument("--lane-id", required=True)
    parser.add_argument("--instance-job-id", required=True)
    parser.add_argument(
        "--active-claims-path",
        type=Path,
        default=Path(".omx/state/active_lane_dispatch_claims.md"),
    )
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--selected-experiment-id", action="append", default=[])
    parser.add_argument(
        "--storage-expected-bytes-per-row",
        type=int,
        default=DEFAULT_STORAGE_EXPECTED_BYTES_PER_ROW,
    )
    parser.add_argument("--storage-reserve-free-gb", type=float, default=40.0)
    parser.add_argument(
        "--local-mlx-timeout-seconds",
        type=int,
        default=DEFAULT_LOCAL_MLX_LONG_TRAINING_TIMEOUT_SECONDS,
        help=(
            "explicit timeout for admitted local_mlx long-training rows; "
            "default is 12h so full HiNeRV/SNeRV rows are not killed by "
            "short generic queue defaults"
        ),
    )
    parser.add_argument("--allowed-output-root", action="append", default=[])
    parser.add_argument(
        "--skip-active-local-mlx-process-scan",
        action="store_true",
        help=(
            "Disable the default local process guard. By default the CLI scans "
            "for already-running HiNeRV/SNeRV local-MLX training processes and "
            "fails closed instead of admitting another queue concurrently."
        ),
    )
    parser.add_argument("--expected-output-json-sha256")
    parser.add_argument("--expected-output-md-sha256")
    parser.add_argument("--expected-output-queue-sha256")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    source = args.consumer_verdict if args.consumer_verdict.is_absolute() else REPO_ROOT / args.consumer_verdict
    payload = read_json(source)
    if not isinstance(payload, dict):
        raise TypeError(f"{source}: expected JSON object")
    allowed_output_roots = tuple(args.allowed_output_root) or (
        "/Volumes/VertigoDataTier/pact",
        "/Volumes/APDataStore/pact",
    )
    admission = build_nerv_long_training_campaign_execution_admission(
        payload,
        repo_root=REPO_ROOT,
        active_claims_path=args.active_claims_path,
        lane_id=args.lane_id,
        instance_job_id=args.instance_job_id,
        queue_id=args.queue_id,
        limit=args.limit,
        selected_experiment_ids=tuple(args.selected_experiment_id),
        storage_expected_bytes_per_row=args.storage_expected_bytes_per_row,
        storage_reserve_free_gb=args.storage_reserve_free_gb,
        local_mlx_timeout_seconds=args.local_mlx_timeout_seconds,
        allowed_output_roots=allowed_output_roots,
        active_local_mlx_processes=()
        if args.skip_active_local_mlx_process_scan
        else tuple(_discover_active_local_mlx_processes()),
    )
    output_json = args.output_json or _default_output_json()
    if not output_json.is_absolute():
        output_json = REPO_ROOT / output_json
    json_result = write_json_artifact(
        output_json,
        admission,
        allow_overwrite=args.expected_output_json_sha256 is not None,
        expected_existing_sha256=args.expected_output_json_sha256,
    )
    md_result = None
    if args.output_md:
        output_md = args.output_md if args.output_md.is_absolute() else REPO_ROOT / args.output_md
        md_result = write_text_artifact(
            output_md,
            render_nerv_long_training_campaign_execution_admission_markdown(admission),
            allow_overwrite=args.expected_output_md_sha256 is not None,
            expected_existing_sha256=args.expected_output_md_sha256,
        )
    queue_result = None
    if args.output_queue:
        if admission.get("experiment_queue") is None:
            raise SystemExit(
                "FATAL: admission did not produce experiment_queue; blockers="
                + ",".join(str(item) for item in admission.get("blockers", []))
            )
        output_queue = args.output_queue if args.output_queue.is_absolute() else REPO_ROOT / args.output_queue
        queue_result = write_json_artifact(
            output_queue,
            admission["experiment_queue"],
            allow_overwrite=args.expected_output_queue_sha256 is not None,
            expected_existing_sha256=args.expected_output_queue_sha256,
        )

    print(
        json.dumps(
            {
                "schema": admission["schema"],
                "experiment_queue_ready": admission["experiment_queue_ready"],
                "admitted_experiment_count": admission["admitted_experiment_count"],
                "blocker_count": len(admission["blockers"]),
                "score_claim": admission["score_claim"],
                "ready_for_exact_eval_dispatch": admission[
                    "ready_for_exact_eval_dispatch"
                ],
                "output_json": json_result.path,
                "output_queue": None if queue_result is None else queue_result.path,
                "output_md": None if md_result is None else md_result.path,
            },
            sort_keys=True,
        )
    )
    return 0


def _discover_active_local_mlx_processes() -> list[dict[str, object]]:
    proc = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,stat=,etime=,command="],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return []
    own_pid = os.getpid()
    rows: list[dict[str, object]] = []
    for line in proc.stdout.splitlines():
        parts = line.strip().split(None, 4)
        if len(parts) < 5:
            continue
        pid_text, ppid_text, stat, etime, command = parts
        pid = _int_or_none(pid_text)
        if pid is None or pid == own_pid:
            continue
        if _is_local_mlx_training_command(command):
            rows.append(
                {
                    "pid": pid,
                    "ppid": _int_or_none(ppid_text),
                    "stat": stat,
                    "etime": etime,
                    "command": command,
                }
            )
    return rows


def _is_local_mlx_training_command(command: str) -> bool:
    text = command.lower()
    if "run_compact_renderer_mlx_spine_runner.py" in text:
        return "--execute-family" in text and (
            "hi_nerv" in text or "snerv" in text
        )
    return bool(
        "tools/experiment_queue.py" in text
        and "nerv_long_training_campaign" in text
        and "run-worker" in text
    )


def _int_or_none(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
