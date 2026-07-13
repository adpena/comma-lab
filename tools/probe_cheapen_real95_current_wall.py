#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Read-only current-wall receipt from the sacred v7.5.2 n600 run.

The live launch did not enable the trainer's component profiler.  This tool
therefore measures only quantities present in durable logs and fails closed on
the requested scorer/render/R/loss decomposition instead of allocating the
unprofiled residual by an old 78/22 ratio.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = REPO / "experiments/results/levelset_v752_baseline_20260710T185913Z"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()


def _rows(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for raw in path.read_text(errors="replace").splitlines():
        start = raw.find("{")
        if start < 0:
            continue
        try:
            payload = json.loads(raw[start:])
        except (ValueError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            result.append(payload)
    return result


def build_receipt(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    log_path = run_dir / "daemon.log"
    launch_path = run_dir / "launch.sh"
    resume_path = run_dir / "levelset_resume_state.npz"
    for path in (log_path, launch_path, resume_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    rows = _rows(log_path)
    verdicts = [
        row
        for row in rows
        if row.get("stage") == "verdict"
        and int(row.get("epoch", -1)) >= 25
        and row.get("ts")
    ]
    done = {
        int(row["epoch"]): float(row["secs"])
        for row in rows
        if row.get("stage") == "verdict_async_done"
    }
    if len(verdicts) < 3 or not done:
        raise ValueError("insufficient durable verdict timing rows")
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    interval_rows: list[dict[str, Any]] = []
    corrected_epoch_seconds: list[float] = []
    for left, right in pairwise(verdicts):
        ep_left, ep_right = int(left["epoch"]), int(right["epoch"])
        if ep_left not in done or ep_right not in done:
            continue
        completion_delta = (
            datetime.strptime(str(right["ts"]), fmt)
            - datetime.strptime(str(left["ts"]), fmt)
        ).total_seconds()
        # verdict timestamp is the async completion.  Remove the change in CPU
        # service time between endpoints to recover boundary-to-boundary train
        # time: (C_b-C_a) - (V_b-V_a).
        train_delta = completion_delta - (done[ep_right] - done[ep_left])
        epoch_seconds = train_delta / (ep_right - ep_left)
        corrected_epoch_seconds.append(epoch_seconds)
        interval_rows.append(
            {
                "epoch_start": ep_left,
                "epoch_end": ep_right,
                "completion_delta_s": completion_delta,
                "cpu_service_delta_s": done[ep_right] - done[ep_left],
                "derived_training_epoch_s": epoch_seconds,
            }
        )
    if not corrected_epoch_seconds:
        raise ValueError("no intervals have paired async service-time rows")

    launch = launch_path.read_text()
    lever_checks = {
        "custom_grouped_backward": "TAC_MLX_CUSTOM_GROUPED_BACKWARD=1" in launch,
        "custom_persistence_pool": "TAC_MLX_CUSTOM_PERSISTENCE_POOL=1" in launch,
        "fused_R": "--fused-r-kernel" in launch,
        "async_cpu_verdict": "--async-verdict" in launch,
        "safe_hosc_compile": "--safe-compile-regions hosc_activation" in launch,
        "micro_batch_pairs": "--micro-batch-pairs" in launch,
    }
    epoch_s = statistics.median(corrected_epoch_seconds)
    cpu_service_s = statistics.median(done.values())
    cadence = 25
    components = {
        "mlx_scorer_forward_s_per_epoch": None,
        "mlx_scorer_backward_s_per_epoch": None,
        "render_s_per_epoch": None,
        "R_s_per_epoch": None,
        "loss_terms_s_per_epoch": None,
        "cpu_torch_verdict_service_s_per_call": cpu_service_s,
        "cpu_torch_verdict_service_amortized_s_per_epoch": cpu_service_s / cadence,
        "cpu_torch_verdict_critical_path_s_per_epoch": 0.0,
        "unallocated_training_critical_path_s_per_epoch": epoch_s,
    }
    return {
        "schema": "cheapen_real95_current_wall.v1",
        "written_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "lane_id": "lane_cheapen_real95_tilehalo_fp16_20260713",
        "axis": "[macOS-MLX training wall + macOS-CPU advisory verdict; NON-PROMOTABLE]",
        "provenance": {
            "git_sha": _git_sha(),
            "probe_source": "tools/probe_cheapen_real95_current_wall.py",
            "probe_source_sha256": _sha256(Path(__file__).resolve()),
            "run_dir": str(run_dir.relative_to(REPO)),
            "run_dir_mutated": False,
            "daemon_log": str(log_path.relative_to(REPO)),
            "daemon_log_sha256": _sha256(log_path),
            "launch_script": str(launch_path.relative_to(REPO)),
            "launch_script_sha256": _sha256(launch_path),
            "resume_checkpoint": str(resume_path.relative_to(REPO)),
            "resume_checkpoint_sha256": _sha256(resume_path),
        },
        "current_launch_speed_levers": lever_checks,
        "all_requested_speed_levers_on": all(lever_checks.values()),
        "speed_lever_caveat": (
            "micro-batch is OFF in the canonical live launch; all currently admitted neutral "
            "levers are ON, but the prompt's micro-batch composition is not this trajectory"
        ),
        "measured_wall": {
            "evidence_status": "DERIVED_FROM_MEASURED_N600_LOG_TIMESTAMPS",
            "epoch_window": [interval_rows[0]["epoch_start"], interval_rows[-1]["epoch_end"]],
            "n_pairs_per_epoch": 600,
            "optimizer_chunks_per_epoch": 75,
            "median_training_epoch_s": epoch_s,
            "median_s_per_pair_visit": epoch_s / 600.0,
            "median_s_per_optimizer_chunk": epoch_s / 75.0,
            "intervals": interval_rows,
            "async_cpu_verdict_seconds_by_epoch": done,
        },
        "canonical_equation_ready_row": {
            "status": "INCOMPLETE_BLOCKED_NOT_COMPOSABLE",
            "total_training_critical_path_s_per_epoch": epoch_s,
            "components": components,
            "component_evidence_status": {
                "mlx_scorer_forward": "BLOCKED_NOT_MEASURED",
                "mlx_scorer_backward": "BLOCKED_NOT_MEASURED",
                "render": "BLOCKED_NOT_MEASURED",
                "R": "BLOCKED_NOT_MEASURED",
                "loss_terms": "BLOCKED_NOT_MEASURED",
                "cpu_torch_verdict_service": "MEASURED",
                "cpu_torch_verdict_critical_path": "MEASURED_OBSERVED_ZERO_ASYNC_WAIT",
                "unallocated_training_critical_path": "DERIVED_RESIDUAL",
            },
            "sum_known_critical_path_s_per_epoch": 0.0,
            "unallocated_fraction": 1.0,
            "composition_admissible": False,
        },
        "blocker": {
            "classification": "EXECUTION_ENVIRONMENT_PLUS_MISSING_LIVE_INSTRUMENTATION",
            "detail": (
                "live trainer --profile-timing was OFF and this sandbox exposes no Metal device; "
                "the stale 78/22 CPU or old stripped-B8 MLX rows are explicitly not substituted"
            ),
            "recovery": (
                "run the new precision probe from a Metal-enabled host process; the exact full "
                "step split additionally requires a future read-only extraction harness because "
                "the current loss closure is nested and the sacred run has no component profile"
            ),
        },
        "pointer_delta": "ZERO; read-only wall receipt",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    out = args.out.resolve()
    if str(out).startswith("/tmp/") or str(out).startswith("/private/tmp/"):
        raise SystemExit("refusing /tmp durable evidence path")
    payload = build_receipt(args.run_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
