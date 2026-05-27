# SPDX-License-Identifier: MIT
"""Queue-owned deterministic replay verification for MLX master gradients."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .experiment_queue import QUEUE_SCHEMA, ExperimentQueueError, normalize_queue_definition

MLX_MASTER_GRADIENT_REPLAY_QUEUE_SCHEMA = "mlx_master_gradient_replay_queue.v1"
MLX_MASTER_GRADIENT_REPLAY_BUNDLE_SCHEMA = "mlx_master_gradient_replay_bundle.v1"
MLX_MASTER_GRADIENT_REPLAY_RERUN_SCHEMA = "mlx_master_gradient_replay_rerun.v1"
TOOL_NAME = "comma_lab.scheduler.mlx_master_gradient_replay_queue"
RERUN_TOOL = "tools/rerun_mlx_master_gradient_replay.py"
FALSE_AUTHORITY = {
    "score_claim": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_id(value: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in value)
    return out.strip("._-").lower() or "mlx_master_gradient_replay"


def _repo_rel(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve(path: str | Path, *, repo_root: Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else repo_root / candidate


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ExperimentQueueError(f"{label}: invalid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ExperimentQueueError(f"{label}: expected JSON object")
    return payload


def _file_record(path: Path, *, repo_root: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ExperimentQueueError(f"required replay bundle missing: {path}")
    return {
        "path": _repo_rel(path, repo_root),
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _require_false_authority(payload: Mapping[str, Any], *, label: str) -> None:
    for key in FALSE_AUTHORITY:
        if payload.get(key) is not False:
            raise ExperimentQueueError(f"{label}: {key} must be explicit false")


def _validate_bundle(path: Path, *, repo_root: Path, index: int) -> dict[str, Any]:
    payload = _load_json(path, label=f"replay_bundle[{index}]")
    if payload.get("schema") != MLX_MASTER_GRADIENT_REPLAY_BUNDLE_SCHEMA:
        raise ExperimentQueueError(
            f"replay_bundle[{index}] schema must be "
            f"{MLX_MASTER_GRADIENT_REPLAY_BUNDLE_SCHEMA}"
        )
    if payload.get("tool") != "tools/extract_master_gradient_mlx.py":
        raise ExperimentQueueError(
            f"replay_bundle[{index}] must target tools/extract_master_gradient_mlx.py"
        )
    gate = payload.get("calibration_gate")
    if not isinstance(gate, Mapping):
        raise ExperimentQueueError(f"replay_bundle[{index}] calibration_gate missing")
    _require_false_authority(gate, label=f"replay_bundle[{index}].calibration_gate")
    record = _file_record(path, repo_root=repo_root)
    record["archive_sha256"] = (
        payload.get("archive", {}).get("sha256")
        if isinstance(payload.get("archive"), Mapping)
        else None
    )
    record["source_npy_sha256"] = (
        payload.get("output", {}).get("npy_sha256")
        if isinstance(payload.get("output"), Mapping)
        else None
    )
    return record


def _postconditions(summary_path: Path, diff_path: Path, *, repo_root: Path) -> list[dict[str, Any]]:
    summary_rel = _repo_rel(summary_path, repo_root)
    diff_rel = _repo_rel(diff_path, repo_root)
    return [
        {"type": "path_exists", "path": summary_rel},
        {
            "type": "json_equals",
            "path": summary_rel,
            "key": "schema",
            "equals": MLX_MASTER_GRADIENT_REPLAY_RERUN_SCHEMA,
        },
        {
            "type": "json_equals",
            "path": summary_rel,
            "key": "matched",
            "equals": True,
        },
        {"type": "json_false_authority", "path": summary_rel},
        {"type": "path_exists", "path": diff_rel},
        {"type": "json_false_authority", "path": diff_rel},
    ]


def build_mlx_master_gradient_replay_queue(
    *,
    replay_bundle_paths: Sequence[str | Path],
    output_root: str | Path,
    queue_id: str,
    repo_root: str | Path,
    local_mlx_concurrency: int = 1,
    timeout_seconds: int = 0,
    strict: bool = True,
    append_manifest: bool = False,
    python_executable: str = ".venv/bin/python",
) -> dict[str, Any]:
    """Build an experiment queue that reruns MLX replay bundles and diffs them."""

    if not replay_bundle_paths:
        raise ExperimentQueueError("at least one replay bundle is required")
    if local_mlx_concurrency < 1:
        raise ExperimentQueueError("local_mlx_concurrency must be >= 1")
    if timeout_seconds < 0:
        raise ExperimentQueueError("timeout_seconds must be non-negative")
    repo = Path(repo_root)
    queue_safe_id = _safe_id(queue_id)
    output_dir = _resolve(output_root, repo_root=repo) / queue_safe_id
    bundle_paths = [
        _resolve(path, repo_root=repo) for path in replay_bundle_paths
    ]
    source_records = [
        _validate_bundle(path, repo_root=repo, index=index)
        for index, path in enumerate(bundle_paths)
    ]

    steps: list[dict[str, Any]] = []
    for index, (path, record) in enumerate(zip(bundle_paths, source_records, strict=True), start=1):
        step_id = f"rerun_mlx_master_gradient_replay_{index:04d}"
        run_id = _safe_id(f"{index:04d}_{Path(record['path']).stem}")
        summary_path = output_dir / run_id / "replay_rerun_summary.json"
        diff_path = output_dir / run_id / "replay_diff.json"
        command = [
            python_executable,
            RERUN_TOOL,
            _repo_rel(path, repo),
            "--output-dir",
            _repo_rel(output_dir, repo),
            "--run-id",
            run_id,
        ]
        if strict:
            command.append("--strict")
        if append_manifest:
            command.append("--append-manifest")
        steps.append(
            {
                "id": step_id,
                "kind": "command",
                "command": command,
                "requires": [],
                "resources": {"kind": "local_mlx"},
                "timeout_seconds": timeout_seconds,
                "postconditions": _postconditions(
                    summary_path,
                    diff_path,
                    repo_root=repo,
                ),
                "telemetry": {
                    "schema": MLX_MASTER_GRADIENT_REPLAY_QUEUE_SCHEMA,
                    "artifact_paths": [
                        _repo_rel(output_dir / run_id, repo),
                        _repo_rel(summary_path, repo),
                        _repo_rel(diff_path, repo),
                    ],
                    "input_artifact_paths": [record["path"]],
                    "recursive": True,
                },
            }
        )

    metadata = {
        "schema": MLX_MASTER_GRADIENT_REPLAY_QUEUE_SCHEMA,
        "tool": TOOL_NAME,
        "generated_at_utc": _utc_now(),
        "source_artifacts": {"replay_bundles": source_records},
        "output_root": _repo_rel(output_dir, repo),
        "strict": bool(strict),
        "append_manifest": bool(append_manifest),
        "candidate_generation_only": True,
        "deterministic_replay_verification_only": True,
        "requires_contest_cpu_or_cuda_auth_eval_before_dispatch": True,
        **FALSE_AUTHORITY,
    }
    return normalize_queue_definition(
        {
            "schema": QUEUE_SCHEMA,
            "queue_id": queue_id,
            "controls": {
                "mode": "running",
                "local_first": True,
                "max_concurrency": {
                    "local_mlx": int(local_mlx_concurrency),
                    "local_cpu": 1,
                },
            },
            "experiments": [
                {
                    "id": f"mlx_master_gradient_replay_{queue_safe_id}",
                    "lane_id": "mlx_master_gradient_replay_verification",
                    "priority": 10,
                    "metadata": metadata,
                    "steps": steps,
                }
            ],
        }
    )
