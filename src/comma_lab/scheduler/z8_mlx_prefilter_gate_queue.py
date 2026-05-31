# SPDX-License-Identifier: MIT
"""Queue-owned MLX prefilter gate for byte-closed Z8 candidates."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from comma_lab.local_exact_auth_gate_learning import (
    LOCAL_EXACT_AUTH_GATE_LEARNING_SIGNAL_SCHEMA,
)
from comma_lab.operator_storage_waterfall import operator_cold_store_roots
from comma_lab.scheduler.experiment_queue import QUEUE_SCHEMA, normalize_queue_definition
from tac.optimization.dqs1_materializer_feedback_bridge import FALSE_AUTHORITY
from tac.optimization.proxy_candidate_contract import require_no_truthy_authority_fields
from tac.repo_io import sha256_file

Z8_MLX_PREFILTER_GATE_QUEUE_SCHEMA = "z8_mlx_prefilter_gate_queue_metadata.v1"
Z8_MLX_REPLAY_SCHEMA = "z8_full_video_mlx_replay.v1"
LOCAL_EXACT_AUTH_GATE_SCHEMA = "local_candidate_exact_auth_gate.v1"
Z8_ARTIFACT_RETENTION_SCHEMA = "comma_lab.artifact_retention_plan.v1"


class Z8MlxPrefilterGateQueueError(ValueError):
    """Raised when a Z8 MLX prefilter queue cannot be built."""


def _safe_id(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower()
    return text or "candidate"


def _resolve(path: str | Path, repo_root: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else Path(repo_root) / value


def _repo_rel(path: str | Path, repo_root: str | Path) -> str:
    value = Path(path)
    repo = Path(repo_root)
    try:
        return value.resolve(strict=False).relative_to(repo.resolve(strict=False)).as_posix()
    except ValueError:
        return value.as_posix()


def _require_file(path: str | Path, *, repo_root: str | Path, label: str) -> Path:
    resolved = _resolve(path, repo_root)
    if not resolved.is_file():
        raise Z8MlxPrefilterGateQueueError(f"{label} missing: {path}")
    return resolved


def _candidate_id(row: Mapping[str, Any], index: int) -> str:
    raw = str(row.get("candidate_id") or row.get("id") or f"candidate_{index:04d}")
    return _safe_id(raw)


def _candidate_archive(row: Mapping[str, Any], *, repo_root: Path) -> Path:
    for key in ("candidate_archive_bin", "archive_bin", "candidate_bin", "path"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return _require_file(value, repo_root=repo_root, label=key)
    raise Z8MlxPrefilterGateQueueError(
        "candidate row must include candidate_archive_bin/archive_bin/candidate_bin/path"
    )


def _optional_archive_zip(row: Mapping[str, Any], *, repo_root: Path) -> Path | None:
    value = row.get("archive_zip") or row.get("byte_closed_archive_zip")
    if value is None or not str(value).strip():
        return None
    return _require_file(value, repo_root=repo_root, label="archive_zip")


def _replay_command(
    *,
    candidate_archive_bin: Path,
    reference_pairs_npy: Path,
    out_json: Path,
    repo_root: Path,
    upstream_dir: Path,
    archive_zip: Path | None,
    pair_chunk_size: int,
    scorer_hw: tuple[int, int],
    device: str,
) -> list[str]:
    command = [
        ".venv/bin/python",
        "tools/replay_z8_full_video_mlx_candidate.py",
        "--candidate-archive-bin",
        _repo_rel(candidate_archive_bin, repo_root),
        "--reference-pairs-npy",
        _repo_rel(reference_pairs_npy, repo_root),
        "--out-json",
        _repo_rel(out_json, repo_root),
        "--upstream-dir",
        _repo_rel(upstream_dir, repo_root),
        "--pair-chunk-size",
        str(int(pair_chunk_size)),
        "--scorer-hw",
        f"{int(scorer_hw[0])},{int(scorer_hw[1])}",
        "--device",
        str(device),
    ]
    if archive_zip is not None:
        command.extend(["--archive-zip", _repo_rel(archive_zip, repo_root)])
    return command


def _gate_command(
    *,
    replay_json: Path,
    out_json: Path,
    repo_root: Path,
    auth_frontier_score: float,
    mlx_target_action: float,
) -> list[str]:
    return [
        ".venv/bin/python",
        "tools/gate_local_candidate_for_exact_auth.py",
        "--mlx-prefilter-summary-json",
        _repo_rel(replay_json, repo_root),
        "--auth-frontier-score",
        repr(float(auth_frontier_score)),
        "--mlx-target-action",
        repr(float(mlx_target_action)),
        "--out-json",
        _repo_rel(out_json, repo_root),
        "--success-on-blocked",
    ]


def _learning_command(
    *,
    replay_json: Path,
    gate_json: Path,
    learning_json: Path,
    repo_root: Path,
    candidate_id: str,
    lane_id: str,
    family_id: str,
    posterior_path: Path | None,
) -> list[str]:
    command = [
        ".venv/bin/python",
        "tools/record_local_exact_auth_gate_learning.py",
        "--gate-report-json",
        _repo_rel(gate_json, repo_root),
        "--replay-summary-json",
        _repo_rel(replay_json, repo_root),
        "--candidate-id",
        str(candidate_id),
        "--lane-id",
        str(lane_id),
        "--family-id",
        str(family_id),
        "--out-json",
        _repo_rel(learning_json, repo_root),
    ]
    if posterior_path is not None:
        command.extend(["--posterior-path", _repo_rel(posterior_path, repo_root)])
    return command


def _cleanup_command(
    *,
    output_dir: Path,
    cleanup_json: Path,
    cleanup_journal: Path,
    repo_root: Path,
    cold_store_roots: Sequence[str],
    min_bytes: str,
    reserve_gb: float,
) -> list[str]:
    command = [
        ".venv/bin/python",
        "tools/compact_experiment_artifacts.py",
        _repo_rel(output_dir, repo_root),
        "--min-bytes",
        str(min_bytes),
        "--json-output",
        _repo_rel(cleanup_json, repo_root),
        "--journal-output",
        _repo_rel(cleanup_journal, repo_root),
        "--execute",
        "--action",
        "move",
        "--cold-store-reserve-gb",
        str(float(reserve_gb)),
    ]
    for root in cold_store_roots:
        command.extend(["--cold-store-root", str(root)])
    return command


def _artifact_record(path: Path, *, repo_root: Path) -> dict[str, Any]:
    return {
        "path": _repo_rel(path, repo_root),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def build_z8_mlx_prefilter_gate_queue(
    candidates: Sequence[Mapping[str, Any]],
    *,
    queue_id: str,
    repo_root: str | Path,
    reference_pairs_npy: str | Path,
    output_root: str | Path,
    auth_frontier_score: float,
    mlx_target_action: float | None = None,
    lane_id: str = "z8_mlx_prefilter_gate",
    upstream_dir: str | Path = "upstream",
    pair_chunk_size: int = 32,
    scorer_hw: tuple[int, int] = (384, 512),
    mlx_device: str = "cpu",
    local_cpu_concurrency: int = 1,
    local_mlx_concurrency: int = 1,
    local_io_concurrency: int = 1,
    family_id: str = "z8_hierarchical_predictive_coding",
    posterior_path: str | Path | None = None,
    enable_learning_signal: bool = True,
    enable_auto_cleanup: bool = True,
    cleanup_min_bytes: str = "100MiB",
    cleanup_cold_store_roots: Sequence[str] = (),
    cleanup_cold_store_reserve_gb: float = 40.0,
    timeout_seconds: int = 0,
) -> dict[str, Any]:
    """Compile Z8 candidate MLX replay + gate work into ``experiment_queue.v1``."""

    if not candidates:
        raise Z8MlxPrefilterGateQueueError("at least one candidate is required")
    if not str(queue_id).strip():
        raise Z8MlxPrefilterGateQueueError("queue_id is required")
    if not str(lane_id).strip():
        raise Z8MlxPrefilterGateQueueError("lane_id is required")
    if local_cpu_concurrency < 1 or local_mlx_concurrency < 1 or local_io_concurrency < 1:
        raise Z8MlxPrefilterGateQueueError("local concurrency values must be >= 1")
    if pair_chunk_size < 1:
        raise Z8MlxPrefilterGateQueueError("pair_chunk_size must be >= 1")
    if timeout_seconds < 0:
        raise Z8MlxPrefilterGateQueueError("timeout_seconds must be >= 0")
    if len(scorer_hw) != 2 or scorer_hw[0] <= 0 or scorer_hw[1] <= 0:
        raise Z8MlxPrefilterGateQueueError("scorer_hw must contain positive H,W")
    if mlx_device not in {"cpu", "gpu"}:
        raise Z8MlxPrefilterGateQueueError("mlx_device must be cpu or gpu")
    repo = Path(repo_root)
    reference = _require_file(reference_pairs_npy, repo_root=repo, label="reference_pairs_npy")
    upstream = _resolve(upstream_dir, repo)
    root = _resolve(output_root, repo)
    posterior = None if posterior_path is None else _resolve(posterior_path, repo)
    effective_cold_roots = tuple(
        cleanup_cold_store_roots
        or operator_cold_store_roots()
    )
    target = float(auth_frontier_score if mlx_target_action is None else mlx_target_action)
    if auth_frontier_score <= 0.0 or target <= 0.0:
        raise Z8MlxPrefilterGateQueueError("auth_frontier_score and mlx_target_action must be positive")

    queue_metadata = {
        "schema": Z8_MLX_PREFILTER_GATE_QUEUE_SCHEMA,
        "tool": "comma_lab.scheduler.z8_mlx_prefilter_gate_queue",
        "candidate_count": len(candidates),
        "reference_pairs_npy": _artifact_record(reference, repo_root=repo),
        "auth_frontier_score": float(auth_frontier_score),
        "mlx_target_action": target,
        "local_first": True,
        "mlx_before_cpu": True,
        "requires_exact_eval_before_promotion": True,
        "output_root": _repo_rel(root, repo),
        "auto_cleanup_enabled": bool(enable_auto_cleanup),
        "cleanup_cold_store_roots": list(effective_cold_roots),
        "cleanup_min_bytes": str(cleanup_min_bytes),
        "learning_signal_enabled": bool(enable_learning_signal),
        "learning_posterior_path": (
            None if posterior is None else _repo_rel(posterior, repo)
        ),
        "reproducibility_contract": {
            "schema": "z8_mlx_prefilter_gate_reproducibility_contract.v1",
            "candidate_inputs_hashed": True,
            "queue_steps_have_file_backed_postconditions": True,
            "gate_learning_identity_uses_artifact_sha256_not_wall_clock": True,
            "raw_scratch_cleanup_is_queue_owned": bool(enable_auto_cleanup),
            "exact_auth_required_before_promotion": True,
        },
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        queue_metadata,
        context="z8_mlx_prefilter_gate_queue.metadata",
    )

    experiments: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        cid = _candidate_id(candidate, index)
        candidate_bin = _candidate_archive(candidate, repo_root=repo)
        archive_zip = _optional_archive_zip(candidate, repo_root=repo)
        out_dir = _resolve(candidate.get("output_dir") or root / cid, repo)
        replay_json = out_dir / "full_video_mlx_replay.json"
        gate_json = out_dir / "mlx_prefilter_exact_auth_gate.json"
        learning_json = out_dir / "mlx_prefilter_gate_learning_signal.json"
        cleanup_json = out_dir / "artifact_retention_cleanup.json"
        cleanup_journal = out_dir / "artifact_retention_cleanup.json.journal.jsonl"
        replay_step = {
            "id": "run_z8_full_video_mlx_replay",
            "kind": "command",
            "command": _replay_command(
                candidate_archive_bin=candidate_bin,
                reference_pairs_npy=reference,
                out_json=replay_json,
                repo_root=repo,
                upstream_dir=upstream,
                archive_zip=archive_zip,
                pair_chunk_size=pair_chunk_size,
                scorer_hw=scorer_hw,
                device=mlx_device,
            ),
            "resources": {"kind": "local_mlx"},
            "timeout_seconds": timeout_seconds,
            "postconditions": [
                {"type": "path_exists", "path": _repo_rel(replay_json, repo)},
                {
                    "type": "json_equals",
                    "path": _repo_rel(replay_json, repo),
                    "key": "schema",
                    "equals": Z8_MLX_REPLAY_SCHEMA,
                },
                {"type": "json_false_authority", "path": _repo_rel(replay_json, repo)},
            ],
            "telemetry": {
                "schema": "z8_mlx_prefilter_replay_step_telemetry.v1",
                "artifact_paths": [_repo_rel(replay_json, repo)],
                "input_artifact_paths": [_repo_rel(candidate_bin, repo), _repo_rel(reference, repo)],
                "recursive": False,
            },
        }
        if archive_zip is not None:
            replay_step["telemetry"]["input_artifact_paths"].append(
                _repo_rel(archive_zip, repo)
            )
        gate_step = {
            "id": "gate_mlx_prefilter_for_exact_auth",
            "kind": "command",
            "requires": ["run_z8_full_video_mlx_replay"],
            "command": _gate_command(
                replay_json=replay_json,
                out_json=gate_json,
                repo_root=repo,
                auth_frontier_score=float(auth_frontier_score),
                mlx_target_action=target,
            ),
            "resources": {"kind": "local_cpu"},
            "timeout_seconds": timeout_seconds,
            "postconditions": [
                {"type": "path_exists", "path": _repo_rel(gate_json, repo)},
                {
                    "type": "json_equals",
                    "path": _repo_rel(gate_json, repo),
                    "key": "schema",
                    "equals": LOCAL_EXACT_AUTH_GATE_SCHEMA,
                },
                {"type": "json_false_authority", "path": _repo_rel(gate_json, repo)},
            ],
            "telemetry": {
                "schema": "z8_mlx_prefilter_gate_step_telemetry.v1",
                "artifact_paths": [_repo_rel(gate_json, repo)],
                "input_artifact_paths": [_repo_rel(replay_json, repo)],
                "recursive": False,
            },
        }
        steps = [replay_step, gate_step]
        if enable_learning_signal:
            learning_step = {
                "id": "record_mlx_prefilter_gate_learning",
                "kind": "command",
                "requires": ["gate_mlx_prefilter_for_exact_auth"],
                "command": _learning_command(
                    replay_json=replay_json,
                    gate_json=gate_json,
                    learning_json=learning_json,
                    repo_root=repo,
                    candidate_id=cid,
                    lane_id=lane_id,
                    family_id=family_id,
                    posterior_path=posterior,
                ),
                "resources": {"kind": "local_cpu"},
                "timeout_seconds": timeout_seconds,
                "postconditions": [
                    {"type": "path_exists", "path": _repo_rel(learning_json, repo)},
                    {
                        "type": "json_equals",
                        "path": _repo_rel(learning_json, repo),
                        "key": "schema",
                        "equals": LOCAL_EXACT_AUTH_GATE_LEARNING_SIGNAL_SCHEMA,
                    },
                    {"type": "json_false_authority", "path": _repo_rel(learning_json, repo)},
                ],
                "telemetry": {
                    "schema": "z8_mlx_prefilter_gate_learning_step_telemetry.v1",
                    "artifact_paths": [_repo_rel(learning_json, repo)],
                    "input_artifact_paths": [
                        _repo_rel(replay_json, repo),
                        _repo_rel(gate_json, repo),
                    ],
                    "recursive": False,
                },
            }
            steps.append(learning_step)
        if enable_auto_cleanup:
            cleanup_requires = [
                "record_mlx_prefilter_gate_learning"
                if enable_learning_signal
                else "gate_mlx_prefilter_for_exact_auth"
            ]
            cleanup_step = {
                "id": "cleanup_rebuildable_raw_scratch",
                "kind": "command",
                "requires": cleanup_requires,
                "command": _cleanup_command(
                    output_dir=out_dir,
                    cleanup_json=cleanup_json,
                    cleanup_journal=cleanup_journal,
                    repo_root=repo,
                    cold_store_roots=effective_cold_roots,
                    min_bytes=cleanup_min_bytes,
                    reserve_gb=cleanup_cold_store_reserve_gb,
                ),
                "resources": {"kind": "local_io_heavy"},
                "timeout_seconds": timeout_seconds,
                "postconditions": [
                    {"type": "path_exists", "path": _repo_rel(cleanup_json, repo)},
                    {
                        "type": "json_equals",
                        "path": _repo_rel(cleanup_json, repo),
                        "key": "plan.schema",
                        "equals": Z8_ARTIFACT_RETENTION_SCHEMA,
                    },
                    {
                        "type": "json_false_authority",
                        "path": _repo_rel(cleanup_json, repo),
                        "required_false": [
                            "plan.score_claim",
                            "plan.promotion_eligible",
                            "plan.ready_for_exact_eval_dispatch",
                        ],
                        "false_or_missing": [],
                    },
                ],
                "telemetry": {
                    "schema": "z8_mlx_prefilter_cleanup_step_telemetry.v1",
                    "artifact_paths": [
                        _repo_rel(cleanup_json, repo),
                        _repo_rel(cleanup_journal, repo),
                    ],
                    "input_artifact_paths": [_repo_rel(out_dir, repo)],
                    "recursive": True,
                },
            }
            steps.append(cleanup_step)
        experiment_metadata = {
            **queue_metadata,
            "candidate_id": cid,
            "family_id": family_id,
            "candidate_archive_bin": _artifact_record(candidate_bin, repo_root=repo),
            "archive_zip": (
                _artifact_record(archive_zip, repo_root=repo)
                if archive_zip is not None
                else None
            ),
            "mlx_replay_json": _repo_rel(replay_json, repo),
            "mlx_prefilter_gate_json": _repo_rel(gate_json, repo),
            "mlx_prefilter_gate_learning_signal_json": (
                _repo_rel(learning_json, repo) if enable_learning_signal else None
            ),
            "artifact_retention_cleanup_json": (
                _repo_rel(cleanup_json, repo) if enable_auto_cleanup else None
            ),
            "cpu_replay_allowed_only_after_gate_action": "run_local_cpu_replay",
            **FALSE_AUTHORITY,
        }
        require_no_truthy_authority_fields(
            experiment_metadata,
            context=f"z8_mlx_prefilter_gate_queue.experiment:{cid}",
        )
        experiments.append(
            {
                "id": f"z8_mlx_prefilter_{index:04d}_{cid}",
                "lane_id": lane_id,
                "priority": 10 + index,
                "metadata": experiment_metadata,
                "steps": steps,
            }
        )

    return normalize_queue_definition(
        {
            "schema": QUEUE_SCHEMA,
            "queue_id": str(queue_id),
            "controls": {
                "mode": "running",
                "local_first": True,
                "max_concurrency": {
                    "local_cpu": int(local_cpu_concurrency),
                    "local_mlx": int(local_mlx_concurrency),
                    "local_io_heavy": int(local_io_concurrency),
                },
            },
            "experiments": experiments,
        }
    )


__all__ = [
    "LOCAL_EXACT_AUTH_GATE_SCHEMA",
    "Z8_ARTIFACT_RETENTION_SCHEMA",
    "Z8_MLX_PREFILTER_GATE_QUEUE_SCHEMA",
    "Z8_MLX_REPLAY_SCHEMA",
    "Z8MlxPrefilterGateQueueError",
    "build_z8_mlx_prefilter_gate_queue",
]
