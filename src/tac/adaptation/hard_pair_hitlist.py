# SPDX-License-Identifier: MIT
"""Build hard-pair hitlists from full-video scorer-response components."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "nerv_hard_pair_hitlist.v1"
FALSE_AUTHORITY = {
    "score_claim": False,
    "frontier_score_claim": False,
    "rank_or_kill_eligible": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}


def build_hard_pair_hitlist_from_mlx_response(
    *,
    mlx_response: Mapping[str, Any],
    mlx_response_path: str | Path | None = None,
    top_k: int | None = None,
    top_fraction: float | None = None,
    min_pairs: int = 16,
) -> dict[str, Any]:
    """Rank pairs by scorer marginal and emit a runner-consumable hitlist."""

    if str(mlx_response.get("schema") or "") != "mlx_scorer_response.v1":
        raise ValueError("mlx_response must have schema 'mlx_scorer_response.v1'")
    artifacts = dict((mlx_response.get("components") or {}).get("artifacts") or {})
    pose_artifact = dict(artifacts.get("posenet_distortion") or {})
    seg_artifact = dict(artifacts.get("segnet_distortion") or {})
    pose_path = _required_path(pose_artifact, "posenet_distortion")
    seg_path = _required_path(seg_artifact, "segnet_distortion")
    pose = np.asarray(np.load(pose_path), dtype=np.float64)
    seg = np.asarray(np.load(seg_path), dtype=np.float64)
    if pose.ndim != 1 or seg.ndim != 1 or pose.shape != seg.shape:
        raise ValueError(
            "posenet_distortion and segnet_distortion must be 1-D arrays with "
            f"matching shape, got {pose.shape} and {seg.shape}"
        )
    pair_count = int(pose.shape[0])
    if pair_count <= 0:
        raise ValueError("component arrays are empty")
    requested_k = _resolve_top_k(
        pair_count=pair_count,
        top_k=top_k,
        top_fraction=top_fraction,
        min_pairs=min_pairs,
    )
    avg_pose = _finite_float(
        mlx_response.get("avg_posenet_dist"),
        default=float(np.mean(pose, dtype=np.float64)),
    )
    pose_marginal = 0.0 if avg_pose <= 0.0 else 5.0 / math.sqrt(10.0 * avg_pose)
    seg_marginal = 100.0
    score_marginal = seg_marginal * seg + pose_marginal * pose
    order = np.lexsort((np.arange(pair_count), -score_marginal))
    selected_local = [int(idx) for idx in order[:requested_k]]
    source_pair_frames = _source_pair_frames_for_response(
        mlx_response,
        expected_count=pair_count,
    )
    selected_pairs = [int(idx) for idx in selected_local]
    response_path = (
        Path(mlx_response_path).expanduser().resolve(strict=False)
        if mlx_response_path is not None
        else None
    )
    return {
        "schema": SCHEMA,
        "created_utc": datetime.now(UTC).isoformat(),
        "source_mlx_response_path": response_path.as_posix() if response_path else None,
        "source_mlx_response_sha256": _sha256_file(response_path)
        if response_path
        else None,
        "source_archive_sha256": mlx_response.get("archive_sha256"),
        "source_archive_bytes": mlx_response.get("archive_size_bytes"),
        "response_family": mlx_response.get("response_family"),
        "hardware_substrate": mlx_response.get("hardware_substrate"),
        "evidence_tag": mlx_response.get("evidence_tag"),
        "pair_count": pair_count,
        "top_k": requested_k,
        "top_fraction": float(requested_k) / float(pair_count),
        "ranking_formula": "100*segnet_distortion + 5/sqrt(10*avg_posenet_dist)*posenet_distortion",
        "avg_posenet_dist": avg_pose,
        "pose_marginal_weight": pose_marginal,
        "segnet_marginal_weight": seg_marginal,
        "pair_indices": selected_pairs,
        "ranked_pairs": [
            {
                "rank": rank + 1,
                "pair_index": int(local_idx),
                "local_row_index": int(local_idx),
                "source_frame_pair": (
                    [int(item) for item in source_pair_frames[local_idx].tolist()]
                    if source_pair_frames is not None
                    else None
                ),
                "score_marginal": float(score_marginal[local_idx]),
                "posenet_distortion": float(pose[local_idx]),
                "segnet_distortion": float(seg[local_idx]),
            }
            for rank, local_idx in enumerate(selected_local)
        ],
        "component_artifacts": {
            "posenet_distortion": _artifact_with_verified_sha(pose_path, pose_artifact),
            "segnet_distortion": _artifact_with_verified_sha(seg_path, seg_artifact),
        },
        "consumable_by": [
            "tools/run_compact_renderer_mlx_spine_runner.py --prioritized-pair-indices-file"
        ],
        "producer": "tac.adaptation.hard_pair_hitlist",
        **FALSE_AUTHORITY,
    }


def write_hard_pair_hitlist_from_mlx_response(
    *,
    mlx_response: Mapping[str, Any],
    output_json: str | Path,
    mlx_response_path: str | Path | None = None,
    top_k: int | None = None,
    top_fraction: float | None = None,
    min_pairs: int = 16,
) -> dict[str, Any]:
    payload = build_hard_pair_hitlist_from_mlx_response(
        mlx_response=mlx_response,
        mlx_response_path=mlx_response_path,
        top_k=top_k,
        top_fraction=top_fraction,
        min_pairs=min_pairs,
    )
    out = Path(output_json).expanduser().resolve(strict=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _resolve_top_k(
    *,
    pair_count: int,
    top_k: int | None,
    top_fraction: float | None,
    min_pairs: int,
) -> int:
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive")
    if top_fraction is not None and not (0.0 < float(top_fraction) <= 1.0):
        raise ValueError("top_fraction must be in (0, 1]")
    if top_k is not None:
        return min(pair_count, int(top_k))
    if top_fraction is not None:
        return min(pair_count, max(int(min_pairs), math.ceil(pair_count * float(top_fraction))))
    return min(pair_count, max(int(min_pairs), math.ceil(pair_count * 0.16)))


def _source_pair_frames_for_response(
    mlx_response: Mapping[str, Any],
    *,
    expected_count: int,
) -> np.ndarray | None:
    source_run = dict(mlx_response.get("source_cache_run") or {})
    candidate_cache_dir = source_run.get("candidate_cache_dir")
    if candidate_cache_dir:
        pair_path = Path(str(candidate_cache_dir)).expanduser() / "pair_indices.npy"
        if pair_path.is_file():
            pairs = np.asarray(np.load(pair_path))
            if pairs.shape[0] == expected_count:
                if pairs.ndim == 2:
                    return pairs.astype(np.int64, copy=False)
                if pairs.ndim == 1:
                    return pairs.reshape(-1, 1).astype(np.int64, copy=False)
    return None


def _required_path(artifact: Mapping[str, Any], name: str) -> Path:
    path = artifact.get("path")
    if not path:
        raise ValueError(f"mlx_response components.artifacts.{name}.path missing")
    resolved = Path(str(path)).expanduser().resolve(strict=False)
    if not resolved.is_file():
        raise FileNotFoundError(f"{name} artifact does not exist: {resolved}")
    return resolved


def _artifact_with_verified_sha(path: Path, artifact: Mapping[str, Any]) -> dict[str, Any]:
    actual_sha = _sha256_file(path)
    expected_sha = artifact.get("sha256")
    if expected_sha and actual_sha != expected_sha:
        raise ValueError(
            f"artifact sha256 mismatch for {path}: expected {expected_sha}, got {actual_sha}"
        )
    return {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": actual_sha,
    }


def _finite_float(value: Any, *, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        out = float(default)
    if not math.isfinite(out):
        return float(default)
    return out


def _sha256_file(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "SCHEMA",
    "build_hard_pair_hitlist_from_mlx_response",
    "write_hard_pair_hitlist_from_mlx_response",
]
