# SPDX-License-Identifier: MIT
"""Split non-authoritative MLX scorer-response payloads into window rows."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from tac.auth_eval_schema import contest_formula_score
from tac.local_acceleration.mlx_scorer_response import SCHEMA_VERSION

WINDOW_SPLIT_SCHEMA = "mlx_scorer_response_window_split.v1"


class MLXResponseWindowSplitError(ValueError):
    """Raised when an MLX scorer-response payload cannot be split safely."""


def split_mlx_scorer_response_windows(
    *,
    response_payload: dict[str, Any],
    posenet_distortion: np.ndarray,
    segnet_distortion: np.ndarray,
    output_dir: str | Path,
    window_pairs: int,
    stride_pairs: int | None = None,
    max_windows: int | None = None,
    prefix: str = "window",
    components_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Write per-window MLX scorer-response JSON files from one parent response."""

    _validate_parent_response(response_payload)
    window = _positive_int(window_pairs, "window_pairs")
    stride = window if stride_pairs is None else _positive_int(stride_pairs, "stride_pairs")
    if max_windows is not None:
        max_windows = _positive_int(max_windows, "max_windows")
    if not prefix:
        raise MLXResponseWindowSplitError("prefix must be non-empty")

    pose = np.asarray(posenet_distortion, dtype=np.float32)
    seg = np.asarray(segnet_distortion, dtype=np.float32)
    if pose.ndim != 1 or seg.ndim != 1:
        raise MLXResponseWindowSplitError("distortion arrays must be rank-1")
    if pose.shape != seg.shape:
        raise MLXResponseWindowSplitError(
            f"posenet/segnet distortion shape mismatch: {pose.shape} vs {seg.shape}"
        )
    n_samples = int(response_payload["n_samples"])
    if pose.shape[0] != n_samples:
        raise MLXResponseWindowSplitError(
            f"distortion length {pose.shape[0]} does not match n_samples {n_samples}"
        )
    pair_window = response_payload.get("pair_window")
    if not isinstance(pair_window, list) or len(pair_window) != 2:
        raise MLXResponseWindowSplitError("response pair_window must be a length-2 list")
    parent_start = int(pair_window[0])
    parent_stop = int(pair_window[1])
    if parent_stop - parent_start != n_samples:
        raise MLXResponseWindowSplitError("response pair_window does not match n_samples")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    component_root = None if components_dir is None else Path(components_dir)
    if component_root is not None:
        component_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    cursor = 0
    while cursor < n_samples:
        rel_stop = min(n_samples, cursor + window)
        if rel_stop <= cursor:
            break
        if max_windows is not None and len(rows) >= max_windows:
            break
        abs_start = parent_start + cursor
        abs_stop = parent_start + rel_stop
        pose_slice = np.ascontiguousarray(pose[cursor:rel_stop])
        seg_slice = np.ascontiguousarray(seg[cursor:rel_stop])
        child = _window_payload(
            response_payload=response_payload,
            pose_distortion=pose_slice,
            seg_distortion=seg_slice,
            abs_start=abs_start,
            abs_stop=abs_stop,
            components_dir=component_root,
            prefix=prefix,
        )
        child_path = out_dir / f"{prefix}_{abs_start:04d}_{abs_stop:04d}.json"
        child_path.write_text(json.dumps(_jsonable(child), indent=2, sort_keys=True) + "\n")
        rows.append(
            {
                "path": str(child_path),
                "pair_window": [abs_start, abs_stop],
                "n_samples": abs_stop - abs_start,
                "canonical_score": child["canonical_score"],
                "avg_posenet_dist": child["avg_posenet_dist"],
                "avg_segnet_dist": child["avg_segnet_dist"],
                "score_claim": False,
            }
        )
        cursor += stride

    return {
        "schema_version": WINDOW_SPLIT_SCHEMA,
        "source_schema_version": response_payload.get("schema_version"),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "candidate_generation_only": True,
        "parent_pair_window": [parent_start, parent_stop],
        "window_pairs": window,
        "stride_pairs": stride,
        "max_windows": max_windows,
        "window_count": len(rows),
        "rows": rows,
    }


def load_distortion_components_from_response(
    response_payload: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Load component arrays referenced by a scorer-response payload."""

    components = response_payload.get("components")
    if not isinstance(components, dict):
        raise MLXResponseWindowSplitError("response components must be an object")
    artifacts = components.get("artifacts")
    if not isinstance(artifacts, dict):
        raise MLXResponseWindowSplitError("response components.artifacts must be an object")
    pose_path = _artifact_path(artifacts, "posenet_distortion")
    seg_path = _artifact_path(artifacts, "segnet_distortion")
    return (
        np.load(pose_path).astype(np.float32, copy=False),
        np.load(seg_path).astype(np.float32, copy=False),
    )


def _window_payload(
    *,
    response_payload: dict[str, Any],
    pose_distortion: np.ndarray,
    seg_distortion: np.ndarray,
    abs_start: int,
    abs_stop: int,
    components_dir: Path | None,
    prefix: str,
) -> dict[str, Any]:
    child = copy.deepcopy(response_payload)
    pose_avg = float(np.mean(pose_distortion, dtype=np.float64))
    seg_avg = float(np.mean(seg_distortion, dtype=np.float64))
    archive_bytes = int(child["archive_size_bytes"])
    score = contest_formula_score(
        seg_dist=seg_avg,
        pose_dist=pose_avg,
        archive_bytes=archive_bytes,
    )
    child["canonical_score"] = score
    child["score_recomputed_from_components"] = score
    child["avg_posenet_dist"] = pose_avg
    child["avg_segnet_dist"] = seg_avg
    child["n_samples"] = int(abs_stop - abs_start)
    child["start_pair"] = int(abs_start)
    child["max_pairs"] = int(abs_stop - abs_start)
    child["pair_window"] = [int(abs_start), int(abs_stop)]
    child["elapsed_seconds"] = None
    parent_run_id = child.get("run_id") or "mlx_scorer_response"
    child["run_id"] = f"{parent_run_id}:window:{abs_start}:{abs_stop}"
    artifacts: dict[str, Any] = {}
    if components_dir is not None:
        win_dir = components_dir / f"{prefix}_{abs_start:04d}_{abs_stop:04d}"
        win_dir.mkdir(parents=True, exist_ok=True)
        pose_path = win_dir / "posenet_distortion.npy"
        seg_path = win_dir / "segnet_distortion.npy"
        np.save(pose_path, pose_distortion)
        np.save(seg_path, seg_distortion)
        artifacts = {
            "posenet_distortion": _artifact_record(pose_path),
            "segnet_distortion": _artifact_record(seg_path),
        }
    child["components"] = {
        "posenet_shape": list(pose_distortion.shape),
        "segnet_shape": list(seg_distortion.shape),
        "posenet_sha256": _array_sha256(pose_distortion),
        "segnet_sha256": _array_sha256(seg_distortion),
        "artifacts": artifacts,
    }
    child["window_split_source"] = {
        "schema_version": WINDOW_SPLIT_SCHEMA,
        "parent_pair_window": response_payload.get("pair_window"),
        "parent_n_samples": response_payload.get("n_samples"),
    }
    return child


def _validate_parent_response(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise MLXResponseWindowSplitError("response payload must be an object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise MLXResponseWindowSplitError(f"response schema_version must be {SCHEMA_VERSION}")
    for field in (
        "score_claim",
        "score_claim_valid",
        "promotion_eligible",
        "rank_or_kill_eligible",
        "ready_for_exact_eval_dispatch",
    ):
        if payload.get(field) is not False:
            raise MLXResponseWindowSplitError(f"response {field} must be false")
    if payload.get("candidate_generation_only") is not True:
        raise MLXResponseWindowSplitError("response candidate_generation_only must be true")
    if int(payload.get("batch_pairs", 0)) != 1:
        raise MLXResponseWindowSplitError("only singleton batch_pairs=1 responses may be split")
    for field in ("archive_size_bytes", "n_samples"):
        _positive_int(payload.get(field), field)


def _artifact_path(artifacts: dict[str, Any], key: str) -> Path:
    item = artifacts.get(key)
    if not isinstance(item, dict) or not item.get("path"):
        raise MLXResponseWindowSplitError(f"missing component artifact {key}")
    path = Path(str(item["path"]))
    if not path.exists():
        raise MLXResponseWindowSplitError(f"component artifact does not exist: {path}")
    return path


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise MLXResponseWindowSplitError(f"{label} must be an integer")
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise MLXResponseWindowSplitError(f"{label} must be an integer") from exc
    if out <= 0:
        raise MLXResponseWindowSplitError(f"{label} must be positive")
    return out


def _artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _array_sha256(arr: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(contiguous.dtype).encode("utf-8"))
    h.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode("utf-8"))
    h.update(contiguous.tobytes())
    return h.hexdigest()


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


__all__ = [
    "WINDOW_SPLIT_SCHEMA",
    "MLXResponseWindowSplitError",
    "load_distortion_components_from_response",
    "split_mlx_scorer_response_windows",
]
