# SPDX-License-Identifier: MIT
"""Patch pair-window MLX responses onto a full-video HPRC baseline."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from tac.auth_eval_schema import contest_formula_score
from tac.substrates.hprc.archive_candidate import FALSE_AUTHORITY

HPRC_INCREMENTAL_PAIR_RESPONSE_SCHEMA = "hprc_incremental_pair_response.v1"


def build_hprc_incremental_pair_response_report(
    *,
    profile_path: str | Path,
    candidate_variant_id: str,
    candidate_response_path: str | Path,
    candidate_cache_dir: str | Path,
    materialization_report_path: str | Path,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    """Return a full-video advisory report by patching changed pairs only."""

    root = Path(repo_root).expanduser().resolve(strict=False)
    profile_file = _resolve(profile_path, base=root)
    profile = _load_json_object(profile_file)
    baseline_variant = _variant_row(profile, "baseline")
    candidate_variant = _variant_row(profile, candidate_variant_id)
    baseline_response = _load_json_object(Path(str(baseline_variant["mlx_response"])))
    candidate_response = _load_json_object(_resolve(candidate_response_path, base=root))
    materialization_report = _load_json_object(
        _resolve(materialization_report_path, base=root)
    )
    candidate_cache = _resolve(candidate_cache_dir, base=root)
    pair_indices = np.load(candidate_cache / "pair_indices.npy")
    if pair_indices.ndim != 2 or pair_indices.shape[1] != 2:
        raise ValueError(f"candidate pair_indices must have shape (N,2): {pair_indices.shape}")
    changed_pair_rows = (pair_indices[:, 0] // 2).astype(np.int64)
    if not np.array_equal(pair_indices[:, 1], pair_indices[:, 0] + 1):
        raise ValueError("candidate pair_indices must be adjacent frame pairs")

    baseline_pose, baseline_seg = _component_arrays(baseline_response)
    candidate_pose, candidate_seg = _component_arrays(candidate_response)
    if len(changed_pair_rows) != len(candidate_pose) or len(changed_pair_rows) != len(candidate_seg):
        raise ValueError("candidate response component lengths do not match pair index count")
    if np.any(changed_pair_rows < 0) or np.any(changed_pair_rows >= len(baseline_pose)):
        raise ValueError("candidate changed pair rows exceed baseline component length")

    patched_pose = np.asarray(baseline_pose, dtype=np.float64).copy()
    patched_seg = np.asarray(baseline_seg, dtype=np.float64).copy()
    patched_pose[changed_pair_rows] = np.asarray(candidate_pose, dtype=np.float64)
    patched_seg[changed_pair_rows] = np.asarray(candidate_seg, dtype=np.float64)

    baseline_archive_bytes = int(baseline_variant["archive_zip_bytes"])
    candidate_archive_bytes = int(candidate_variant["archive_zip_bytes"])
    baseline_score = contest_formula_score(
        seg_dist=float(np.mean(baseline_seg, dtype=np.float64)),
        pose_dist=float(np.mean(baseline_pose, dtype=np.float64)),
        archive_bytes=baseline_archive_bytes,
    )
    patched_score = contest_formula_score(
        seg_dist=float(np.mean(patched_seg, dtype=np.float64)),
        pose_dist=float(np.mean(patched_pose, dtype=np.float64)),
        archive_bytes=candidate_archive_bytes,
    )
    return {
        "schema": HPRC_INCREMENTAL_PAIR_RESPONSE_SCHEMA,
        "generated_at_utc": _utc_stamp(),
        "repo_root": root.as_posix(),
        "source_profile_path": profile_file.as_posix(),
        "candidate_variant_id": candidate_variant_id,
        "candidate_response_path": _resolve(candidate_response_path, base=root).as_posix(),
        "candidate_cache_dir": candidate_cache.as_posix(),
        "materialization_report_path": _resolve(
            materialization_report_path,
            base=root,
        ).as_posix(),
        "materialization_pair_scope": {
            "selected_pair_count": int(materialization_report.get("cached_pair_count") or 0),
            "selected_pair_ranges": materialization_report.get(
                "hprc_direct_cache_report",
                {},
            ).get("selected_pair_ranges"),
            "pair_index_scope": materialization_report.get(
                "hprc_direct_cache_report",
                {},
            ).get("pair_index_scope"),
        },
        "changed_pair_rows": [int(value) for value in changed_pair_rows.tolist()],
        "full_video_pair_count": len(baseline_pose),
        "baseline_archive_bytes": baseline_archive_bytes,
        "candidate_archive_bytes": candidate_archive_bytes,
        "archive_bytes_removed_vs_baseline": baseline_archive_bytes - candidate_archive_bytes,
        "baseline_mlx_score_advisory_recomputed": baseline_score,
        "patched_full_video_mlx_score_advisory": patched_score,
        "delta_total_mlx_score_advisory": patched_score - baseline_score,
        "delta_avg_posenet_dist": float(np.mean(patched_pose) - np.mean(baseline_pose)),
        "delta_avg_segnet_dist": float(np.mean(patched_seg) - np.mean(baseline_seg)),
        "incremental_assumption": (
            "DistortionNet response is evaluated pair-locally, so unchanged pair "
            "component rows are safely reused from the singleton full-video baseline."
        ),
        "blockers": [
            "incremental_mlx_response_is_advisory_not_score_authority",
            "singleton_full_replay_required_before_promotion",
            "contest_cpu_cuda_exact_eval_not_executed",
        ],
        **FALSE_AUTHORITY,
    }


def write_hprc_incremental_pair_response_report(
    *,
    output_path: str | Path,
    report: dict[str, Any],
    allow_overwrite: bool = False,
) -> Path:
    """Write a deterministic incremental response report."""

    path = Path(output_path)
    if path.exists() and not allow_overwrite:
        raise FileExistsError(f"output exists; pass allow_overwrite=True: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return path


def _variant_row(profile: dict[str, Any], variant_id: str) -> dict[str, Any]:
    rows = profile.get("variant_rows")
    if not isinstance(rows, list):
        raise ValueError("profile missing variant_rows")
    matches = [
        row
        for row in rows
        if isinstance(row, dict) and row.get("variant_id") == variant_id
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one variant row for {variant_id!r}")
    return matches[0]


def _component_arrays(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    artifacts = payload.get("components", {}).get("artifacts", {})
    pose_path = artifacts.get("posenet_distortion", {}).get("path")
    seg_path = artifacts.get("segnet_distortion", {}).get("path")
    if not pose_path or not seg_path:
        raise ValueError("MLX response payload missing component artifact paths")
    return np.load(pose_path).astype(np.float32), np.load(seg_path).astype(np.float32)


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return payload


def _resolve(path: str | Path, *, base: Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (base / p).resolve(strict=False)


def _utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


__all__ = [
    "HPRC_INCREMENTAL_PAIR_RESPONSE_SCHEMA",
    "build_hprc_incremental_pair_response_report",
    "write_hprc_incremental_pair_response_report",
]
