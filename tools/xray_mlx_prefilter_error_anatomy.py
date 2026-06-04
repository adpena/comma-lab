#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Analyze MLX prefilter scorer artifacts by pair, tensor, raw, and byte axis.

This is a diagnostic bridge from compact-carrier receiver proofs into the XRay
and hard-pair tooling. It consumes an existing ``mlx_scorer_response.v1`` style
profile plus retained scorer-input caches, emits per-pair component tails and
optional scorer-input pixel/tensor deltas, and records a fail-closed direct-VJP
work order. It does not score, promote, dispatch, or claim auth-eval authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

SCHEMA = "mlx_prefilter_error_anatomy.v1"
TOOL = "tools/xray_mlx_prefilter_error_anatomy.py"
CONTEST_RATE_NORMALIZER_BYTES = 37_545_489
FALSE_AUTHORITY: dict[str, bool] = {
    "research_only": True,
    "score_claim": False,
    "score_claim_valid": False,
    "dispatch_attempted": False,
    "promotion_eligible": False,
    "promotable": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}
SCORER_CACHE_ARRAYS = ("pair_indices", "posenet_yuv6_pair", "segnet_last_rgb")


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_finite_float(value: Any, *, label: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{label} is not finite")
    return out


def _artifact_path(record: Any) -> Path | None:
    if not isinstance(record, Mapping):
        return None
    value = record.get("path")
    if not isinstance(value, str) or not value:
        return None
    path = Path(value).expanduser().resolve(strict=False)
    return path if path.is_file() else None


def _cache_root(cache: Any) -> Path | None:
    if not isinstance(cache, Mapping):
        return None
    value = cache.get("path")
    if not isinstance(value, str) or not value:
        return None
    path = Path(value).expanduser().resolve(strict=False)
    return path if path.is_dir() else None


def _profile_cache(profile: Mapping[str, Any], side: str) -> dict[str, Any] | None:
    out: dict[str, Any] | None = None
    cache_identity = profile.get("cache_identity")
    if isinstance(cache_identity, Mapping) and isinstance(cache_identity.get(side), dict):
        out = dict(cache_identity[side])
    prefilter = profile.get("hinerv_receiver_raw_cache_prefilter")
    if isinstance(prefilter, Mapping):
        key = f"{side}_cache_manifest"
        if isinstance(prefilter.get(key), dict):
            manifest = dict(prefilter[key])
            root = _infer_cache_root_from_manifest(manifest)
            if root is not None:
                manifest.setdefault("path", root.as_posix())
            if out is None:
                return manifest
            merged = dict(manifest)
            merged.update(out)
            for rich_key in ("artifacts", "path"):
                if rich_key in manifest and rich_key not in out:
                    merged[rich_key] = manifest[rich_key]
            return merged
    return out


def _infer_cache_root_from_manifest(manifest: Mapping[str, Any]) -> Path | None:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        return None
    for key in SCORER_CACHE_ARRAYS:
        path = _artifact_path(artifacts.get(key))
        if path is not None:
            return path.parent
    return None


def _cache_array_path(cache: Mapping[str, Any] | None, key: str) -> Path | None:
    if cache is None:
        return None
    artifacts = cache.get("artifacts")
    if isinstance(artifacts, Mapping):
        path = _artifact_path(artifacts.get(key))
        if path is not None:
            return path
    root = _cache_root(cache)
    if root is not None:
        path = root / f"{key}.npy"
        if path.is_file():
            return path
    return None


def _cache_has_hot_arrays(cache: Mapping[str, Any] | None) -> bool:
    return all(_cache_array_path(cache, key) is not None for key in SCORER_CACHE_ARRAYS)


def _load_cache_manifest_dir(cache_dir: str | Path) -> dict[str, Any]:
    root = Path(cache_dir).expanduser().resolve(strict=False)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"cache override missing manifest.json: {manifest_path}")
    manifest = _load_json_object(manifest_path)
    manifest.setdefault("path", root.as_posix())
    return manifest


def _regenerate_hot_cache_if_needed(
    *,
    profile_cache: dict[str, Any] | None,
    side: str,
    output_root: Path | None,
    batch_pairs: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if profile_cache is None:
        return None, None
    if _cache_has_hot_arrays(profile_cache):
        return profile_cache, {
            "side": side,
            "action": "reused_existing_hot_cache",
            "cache_dir": str(profile_cache.get("path")),
        }
    if output_root is None:
        return profile_cache, {
            "side": side,
            "action": "not_regenerated",
            "blockers": [f"{side}_hot_cache_missing_and_no_regeneration_root"],
        }
    source_value = profile_cache.get("source")
    if not isinstance(source_value, str) or not source_value:
        return profile_cache, {
            "side": side,
            "action": "not_regenerated",
            "blockers": [f"{side}_hot_cache_source_missing_for_regeneration"],
        }
    source = Path(source_value).expanduser().resolve(strict=False)
    if not source.exists():
        return profile_cache, {
            "side": side,
            "action": "not_regenerated",
            "blockers": [f"{side}_hot_cache_source_missing_on_disk:{source.as_posix()}"],
        }

    out_dir = output_root.expanduser().resolve(strict=False) / f"{side}_regenerated_scorer_cache"
    manifest_path = out_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = _load_cache_manifest_dir(out_dir)
        if _cache_has_hot_arrays(manifest):
            return manifest, {
                "side": side,
                "action": "reused_regenerated_hot_cache",
                "cache_dir": out_dir.as_posix(),
                "manifest": manifest_path.as_posix(),
            }
        existing = [
            (out_dir / f"{key}.npy").as_posix()
            for key in SCORER_CACHE_ARRAYS
            if (out_dir / f"{key}.npy").exists()
        ]
        raise ValueError(
            "refusing partial regenerated scorer cache with manifest but missing arrays: "
            + ", ".join(existing)
        )

    from tac.local_acceleration.mlx_preprocess import (
        write_scorer_input_cache_from_raw_file,
        write_scorer_input_cache_from_video_file,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    source_kind = str(profile_cache.get("source_kind") or "")
    common = {
        "archive_sha256": profile_cache.get("archive_sha256")
        if isinstance(profile_cache.get("archive_sha256"), str)
        else None,
        "inflated_outputs_aggregate_sha256": profile_cache.get("inflated_outputs_aggregate_sha256")
        if isinstance(profile_cache.get("inflated_outputs_aggregate_sha256"), str)
        else None,
        "max_pairs": int(profile_cache.get("pair_count"))
        if isinstance(profile_cache.get("pair_count"), int)
        else None,
        "batch_pairs": int(batch_pairs),
    }
    if source_kind == "raw" or source.suffix.lower() == ".raw":
        manifest = write_scorer_input_cache_from_raw_file(source, out_dir, **common)
        normalized_kind = "raw"
    else:
        manifest = write_scorer_input_cache_from_video_file(source, out_dir, **common)
        normalized_kind = source_kind or "video"
    manifest.setdefault("path", out_dir.as_posix())
    return manifest, {
        "schema": "xray_regenerated_hot_scorer_cache.v1",
        "side": side,
        "action": "regenerated_hot_cache_from_profile_source",
        "source": source.as_posix(),
        "source_kind": normalized_kind,
        "cache_dir": out_dir.as_posix(),
        "manifest": (out_dir / "manifest.json").as_posix(),
        "batch_pairs": int(batch_pairs),
        "array_sha256": dict(manifest.get("array_sha256") or {}),
        "artifacts": {
            key: _artifact_record(_cache_array_path(manifest, key))
            for key in SCORER_CACHE_ARRAYS
        },
        **FALSE_AUTHORITY,
    }


def _load_component_arrays(profile: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    artifacts = profile.get("components", {}).get("artifacts") if isinstance(profile.get("components"), Mapping) else None
    if not isinstance(artifacts, Mapping):
        raise ValueError("profile missing components.artifacts")
    pose_path = _artifact_path(artifacts.get("posenet_distortion"))
    seg_path = _artifact_path(artifacts.get("segnet_distortion"))
    if pose_path is None or seg_path is None:
        raise ValueError("profile missing retained posenet_distortion/segnet_distortion arrays")
    pose = np.load(pose_path, mmap_mode="r")
    seg = np.load(seg_path, mmap_mode="r")
    if pose.ndim != 1 or seg.ndim != 1 or pose.shape != seg.shape:
        raise ValueError(f"component arrays must be matching rank-1 arrays, got {pose.shape} and {seg.shape}")
    if pose.shape[0] < 1:
        raise ValueError("component arrays are empty")
    meta = {
        "posenet_distortion": _artifact_record(pose_path),
        "segnet_distortion": _artifact_record(seg_path),
    }
    return pose, seg, meta


def _artifact_record(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return {
        "path": path.as_posix(),
        "bytes": path.stat().st_size if path.exists() else None,
        "sha256": _sha256_file(path) if path.is_file() else None,
    }


def _quantiles(values: np.ndarray, qs: tuple[float, ...] = (0.0, 0.5, 0.9, 0.95, 0.99, 1.0)) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {f"q{int(q * 100):02d}": _as_finite_float(np.quantile(arr, q), label=f"quantile_{q}") for q in qs}


def _pair_indices_for_rows(candidate_cache: Mapping[str, Any] | None, n_pairs: int) -> np.ndarray:
    path = _cache_array_path(candidate_cache, "pair_indices")
    if path is None:
        return np.stack(
            [
                np.arange(0, n_pairs * 2, 2, dtype=np.int64),
                np.arange(1, n_pairs * 2 + 1, 2, dtype=np.int64),
            ],
            axis=1,
        )
    arr = np.load(path, mmap_mode="r")
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < n_pairs:
        raise ValueError(f"pair_indices must have shape (>=n_pairs,2), got {arr.shape}")
    return np.asarray(arr[:n_pairs], dtype=np.int64)


def _component_rows(
    *,
    pose_dist: np.ndarray,
    seg_dist: np.ndarray,
    pair_indices: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx in range(int(pose_dist.shape[0])):
        pose_value = _as_finite_float(pose_dist[idx], label=f"pose_dist[{idx}]")
        seg_value = _as_finite_float(seg_dist[idx], label=f"seg_dist[{idx}]")
        pose_term = math.sqrt(max(0.0, 10.0 * pose_value))
        seg_term = 100.0 * seg_value
        rows.append(
            {
                "pair_idx": int(idx),
                "frame_indices": [int(pair_indices[idx, 0]), int(pair_indices[idx, 1])],
                "pose_dist": pose_value,
                "seg_dist": seg_value,
                "pose_score_contribution": pose_term,
                "seg_score_contribution": seg_term,
                "component_score_no_rate": pose_term + seg_term,
            }
        )
    return rows


def _axis_tuple(rank: int) -> tuple[int, ...]:
    return tuple(range(1, rank))


def _per_pair_delta_stats(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    n_pairs: int,
    chunk_pairs: int,
    changed_eps: float,
) -> dict[str, np.ndarray]:
    if candidate.shape[0] < n_pairs or reference.shape[0] < n_pairs:
        raise ValueError(
            f"cache arrays shorter than component pairs: {candidate.shape[0]} and {reference.shape[0]} vs {n_pairs}"
        )
    if candidate.shape[1:] != reference.shape[1:]:
        raise ValueError(f"cache array shape mismatch: {candidate.shape[1:]} vs {reference.shape[1:]}")
    mean_abs = np.empty(n_pairs, dtype=np.float32)
    rmse = np.empty(n_pairs, dtype=np.float32)
    changed = np.empty(n_pairs, dtype=np.float32)
    max_abs = np.empty(n_pairs, dtype=np.float32)
    axes = _axis_tuple(candidate.ndim)
    for start in range(0, n_pairs, chunk_pairs):
        stop = min(n_pairs, start + chunk_pairs)
        cand = np.asarray(candidate[start:stop], dtype=np.float32)
        ref = np.asarray(reference[start:stop], dtype=np.float32)
        diff = cand - ref
        abs_diff = np.abs(diff)
        mean_abs[start:stop] = np.mean(abs_diff, axis=axes)
        rmse[start:stop] = np.sqrt(np.mean(diff * diff, axis=axes))
        changed[start:stop] = np.mean(abs_diff > changed_eps, axis=axes)
        max_abs[start:stop] = np.max(abs_diff, axis=axes)
    return {
        "mean_abs": mean_abs,
        "rmse": rmse,
        "changed_fraction": changed,
        "max_abs": max_abs,
    }


def _attach_pixel_xray(
    *,
    rows: list[dict[str, Any]],
    candidate_cache: Mapping[str, Any] | None,
    reference_cache: Mapping[str, Any] | None,
    chunk_pairs: int,
    changed_eps: float,
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    stats_by_name: dict[str, dict[str, np.ndarray]] = {}
    for key, row_prefix in (("segnet_last_rgb", "segnet_cache"), ("posenet_yuv6_pair", "posenet_cache")):
        candidate_path = _cache_array_path(candidate_cache, key)
        reference_path = _cache_array_path(reference_cache, key)
        if candidate_path is None:
            blockers.append(f"candidate_{key}_array_missing_for_pixel_xray")
            continue
        if reference_path is None:
            blockers.append(f"reference_{key}_array_missing_for_pixel_xray")
            continue
        candidate = np.load(candidate_path, mmap_mode="r")
        reference = np.load(reference_path, mmap_mode="r")
        stats = _per_pair_delta_stats(
            candidate,
            reference,
            n_pairs=len(rows),
            chunk_pairs=chunk_pairs,
            changed_eps=changed_eps,
        )
        stats_by_name[row_prefix] = stats
        for idx, row in enumerate(rows):
            row[f"{row_prefix}_mean_abs_delta"] = _as_finite_float(stats["mean_abs"][idx], label=f"{key}.mean_abs[{idx}]")
            row[f"{row_prefix}_rmse_delta"] = _as_finite_float(stats["rmse"][idx], label=f"{key}.rmse[{idx}]")
            row[f"{row_prefix}_changed_fraction"] = _as_finite_float(
                stats["changed_fraction"][idx],
                label=f"{key}.changed_fraction[{idx}]",
            )
            row[f"{row_prefix}_max_abs_delta"] = _as_finite_float(stats["max_abs"][idx], label=f"{key}.max_abs[{idx}]")
    summary: dict[str, Any] = {}
    for row_prefix, stats in stats_by_name.items():
        summary[row_prefix] = {
            metric: {
                "mean": _as_finite_float(np.mean(values), label=f"{row_prefix}.{metric}.mean"),
                "max": _as_finite_float(np.max(values), label=f"{row_prefix}.{metric}.max"),
                "quantiles": _quantiles(values),
            }
            for metric, values in stats.items()
        }
    return summary, blockers


def _top_rows(rows: list[dict[str, Any]], field: str, top_k: int) -> list[dict[str, Any]]:
    if top_k <= 0:
        return []
    return list(
        sorted(
            rows,
            key=lambda item: float(item.get(field, 0.0)),
            reverse=True,
        )[:top_k]
    )


def _cache_summary(cache: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if cache is None:
        return None
    artifacts: dict[str, Any] = {}
    for key in SCORER_CACHE_ARRAYS:
        artifacts[key] = _artifact_record(_cache_array_path(cache, key))
    return {
        "path": str(cache.get("path")) if cache.get("path") is not None else None,
        "pair_count": cache.get("pair_count"),
        "pair_indices_shape": cache.get("pair_indices_shape"),
        "posenet_yuv6_pair_shape": cache.get("posenet_yuv6_pair_shape"),
        "segnet_last_rgb_shape": cache.get("segnet_last_rgb_shape"),
        "array_sha256": dict(cache.get("array_sha256") or {}),
        "artifacts": artifacts,
    }


def _direct_vjp_work_order(
    *,
    report_path: Path | None,
    candidate_cache: Mapping[str, Any] | None,
    reference_cache: Mapping[str, Any] | None,
    rows: list[dict[str, Any]],
    component_summary: Mapping[str, Any],
    archive: Path | None,
) -> dict[str, Any]:
    blockers: list[str] = []
    for side, cache in (("candidate", candidate_cache), ("reference", reference_cache)):
        for key in SCORER_CACHE_ARRAYS:
            if _cache_array_path(cache, key) is None:
                blockers.append(f"{side}_{key}_array_missing_for_direct_full_scorer_vjp")
    ready = not blockers
    commands: dict[str, str] = {}
    if ready:
        output_root = (
            str(report_path.parent / "direct_full_scorer_vjp")
            if report_path is not None
            else "<xray-output-dir>/direct_full_scorer_vjp"
        )
        common_args = [
            "uv",
            "run",
            "python",
            "tools/run_full_scorer_vjp_from_scorer_cache.py",
            "--candidate-cache-dir",
            str(_cache_root(candidate_cache) or candidate_cache.get("path")),
            "--reference-cache-dir",
            str(_cache_root(reference_cache) or reference_cache.get("path")),
            "--pair-window-json",
            str(report_path) if report_path is not None else "<this-xray-report.json>",
            "--shard-pairs",
            "4",
        ]
        if archive is not None:
            common_args.extend(["--archive", str(archive)])
        acquisitions = {
            "mlx_pose_p19_full_video": [
                *common_args,
                "--output-dir",
                f"{output_root}/mlx_pose_p19_full_video",
                "--backend",
                "mlx",
                "--device-type",
                "auto",
                "--seg-ce-weight",
                "0",
            ],
            "torch_joint_p18_p19_full_video": [
                *common_args,
                "--output-dir",
                f"{output_root}/torch_joint_p18_p19_full_video",
                "--backend",
                "torch",
                "--torch-device",
                "cpu",
            ],
        }
        commands = {name: " ".join(shlex.quote(item) for item in args) for name, args in acquisitions.items()}
    return {
        "schema": "direct_full_scorer_vjp_work_order.v1",
        "required": True,
        "reason": "component_error_is_broad_and_distortion_axis_blocks_score_lowering",
        "authority_contract": {
            "budget_spend_allowed_before_full_reduction": False,
            "reduction": "exact_full_video_all_pairs_accumulated_before_update",
            "backend_contract": "numpy_cache_spine_with_mlx_first_and_torch_fallback",
            "score_claim": False,
        },
        "acquisition_lanes": {
            "mlx_pose_p19_full_video": {
                "status": "preferred_parity_safe_for_pose",
                "math": "P19 pairwise pose Fisher/VJP only; SegNet branch disabled so MLX SegNet-gradient drift cannot poison pose.",
                "backend": "mlx",
                "device_type": "auto",
                "seg_ce_weight": 0.0,
            },
            "torch_joint_p18_p19_full_video": {
                "status": "verified_fallback_for_joint_surface",
                "math": "P18 SegNet boundary surrogate plus P19 pose VJP from the same NumPy scorer-input cache spine.",
                "backend": "torch",
                "torch_device": "cpu",
            },
        },
        "ready_for_vjp_materialization": ready,
        "blockers": blockers,
        "recommended_pair_priority": [int(row["pair_idx"]) for row in _top_rows(rows, "component_score_no_rate", 32)],
        "component_summary": dict(component_summary),
        "next_command": commands.get("torch_joint_p18_p19_full_video") if commands else None,
        "next_commands": commands,
    }


def _analysis_dimensions() -> list[dict[str, Any]]:
    return [
        {"axis": "archive_byte", "unit": "archive.zip byte", "optimization": "fixed contest price 25/N per byte"},
        {"axis": "payload_bitstream", "unit": "section/stream/bitplane", "optimization": "coder and quantizer choice per section"},
        {"axis": "decoder_weight", "unit": "tensor/channel/group", "optimization": "QAT, saliency, int8/int4/int2/zero waterfill"},
        {"axis": "latent_or_selector", "unit": "pair/token/class selector", "optimization": "keep only measured value-per-byte"},
        {"axis": "frame", "unit": "frame0/frame1", "optimization": "SegNet sees last frame; PoseNet sees both"},
        {"axis": "pair", "unit": "non-overlapping frame pair", "optimization": "hard-pair curriculum and exact VJP shard priority"},
        {"axis": "scorer_input_tensor", "unit": "SegNet RGB / PoseNet YUV6", "optimization": "direct scorer-space error and gradient"},
        {"axis": "raw_pixel", "unit": "camera RGB pixel", "optimization": "receiver-output locality, boundary repair, sparse residuals"},
        {"axis": "seg_class_boundary", "unit": "class/margin/boundary pixel", "optimization": "argmax-flip risk P18"},
        {"axis": "pose_axis_nullspace", "unit": "six pose axes / Mahalanobis null subset", "optimization": "true P19 protection"},
        {"axis": "time_curriculum", "unit": "epoch/stage/checkpoint", "optimization": "score-aware long-run control and hard-pair replay"},
        {"axis": "backend_drift", "unit": "MLX/Torch/CPU/CUDA", "optimization": "parity gates before authority transfer"},
        {"axis": "provenance_cleanup", "unit": "hash/path/env/argv", "optimization": "rebuildable artifact hygiene without signal loss"},
    ]


def build_report_from_profile(
    *,
    mlx_profile: str | Path,
    archive: str | Path | None = None,
    candidate_cache_dir: str | Path | None = None,
    reference_cache_dir: str | Path | None = None,
    regenerate_missing_hot_caches_root: str | Path | None = None,
    regenerate_batch_pairs: int = 4,
    label: str | None = None,
    top_k: int = 32,
    compute_pixel_xray: bool = True,
    pixel_chunk_pairs: int = 8,
    changed_eps: float = 0.0,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    profile_path = Path(mlx_profile).expanduser().resolve(strict=False)
    profile = _load_json_object(profile_path)
    archive_path = Path(archive).expanduser().resolve(strict=False) if archive is not None else None
    pose_dist, seg_dist, component_artifacts = _load_component_arrays(profile)
    candidate_cache = (
        _load_cache_manifest_dir(candidate_cache_dir)
        if candidate_cache_dir is not None
        else _profile_cache(profile, "candidate")
    )
    reference_cache = (
        _load_cache_manifest_dir(reference_cache_dir)
        if reference_cache_dir is not None
        else _profile_cache(profile, "reference")
    )
    hot_cache_root = (
        Path(regenerate_missing_hot_caches_root).expanduser().resolve(strict=False)
        if regenerate_missing_hot_caches_root is not None
        else None
    )
    hot_cache_records: list[dict[str, Any]] = []
    if candidate_cache_dir is None:
        candidate_cache, record = _regenerate_hot_cache_if_needed(
            profile_cache=candidate_cache,
            side="candidate",
            output_root=hot_cache_root,
            batch_pairs=regenerate_batch_pairs,
        )
        if record is not None:
            hot_cache_records.append(record)
    else:
        hot_cache_records.append(
            {
                "side": "candidate",
                "action": "explicit_hot_cache_override",
                "cache_dir": str(candidate_cache_dir),
            }
        )
    if reference_cache_dir is None:
        reference_cache, record = _regenerate_hot_cache_if_needed(
            profile_cache=reference_cache,
            side="reference",
            output_root=hot_cache_root,
            batch_pairs=regenerate_batch_pairs,
        )
        if record is not None:
            hot_cache_records.append(record)
    else:
        hot_cache_records.append(
            {
                "side": "reference",
                "action": "explicit_hot_cache_override",
                "cache_dir": str(reference_cache_dir),
            }
        )
    pair_indices = _pair_indices_for_rows(candidate_cache, int(pose_dist.shape[0]))
    rows = _component_rows(pose_dist=pose_dist, seg_dist=seg_dist, pair_indices=pair_indices)

    blockers = ["mlx_prefilter_error_anatomy_is_false_authority"]
    pixel_summary: dict[str, Any] = {"computed": False}
    pixel_blockers: list[str] = []
    if compute_pixel_xray:
        pixel_metrics, pixel_blockers = _attach_pixel_xray(
            rows=rows,
            candidate_cache=candidate_cache,
            reference_cache=reference_cache,
            chunk_pairs=pixel_chunk_pairs,
            changed_eps=changed_eps,
        )
        pixel_summary = {"computed": not pixel_blockers, "metrics": pixel_metrics, "blockers": pixel_blockers}
        blockers.extend(pixel_blockers)
    else:
        pixel_summary = {"computed": False, "blockers": ["pixel_xray_disabled_by_operator"]}

    pose_values = np.asarray(pose_dist, dtype=np.float64)
    seg_values = np.asarray(seg_dist, dtype=np.float64)
    pose_terms = np.sqrt(np.maximum(0.0, 10.0 * pose_values))
    seg_terms = 100.0 * seg_values
    component_terms = pose_terms + seg_terms
    component_summary = {
        "avg_posenet_dist": _as_finite_float(np.mean(pose_values), label="avg_posenet_dist"),
        "avg_segnet_dist": _as_finite_float(np.mean(seg_values), label="avg_segnet_dist"),
        "pose_score_contribution": _as_finite_float(np.mean(pose_terms), label="pose_score_contribution"),
        "seg_score_contribution": _as_finite_float(np.mean(seg_terms), label="seg_score_contribution"),
        "component_score_no_rate": _as_finite_float(np.mean(component_terms), label="component_score_no_rate"),
        "posenet_dist_quantiles": _quantiles(pose_values),
        "segnet_dist_quantiles": _quantiles(seg_values),
        "component_score_quantiles": _quantiles(component_terms),
        "n_pairs": len(rows),
    }
    report_output_path = Path(report_path).expanduser().resolve(strict=False) if report_path else None
    report = {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "producer": TOOL,
        "label": label or profile_path.stem,
        "mlx_profile": {
            "path": profile_path.as_posix(),
            "sha256": _sha256_file(profile_path),
            "schema": profile.get("schema") or profile.get("schema_version"),
            "score_axis": profile.get("score_axis"),
            "evidence_grade": profile.get("evidence_grade"),
            "evidence_tag": profile.get("evidence_tag"),
        },
        "archive": _archive_record(archive_path, profile),
        "receiver_raw": _receiver_raw_record(profile),
        "component_artifacts": component_artifacts,
        "cache_identity": {
            "candidate": _cache_summary(candidate_cache),
            "reference": _cache_summary(reference_cache),
            "pair_indices_equal": (profile.get("cache_identity") or {}).get("pair_indices_equal")
            if isinstance(profile.get("cache_identity"), Mapping)
            else None,
        },
        "hot_cache_recovery": {
            "records": hot_cache_records,
            "regeneration_root": hot_cache_root.as_posix() if hot_cache_root is not None else None,
            "regenerate_batch_pairs": int(regenerate_batch_pairs),
            **FALSE_AUTHORITY,
        },
        "component_summary": component_summary,
        "pixel_summary": pixel_summary,
        "top_pairs": {
            "combined": _top_rows(rows, "component_score_no_rate", top_k),
            "pose": _top_rows(rows, "pose_score_contribution", top_k),
            "segnet": _top_rows(rows, "seg_score_contribution", top_k),
            "segnet_cache_delta": _top_rows(rows, "segnet_cache_mean_abs_delta", top_k),
            "posenet_cache_delta": _top_rows(rows, "posenet_cache_mean_abs_delta", top_k),
        },
        "rows": rows,
        "analysis_dimensions": _analysis_dimensions(),
        "blockers": _ordered_unique(blockers),
        **FALSE_AUTHORITY,
    }
    report["direct_full_scorer_vjp_work_order"] = _direct_vjp_work_order(
        report_path=report_output_path,
        candidate_cache=candidate_cache,
        reference_cache=reference_cache,
        rows=rows,
        component_summary=component_summary,
        archive=archive_path,
    )
    return report


def _archive_record(path: Path | None, profile: Mapping[str, Any]) -> dict[str, Any] | None:
    if path is None:
        size = profile.get("archive_size_bytes")
        sha = profile.get("archive_sha256")
        return {
            "path": None,
            "bytes": int(size) if isinstance(size, int) else None,
            "sha256": sha if isinstance(sha, str) else None,
            "rate_term": (25.0 * float(size) / CONTEST_RATE_NORMALIZER_BYTES) if isinstance(size, int) else None,
        }
    bytes_value = path.stat().st_size if path.exists() else None
    return {
        "path": path.as_posix(),
        "bytes": bytes_value,
        "sha256": _sha256_file(path) if path.is_file() else None,
        "rate_term": (25.0 * float(bytes_value) / CONTEST_RATE_NORMALIZER_BYTES) if bytes_value is not None else None,
    }


def _receiver_raw_record(profile: Mapping[str, Any]) -> dict[str, Any] | None:
    prefilter = profile.get("hinerv_receiver_raw_cache_prefilter")
    value = prefilter.get("receiver_output_path") if isinstance(prefilter, Mapping) else None
    path = Path(str(value)).expanduser().resolve(strict=False) if isinstance(value, str) and value else None
    return {
        "path": path.as_posix() if path is not None else None,
        "bytes": path.stat().st_size if path is not None and path.is_file() else None,
        "sha256": profile.get("raw_sha256") if isinstance(profile.get("raw_sha256"), str) else None,
    }


def _ordered_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out


def write_report_outputs(report: dict[str, Any], output_dir: str | Path) -> dict[str, Path]:
    out = Path(output_dir).expanduser().resolve(strict=False)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "mlx_prefilter_error_anatomy.json"
    jsonl_path = out / "mlx_prefilter_error_anatomy.rows.jsonl"
    md_path = out / "mlx_prefilter_error_anatomy.md"
    report["report_path"] = json_path.as_posix()
    if isinstance(report.get("direct_full_scorer_vjp_work_order"), dict):
        report["direct_full_scorer_vjp_work_order"]["source_xray_report"] = json_path.as_posix()
        for key in ("next_command",):
            command = report["direct_full_scorer_vjp_work_order"].get(key)
            if isinstance(command, str):
                report["direct_full_scorer_vjp_work_order"][key] = command.replace("<this-xray-report.json>", json_path.as_posix())
        next_commands = report["direct_full_scorer_vjp_work_order"].get("next_commands")
        if isinstance(next_commands, dict):
            report["direct_full_scorer_vjp_work_order"]["next_commands"] = {
                name: command.replace("<this-xray-report.json>", json_path.as_posix())
                if isinstance(command, str)
                else command
                for name, command in next_commands.items()
            }
    _write_json(json_path, report)
    _write_jsonl(jsonl_path, report["rows"])
    md_path.write_text(_markdown_summary(report), encoding="utf-8")
    return {"json": json_path, "jsonl": jsonl_path, "markdown": md_path}


def _markdown_summary(report: Mapping[str, Any]) -> str:
    component = report["component_summary"]
    archive = report.get("archive") or {}
    vjp = report.get("direct_full_scorer_vjp_work_order") or {}
    lines = [
        f"# MLX Prefilter Error Anatomy: {report.get('label')}",
        "",
        "Diagnostic only: no score claim, promotion, rank, kill, or dispatch authority.",
        "",
        "## Summary",
        f"- Pairs: `{component.get('n_pairs')}`",
        f"- Avg SegNet distortion: `{component.get('avg_segnet_dist')}`",
        f"- Avg PoseNet distortion: `{component.get('avg_posenet_dist')}`",
        f"- Mean component score without rate: `{component.get('component_score_no_rate')}`",
        f"- Archive bytes: `{archive.get('bytes')}`",
        f"- Archive rate term: `{archive.get('rate_term')}`",
        f"- Direct full-scorer VJP ready: `{vjp.get('ready_for_vjp_materialization')}`",
        f"- Blockers: `{', '.join(str(item) for item in report.get('blockers', []))}`",
        "",
        "## Top Combined Pairs",
        "| pair | frames | component | seg | pose | seg_cache_mae | pose_cache_mae |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in (report.get("top_pairs", {}).get("combined") or [])[:16]:
        lines.append(
            "| {pair} | {frames} | {component:.6g} | {seg:.6g} | {pose:.6g} | {seg_mae:.6g} | {pose_mae:.6g} |".format(
                pair=int(row["pair_idx"]),
                frames=",".join(str(v) for v in row.get("frame_indices", [])),
                component=float(row.get("component_score_no_rate", 0.0)),
                seg=float(row.get("seg_dist", 0.0)),
                pose=float(row.get("pose_dist", 0.0)),
                seg_mae=float(row.get("segnet_cache_mean_abs_delta", 0.0)),
                pose_mae=float(row.get("posenet_cache_mean_abs_delta", 0.0)),
            )
        )
    lines.extend(
        [
            "",
            "## Direct VJP Work Order",
            "```json",
            json.dumps(_jsonable(vjp), indent=2, sort_keys=True),
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-profile", required=True, type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--candidate-cache-dir", type=Path)
    parser.add_argument("--reference-cache-dir", type=Path)
    parser.add_argument(
        "--regenerate-missing-hot-caches-root",
        type=Path,
        help=(
            "SSD/root directory where missing cleaned scorer cache tensors may be "
            "regenerated from profile manifest sources. Existing valid caches are reused."
        ),
    )
    parser.add_argument("--regenerate-batch-pairs", type=int, default=4)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--label")
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--pixel-chunk-pairs", type=int, default=8)
    parser.add_argument("--changed-eps", type=float, default=0.0)
    parser.add_argument("--skip-pixel-xray", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.top_k < 0:
        raise SystemExit("--top-k must be >= 0")
    if args.pixel_chunk_pairs < 1:
        raise SystemExit("--pixel-chunk-pairs must be >= 1")
    if args.regenerate_batch_pairs < 1:
        raise SystemExit("--regenerate-batch-pairs must be >= 1")
    if args.changed_eps < 0.0:
        raise SystemExit("--changed-eps must be >= 0")
    json_path = args.output_dir.expanduser().resolve(strict=False) / "mlx_prefilter_error_anatomy.json"
    report = build_report_from_profile(
        mlx_profile=args.mlx_profile,
        archive=args.archive,
        candidate_cache_dir=args.candidate_cache_dir,
        reference_cache_dir=args.reference_cache_dir,
        regenerate_missing_hot_caches_root=args.regenerate_missing_hot_caches_root,
        regenerate_batch_pairs=args.regenerate_batch_pairs,
        label=args.label,
        top_k=args.top_k,
        compute_pixel_xray=not args.skip_pixel_xray,
        pixel_chunk_pairs=args.pixel_chunk_pairs,
        changed_eps=args.changed_eps,
        report_path=json_path,
    )
    outputs = write_report_outputs(report, args.output_dir)
    print(json.dumps({key: path.as_posix() for key, path in outputs.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
