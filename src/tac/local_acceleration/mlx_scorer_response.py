# SPDX-License-Identifier: MIT
"""Run MLX scorer responses from fixed scorer-input caches.

This module is intentionally non-authoritative.  It consumes NumPy scorer-input
caches whose identity can be audited against auth-eval provenance, runs the
local MLX PoseNet/SegNet adapters, and writes a JSON payload that downstream
fidelity gates can compare against contest CPU/CUDA evaluator outputs.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tac.auth_eval_schema import ORIGINAL_VIDEO_BYTES, contest_formula_score
from tac.local_acceleration import EVIDENCE_GRADE_MLX, EVIDENCE_TAG_MLX
from tac.local_acceleration.mlx_cache_audit import cache_audit_stamp_blockers
from tac.local_acceleration.mlx_scorer_adapters import (
    run_mlx_distortion_scorer_nchw,
    scorer_distortion_components_numpy,
    temporary_mlx_device,
    torch_distortion_net_to_mlx,
)

SCHEMA_VERSION = "mlx_scorer_response.v1"
GPU_RESEARCH_SIGNAL_BLOCKER = "mlx_gpu_scorer_response_requires_explicit_research_signal_allowance"
GPU_BATCH_SHAPE_BLOCKER = "mlx_gpu_scorer_response_requires_singleton_batches_until_invariance_passes"
BATCH_SHAPE_RESEARCH_SIGNAL_BLOCKER = (
    "mlx_scorer_response_non_singleton_batches_require_explicit_batch_shape_research_signal_allowance"
)
CANDIDATE_CACHE_TRANSFER_BLOCKER = (
    "mlx_scorer_response_candidate_cache_requires_passing_auth_identity_audit"
)
LOCAL_ADVISORY_CACHE_IDENTITY_BLOCKER = (
    "mlx_scorer_response_local_advisory_cache_identity_requires_explicit_allowance"
)
CACHE_INTEGRITY_BLOCKER = "mlx_scorer_input_cache_integrity_failed"
AUDIT_STAMP_DEREFERENCE_BLOCKER = "mlx_scorer_response_cache_audit_stamp_dereference_failed"


@dataclass(frozen=True)
class ScorerInputCache:
    """Loaded fixed scorer-input cache."""

    root: Path
    manifest: dict[str, Any]
    segnet_last_rgb: np.ndarray
    posenet_yuv6_pair: np.ndarray
    pair_indices: np.ndarray
    cache_integrity: dict[str, Any]


def load_scorer_input_cache(cache_dir: str | Path, *, mmap_mode: str | None = "r") -> ScorerInputCache:
    """Load a full tensor scorer-input cache and validate basic shape contracts."""

    root = Path(cache_dir)
    manifest_path = root / "manifest.json"
    manifest = _load_json_object(manifest_path)
    if manifest.get("hash_only") is True:
        raise ValueError(f"cache is hash-only and has no tensors: {manifest_path}")

    seg_path = root / "segnet_last_rgb.npy"
    pose_path = root / "posenet_yuv6_pair.npy"
    pair_path = root / "pair_indices.npy"
    missing = [str(path) for path in (seg_path, pose_path, pair_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing scorer-input cache arrays: {missing}")

    seg = np.load(seg_path, mmap_mode=mmap_mode)
    pose = np.load(pose_path, mmap_mode=mmap_mode)
    pairs = np.load(pair_path, mmap_mode=mmap_mode)
    _validate_cache_shapes(manifest, seg, pose, pairs)
    cache_integrity = _verify_cache_integrity(
        root=root,
        manifest=manifest,
        seg=seg,
        pose=pose,
        pairs=pairs,
    )
    return ScorerInputCache(
        root=root,
        manifest=manifest,
        segnet_last_rgb=seg,
        posenet_yuv6_pair=pose,
        pair_indices=pairs,
        cache_integrity=cache_integrity,
    )


def build_mlx_scorer_response_payload(
    *,
    reference_cache_dir: str | Path,
    candidate_cache_dir: str | Path,
    archive_size_bytes: int,
    repo_root: str | Path = ".",
    batch_pairs: int = 1,
    device_type: str = "cpu",
    components_dir: str | Path | None = None,
    progress_every: int = 0,
    start_pair: int = 0,
    max_pairs: int | None = None,
    allow_gpu_research_signal: bool = False,
    allow_batch_shape_research_signal: bool = False,
    allow_unaudited_candidate_cache_debug: bool = False,
    allow_local_cpu_advisory_cache_identity: bool = False,
    response_family: str | None = None,
) -> dict[str, Any]:
    """Run MLX scorer responses for reference/candidate caches and summarize metrics."""

    if int(archive_size_bytes) < 0:
        raise ValueError(f"archive_size_bytes must be non-negative, got {archive_size_bytes}")
    batch_pairs_int = int(batch_pairs)
    if batch_pairs_int < 1:
        raise ValueError(f"batch_pairs must be >= 1, got {batch_pairs}")
    if device_type not in {"cpu", "gpu"}:
        raise ValueError(f"device_type must be 'cpu' or 'gpu', got {device_type!r}")
    if device_type == "gpu" and not allow_gpu_research_signal:
        raise ValueError(
            f"{GPU_RESEARCH_SIGNAL_BLOCKER}: device_type='gpu' is local MLX "
            "prescreen signal only; pass allow_gpu_research_signal=True after "
            "recording CPU-transfer calibration or a research-only rationale"
        )
    if batch_pairs_int != 1 and not allow_batch_shape_research_signal:
        raise ValueError(
            f"{BATCH_SHAPE_RESEARCH_SIGNAL_BLOCKER}: clean-head FEC6 parity "
            "found batch-shape-sensitive SegNet argmax drift for MLX scorer "
            "responses; use batch_pairs=1 for production local signal or pass "
            "allow_batch_shape_research_signal=True only for explicitly "
            "recorded batch-shape research probes"
        )
    if int(progress_every) < 0:
        raise ValueError(f"progress_every must be >= 0, got {progress_every}")
    if int(start_pair) < 0:
        raise ValueError(f"start_pair must be >= 0, got {start_pair}")
    if max_pairs is not None and int(max_pairs) < 1:
        raise ValueError(f"max_pairs must be >= 1 when set, got {max_pairs}")
    family = _normalize_response_family(response_family)

    reference = load_scorer_input_cache(reference_cache_dir)
    candidate = load_scorer_input_cache(candidate_cache_dir)
    _validate_cache_pairing(reference, candidate)
    candidate_cache_identity_mode = _validate_candidate_transfer_cache(
        candidate,
        allow_unaudited_candidate_cache_debug=allow_unaudited_candidate_cache_debug,
        allow_local_cpu_advisory_cache_identity=allow_local_cpu_advisory_cache_identity,
    )

    started = time.time()
    dist = _load_upstream_distortion_net(Path(repo_root).resolve())
    pose_chunks: list[np.ndarray] = []
    seg_chunks: list[np.ndarray] = []
    total_pair_count = int(reference.pair_indices.shape[0])
    start = int(start_pair)
    if start >= total_pair_count:
        raise ValueError(f"start_pair {start} is outside cache pair count {total_pair_count}")
    stop_exclusive = total_pair_count if max_pairs is None else min(total_pair_count, start + int(max_pairs))
    pair_count = stop_exclusive - start

    with temporary_mlx_device(device_type):
        adapter = torch_distortion_net_to_mlx(dist)
        for batch_index, batch_start in enumerate(
            range(start, stop_exclusive, int(batch_pairs)),
            start=1,
        ):
            stop = min(stop_exclusive, batch_start + int(batch_pairs))
            ref_pose = np.asarray(reference.posenet_yuv6_pair[batch_start:stop], dtype=np.float32)
            ref_seg = np.asarray(reference.segnet_last_rgb[batch_start:stop], dtype=np.float32)
            cand_pose = np.asarray(candidate.posenet_yuv6_pair[batch_start:stop], dtype=np.float32)
            cand_seg = np.asarray(candidate.segnet_last_rgb[batch_start:stop], dtype=np.float32)
            ref_outputs = run_mlx_distortion_scorer_nchw(
                adapter,
                ref_pose,
                ref_seg,
            )
            if np.array_equal(ref_pose, cand_pose) and np.array_equal(ref_seg, cand_seg):
                cand_outputs = ref_outputs
            else:
                cand_outputs = run_mlx_distortion_scorer_nchw(
                    adapter,
                    cand_pose,
                    cand_seg,
                )
            components = scorer_distortion_components_numpy(ref_outputs, cand_outputs)
            pose_chunks.append(components["posenet"])
            seg_chunks.append(components["segnet"])
            if progress_every and batch_index % int(progress_every) == 0:
                elapsed = time.time() - started
                done = stop - start
                rate = done / elapsed if elapsed > 0 else 0.0
                print(
                    json.dumps(
                        {
                            "event": "mlx_scorer_response_progress",
                            "done_pairs": done,
                            "total_pairs": pair_count,
                            "pairs_per_second": rate,
                            "elapsed_seconds": elapsed,
                        },
                        sort_keys=True,
                    ),
                    file=sys.stderr,
                    flush=True,
                )

    pose_distortion = np.concatenate(pose_chunks).astype(np.float32, copy=False)
    seg_distortion = np.concatenate(seg_chunks).astype(np.float32, copy=False)
    pose_avg = float(np.mean(pose_distortion, dtype=np.float64))
    seg_avg = float(np.mean(seg_distortion, dtype=np.float64))
    archive_bytes = int(archive_size_bytes)
    rate_unscaled = archive_bytes / ORIGINAL_VIDEO_BYTES
    rate_contribution = 25.0 * rate_unscaled
    score = contest_formula_score(
        seg_dist=seg_avg,
        pose_dist=pose_avg,
        archive_bytes=archive_bytes,
    )

    artifacts = _write_component_artifacts(
        components_dir,
        pose_distortion=pose_distortion,
        seg_distortion=seg_distortion,
    )
    elapsed = time.time() - started
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_grade": EVIDENCE_GRADE_MLX,
        "evidence_tag": EVIDENCE_TAG_MLX,
        "score_axis": EVIDENCE_TAG_MLX,
        "response_family": family,
        "hardware_substrate": f"MLX {device_type}",
        "gpu_research_signal_allowed": bool(allow_gpu_research_signal),
        "batch_shape_research_signal_allowed": bool(allow_batch_shape_research_signal),
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "promotable": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "candidate_generation_only": True,
        "requires_exact_eval_before_promotion": True,
        "canonical_score": score,
        "score_recomputed_from_components": score,
        "canonical_score_source": "score_recomputed_from_components",
        "avg_posenet_dist": pose_avg,
        "avg_segnet_dist": seg_avg,
        "archive_size_bytes": archive_bytes,
        "rate_unscaled": rate_unscaled,
        "score_rate_contribution": rate_contribution,
        "n_samples": pair_count,
        "total_cache_pairs": total_pair_count,
        "start_pair": start,
        "max_pairs": None if max_pairs is None else int(max_pairs),
        "pair_window": [start, stop_exclusive],
        "batch_pairs": int(batch_pairs),
        "elapsed_seconds": elapsed,
        "components": {
            "posenet_shape": list(pose_distortion.shape),
            "segnet_shape": list(seg_distortion.shape),
            "posenet_sha256": _array_sha256(pose_distortion),
            "segnet_sha256": _array_sha256(seg_distortion),
            "artifacts": artifacts,
        },
        "cache_identity": {
            "reference": _cache_identity(reference),
            "candidate": _cache_identity(candidate),
            "pair_indices_equal": True,
        },
        "cache_integrity": {
            "reference": reference.cache_integrity,
            "candidate": candidate.cache_integrity,
        },
        "archive_sha256": _manifest_string(candidate.manifest, "archive_sha256"),
        "inflated_outputs_aggregate_sha256": _manifest_string(
            candidate.manifest,
            "inflated_outputs_aggregate_sha256",
        ),
        "raw_sha256": _manifest_string(candidate.manifest, "raw_sha256"),
        "device_contract": {
            "gpu_research_signal_blocker": GPU_RESEARCH_SIGNAL_BLOCKER,
            "gpu_batch_shape_blocker": GPU_BATCH_SHAPE_BLOCKER,
            "batch_shape_research_signal_blocker": BATCH_SHAPE_RESEARCH_SIGNAL_BLOCKER,
            "gpu_research_signal_required": device_type == "gpu",
            "batch_shape_research_signal_required": batch_pairs_int != 1,
            "unaudited_candidate_cache_debug_allowed": bool(
                allow_unaudited_candidate_cache_debug
            ),
            "local_cpu_advisory_cache_identity_allowed": bool(
                allow_local_cpu_advisory_cache_identity
            ),
            "candidate_cache_identity_mode": candidate_cache_identity_mode,
            "candidate_cache_transfer_blocker": CANDIDATE_CACHE_TRANSFER_BLOCKER,
            "local_advisory_cache_identity_blocker": LOCAL_ADVISORY_CACHE_IDENTITY_BLOCKER,
            "gpu_batch_pairs_allowed_without_invariance_override": 1,
            "batch_pairs_allowed_without_invariance_override": 1,
            "allowed_uses": _allowed_uses_for_device_contract(
                candidate_cache_identity_mode=candidate_cache_identity_mode,
                device_type=device_type,
                batch_pairs=batch_pairs_int,
            ),
            "forbidden_uses": [
                "auth_eval",
                "score_claim",
                "promotion",
                "rank_or_kill",
                "leaderboard_claim",
                "replacement_for_cuda_t4_or_linux_x86_64_eval",
            ],
        },
    }


def _allowed_uses_for_device_contract(
    *,
    candidate_cache_identity_mode: str,
    device_type: str,
    batch_pairs: int,
) -> list[str]:
    if candidate_cache_identity_mode == "local_cpu_advisory_identity":
        return [
            "local_mlx_debug_against_matching_local_cpu_advisory_raw",
            "local_speed_quality_delta_measurement",
        ]
    if candidate_cache_identity_mode == "unaudited_debug_override":
        return [
            "local_tensor_ingestion_debug_only",
            "non_authoritative_component_smoke",
        ]
    if device_type == "gpu" or batch_pairs != 1:
        return [
            "local_mlx_research_signal_after_cpu_anchor",
            "batch_shape_or_device_drift_probe",
            "candidate_generation_prior",
        ]
    return [
        "local_mlx_training_gradient_shaping",
        "local_sweep_reranking_after_transfer_and_score_calibration",
        "candidate_generation_prior",
        "signal_exposure",
        "prepaid_dispatch_spend_filter_after_score_calibration",
    ]


def write_mlx_scorer_response_payload(payload: dict[str, Any], output: str | Path) -> None:
    """Write a scorer-response JSON payload."""

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_upstream_distortion_net(repo_root: Path) -> Any:
    upstream_dir = repo_root / "upstream"
    if not upstream_dir.is_dir():
        raise FileNotFoundError(f"missing upstream directory: {upstream_dir}")
    old_path = list(sys.path)
    try:
        if str(upstream_dir) not in sys.path:
            sys.path.insert(0, str(upstream_dir))
        import modules  # type: ignore[import-not-found]
        import torch

        dist = modules.DistortionNet().eval()
        dist.load_state_dicts(
            modules.posenet_sd_path,
            modules.segnet_sd_path,
            torch.device("cpu"),
        )
        return dist.eval()
    finally:
        sys.path[:] = old_path


def _validate_cache_shapes(
    manifest: dict[str, Any],
    seg: np.ndarray,
    pose: np.ndarray,
    pairs: np.ndarray,
) -> None:
    if seg.ndim != 4 or pose.ndim != 4 or pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError(
            "expected segnet_last_rgb rank-4, posenet_yuv6_pair rank-4, "
            f"and pair_indices shape (N, 2); got {seg.shape}, {pose.shape}, {pairs.shape}"
        )
    if seg.shape[0] != pose.shape[0] or seg.shape[0] != pairs.shape[0]:
        raise ValueError(
            "cache pair counts disagree: "
            f"seg={seg.shape[0]} pose={pose.shape[0]} pair_indices={pairs.shape[0]}"
        )
    for manifest_key, actual_shape in (
        ("segnet_last_rgb_shape", seg.shape),
        ("posenet_yuv6_pair_shape", pose.shape),
        ("pair_indices_shape", pairs.shape),
    ):
        expected_shape = manifest.get(manifest_key)
        if expected_shape is not None and list(actual_shape) != list(expected_shape):
            raise ValueError(
                f"{manifest_key} mismatch: manifest={expected_shape} actual={list(actual_shape)}"
            )


def _verify_cache_integrity(
    *,
    root: Path,
    manifest: dict[str, Any],
    seg: np.ndarray,
    pose: np.ndarray,
    pairs: np.ndarray,
) -> dict[str, Any]:
    blockers: list[str] = []
    array_actual = {
        "segnet_last_rgb": _array_sha256(seg),
        "posenet_yuv6_pair": _array_sha256(pose),
        "pair_indices": _array_sha256(pairs),
    }
    artifact_paths = {
        "segnet_last_rgb": root / "segnet_last_rgb.npy",
        "posenet_yuv6_pair": root / "posenet_yuv6_pair.npy",
        "pair_indices": root / "pair_indices.npy",
    }
    artifact_actual = {
        key: {
            "bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        for key, path in artifact_paths.items()
    }

    expected_array = manifest.get("array_sha256")
    if not isinstance(expected_array, dict):
        blockers.append("array_sha256_missing")
        expected_array = {}
    for key, actual in array_actual.items():
        expected = expected_array.get(key)
        if expected is None:
            blockers.append(f"array_sha256_{key}_missing")
        elif expected != actual:
            blockers.append(f"array_sha256_{key}_mismatch")

    expected_artifacts = manifest.get("artifacts")
    if not isinstance(expected_artifacts, dict):
        blockers.append("artifacts_missing")
        expected_artifacts = {}
    for key, actual in artifact_actual.items():
        expected = expected_artifacts.get(key)
        if not isinstance(expected, dict):
            blockers.append(f"artifact_{key}_missing")
            continue
        if expected.get("bytes") != actual["bytes"]:
            blockers.append(f"artifact_{key}_bytes_mismatch")
        if expected.get("sha256") != actual["sha256"]:
            blockers.append(f"artifact_{key}_sha256_mismatch")

    hash_domain = manifest.get("hash_domain")
    if not isinstance(hash_domain, str) or not hash_domain:
        blockers.append("hash_domain_missing")

    result = {
        "passed": not blockers,
        "blockers": blockers,
        "hash_domain": hash_domain,
        "array_sha256": array_actual,
        "artifact_sha256": {
            key: value["sha256"] for key, value in artifact_actual.items()
        },
        "artifact_bytes": {
            key: value["bytes"] for key, value in artifact_actual.items()
        },
    }
    if blockers:
        raise ValueError(f"{CACHE_INTEGRITY_BLOCKER}: {blockers}")
    return result


def _validate_cache_pairing(reference: ScorerInputCache, candidate: ScorerInputCache) -> None:
    if reference.segnet_last_rgb.shape != candidate.segnet_last_rgb.shape:
        raise ValueError(
            "reference/candidate segnet shape mismatch: "
            f"{reference.segnet_last_rgb.shape} vs {candidate.segnet_last_rgb.shape}"
        )
    if reference.posenet_yuv6_pair.shape != candidate.posenet_yuv6_pair.shape:
        raise ValueError(
            "reference/candidate posenet shape mismatch: "
            f"{reference.posenet_yuv6_pair.shape} vs {candidate.posenet_yuv6_pair.shape}"
        )
    if not np.array_equal(reference.pair_indices, candidate.pair_indices):
        raise ValueError("reference/candidate pair_indices differ")


def _validate_candidate_transfer_cache(
    candidate: ScorerInputCache,
    *,
    allow_unaudited_candidate_cache_debug: bool,
    allow_local_cpu_advisory_cache_identity: bool,
) -> str:
    manifest = candidate.manifest
    audit = manifest.get("auth_eval_identity_audit")
    audit_blockers = cache_audit_stamp_blockers(
        manifest,
        cache_root=candidate.root,
        stamp_key="auth_eval_identity_audit",
        expected_verdict="PASS_CACHE_AUTH_EVAL_IDENTITY",
        require_identity_residual_zero=True,
        require_cache_shapes=True,
    )
    audit_ok = (
        manifest.get("eligible_for_local_mlx_transfer_calibration") is True
        and isinstance(audit, dict)
        and audit.get("verdict") == "PASS_CACHE_AUTH_EVAL_IDENTITY"
        and audit.get("passed") is True
        and audit.get("identity_residual") == 0
        and not audit_blockers
    )
    if audit_ok:
        return "auth_eval_identity"
    local_audit = manifest.get("local_cpu_advisory_cache_identity_audit")
    local_audit_blockers = cache_audit_stamp_blockers(
        manifest,
        cache_root=candidate.root,
        stamp_key="local_cpu_advisory_cache_identity_audit",
        expected_verdict="PASS_CACHE_LOCAL_CPU_ADVISORY_IDENTITY",
        require_identity_residual_zero=False,
        require_cache_shapes=False,
    )
    local_audit_ok = (
        allow_local_cpu_advisory_cache_identity
        and manifest.get("eligible_for_local_mlx_local_advisory_debug") is True
        and isinstance(local_audit, dict)
        and local_audit.get("verdict") == "PASS_CACHE_LOCAL_CPU_ADVISORY_IDENTITY"
        and local_audit.get("passed") is True
        and not local_audit_blockers
    )
    if local_audit_ok:
        return "local_cpu_advisory_identity"
    if allow_unaudited_candidate_cache_debug:
        return "unaudited_debug_override"
    if (
        manifest.get("eligible_for_local_mlx_local_advisory_debug") is True
        and isinstance(local_audit, dict)
        and local_audit.get("verdict") == "PASS_CACHE_LOCAL_CPU_ADVISORY_IDENTITY"
        and local_audit.get("passed") is True
    ):
        detail = (
            f"; {AUDIT_STAMP_DEREFERENCE_BLOCKER}: {', '.join(local_audit_blockers)}"
            if local_audit_blockers
            else ""
        )
        raise ValueError(
            f"{LOCAL_ADVISORY_CACHE_IDENTITY_BLOCKER}: candidate cache has only "
            "local CPU-advisory identity, not contest auth-axis identity; pass "
            "allow_local_cpu_advisory_cache_identity=True for local debug/speed "
            f"delta runs{detail}"
        )
    detail = (
        f"; {AUDIT_STAMP_DEREFERENCE_BLOCKER}: {', '.join(audit_blockers)}"
        if isinstance(audit, dict) and audit_blockers
        else ""
    )
    raise ValueError(
        f"{CANDIDATE_CACHE_TRANSFER_BLOCKER}: candidate cache must be stamped by "
        "tools/materialize_mlx_scorer_cache_from_auth_eval.py or an equivalent "
        f"PASS_CACHE_AUTH_EVAL_IDENTITY audit before response scoring{detail}"
    )


def _write_component_artifacts(
    components_dir: str | Path | None,
    *,
    pose_distortion: np.ndarray,
    seg_distortion: np.ndarray,
) -> dict[str, Any]:
    if components_dir is None:
        return {}
    out = Path(components_dir)
    out.mkdir(parents=True, exist_ok=True)
    pose_path = out / "posenet_distortion.npy"
    seg_path = out / "segnet_distortion.npy"
    np.save(pose_path, pose_distortion)
    np.save(seg_path, seg_distortion)
    return {
        "posenet_distortion": _artifact_record(pose_path),
        "segnet_distortion": _artifact_record(seg_path),
    }


def _cache_identity(cache: ScorerInputCache) -> dict[str, Any]:
    manifest = cache.manifest
    return {
        "path": str(cache.root),
        "archive_sha256": _manifest_string(manifest, "archive_sha256"),
        "inflated_outputs_aggregate_sha256": _manifest_string(
            manifest,
            "inflated_outputs_aggregate_sha256",
        ),
        "raw_sha256": _manifest_string(manifest, "raw_sha256"),
        "array_sha256": manifest.get("array_sha256"),
        "hash_domain": manifest.get("hash_domain"),
        "segnet_last_rgb_shape": manifest.get("segnet_last_rgb_shape"),
        "posenet_yuv6_pair_shape": manifest.get("posenet_yuv6_pair_shape"),
        "pair_indices_shape": manifest.get("pair_indices_shape"),
        "pair_count": int(cache.pair_indices.shape[0]),
        "eligible_for_local_mlx_transfer_calibration": bool(
            manifest.get("eligible_for_local_mlx_transfer_calibration")
        ),
        "auth_eval_identity_audit": manifest.get("auth_eval_identity_audit"),
        "cache_integrity": cache.cache_integrity,
    }


def _artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


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


def _manifest_string(manifest: dict[str, Any], key: str) -> str | None:
    value = manifest.get(key)
    return value if isinstance(value, str) and value else None


def _normalize_response_family(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        raise ValueError("response_family must be non-empty when provided")
    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789_-.")
    if any(ch not in allowed for ch in text):
        raise ValueError(
            "response_family may contain only lowercase letters, digits, underscore, dash, or dot"
        )
    return text


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
    "CACHE_INTEGRITY_BLOCKER",
    "CANDIDATE_CACHE_TRANSFER_BLOCKER",
    "GPU_BATCH_SHAPE_BLOCKER",
    "GPU_RESEARCH_SIGNAL_BLOCKER",
    "SCHEMA_VERSION",
    "ScorerInputCache",
    "build_mlx_scorer_response_payload",
    "load_scorer_input_cache",
    "write_mlx_scorer_response_payload",
]
