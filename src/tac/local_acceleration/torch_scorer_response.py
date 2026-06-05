# SPDX-License-Identifier: MIT
"""PyTorch CPU scorer responses from fixed scorer-input caches.

This mirrors the MLX scorer-response cache path but keeps the upstream Torch
DistortionNet on CPU.  Its purpose is local drift triage: when a compact
renderer collapses in scorer space, this tells us whether MLX is lying or the
candidate tensors are already bad under the upstream implementation.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from tac.auth_eval_schema import ORIGINAL_VIDEO_BYTES, contest_formula_score
from tac.local_acceleration.mlx_scorer_adapters import (
    scorer_distortion_components_numpy,
)
from tac.local_acceleration.mlx_scorer_response import (
    LOCAL_ADVISORY_CACHE_IDENTITY_BLOCKER,
    MANIFEST_CACHE_INTEGRITY_MODE,
    VALID_CACHE_INTEGRITY_MODES,
    _array_sha256,
    _build_cache_pairing_plan,
    _cache_identity,
    _jsonable,
    _load_upstream_distortion_net,
    _manifest_string,
    _normalize_response_family,
    _resolve_upstream_dir,
    _validate_candidate_transfer_cache,
    _write_component_artifacts,
    load_scorer_input_cache,
)
from tac.local_acceleration.mlx_scorer_torch_parity import (
    run_torch_distortion_scorer_nchw,
)

SCHEMA_VERSION = "torch_cpu_scorer_response.v1"
EVIDENCE_TAG_TORCH_CPU = "[macOS-PyTorch-CPU advisory]"


def build_torch_cpu_scorer_response_payload(
    *,
    reference_cache_dir: str | Path,
    candidate_cache_dir: str | Path,
    archive_size_bytes: int,
    repo_root: str | Path = ".",
    upstream_dir: str | Path | None = None,
    batch_pairs: int = 1,
    start_pair: int = 0,
    max_pairs: int | None = None,
    components_dir: str | Path | None = None,
    progress_every: int = 0,
    response_family: str | None = None,
    allow_unaudited_candidate_cache_debug: bool = False,
    allow_local_cpu_advisory_cache_identity: bool = False,
    cache_integrity_mode: str = MANIFEST_CACHE_INTEGRITY_MODE,
) -> dict[str, Any]:
    """Run upstream PyTorch CPU scorer response on aligned cache rows."""

    if int(archive_size_bytes) < 0:
        raise ValueError(f"archive_size_bytes must be non-negative, got {archive_size_bytes}")
    batch_pairs_int = int(batch_pairs)
    if batch_pairs_int < 1:
        raise ValueError(f"batch_pairs must be >= 1, got {batch_pairs}")
    if int(start_pair) < 0:
        raise ValueError(f"start_pair must be >= 0, got {start_pair}")
    if max_pairs is not None and int(max_pairs) < 1:
        raise ValueError(f"max_pairs must be >= 1 when set, got {max_pairs}")
    if int(progress_every) < 0:
        raise ValueError(f"progress_every must be >= 0, got {progress_every}")
    if cache_integrity_mode not in VALID_CACHE_INTEGRITY_MODES:
        raise ValueError(
            "cache_integrity_mode must be one of "
            f"{sorted(VALID_CACHE_INTEGRITY_MODES)}, got {cache_integrity_mode!r}"
        )

    started = time.time()
    family = _normalize_response_family(response_family)
    reference = load_scorer_input_cache(
        reference_cache_dir,
        integrity_mode=cache_integrity_mode,
    )
    candidate = load_scorer_input_cache(
        candidate_cache_dir,
        integrity_mode=cache_integrity_mode,
    )
    pairing_plan = _build_cache_pairing_plan(reference, candidate)
    candidate_cache_identity_mode = _validate_candidate_transfer_cache(
        candidate,
        allow_unaudited_candidate_cache_debug=allow_unaudited_candidate_cache_debug,
        allow_local_cpu_advisory_cache_identity=allow_local_cpu_advisory_cache_identity,
    )
    scorer_upstream_dir = _resolve_upstream_dir(
        Path(repo_root).resolve(),
        upstream_dir=upstream_dir,
    )
    dist = _load_upstream_distortion_net(scorer_upstream_dir)

    total_pair_count = int(candidate.pair_indices.shape[0])
    start = int(start_pair)
    if start >= total_pair_count:
        raise ValueError(f"start_pair {start} is outside cache pair count {total_pair_count}")
    stop_exclusive = (
        total_pair_count if max_pairs is None else min(total_pair_count, start + int(max_pairs))
    )
    pair_count = stop_exclusive - start
    pose_chunks: list[np.ndarray] = []
    seg_chunks: list[np.ndarray] = []
    for batch_index, batch_start in enumerate(
        range(start, stop_exclusive, batch_pairs_int),
        start=1,
    ):
        stop = min(stop_exclusive, batch_start + batch_pairs_int)
        ref_rows = pairing_plan.reference_row_indices[batch_start:stop]
        ref_pose = np.asarray(reference.posenet_yuv6_pair[ref_rows], dtype=np.float32)
        ref_seg = np.asarray(reference.segnet_last_rgb[ref_rows], dtype=np.float32)
        cand_pose = np.asarray(
            candidate.posenet_yuv6_pair[batch_start:stop],
            dtype=np.float32,
        )
        cand_seg = np.asarray(
            candidate.segnet_last_rgb[batch_start:stop],
            dtype=np.float32,
        )
        ref_outputs = run_torch_distortion_scorer_nchw(dist, ref_pose, ref_seg)
        if np.array_equal(ref_pose, cand_pose) and np.array_equal(ref_seg, cand_seg):
            cand_outputs = ref_outputs
        else:
            cand_outputs = run_torch_distortion_scorer_nchw(dist, cand_pose, cand_seg)
        components = scorer_distortion_components_numpy(ref_outputs, cand_outputs)
        pose_chunks.append(components["posenet"])
        seg_chunks.append(components["segnet"])
        if progress_every and batch_index % int(progress_every) == 0:
            elapsed = time.time() - started
            done = stop - start
            rate = done / elapsed if elapsed > 0.0 else 0.0
            print(
                json.dumps(
                    {
                        "event": "torch_cpu_scorer_response_progress",
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
    candidate_cache_identity = _cache_identity(candidate)
    candidate_cache_identity["candidate_cache_identity_mode"] = candidate_cache_identity_mode
    return {
        "schema": SCHEMA_VERSION,
        "schema_version": SCHEMA_VERSION,
        "evidence_tag": EVIDENCE_TAG_TORCH_CPU,
        "score_axis": EVIDENCE_TAG_TORCH_CPU,
        "response_family": family,
        "hardware_substrate": "PyTorch CPU",
        "scorer_upstream": {
            "schema": "torch_cpu_scorer_response_upstream_snapshot.v1",
            "upstream_dir": scorer_upstream_dir.as_posix(),
        },
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
        "score_rate_contribution": 25.0 * rate_unscaled,
        "n_samples": pair_count,
        "total_cache_pairs": total_pair_count,
        "candidate_cache_pairs": total_pair_count,
        "reference_cache_pairs": int(reference.pair_indices.shape[0]),
        "start_pair": start,
        "max_pairs": None if max_pairs is None else int(max_pairs),
        "pair_window": [start, stop_exclusive],
        "source_pair_window": [
            candidate.pair_indices[start].tolist(),
            candidate.pair_indices[stop_exclusive - 1].tolist(),
        ],
        "batch_pairs": batch_pairs_int,
        "elapsed_seconds": time.time() - started,
        "components": {
            "posenet_shape": list(pose_distortion.shape),
            "segnet_shape": list(seg_distortion.shape),
            "posenet_sha256": _array_sha256(pose_distortion),
            "segnet_sha256": _array_sha256(seg_distortion),
            "artifacts": artifacts,
        },
        "cache_identity": {
            "reference": _cache_identity(reference),
            "candidate": candidate_cache_identity,
            "pair_indices_equal": pairing_plan.pair_indices_equal,
            "pair_index_alignment_mode": pairing_plan.alignment_mode,
            "reference_row_indices_sha256": _array_sha256(
                pairing_plan.reference_row_indices.astype(np.int64, copy=False)
            ),
            "reference_row_window": [
                int(pairing_plan.reference_row_indices[start]),
                int(pairing_plan.reference_row_indices[stop_exclusive - 1]),
            ],
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
            "candidate_cache_identity_mode": candidate_cache_identity_mode,
            "local_advisory_cache_identity_blocker": LOCAL_ADVISORY_CACHE_IDENTITY_BLOCKER,
            "unaudited_candidate_cache_debug_allowed": bool(
                allow_unaudited_candidate_cache_debug
            ),
            "local_cpu_advisory_cache_identity_allowed": bool(
                allow_local_cpu_advisory_cache_identity
            ),
            "allowed_uses": [
                "local_torch_cpu_debug_against_matching_cache",
                "mlx_drift_triage",
                "non_authoritative_component_smoke",
            ],
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


def write_torch_cpu_scorer_response_payload(
    payload: dict[str, Any],
    output: str | Path,
) -> None:
    """Write a Torch CPU scorer-response payload with stable formatting."""

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "EVIDENCE_TAG_TORCH_CPU",
    "SCHEMA_VERSION",
    "build_torch_cpu_scorer_response_payload",
    "write_torch_cpu_scorer_response_payload",
]
