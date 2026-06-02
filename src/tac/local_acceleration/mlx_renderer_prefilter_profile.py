# SPDX-License-Identifier: MIT
"""MLX renderer scorer-prefilter profiles for archive-runnable carriers.

This module turns an in-memory MLX renderer into the existing HPRC MLX
prefilter dialect. It is deliberately a local research-signal surface: it runs
the MLX scorer mirror against decoded renderer frames, records deterministic
component summaries and output hashes, and leaves contest CPU/CUDA exact eval as
the only score authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from tac.auth_eval_schema import ORIGINAL_VIDEO_BYTES, contest_formula_score
from tac.local_acceleration import EVIDENCE_GRADE_MLX, EVIDENCE_TAG_MLX
from tac.substrates.hprc.mlx_prefilter_coverage import (
    HPRC_MLX_COMPONENT_PROFILE_SCHEMA,
)
from tac.substrates.hprc.resolution_contract import CONTEST_PAIR_COUNT

FALSE_AUTHORITY: dict[str, bool] = {
    "score_claim": False,
    "score_claim_valid": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "rank_or_kill_eligible": False,
    "promotable": False,
}


def write_mlx_renderer_prefilter_profile(
    *,
    bundle: Any,
    output_path: str | Path,
    archive_bytes: int,
    archive_sha256: str,
    upstream_dir: str | Path,
    scorer_device: str = "cpu",
    scorer_batch_pairs: int = 1,
    required_pairs: int = CONTEST_PAIR_COUNT,
    run_id: str | None = None,
    source_video_path: str | Path | None = None,
    progress_jsonl_path: str | Path | None = None,
    progress_every: int = 0,
) -> dict[str, Any]:
    """Run the MLX scorer mirror on ``bundle`` and write a prefilter profile.

    Args:
        bundle: Shared MLX score-aware ``RendererBundle``. The bundle supplies
            the trained renderer and canonical target RGB frames.
        output_path: JSON profile path.
        archive_bytes: Byte size of the archive emitted from this renderer.
        archive_sha256: SHA-256 of that archive.
        upstream_dir: Upstream scorer/runtime directory used to load scorer
            weights.
        scorer_device: Device used while importing upstream PyTorch scorers
            before conversion to MLX. Usually ``"cpu"``.
        scorer_batch_pairs: Number of pairs per MLX scorer forward. Strict
            replay gates currently require singleton batches for full-video
            promotion, so callers should keep this at ``1`` when they want the
            profile to unlock local CPU replay automatically.
        required_pairs: Full-video pair count required by downstream gates.
        run_id: Optional provenance label.
        source_video_path: Optional source video path recorded for provenance.
        progress_jsonl_path: Optional JSONL path for crash-visible chunk
            progress. Rows are false-authority telemetry only.
        progress_every: Emit one progress row every N chunks. ``0`` disables
            progress rows; when enabled, the final chunk is always emitted.

    Returns:
        The written profile.
    """

    from tac.local_acceleration.mlx_scorer_adapters import (
        load_mlx_distortion_scorer_adapter_from_upstream,
        temporary_mlx_device,
    )

    with temporary_mlx_device(scorer_device):
        adapter = load_mlx_distortion_scorer_adapter_from_upstream(
            upstream_dir,
            device=scorer_device,
        )
        profile = build_mlx_renderer_prefilter_profile_loaded(
            bundle=bundle,
            adapter=adapter,
            archive_bytes=archive_bytes,
            archive_sha256=archive_sha256,
            scorer_batch_pairs=scorer_batch_pairs,
            required_pairs=required_pairs,
            run_id=run_id,
            source_video_path=source_video_path,
            upstream_dir=upstream_dir,
            scorer_device=scorer_device,
            progress_jsonl_path=progress_jsonl_path,
            progress_every=progress_every,
        )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(_jsonable(profile), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return profile


def build_mlx_renderer_prefilter_profile_loaded(
    *,
    bundle: Any,
    adapter: Any,
    archive_bytes: int,
    archive_sha256: str,
    scorer_batch_pairs: int = 1,
    required_pairs: int = CONTEST_PAIR_COUNT,
    run_id: str | None = None,
    source_video_path: str | Path | None = None,
    upstream_dir: str | Path | None = None,
    scorer_device: str = "cpu",
    progress_jsonl_path: str | Path | None = None,
    progress_every: int = 0,
) -> dict[str, Any]:
    """Build a profile with a pre-loaded MLX scorer adapter.

    This split keeps the heavy scorer import out of tests and lets future
    queue runners reuse one loaded scorer across many candidate renderers.
    """

    import mlx.core as mx

    from tac.local_acceleration.pr95_hnerv_mlx_training import (
        resize_nhwc_align_corners_false,
        rgb_to_yuv6_mlx,
    )
    from tac.substrates._shared.mlx_score_aware.loss import decode_frames_nhwc01

    pair_count = _positive_int(bundle.num_pairs, "bundle.num_pairs")
    batch_pairs = _positive_int(scorer_batch_pairs, "scorer_batch_pairs")
    archive_size = _positive_int(archive_bytes, "archive_bytes")
    archive_hash = _required_sha256(archive_sha256, "archive_sha256")
    required = _positive_int(required_pairs, "required_pairs")
    progress_interval = _nonnegative_int(progress_every, "progress_every")
    progress_path = (
        Path(progress_jsonl_path).expanduser().resolve(strict=False)
        if progress_jsonl_path is not None
        else None
    )
    if progress_path is not None:
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_path.write_text("", encoding="utf-8")

    started = time.time()
    pose_sum = 0.0
    seg_sum = 0.0
    count = 0
    output_hashes = _OutputHashes()
    chunk_elapsed_seconds: list[float] = []
    chunk_pairs_per_second: list[float] = []

    total_chunks = math.ceil(pair_count / float(batch_pairs))
    for chunk_index, start in enumerate(range(0, pair_count, batch_pairs), start=1):
        chunk_started = time.time()
        stop = min(pair_count, start + batch_pairs)
        idx = mx.arange(start, stop)
        cand_0, cand_1 = decode_frames_nhwc01(bundle, idx)
        ref_0 = bundle.target_rgb_0[idx]
        ref_1 = bundle.target_rgb_1[idx]
        cand_pair = mx.stack([cand_0, cand_1], axis=1) * 255.0
        ref_pair = mx.stack([ref_0, ref_1], axis=1) * 255.0
        cand_pose, cand_seg = _scorer_inputs_from_pair_rgb255(
            cand_pair,
            resize_nhwc_align_corners_false=resize_nhwc_align_corners_false,
            rgb_to_yuv6_mlx=rgb_to_yuv6_mlx,
        )
        ref_pose, ref_seg = _scorer_inputs_from_pair_rgb255(
            ref_pair,
            resize_nhwc_align_corners_false=resize_nhwc_align_corners_false,
            rgb_to_yuv6_mlx=rgb_to_yuv6_mlx,
        )
        cand_out = adapter(cand_pose, cand_seg)
        ref_out = adapter(ref_pose, ref_seg)
        cand_pose_np = np.asarray(cand_out["posenet"]["pose"], dtype=np.float32)
        ref_pose_np = np.asarray(ref_out["posenet"]["pose"], dtype=np.float32)
        cand_seg_np = np.asarray(cand_out["segnet"], dtype=np.float32)
        ref_seg_np = np.asarray(ref_out["segnet"], dtype=np.float32)
        output_hashes.update(
            candidate_pose=cand_pose_np,
            reference_pose=ref_pose_np,
            candidate_seg=cand_seg_np,
            reference_seg=ref_seg_np,
        )
        pose = _pose_distortion(cand_pose_np, ref_pose_np)
        seg = _segnet_argmax_distortion_nhwc(cand_seg_np, ref_seg_np)
        pose_sum += float(np.sum(pose, dtype=np.float64))
        seg_sum += float(np.sum(seg, dtype=np.float64))
        count += int(pose.shape[0])
        mx.eval(cand_pose, cand_seg, ref_pose, ref_seg)
        chunk_elapsed = time.time() - chunk_started
        chunk_pair_count = int(stop - start)
        chunk_rate = (
            float(chunk_pair_count) / chunk_elapsed if chunk_elapsed > 0.0 else None
        )
        chunk_elapsed_seconds.append(chunk_elapsed)
        if chunk_rate is not None:
            chunk_pairs_per_second.append(chunk_rate)
        if _should_emit_progress(
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            progress_every=progress_interval,
        ):
            elapsed_so_far = time.time() - started
            avg_pose_so_far = pose_sum / float(count)
            avg_seg_so_far = seg_sum / float(count)
            _append_progress_row(
                progress_path,
                {
                    "schema": "mlx_renderer_prefilter_progress.v1",
                    "producer": "tac.local_acceleration.mlx_renderer_prefilter_profile",
                    "generated_utc": datetime.now(UTC).isoformat(),
                    "run_id": run_id,
                    "archive_sha256": archive_hash,
                    "archive_bytes": archive_size,
                    "scorer_device": scorer_device,
                    "scorer_batch_pairs": batch_pairs,
                    "chunk_index": chunk_index,
                    "chunk_count": total_chunks,
                    "pair_start": int(start),
                    "pair_stop": int(stop),
                    "pair_count": chunk_pair_count,
                    "cumulative_pair_count": int(count),
                    "required_pairs": required,
                    "elapsed_seconds": elapsed_so_far,
                    "chunk_elapsed_seconds": chunk_elapsed,
                    "chunk_pairs_per_second": chunk_rate,
                    "cumulative_pairs_per_second": (
                        float(count) / elapsed_so_far
                        if elapsed_so_far > 0.0
                        else None
                    ),
                    "cumulative_avg_posenet_dist": avg_pose_so_far,
                    "cumulative_avg_segnet_dist": avg_seg_so_far,
                    "cumulative_canonical_score": contest_formula_score(
                        seg_dist=avg_seg_so_far,
                        pose_dist=avg_pose_so_far,
                        archive_bytes=archive_size,
                    ),
                    **FALSE_AUTHORITY,
                    "authority": (
                        "macOS MLX progress telemetry only; not a contest score, "
                        "promotion, rank, or kill authority."
                    ),
                },
            )

    if count != pair_count:
        raise ValueError(f"internal pair count mismatch: accumulated {count}, expected {pair_count}")
    avg_pose = pose_sum / float(count)
    avg_seg = seg_sum / float(count)
    canonical_score = contest_formula_score(
        seg_dist=avg_seg,
        pose_dist=avg_pose,
        archive_bytes=archive_size,
    )
    elapsed = time.time() - started
    throughput = float(count) / elapsed if elapsed > 0.0 else None
    full_scope = (
        "executed"
        if pair_count >= required
        else "sampled_prefix_requires_full_video_rerun"
    )
    blockers: list[str] = [
        "mlx_local_replay_not_contest_auth_axis",
    ]
    if pair_count < required:
        blockers.append("partial_coverage_mlx_replay_not_score_authority")
    if batch_pairs != 1:
        blockers.append("mlx_profile_batch_pairs_not_singleton")
    return {
        "schema": HPRC_MLX_COMPONENT_PROFILE_SCHEMA,
        "producer": "tac.local_acceleration.mlx_renderer_prefilter_profile",
        "generated_utc": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "source_video_path": str(source_video_path) if source_video_path else None,
        "upstream_dir": str(upstream_dir) if upstream_dir else None,
        "scorer_device": scorer_device,
        "evidence_grade": EVIDENCE_GRADE_MLX,
        "evidence_tag": EVIDENCE_TAG_MLX,
        **FALSE_AUTHORITY,
        "candidate_generation_only": True,
        "requires_exact_eval_before_promotion": True,
        "archive_size_bytes": archive_size,
        "archive_bytes": archive_size,
        "archive_sha256": archive_hash,
        "original_video_bytes": ORIGINAL_VIDEO_BYTES,
        "max_pairs": pair_count,
        "num_pairs": pair_count,
        "n_samples": pair_count,
        "scorer_batch_pairs": batch_pairs,
        "scope_status": {"full_video": full_scope},
        "score_components": {
            "avg_posenet_dist": avg_pose,
            "avg_segnet_dist": avg_seg,
            "pose_term": math.sqrt(10.0 * avg_pose),
            "seg_term": 100.0 * avg_seg,
            "rate_term": 25.0 * archive_size / ORIGINAL_VIDEO_BYTES,
            "rate_unscaled": archive_size / ORIGINAL_VIDEO_BYTES,
            "canonical_score": canonical_score,
        },
        "mlx_response_summary": {
            "batch_pairs": batch_pairs,
            "max_pairs": pair_count,
            "n_samples": pair_count,
            "candidate_cache_pairs": pair_count,
            "reference_cache_pairs": pair_count,
            "local_score_estimate": canonical_score,
        },
        "response_metadata": {
            "batch_pairs": batch_pairs,
            "num_pairs": pair_count,
            "local_score_estimate": canonical_score,
            "elapsed_seconds": elapsed,
            "pair_throughput_per_second": throughput,
            "scorer_layout": "direct_mlx_nhwc",
        },
        "progress": {
            "schema": "mlx_renderer_prefilter_progress_summary.v1",
            "progress_jsonl_path": progress_path.as_posix() if progress_path else None,
            "progress_every": progress_interval,
            "chunk_count": total_chunks,
            "batch_pairs": batch_pairs,
            "pair_throughput_per_second": throughput,
            "mean_chunk_elapsed_seconds": (
                statistics.fmean(chunk_elapsed_seconds)
                if chunk_elapsed_seconds
                else None
            ),
            "median_chunk_elapsed_seconds": (
                statistics.median(chunk_elapsed_seconds)
                if chunk_elapsed_seconds
                else None
            ),
            "slowest_chunk_seconds": (
                max(chunk_elapsed_seconds) if chunk_elapsed_seconds else None
            ),
            "fastest_chunk_seconds": (
                min(chunk_elapsed_seconds) if chunk_elapsed_seconds else None
            ),
            "mean_chunk_pairs_per_second": (
                statistics.fmean(chunk_pairs_per_second)
                if chunk_pairs_per_second
                else None
            ),
            "cache_reuse_contract": (
                "build_mlx_renderer_prefilter_profile_loaded accepts a preloaded "
                "MLX scorer adapter so queue runners can reuse scorer weights "
                "across candidates without re-importing upstream PyTorch scorers."
            ),
            **FALSE_AUTHORITY,
        },
        "component_output_hashes": output_hashes.hexdigests(),
        "section_value_rows": [],
        "blockers": blockers,
        "authority_status": (
            "macOS MLX scorer replay is a local prefilter only. It can route "
            "local CPU replay after coverage/calibration gates, but it is not "
            "a contest score, promotion, rank, or kill authority."
        ),
    }


def write_mlx_renderer_prefilter_failure_profile(
    *,
    output_path: str | Path,
    archive_bytes: int | None,
    archive_sha256: str | None,
    num_pairs: int,
    failure: str,
    required_pairs: int = CONTEST_PAIR_COUNT,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Write a supported-schema failure profile so gates preserve the blocker."""

    pairs = int(num_pairs)
    profile = {
        "schema": HPRC_MLX_COMPONENT_PROFILE_SCHEMA,
        "producer": "tac.local_acceleration.mlx_renderer_prefilter_profile",
        "generated_utc": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "evidence_grade": EVIDENCE_GRADE_MLX,
        "evidence_tag": EVIDENCE_TAG_MLX,
        **FALSE_AUTHORITY,
        "candidate_generation_only": True,
        "requires_exact_eval_before_promotion": True,
        "archive_size_bytes": int(archive_bytes) if archive_bytes else None,
        "archive_bytes": int(archive_bytes) if archive_bytes else None,
        "archive_sha256": archive_sha256,
        "max_pairs": pairs,
        "num_pairs": pairs,
        "n_samples": pairs,
        "scorer_batch_pairs": 1,
        "scope_status": {"full_video": "profile_failed"},
        "mlx_response_summary": {
            "batch_pairs": 1,
            "max_pairs": pairs,
            "n_samples": pairs,
        },
        "section_value_rows": [],
        "blockers": [
            "mlx_renderer_prefilter_profile_failed",
            f"mlx_renderer_prefilter_exception:{failure}",
        ],
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(_jsonable(profile), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return profile


class _OutputHashes:
    def __init__(self) -> None:
        self._digests = {
            "candidate_pose": hashlib.sha256(),
            "reference_pose": hashlib.sha256(),
            "candidate_seg": hashlib.sha256(),
            "reference_seg": hashlib.sha256(),
        }

    def update(
        self,
        *,
        candidate_pose: np.ndarray,
        reference_pose: np.ndarray,
        candidate_seg: np.ndarray,
        reference_seg: np.ndarray,
    ) -> None:
        for name, array in (
            ("candidate_pose", candidate_pose),
            ("reference_pose", reference_pose),
            ("candidate_seg", candidate_seg),
            ("reference_seg", reference_seg),
        ):
            _update_array_digest(self._digests[name], array)

    def hexdigests(self) -> dict[str, str]:
        return {name: digest.hexdigest() for name, digest in self._digests.items()}


def _scorer_inputs_from_pair_rgb255(
    pair_rgb_nhwc: Any,
    *,
    resize_nhwc_align_corners_false: Any,
    rgb_to_yuv6_mlx: Any,
) -> tuple[Any, Any]:
    import mlx.core as mx

    if len(pair_rgb_nhwc.shape) != 5 or int(pair_rgb_nhwc.shape[1]) != 2:
        raise ValueError(f"expected (B,2,H,W,3), got {tuple(pair_rgb_nhwc.shape)}")
    b, t, h, w, c = [int(v) for v in pair_rgb_nhwc.shape]
    if c != 3:
        raise ValueError(f"expected RGB channel count 3, got {c}")
    flat = mx.reshape(pair_rgb_nhwc, (b * t, h, w, c))
    if (h, w) != (384, 512):
        flat = resize_nhwc_align_corners_false(flat, size=(384, 512), mode="bilinear")
    resized = mx.reshape(flat, (b, t, 384, 512, c))
    segnet_last_rgb = resized[:, -1, :, :, :]
    yuv6 = rgb_to_yuv6_mlx(resized)
    yuv6 = mx.transpose(yuv6, (0, 2, 3, 1, 4))
    posenet_yuv6_pair = mx.reshape(yuv6, (b, 192, 256, 12))
    return posenet_yuv6_pair, segnet_last_rgb


def _pose_distortion(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    diff = np.asarray(candidate[..., :6] - reference[..., :6], dtype=np.float32)
    axes = tuple(range(1, diff.ndim))
    return np.mean(np.square(diff), axis=axes, dtype=np.float32)


def _segnet_argmax_distortion_nhwc(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if candidate.ndim != 4 or reference.ndim != 4:
        raise ValueError(
            f"expected SegNet NHWC logits, got candidate={candidate.shape} reference={reference.shape}"
        )
    diff = np.argmax(reference, axis=-1) != np.argmax(candidate, axis=-1)
    return np.mean(diff.astype(np.float32), axis=(1, 2), dtype=np.float32)


def _update_array_digest(digest: hashlib._Hash, array: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(array)
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(json.dumps(list(contiguous.shape)).encode("utf-8"))
    digest.update(contiguous.tobytes())


def _positive_int(value: Any, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer, got {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be positive, got {parsed}")
    return parsed


def _nonnegative_int(value: Any, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a nonnegative integer, got {value!r}") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be nonnegative, got {parsed}")
    return parsed


def _required_sha256(value: Any, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text.lower()):
        raise ValueError(f"{name} must be a 64-char lowercase SHA-256 hex string")
    return text.lower()


def _should_emit_progress(
    *,
    chunk_index: int,
    total_chunks: int,
    progress_every: int,
) -> bool:
    if progress_every <= 0:
        return False
    if chunk_index >= total_chunks:
        return True
    return chunk_index % progress_every == 0


def _append_progress_row(path: Path | None, row: dict[str, Any]) -> None:
    if path is None:
        return
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(_jsonable(row), sort_keys=True, allow_nan=False))
        fh.write("\n")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    return value


__all__ = [
    "build_mlx_renderer_prefilter_profile_loaded",
    "write_mlx_renderer_prefilter_failure_profile",
    "write_mlx_renderer_prefilter_profile",
]
