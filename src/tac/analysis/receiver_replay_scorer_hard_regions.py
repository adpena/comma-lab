# SPDX-License-Identifier: MIT
"""Mine receiver-replay SegNet hard regions from argmax and scorer cache artifacts.

The core miner is NumPy-only: it accepts candidate/reference SegNet argmax
surfaces and returns confusion, per-pair/per-class hard-region rows, and
optional connected components.  Optional helpers can derive argmax surfaces from
receiver scorer-input caches, but every emitted report is local false-authority
diagnostic evidence only.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "receiver_replay_scorer_hard_regions.v1"
TOOL = "tac.analysis.receiver_replay_scorer_hard_regions"
SEGNET_SCORE_WEIGHT = 100.0
DEFAULT_MIN_CLASS_COUNT = 5
FALSE_AUTHORITY = {
    "score_claim": False,
    "score_claim_valid": False,
    "dispatch_attempted": False,
    "gpu_launched": False,
    "promotion_eligible": False,
    "promotable": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}


class ReceiverReplayHardRegionError(ValueError):
    """Raised when hard-region inputs cannot be trusted even as local diagnostics."""


def build_receiver_replay_scorer_hard_region_report(
    *,
    candidate_argmax: np.ndarray,
    reference_argmax: np.ndarray,
    pair_indices: np.ndarray | None = None,
    posenet_distortion: np.ndarray | Sequence[float] | None = None,
    segnet_distortion: np.ndarray | Sequence[float] | None = None,
    label: str = "receiver_replay_scorer_hard_regions",
    min_class_count: int = DEFAULT_MIN_CLASS_COUNT,
    top_components: int = 32,
    include_solved_confusions: bool = False,
    source_artifacts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic hard-region report from SegNet argmax arrays.

    ``candidate_argmax`` and ``reference_argmax`` are expected as ``(N,H,W)`` or
    ``(H,W)`` integer arrays.  The report prices unsolved mass in SegNet score
    units, using the contest ``100 * d_seg`` term over the provided sample.
    ``posenet_distortion`` is optional per-pair context from scorer-response
    component artifacts; it is reported as local marginal context, not as score
    authority.
    """

    candidate = _normalize_argmax(candidate_argmax, "candidate_argmax")
    reference = _normalize_argmax(reference_argmax, "reference_argmax")
    if candidate.shape != reference.shape:
        raise ReceiverReplayHardRegionError(
            "candidate/reference argmax shapes must match; "
            f"got {candidate.shape} vs {reference.shape}"
        )
    pair_count, height, width = (int(v) for v in candidate.shape)
    if pair_count <= 0 or height <= 0 or width <= 0:
        raise ReceiverReplayHardRegionError(f"argmax arrays must be non-empty, got {candidate.shape}")
    class_count = _resolve_class_count(candidate, reference, min_class_count=min_class_count)
    pair_index_array = _normalize_pair_indices(pair_indices, pair_count=pair_count)
    pose = _normalize_optional_vector(posenet_distortion, pair_count=pair_count, label="posenet_distortion")
    seg = _normalize_optional_vector(segnet_distortion, pair_count=pair_count, label="segnet_distortion")

    total_pixels = int(candidate.size)
    pair_pixels = int(height * width)
    global_confusion = _confusion_matrix(reference.reshape(-1), candidate.reshape(-1), class_count=class_count)
    mismatch = candidate != reference
    mismatch_pixels = int(np.count_nonzero(mismatch))
    segnet_score_contribution = _score_mass(mismatch_pixels, total_pixels)
    pose_marginal_weight = _pose_marginal_weight(pose)

    hard_records: list[dict[str, Any]] = []
    per_pair: list[dict[str, Any]] = []
    for row_index in range(pair_count):
        pair_confusion = _confusion_matrix(
            reference[row_index].reshape(-1),
            candidate[row_index].reshape(-1),
            class_count=class_count,
        )
        pair_mismatch_pixels = int(pair_pixels - np.trace(pair_confusion))
        pair_records = _hard_records_for_pair(
            pair_confusion=pair_confusion,
            pair_index=row_index,
            source_frame_pair=_source_frame_pair(pair_index_array, row_index),
            total_pixels=total_pixels,
            pair_pixels=pair_pixels,
            posenet_distortion=None if pose is None else float(pose[row_index]),
            pose_marginal_weight=pose_marginal_weight,
            scorer_response_segnet_distortion=None if seg is None else float(seg[row_index]),
            include_solved_confusions=include_solved_confusions,
        )
        hard_records.extend(pair_records)
        per_pair.append(
            {
                "local_row_index": int(row_index),
                "source_frame_pair": _source_frame_pair(pair_index_array, row_index),
                "pair_pixels": pair_pixels,
                "mismatch_pixels": pair_mismatch_pixels,
                "argmax_disagreement_rate": _ratio(pair_mismatch_pixels, pair_pixels),
                "pair_segnet_score_contribution": _score_mass(pair_mismatch_pixels, pair_pixels),
                "sample_weighted_segnet_score_contribution": _score_mass(pair_mismatch_pixels, total_pixels),
                "posenet_distortion": None if pose is None else float(pose[row_index]),
                "pose_marginal_score_contribution": (
                    None if pose is None else float(pose[row_index] * pose_marginal_weight)
                ),
                "scorer_response_segnet_distortion": None if seg is None else float(seg[row_index]),
                "argmax_pair_segnet_distortion": _ratio(pair_mismatch_pixels, pair_pixels),
                "confusion_matrix": _matrix_to_jsonable(pair_confusion),
                "hard_region_records": pair_records,
            }
        )

    hard_records.sort(key=_hard_record_sort_key)
    for rank, row in enumerate(hard_records, start=1):
        row["rank"] = rank

    components = _top_connected_components(
        candidate=candidate,
        reference=reference,
        hard_records=hard_records,
        total_pixels=total_pixels,
        pair_pixels=pair_pixels,
        top_components=top_components,
    )

    global_confusion_records = _confusion_records(
        global_confusion,
        total_pixels=total_pixels,
        include_solved=True,
    )
    return {
        "schema": SCHEMA,
        "tool": TOOL,
        "generated_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "label": str(label),
        "evidence_grade": "local_receiver_replay_scorer_hard_region_false_authority",
        "evidence_tag": "[local receiver-replay hard-region signal]",
        "authority": {
            **FALSE_AUTHORITY,
            "research_only": True,
            "notes": [
                "local_argmax_or_cache_diagnostic_only",
                "not_a_score_claim",
                "not_rank_kill_or_promotion_authority",
            ],
        },
        **FALSE_AUTHORITY,
        "array_identity": {
            "candidate_argmax_sha256": _array_sha256(candidate),
            "reference_argmax_sha256": _array_sha256(reference),
            "pair_indices_sha256": None if pair_index_array is None else _array_sha256(pair_index_array),
            "posenet_distortion_sha256": None if pose is None else _array_sha256(pose),
            "segnet_distortion_sha256": None if seg is None else _array_sha256(seg),
        },
        "source_artifacts": dict(source_artifacts or {}),
        "pair_count": pair_count,
        "argmax_shape": [pair_count, height, width],
        "class_count": class_count,
        "total_pixels": total_pixels,
        "mismatch_pixels": mismatch_pixels,
        "argmax_disagreement_rate": _ratio(mismatch_pixels, total_pixels),
        "segnet_score_contribution": segnet_score_contribution,
        "score_weighting": {
            "segnet_score_weight": SEGNET_SCORE_WEIGHT,
            "segnet_score_formula": "100 * mismatch_pixels / total_pixels",
            "pose_pair_context_formula": (
                "posenet_distortion * 5/sqrt(10*avg_posenet_dist); local marginal context only"
            ),
            "pose_marginal_weight": pose_marginal_weight,
            "has_posenet_pair_scores": pose is not None,
            "has_scorer_response_segnet_distortion": seg is not None,
        },
        "confusion_matrix": _matrix_to_jsonable(global_confusion),
        "confusion_records": global_confusion_records,
        "hard_region_records": hard_records,
        "top_connected_components": components,
        "per_pair": per_pair,
    }


def write_receiver_replay_scorer_hard_region_report(
    *,
    output_json: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build and write a hard-region report as JSON."""

    payload = build_receiver_replay_scorer_hard_region_report(**kwargs)
    out = Path(output_json).expanduser().resolve(strict=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def build_hard_region_recon_pixel_weight(
    report: Mapping[str, Any],
    *,
    output_height: int,
    output_width: int,
    pair_count: int | None = None,
    frame_index: int = 1,
    base_weight: float = 1.0,
    score_gain: float = 2.0,
    component_gain: float = 1.0,
    normalize: str = "mean",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Convert hard-region rows into a trainer-consumable recon weight map.

    The output shape is ``(N,2,H,W,1)``, matching the existing MLX
    ``recon_pixel_weight`` channel.  The map is deliberately SegNet-last-frame
    biased by default because upstream evaluates SegNet on frame 1; callers can
    change ``frame_index`` for ablations, but the default path is contest
    geometry rather than human visual repair.
    """

    if int(output_height) <= 0 or int(output_width) <= 0:
        raise ReceiverReplayHardRegionError(
            f"output_height/output_width must be positive; got {output_height}x{output_width}"
        )
    if int(frame_index) not in (0, 1):
        raise ReceiverReplayHardRegionError(f"frame_index must be 0 or 1, got {frame_index}")
    if normalize not in {"mean", "none"}:
        raise ReceiverReplayHardRegionError(f"normalize must be 'mean' or 'none', got {normalize!r}")
    for name, value in (
        ("base_weight", base_weight),
        ("score_gain", score_gain),
        ("component_gain", component_gain),
    ):
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ReceiverReplayHardRegionError(f"{name} must be finite and non-negative")

    report_pairs = int(report.get("pair_count") or 0)
    n = int(pair_count) if pair_count is not None else report_pairs
    if n <= 0:
        raise ReceiverReplayHardRegionError("pair_count must be positive or present in report")
    if report_pairs > 0 and n < report_pairs:
        raise ReceiverReplayHardRegionError(
            f"requested pair_count={n} is smaller than report pair_count={report_pairs}"
        )

    weight = np.full(
        (n, 2, int(output_height), int(output_width), 1),
        float(base_weight),
        dtype=np.float32,
    )
    source_shape = report.get("argmax_shape")
    if (
        not isinstance(source_shape, Sequence)
        or len(source_shape) != 3
        or int(source_shape[1]) <= 0
        or int(source_shape[2]) <= 0
    ):
        raise ReceiverReplayHardRegionError("report argmax_shape must be [N,H,W]")
    src_h = int(source_shape[1])
    src_w = int(source_shape[2])
    hard_records = list(report.get("hard_region_records") or [])
    components = list(report.get("top_connected_components") or [])
    applied_components = 0
    applied_records = 0
    max_delta = 0.0

    for row in hard_records:
        if bool(row.get("solved")):
            continue
        local_row = int(row.get("local_row_index", -1))
        if local_row < 0 or local_row >= n:
            continue
        score_mass = _finite_nonnegative(row.get("score_weighted_unsolved_mass"))
        pixel_fraction = _finite_nonnegative(row.get("pixel_fraction_of_pair"))
        if score_mass <= 0.0 and pixel_fraction <= 0.0:
            continue
        delta = float(score_gain) * score_mass + float(component_gain) * pixel_fraction
        if delta <= 0.0:
            continue
        mask = _bbox_mask_for_hard_record(
            row,
            components=components,
            src_h=src_h,
            src_w=src_w,
            dst_h=int(output_height),
            dst_w=int(output_width),
        )
        frame_weight = weight[local_row, int(frame_index), :, :, 0]
        if mask is None:
            frame_weight += np.float32(delta)
        else:
            frame_weight[mask] += np.float32(delta)
            applied_components += 1
        max_delta = max(max_delta, delta)
        applied_records += 1

    if normalize == "mean":
        mean = float(np.mean(weight, dtype=np.float64))
        if not math.isfinite(mean) or mean <= 0.0:
            raise ReceiverReplayHardRegionError("hard-region recon weight mean is non-positive")
        weight = np.ascontiguousarray((weight / mean).astype(np.float32))
    metadata = {
        "schema": "receiver_replay_hard_region_recon_pixel_weight.v1",
        "source_report_schema": report.get("schema"),
        "source_report_label": report.get("label"),
        "evidence_grade": "local_receiver_replay_scorer_hard_region_false_authority",
        "evidence_tag": "[local receiver-replay hard-region signal]",
        "pair_count": n,
        "height": int(output_height),
        "width": int(output_width),
        "shape": [int(v) for v in weight.shape],
        "target_frame_index": int(frame_index),
        "base_weight": float(base_weight),
        "score_gain": float(score_gain),
        "component_gain": float(component_gain),
        "normalize": normalize,
        "applied_hard_region_records": int(applied_records),
        "applied_component_bboxes": int(applied_components),
        "max_unnormalized_delta": float(max_delta),
        "weight_stats": _weight_stats(weight),
        "source_report_identity": dict(report.get("array_identity") or {}),
        **FALSE_AUTHORITY,
    }
    return weight, metadata


def write_hard_region_recon_pixel_weight_artifact(
    *,
    report: Mapping[str, Any],
    output_dir: str | Path,
    output_height: int,
    output_width: int,
    pair_count: int | None = None,
    frame_index: int = 1,
    base_weight: float = 1.0,
    score_gain: float = 2.0,
    component_gain: float = 1.0,
    normalize: str = "mean",
    allow_overwrite: bool = False,
) -> dict[str, Any]:
    """Write a false-authority hard-region recon-weight NPZ + manifest."""

    out = Path(output_dir).expanduser().resolve(strict=False)
    if out.exists() and any(out.iterdir()) and not allow_overwrite:
        raise FileExistsError(f"output_dir is non-empty; pass allow_overwrite=True: {out}")
    out.mkdir(parents=True, exist_ok=True)
    weight, metadata = build_hard_region_recon_pixel_weight(
        report,
        output_height=output_height,
        output_width=output_width,
        pair_count=pair_count,
        frame_index=frame_index,
        base_weight=base_weight,
        score_gain=score_gain,
        component_gain=component_gain,
        normalize=normalize,
    )
    weight_path = out / "receiver_replay_hard_region_recon_pixel_weight.npz"
    manifest_path = out / "receiver_replay_hard_region_recon_pixel_weight_manifest.json"
    np.savez_compressed(weight_path, weight=weight)
    manifest = {
        "schema": "receiver_replay_hard_region_recon_pixel_weight_manifest.v1",
        "manifest_path": manifest_path.as_posix(),
        "weight_path": weight_path.as_posix(),
        "weight_sha256": _file_sha256(weight_path),
        "weight_array_sha256": _array_sha256(weight),
        "weight_bytes": int(weight_path.stat().st_size),
        "metadata": metadata,
        "consumption": {
            "training_arg": "--recon-pixel-weight-path",
            "training_consumption_recommended": bool(
                metadata["applied_hard_region_records"] > 0
            ),
            "auto_discovery_eligible": False,
            "reason": "receiver_replay_hard_region_surface_requires_explicit_operator_or_runner_selection",
        },
        **FALSE_AUTHORITY,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_argmax_array(path: str | Path) -> np.ndarray:
    """Load a direct argmax array from ``.npy`` or JSON/JSONL-like JSON."""

    resolved = Path(path).expanduser().resolve(strict=False)
    if not resolved.is_file():
        raise FileNotFoundError(f"argmax array does not exist: {resolved}")
    if resolved.suffix.lower() == ".npy":
        return np.asarray(np.load(resolved, mmap_mode="r"))
    try:
        return np.asarray(json.loads(resolved.read_text(encoding="utf-8")))
    except json.JSONDecodeError as exc:
        raise ReceiverReplayHardRegionError(f"unsupported argmax array format for {resolved}") from exc


def load_pair_indices(path: str | Path) -> np.ndarray:
    """Load pair indices from ``.npy`` or JSON."""

    resolved = Path(path).expanduser().resolve(strict=False)
    if not resolved.is_file():
        raise FileNotFoundError(f"pair indices do not exist: {resolved}")
    if resolved.suffix.lower() == ".npy":
        return np.asarray(np.load(resolved, mmap_mode="r"))
    try:
        return np.asarray(json.loads(resolved.read_text(encoding="utf-8")))
    except json.JSONDecodeError as exc:
        raise ReceiverReplayHardRegionError(f"unsupported pair indices format for {resolved}") from exc


def load_component_vector(path: str | Path, *, label: str) -> np.ndarray:
    """Load a per-pair scorer-response component vector."""

    resolved = Path(path).expanduser().resolve(strict=False)
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} component vector does not exist: {resolved}")
    if resolved.suffix.lower() == ".npy":
        return np.asarray(np.load(resolved, mmap_mode="r"))
    try:
        return np.asarray(json.loads(resolved.read_text(encoding="utf-8")))
    except json.JSONDecodeError as exc:
        raise ReceiverReplayHardRegionError(f"unsupported {label} component vector format for {resolved}") from exc


def load_component_vectors_from_dir(components_dir: str | Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load ``posenet_distortion.npy`` and ``segnet_distortion.npy`` if present."""

    root = Path(components_dir).expanduser().resolve(strict=False)
    pose_path = root / "posenet_distortion.npy"
    seg_path = root / "segnet_distortion.npy"
    pose = load_component_vector(pose_path, label="posenet_distortion") if pose_path.is_file() else None
    seg = load_component_vector(seg_path, label="segnet_distortion") if seg_path.is_file() else None
    return pose, seg


def component_artifacts_from_mlx_response(response: Mapping[str, Any]) -> dict[str, Path]:
    """Return component artifact paths from an MLX scorer-response payload or wrapper."""

    payload: Mapping[str, Any] = response
    nested = response.get("response_payload")
    if isinstance(nested, Mapping):
        payload = nested
    if str(payload.get("schema") or "") != "mlx_scorer_response.v1":
        raise ReceiverReplayHardRegionError("MLX response must have schema 'mlx_scorer_response.v1'")
    artifacts = payload.get("components")
    if isinstance(artifacts, Mapping):
        artifacts = artifacts.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ReceiverReplayHardRegionError("MLX response components.artifacts missing")
    out: dict[str, Path] = {}
    for key in ("posenet_distortion", "segnet_distortion"):
        item = artifacts.get(key)
        if isinstance(item, Mapping) and item.get("path"):
            out[key] = Path(str(item["path"])).expanduser().resolve(strict=False)
    return out


def load_component_vectors_from_mlx_response(path: str | Path) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    """Load per-pair component vectors from an MLX scorer-response JSON payload."""

    response_path = Path(path).expanduser().resolve(strict=False)
    try:
        response = json.loads(response_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReceiverReplayHardRegionError(f"could not parse MLX response JSON: {response_path}") from exc
    artifacts = component_artifacts_from_mlx_response(response)
    pose = (
        load_component_vector(artifacts["posenet_distortion"], label="posenet_distortion")
        if "posenet_distortion" in artifacts
        else None
    )
    seg = (
        load_component_vector(artifacts["segnet_distortion"], label="segnet_distortion")
        if "segnet_distortion" in artifacts
        else None
    )
    source: dict[str, Any] = {
        "mlx_response_path": response_path.as_posix(),
        "mlx_response_sha256": _file_sha256(response_path),
    }
    for key, artifact_path in artifacts.items():
        source[key] = _artifact_record(artifact_path)
    return pose, seg, source


def infer_cache_dirs_from_mlx_response(path: str | Path) -> tuple[Path | None, Path | None]:
    """Best-effort extraction of candidate/reference cache dirs from response JSON."""

    response_path = Path(path).expanduser().resolve(strict=False)
    payload = json.loads(response_path.read_text(encoding="utf-8"))
    candidate = _first_path_value(payload, ("candidate_cache_dir", "source_cache_run.candidate_cache_dir"))
    reference = _first_path_value(payload, ("reference_cache_dir", "source_cache_run.reference_cache_dir"))
    nested = payload.get("response_payload")
    if isinstance(nested, Mapping):
        candidate = candidate or _first_path_value(nested, ("source_cache_run.candidate_cache_dir",))
        reference = reference or _first_path_value(nested, ("source_cache_run.reference_cache_dir",))
    return (
        Path(candidate).expanduser().resolve(strict=False) if candidate else None,
        Path(reference).expanduser().resolve(strict=False) if reference else None,
    )


def build_segnet_argmax_arrays_from_cache_dirs(
    *,
    candidate_cache_dir: str | Path,
    reference_cache_dir: str | Path,
    upstream_dir: str | Path,
    sample_pairs: int | None = None,
    batch_frames: int = 4,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Run the existing MLX SegNet adapter on receiver scorer-input caches."""

    candidate_root = Path(candidate_cache_dir).expanduser().resolve(strict=False)
    reference_root = Path(reference_cache_dir).expanduser().resolve(strict=False)
    upstream = Path(upstream_dir).expanduser().resolve(strict=False)
    cand_raw = np.asarray(np.load(candidate_root / "segnet_last_rgb.npy", mmap_mode="r"))
    ref_raw = np.asarray(np.load(reference_root / "segnet_last_rgb.npy", mmap_mode="r"))
    cand_seg, cand_layout = _normalize_segnet_cache_rgb_nchw(
        cand_raw,
        label="candidate segnet_last_rgb",
    )
    ref_seg, ref_layout = _normalize_segnet_cache_rgb_nchw(
        ref_raw,
        label="reference segnet_last_rgb",
    )
    pair_indices = np.asarray(np.load(candidate_root / "pair_indices.npy", mmap_mode="r"))
    if cand_seg.shape[1:] != ref_seg.shape[1:]:
        raise ReceiverReplayHardRegionError(
            f"candidate/reference segnet cache shapes differ: {cand_seg.shape} vs {ref_seg.shape}"
        )
    n = int(min(cand_seg.shape[0], ref_seg.shape[0]))
    if sample_pairs is not None:
        if int(sample_pairs) <= 0:
            raise ReceiverReplayHardRegionError(f"sample_pairs must be positive, got {sample_pairs}")
        n = min(n, int(sample_pairs))
    if n <= 0:
        raise ReceiverReplayHardRegionError("cache arrays contain no sample pairs")

    logits_fn = _build_real_mlx_segnet_logits_fn(upstream_dir=upstream, device=device)
    cand_argmax = _run_segnet_argmax_batches(
        np.asarray(cand_seg[:n], dtype=np.float32),
        logits_fn=logits_fn,
        batch_frames=batch_frames,
    )
    ref_argmax = _run_segnet_argmax_batches(
        np.asarray(ref_seg[:n], dtype=np.float32),
        logits_fn=logits_fn,
        batch_frames=batch_frames,
    )
    source = {
        "candidate_cache_dir": candidate_root.as_posix(),
        "reference_cache_dir": reference_root.as_posix(),
        "upstream_dir": upstream.as_posix(),
        "segnet_backend": "mlx_segnet_adapter",
        "segnet_device": str(device),
        "sample_pairs": n,
        "candidate_cache_layout": cand_layout,
        "reference_cache_layout": ref_layout,
        "candidate_cache_manifest": _optional_artifact_record(candidate_root / "manifest.json"),
        "reference_cache_manifest": _optional_artifact_record(reference_root / "manifest.json"),
    }
    return cand_argmax, ref_argmax, np.asarray(pair_indices[:n]), source


def _normalize_segnet_cache_rgb_nchw(value: np.ndarray, *, label: str) -> tuple[np.ndarray, str]:
    arr = np.asarray(value)
    if arr.ndim != 4:
        raise ReceiverReplayHardRegionError(
            f"cache {label} must be rank-4 RGB, got shape={arr.shape}"
        )
    if int(arr.shape[1]) == 3:
        return np.ascontiguousarray(arr.astype(np.float32, copy=False)), "NCHW"
    if int(arr.shape[-1]) == 3:
        return np.ascontiguousarray(np.transpose(arr, (0, 3, 1, 2)).astype(np.float32, copy=False)), "NHWC_to_NCHW"
    raise ReceiverReplayHardRegionError(
        f"cache {label} must have RGB channel dimension 3 in NCHW or NHWC layout, got shape={arr.shape}"
    )


def _normalize_argmax(value: np.ndarray, label: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim == 2:
        arr = arr.reshape(1, *arr.shape)
    elif arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0]
    elif arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 3:
        raise ReceiverReplayHardRegionError(f"{label} must have shape (N,H,W) or (H,W), got {arr.shape}")
    if not (np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.bool_)):
        raise ReceiverReplayHardRegionError(f"{label} must be an integer argmax array, got dtype={arr.dtype}")
    out = np.asarray(arr, dtype=np.int64)
    if out.size == 0:
        raise ReceiverReplayHardRegionError(f"{label} must be non-empty")
    if int(np.min(out)) < 0:
        raise ReceiverReplayHardRegionError(f"{label} contains negative class ids")
    return np.ascontiguousarray(out)


def _normalize_pair_indices(value: np.ndarray | None, *, pair_count: int) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.ndim == 1:
        if arr.shape[0] != pair_count:
            raise ReceiverReplayHardRegionError(f"pair_indices length {arr.shape[0]} does not match {pair_count}")
        return np.ascontiguousarray(arr.astype(np.int64, copy=False))
    if arr.ndim == 2:
        if arr.shape[0] != pair_count:
            raise ReceiverReplayHardRegionError(f"pair_indices rows {arr.shape[0]} do not match {pair_count}")
        return np.ascontiguousarray(arr.astype(np.int64, copy=False))
    raise ReceiverReplayHardRegionError(f"pair_indices must be rank-1 or rank-2, got {arr.shape}")


def _normalize_optional_vector(
    value: np.ndarray | Sequence[float] | None,
    *,
    pair_count: int,
    label: str,
) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1 or arr.shape[0] != pair_count:
        raise ReceiverReplayHardRegionError(f"{label} must be a 1-D vector of length {pair_count}, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ReceiverReplayHardRegionError(f"{label} contains non-finite values")
    return np.ascontiguousarray(arr)


def _resolve_class_count(candidate: np.ndarray, reference: np.ndarray, *, min_class_count: int) -> int:
    if int(min_class_count) < 1:
        raise ReceiverReplayHardRegionError(f"min_class_count must be positive, got {min_class_count}")
    max_class = int(max(np.max(candidate), np.max(reference), 0))
    return max(max_class + 1, int(min_class_count))


def _confusion_matrix(reference: np.ndarray, candidate: np.ndarray, *, class_count: int) -> np.ndarray:
    encoded = np.asarray(reference, dtype=np.int64) * int(class_count) + np.asarray(candidate, dtype=np.int64)
    return np.bincount(encoded, minlength=int(class_count) * int(class_count)).reshape(
        int(class_count),
        int(class_count),
    )


def _hard_records_for_pair(
    *,
    pair_confusion: np.ndarray,
    pair_index: int,
    source_frame_pair: list[int] | None,
    total_pixels: int,
    pair_pixels: int,
    posenet_distortion: float | None,
    pose_marginal_weight: float,
    scorer_response_segnet_distortion: float | None,
    include_solved_confusions: bool,
) -> list[dict[str, Any]]:
    target_pixels_by_class = pair_confusion.sum(axis=1)
    records: list[dict[str, Any]] = []
    for target_class in range(pair_confusion.shape[0]):
        target_pixels = int(target_pixels_by_class[target_class])
        if target_pixels <= 0:
            continue
        for predicted_class in range(pair_confusion.shape[1]):
            pixel_count = int(pair_confusion[target_class, predicted_class])
            if pixel_count <= 0:
                continue
            solved = target_class == predicted_class
            if solved and not include_solved_confusions:
                continue
            unsolved_pixels = 0 if solved else pixel_count
            records.append(
                {
                    "local_row_index": int(pair_index),
                    "source_frame_pair": source_frame_pair,
                    "target_class": int(target_class),
                    "predicted_class": int(predicted_class),
                    "solved": bool(solved),
                    "pixel_count": pixel_count,
                    "unsolved_pixels": int(unsolved_pixels),
                    "target_mass_pixels": target_pixels,
                    "target_mass_fraction_of_pair": _ratio(target_pixels, pair_pixels),
                    "pixel_fraction_of_pair": _ratio(pixel_count, pair_pixels),
                    "pixel_fraction_of_target_region": _ratio(pixel_count, target_pixels),
                    "score_weighted_unsolved_mass": _score_mass(unsolved_pixels, total_pixels),
                    "pair_score_weighted_unsolved_mass": _score_mass(unsolved_pixels, pair_pixels),
                    "posenet_distortion": posenet_distortion,
                    "pose_marginal_score_contribution": (
                        None if posenet_distortion is None else float(posenet_distortion * pose_marginal_weight)
                    ),
                    "scorer_response_segnet_distortion": scorer_response_segnet_distortion,
                }
            )
    return records


def _confusion_records(
    confusion: np.ndarray,
    *,
    total_pixels: int,
    include_solved: bool,
) -> list[dict[str, Any]]:
    target_pixels_by_class = confusion.sum(axis=1)
    records: list[dict[str, Any]] = []
    for target_class in range(confusion.shape[0]):
        target_pixels = int(target_pixels_by_class[target_class])
        for predicted_class in range(confusion.shape[1]):
            pixel_count = int(confusion[target_class, predicted_class])
            if pixel_count <= 0:
                continue
            solved = target_class == predicted_class
            if solved and not include_solved:
                continue
            records.append(
                {
                    "target_class": int(target_class),
                    "predicted_class": int(predicted_class),
                    "solved": bool(solved),
                    "pixel_count": pixel_count,
                    "unsolved_pixels": 0 if solved else pixel_count,
                    "target_mass_pixels": target_pixels,
                    "target_mass_fraction_of_total": _ratio(target_pixels, total_pixels),
                    "pixel_fraction_of_target_region": _ratio(pixel_count, target_pixels),
                    "score_weighted_unsolved_mass": _score_mass(0 if solved else pixel_count, total_pixels),
                }
            )
    records.sort(
        key=lambda row: (
            -float(row["score_weighted_unsolved_mass"]),
            -int(row["pixel_count"]),
            int(row["target_class"]),
            int(row["predicted_class"]),
        )
    )
    return records


def _top_connected_components(
    *,
    candidate: np.ndarray,
    reference: np.ndarray,
    hard_records: list[dict[str, Any]],
    total_pixels: int,
    pair_pixels: int,
    top_components: int,
) -> list[dict[str, Any]]:
    if int(top_components) <= 0:
        return []
    components: list[dict[str, Any]] = []
    seen_masks: set[tuple[int, int, int]] = set()
    for record in hard_records:
        if record.get("solved") is True:
            continue
        key = (int(record["local_row_index"]), int(record["target_class"]), int(record["predicted_class"]))
        if key in seen_masks:
            continue
        seen_masks.add(key)
        pair_index, target_class, predicted_class = key
        mask = (reference[pair_index] == target_class) & (candidate[pair_index] == predicted_class)
        for component in _connected_components_4(mask):
            components.append(
                {
                    "local_row_index": pair_index,
                    "source_frame_pair": record.get("source_frame_pair"),
                    "target_class": target_class,
                    "predicted_class": predicted_class,
                    "pixel_count": int(component["pixel_count"]),
                    "bbox_y0x0y1x1_exclusive": component["bbox_y0x0y1x1_exclusive"],
                    "centroid_yx": component["centroid_yx"],
                    "score_weighted_unsolved_mass": _score_mass(int(component["pixel_count"]), total_pixels),
                    "pair_score_weighted_unsolved_mass": _score_mass(int(component["pixel_count"]), pair_pixels),
                }
            )
    components.sort(
        key=lambda row: (
            -int(row["pixel_count"]),
            int(row["local_row_index"]),
            int(row["target_class"]),
            int(row["predicted_class"]),
            row["bbox_y0x0y1x1_exclusive"],
        )
    )
    return components[: int(top_components)]


def _connected_components_4(mask: np.ndarray) -> list[dict[str, Any]]:
    mask_bool = np.ascontiguousarray(mask, dtype=bool)
    height, width = (int(v) for v in mask_bool.shape)
    flat_mask = mask_bool.reshape(-1)
    visited = np.zeros(flat_mask.shape, dtype=bool)
    components: list[dict[str, Any]] = []
    for raw_start in np.flatnonzero(flat_mask):
        start = int(raw_start)
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        count = 0
        y_sum = 0
        x_sum = 0
        min_y = height
        min_x = width
        max_y = -1
        max_x = -1
        while stack:
            idx = stack.pop()
            y, x = divmod(idx, width)
            count += 1
            y_sum += y
            x_sum += x
            min_y = min(min_y, y)
            min_x = min(min_x, x)
            max_y = max(max_y, y)
            max_x = max(max_x, x)
            if x > 0:
                _maybe_push(idx - 1, flat_mask, visited, stack)
            if x + 1 < width:
                _maybe_push(idx + 1, flat_mask, visited, stack)
            if y > 0:
                _maybe_push(idx - width, flat_mask, visited, stack)
            if y + 1 < height:
                _maybe_push(idx + width, flat_mask, visited, stack)
        components.append(
            {
                "pixel_count": int(count),
                "bbox_y0x0y1x1_exclusive": [int(min_y), int(min_x), int(max_y + 1), int(max_x + 1)],
                "centroid_yx": [float(y_sum / count), float(x_sum / count)],
            }
        )
    components.sort(key=lambda row: (-int(row["pixel_count"]), row["bbox_y0x0y1x1_exclusive"]))
    return components


def _finite_nonnegative(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(out) or out < 0.0:
        return 0.0
    return out


def _bbox_mask_for_hard_record(
    row: Mapping[str, Any],
    *,
    components: Sequence[Any],
    src_h: int,
    src_w: int,
    dst_h: int,
    dst_w: int,
) -> np.ndarray | None:
    local_row = int(row.get("local_row_index", -1))
    target_class = int(row.get("target_class", -1))
    predicted_class = int(row.get("predicted_class", -1))
    mask = np.zeros((dst_h, dst_w), dtype=bool)
    matched = False
    for raw_component in components:
        if not isinstance(raw_component, Mapping):
            continue
        if (
            int(raw_component.get("local_row_index", -2)) != local_row
            or int(raw_component.get("target_class", -2)) != target_class
            or int(raw_component.get("predicted_class", -2)) != predicted_class
        ):
            continue
        bbox = raw_component.get("bbox_y0x0y1x1_exclusive")
        if not isinstance(bbox, Sequence) or len(bbox) != 4:
            continue
        y0, x0, y1, x1 = (int(v) for v in bbox)
        dy0 = _scale_floor(y0, src_h, dst_h)
        dx0 = _scale_floor(x0, src_w, dst_w)
        dy1 = _scale_ceil(y1, src_h, dst_h)
        dx1 = _scale_ceil(x1, src_w, dst_w)
        dy0 = max(0, min(dst_h, dy0))
        dy1 = max(0, min(dst_h, dy1))
        dx0 = max(0, min(dst_w, dx0))
        dx1 = max(0, min(dst_w, dx1))
        if dy1 <= dy0 or dx1 <= dx0:
            continue
        mask[dy0:dy1, dx0:dx1] = True
        matched = True
    return mask if matched else None


def _scale_floor(value: int, src: int, dst: int) -> int:
    return math.floor(float(value) * float(dst) / float(src))


def _scale_ceil(value: int, src: int, dst: int) -> int:
    return math.ceil(float(value) * float(dst) / float(src))


def _weight_stats(array: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(array, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    return {
        "shape": [int(v) for v in np.asarray(array).shape],
        "dtype": str(np.asarray(array).dtype),
        "min": 0.0 if finite.size == 0 else float(np.min(finite)),
        "max": 0.0 if finite.size == 0 else float(np.max(finite)),
        "mean": 0.0 if finite.size == 0 else float(np.mean(finite)),
        "std": 0.0 if finite.size == 0 else float(np.std(finite)),
        "nonfinite_count": int(arr.size - finite.size),
        "nonzero_fraction": float(np.count_nonzero(arr) / max(int(arr.size), 1)),
    }


def _maybe_push(idx: int, flat_mask: np.ndarray, visited: np.ndarray, stack: list[int]) -> None:
    if not visited[idx] and bool(flat_mask[idx]):
        visited[idx] = True
        stack.append(idx)


def _run_segnet_argmax_batches(
    frames_nchw: np.ndarray,
    *,
    logits_fn: Any,
    batch_frames: int,
) -> np.ndarray:
    if int(batch_frames) < 1:
        raise ReceiverReplayHardRegionError(f"batch_frames must be positive, got {batch_frames}")
    chunks: list[np.ndarray] = []
    for start in range(0, int(frames_nchw.shape[0]), int(batch_frames)):
        chunk = np.ascontiguousarray(frames_nchw[start : start + int(batch_frames)])
        logits = np.asarray(logits_fn(chunk), dtype=np.float32)
        class_axis = _infer_segnet_logits_class_axis(logits)
        logits_last = np.moveaxis(logits, class_axis, -1)
        chunks.append(np.argmax(logits_last, axis=-1).astype(np.int16))
    return np.concatenate(chunks, axis=0)


def _infer_segnet_logits_class_axis(logits: np.ndarray) -> int:
    if logits.ndim != 4:
        raise ReceiverReplayHardRegionError(f"SegNet logits must be rank-4, got shape={logits.shape}")
    if int(logits.shape[1]) == DEFAULT_MIN_CLASS_COUNT:
        return 1
    if int(logits.shape[-1]) == DEFAULT_MIN_CLASS_COUNT:
        return -1
    if 2 <= int(logits.shape[1]) <= 32 and int(logits.shape[2]) > 32:
        return 1
    if 2 <= int(logits.shape[-1]) <= 32 and int(logits.shape[1]) > 32:
        return -1
    raise ReceiverReplayHardRegionError(f"cannot infer SegNet class axis from shape={logits.shape}")


def _build_real_mlx_segnet_logits_fn(*, upstream_dir: Path, device: str) -> Any:
    from tac.local_acceleration.mlx_scorer_adapters import (
        run_mlx_segnet_nchw,
        temporary_mlx_device,
        torch_segnet_to_mlx,
    )
    from tac.scorer import load_default_segnet

    if device not in {"cpu", "gpu"}:
        raise ReceiverReplayHardRegionError(f"MLX SegNet cache mining device must be 'cpu' or 'gpu', got {device!r}")
    segnet = load_default_segnet(upstream_dir, device="cpu")
    segnet.eval()
    adapter = torch_segnet_to_mlx(segnet)

    def logits_fn(x_nchw: np.ndarray) -> np.ndarray:
        with temporary_mlx_device(device):
            return np.asarray(run_mlx_segnet_nchw(adapter, x_nchw), dtype=np.float32)

    return logits_fn


def _pose_marginal_weight(pose: np.ndarray | None) -> float:
    if pose is None or pose.size == 0:
        return 0.0
    avg_pose = float(np.mean(pose, dtype=np.float64))
    if not math.isfinite(avg_pose) or avg_pose <= 0.0:
        return 0.0
    return 5.0 / math.sqrt(10.0 * avg_pose)


def _score_mass(pixel_count: int, total_pixels: int) -> float:
    return SEGNET_SCORE_WEIGHT * _ratio(pixel_count, total_pixels)


def _ratio(num: int | float, den: int | float) -> float:
    denominator = float(den)
    if denominator <= 0.0:
        return 0.0
    return float(num) / denominator


def _source_frame_pair(pair_indices: np.ndarray | None, row_index: int) -> list[int] | None:
    if pair_indices is None:
        return None
    value = pair_indices[row_index]
    if np.asarray(value).ndim == 0:
        return [int(value)]
    return [int(v) for v in np.asarray(value).reshape(-1).tolist()]


def _hard_record_sort_key(row: Mapping[str, Any]) -> tuple[float, int, int, int, int]:
    return (
        -float(row["score_weighted_unsolved_mass"]),
        -int(row["pixel_count"]),
        int(row["local_row_index"]),
        int(row["target_class"]),
        int(row["predicted_class"]),
    )


def _matrix_to_jsonable(matrix: np.ndarray) -> list[list[int]]:
    return [[int(v) for v in row] for row in np.asarray(matrix).tolist()]


def _array_sha256(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode("utf-8"))
    digest.update(json.dumps([int(v) for v in arr.shape]).encode("utf-8"))
    digest.update(memoryview(arr).cast("B"))
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_record(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=False)
    if not resolved.is_file():
        raise FileNotFoundError(f"artifact does not exist: {resolved}")
    return {
        "path": resolved.as_posix(),
        "bytes": int(resolved.stat().st_size),
        "sha256": _file_sha256(resolved),
    }


def _optional_artifact_record(path: Path) -> dict[str, Any] | None:
    return _artifact_record(path) if path.is_file() else None


def _first_path_value(payload: Mapping[str, Any], dotted_keys: Sequence[str]) -> str | None:
    for dotted in dotted_keys:
        value: Any = payload
        for part in dotted.split("."):
            if not isinstance(value, Mapping) or part not in value:
                value = None
                break
            value = value[part]
        if value:
            return str(value)
    return None


__all__ = [
    "FALSE_AUTHORITY",
    "SCHEMA",
    "ReceiverReplayHardRegionError",
    "build_hard_region_recon_pixel_weight",
    "build_receiver_replay_scorer_hard_region_report",
    "build_segnet_argmax_arrays_from_cache_dirs",
    "component_artifacts_from_mlx_response",
    "infer_cache_dirs_from_mlx_response",
    "load_argmax_array",
    "load_component_vector",
    "load_component_vectors_from_dir",
    "load_component_vectors_from_mlx_response",
    "load_pair_indices",
    "write_hard_region_recon_pixel_weight_artifact",
    "write_receiver_replay_scorer_hard_region_report",
]
