#!/usr/bin/env python3
"""Alpha-Geo-0 CPU geometry diagnostics for NeRV mask streams.

This compares a decoded candidate mask stream against the decoded baseline
archive mask stream. It intentionally stays below scorer/PoseNet evaluation:
all metrics operate on integer class tensors on CPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


CLASS_NAMES: dict[int, str] = {
    0: "road",
    1: "lane_marking",
    2: "vehicle_undrivable",
    3: "sky_or_movable",
    4: "background",
}
DEFAULT_BOUNDARY_RADII: tuple[int, ...] = (1, 2, 3, 5)


@dataclass(frozen=True)
class GeometryThresholds:
    """Pass/fail gates for Alpha tensor-only geometry diagnostics."""

    global_disagreement_max: float | None = 0.003
    boundary_band_disagreement_max: dict[int, float] = field(
        default_factory=lambda: {1: 0.005, 2: 0.005, 3: 0.005, 5: 0.005}
    )
    stable_region_false_flip_rate_max: float | None = 0.004
    pair_transition_disagreement_max: float | None = 0.004
    pair_transition_f1_min: float | None = None
    class_recall_min: dict[int, float] = field(default_factory=dict)
    tiny_speckle_rate_max: float | None = 0.0005
    max_component_centroid_jump_px: float | None = 1.0
    missing_component_rate_max: float | None = 0.0

    @classmethod
    def from_preset(cls, preset: str) -> "GeometryThresholds":
        if preset == "none":
            return cls(
                global_disagreement_max=None,
                boundary_band_disagreement_max={},
                stable_region_false_flip_rate_max=None,
                pair_transition_disagreement_max=None,
                pair_transition_f1_min=None,
                class_recall_min={},
                tiny_speckle_rate_max=None,
                max_component_centroid_jump_px=None,
                missing_component_rate_max=None,
            )
        if preset == "exploratory":
            return cls()
        if preset == "promotion":
            return cls(
                global_disagreement_max=0.001,
                boundary_band_disagreement_max={1: 0.002, 2: 0.002, 3: 0.002, 5: 0.002},
                stable_region_false_flip_rate_max=0.002,
                pair_transition_disagreement_max=0.002,
                pair_transition_f1_min=None,
                class_recall_min={1: 0.999, 2: 0.999},
                tiny_speckle_rate_max=0.0001,
                max_component_centroid_jump_px=1.0,
                missing_component_rate_max=0.0,
            )
        raise ValueError(f"unknown threshold preset: {preset!r}")


@dataclass(frozen=True)
class ComponentOptions:
    tiny_component_max_area: int = 8
    speckle_mismatch_fraction_min: float = 0.5
    centroid_component_min_area: int = 9


@dataclass(frozen=True)
class ResidualRankingOptions:
    """CPU-only residual repair ranking for decoded mask disagreements."""

    max_regions: int = 20
    min_area: int = 1
    boundary_radius: int = 2
    near_field_y_fraction: float = 0.60
    critical_classes: tuple[int, ...] = (1, 2)


@dataclass(frozen=True)
class VisualPrimitiveOptions:
    """CPU-only Alpha visual primitive extraction options."""

    frame_stride: int = 1
    component_connectivity: int = 4
    critical_component_min_area: int = 9
    tiny_component_max_area: int = 8
    polyline_ordered: bool = False
    boundary_distance_sample_cap: int = 4096
    boundary_distance_global_sample_cap: int = 262_144
    track_min_iou: float = 0.05
    track_max_centroid_jump_px: float = 16.0
    pose_sensitive_y_fraction_min: float = 0.60
    max_component_failures: int = 20
    critical_classes: tuple[int, ...] = (1, 2)
    component_classes: tuple[int, ...] = ()
    track_classes: tuple[int, ...] = ()
    temporal_tracks_enabled: bool = True


def _as_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _as_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_as_jsonable(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, torch.Tensor):
        return _as_jsonable(value.detach().cpu().tolist())
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _safe_rate(numer: int, denom: int, *, empty_value: float = 0.0) -> float:
    return empty_value if denom == 0 else float(numer) / float(denom)


def _safe_f1(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = _safe_rate(tp, tp + fp, empty_value=1.0)
    recall = _safe_rate(tp, tp + fn, empty_value=1.0)
    denom = precision + recall
    f1 = 1.0 if denom == 0 else 2.0 * precision * recall / denom
    return {"precision": precision, "recall": recall, "f1": f1}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _mask_tensor_sha256(masks: torch.Tensor) -> str:
    normalized = _normalize_mask_tensor(masks, name="mask_sha256")
    digest = hashlib.sha256()
    digest.update(str(tuple(normalized.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(normalized.dtype).encode("ascii"))
    digest.update(b"\0")
    arr = normalized.contiguous().numpy()
    digest.update(memoryview(arr))
    return digest.hexdigest()


def _validated_zip_infos(zf: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    infos: dict[str, zipfile.ZipInfo] = {}
    for info in zf.infolist():
        member_path = PurePosixPath(info.filename)
        if member_path.is_absolute() or ".." in member_path.parts:
            raise ValueError(f"unsafe archive member path: {info.filename!r}")
        if info.filename in infos:
            raise ValueError(f"duplicate archive member: {info.filename!r}")
        infos[info.filename] = info
    return infos


def _normalize_mask_tensor(masks: torch.Tensor, *, name: str) -> torch.Tensor:
    if not isinstance(masks, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor; got {type(masks).__name__}")
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3:
        raise ValueError(f"{name} must have shape (T,H,W); got {tuple(masks.shape)}")
    if masks.numel() == 0:
        raise ValueError(f"{name} must be non-empty")
    if torch.is_floating_point(masks):
        if not torch.equal(masks, masks.round()):
            raise ValueError(f"{name} floating tensor contains non-integer class IDs")
        masks = masks.round()
    masks = masks.detach().cpu().to(torch.int64)
    if int(masks.min().item()) < 0:
        raise ValueError(f"{name} contains negative class IDs")
    max_id = int(masks.max().item())
    if max_id <= 255:
        return masks.to(torch.uint8).contiguous()
    return masks.contiguous()


def _validate_pair(
    baseline: torch.Tensor,
    candidate: torch.Tensor,
    *,
    num_classes: int | None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    baseline = _normalize_mask_tensor(baseline, name="baseline")
    candidate = _normalize_mask_tensor(candidate, name="candidate")
    if tuple(baseline.shape) != tuple(candidate.shape):
        raise ValueError(
            f"baseline and candidate shapes must match; got {tuple(baseline.shape)} vs {tuple(candidate.shape)}"
        )
    observed_classes = max(int(baseline.max().item()), int(candidate.max().item())) + 1
    if num_classes is None:
        num_classes = max(5, observed_classes)
    if num_classes <= 0:
        raise ValueError(f"num_classes must be > 0; got {num_classes}")
    if observed_classes > num_classes:
        raise ValueError(
            f"observed class ID {observed_classes - 1} exceeds num_classes={num_classes}"
        )
    return baseline, candidate, int(num_classes)


def _frame_chunks(num_frames: int, chunk_frames: int) -> list[tuple[int, int]]:
    if chunk_frames <= 0:
        raise ValueError(f"chunk_frames must be > 0; got {chunk_frames}")
    return [(start, min(start + chunk_frames, num_frames)) for start in range(0, num_frames, chunk_frames)]


def _compute_boundary_mask(masks: torch.Tensor) -> torch.Tensor:
    boundary = torch.zeros_like(masks, dtype=torch.bool)
    boundary[:, :, 1:] |= masks[:, :, 1:] != masks[:, :, :-1]
    boundary[:, :, :-1] |= masks[:, :, 1:] != masks[:, :, :-1]
    boundary[:, 1:, :] |= masks[:, 1:, :] != masks[:, :-1, :]
    boundary[:, :-1, :] |= masks[:, 1:, :] != masks[:, :-1, :]
    return boundary


def _dilate_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius < 0:
        raise ValueError(f"boundary radius must be >= 0; got {radius}")
    if radius == 0:
        return mask
    pooled = F.max_pool2d(
        mask.to(torch.float32).unsqueeze(1),
        kernel_size=2 * radius + 1,
        stride=1,
        padding=radius,
    )
    return pooled[:, 0] > 0


def _global_and_confusion(
    baseline: torch.Tensor,
    candidate: torch.Tensor,
    *,
    num_classes: int,
    chunk_frames: int,
) -> tuple[dict[str, Any], torch.Tensor]:
    total = int(baseline.numel())
    disagree = 0
    confusion_flat = torch.zeros(num_classes * num_classes, dtype=torch.int64)
    for start, end in _frame_chunks(baseline.shape[0], chunk_frames):
        b = baseline[start:end].to(torch.int64)
        c = candidate[start:end].to(torch.int64)
        disagree += int((b != c).sum().item())
        idx = (b * num_classes + c).reshape(-1)
        confusion_flat += torch.bincount(idx, minlength=num_classes * num_classes)

    confusion = confusion_flat.reshape(num_classes, num_classes)
    global_metrics = {
        "total_pixels": total,
        "disagreement_pixels": disagree,
        "global_disagreement": _safe_rate(disagree, total),
        "global_agreement": 1.0 - _safe_rate(disagree, total),
    }
    return global_metrics, confusion


def _class_metrics(confusion: torch.Tensor) -> dict[str, Any]:
    num_classes = confusion.shape[0]
    rows = confusion.sum(dim=1)
    cols = confusion.sum(dim=0)
    total = int(confusion.sum().item())
    row_normalized: list[list[float | None]] = []
    classes: dict[str, Any] = {}
    for cls in range(num_classes):
        support = int(rows[cls].item())
        predicted = int(cols[cls].item())
        tp = int(confusion[cls, cls].item())
        precision = None if predicted == 0 else float(tp) / float(predicted)
        recall = None if support == 0 else float(tp) / float(support)
        if precision is None or recall is None:
            f1 = None
        else:
            f1 = 1.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
        classes[str(cls)] = {
            "name": CLASS_NAMES.get(cls, f"class_{cls}"),
            "support_pixels": support,
            "predicted_pixels": predicted,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        row = []
        for pred in range(num_classes):
            row.append(None if support == 0 else float(confusion[cls, pred].item()) / float(support))
        row_normalized.append(row)

    return {
        "confusion_matrix": confusion.tolist(),
        "confusion_matrix_row_normalized": row_normalized,
        "classes": classes,
        "total_pixels": total,
    }


def _boundary_metrics(
    baseline: torch.Tensor,
    candidate: torch.Tensor,
    *,
    radii: tuple[int, ...],
    chunk_frames: int,
) -> dict[str, Any]:
    accum = {int(radius): {"band_pixels": 0, "disagreement_pixels": 0} for radius in radii}
    for start, end in _frame_chunks(baseline.shape[0], chunk_frames):
        b = baseline[start:end]
        c = candidate[start:end]
        diff = b != c
        boundary = _compute_boundary_mask(b)
        for radius in radii:
            band = _dilate_mask(boundary, int(radius))
            accum[int(radius)]["band_pixels"] += int(band.sum().item())
            accum[int(radius)]["disagreement_pixels"] += int((diff & band).sum().item())

    out: dict[str, Any] = {}
    for radius in sorted(accum):
        band_pixels = accum[radius]["band_pixels"]
        disagreement_pixels = accum[radius]["disagreement_pixels"]
        out[str(radius)] = {
            "radius_px": radius,
            "band_pixels": band_pixels,
            "disagreement_pixels": disagreement_pixels,
            "band_fraction": _safe_rate(band_pixels, int(baseline.numel())),
            "disagreement_rate": _safe_rate(disagreement_pixels, band_pixels),
        }
    return out


def _temporal_metrics(
    baseline: torch.Tensor,
    candidate: torch.Tensor,
    *,
    worst_pair_count: int,
    chunk_frames: int,
) -> dict[str, Any]:
    frames, height, width = baseline.shape
    pixels_per_frame = int(height * width)
    if frames < 2:
        return {
            "num_pairs": 0,
            "stable_region": {
                "stable_pixels": 0,
                "false_flip_pixels": 0,
                "false_flip_rate": 0.0,
            },
            "pair_transition": {
                "tp_pixels": 0,
                "fp_pixels": 0,
                "fn_pixels": 0,
                "tn_pixels": 0,
                "disagreement_pixels": 0,
                "total_pixels": 0,
                "disagreement_rate": 0.0,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
            },
            "worst_frame_pairs": [],
        }

    stable_pixels = 0
    false_flip_pixels = 0
    tp = fp = fn = tn = 0
    pair_rows: list[dict[str, Any]] = []
    for start in range(0, frames - 1, chunk_frames):
        end_pair = min(start + chunk_frames, frames - 1)
        b = baseline[start : end_pair + 1]
        c = candidate[start : end_pair + 1]
        base_transition = b[1:] != b[:-1]
        cand_transition = c[1:] != c[:-1]
        stable = ~base_transition
        false_flips = stable & cand_transition
        transition_disagree = base_transition ^ cand_transition

        stable_pixels += int(stable.sum().item())
        false_flip_pixels += int(false_flips.sum().item())
        tp += int((base_transition & cand_transition).sum().item())
        fp += int((~base_transition & cand_transition).sum().item())
        fn += int((base_transition & ~cand_transition).sum().item())
        tn += int((~base_transition & ~cand_transition).sum().item())

        pair_count = end_pair - start
        frame_diff = b != c
        pair_frame_disagree = (
            frame_diff[:-1].reshape(pair_count, -1).sum(dim=1)
            + frame_diff[1:].reshape(pair_count, -1).sum(dim=1)
        )
        transition_disagree_counts = transition_disagree.reshape(pair_count, -1).sum(dim=1)
        false_flip_counts = false_flips.reshape(pair_count, -1).sum(dim=1)
        stable_counts = stable.reshape(pair_count, -1).sum(dim=1)
        for local_idx in range(pair_count):
            stable_count = int(stable_counts[local_idx].item())
            false_count = int(false_flip_counts[local_idx].item())
            transition_count = int(transition_disagree_counts[local_idx].item())
            pair_rows.append(
                {
                    "pair_index": start + local_idx,
                    "frames": [start + local_idx, start + local_idx + 1],
                    "pair_frame_disagreement_rate": float(pair_frame_disagree[local_idx].item())
                    / float(2 * pixels_per_frame),
                    "transition_disagreement_rate": float(transition_count) / float(pixels_per_frame),
                    "stable_false_flip_rate": _safe_rate(false_count, stable_count),
                    "transition_disagreement_pixels": transition_count,
                    "stable_false_flip_pixels": false_count,
                }
            )

    f1_metrics = _safe_f1(tp, fp, fn)
    total_transition_pixels = (frames - 1) * pixels_per_frame
    transition_disagreement_pixels = fp + fn
    pair_rows.sort(
        key=lambda row: (
            row["transition_disagreement_rate"],
            row["stable_false_flip_rate"],
            row["pair_frame_disagreement_rate"],
        ),
        reverse=True,
    )
    return {
        "num_pairs": frames - 1,
        "stable_region": {
            "stable_pixels": stable_pixels,
            "false_flip_pixels": false_flip_pixels,
            "false_flip_rate": _safe_rate(false_flip_pixels, stable_pixels),
        },
        "pair_transition": {
            "tp_pixels": tp,
            "fp_pixels": fp,
            "fn_pixels": fn,
            "tn_pixels": tn,
            "disagreement_pixels": transition_disagreement_pixels,
            "total_pixels": total_transition_pixels,
            "disagreement_rate": _safe_rate(transition_disagreement_pixels, total_transition_pixels),
            **f1_metrics,
        },
        "worst_frame_pairs": pair_rows[: max(0, worst_pair_count)],
    }


def _load_scipy_ndimage():
    try:
        from scipy import ndimage as ndi

        return ndi
    except Exception:
        return None


def _label_components_fallback(mask: np.ndarray) -> tuple[np.ndarray, int]:
    labels = np.zeros(mask.shape, dtype=np.int32)
    current = 0
    height, width = mask.shape
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or labels[y, x] != 0:
                continue
            current += 1
            stack = [(y, x)]
            labels[y, x] = current
            while stack:
                cy, cx = stack.pop()
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and labels[ny, nx] == 0:
                        labels[ny, nx] = current
                        stack.append((ny, nx))
    return labels, current


def _label_components(mask: np.ndarray, ndi: Any) -> tuple[np.ndarray, int]:
    if ndi is None:
        return _label_components_fallback(mask)
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    labels, count = ndi.label(mask, structure=structure)
    return labels.astype(np.int32, copy=False), int(count)


def _boundary_mask_np(mask: np.ndarray) -> np.ndarray:
    boundary = np.zeros(mask.shape, dtype=bool)
    boundary[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    boundary[:, :-1] |= mask[:, 1:] != mask[:, :-1]
    boundary[1:, :] |= mask[1:, :] != mask[:-1, :]
    boundary[:-1, :] |= mask[1:, :] != mask[:-1, :]
    return boundary


def _dilate_mask_np(mask: np.ndarray, radius: int, ndi: Any) -> np.ndarray:
    if radius < 0:
        raise ValueError(f"residual boundary radius must be >= 0; got {radius}")
    if radius == 0:
        return mask
    if ndi is not None:
        structure = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
        return ndi.binary_dilation(mask, structure=structure).astype(bool, copy=False)
    height, width = mask.shape
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    out = np.zeros_like(mask, dtype=bool)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            out |= padded[dy : dy + height, dx : dx + width]
    return out


def _component_centroids(labels: np.ndarray, count: int) -> dict[int, tuple[float, float]]:
    if count == 0:
        return {}
    yy, xx = np.indices(labels.shape)
    flat_labels = labels.reshape(-1)
    area = np.bincount(flat_labels, minlength=count + 1).astype(np.float64)
    sum_y = np.bincount(flat_labels, weights=yy.reshape(-1), minlength=count + 1)
    sum_x = np.bincount(flat_labels, weights=xx.reshape(-1), minlength=count + 1)
    centroids: dict[int, tuple[float, float]] = {}
    for label_id in range(1, count + 1):
        if area[label_id] > 0:
            centroids[label_id] = (float(sum_y[label_id] / area[label_id]), float(sum_x[label_id] / area[label_id]))
    return centroids


def _component_metrics(
    baseline: torch.Tensor,
    candidate: torch.Tensor,
    *,
    num_classes: int,
    options: ComponentOptions,
) -> dict[str, Any]:
    ndi = _load_scipy_ndimage()
    baseline_np = baseline.numpy()
    candidate_np = candidate.numpy()
    total_pixels = int(baseline.numel())
    speckle_pixels = 0
    speckle_components = 0
    candidate_tiny_components = 0
    candidate_components = 0
    matched_components = 0
    missing_components = 0
    centroid_jumps: list[float] = []
    per_class: dict[str, Any] = {
        str(cls): {
            "name": CLASS_NAMES.get(cls, f"class_{cls}"),
            "candidate_components": 0,
            "candidate_tiny_components": 0,
            "speckle_components": 0,
            "speckle_pixels": 0,
            "matched_components": 0,
            "missing_components": 0,
        }
        for cls in range(num_classes)
    }

    for frame_idx in range(baseline_np.shape[0]):
        base_frame = baseline_np[frame_idx]
        cand_frame = candidate_np[frame_idx]
        for cls in range(num_classes):
            base_mask = base_frame == cls
            cand_mask = cand_frame == cls
            cand_labels, cand_count = _label_components(cand_mask, ndi)
            candidate_components += cand_count
            per_class[str(cls)]["candidate_components"] += cand_count
            if cand_count:
                areas = np.bincount(cand_labels.reshape(-1), minlength=cand_count + 1)
                for label_id in range(1, cand_count + 1):
                    area = int(areas[label_id])
                    if area <= options.tiny_component_max_area:
                        candidate_tiny_components += 1
                        per_class[str(cls)]["candidate_tiny_components"] += 1
                        component = cand_labels == label_id
                        mismatch = int(np.count_nonzero(base_frame[component] != cls))
                        if area > 0 and mismatch / area >= options.speckle_mismatch_fraction_min:
                            speckle_components += 1
                            speckle_pixels += area
                            per_class[str(cls)]["speckle_components"] += 1
                            per_class[str(cls)]["speckle_pixels"] += area

            base_labels, base_count = _label_components(base_mask, ndi)
            if base_count == 0:
                continue
            base_areas = np.bincount(base_labels.reshape(-1), minlength=base_count + 1)
            base_centroids = _component_centroids(base_labels, base_count)
            cand_centroids = _component_centroids(cand_labels, cand_count)
            for base_label in range(1, base_count + 1):
                if int(base_areas[base_label]) < options.centroid_component_min_area:
                    continue
                component = base_labels == base_label
                overlaps = np.bincount(cand_labels[component], minlength=cand_count + 1)
                if overlaps.shape[0] <= 1 or int(overlaps[1:].max(initial=0)) == 0:
                    missing_components += 1
                    per_class[str(cls)]["missing_components"] += 1
                    continue
                cand_label = int(np.argmax(overlaps[1:]) + 1)
                by, bx = base_centroids[base_label]
                cy, cx = cand_centroids[cand_label]
                jump = math.hypot(cy - by, cx - bx)
                centroid_jumps.append(float(jump))
                matched_components += 1
                per_class[str(cls)]["matched_components"] += 1

    considered_components = matched_components + missing_components
    centroid_jumps_sorted = sorted(centroid_jumps)
    p95_jump = 0.0
    if centroid_jumps_sorted:
        p95_idx = min(len(centroid_jumps_sorted) - 1, int(math.ceil(0.95 * len(centroid_jumps_sorted))) - 1)
        p95_jump = centroid_jumps_sorted[p95_idx]
    return {
        "options": asdict(options),
        "scipy_ndimage": ndi is not None,
        "candidate_components": candidate_components,
        "candidate_tiny_components": candidate_tiny_components,
        "speckle_components": speckle_components,
        "speckle_pixels": speckle_pixels,
        "speckle_rate": _safe_rate(speckle_pixels, total_pixels),
        "centroid": {
            "matched_components": matched_components,
            "missing_components": missing_components,
            "missing_component_rate": _safe_rate(missing_components, considered_components),
            "mean_matched_jump_px": float(np.mean(centroid_jumps)) if centroid_jumps else 0.0,
            "p95_matched_jump_px": p95_jump,
            "max_matched_jump_px": max(centroid_jumps) if centroid_jumps else 0.0,
            "over_1px_rate": _safe_rate(sum(1 for value in centroid_jumps if value > 1.0), len(centroid_jumps)),
        },
        "per_class": per_class,
    }


def _class_histogram(values: np.ndarray, *, num_classes: int) -> dict[str, int]:
    counts = np.bincount(values.astype(np.int64, copy=False), minlength=num_classes)
    return {str(cls): int(counts[cls]) for cls in range(num_classes) if int(counts[cls]) > 0}


def _confusion_pairs_for_region(
    baseline_values: np.ndarray,
    candidate_values: np.ndarray,
    *,
    num_classes: int,
    max_pairs: int = 8,
) -> list[dict[str, Any]]:
    flat = baseline_values.astype(np.int64, copy=False) * num_classes + candidate_values.astype(
        np.int64,
        copy=False,
    )
    counts = np.bincount(flat, minlength=num_classes * num_classes)
    pairs: list[dict[str, Any]] = []
    for pair_idx, count in enumerate(counts):
        if int(count) == 0:
            continue
        baseline_class = pair_idx // num_classes
        candidate_class = pair_idx % num_classes
        pairs.append(
            {
                "baseline_class": int(baseline_class),
                "baseline_name": CLASS_NAMES.get(int(baseline_class), f"class_{baseline_class}"),
                "candidate_class": int(candidate_class),
                "candidate_name": CLASS_NAMES.get(int(candidate_class), f"class_{candidate_class}"),
                "pixels": int(count),
            }
        )
    pairs.sort(key=lambda row: (-row["pixels"], row["baseline_class"], row["candidate_class"]))
    return pairs[:max(0, max_pairs)]


def _residual_priority(
    *,
    baseline_hist: dict[str, int],
    candidate_hist: dict[str, int],
    boundary_pixels: int,
    temporal_pixels: int,
    lower_field: bool,
) -> tuple[int, str, str]:
    lane_pixels = baseline_hist.get("1", 0) + candidate_hist.get("1", 0)
    vehicle_pixels = baseline_hist.get("2", 0) + candidate_hist.get("2", 0)
    road_pixels = baseline_hist.get("0", 0) + candidate_hist.get("0", 0)
    if lane_pixels > 0 and lower_field:
        return 0, "lower_field_lane_marking", "lane_lower_field_residual"
    if lane_pixels > 0:
        return 1, "lane_marking", "lane_component_residual"
    if boundary_pixels > 0 and (road_pixels > 0 or vehicle_pixels > 0):
        return 2, "road_or_vehicle_boundary", "boundary_band_residual"
    if vehicle_pixels > 0 and lower_field:
        return 3, "near_field_vehicle_undrivable", "near_vehicle_box_residual"
    if temporal_pixels > 0:
        return 4, "temporal_transition", "temporal_pair_residual"
    if boundary_pixels > 0:
        return 5, "remaining_boundary_band", "boundary_band_residual"
    return 6, "other_class_disagreement", "class_residual"


def _residual_region_ranking(
    baseline: torch.Tensor,
    candidate: torch.Tensor,
    *,
    num_classes: int,
    options: ResidualRankingOptions,
) -> dict[str, Any]:
    if options.max_regions < 0:
        raise ValueError(f"max residual regions must be >= 0; got {options.max_regions}")
    if options.max_regions == 0:
        return {
            "schema_version": 1,
            "diagnostic": "alpha_geo_residual_region_ranking",
            "score_evidence_grade": "empirical",
            "device": "cpu",
            "scorer_proxy": False,
            "promotion_eligible": False,
            "score_claim_eligible": False,
            "exact_score_source": "experiments/contest_auth_eval.py --device cuda on exact archive bytes",
            "options": _as_jsonable(asdict(options)),
            "scan_skipped": True,
            "skip_reason": "max_regions=0",
            "regions_considered": 0,
            "regions_filtered_below_min_area": 0,
            "regions_returned": 0,
            "regions": [],
        }
    if options.min_area <= 0:
        raise ValueError(f"residual min_area must be > 0; got {options.min_area}")
    if not 0.0 <= options.near_field_y_fraction <= 1.0:
        raise ValueError(
            "residual near_field_y_fraction must be between 0 and 1; "
            f"got {options.near_field_y_fraction}"
        )

    ndi = _load_scipy_ndimage()
    baseline_np = baseline.numpy()
    candidate_np = candidate.numpy()
    frames, height, width = baseline_np.shape
    regions: list[dict[str, Any]] = []
    filtered_below_min_area = 0
    near_field_y = options.near_field_y_fraction * float(height)
    critical_classes = tuple(int(cls) for cls in options.critical_classes)
    critical_set = set(critical_classes)

    for frame_idx in range(frames):
        base_frame = baseline_np[frame_idx]
        cand_frame = candidate_np[frame_idx]
        diff = base_frame != cand_frame
        labels, count = _label_components(diff, ndi)
        if count == 0:
            continue

        areas = np.bincount(labels.reshape(-1), minlength=count + 1)
        boundary_band = _dilate_mask_np(
            _boundary_mask_np(base_frame),
            int(options.boundary_radius),
            ndi,
        )
        temporal_disagree = np.zeros((height, width), dtype=bool)
        if frame_idx > 0:
            temporal_disagree |= (
                (baseline_np[frame_idx] != baseline_np[frame_idx - 1])
                ^ (candidate_np[frame_idx] != candidate_np[frame_idx - 1])
            )
        if frame_idx + 1 < frames:
            temporal_disagree |= (
                (baseline_np[frame_idx + 1] != baseline_np[frame_idx])
                ^ (candidate_np[frame_idx + 1] != candidate_np[frame_idx])
            )

        for label_id in range(1, count + 1):
            area = int(areas[label_id])
            if area < options.min_area:
                filtered_below_min_area += 1
                continue

            region = labels == label_id
            ys, xs = np.nonzero(region)
            y0 = int(ys.min())
            y1 = int(ys.max()) + 1
            x0 = int(xs.min())
            x1 = int(xs.max()) + 1
            baseline_values = base_frame[region]
            candidate_values = cand_frame[region]
            baseline_hist = _class_histogram(baseline_values, num_classes=num_classes)
            candidate_hist = _class_histogram(candidate_values, num_classes=num_classes)
            boundary_pixels = int(np.count_nonzero(region & boundary_band))
            temporal_pixels = int(np.count_nonzero(region & temporal_disagree))
            lower_field = bool(y1 >= near_field_y)
            critical_pixels = int(
                sum(baseline_hist.get(str(cls), 0) + candidate_hist.get(str(cls), 0) for cls in critical_set)
            )
            priority_bucket, priority_label, suggested_repair = _residual_priority(
                baseline_hist=baseline_hist,
                candidate_hist=candidate_hist,
                boundary_pixels=boundary_pixels,
                temporal_pixels=temporal_pixels,
                lower_field=lower_field,
            )
            dominant_baseline = max(
                ((int(cls), count) for cls, count in baseline_hist.items()),
                key=lambda item: (item[1], -item[0]),
            )[0]
            dominant_candidate = max(
                ((int(cls), count) for cls, count in candidate_hist.items()),
                key=lambda item: (item[1], -item[0]),
            )[0]
            sort_key = (
                int(priority_bucket),
                -int(critical_pixels),
                -int(temporal_pixels),
                -int(boundary_pixels),
                -int(area),
                int(frame_idx),
                int(y0),
                int(x0),
            )
            regions.append(
                {
                    "rank": None,
                    "residual_region_id": f"f{frame_idx:04d}_c{label_id:04d}",
                    "frame": int(frame_idx),
                    "frame_range": [int(frame_idx), int(frame_idx)],
                    "box_xyxy": [int(x0), int(y0), int(x1), int(y1)],
                    "centroid_xy": [float(xs.mean()), float(ys.mean())],
                    "area_px": int(area),
                    "area_fraction_of_frame": float(area) / float(height * width),
                    "baseline_class_hist": baseline_hist,
                    "candidate_class_hist": candidate_hist,
                    "dominant_baseline_class": int(dominant_baseline),
                    "dominant_baseline_name": CLASS_NAMES.get(
                        int(dominant_baseline),
                        f"class_{dominant_baseline}",
                    ),
                    "dominant_candidate_class": int(dominant_candidate),
                    "dominant_candidate_name": CLASS_NAMES.get(
                        int(dominant_candidate),
                        f"class_{dominant_candidate}",
                    ),
                    "confusion_pairs": _confusion_pairs_for_region(
                        baseline_values,
                        candidate_values,
                        num_classes=num_classes,
                    ),
                    "critical_class_pixels": int(critical_pixels),
                    "boundary_band_pixels": int(boundary_pixels),
                    "boundary_band_fraction": _safe_rate(boundary_pixels, area),
                    "temporal_transition_disagreement_pixels": int(temporal_pixels),
                    "temporal_transition_disagreement_fraction": _safe_rate(temporal_pixels, area),
                    "lower_field": lower_field,
                    "priority_bucket": int(priority_bucket),
                    "priority_label": priority_label,
                    "suggested_repair": suggested_repair,
                    "estimated_uncompressed_pixels": int(area),
                    "priority_sort_key": [int(v) for v in sort_key],
                    "_sort_key": sort_key,
                }
            )

    regions.sort(key=lambda row: row["_sort_key"])
    kept = regions[: options.max_regions]
    for rank, row in enumerate(kept, start=1):
        row["rank"] = rank
        del row["_sort_key"]

    return {
        "schema_version": 1,
        "diagnostic": "alpha_geo_residual_region_ranking",
        "score_evidence_grade": "empirical",
        "device": "cpu",
        "scorer_proxy": False,
        "promotion_eligible": False,
        "score_claim_eligible": False,
        "exact_score_source": "experiments/contest_auth_eval.py --device cuda on exact archive bytes",
        "options": _as_jsonable(asdict(options)),
        "ranking_key": [
            "priority_bucket ascending",
            "critical_class_pixels descending",
            "temporal_transition_disagreement_pixels descending",
            "boundary_band_pixels descending",
            "area_px descending",
            "frame ascending",
            "y0 ascending",
            "x0 ascending",
        ],
        "priority_buckets": {
            "0": "lower_field_lane_marking",
            "1": "lane_marking",
            "2": "road_or_vehicle_boundary",
            "3": "near_field_vehicle_undrivable",
            "4": "temporal_transition",
            "5": "remaining_boundary_band",
            "6": "other_class_disagreement",
        },
        "regions_considered": int(len(regions)),
        "regions_filtered_below_min_area": int(filtered_below_min_area),
        "regions_returned": int(len(kept)),
        "regions": kept,
    }


def _percentile(values: list[float] | np.ndarray, q: float, *, default: float | None = None) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return default
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return default
    return float(np.percentile(arr, q))


def _stable_u64_hash(indices: np.ndarray, *, seed: int) -> np.ndarray:
    """Vectorized SplitMix64-style hash for deterministic bounded sampling."""

    x = indices.astype(np.uint64, copy=False) + np.uint64(seed) + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def _update_distance_sample_reservoir(
    row: dict[str, Any],
    samples: np.ndarray,
    *,
    cap: int,
    seed: int,
) -> None:
    """Keep a deterministic bounded sample of boundary distances.

    The scalar chamfer and coverage accumulators remain exact when scipy EDT is
    available. This reservoir only bounds the advisory p95 sample population.
    """

    if cap < 0:
        raise ValueError(f"boundary_distance_global_sample_cap must be >= 0; got {cap}")
    sample_count = int(samples.size)
    row["distance_sample_population_seen"] += sample_count
    if sample_count == 0 or cap == 0:
        return

    start = int(row["distance_sample_ordinal"])
    row["distance_sample_ordinal"] = start + sample_count
    indices = np.arange(start, start + sample_count, dtype=np.uint64)
    new_keys = _stable_u64_hash(indices, seed=seed)
    new_values = samples.astype(np.float64, copy=False)

    existing_keys = row.get("sample_keys")
    existing_values = row.get("sample_values")
    if existing_keys is None or existing_values is None:
        combined_keys = new_keys
        combined_values = new_values
    else:
        combined_keys = np.concatenate([existing_keys, new_keys])
        combined_values = np.concatenate([existing_values, new_values])

    if combined_keys.size > cap:
        keep = np.argpartition(combined_keys, cap - 1)[:cap]
        combined_keys = combined_keys[keep]
        combined_values = combined_values[keep]

    row["sample_keys"] = combined_keys
    row["sample_values"] = combined_values


def _binary_perimeter_pixels(mask: np.ndarray) -> int:
    if not np.any(mask):
        return 0
    padded = np.pad(mask.astype(bool, copy=False), 1, mode="constant", constant_values=False)
    center = padded[1:-1, 1:-1]
    interior = (
        center
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    return int(np.count_nonzero(center & ~interior))


def _box_iou_xyxy(a: list[int], b: list[int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0, ix1 - ix0)
    ih = max(0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    union = area_a + area_b - inter
    return _safe_rate(inter, union, empty_value=1.0)


def _component_descriptors(labels: np.ndarray, count: int) -> dict[int, dict[str, Any]]:
    descriptors: dict[int, dict[str, Any]] = {}
    if count == 0:
        return descriptors
    for label_id in range(1, count + 1):
        ys, xs = np.nonzero(labels == label_id)
        area = int(ys.size)
        if area == 0:
            continue
        component = labels == label_id
        descriptors[label_id] = {
            "label_id": int(label_id),
            "component_id": int(label_id - 1),
            "area_px": int(area),
            "box_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
            "centroid_xy": [float(xs.mean()), float(ys.mean())],
            "perimeter_px": _binary_perimeter_pixels(component),
        }
    return descriptors


def _class_touch_mask_4n(frame: np.ndarray, cls: int) -> np.ndarray:
    mask = frame == cls
    touch = mask.copy()
    touch[:, 1:] |= mask[:, :-1]
    touch[:, :-1] |= mask[:, 1:]
    touch[1:, :] |= mask[:-1, :]
    touch[:-1, :] |= mask[1:, :]
    return touch


def _boundary_family_mask(
    frame: np.ndarray,
    boundary: np.ndarray,
    *,
    family: str,
) -> np.ndarray:
    if family == "all":
        return boundary
    class_by_family = {
        "road": 0,
        "lane": 1,
        "vehicle": 2,
    }
    if family not in class_by_family:
        raise ValueError(f"unknown boundary primitive family: {family!r}")
    return boundary & _class_touch_mask_4n(frame, class_by_family[family])


def _directed_boundary_distance_accum(
    source: np.ndarray,
    target: np.ndarray,
    *,
    ndi: Any,
    sample_cap: int,
    max_missing_distance: float,
) -> dict[str, Any]:
    source_count = int(np.count_nonzero(source))
    if source_count == 0:
        return {
            "count": 0,
            "sum": 0.0,
            "coverage_1px": 0,
            "coverage_2px": 0,
            "samples": np.asarray([], dtype=np.float64),
            "exact": True,
        }
    target_count = int(np.count_nonzero(target))
    sample_count = min(source_count, max(0, int(sample_cap)))
    if target_count == 0:
        samples = np.full(sample_count, max_missing_distance, dtype=np.float64)
        return {
            "count": source_count,
            "sum": float(max_missing_distance * source_count),
            "coverage_1px": 0,
            "coverage_2px": 0,
            "samples": samples,
            "exact": True,
        }
    if ndi is not None:
        distances = ndi.distance_transform_edt(~target)[source]
        if sample_count and distances.size > sample_count:
            sample_idx = np.linspace(0, distances.size - 1, sample_count, dtype=np.int64)
            samples = distances[sample_idx].astype(np.float64, copy=False)
        else:
            samples = distances.astype(np.float64, copy=False)
        return {
            "count": source_count,
            "sum": float(distances.sum()),
            "coverage_1px": int(np.count_nonzero(distances <= 1.0)),
            "coverage_2px": int(np.count_nonzero(distances <= 2.0)),
            "samples": samples,
            "exact": True,
        }

    source_points = np.argwhere(source)
    target_points = np.argwhere(target)
    if sample_count and source_points.shape[0] > sample_count:
        source_points = source_points[np.linspace(0, source_points.shape[0] - 1, sample_count, dtype=np.int64)]
    if target_points.shape[0] > max(1, sample_count):
        target_points = target_points[np.linspace(0, target_points.shape[0] - 1, max(1, sample_count), dtype=np.int64)]
    if source_points.size == 0 or target_points.size == 0:
        samples = np.full(sample_count, max_missing_distance, dtype=np.float64)
    else:
        distances = []
        for point in source_points.astype(np.float64):
            delta = target_points.astype(np.float64) - point[None, :]
            distances.append(float(np.sqrt(np.sum(delta * delta, axis=1)).min()))
        samples = np.asarray(distances, dtype=np.float64)
    return {
        "count": source_count,
        "sum": float(samples.mean() * source_count) if samples.size else 0.0,
        "coverage_1px": int(round(_safe_rate(int(np.count_nonzero(samples <= 1.0)), samples.size) * source_count)),
        "coverage_2px": int(round(_safe_rate(int(np.count_nonzero(samples <= 2.0)), samples.size) * source_count)),
        "samples": samples,
        "exact": False,
    }


def _visual_boundary_primitives(
    baseline_np: np.ndarray,
    candidate_np: np.ndarray,
    *,
    options: VisualPrimitiveOptions,
) -> dict[str, Any]:
    ndi = _load_scipy_ndimage()
    frames, height, width = baseline_np.shape
    max_missing_distance = float(math.hypot(height, width))
    families = ("all", "road", "lane", "vehicle")
    work: dict[str, dict[str, Any]] = {
        family: {
            "baseline_boundary_pixels": 0,
            "candidate_boundary_pixels": 0,
            "base_to_candidate_count": 0,
            "candidate_to_base_count": 0,
            "base_to_candidate_sum": 0.0,
            "candidate_to_base_sum": 0.0,
            "base_coverage_1px": 0,
            "base_coverage_2px": 0,
            "candidate_coverage_1px": 0,
            "candidate_coverage_2px": 0,
            "sample_keys": None,
            "sample_values": None,
            "distance_sample_ordinal": 0,
            "distance_sample_population_seen": 0,
            "exact_distances": True,
        }
        for family in families
    }
    family_sample_seed = {family: 101 + idx * 65_537 for idx, family in enumerate(families)}

    for frame_idx in range(frames):
        base_frame = baseline_np[frame_idx]
        cand_frame = candidate_np[frame_idx]
        base_boundary = _boundary_mask_np(base_frame)
        cand_boundary = _boundary_mask_np(cand_frame)
        for family in families:
            base_family = _boundary_family_mask(base_frame, base_boundary, family=family)
            cand_family = _boundary_family_mask(cand_frame, cand_boundary, family=family)
            row = work[family]
            row["baseline_boundary_pixels"] += int(np.count_nonzero(base_family))
            row["candidate_boundary_pixels"] += int(np.count_nonzero(cand_family))
            b2c = _directed_boundary_distance_accum(
                base_family,
                cand_family,
                ndi=ndi,
                sample_cap=options.boundary_distance_sample_cap,
                max_missing_distance=max_missing_distance,
            )
            c2b = _directed_boundary_distance_accum(
                cand_family,
                base_family,
                ndi=ndi,
                sample_cap=options.boundary_distance_sample_cap,
                max_missing_distance=max_missing_distance,
            )
            row["base_to_candidate_count"] += int(b2c["count"])
            row["candidate_to_base_count"] += int(c2b["count"])
            row["base_to_candidate_sum"] += float(b2c["sum"])
            row["candidate_to_base_sum"] += float(c2b["sum"])
            row["base_coverage_1px"] += int(b2c["coverage_1px"])
            row["base_coverage_2px"] += int(b2c["coverage_2px"])
            row["candidate_coverage_1px"] += int(c2b["coverage_1px"])
            row["candidate_coverage_2px"] += int(c2b["coverage_2px"])
            _update_distance_sample_reservoir(
                row,
                b2c["samples"],
                cap=options.boundary_distance_global_sample_cap,
                seed=family_sample_seed[family],
            )
            _update_distance_sample_reservoir(
                row,
                c2b["samples"],
                cap=options.boundary_distance_global_sample_cap,
                seed=family_sample_seed[family],
            )
            row["exact_distances"] = bool(row["exact_distances"] and b2c["exact"] and c2b["exact"])

    out: dict[str, Any] = {}
    for family, row in work.items():
        combined_samples = (
            row["sample_values"].astype(np.float64, copy=False)
            if row["sample_values"] is not None
            else np.asarray([], dtype=np.float64)
        )
        b_count = int(row["base_to_candidate_count"])
        c_count = int(row["candidate_to_base_count"])
        b_mean = _safe_rate(float(row["base_to_candidate_sum"]), b_count)
        c_mean = _safe_rate(float(row["candidate_to_base_sum"]), c_count)
        out[family] = {
            "polyline_ordered": False,
            "primitive_family": family,
            "baseline_boundary_pixels": int(row["baseline_boundary_pixels"]),
            "candidate_boundary_pixels": int(row["candidate_boundary_pixels"]),
            "directed_chamfer_baseline_to_candidate_px": b_mean,
            "directed_chamfer_candidate_to_baseline_px": c_mean,
            "bidirectional_chamfer_px": 0.5 * (b_mean + c_mean),
            "hausdorff_p95_px": _percentile(combined_samples, 95, default=None),
            "baseline_coverage_at_1px": _safe_rate(int(row["base_coverage_1px"]), b_count, empty_value=1.0),
            "baseline_coverage_at_2px": _safe_rate(int(row["base_coverage_2px"]), b_count, empty_value=1.0),
            "candidate_coverage_at_1px": _safe_rate(int(row["candidate_coverage_1px"]), c_count, empty_value=1.0),
            "candidate_coverage_at_2px": _safe_rate(int(row["candidate_coverage_2px"]), c_count, empty_value=1.0),
            "distance_sample_count": int(combined_samples.size),
            "distance_sample_population_seen": int(row["distance_sample_population_seen"]),
            "distance_sample_per_frame_cap": int(options.boundary_distance_sample_cap),
            "distance_sample_global_cap": int(options.boundary_distance_global_sample_cap),
            "distance_sample_method": "deterministic_hash_reservoir_from_frame_samples",
            "distance_method": "scipy_distance_transform_edt" if row["exact_distances"] else "sampled_bruteforce_without_scipy",
        }
    return {
        "summary": "unordered_boundary_point_sets",
        "families": out,
    }


def _match_component_edges(
    labels_a: np.ndarray,
    desc_a: dict[int, dict[str, Any]],
    labels_b: np.ndarray,
    desc_b: dict[int, dict[str, Any]],
    *,
    options: VisualPrimitiveOptions,
) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    if not desc_a or not desc_b:
        return edges
    for label_a, row_a in desc_a.items():
        if int(row_a["area_px"]) < options.critical_component_min_area:
            continue
        component = labels_a == label_a
        overlaps = np.bincount(labels_b[component], minlength=max(desc_b) + 1)
        best_label = 0
        best_iou = 0.0
        if overlaps.shape[0] > 1 and int(overlaps[1:].max(initial=0)) > 0:
            best_label = int(np.argmax(overlaps[1:]) + 1)
            inter = int(overlaps[best_label])
            area_b = int(desc_b[best_label]["area_px"])
            best_iou = _safe_rate(inter, int(row_a["area_px"]) + area_b - inter)
        method = None
        if best_label and best_iou >= options.track_min_iou:
            method = "iou"
        else:
            ax, ay = row_a["centroid_xy"]
            nearest_label = 0
            nearest_dist = float("inf")
            for label_b, row_b in desc_b.items():
                bx, by = row_b["centroid_xy"]
                dist = float(math.hypot(bx - ax, by - ay))
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_label = label_b
            if nearest_label and nearest_dist <= options.track_max_centroid_jump_px:
                best_label = nearest_label
                method = "centroid"
        if not best_label or method is None:
            continue
        bx, by = desc_b[best_label]["centroid_xy"]
        ax, ay = row_a["centroid_xy"]
        edges.append(
            {
                "from_label": int(label_a),
                "to_label": int(best_label),
                "method": method,
                "iou": float(best_iou),
                "centroid_velocity_px": float(math.hypot(bx - ax, by - ay)),
            }
        )
    return edges


def _visual_temporal_primitives(
    baseline_np: np.ndarray,
    candidate_np: np.ndarray,
    *,
    num_classes: int,
    existing_temporal: dict[str, Any],
    options: VisualPrimitiveOptions,
) -> dict[str, Any]:
    ndi = _load_scipy_ndimage()
    frames = baseline_np.shape[0]
    track_classes = (
        tuple(range(num_classes))
        if not options.track_classes
        else tuple(int(cls) for cls in options.track_classes)
    )
    track_class_set = set(track_classes)
    per_class: dict[str, Any] = {
        str(cls): {
            "name": CLASS_NAMES.get(cls, f"class_{cls}"),
            "track_evaluated": bool(options.temporal_tracks_enabled and cls in track_class_set),
            "baseline_track_edges": 0,
            "candidate_track_edges": 0,
            "candidate_edge_count_delta": 0,
            "track_survival_recall_proxy": 1.0,
            "track_fragmentation_rate_proxy": 0.0,
            "baseline_velocity_p95_px": None,
            "candidate_velocity_p95_px": None,
        }
        for cls in range(num_classes)
    }
    if not options.temporal_tracks_enabled:
        return {
            "track_proxy_method": "disabled_by_visual_primitive_options",
            "track_proxy_limitations": (
                "temporal component-track proxies skipped for bounded CPU triage; "
                "full scalar temporal mask metrics remain in existing_temporal_mask_metrics"
            ),
            "track_classes": [],
            "critical_classes": [int(cls) for cls in options.critical_classes],
            "critical_track_survival_recall_proxy": None,
            "critical_track_fragmentation_rate_proxy": None,
            "per_class": per_class,
            "existing_temporal_mask_metrics": existing_temporal,
            "promotion_eligible": False,
            "score_claim_eligible": False,
        }
    if frames < 2:
        return {
            "track_proxy_method": "same_class_iou_then_centroid_cpu",
            "track_classes": [int(cls) for cls in track_classes],
            "per_class": per_class,
            "existing_temporal_mask_metrics": existing_temporal,
        }

    velocity_work: dict[str, dict[str, list[float]]] = {
        str(cls): {"baseline": [], "candidate": []} for cls in range(num_classes)
    }
    for frame_idx in range(frames - 1):
        for cls in track_classes:
            key = str(cls)
            base_a, base_a_count = _label_components(baseline_np[frame_idx] == cls, ndi)
            base_b, base_b_count = _label_components(baseline_np[frame_idx + 1] == cls, ndi)
            cand_a, cand_a_count = _label_components(candidate_np[frame_idx] == cls, ndi)
            cand_b, cand_b_count = _label_components(candidate_np[frame_idx + 1] == cls, ndi)
            base_edges = _match_component_edges(
                base_a,
                _component_descriptors(base_a, base_a_count),
                base_b,
                _component_descriptors(base_b, base_b_count),
                options=options,
            )
            cand_edges = _match_component_edges(
                cand_a,
                _component_descriptors(cand_a, cand_a_count),
                cand_b,
                _component_descriptors(cand_b, cand_b_count),
                options=options,
            )
            per_class[key]["baseline_track_edges"] += len(base_edges)
            per_class[key]["candidate_track_edges"] += len(cand_edges)
            velocity_work[key]["baseline"].extend(float(edge["centroid_velocity_px"]) for edge in base_edges)
            velocity_work[key]["candidate"].extend(float(edge["centroid_velocity_px"]) for edge in cand_edges)

    for cls in range(num_classes):
        key = str(cls)
        baseline_edges = int(per_class[key]["baseline_track_edges"])
        candidate_edges = int(per_class[key]["candidate_track_edges"])
        per_class[key]["candidate_edge_count_delta"] = int(candidate_edges - baseline_edges)
        per_class[key]["track_survival_recall_proxy"] = _safe_rate(
            min(candidate_edges, baseline_edges),
            baseline_edges,
            empty_value=1.0,
        )
        per_class[key]["track_fragmentation_rate_proxy"] = _safe_rate(
            max(0, baseline_edges - candidate_edges),
            baseline_edges,
        )
        per_class[key]["baseline_velocity_p95_px"] = _percentile(
            velocity_work[key]["baseline"],
            95,
            default=None,
        )
        per_class[key]["candidate_velocity_p95_px"] = _percentile(
            velocity_work[key]["candidate"],
            95,
            default=None,
        )

    critical = [per_class[str(cls)] for cls in options.critical_classes if str(cls) in per_class]
    baseline_critical_edges = sum(int(row["baseline_track_edges"]) for row in critical)
    candidate_critical_edges = sum(int(row["candidate_track_edges"]) for row in critical)
    return {
        "track_proxy_method": "same_class_iou_then_centroid_cpu",
        "track_proxy_limitations": (
            "diagnostic component-edge counts only; exact score truth remains CUDA auth eval"
        ),
        "critical_classes": [int(cls) for cls in options.critical_classes],
        "track_classes": [int(cls) for cls in track_classes],
        "critical_track_survival_recall_proxy": _safe_rate(
            min(candidate_critical_edges, baseline_critical_edges),
            baseline_critical_edges,
            empty_value=1.0,
        ),
        "critical_track_fragmentation_rate_proxy": _safe_rate(
            max(0, baseline_critical_edges - candidate_critical_edges),
            baseline_critical_edges,
        ),
        "per_class": per_class,
        "existing_temporal_mask_metrics": existing_temporal,
    }


def _visual_component_primitives(
    baseline_np: np.ndarray,
    candidate_np: np.ndarray,
    *,
    num_classes: int,
    options: VisualPrimitiveOptions,
) -> dict[str, Any]:
    ndi = _load_scipy_ndimage()
    frames, height, width = baseline_np.shape
    pose_y = options.pose_sensitive_y_fraction_min * float(height)
    component_classes = (
        tuple(range(num_classes))
        if not options.component_classes
        else tuple(int(cls) for cls in options.component_classes)
    )
    component_class_set = set(component_classes)
    per_class: dict[str, dict[str, Any]] = {
        str(cls): {
            "name": CLASS_NAMES.get(cls, f"class_{cls}"),
            "component_evaluated": cls in component_class_set,
            "baseline_components": 0,
            "candidate_components": 0,
            "evaluated_baseline_components": 0,
            "evaluated_baseline_area_px": 0,
            "matched_components": 0,
            "missing_components": 0,
            "missing_area_px": 0,
            "split_components": 0,
            "merge_extra_baseline_components": 0,
            "pose_sensitive_missing_components": 0,
            "_box_ious": [],
            "_mask_ious": [],
            "_center_shifts": [],
            "_normalized_center_shifts": [],
            "_area_ratio_errors": [],
        }
        for cls in range(num_classes)
    }
    critical_failures: list[dict[str, Any]] = []
    critical_class_set = set(int(cls) for cls in options.critical_classes)

    for frame_idx in range(frames):
        base_frame = baseline_np[frame_idx]
        cand_frame = candidate_np[frame_idx]
        for cls in component_classes:
            key = str(cls)
            base_labels, base_count = _label_components(base_frame == cls, ndi)
            cand_labels, cand_count = _label_components(cand_frame == cls, ndi)
            base_desc = _component_descriptors(base_labels, base_count)
            cand_desc = _component_descriptors(cand_labels, cand_count)
            per_class[key]["baseline_components"] += len(base_desc)
            per_class[key]["candidate_components"] += len(cand_desc)
            candidate_to_baseline: dict[int, list[int]] = {}

            for base_label, base_row in base_desc.items():
                area = int(base_row["area_px"])
                if area < options.critical_component_min_area:
                    continue
                per_class[key]["evaluated_baseline_components"] += 1
                per_class[key]["evaluated_baseline_area_px"] += area
                component = base_labels == base_label
                overlaps = np.bincount(cand_labels[component], minlength=cand_count + 1)
                overlapping_candidate_labels = [
                    int(label_id)
                    for label_id in range(1, cand_count + 1)
                    if label_id < overlaps.shape[0] and int(overlaps[label_id]) > 0
                ]
                lower_or_pose_sensitive = bool(base_row["box_xyxy"][3] >= pose_y)
                if not overlapping_candidate_labels:
                    per_class[key]["missing_components"] += 1
                    per_class[key]["missing_area_px"] += area
                    if lower_or_pose_sensitive:
                        per_class[key]["pose_sensitive_missing_components"] += 1
                    critical_failures.append(
                        {
                            "failure_type": "missing_component",
                            "frame": int(frame_idx),
                            "class_id": int(cls),
                            "class_name": CLASS_NAMES.get(cls, f"class_{cls}"),
                            "baseline_component_id": int(base_row["component_id"]),
                            "candidate_component_id": None,
                            "area_px": area,
                            "box_xyxy": base_row["box_xyxy"],
                            "centroid_xy": base_row["centroid_xy"],
                            "perimeter_px": int(base_row["perimeter_px"]),
                            "box_iou": 0.0,
                            "mask_iou": 0.0,
                            "centroid_jump_px": None,
                            "area_ratio_error": None,
                            "split_candidate_components": 0,
                            "pose_sensitive": lower_or_pose_sensitive,
                        }
                    )
                    continue

                best_label = max(overlapping_candidate_labels, key=lambda label_id: int(overlaps[label_id]))
                candidate_to_baseline.setdefault(best_label, []).append(base_label)
                cand_row = cand_desc[best_label]
                intersection = int(overlaps[best_label])
                candidate_area = int(cand_row["area_px"])
                union = area + candidate_area - intersection
                mask_iou = _safe_rate(intersection, union)
                box_iou = _box_iou_xyxy(base_row["box_xyxy"], cand_row["box_xyxy"])
                bx, by = base_row["centroid_xy"]
                cx, cy = cand_row["centroid_xy"]
                center_shift = float(math.hypot(cx - bx, cy - by))
                normalized_shift = center_shift / max(1.0, math.sqrt(float(area)))
                area_ratio_error = abs(float(candidate_area) / float(area) - 1.0)
                split_count = len(overlapping_candidate_labels)
                per_class[key]["matched_components"] += 1
                per_class[key]["_box_ious"].append(box_iou)
                per_class[key]["_mask_ious"].append(mask_iou)
                per_class[key]["_center_shifts"].append(center_shift)
                per_class[key]["_normalized_center_shifts"].append(normalized_shift)
                per_class[key]["_area_ratio_errors"].append(area_ratio_error)
                if split_count > 1:
                    per_class[key]["split_components"] += 1
                if (
                    cls in critical_class_set
                    and (box_iou < 0.90 or center_shift > 1.0 or split_count > 1 or area_ratio_error > 0.10)
                ):
                    critical_failures.append(
                        {
                            "failure_type": "degraded_match",
                            "frame": int(frame_idx),
                            "class_id": int(cls),
                            "class_name": CLASS_NAMES.get(cls, f"class_{cls}"),
                            "baseline_component_id": int(base_row["component_id"]),
                            "candidate_component_id": int(cand_row["component_id"]),
                            "area_px": area,
                            "box_xyxy": base_row["box_xyxy"],
                            "candidate_box_xyxy": cand_row["box_xyxy"],
                            "centroid_xy": base_row["centroid_xy"],
                            "candidate_centroid_xy": cand_row["centroid_xy"],
                            "perimeter_px": int(base_row["perimeter_px"]),
                            "box_iou": box_iou,
                            "mask_iou": mask_iou,
                            "centroid_jump_px": center_shift,
                            "centroid_jump_norm": normalized_shift,
                            "area_ratio_error": area_ratio_error,
                            "split_candidate_components": int(split_count),
                            "pose_sensitive": lower_or_pose_sensitive,
                        }
                    )

            for matched_base_labels in candidate_to_baseline.values():
                if len(matched_base_labels) > 1:
                    per_class[key]["merge_extra_baseline_components"] += len(matched_base_labels) - 1

    for cls in range(num_classes):
        key = str(cls)
        row = per_class[key]
        evaluated = int(row["evaluated_baseline_components"])
        matched = int(row["matched_components"])
        baseline_area = int(row["evaluated_baseline_area_px"])
        row["missing_rate"] = _safe_rate(int(row["missing_components"]), evaluated)
        row["missing_area_rate"] = _safe_rate(int(row["missing_area_px"]), baseline_area)
        row["box_iou_p05"] = _percentile(row["_box_ious"], 5, default=None)
        row["box_iou_p50"] = _percentile(row["_box_ious"], 50, default=None)
        row["mask_iou_p05"] = _percentile(row["_mask_ious"], 5, default=None)
        row["centroid_jump_p50_px"] = _percentile(row["_center_shifts"], 50, default=None)
        row["centroid_jump_p95_px"] = _percentile(row["_center_shifts"], 95, default=None)
        row["centroid_jump_max_px"] = max(row["_center_shifts"]) if row["_center_shifts"] else None
        row["centroid_jump_norm_p95"] = _percentile(row["_normalized_center_shifts"], 95, default=None)
        row["centroid_over_1px_rate"] = _safe_rate(
            sum(1 for value in row["_center_shifts"] if value > 1.0),
            matched,
        )
        row["centroid_over_2px_rate"] = _safe_rate(
            sum(1 for value in row["_center_shifts"] if value > 2.0),
            matched,
        )
        row["box_area_ratio_error_p95"] = _percentile(row["_area_ratio_errors"], 95, default=None)
        row["split_rate"] = _safe_rate(int(row["split_components"]), evaluated)
        row["merge_rate"] = _safe_rate(int(row["merge_extra_baseline_components"]), matched)
        for private_key in (
            "_box_ious",
            "_mask_ious",
            "_center_shifts",
            "_normalized_center_shifts",
            "_area_ratio_errors",
        ):
            del row[private_key]

    priority_by_class = {1: 0, 2: 1, 0: 2}
    for row in critical_failures:
        class_id = int(row["class_id"])
        missing_bucket = 0 if row["failure_type"] == "missing_component" else 1
        centroid_jump = row["centroid_jump_px"]
        row["_sort_key"] = (
            priority_by_class.get(class_id, 3),
            missing_bucket,
            -int(row["area_px"]),
            -(float(centroid_jump) if centroid_jump is not None else float("inf")),
            float(row["box_iou"]),
            int(row["frame"]),
            int(row["box_xyxy"][1]),
            int(row["box_xyxy"][0]),
        )
    critical_failures.sort(key=lambda row: row["_sort_key"])
    kept_failures = critical_failures[: max(0, options.max_component_failures)]
    for rank, row in enumerate(kept_failures, start=1):
        row["rank"] = int(rank)
        del row["_sort_key"]

    critical_rows = [per_class[str(cls)] for cls in options.critical_classes if str(cls) in per_class]
    critical_evaluated = sum(int(row["evaluated_baseline_components"]) for row in critical_rows)
    critical_missing = sum(int(row["missing_components"]) for row in critical_rows)
    critical_area = sum(int(row["evaluated_baseline_area_px"]) for row in critical_rows)
    critical_missing_area = sum(int(row["missing_area_px"]) for row in critical_rows)
    return {
        "connectivity": int(options.component_connectivity),
        "critical_component_min_area": int(options.critical_component_min_area),
        "component_classes": [int(cls) for cls in component_classes],
        "per_class": per_class,
        "critical_summary": {
            "critical_classes": [int(cls) for cls in options.critical_classes],
            "evaluated_baseline_components": int(critical_evaluated),
            "missing_components": int(critical_missing),
            "missing_rate": _safe_rate(critical_missing, critical_evaluated),
            "evaluated_baseline_area_px": int(critical_area),
            "missing_area_px": int(critical_missing_area),
            "missing_area_rate": _safe_rate(critical_missing_area, critical_area),
            "critical_failures_returned": int(len(kept_failures)),
        },
        "critical_box_failures": kept_failures,
        "shape": {"frames": int(frames), "height": int(height), "width": int(width)},
    }


def _visual_gate_status(
    *,
    global_disagreement: float,
    boundary_2px_disagreement: float | None,
    pair_transition_disagreement: float,
    critical_missing_rate: float,
    critical_missing_area_rate: float,
) -> dict[str, Any]:
    gates = {
        "exploratory_retrain_gate": {
            "global_disagreement_max": 0.003,
            "boundary_2px_disagreement_max": 0.005,
            "pair_transition_disagreement_max": 0.004,
            "critical_missing_rate_max": 0.02,
            "critical_missing_area_rate_max": 0.01,
        },
        "exact_eval_spend_gate": {
            "global_disagreement_max": 0.001,
            "boundary_2px_disagreement_max": 0.002,
            "pair_transition_disagreement_max": 0.002,
            "critical_missing_rate_max": 0.001,
            "critical_missing_area_rate_max": 0.0005,
        },
    }
    out: dict[str, Any] = {}
    for gate_name, thresholds in gates.items():
        blockers: list[str] = []
        if global_disagreement > thresholds["global_disagreement_max"]:
            blockers.append("global_disagreement")
        if boundary_2px_disagreement is None:
            blockers.append("boundary_2px_disagreement_missing")
        elif boundary_2px_disagreement > thresholds["boundary_2px_disagreement_max"]:
            blockers.append("boundary_2px_disagreement")
        if pair_transition_disagreement > thresholds["pair_transition_disagreement_max"]:
            blockers.append("pair_transition_disagreement")
        if critical_missing_rate > thresholds["critical_missing_rate_max"]:
            blockers.append("critical_missing_rate")
        if critical_missing_area_rate > thresholds["critical_missing_area_rate_max"]:
            blockers.append("critical_missing_area_rate")
        out[gate_name] = {
            "passed": len(blockers) == 0,
            "blockers": blockers,
            "thresholds": thresholds,
            "observed": {
                "global_disagreement": global_disagreement,
                "boundary_2px_disagreement": boundary_2px_disagreement,
                "pair_transition_disagreement": pair_transition_disagreement,
                "critical_missing_rate": critical_missing_rate,
                "critical_missing_area_rate": critical_missing_area_rate,
            },
            "promotion_eligible": False,
            "score_claim_eligible": False,
        }
    return out


def _gate_gap(value: float | None, threshold: float | None) -> float | None:
    if value is None or threshold is None or threshold <= 0:
        return None
    return float(value) / float(threshold)


def _alpha_geo_repair_retrain_spec(
    *,
    existing_metrics: dict[str, Any],
    component_primitives: dict[str, Any],
    boundary_primitives: dict[str, Any],
    temporal_primitives: dict[str, Any],
    gate_status: dict[str, Any],
) -> dict[str, Any]:
    """Translate Alpha-Geo visual blockers into deterministic next specs.

    This is intentionally an empirical planning packet. It does not authorize
    retraining, L2 clearance, or exact-eval spend.
    """

    exploratory_gate = gate_status["exploratory_retrain_gate"]
    exact_gate = gate_status["exact_eval_spend_gate"]
    observed = exact_gate["observed"]
    exact_thresholds = exact_gate["thresholds"]
    exploratory_thresholds = exploratory_gate["thresholds"]
    blockers = [str(item) for item in exact_gate["blockers"]]
    blocker_set = set(blockers)
    critical_summary = component_primitives["critical_summary"]
    per_class = component_primitives["per_class"]
    boundary_families = boundary_primitives["families"]
    scalar_temporal = temporal_primitives.get(
        "existing_temporal_mask_metrics",
        existing_metrics["temporal"],
    )

    metric_gaps = {
        name: {
            "observed": observed.get(name),
            "exploratory_threshold": exploratory_thresholds.get(f"{name}_max"),
            "exact_eval_spend_threshold": exact_thresholds.get(f"{name}_max"),
            "exploratory_gap": _gate_gap(
                observed.get(name),
                exploratory_thresholds.get(f"{name}_max"),
            ),
            "exact_eval_spend_gap": _gate_gap(
                observed.get(name),
                exact_thresholds.get(f"{name}_max"),
            ),
        }
        for name in (
            "global_disagreement",
            "boundary_2px_disagreement",
            "pair_transition_disagreement",
            "critical_missing_rate",
            "critical_missing_area_rate",
        )
    }

    class_focus = []
    for cls in critical_summary["critical_classes"]:
        row = per_class.get(str(cls), {})
        class_focus.append(
            {
                "class_id": int(cls),
                "class_name": row.get("name", CLASS_NAMES.get(int(cls), f"class_{cls}")),
                "missing_rate": row.get("missing_rate"),
                "missing_area_rate": row.get("missing_area_rate"),
                "evaluated_baseline_components": row.get("evaluated_baseline_components"),
                "missing_components": row.get("missing_components"),
                "pose_sensitive_missing_components": row.get("pose_sensitive_missing_components"),
                "box_iou_p05": row.get("box_iou_p05"),
                "centroid_jump_p95_px": row.get("centroid_jump_p95_px"),
            }
        )

    blocker_to_spec_ids: dict[str, list[str]] = {name: [] for name in blockers}
    training_specs: list[dict[str, Any]] = []

    def _add_training_spec(spec: dict[str, Any], blocker_names: tuple[str, ...]) -> None:
        training_specs.append(spec)
        for blocker_name in blocker_names:
            if blocker_name in blocker_to_spec_ids:
                blocker_to_spec_ids[blocker_name].append(str(spec["spec_id"]))

    if "critical_missing_rate" in blocker_set or "critical_missing_area_rate" in blocker_set:
        _add_training_spec(
            {
                "spec_id": "critical_component_recall_retrain",
                "objective": "preserve decoded-baseline lane/vehicle components before rate optimization",
                "target_source": "decoded_baseline_masks_mkv",
                "must_use_existing_training_mode": "experiments/train_nerv_mask.py --gt-masks-source decoded-baseline",
                "sample_weighting": [
                    "oversample frames listed in visual_primitives.primitive_metrics.connected_components.critical_box_failures",
                    "weight class 1 and class 2 component interiors above all-background/global CE",
                    "weight pose-sensitive components whose box y1 >= 0.60 * height",
                ],
                "success_criteria": {
                    "exploratory": {
                        "critical_missing_rate_max": exploratory_thresholds["critical_missing_rate_max"],
                        "critical_missing_area_rate_max": exploratory_thresholds[
                            "critical_missing_area_rate_max"
                        ],
                    },
                    "exact_eval_spend_review": {
                        "critical_missing_rate_max": exact_thresholds["critical_missing_rate_max"],
                        "critical_missing_area_rate_max": exact_thresholds[
                            "critical_missing_area_rate_max"
                        ],
                    },
                },
                "observed_focus": {
                    "critical_summary": critical_summary,
                    "critical_classes": class_focus,
                    "critical_box_failure_count_returned": len(
                        component_primitives.get("critical_box_failures", [])
                    ),
                },
                "promotion_eligible": False,
                "score_claim_eligible": False,
            },
            ("critical_missing_rate", "critical_missing_area_rate"),
        )

    if "boundary_2px_disagreement" in blocker_set:
        _add_training_spec(
            {
                "spec_id": "boundary_band_retrain",
                "objective": "match decoded-baseline boundary geometry within the 2px Alpha-Geo band",
                "target_source": "decoded_baseline_masks_mkv",
                "boundary_radii_px": [1, 2, 3, 5],
                "sample_weighting": [
                    "weight baseline 1px and 2px boundary bands for all classes",
                    "apply extra weight to lane and road boundary families because lane baseline coverage is lowest",
                    "keep class targets integer; no scorer or PoseNet proxy is part of this diagnostic spec",
                ],
                "observed_focus": {
                    "scalar_boundary_2px_disagreement": observed["boundary_2px_disagreement"],
                    "boundary_families": {
                        family: {
                            "baseline_coverage_at_2px": row.get("baseline_coverage_at_2px"),
                            "candidate_coverage_at_2px": row.get("candidate_coverage_at_2px"),
                            "bidirectional_chamfer_px": row.get("bidirectional_chamfer_px"),
                            "hausdorff_p95_px": row.get("hausdorff_p95_px"),
                        }
                        for family, row in boundary_families.items()
                    },
                },
                "success_criteria": {
                    "exploratory_boundary_2px_disagreement_max": exploratory_thresholds[
                        "boundary_2px_disagreement_max"
                    ],
                    "exact_eval_spend_boundary_2px_disagreement_max": exact_thresholds[
                        "boundary_2px_disagreement_max"
                    ],
                },
                "promotion_eligible": False,
                "score_claim_eligible": False,
            },
            ("boundary_2px_disagreement",),
        )

    if "pair_transition_disagreement" in blocker_set:
        _add_training_spec(
            {
                "spec_id": "temporal_transition_retrain",
                "objective": "preserve decoded-baseline adjacent-frame transition masks",
                "target_source": "decoded_baseline_masks_mkv",
                "sample_weighting": [
                    "train adjacent frame pairs, not only independent coordinates",
                    "oversample worst_frame_pairs in descending transition_disagreement_rate order",
                    "penalize false-negative baseline transitions separately from false-positive candidate transitions",
                ],
                "observed_focus": {
                    "pair_transition": scalar_temporal["pair_transition"],
                    "stable_region": scalar_temporal["stable_region"],
                    "worst_pair_indices": [
                        int(row["pair_index"]) for row in scalar_temporal.get("worst_frame_pairs", [])[:10]
                    ],
                },
                "success_criteria": {
                    "exploratory_pair_transition_disagreement_max": exploratory_thresholds[
                        "pair_transition_disagreement_max"
                    ],
                    "exact_eval_spend_pair_transition_disagreement_max": exact_thresholds[
                        "pair_transition_disagreement_max"
                    ],
                },
                "promotion_eligible": False,
                "score_claim_eligible": False,
            },
            ("pair_transition_disagreement",),
        )

    if "global_disagreement" in blocker_set:
        _add_training_spec(
            {
                "spec_id": "decoded_baseline_global_overfit",
                "objective": "reduce all-pixel Hamming disagreement against decoded baseline before any archive score spend",
                "target_source": "decoded_baseline_masks_mkv",
                "sample_weighting": [
                    "keep uniform decoded-baseline CE as the base objective",
                    "do not train against fresh SegNet argmax labels for Alpha-Geo repair",
                    "do not lower component or boundary weights to chase byte count until geometry gates pass",
                ],
                "observed_focus": {
                    "global": existing_metrics["global"],
                    "class_focus": class_focus,
                },
                "success_criteria": {
                    "exploratory_global_disagreement_max": exploratory_thresholds[
                        "global_disagreement_max"
                    ],
                    "exact_eval_spend_global_disagreement_max": exact_thresholds[
                        "global_disagreement_max"
                    ],
                },
                "promotion_eligible": False,
                "score_claim_eligible": False,
            },
            ("global_disagreement",),
        )

    repair_specs = [
        {
            "spec_id": "charged_sparse_critical_residual_after_retrain",
            "activation_rule": (
                "only after decoded-baseline retrain reduces global and boundary blockers but "
                "critical component blockers remain"
            ),
            "source_regions": [
                "visual_primitives.primitive_metrics.connected_components.critical_box_failures",
                "residual_region_ranking.regions after rerun with --residual-region-count > 0",
            ],
            "byte_accounting": "all residual side information must live inside archive.zip",
            "residual_atoms": [
                "lane_lower_field_residual",
                "boundary_band_residual",
                "temporal_pair_residual",
                "near_vehicle_box_residual",
            ],
            "promotion_eligible": False,
            "score_claim_eligible": False,
        }
    ]

    return {
        "schema_version": 1,
        "diagnostic": "alpha_geo_repair_retrain_spec_v1",
        "score_evidence_grade": "empirical",
        "device": "cpu",
        "scorer_proxy": False,
        "promotion_eligible": False,
        "score_claim_eligible": False,
        "exact_eval_claim": False,
        "l2_clearance_created": False,
        "l2_status": "blocked_until_external_clearance_packet_and_clean_reviews",
        "source_gate": "visual_primitives.pass_fail.exact_eval_spend_gate",
        "blockers": blockers,
        "blocker_to_spec_ids": blocker_to_spec_ids,
        "metric_gaps": metric_gaps,
        "training_specs": training_specs,
        "repair_specs": repair_specs,
        "currently_admissible_commands": [
            {
                "name": "materialize_residual_repair_regions_cpu_only",
                "requires_l2_clearance": False,
                "score_claim_eligible": False,
                "command": [
                    ".venv/bin/python",
                    "experiments/diagnose_nerv_geometry.py",
                    "--baseline",
                    "<baseline_archive.zip>",
                    "--baseline-member",
                    "masks.mkv",
                    "--candidate",
                    "<candidate_archive.zip>",
                    "--candidate-member",
                    "masks.nrv",
                    "--num-frames",
                    "1200",
                    "--height",
                    "384",
                    "--width",
                    "512",
                    "--threshold-preset",
                    "none",
                    "--residual-region-count",
                    "200",
                    "--visual-component-classes",
                    "1,2",
                    "--visual-disable-temporal-tracks",
                    "--output-json",
                    "<evidence_dir>/alpha_geo_residual_regions.json",
                ],
            }
        ],
        "blocked_commands": [
            {
                "name": "lane12_decoded_baseline_retrain_build_only",
                "blocked_by": "missing_valid_l2_clearance_packet",
                "requires_l2_clearance": True,
                "score_claim_eligible": False,
                "command_template": (
                    "RUN_AUTH_EVAL=0 GT_MASKS_SOURCE=decoded-baseline "
                    "DECODED_BASELINE_PATH=<baseline_archive.zip> "
                    "DECODED_BASELINE_MEMBER=masks.mkv "
                    "scripts/remote_lane_nerv.sh"
                ),
            },
            {
                "name": "exact_cuda_auth_eval",
                "blocked_by": (
                    "requires passing Alpha-Geo packet, pose-regeneration provenance, "
                    "archive custody, and external L2/Grand Council gates"
                ),
                "requires_l2_clearance": True,
                "score_claim_eligible": False,
                "command_template": (
                    ".venv/bin/python experiments/contest_auth_eval.py "
                    "--archive <candidate_archive.zip> "
                    "--inflate-sh submissions/robust_current/inflate.sh "
                    "--upstream-dir upstream --device cuda --keep-work-dir "
                    "--work-dir <evidence_dir>/auth_eval_work"
                ),
            },
        ],
    }


def _alpha_geo_visual_primitives_packet(
    baseline: torch.Tensor,
    candidate: torch.Tensor,
    *,
    num_classes: int,
    boundary_radii: tuple[int, ...],
    existing_metrics: dict[str, Any],
    options: VisualPrimitiveOptions,
) -> dict[str, Any]:
    if options.component_connectivity != 4:
        raise ValueError("alpha visual primitives currently support only 4-connected components")
    if options.frame_stride <= 0:
        raise ValueError("visual primitive frame_stride must be > 0")
    if options.polyline_ordered:
        raise ValueError("ordered boundary polyline extraction is not implemented in this CPU packet")
    if options.boundary_distance_sample_cap < 0:
        raise ValueError("boundary_distance_sample_cap must be >= 0")
    if options.boundary_distance_global_sample_cap < 0:
        raise ValueError("boundary_distance_global_sample_cap must be >= 0")
    if not 0.0 <= options.track_min_iou <= 1.0:
        raise ValueError("track_min_iou must be between 0 and 1")
    if options.track_max_centroid_jump_px < 0.0:
        raise ValueError("track_max_centroid_jump_px must be >= 0")
    for field_name, classes in (
        ("critical_classes", options.critical_classes),
        ("component_classes", options.component_classes),
        ("track_classes", options.track_classes),
    ):
        for cls in classes:
            if int(cls) < 0 or int(cls) >= num_classes:
                raise ValueError(f"{field_name} contains class {cls}, outside [0, {num_classes})")
    critical_set = set(int(cls) for cls in options.critical_classes)
    if options.component_classes and not critical_set.issubset(set(int(cls) for cls in options.component_classes)):
        raise ValueError("visual component_classes must include every critical class")
    if options.track_classes and not critical_set.issubset(set(int(cls) for cls in options.track_classes)):
        raise ValueError("visual track_classes must include every critical class")
    baseline_full_sha = _mask_tensor_sha256(baseline)
    candidate_full_sha = _mask_tensor_sha256(candidate)
    if options.frame_stride == 1:
        visual_baseline = baseline
        visual_candidate = candidate
    else:
        visual_baseline = baseline[:: options.frame_stride].contiguous()
        visual_candidate = candidate[:: options.frame_stride].contiguous()
    baseline_np = visual_baseline.numpy()
    candidate_np = visual_candidate.numpy()
    component_primitives = _visual_component_primitives(
        baseline_np,
        candidate_np,
        num_classes=num_classes,
        options=options,
    )
    boundary_primitives = _visual_boundary_primitives(
        baseline_np,
        candidate_np,
        options=options,
    )
    temporal_primitives = _visual_temporal_primitives(
        baseline_np,
        candidate_np,
        num_classes=num_classes,
        existing_temporal=existing_metrics["temporal"],
        options=options,
    )
    boundary_2px = existing_metrics["boundary_bands"].get("2", {}).get("disagreement_rate")
    gate_status = _visual_gate_status(
        global_disagreement=float(existing_metrics["global"]["global_disagreement"]),
        boundary_2px_disagreement=None if boundary_2px is None else float(boundary_2px),
        pair_transition_disagreement=float(existing_metrics["temporal"]["pair_transition"]["disagreement_rate"]),
        critical_missing_rate=float(component_primitives["critical_summary"]["missing_rate"]),
        critical_missing_area_rate=float(component_primitives["critical_summary"]["missing_area_rate"]),
    )
    exact_gate = gate_status["exact_eval_spend_gate"]
    repair_retrain_spec = _alpha_geo_repair_retrain_spec(
        existing_metrics=existing_metrics,
        component_primitives=component_primitives,
        boundary_primitives=boundary_primitives,
        temporal_primitives=temporal_primitives,
        gate_status=gate_status,
    )
    return {
        "schema_version": 1,
        "diagnostic": "alpha_geo_visual_primitives_v1",
        "score_evidence_grade": "empirical",
        "device": "cpu",
        "scorer_proxy": False,
        "scorer_network_loaded": False,
        "promotion_eligible": False,
        "score_claim_eligible": False,
        "exact_eval_claim": False,
        "exact_score_source": (
            "none in this packet; exact score requires archive.zip -> inflate.sh -> "
            "upstream/evaluate.py on CUDA"
        ),
        "source": {
            "baseline_mask_sha256": baseline_full_sha,
            "candidate_mask_sha256": candidate_full_sha,
            "visual_baseline_mask_sha256": _mask_tensor_sha256(visual_baseline),
            "visual_candidate_mask_sha256": _mask_tensor_sha256(visual_candidate),
            "visual_frame_stride": int(options.frame_stride),
            "visual_frame_count": int(visual_baseline.shape[0]),
            "diagnose_nerv_geometry_json": None,
        },
        "shape": {
            "frames": int(baseline.shape[0]),
            "height": int(baseline.shape[1]),
            "width": int(baseline.shape[2]),
            "num_classes": int(num_classes),
        },
        "visual_shape": {
            "frames": int(visual_baseline.shape[0]),
            "height": int(visual_baseline.shape[1]),
            "width": int(visual_baseline.shape[2]),
            "num_classes": int(num_classes),
        },
        "extraction_config": {
            **_as_jsonable(asdict(options)),
            "boundary_radii_px": [int(radius) for radius in boundary_radii],
            "component_ordering": "frame,class,decreasing_area,y0,x0,y1,x1",
            "box_coordinate_convention": "half_open_xyxy",
        },
        "existing_geometry_metrics": {
            "global": existing_metrics["global"],
            "boundary_bands": existing_metrics["boundary_bands"],
            "temporal": existing_metrics["temporal"],
            "components": {
                "speckle_rate": existing_metrics["components"]["speckle_rate"],
                "centroid": existing_metrics["components"]["centroid"],
            },
            "residual_region_ranking_summary": {
                "regions_considered": existing_metrics["residual_region_ranking"]["regions_considered"],
                "regions_returned": existing_metrics["residual_region_ranking"]["regions_returned"],
            },
        },
        "primitive_metrics": {
            "connected_components": component_primitives,
            "boundary_primitives": boundary_primitives,
            "temporal_primitives": temporal_primitives,
        },
        "pass_fail": gate_status,
        "repair_retrain_spec": repair_retrain_spec,
        "next_action": {
            "exact_eval_spend_recommended_for_review": bool(exact_gate["passed"]),
            "recommendation": (
                "candidate_plausible_for_exact_eval_budget_review"
                if exact_gate["passed"]
                else "repair_or_retrain_before_exact_eval_spend"
            ),
            "blockers": list(exact_gate["blockers"]),
            "review_order": [
                "critical_box_failures",
                "lane_boundary_coverage",
                "road_boundary_coverage",
                "worst_frame_pairs",
                "residual_region_ranking",
            ],
            "promotion_eligible": False,
            "score_claim_eligible": False,
        },
    }


def _record_check(
    checks: dict[str, Any],
    name: str,
    *,
    value: float | None,
    op: str,
    threshold: float | None,
    skipped: bool = False,
    reason: str | None = None,
) -> None:
    if skipped or threshold is None or value is None:
        checks[name] = {
            "value": value,
            "op": op,
            "threshold": threshold,
            "passed": None,
            "skipped": True,
            "reason": reason or "threshold not configured",
        }
        return
    if op == "<=":
        passed = value <= threshold
    elif op == ">=":
        passed = value >= threshold
    else:
        raise ValueError(f"unsupported threshold op: {op}")
    checks[name] = {
        "value": value,
        "op": op,
        "threshold": threshold,
        "passed": bool(passed),
        "skipped": False,
    }


def evaluate_thresholds(metrics: dict[str, Any], thresholds: GeometryThresholds) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    _record_check(
        checks,
        "global_disagreement",
        value=metrics["global"]["global_disagreement"],
        op="<=",
        threshold=thresholds.global_disagreement_max,
    )
    for radius, threshold in sorted(thresholds.boundary_band_disagreement_max.items()):
        radius_key = str(radius)
        band = metrics["boundary_bands"].get(radius_key)
        _record_check(
            checks,
            f"boundary_band_{radius}px_disagreement",
            value=None if band is None else band["disagreement_rate"],
            op="<=",
            threshold=threshold,
            skipped=band is None,
            reason=f"boundary radius {radius}px was not computed" if band is None else None,
        )
    _record_check(
        checks,
        "stable_region_false_flip_rate",
        value=metrics["temporal"]["stable_region"]["false_flip_rate"],
        op="<=",
        threshold=thresholds.stable_region_false_flip_rate_max,
    )
    _record_check(
        checks,
        "pair_transition_disagreement_rate",
        value=metrics["temporal"]["pair_transition"]["disagreement_rate"],
        op="<=",
        threshold=thresholds.pair_transition_disagreement_max,
    )
    _record_check(
        checks,
        "pair_transition_f1",
        value=metrics["temporal"]["pair_transition"]["f1"],
        op=">=",
        threshold=thresholds.pair_transition_f1_min,
    )
    for cls, threshold in sorted(thresholds.class_recall_min.items()):
        class_row = metrics["per_class"]["classes"].get(str(cls))
        support = 0 if class_row is None else int(class_row["support_pixels"])
        _record_check(
            checks,
            f"class_{cls}_recall",
            value=None if class_row is None else class_row["recall"],
            op=">=",
            threshold=threshold,
            skipped=class_row is None or support == 0,
            reason=f"class {cls} absent from baseline" if class_row is not None and support == 0 else None,
        )
    _record_check(
        checks,
        "tiny_speckle_rate",
        value=metrics["components"]["speckle_rate"],
        op="<=",
        threshold=thresholds.tiny_speckle_rate_max,
    )
    _record_check(
        checks,
        "max_component_centroid_jump_px",
        value=metrics["components"]["centroid"]["max_matched_jump_px"],
        op="<=",
        threshold=thresholds.max_component_centroid_jump_px,
    )
    _record_check(
        checks,
        "missing_component_rate",
        value=metrics["components"]["centroid"]["missing_component_rate"],
        op="<=",
        threshold=thresholds.missing_component_rate_max,
    )
    pass_values = [entry["passed"] for entry in checks.values() if not entry["skipped"]]
    return {
        "thresholds": _as_jsonable(asdict(thresholds)),
        "checks": checks,
        "overall_pass": bool(all(pass_values)) if pass_values else None,
    }


def compute_nerv_geometry_diagnostics(
    baseline_masks: torch.Tensor,
    candidate_masks: torch.Tensor,
    *,
    num_classes: int | None = 5,
    boundary_radii: tuple[int, ...] = DEFAULT_BOUNDARY_RADII,
    thresholds: GeometryThresholds | None = None,
    component_options: ComponentOptions | None = None,
    residual_ranking_options: ResidualRankingOptions | None = None,
    visual_primitive_options: VisualPrimitiveOptions | None = None,
    chunk_frames: int = 16,
    worst_pair_count: int = 10,
) -> dict[str, Any]:
    """Compute Alpha geometry diagnostics on CPU mask tensors.

    Args:
        baseline_masks: decoded baseline archive masks, shape ``(T,H,W)``.
        candidate_masks: decoded candidate masks, shape ``(T,H,W)``.
        num_classes: class count. Defaults to 5 for the contest SegNet masks.
        boundary_radii: boundary dilation radii in pixels.
        thresholds: optional JSON pass/fail thresholds.
        component_options: connected-component/speckle options.
        residual_ranking_options: optional CPU-only residual repair ranking options.
        visual_primitive_options: optional CPU-only visual primitive packet options.
        chunk_frames: frame chunk size for tensor metrics.
        worst_pair_count: number of worst frame pairs to include.

    Returns:
        JSON-serializable diagnostic dictionary.
    """
    baseline, candidate, resolved_classes = _validate_pair(
        baseline_masks,
        candidate_masks,
        num_classes=num_classes,
    )
    if component_options is None:
        component_options = ComponentOptions()
    if residual_ranking_options is None:
        residual_ranking_options = ResidualRankingOptions()
    if visual_primitive_options is None:
        visual_primitive_options = VisualPrimitiveOptions(
            critical_component_min_area=component_options.centroid_component_min_area,
            tiny_component_max_area=component_options.tiny_component_max_area,
            critical_classes=residual_ranking_options.critical_classes,
        )
    global_metrics, confusion = _global_and_confusion(
        baseline,
        candidate,
        num_classes=resolved_classes,
        chunk_frames=chunk_frames,
    )
    metrics = {
        "schema_version": 1,
        "diagnostic": "alpha_geo_0_nerv_geometry",
        "score_evidence_grade": "empirical",
        "scorer_proxy": False,
        "device": "cpu",
        "shape": {
            "frames": int(baseline.shape[0]),
            "height": int(baseline.shape[1]),
            "width": int(baseline.shape[2]),
            "num_classes": resolved_classes,
        },
        "global": global_metrics,
        "per_class": _class_metrics(confusion),
        "boundary_bands": _boundary_metrics(
            baseline,
            candidate,
            radii=tuple(int(r) for r in boundary_radii),
            chunk_frames=chunk_frames,
        ),
        "temporal": _temporal_metrics(
            baseline,
            candidate,
            worst_pair_count=worst_pair_count,
            chunk_frames=chunk_frames,
        ),
        "components": _component_metrics(
            baseline,
            candidate,
            num_classes=resolved_classes,
            options=component_options,
        ),
        "residual_region_ranking": _residual_region_ranking(
            baseline,
            candidate,
            num_classes=resolved_classes,
            options=residual_ranking_options,
        ),
    }
    metrics["visual_primitives"] = _alpha_geo_visual_primitives_packet(
        baseline,
        candidate,
        num_classes=resolved_classes,
        boundary_radii=tuple(int(r) for r in boundary_radii),
        existing_metrics=metrics,
        options=visual_primitive_options,
    )
    if thresholds is not None:
        metrics["pass_fail"] = evaluate_thresholds(metrics, thresholds)
    return _as_jsonable(metrics)


def _parse_class_recall_thresholds(raw: str | None) -> dict[int, float] | None:
    if raw is None:
        return None
    raw = raw.strip()
    if raw == "":
        return {}
    out: dict[int, float] = {}
    for item in raw.split(","):
        cls_s, value_s = item.split(":", 1)
        out[int(cls_s)] = float(value_s)
    return out


def _parse_int_tuple(raw: str) -> tuple[int, ...]:
    raw = raw.strip()
    if raw == "":
        return ()
    return tuple(int(item.strip()) for item in raw.split(",") if item.strip())


def _load_tensor_file(path: Path) -> torch.Tensor:
    if path.suffix == ".npy":
        return torch.from_numpy(np.load(path))
    if path.suffix == ".npz":
        data = np.load(path)
        for key in ("masks", "mask_classes", "class_ids"):
            if key in data:
                return torch.from_numpy(data[key])
        raise KeyError(f"{path} has no masks/mask_classes/class_ids array")
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, dict):
        for key in ("masks", "mask_classes", "class_ids"):
            if key in obj and isinstance(obj[key], torch.Tensor):
                return obj[key]
    raise TypeError(f"{path} did not contain a mask tensor")


def _render_nerv_mask_argmax_streaming(
    codec: torch.nn.Module,
    *,
    num_frames: int,
    height: int,
    width: int,
    batch_size: int,
) -> torch.Tensor:
    """Render a NeRV mask stream without materializing the full coord grid."""

    if num_frames <= 0 or height <= 0 or width <= 0:
        raise ValueError(
            "streaming NeRV render dimensions must be positive; "
            f"got frames={num_frames}, height={height}, width={width}"
        )
    if batch_size <= 0:
        raise ValueError(f"nrv_batch_size must be > 0; got {batch_size}")

    codec.to("cpu").eval()
    ts = torch.linspace(-1.0, 1.0, int(num_frames), dtype=torch.float32)
    ys = torch.linspace(-1.0, 1.0, int(height), dtype=torch.float32)
    xs = torch.linspace(-1.0, 1.0, int(width), dtype=torch.float32)
    frame_pixels = int(height) * int(width)
    total = int(num_frames) * frame_pixels
    out_argmax = torch.empty(total, dtype=torch.uint8)

    with torch.no_grad():
        for start in range(0, total, int(batch_size)):
            end = min(start + int(batch_size), total)
            flat = torch.arange(start, end, dtype=torch.int64)
            frame_idx = torch.div(flat, frame_pixels, rounding_mode="floor")
            rem = flat - frame_idx * frame_pixels
            y_idx = torch.div(rem, int(width), rounding_mode="floor")
            x_idx = rem - y_idx * int(width)
            coords = torch.stack([ts[frame_idx], ys[y_idx], xs[x_idx]], dim=-1)
            pred = codec(coords).argmax(dim=-1).to(torch.uint8)
            out_argmax[start:end] = pred.cpu()

    return out_argmax.reshape(int(num_frames), int(height), int(width))


def _mask_cache_fingerprint(
    metadata: dict[str, Any],
    *,
    expected_frames: int | None,
    height: int | None,
    width: int | None,
) -> dict[str, Any]:
    payload_name = str(metadata.get("archive_member_resolved") or metadata["path"])
    return {
        "schema_version": 1,
        "cache_kind": "diagnose_nerv_geometry_predecoded_mask_stream",
        "decoder_semantics": "class_id_tensor_THW_v1",
        "source_sha256": metadata["source_sha256"],
        "source_size_bytes": int(metadata["source_size_bytes"]),
        "source_suffix": Path(str(metadata["path"])).suffix.lower(),
        "payload_suffix": Path(payload_name).suffix.lower(),
        "archive_member_resolved": metadata.get("archive_member_resolved"),
        "archive_member_sha256": metadata.get("archive_member_sha256"),
        "archive_member_size_bytes": metadata.get("archive_member_size_bytes"),
        "expected_frames": None if expected_frames is None else int(expected_frames),
        "height": None if height is None else int(height),
        "width": None if width is None else int(width),
    }


def _mask_cache_key(fingerprint: dict[str, Any]) -> str:
    encoded = json.dumps(fingerprint, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mask_cache_paths(cache_dir: Path, cache_key: str) -> tuple[Path, Path]:
    return cache_dir / f"{cache_key}.pt", cache_dir / f"{cache_key}.json"


def _cache_record_common(
    *,
    cache_dir: Path,
    cache_key: str,
    tensor_path: Path,
    metadata_path: Path,
    status: str,
) -> dict[str, Any]:
    return {
        "enabled": True,
        "schema_version": 1,
        "status": status,
        "cache_dir": str(cache_dir),
        "cache_key": cache_key,
        "tensor_path": str(tensor_path),
        "metadata_path": str(metadata_path),
        "promotion_eligible": False,
        "score_claim_eligible": False,
        "score_evidence_grade": "empirical",
    }


def _load_predecoded_mask_cache(
    *,
    cache_dir: Path,
    cache_key: str,
    fingerprint: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]] | None:
    tensor_path, metadata_path = _mask_cache_paths(cache_dir, cache_key)
    if not tensor_path.exists() or not metadata_path.exists():
        return None
    try:
        cache_metadata = json.loads(metadata_path.read_text())
    except json.JSONDecodeError:
        return None
    if cache_metadata.get("fingerprint") != fingerprint:
        return None

    masks = _normalize_mask_tensor(_load_tensor_file(tensor_path), name="cached_decoded_mask")
    decoded_sha = _mask_tensor_sha256(masks)
    expected_sha = cache_metadata.get("decoded_mask_sha256")
    if expected_sha != decoded_sha:
        raise ValueError(
            f"predecoded mask cache sha mismatch for {tensor_path}: "
            f"metadata={expected_sha} tensor={decoded_sha}"
        )
    cache_record = _cache_record_common(
        cache_dir=cache_dir,
        cache_key=cache_key,
        tensor_path=tensor_path,
        metadata_path=metadata_path,
        status="hit",
    )
    cache_record.update(
        {
            "decoded_mask_sha256": decoded_sha,
            "decoded_mask_shape": [int(v) for v in masks.shape],
            "decoded_mask_dtype": str(masks.dtype),
        }
    )
    return masks, cache_record


def _write_predecoded_mask_cache(
    masks: torch.Tensor,
    *,
    cache_dir: Path,
    cache_key: str,
    fingerprint: dict[str, Any],
    decode_request: dict[str, Any],
) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    tensor_path, metadata_path = _mask_cache_paths(cache_dir, cache_key)
    normalized = _normalize_mask_tensor(masks, name="decoded_mask")
    decoded_sha = _mask_tensor_sha256(normalized)

    with tempfile.NamedTemporaryFile(
        prefix=f".{cache_key}.",
        suffix=".pt.tmp",
        dir=cache_dir,
        delete=False,
    ) as tmp_fh:
        tmp_tensor_path = Path(tmp_fh.name)
    with tempfile.NamedTemporaryFile(
        prefix=f".{cache_key}.",
        suffix=".json.tmp",
        dir=cache_dir,
        delete=False,
    ) as tmp_fh:
        tmp_metadata_path = Path(tmp_fh.name)
    try:
        torch.save({"masks": normalized}, tmp_tensor_path)
        cache_metadata = {
            "schema_version": 1,
            "cache_kind": "diagnose_nerv_geometry_predecoded_mask_stream",
            "fingerprint": fingerprint,
            "decode_request": _as_jsonable(decode_request),
            "decoded_mask_sha256": decoded_sha,
            "decoded_mask_sha256_algo": "sha256(shape,dtype,contiguous-raw-bytes)",
            "decoded_mask_shape": [int(v) for v in normalized.shape],
            "decoded_mask_dtype": str(normalized.dtype),
            "tensor_file": tensor_path.name,
            "promotion_eligible": False,
            "score_claim_eligible": False,
            "score_evidence_grade": "empirical",
        }
        tmp_metadata_path.write_text(json.dumps(cache_metadata, indent=2, sort_keys=True) + "\n")
        tmp_tensor_path.replace(tensor_path)
        tmp_metadata_path.replace(metadata_path)
    finally:
        for tmp_path in (tmp_tensor_path, tmp_metadata_path):
            if tmp_path.exists():
                tmp_path.unlink()

    cache_record = _cache_record_common(
        cache_dir=cache_dir,
        cache_key=cache_key,
        tensor_path=tensor_path,
        metadata_path=metadata_path,
        status="miss_written",
    )
    cache_record.update(
        {
            "decoded_mask_sha256": decoded_sha,
            "decoded_mask_shape": [int(v) for v in normalized.shape],
            "decoded_mask_dtype": str(normalized.dtype),
        }
    )
    return cache_record


def _load_mask_stream(
    path: Path,
    *,
    archive_member: str | None,
    expected_frames: int | None,
    height: int | None,
    width: int | None,
    nrv_batch_size: int,
) -> torch.Tensor:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".zip":
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            with zipfile.ZipFile(path, "r") as zf:
                infos = _validated_zip_infos(zf)
                names = set(infos)
                member = archive_member
                if member is None:
                    if "masks.nrv" in names:
                        member = "masks.nrv"
                    elif "masks.mkv" in names:
                        member = "masks.mkv"
                    else:
                        raise FileNotFoundError(f"{path} contains neither masks.nrv nor masks.mkv")
                member_path = PurePosixPath(member)
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise ValueError(f"unsafe archive member path: {member!r}")
                if member not in names:
                    raise FileNotFoundError(f"{path} missing archive member {member!r}")
                local_member = td_path / member_path.name
                local_member.write_bytes(zf.read(member))
            return _load_mask_stream(
                local_member,
                archive_member=None,
                expected_frames=expected_frames,
                height=height,
                width=width,
                nrv_batch_size=nrv_batch_size,
            )
    if path.suffix in {".pt", ".pth", ".npy", ".npz"}:
        return _load_tensor_file(path)
    if path.suffix == ".nrv":
        if expected_frames is None or height is None or width is None:
            raise ValueError("loading .nrv requires --num-frames, --height, and --width")
        from tac.nerv_mask_codec import decode_nerv_codec

        codec = decode_nerv_codec(path.read_bytes())
        return _render_nerv_mask_argmax_streaming(
            codec,
            num_frames=expected_frames,
            height=height,
            width=width,
            batch_size=nrv_batch_size,
        )
    if path.suffix.lower() in {".mkv", ".mp4", ".webm"}:
        from tac.mask_codec import decode_masks

        return decode_masks(path, expected_frames=expected_frames)
    raise ValueError(f"unsupported mask path suffix for {path}")


def _resolve_archive_member_metadata(
    path: Path,
    *,
    archive_member: str | None,
) -> dict[str, Any]:
    with zipfile.ZipFile(path, "r") as zf:
        infos = _validated_zip_infos(zf)
        names = set(infos)
        resolved_member = archive_member
        if resolved_member is None:
            if "masks.nrv" in names:
                resolved_member = "masks.nrv"
            elif "masks.mkv" in names:
                resolved_member = "masks.mkv"
            else:
                raise FileNotFoundError(f"{path} contains neither masks.nrv nor masks.mkv")
        member_path = PurePosixPath(resolved_member)
        if member_path.is_absolute() or ".." in member_path.parts:
            raise ValueError(f"unsafe archive member path: {resolved_member!r}")
        if resolved_member not in infos:
            raise FileNotFoundError(f"{path} missing archive member {resolved_member!r}")
        info = infos[resolved_member]
        data = zf.read(resolved_member)
    return {
        "archive_member_requested": archive_member,
        "archive_member_resolved": resolved_member,
        "archive_member_size_bytes": int(info.file_size),
        "archive_member_compressed_bytes": int(info.compress_size),
        "archive_member_sha256": _sha256_bytes(data),
    }


def _load_mask_stream_with_metadata(
    path: Path,
    *,
    archive_member: str | None,
    expected_frames: int | None,
    height: int | None,
    width: int | None,
    nrv_batch_size: int,
    mask_cache_dir: Path | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    metadata: dict[str, Any] = {
        "path": str(path),
        "source_size_bytes": int(path.stat().st_size),
        "source_sha256": _sha256_file(path),
    }
    effective_member = archive_member
    if path.suffix == ".zip":
        archive_metadata = _resolve_archive_member_metadata(
            path,
            archive_member=archive_member,
        )
        metadata.update(archive_metadata)
        effective_member = str(archive_metadata["archive_member_resolved"])
    payload_suffix = Path(str(effective_member or path)).suffix.lower()
    if payload_suffix == ".nrv":
        metadata["nrv_decode_mode"] = "bounded_cpu_coordinate_stream_v1"
    decode_request = {
        "expected_frames": expected_frames,
        "height": height,
        "width": width,
        "nrv_batch_size": nrv_batch_size,
        "nrv_decode_mode": metadata.get("nrv_decode_mode"),
        "archive_member": effective_member,
    }
    fingerprint = _mask_cache_fingerprint(
        metadata,
        expected_frames=expected_frames,
        height=height,
        width=width,
    )
    cache_key = _mask_cache_key(fingerprint)
    if mask_cache_dir is not None:
        cache_dir = Path(mask_cache_dir)
        cached = _load_predecoded_mask_cache(
            cache_dir=cache_dir,
            cache_key=cache_key,
            fingerprint=fingerprint,
        )
        if cached is not None:
            masks, cache_record = cached
            metadata.update(
                {
                    "decoded_mask_sha256": cache_record["decoded_mask_sha256"],
                    "decoded_mask_sha256_algo": "sha256(shape,dtype,contiguous-raw-bytes)",
                    "decoded_mask_shape": cache_record["decoded_mask_shape"],
                    "decoded_mask_dtype": cache_record["decoded_mask_dtype"],
                    "predecoded_cache": _as_jsonable(cache_record),
                }
            )
            return masks, metadata

    masks = _load_mask_stream(
        path,
        archive_member=effective_member,
        expected_frames=expected_frames,
        height=height,
        width=width,
        nrv_batch_size=nrv_batch_size,
    )
    normalized = _normalize_mask_tensor(masks, name="decoded_mask")
    cache_record = None
    if mask_cache_dir is not None:
        cache_record = _write_predecoded_mask_cache(
            normalized,
            cache_dir=Path(mask_cache_dir),
            cache_key=cache_key,
            fingerprint=fingerprint,
            decode_request=decode_request,
        )
    metadata.update(
        {
            "decoded_mask_sha256": _mask_tensor_sha256(normalized),
            "decoded_mask_sha256_algo": "sha256(shape,dtype,contiguous-raw-bytes)",
            "decoded_mask_shape": [int(v) for v in normalized.shape],
            "decoded_mask_dtype": str(normalized.dtype),
            "predecoded_cache": _as_jsonable(
                cache_record
                if cache_record is not None
                else {
                    "enabled": False,
                    "status": "disabled",
                    "promotion_eligible": False,
                    "score_claim_eligible": False,
                    "score_evidence_grade": "empirical",
                }
            ),
        }
    )
    return normalized, metadata


def _build_thresholds_from_args(args: argparse.Namespace) -> GeometryThresholds | None:
    if args.threshold_preset == "none":
        return None
    thresholds = GeometryThresholds.from_preset(args.threshold_preset)
    boundary = dict(thresholds.boundary_band_disagreement_max)
    if args.boundary_max is not None:
        boundary = {radius: args.boundary_max for radius in args.boundary_radii}
    for radius, value in (
        (1, args.boundary_1px_max),
        (2, args.boundary_2px_max),
        (3, args.boundary_3px_max),
        (5, args.boundary_5px_max),
    ):
        if value is not None:
            boundary[radius] = value
    class_recall = thresholds.class_recall_min
    parsed_class_recall = _parse_class_recall_thresholds(args.class_recall_min)
    if parsed_class_recall is not None:
        class_recall = parsed_class_recall
    return GeometryThresholds(
        global_disagreement_max=(
            thresholds.global_disagreement_max if args.global_max is None else args.global_max
        ),
        boundary_band_disagreement_max=boundary,
        stable_region_false_flip_rate_max=(
            thresholds.stable_region_false_flip_rate_max
            if args.stable_false_flip_max is None
            else args.stable_false_flip_max
        ),
        pair_transition_disagreement_max=(
            thresholds.pair_transition_disagreement_max
            if args.pair_transition_disagreement_max is None
            else args.pair_transition_disagreement_max
        ),
        pair_transition_f1_min=(
            thresholds.pair_transition_f1_min if args.pair_transition_f1_min is None else args.pair_transition_f1_min
        ),
        class_recall_min=class_recall,
        tiny_speckle_rate_max=(
            thresholds.tiny_speckle_rate_max if args.tiny_speckle_rate_max is None else args.tiny_speckle_rate_max
        ),
        max_component_centroid_jump_px=(
            thresholds.max_component_centroid_jump_px
            if args.max_component_centroid_jump_px is None
            else args.max_component_centroid_jump_px
        ),
        missing_component_rate_max=(
            thresholds.missing_component_rate_max
            if args.missing_component_rate_max is None
            else args.missing_component_rate_max
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="Decoded baseline tensor/video/archive path.")
    parser.add_argument("--candidate", type=Path, required=True, help="Decoded candidate tensor/video/archive/.nrv path.")
    parser.add_argument("--output-json", type=Path, required=True, help="Path to write diagnostics JSON.")
    parser.add_argument("--baseline-member", type=str, default=None, help="Archive member for --baseline when it is a zip.")
    parser.add_argument("--candidate-member", type=str, default=None, help="Archive member for --candidate when it is a zip.")
    parser.add_argument("--num-frames", type=int, default=None, help="Required when loading .nrv.")
    parser.add_argument("--height", type=int, default=None, help="Required when loading .nrv.")
    parser.add_argument("--width", type=int, default=None, help="Required when loading .nrv.")
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--boundary-radii", type=int, nargs="+", default=list(DEFAULT_BOUNDARY_RADII))
    parser.add_argument("--chunk-frames", type=int, default=16)
    parser.add_argument("--worst-pair-count", type=int, default=10)
    parser.add_argument("--nrv-batch-size", type=int, default=262_144)
    parser.add_argument(
        "--mask-cache-dir",
        type=Path,
        default=None,
        help=(
            "Optional deterministic decoded-mask cache directory. Cache keys "
            "are derived from source/member hashes and requested dimensions; "
            "cache metadata is empirical/no-claim only."
        ),
    )
    parser.add_argument("--tiny-component-max-area", type=int, default=8)
    parser.add_argument("--speckle-mismatch-fraction-min", type=float, default=0.5)
    parser.add_argument("--centroid-component-min-area", type=int, default=9)
    parser.add_argument("--residual-region-count", type=int, default=20)
    parser.add_argument("--residual-region-min-area", type=int, default=1)
    parser.add_argument("--residual-region-boundary-radius", type=int, default=2)
    parser.add_argument("--residual-region-near-field-y-fraction", type=float, default=0.60)
    parser.add_argument("--residual-region-critical-classes", type=str, default="1,2")
    parser.add_argument(
        "--visual-frame-stride",
        type=int,
        default=1,
        help=(
            "Subsample frames for the expensive advisory visual-primitives "
            "packet. Full global/boundary/temporal scalar metrics still use "
            "all frames."
        ),
    )
    parser.add_argument(
        "--visual-boundary-distance-sample-cap",
        type=int,
        default=4096,
        help="Per-frame boundary-distance sample cap for fallback distance summaries.",
    )
    parser.add_argument(
        "--visual-boundary-distance-global-sample-cap",
        type=int,
        default=262_144,
        help=(
            "Global deterministic reservoir cap for advisory boundary-distance "
            "p95 samples. Scalar chamfer/coverage accumulators still use all "
            "available boundary pixels when scipy EDT is available."
        ),
    )
    parser.add_argument("--visual-track-min-iou", type=float, default=0.05)
    parser.add_argument("--visual-track-max-centroid-jump-px", type=float, default=16.0)
    parser.add_argument("--visual-max-component-failures", type=int, default=20)
    parser.add_argument(
        "--visual-component-classes",
        type=str,
        default="",
        help=(
            "Comma-separated class IDs for advisory visual component extraction. "
            "Empty means all classes. Any non-empty subset must include the critical classes."
        ),
    )
    parser.add_argument(
        "--visual-track-classes",
        type=str,
        default="",
        help=(
            "Comma-separated class IDs for advisory temporal track proxies. "
            "Empty means all classes. Any non-empty subset must include the critical classes."
        ),
    )
    parser.add_argument(
        "--visual-disable-temporal-tracks",
        action="store_true",
        help=(
            "Skip advisory component-track proxies while preserving full scalar "
            "temporal mask metrics. This is empirical/no-claim triage only."
        ),
    )
    parser.add_argument("--threshold-preset", choices=["exploratory", "promotion", "none"], default="exploratory")
    parser.add_argument("--global-max", type=float, default=None)
    parser.add_argument("--boundary-max", type=float, default=None, help="Override every configured boundary radius gate.")
    parser.add_argument("--boundary-1px-max", type=float, default=None)
    parser.add_argument("--boundary-2px-max", type=float, default=None)
    parser.add_argument("--boundary-3px-max", type=float, default=None)
    parser.add_argument("--boundary-5px-max", type=float, default=None)
    parser.add_argument("--stable-false-flip-max", type=float, default=None)
    parser.add_argument("--pair-transition-disagreement-max", type=float, default=None)
    parser.add_argument("--pair-transition-f1-min", type=float, default=None)
    parser.add_argument("--class-recall-min", type=str, default=None, help="Comma list like '1:0.999,2:0.999'.")
    parser.add_argument("--tiny-speckle-rate-max", type=float, default=None)
    parser.add_argument("--max-component-centroid-jump-px", type=float, default=None)
    parser.add_argument("--missing-component-rate-max", type=float, default=None)
    args = parser.parse_args()

    baseline_member = args.baseline_member
    if baseline_member is None and args.baseline.suffix == ".zip":
        baseline_member = "masks.mkv"
    baseline, baseline_metadata = _load_mask_stream_with_metadata(
        args.baseline,
        archive_member=baseline_member,
        expected_frames=args.num_frames,
        height=args.height,
        width=args.width,
        nrv_batch_size=args.nrv_batch_size,
        mask_cache_dir=args.mask_cache_dir,
    )
    if args.num_frames is None:
        args.num_frames = int(_normalize_mask_tensor(baseline, name="baseline").shape[0])
    if args.height is None:
        args.height = int(_normalize_mask_tensor(baseline, name="baseline").shape[1])
    if args.width is None:
        args.width = int(_normalize_mask_tensor(baseline, name="baseline").shape[2])
    candidate, candidate_metadata = _load_mask_stream_with_metadata(
        args.candidate,
        archive_member=args.candidate_member,
        expected_frames=args.num_frames,
        height=args.height,
        width=args.width,
        nrv_batch_size=args.nrv_batch_size,
        mask_cache_dir=args.mask_cache_dir,
    )
    thresholds = _build_thresholds_from_args(args)
    result = compute_nerv_geometry_diagnostics(
        baseline,
        candidate,
        num_classes=args.num_classes,
        boundary_radii=tuple(args.boundary_radii),
        thresholds=thresholds,
        component_options=ComponentOptions(
            tiny_component_max_area=args.tiny_component_max_area,
            speckle_mismatch_fraction_min=args.speckle_mismatch_fraction_min,
            centroid_component_min_area=args.centroid_component_min_area,
        ),
        residual_ranking_options=ResidualRankingOptions(
            max_regions=args.residual_region_count,
            min_area=args.residual_region_min_area,
            boundary_radius=args.residual_region_boundary_radius,
            near_field_y_fraction=args.residual_region_near_field_y_fraction,
            critical_classes=_parse_int_tuple(args.residual_region_critical_classes),
        ),
        visual_primitive_options=VisualPrimitiveOptions(
            frame_stride=args.visual_frame_stride,
            critical_component_min_area=args.centroid_component_min_area,
            tiny_component_max_area=args.tiny_component_max_area,
            boundary_distance_sample_cap=args.visual_boundary_distance_sample_cap,
            boundary_distance_global_sample_cap=args.visual_boundary_distance_global_sample_cap,
            track_min_iou=args.visual_track_min_iou,
            track_max_centroid_jump_px=args.visual_track_max_centroid_jump_px,
            max_component_failures=args.visual_max_component_failures,
            critical_classes=_parse_int_tuple(args.residual_region_critical_classes),
            component_classes=_parse_int_tuple(args.visual_component_classes),
            track_classes=_parse_int_tuple(args.visual_track_classes),
            temporal_tracks_enabled=not args.visual_disable_temporal_tracks,
        ),
        chunk_frames=args.chunk_frames,
        worst_pair_count=args.worst_pair_count,
    )
    result["inputs"] = {
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "baseline_member": baseline_metadata.get("archive_member_resolved", baseline_member),
        "candidate_member": candidate_metadata.get("archive_member_resolved", args.candidate_member),
        "baseline_member_requested": baseline_member,
        "candidate_member_requested": args.candidate_member,
        "baseline_source": _as_jsonable(baseline_metadata),
        "candidate_source": _as_jsonable(candidate_metadata),
    }
    result["visual_primitives"]["source"].update(
        {
            "baseline": str(args.baseline),
            "candidate": str(args.candidate),
            "baseline_member": baseline_metadata.get("archive_member_resolved", baseline_member),
            "candidate_member": candidate_metadata.get("archive_member_resolved", args.candidate_member),
            "baseline_source_sha256": baseline_metadata.get("source_sha256"),
            "candidate_source_sha256": candidate_metadata.get("source_sha256"),
            "diagnose_nerv_geometry_json": str(args.output_json),
        }
    )
    result["diagnostic_config"] = {
        "boundary_radii": [int(r) for r in args.boundary_radii],
        "chunk_frames": int(args.chunk_frames),
        "worst_pair_count": int(args.worst_pair_count),
        "nrv_batch_size": int(args.nrv_batch_size),
        "mask_cache_dir": None if args.mask_cache_dir is None else str(args.mask_cache_dir),
        "threshold_preset": args.threshold_preset,
        "thresholds": None if thresholds is None else _as_jsonable(asdict(thresholds)),
        "component_options": _as_jsonable(
            asdict(
                ComponentOptions(
                    tiny_component_max_area=args.tiny_component_max_area,
                    speckle_mismatch_fraction_min=args.speckle_mismatch_fraction_min,
                    centroid_component_min_area=args.centroid_component_min_area,
                )
            )
        ),
        "residual_ranking_options": _as_jsonable(
            asdict(
                ResidualRankingOptions(
                    max_regions=args.residual_region_count,
                    min_area=args.residual_region_min_area,
                    boundary_radius=args.residual_region_boundary_radius,
                    near_field_y_fraction=args.residual_region_near_field_y_fraction,
                    critical_classes=_parse_int_tuple(args.residual_region_critical_classes),
                )
            )
        ),
        "visual_primitive_options": _as_jsonable(
            asdict(
                VisualPrimitiveOptions(
                    frame_stride=args.visual_frame_stride,
                    critical_component_min_area=args.centroid_component_min_area,
                    tiny_component_max_area=args.tiny_component_max_area,
                    boundary_distance_sample_cap=args.visual_boundary_distance_sample_cap,
                    boundary_distance_global_sample_cap=args.visual_boundary_distance_global_sample_cap,
                    track_min_iou=args.visual_track_min_iou,
                    track_max_centroid_jump_px=args.visual_track_max_centroid_jump_px,
                    max_component_failures=args.visual_max_component_failures,
                    critical_classes=_parse_int_tuple(args.residual_region_critical_classes),
                    component_classes=_parse_int_tuple(args.visual_component_classes),
                    track_classes=_parse_int_tuple(args.visual_track_classes),
                    temporal_tracks_enabled=not args.visual_disable_temporal_tracks,
                )
            )
        ),
    }
    result["command"] = {
        "tool": "experiments/diagnose_nerv_geometry.py",
        "argv": list(sys.argv),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    overall = result.get("pass_fail", {}).get("overall_pass")
    print(
        f"[alpha-geo-0] wrote {args.output_json} "
        f"global={result['global']['global_disagreement']:.6g} "
        f"pair_transition={result['temporal']['pair_transition']['disagreement_rate']:.6g} "
        f"pass={overall}",
        flush=True,
    )
    return 0 if overall is not False else 2


if __name__ == "__main__":
    raise SystemExit(main())
