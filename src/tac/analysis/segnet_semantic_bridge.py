# SPDX-License-Identifier: MIT
"""Semantic bridge for SegNet boundary repair and adapter lanes.

The contest SegNet component is a hard argmax-disagreement loss over scorer-grid
pixels. This module turns source-vs-candidate SegNet logits into a typed,
false-authority artifact that postfilters, LoRA/adapters, deterministic repair
rules, and selector codecs can consume without reinterpreting readiness fields.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from tac.analysis.segnet_boundary_marginals import (
    boundary_mask_from_labels,
    logit_margin,
)
from tac.semantic_label_contract import (
    CONTEST_SEGNET_CLASS_NAMES,
    CONTEST_SEGNET_CLASSES,
    NUM_CONTEST_SEGNET_CLASSES,
)

SEGNET_SEMANTIC_BRIDGE_SCHEMA: Final[str] = "segnet_semantic_bridge.v1"
SEGNET_SEMANTIC_SURFACE_ARTIFACTS_SCHEMA: Final[str] = (
    "segnet_semantic_bridge_surface_artifacts.v1"
)
SEGNET_SEMANTIC_SURFACE_ARRAY_NAMES: Final[tuple[str, ...]] = (
    "source_argmax",
    "candidate_argmax",
    "source_top2",
    "source_margin",
    "candidate_margin",
    "boundary_mask",
    "wrong_mask",
    "hinge_map",
    "sample_ids",
)

FALSE_AUTHORITY: Final[dict[str, bool]] = {
    "score_claim": False,
    "score_claim_valid": False,
    "score_claim_eligible": False,
    "score_authority": False,
    "dispatch_attempted": False,
    "promotable": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "budget_spend_allowed": False,
    "ready_for_budget_spend": False,
    "ready_for_exact_eval_dispatch": False,
}

GENERALIZATION_MODES: Final[tuple[str, ...]] = (
    "contest_fixed_dataset",
    "fleet_adaptable",
    "mixed",
)


class SegnetSemanticBridgeError(ValueError):
    """Raised when semantic bridge inputs are not contest-shape coherent."""


@dataclass(frozen=True)
class SemanticBridgeConfig:
    """Configuration for source-vs-candidate SegNet semantic analysis."""

    candidate_id: str
    generalization_mode: str = "mixed"
    boundary_dilation: int = 5
    low_margin_threshold: float = 1.0
    hinge_margin: float = 0.25
    axis_tag: str = "[analysis; SegNet source-vs-candidate semantic bridge]"

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise SegnetSemanticBridgeError("candidate_id must be non-empty")
        if self.generalization_mode not in GENERALIZATION_MODES:
            raise SegnetSemanticBridgeError(
                "generalization_mode must be one of "
                f"{GENERALIZATION_MODES!r}; got {self.generalization_mode!r}"
            )
        if self.boundary_dilation < 1:
            raise SegnetSemanticBridgeError("boundary_dilation must be >= 1")
        if self.low_margin_threshold <= 0.0:
            raise SegnetSemanticBridgeError("low_margin_threshold must be > 0")
        if self.hinge_margin <= 0.0:
            raise SegnetSemanticBridgeError("hinge_margin must be > 0")


def build_segnet_semantic_bridge(
    *,
    source_logits: np.ndarray,
    candidate_logits: np.ndarray,
    config: SemanticBridgeConfig,
    sample_ids: list[int] | None = None,
    pair_component_rows: Mapping[int, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a planner-consumable semantic bridge from SegNet logits.

    Args:
        source_logits: Source/ground-truth SegNet logits, shape ``(N, C, H, W)``.
        candidate_logits: Candidate/inflated SegNet logits, same shape.
        config: Bridge configuration.
        sample_ids: Optional source pair/frame ids for each sample.
        pair_component_rows: Optional rows keyed by pair id, usually from
            ``pair_component_error_xray_v1``. These are copied as context only.

    Returns:
        A JSON-serializable false-authority artifact. It does not claim score.
    """

    source = _logits4(source_logits, "source_logits")
    candidate = _logits4(candidate_logits, "candidate_logits")
    if source.shape != candidate.shape:
        raise SegnetSemanticBridgeError(
            f"source_logits shape {source.shape} != candidate_logits shape {candidate.shape}"
        )
    n_samples, n_classes, height, width = source.shape
    if n_samples < 1:
        raise SegnetSemanticBridgeError("at least one sample is required")
    if n_classes != NUM_CONTEST_SEGNET_CLASSES:
        raise SegnetSemanticBridgeError(
            f"expected {NUM_CONTEST_SEGNET_CLASSES} SegNet classes; got {n_classes}"
        )
    if sample_ids is None:
        sample_ids = list(range(n_samples))
    if len(sample_ids) != n_samples:
        raise SegnetSemanticBridgeError(
            f"sample_ids length {len(sample_ids)} does not match N={n_samples}"
        )

    source_labels = source.argmax(axis=1).astype(np.int64)
    candidate_labels = candidate.argmax(axis=1).astype(np.int64)
    wrong = source_labels != candidate_labels
    boundary = boundary_mask_from_labels(source_labels, dilation=config.boundary_dilation)
    interior = ~boundary
    source_top2 = top2_class_indices(source)
    source_margins = logit_margin(source)
    candidate_margins = logit_margin(candidate)
    low_margin = source_margins <= float(config.low_margin_threshold)
    hinge_map = crammer_singer_hinge_for_targets(
        candidate,
        source_labels,
        margin=config.hinge_margin,
    )
    top1_top2_error = wrong & (candidate_labels == source_top2)
    out_of_pair_error = wrong & ~top1_top2_error

    confusion = _confusion_matrix(source_labels, candidate_labels, n_classes)
    source_top1_top2_surface = _confusion_matrix(source_labels, source_top2, n_classes)
    boundary_confusion = _confusion_matrix(
        source_labels[boundary],
        candidate_labels[boundary],
        n_classes,
    )
    interior_confusion = _confusion_matrix(
        source_labels[interior],
        candidate_labels[interior],
        n_classes,
    )

    pair_rows = _pair_rows(
        sample_ids=sample_ids,
        source_labels=source_labels,
        wrong=wrong,
        top1_top2_error=top1_top2_error,
        out_of_pair_error=out_of_pair_error,
        boundary=boundary,
        interior=interior,
        low_margin=low_margin,
        source_margins=source_margins,
        candidate_margins=candidate_margins,
        hinge_map=hinge_map,
        pair_component_rows=pair_component_rows or {},
    )
    class_rows = _class_rows(
        confusion=confusion,
        boundary_confusion=boundary_confusion,
        interior_confusion=interior_confusion,
    )
    summary = _summary(
        wrong=wrong,
        top1_top2_error=top1_top2_error,
        out_of_pair_error=out_of_pair_error,
        boundary=boundary,
        interior=interior,
        low_margin=low_margin,
        source_margins=source_margins,
        candidate_margins=candidate_margins,
        hinge_map=hinge_map,
    )
    backlog = _executable_backlog(summary=summary, class_rows=class_rows, config=config)

    return {
        "schema": SEGNET_SEMANTIC_BRIDGE_SCHEMA,
        "candidate_id": config.candidate_id,
        "axis_tag": config.axis_tag,
        "evidence_grade": "analysis_false_authority_segnet_semantic_bridge",
        "evidence_semantics": (
            "source-vs-candidate SegNet argmax, margin, boundary, and class "
            "transition analysis for repair acquisition; build archive bytes, "
            "prove receiver runtime consumption, and run exact eval before any "
            "score or promotion claim"
        ),
        "generalization_mode": config.generalization_mode,
        "contest_overfit_policy": {
            "allowed_for_contest_fixed_dataset": config.generalization_mode
            in {"contest_fixed_dataset", "mixed"},
            "must_not_be_rebranded_as_fleet_ready": True,
            "fleet_adaptable_rows_require_holdout_or_online_calibration": True,
        },
        "mathematical_objective": {
            "score_ground_truth": "SegNet component is mean argmax disagreement over scorer-grid pixels",
            "local_repair_loss": "crammer_singer_multiclass_hinge_on_raw_logits",
            "loss_formula": "relu(max_{j!=y_i} z_ij - z_i,y_i + margin_i)",
            "zero_condition": "zero iff target argmax wins by margin",
            "soft_loss_role": (
                "KL, target-vs-rest, and decision-KD are diagnostic/comparison "
                "arms because they can match teacher uncertainty at boundaries"
            ),
            "global_action": (
                "minimize sum_i repair_value_i * hinge_i + lambda_rate * bytes "
                "+ lambda_pose * pose_risk under archive/runtime custody gates"
            ),
        },
        "scorer_grid": {
            "height": int(height),
            "width": int(width),
            "num_classes": int(n_classes),
            "class_names": dict(CONTEST_SEGNET_CLASS_NAMES),
        },
        "config": {
            "boundary_dilation": int(config.boundary_dilation),
            "low_margin_threshold": float(config.low_margin_threshold),
            "hinge_margin": float(config.hinge_margin),
        },
        "summary": summary,
        "confusion_matrix_source_to_candidate": confusion.astype(int).tolist(),
        "source_top1_top2_surface": source_top1_top2_surface.astype(int).tolist(),
        "boundary_confusion_matrix_source_to_candidate": boundary_confusion.astype(int).tolist(),
        "interior_confusion_matrix_source_to_candidate": interior_confusion.astype(int).tolist(),
        "class_rows": class_rows,
        "dominant_error_pairs_real_world": _dominant_error_pairs_real_world(confusion),
        "sample_rows": pair_rows,
        "recommended_training": {
            "segnet_distillation_objective": "boundary_argmax_hinge",
            "segnet_hinge_margin": float(config.hinge_margin),
            "teacher_loss_verdict": (
                "boundary_argmax_hinge_required_out_of_pair_impostor_mass_material"
                if summary["error_is_out_of_pair_spread_fraction"] >= 0.10
                else "decision_kd_diagnostic_arm_errors_mostly_top1_top2"
            ),
            "selection_rule": (
                "rank by hinge_mass_per_sample, boundary_wrong_fraction, "
                "class-transition concentration, and byte-credit pressure"
            ),
            "parallel_candidate_lanes": [
                "deterministic_boundary_repair",
                "deterministic_boundary_postfilter",
                "mlx_lora_or_dora_boundary_adapter",
                "contest_fixed_selector_correction_mask",
                "fleet_adaptable_boundary_rule_induction",
            ],
        },
        "executable_backlog": backlog,
        "missing_optional_context": [
            "aligned_pixel_delta_on_scorer_grid",
            "ego_motion_pose_teacher_context",
            "archive_byte_credit_curve",
            "receiver_runtime_proof",
            "exact_cpu_cuda_axis_payload",
        ],
        **FALSE_AUTHORITY,
    }


def top2_class_indices(logits: np.ndarray) -> np.ndarray:
    """Return the second-highest class id per scorer-grid pixel."""

    arr = _logits4(logits, "logits")
    order = np.argsort(arr, axis=1)
    return order[:, -2, :, :].astype(np.int64)


def crammer_singer_hinge_for_targets(
    logits: np.ndarray,
    targets: np.ndarray,
    *,
    margin: float,
) -> np.ndarray:
    """Return Crammer-Singer multiclass hinge per pixel.

    The spelling keeps the historical "Crammer-Singer" name visible while the
    function remains a small numpy primitive suitable for tests and tooling.
    """

    arr = _logits4(logits, "logits")
    target_arr = np.asarray(targets)
    if target_arr.shape != (arr.shape[0], arr.shape[2], arr.shape[3]):
        raise SegnetSemanticBridgeError(
            f"targets shape {target_arr.shape} does not match logits spatial shape "
            f"{(arr.shape[0], arr.shape[2], arr.shape[3])}"
        )
    if margin <= 0.0:
        raise SegnetSemanticBridgeError("margin must be > 0")
    if target_arr.min() < 0 or target_arr.max() >= arr.shape[1]:
        raise SegnetSemanticBridgeError("targets contain class ids outside logits channels")
    flat_targets = target_arr.astype(np.int64)
    target_logits = np.take_along_axis(arr, flat_targets[:, None, :, :], axis=1)[:, 0]
    impostor_logits = arr.copy()
    np.put_along_axis(
        impostor_logits,
        flat_targets[:, None, :, :],
        -np.inf,
        axis=1,
    )
    max_impostor = impostor_logits.max(axis=1)
    return np.maximum(0.0, max_impostor - target_logits + float(margin))


def build_segnet_semantic_surface_arrays(
    *,
    source_logits: np.ndarray,
    candidate_logits: np.ndarray,
    sample_ids: list[int] | None = None,
    boundary_dilation: int = 5,
    hinge_margin: float = 0.25,
) -> dict[str, np.ndarray]:
    """Return executable per-pixel surfaces consumed by repair/adapters.

    This is deliberately a TAC library primitive rather than a CLI helper so
    queue builders, MLX training rows, deterministic postfilters, and review
    tests all share the same semantic contract.
    """

    source = _logits4(source_logits, "source_logits")
    candidate = _logits4(candidate_logits, "candidate_logits")
    if source.shape != candidate.shape:
        raise SegnetSemanticBridgeError(
            f"source_logits shape {source.shape} != candidate_logits shape {candidate.shape}"
        )
    if source.shape[1] != NUM_CONTEST_SEGNET_CLASSES:
        raise SegnetSemanticBridgeError(
            f"expected {NUM_CONTEST_SEGNET_CLASSES} SegNet classes; got {source.shape[1]}"
        )
    if boundary_dilation < 1:
        raise SegnetSemanticBridgeError("boundary_dilation must be >= 1")
    if hinge_margin <= 0.0:
        raise SegnetSemanticBridgeError("hinge_margin must be > 0")
    sample_ids_arr = (
        np.arange(source.shape[0], dtype=np.int64)
        if sample_ids is None
        else np.asarray(sample_ids, dtype=np.int64)
    )
    if sample_ids_arr.shape != (source.shape[0],):
        raise SegnetSemanticBridgeError(
            f"sample_ids length {sample_ids_arr.size} does not match N={source.shape[0]}"
        )

    source_labels = source.argmax(axis=1).astype(np.uint8)
    candidate_labels = candidate.argmax(axis=1).astype(np.uint8)
    source_labels_i64 = source_labels.astype(np.int64)
    return {
        "source_argmax": source_labels,
        "candidate_argmax": candidate_labels,
        "source_top2": top2_class_indices(source).astype(np.uint8),
        "source_margin": logit_margin(source).astype(np.float32),
        "candidate_margin": logit_margin(candidate).astype(np.float32),
        "boundary_mask": boundary_mask_from_labels(
            source_labels_i64,
            dilation=boundary_dilation,
        ).astype(np.uint8),
        "wrong_mask": (source_labels != candidate_labels).astype(np.uint8),
        "hinge_map": crammer_singer_hinge_for_targets(
            candidate,
            source_labels_i64,
            margin=hinge_margin,
        ).astype(np.float32),
        "sample_ids": sample_ids_arr,
    }


def write_segnet_semantic_surface_npz(
    *,
    source_logits: np.ndarray,
    candidate_logits: np.ndarray,
    sample_ids: list[int] | None,
    boundary_dilation: int,
    hinge_margin: float,
    path: str | Path,
    allow_overwrite: bool = False,
) -> dict[str, Any]:
    """Write canonical per-pixel semantic surfaces and return a custody record."""

    surface_path = Path(path)
    if surface_path.exists() and not allow_overwrite:
        raise SegnetSemanticBridgeError(
            f"refusing to overwrite existing surface: {surface_path}"
        )
    arrays = build_segnet_semantic_surface_arrays(
        source_logits=source_logits,
        candidate_logits=candidate_logits,
        sample_ids=sample_ids,
        boundary_dilation=boundary_dilation,
        hinge_margin=hinge_margin,
    )
    surface_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(surface_path, **arrays)
    return {
        "path": str(surface_path),
        "bytes": surface_path.stat().st_size,
        "sha256": _sha256_file(surface_path),
        "arrays": list(SEGNET_SEMANTIC_SURFACE_ARRAY_NAMES),
        "array_shapes": {
            name: list(arrays[name].shape)
            for name in SEGNET_SEMANTIC_SURFACE_ARRAY_NAMES
        },
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _logits4(value: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 4:
        raise SegnetSemanticBridgeError(
            f"{name} must have shape (N, C, H, W); got {arr.shape}"
        )
    if arr.shape[0] < 1 or arr.shape[1] < 2 or arr.shape[2] < 1 or arr.shape[3] < 1:
        raise SegnetSemanticBridgeError(f"{name} has invalid shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise SegnetSemanticBridgeError(f"{name} must contain only finite values")
    return arr


def _confusion_matrix(
    source_labels: np.ndarray,
    candidate_labels: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    source = np.asarray(source_labels, dtype=np.int64).reshape(-1)
    candidate = np.asarray(candidate_labels, dtype=np.int64).reshape(-1)
    if source.shape != candidate.shape:
        raise SegnetSemanticBridgeError("confusion inputs must have matching shapes")
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    if source.size == 0:
        return matrix
    if source.min() < 0 or source.max() >= n_classes:
        raise SegnetSemanticBridgeError("source labels outside class range")
    if candidate.min() < 0 or candidate.max() >= n_classes:
        raise SegnetSemanticBridgeError("candidate labels outside class range")
    np.add.at(matrix, (source, candidate), 1)
    return matrix


def _summary(
    *,
    wrong: np.ndarray,
    top1_top2_error: np.ndarray,
    out_of_pair_error: np.ndarray,
    boundary: np.ndarray,
    interior: np.ndarray,
    low_margin: np.ndarray,
    source_margins: np.ndarray,
    candidate_margins: np.ndarray,
    hinge_map: np.ndarray,
) -> dict[str, Any]:
    total = int(wrong.size)
    boundary_pixels = int(boundary.sum())
    interior_pixels = int(interior.sum())
    wrong_pixels = int(wrong.sum())
    top1_top2_wrong = int(top1_top2_error.sum())
    out_of_pair_wrong = int(out_of_pair_error.sum())
    boundary_wrong = int((wrong & boundary).sum())
    interior_wrong = int((wrong & interior).sum())
    low_margin_pixels = int(low_margin.sum())
    low_margin_wrong = int((wrong & low_margin).sum())
    return {
        "n_samples": int(wrong.shape[0]),
        "pixels_per_sample": int(np.prod(wrong.shape[1:])),
        "total_pixels": total,
        "wrong_pixels": wrong_pixels,
        "argmax_disagreement_rate": _ratio(wrong_pixels, total),
        "error_is_top1_top2_flip_pixels": top1_top2_wrong,
        "error_is_top1_top2_flip_fraction": _ratio(top1_top2_wrong, wrong_pixels),
        "error_is_out_of_pair_spread_pixels": out_of_pair_wrong,
        "error_is_out_of_pair_spread_fraction": _ratio(out_of_pair_wrong, wrong_pixels),
        "boundary_pixels": boundary_pixels,
        "boundary_fraction": _ratio(boundary_pixels, total),
        "boundary_wrong_pixels": boundary_wrong,
        "boundary_wrong_fraction_of_all_pixels": _ratio(boundary_wrong, total),
        "boundary_error_share": _ratio(boundary_wrong, wrong_pixels),
        "interior_pixels": interior_pixels,
        "interior_wrong_pixels": interior_wrong,
        "interior_wrong_fraction_of_all_pixels": _ratio(interior_wrong, total),
        "low_source_margin_pixels": low_margin_pixels,
        "low_source_margin_wrong_pixels": low_margin_wrong,
        "low_source_margin_error_share": _ratio(low_margin_wrong, wrong_pixels),
        "mean_source_margin": float(source_margins.mean()),
        "p10_source_margin": float(np.percentile(source_margins, 10.0)),
        "mean_candidate_margin": float(candidate_margins.mean()),
        "p10_candidate_margin": float(np.percentile(candidate_margins, 10.0)),
        "hinge_loss_sum": float(hinge_map.sum()),
        "hinge_loss_mean": float(hinge_map.mean()),
        "wrong_hinge_loss_sum": float(hinge_map[wrong].sum()) if wrong_pixels else 0.0,
    }


def _class_rows(
    *,
    confusion: np.ndarray,
    boundary_confusion: np.ndarray,
    interior_confusion: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for class_id in range(confusion.shape[0]):
        row = confusion[class_id]
        source_pixels = int(row.sum())
        correct = int(row[class_id])
        wrong = source_pixels - correct
        top_impostors = [
            {
                "candidate_class_id": int(idx),
                "candidate_class_name": CONTEST_SEGNET_CLASS_NAMES[int(idx)],
                "pixels": int(row[idx]),
                "share_of_source_class": _ratio(int(row[idx]), source_pixels),
            }
            for idx in np.argsort(row)[::-1]
            if int(idx) != class_id and int(row[idx]) > 0
        ][:3]
        rows.append(
            {
                "class_id": int(class_id),
                "class_name": CONTEST_SEGNET_CLASS_NAMES[class_id],
                "source_pixels": source_pixels,
                "correct_pixels": correct,
                "wrong_pixels": wrong,
                "error_rate_within_source_class": _ratio(wrong, source_pixels),
                "boundary_wrong_pixels": int(
                    boundary_confusion[class_id].sum()
                    - boundary_confusion[class_id, class_id]
                ),
                "interior_wrong_pixels": int(
                    interior_confusion[class_id].sum()
                    - interior_confusion[class_id, class_id]
                ),
                "top_impostors": top_impostors,
            }
        )
    return rows


def _pair_rows(
    *,
    sample_ids: list[int],
    source_labels: np.ndarray,
    wrong: np.ndarray,
    top1_top2_error: np.ndarray,
    out_of_pair_error: np.ndarray,
    boundary: np.ndarray,
    interior: np.ndarray,
    low_margin: np.ndarray,
    source_margins: np.ndarray,
    candidate_margins: np.ndarray,
    hinge_map: np.ndarray,
    pair_component_rows: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for local_idx, sample_id in enumerate(sample_ids):
        sample_wrong = wrong[local_idx]
        sample_top1_top2 = top1_top2_error[local_idx]
        sample_out_of_pair = out_of_pair_error[local_idx]
        sample_boundary = boundary[local_idx]
        sample_interior = interior[local_idx]
        sample_low = low_margin[local_idx]
        total = int(sample_wrong.size)
        wrong_pixels = int(sample_wrong.sum())
        row: dict[str, Any] = {
            "sample_index": int(local_idx),
            "pair_idx": int(sample_id),
            "pixels": total,
            "wrong_pixels": wrong_pixels,
            "argmax_disagreement_rate": _ratio(wrong_pixels, total),
            "top1_top2_error_pixels": int(sample_top1_top2.sum()),
            "top1_top2_error_fraction": _ratio(int(sample_top1_top2.sum()), wrong_pixels),
            "out_of_pair_error_pixels": int(sample_out_of_pair.sum()),
            "out_of_pair_error_fraction": _ratio(int(sample_out_of_pair.sum()), wrong_pixels),
            "boundary_wrong_pixels": int((sample_wrong & sample_boundary).sum()),
            "boundary_wrong_fraction": _ratio(
                int((sample_wrong & sample_boundary).sum()),
                max(1, int(sample_boundary.sum())),
            ),
            "interior_wrong_pixels": int((sample_wrong & sample_interior).sum()),
            "low_source_margin_wrong_pixels": int((sample_wrong & sample_low).sum()),
            "mean_source_margin": float(source_margins[local_idx].mean()),
            "p10_source_margin": float(np.percentile(source_margins[local_idx], 10.0)),
            "mean_candidate_margin": float(candidate_margins[local_idx].mean()),
            "hinge_loss_sum": float(hinge_map[local_idx].sum()),
            "dominant_source_classes": _dominant_source_classes(source_labels[local_idx]),
        }
        context = pair_component_rows.get(int(sample_id))
        if context:
            row["pair_component_context"] = {
                key: context[key]
                for key in (
                    "pose_dist",
                    "seg_dist",
                    "pose_score_contribution",
                    "seg_score_contribution",
                    "component_score_no_rate",
                )
                if key in context
            }
        rows.append(row)
    rows.sort(key=lambda item: (item["hinge_loss_sum"], item["wrong_pixels"]), reverse=True)
    return rows


def _dominant_source_classes(labels: np.ndarray) -> list[dict[str, Any]]:
    counts = np.bincount(labels.reshape(-1), minlength=NUM_CONTEST_SEGNET_CLASSES)
    total = int(counts.sum())
    return [
        {
            "class_id": int(idx),
            "class_name": CONTEST_SEGNET_CLASS_NAMES[int(idx)],
            "pixel_share": _ratio(int(counts[idx]), total),
        }
        for idx in np.argsort(counts)[::-1][:3]
        if int(counts[idx]) > 0
    ]


def _dominant_error_pairs_real_world(confusion: np.ndarray) -> list[dict[str, Any]]:
    descriptions = {
        item.class_id: item.description for item in CONTEST_SEGNET_CLASSES
    }
    pairs: list[tuple[int, int, int]] = []
    for source_class in range(confusion.shape[0]):
        for candidate_class in range(confusion.shape[1]):
            count = int(confusion[source_class, candidate_class])
            if source_class != candidate_class and count > 0:
                pairs.append((count, source_class, candidate_class))
    pairs.sort(reverse=True)
    total_error = sum(count for count, _, _ in pairs)
    return [
        {
            "source_class_id": int(source_class),
            "source_class_name": CONTEST_SEGNET_CLASS_NAMES[source_class],
            "candidate_class_id": int(candidate_class),
            "candidate_class_name": CONTEST_SEGNET_CLASS_NAMES[candidate_class],
            "pixels": int(count),
            "fraction_of_error_mass": _ratio(count, total_error),
            "real_world_read": (
                f"{descriptions[source_class]} -> {descriptions[candidate_class]}"
            ),
        }
        for count, source_class, candidate_class in pairs[:8]
    ]


def _executable_backlog(
    *,
    summary: Mapping[str, Any],
    class_rows: list[Mapping[str, Any]],
    config: SemanticBridgeConfig,
) -> list[dict[str, Any]]:
    highest_error_classes = sorted(
        class_rows,
        key=lambda row: (
            float(row["error_rate_within_source_class"]),
            int(row["wrong_pixels"]),
        ),
        reverse=True,
    )[:3]
    boundary_pressure = float(summary["boundary_error_share"])
    low_margin_pressure = float(summary["low_source_margin_error_share"])
    def _mode_gate(row_mode: str) -> dict[str, Any]:
        enqueueable = (
            config.generalization_mode == "mixed"
            or row_mode == config.generalization_mode
            or (
                config.generalization_mode == "contest_fixed_dataset"
                and row_mode == "mixed"
            )
        )
        return {
            "enqueueable_under_requested_generalization_mode": enqueueable,
            "compatibility_blockers": (
                []
                if enqueueable
                else [
                    f"{row_mode}_lane_not_enqueueable_for_"
                    f"{config.generalization_mode}_bridge"
                ]
            ),
        }

    return [
        {
            "family_id": "deterministic_boundary_repair",
            "generalization_mode": "contest_fixed_dataset",
            "why": (
                "fixed contest videos permit source-indexed correction masks or "
                "class/boundary rule tables when archive bytes can pay for them"
            ),
            "acquisition_features": {
                "boundary_error_share": boundary_pressure,
                "hinge_loss_sum": summary["hinge_loss_sum"],
            },
            "next_materializer_task": (
                "emit byte-closed correction-mask or class-boundary rule-table "
                "candidate and prove inflate.sh consumes it"
            ),
            "score_authority": False,
            **_mode_gate("contest_fixed_dataset"),
        },
        {
            "family_id": "deterministic_boundary_postfilter",
            "generalization_mode": "contest_fixed_dataset",
            "why": (
                "post-decode deterministic filters can target boundary argmax "
                "hinge mass without retraining the representation"
            ),
            "acquisition_features": {
                "boundary_error_share": boundary_pressure,
                "low_source_margin_error_share": low_margin_pressure,
                "hinge_loss_sum": summary["hinge_loss_sum"],
            },
            "next_materializer_task": (
                "emit byte-closed runtime postfilter candidate, bind filter "
                "parameters in archive/runtime custody, and prove inflate.sh consumes it"
            ),
            "score_authority": False,
            **_mode_gate("contest_fixed_dataset"),
        },
        {
            "family_id": "mlx_lora_or_dora_boundary_adapter",
            "generalization_mode": "mixed",
            "why": (
                "small adapter can overfit contest boundary logits while retaining "
                "a fleet path through held-out video calibration"
            ),
            "acquisition_features": {
                "low_source_margin_error_share": low_margin_pressure,
                "top_source_classes": [
                    {
                        "class_id": row["class_id"],
                        "class_name": row["class_name"],
                        "error_rate": row["error_rate_within_source_class"],
                    }
                    for row in highest_error_classes
                ],
            },
            "next_materializer_task": (
                "train MLX adapter with boundary_argmax_hinge plus PoseNet guard; "
                "export only through shared runtime bridge"
            ),
            "score_authority": False,
            **_mode_gate("mixed"),
        },
        {
            "family_id": "fleet_adaptable_boundary_rule_induction",
            "generalization_mode": "fleet_adaptable",
            "why": (
                "production deployment should learn class/boundary behavior from "
                "features available at runtime, not fixed contest labels"
            ),
            "acquisition_features": {
                "class_error_rows": [
                    {
                        "class_id": row["class_id"],
                        "class_name": row["class_name"],
                        "error_rate": row["error_rate_within_source_class"],
                    }
                    for row in highest_error_classes
                ],
            },
            "next_materializer_task": (
                "derive runtime-observable feature rule, validate on holdout, then "
                "compare against contest-overfit lane without merging authority"
            ),
            "score_authority": False,
            **_mode_gate("fleet_adaptable"),
        },
    ]


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "FALSE_AUTHORITY",
    "GENERALIZATION_MODES",
    "SEGNET_SEMANTIC_BRIDGE_SCHEMA",
    "SEGNET_SEMANTIC_SURFACE_ARRAY_NAMES",
    "SEGNET_SEMANTIC_SURFACE_ARTIFACTS_SCHEMA",
    "SegnetSemanticBridgeError",
    "SemanticBridgeConfig",
    "build_segnet_semantic_bridge",
    "build_segnet_semantic_surface_arrays",
    "crammer_singer_hinge_for_targets",
    "top2_class_indices",
    "write_segnet_semantic_surface_npz",
]
