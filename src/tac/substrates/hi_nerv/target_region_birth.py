# SPDX-License-Identifier: MIT
"""Torch-free worst-target-region selection for HiNeRV hard class birth.

The contest SegNet term is an argmax-disagreement *rate*: a class can gain
probability mass everywhere while winning the argmax nowhere, so per-class
aggregate losses can improve while ``target_min_ratio`` stays ``0.0`` and the
receiver argmax probe still fails.  Birth only counts when a specific
connected component of wrong pixels flips its argmax on the receiver surface.

This module prices every connected wrong-region of every target class in
exact contest score units (``100 * unsolved_pixels / total_scored_pixels``),
selects the worst one deterministically, and builds the receipts a scoped
birth actuator must emit.  It is numpy-only on purpose: the same selector and
receipt math serve the live MLX forward, fake-quant forward, archive
parse-back, and inflate replay surfaces without backend drift.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy import ndimage

TARGET_REGION_DEBT_SCHEMA = "hi_nerv_target_region_debt.v1"
TARGET_REGION_BIRTH_RECEIPT_SCHEMA = "hi_nerv_target_region_birth_receipt.v1"
POSE_TRUSTED_BIRTH_ADMISSION_DECISION_SCHEMA = "hi_nerv_pose_trusted_birth_admission_decision.v1"

# Archive-charged tensors the birth actuator may update.  This mirrors the
# live-SegNet-scoped bootstrap list in ``mlx_renderer`` (late surfaces with
# local spatial leverage); early coarse/stem tensors move many pairs at once
# and can break PoseNet geometry, so they stay frozen during birth.
ALLOWED_BIRTH_UPDATE_EXACT: tuple[str, ...] = ("latents_fine",)
ALLOWED_BIRTH_UPDATE_PREFIXES: tuple[str, ...] = (
    "latents_fine.",
    "feature_grids.",
    "fine_injector.",
    "head_rgb_1.",
)
ALLOWED_POSE_COMPENSATION_UPDATE_EXACT: tuple[str, ...] = ("head_rgb_0",)
ALLOWED_POSE_COMPENSATION_UPDATE_PREFIXES: tuple[str, ...] = ("head_rgb_0.",)


def allowed_birth_update_name(name: Any) -> bool:
    """Return True when a flattened parameter name is birth-updatable."""

    flat = ".".join(str(part) for part in name) if isinstance(name, (tuple, list)) else str(name)
    if flat in ALLOWED_BIRTH_UPDATE_EXACT:
        return True
    return any(flat.startswith(prefix) for prefix in ALLOWED_BIRTH_UPDATE_PREFIXES)


def allowed_pose_compensation_update_name(name: Any) -> bool:
    """Return True when a flattened parameter name is frame0 compensation-only."""

    flat = ".".join(str(part) for part in name) if isinstance(name, (tuple, list)) else str(name)
    if flat in ALLOWED_POSE_COMPENSATION_UPDATE_EXACT:
        return True
    return any(flat.startswith(prefix) for prefix in ALLOWED_POSE_COMPENSATION_UPDATE_PREFIXES)


@dataclass(frozen=True)
class TargetRegionDebt:
    """One connected target-class region priced in exact contest score units."""

    batch_index: int
    class_index: int
    region_label: int
    region_pixel_count: int
    region_unsolved_pixel_count: int
    region_hard_ratio: float
    frame_pixel_count: int
    total_scored_pixels: int
    score_debt_units: float
    frame_fraction: float
    bbox_y0: int
    bbox_y1: int
    bbox_x0: int
    bbox_x1: int

    def full_equivalent_score_debt_units(
        self,
        full_eval_total_scored_pixels: int,
    ) -> float:
        """Convert batch-local debt to full-eval score units.

        The stored ``score_debt_units`` uses the batch-local
        ``total_scored_pixels`` normalizer.  Value-per-byte and promotion math
        must use the SAME units as ``evaluate.py`` (all scored pixels of the
        full eval), otherwise a 4-pair smoke region looks ~150x overvalued
        relative to byte price.
        """

        if full_eval_total_scored_pixels <= 0:
            raise ValueError(f"full_eval_total_scored_pixels must be positive; got {full_eval_total_scored_pixels}")
        return float(100.0 * self.region_unsolved_pixel_count / full_eval_total_scored_pixels)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema"] = TARGET_REGION_DEBT_SCHEMA
        # Denominator authority: this row's score units are normalized by the
        # supplied batch only.  Consumers pricing bytes must re-normalize via
        # full_equivalent_score_debt_units(...) before comparing to byte cost.
        payload["normalization_authority"] = "batch_local_scored_pixels"
        payload["score_debt_units_local"] = payload["score_debt_units"]
        # NOTE: deliberately NO score_claim/promotion authority keys here.
        # These rows are embedded under substrate_artifact_metadata, whose
        # harness custody validator REFUSES nested authority/readiness keys
        # (single-custody-surface rule): authority lives only on the canonical
        # TrainingArtifact. Spreading even false-valued copies is forbidden.
        return payload


def _validate_label_pair(
    target_labels: np.ndarray,
    candidate_argmax: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    target = np.asarray(target_labels)
    candidate = np.asarray(candidate_argmax)
    if target.ndim == 4 and target.shape[-1] == 1:
        target = target[..., 0]
    if candidate.ndim == 4 and candidate.shape[-1] == 1:
        candidate = candidate[..., 0]
    if target.ndim != 3 or candidate.ndim != 3:
        raise ValueError(
            "target_labels and candidate_argmax must be BHW (or BHW1); got "
            f"target={target.shape} candidate={candidate.shape}"
        )
    if target.shape != candidate.shape:
        raise ValueError(
            "target_labels and candidate_argmax shapes must match; got "
            f"target={target.shape} candidate={candidate.shape}"
        )
    if target.size == 0:
        raise ValueError("target_labels must be non-empty")
    return target.astype(np.int64, copy=False), candidate.astype(np.int64, copy=False)


def find_target_region_debts(
    target_labels: np.ndarray,
    candidate_argmax: np.ndarray,
    *,
    min_region_pixels: int = 1,
) -> list[TargetRegionDebt]:
    """Price every 4-connected target-class debt component in score units.

    ``score_debt_units = 100 * unsolved_pixels / total_scored_pixels`` where
    ``total_scored_pixels = B * H * W`` for the supplied batch.  The
    normalizer is batch-local and recorded on every row so a 4-pair smoke
    cannot silently masquerade as full-video score units.

    Positive-debt regions are connected components of target pixels that the
    current candidate argmax gets wrong.  Solved classes may still emit zero
    debt rows for legacy no-unsolved checks, but birth actuators never receive
    already-won support inside a positive-debt region.
    """

    if min_region_pixels < 1:
        raise ValueError(f"min_region_pixels must be >= 1; got {min_region_pixels}")
    target, candidate = _validate_label_pair(target_labels, candidate_argmax)
    batch, height, width = target.shape
    total_scored = int(batch * height * width)
    frame_pixels = int(height * width)
    rows: list[TargetRegionDebt] = []
    for batch_index in range(batch):
        frame_target = target[batch_index]
        frame_candidate = candidate[batch_index]
        for class_index in np.unique(frame_target).tolist():
            class_mask = frame_target == class_index
            unsolved_mask = class_mask & (frame_candidate != class_index)
            # Positive-debt rows must be connected WRONG-pixel components. A
            # full semantic component can contain already-solved support; using
            # that as the actuator region lets birth updates destroy pixels
            # that evaluate.py already scores as correct.
            debt_mask = unsolved_mask if np.any(unsolved_mask) else class_mask
            # scipy default 2-D structure is the 4-connected cross.
            labeled, component_count = ndimage.label(debt_mask)
            if component_count == 0:
                continue
            for region_label in range(1, component_count + 1):
                region_mask = labeled == region_label
                region_pixels = int(np.count_nonzero(region_mask))
                if region_pixels < min_region_pixels:
                    continue
                unsolved_pixels = int(np.count_nonzero(region_mask & unsolved_mask))
                ys, xs = np.nonzero(region_mask)
                rows.append(
                    TargetRegionDebt(
                        batch_index=int(batch_index),
                        class_index=int(class_index),
                        region_label=int(region_label),
                        region_pixel_count=region_pixels,
                        region_unsolved_pixel_count=unsolved_pixels,
                        region_hard_ratio=float((region_pixels - unsolved_pixels) / region_pixels),
                        frame_pixel_count=frame_pixels,
                        total_scored_pixels=total_scored,
                        score_debt_units=float(100.0 * unsolved_pixels / total_scored),
                        frame_fraction=float(region_pixels / frame_pixels),
                        bbox_y0=int(ys.min()),
                        bbox_y1=int(ys.max()) + 1,
                        bbox_x0=int(xs.min()),
                        bbox_x1=int(xs.max()) + 1,
                    )
                )
    return rows


def select_worst_target_region(
    debts: Sequence[TargetRegionDebt],
) -> TargetRegionDebt:
    """Return the highest-score-debt region with a deterministic tie-break."""

    if not debts:
        raise ValueError("at least one target-region debt row is required")
    for row in debts:
        if not math.isfinite(row.score_debt_units) or row.score_debt_units < 0.0:
            raise ValueError(
                "score_debt_units must be finite and non-negative; got "
                f"{row.score_debt_units} for batch={row.batch_index} "
                f"class={row.class_index} region={row.region_label}"
            )
    return min(
        debts,
        key=lambda row: (
            -row.score_debt_units,
            row.batch_index,
            row.class_index,
            row.region_label,
        ),
    )


def select_worst_target_region_with_mask(
    target_labels: np.ndarray,
    candidate_argmax: np.ndarray,
    *,
    min_region_pixels: int = 1,
    excluded_region_keys: Sequence[Sequence[int]] | None = None,
    forced_region_key: Sequence[int] | None = None,
) -> tuple[TargetRegionDebt, np.ndarray]:
    """Return the worst region row plus its full-batch BHW float32 mask."""

    target, candidate = _validate_label_pair(target_labels, candidate_argmax)
    debts = find_target_region_debts(
        target_labels,
        candidate_argmax,
        min_region_pixels=min_region_pixels,
    )
    forced: tuple[int, int, int] | None = None
    if forced_region_key is not None:
        forced_parts = tuple(int(part) for part in forced_region_key)
        if len(forced_parts) != 3:
            raise ValueError(
                "forced_region_key must contain exactly batch,class,region; "
                f"got {forced_region_key!r}"
            )
        forced = (forced_parts[0], forced_parts[1], forced_parts[2])
    excluded = {
        (int(parts[0]), int(parts[1]), int(parts[2]))
        for parts in (tuple(key) for key in (excluded_region_keys or ()))
        if len(parts) == 3
    }
    if forced is not None:
        debts = [
            row
            for row in debts
            if (row.batch_index, row.class_index, row.region_label) == forced
        ]
        if not debts:
            raise ValueError(
                "forced target-region key not found in current debt surface: "
                f"batch={forced[0]} class={forced[1]} region={forced[2]}"
            )
    elif excluded:
        filtered = [
            row
            for row in debts
            if (row.batch_index, row.class_index, row.region_label) not in excluded
        ]
        positive = [row for row in filtered if row.region_unsolved_pixel_count > 0]
        debts = positive or filtered
    worst = select_worst_target_region(debts)
    frame_target = target[worst.batch_index]
    frame_candidate = candidate[worst.batch_index]
    if worst.region_unsolved_pixel_count > 0:
        label_mask = (frame_target == worst.class_index) & (
            frame_candidate != worst.class_index
        )
    else:
        label_mask = frame_target == worst.class_index
    labeled, _count = ndimage.label(label_mask)
    mask = np.zeros(target.shape, dtype=np.float32)
    mask[worst.batch_index] = (labeled == worst.region_label).astype(np.float32)
    if int(np.count_nonzero(mask)) != worst.region_pixel_count:
        raise RuntimeError(
            "worst-region mask reconstruction drifted from the priced row; "
            f"mask={int(np.count_nonzero(mask))} row={worst.region_pixel_count}"
        )
    return worst, mask


def region_margin_stats(
    logits_bhwc: np.ndarray,
    region_mask_bhw: np.ndarray,
    class_index: int,
) -> dict[str, float]:
    """Return raw frontier-margin stats for one class within one region.

    The margin convention is PR95's ``impostor - class`` (no floor, no relu):
    negative at a pixel means the class wins the argmax there.  ``p50`` is the
    median over region pixels — the quantity whose *delta* the crux-trace
    consumer ingests as ``worst_region_margin_p50_delta``.
    """

    logits = np.asarray(logits_bhwc, dtype=np.float64)
    mask = np.asarray(region_mask_bhw)
    if logits.ndim != 4:
        raise ValueError(f"logits must be BHWC; got shape {logits.shape}")
    if mask.shape != logits.shape[:3]:
        raise ValueError(f"region mask BHW must match logits BHW; got mask={mask.shape} logits={logits.shape[:3]}")
    class_count = int(logits.shape[-1])
    if not 0 <= int(class_index) < class_count:
        raise ValueError(f"class_index {class_index} outside logits classes {class_count}")
    flat_mask = mask.reshape(-1) > 0.0
    region_pixels = int(np.count_nonzero(flat_mask))
    if region_pixels == 0:
        raise ValueError("region mask selects zero pixels")
    flat_logits = logits.reshape(-1, class_count)[flat_mask]
    class_logit = flat_logits[:, int(class_index)]
    impostor = np.copy(flat_logits)
    impostor[:, int(class_index)] = -np.inf
    impostor_logit = impostor.max(axis=1)
    margin = impostor_logit - class_logit
    hard_won = int(np.count_nonzero(margin < 0.0))
    return {
        "region_pixel_count": float(region_pixels),
        "region_hard_ratio": float(hard_won / region_pixels),
        "region_hard_won_pixels": float(hard_won),
        "margin_min": float(np.min(margin)),
        "margin_p50": float(np.median(margin)),
        "margin_mean": float(np.mean(margin)),
    }


def region_argmax_transition_counts(
    initial_argmax: np.ndarray,
    final_argmax: np.ndarray,
    region_mask_bhw: np.ndarray,
    class_index: int,
) -> dict[str, int]:
    """Disambiguate receiver argmax motion from target-class birth.

    ``argmax_changed_count_region`` is receiver-motion telemetry only: it
    counts churn of ANY class. Birth admission must use the target-class
    transitions — ``wrong_to_target`` (hard won), ``target_to_wrong`` (hard
    lost), and their difference ``net_target_support_delta``. A region can
    post tens of thousands of flips while the net target support is zero or
    negative; conflating the two overclaims birth.
    """

    before = np.asarray(initial_argmax)
    after = np.asarray(final_argmax)
    region = np.asarray(region_mask_bhw) > 0.0
    if before.shape != after.shape or before.shape != region.shape:
        raise ValueError(
            f"argmax/region shapes must match; got before={before.shape} after={after.shape} region={region.shape}"
        )
    cls = int(class_index)
    before_target = before == cls
    after_target = after == cls
    changed = before != after
    wrong_to_target = int(np.count_nonzero(region & ~before_target & after_target))
    target_to_wrong = int(np.count_nonzero(region & before_target & ~after_target))
    wrong_to_wrong = int(np.count_nonzero(region & changed & ~before_target & ~after_target))
    return {
        "argmax_changed_count_region": int(np.count_nonzero(region & changed)),
        "wrong_to_target_count": wrong_to_target,
        "target_to_wrong_count": target_to_wrong,
        "wrong_to_wrong_count": wrong_to_wrong,
        "target_hard_won_count": wrong_to_target,
        "target_hard_lost_count": target_to_wrong,
        "net_target_support_delta": wrong_to_target - target_to_wrong,
    }


def pose_trusted_birth_admission_decision(
    *,
    old_d_seg: float,
    new_d_seg: float,
    old_d_pose: float | None,
    new_d_pose: float | None,
    pose_output_l2_delta: float | None,
    raw_pose_cap_l2: float,
    exact_score_epsilon: float = 1.0e-6,
    catastrophic_relative_cap: float = 0.25,
    catastrophic_pose_score_regression_cap: float = 0.5,
    catastrophic_pose_output_l2_hard_cap: float | None = None,
) -> dict[str, Any]:
    """Return the v6 pose-trusted birth admission decision.

    The exact nonlinear non-rate score is the primary authority:
    ``100*Delta d_seg + sqrt(10*d_pose_new) - sqrt(10*d_pose_old)``.  The old
    raw pose-output cap is retained only as a counterfactual decision surface.
    Catastrophic pose guard is a blow-up catcher, not a tradeoff policer.
    """

    for name, value in (
        ("old_d_seg", old_d_seg),
        ("new_d_seg", new_d_seg),
        ("raw_pose_cap_l2", raw_pose_cap_l2),
        ("exact_score_epsilon", exact_score_epsilon),
        ("catastrophic_relative_cap", catastrophic_relative_cap),
        ("catastrophic_pose_score_regression_cap", catastrophic_pose_score_regression_cap),
    ):
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{name} must be finite and non-negative; got {value}")
    for name, value in (
        ("old_d_pose", old_d_pose),
        ("new_d_pose", new_d_pose),
        ("pose_output_l2_delta", pose_output_l2_delta),
        ("catastrophic_pose_output_l2_hard_cap", catastrophic_pose_output_l2_hard_cap),
    ):
        if value is not None and (not math.isfinite(float(value)) or float(value) < 0.0):
            raise ValueError(f"{name} must be None or finite and non-negative; got {value}")

    seg_score_delta = 100.0 * (float(new_d_seg) - float(old_d_seg))
    pose_score_delta: float | None = None
    exact_delta_score_nonrate: float | None = None
    if old_d_pose is not None and new_d_pose is not None:
        pose_score_delta = math.sqrt(10.0 * float(new_d_pose)) - math.sqrt(10.0 * float(old_d_pose))
        exact_delta_score_nonrate = seg_score_delta + pose_score_delta

    raw_cap_satisfied = pose_output_l2_delta is None or float(pose_output_l2_delta) <= float(raw_pose_cap_l2)
    exact_score_satisfied = (
        exact_delta_score_nonrate is not None
        and float(exact_delta_score_nonrate) < -float(exact_score_epsilon)
    )

    catastrophic_reasons: list[str] = []
    if old_d_pose is None or new_d_pose is None or pose_score_delta is None:
        catastrophic_reasons.append("pose_distortion_endpoint_missing")
    else:
        if float(new_d_pose) > float(old_d_pose) * (1.0 + float(catastrophic_relative_cap)):
            catastrophic_reasons.append("d_pose_relative_cap_exceeded")
        if float(pose_score_delta) > float(catastrophic_pose_score_regression_cap):
            catastrophic_reasons.append("pose_score_regression_cap_exceeded")
    if (
        catastrophic_pose_output_l2_hard_cap is not None
        and pose_output_l2_delta is not None
        and float(pose_output_l2_delta) > float(catastrophic_pose_output_l2_hard_cap)
    ):
        catastrophic_reasons.append("pose_output_l2_hard_cap_exceeded")
    catastrophic_guard_satisfied = not catastrophic_reasons
    accepted = bool(exact_score_satisfied and catastrophic_guard_satisfied)
    would_accept_exact_score_if_raw_cap_disabled = bool(exact_score_satisfied and catastrophic_guard_satisfied)
    would_accept_without_catastrophic_guard = bool(exact_score_satisfied)
    rejection_source: str | None = None
    if not accepted:
        if not exact_score_satisfied:
            rejection_source = "rejected_by_exact_delta_score"
        elif not catastrophic_guard_satisfied:
            rejection_source = "rejected_by_catastrophic_pose_guard"

    return {
        "schema": POSE_TRUSTED_BIRTH_ADMISSION_DECISION_SCHEMA,
        "old_d_seg": float(old_d_seg),
        "new_d_seg": float(new_d_seg),
        "old_d_pose": (None if old_d_pose is None else float(old_d_pose)),
        "new_d_pose": (None if new_d_pose is None else float(new_d_pose)),
        "seg_score_delta": float(seg_score_delta),
        "pose_score_delta": pose_score_delta,
        "exact_delta_score_nonrate": exact_delta_score_nonrate,
        "delta_score_nonrate": exact_delta_score_nonrate,
        "pose_output_l2_delta": (None if pose_output_l2_delta is None else float(pose_output_l2_delta)),
        "raw_pose_cap_l2": float(raw_pose_cap_l2),
        "raw_cap_decision": "satisfied" if raw_cap_satisfied else "violated_counterfactual_only",
        "raw_pose_cap_result": "satisfied" if raw_cap_satisfied else "violated_counterfactual_only",
        "exact_score_epsilon": float(exact_score_epsilon),
        "exact_score_decision": "accepted" if exact_score_satisfied else "rejected",
        "catastrophic_relative_cap": float(catastrophic_relative_cap),
        "catastrophic_pose_score_regression_cap": float(catastrophic_pose_score_regression_cap),
        "catastrophic_pose_output_l2_hard_cap": (
            None
            if catastrophic_pose_output_l2_hard_cap is None
            else float(catastrophic_pose_output_l2_hard_cap)
        ),
        "catastrophic_guard_decision": "satisfied" if catastrophic_guard_satisfied else "rejected",
        "catastrophic_guard_reasons": catastrophic_reasons,
        "accepted": accepted,
        "would_accept_exact_score_if_raw_cap_disabled": would_accept_exact_score_if_raw_cap_disabled,
        "would_accept_without_catastrophic_guard": would_accept_without_catastrophic_guard,
        "rejected_by_raw_pose_cap": False,
        "would_reject_under_raw_pose_cap": bool(not raw_cap_satisfied),
        "rejected_by_exact_delta_score": bool(rejection_source == "rejected_by_exact_delta_score"),
        "rejected_by_catastrophic_pose_guard": bool(rejection_source == "rejected_by_catastrophic_pose_guard"),
        "rejection_source": rejection_source,
        "raw_cap_is_counterfactual_only": True,
    }


def birth_action_id(
    *,
    debt: TargetRegionDebt,
    initial_group_sha256: Mapping[str, str],
    trained_groups: Sequence[str],
) -> str:
    """Return the stable identity of ONE birth action across surfaces.

    L4 survival is only proof when the same action is traced through live,
    fakequant, parse-back, and inflate surfaces; a freshly re-solved birth
    under fakequant is a different experiment. Survival receipts must CARRY
    this id from the live receipt, never recompute it from their own state.
    """

    import hashlib

    payload = "|".join(
        [
            "hinerv_target_region_birth",
            str(debt.batch_index),
            str(debt.class_index),
            str(debt.region_label),
            str(debt.region_pixel_count),
            ",".join(f"{name}={initial_group_sha256[name]}" for name in sorted(initial_group_sha256)),
            ",".join(sorted(str(g) for g in trained_groups)),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_target_region_birth_receipt(
    *,
    debt: TargetRegionDebt,
    before_margin_stats: Mapping[str, float],
    after_margin_stats: Mapping[str, float],
    receiver_uint8_changed_pixels_region: int,
    receiver_uint8_delta_abs_max: float,
    receiver_float_rgb_delta_linf: float,
    argmax_flipped_pixels_region: int,
    accepted_step_count: int,
    rejected_step_count: int,
    blockers: Sequence[str],
    grad_norm_by_group: Mapping[str, float],
    update_norm_by_group: Mapping[str, float],
    updated_parameter_names: Sequence[str],
    pose_guard: Mapping[str, Any],
    runtime_sidecar_bytes: int = 0,
    argmax_transitions: Mapping[str, int] | None = None,
    exact_nonrate: Mapping[str, Any] | None = None,
    candidate_frontier_telemetry: Mapping[str, Any] | None = None,
    pose_compensation: Mapping[str, Any] | None = None,
    action_id: str | None = None,
    surface: str = "live_mlx",
) -> dict[str, Any]:
    """Assemble the actuation receipt with crux-trace-compatible keys.

    The ``receiver_surface_*`` aliases match the trace contract consumed by
    ``tools/trace_nerv_crux.py`` rows so accepted birth updates populate the
    receiver-surface evidence the witness-readiness DAG requires, without the
    consumer needing producer-specific adapters.

    ``pose_compensation`` carries the OPTIONAL frame0-only composite record.
    The frame0 RGB head (``head_rgb_0.*``) is a *compensation* scope, NOT a
    birth scope: SegNet reads frame1 only, so a frame0 update cannot move the
    seg term, and admitting it must never relax ``ALLOWED_BIRTH_UPDATE_*``.
    The compensated parameter names therefore travel under this mapping's
    ``compensation_updated_parameter_names`` key and are deliberately exempt
    from the ``updated_parameter_names`` birth-scope check below.
    """

    margin_p50_delta = float(after_margin_stats["margin_p50"] - before_margin_stats["margin_p50"])
    disallowed = [name for name in updated_parameter_names if not allowed_birth_update_name(name)]
    if disallowed:
        raise ValueError(f"updated parameter names escape the birth scope: {sorted(disallowed)}")
    pose_compensation_payload = dict(pose_compensation) if pose_compensation is not None else None
    if pose_compensation_payload is not None:
        compensation_names = [
            str(name) for name in pose_compensation_payload.get("compensation_updated_parameter_names") or []
        ]
        disallowed_compensation = [
            name
            for name in compensation_names
            if not allowed_pose_compensation_update_name(name) or allowed_birth_update_name(name)
        ]
        if disallowed_compensation:
            raise ValueError(
                "pose compensation parameter names escape the frame0 compensation scope: "
                f"{sorted(disallowed_compensation)}"
            )
    transition_counts = (
        {str(k): int(v) for k, v in argmax_transitions.items()} if argmax_transitions is not None else {}
    )
    receipt: dict[str, Any] = {
        "schema": TARGET_REGION_BIRTH_RECEIPT_SCHEMA,
        "actuator_id": "hinerv_target_region_birth",
        "action_id": action_id,
        "surface": str(surface),
        "family": "hinerv",
        "frame_scope": "frame1_seg_pose_joint",
        "pair_index": int(debt.batch_index),
        "worst_region": debt.as_dict(),
        "before_region_margin_stats": dict(before_margin_stats),
        "after_region_margin_stats": dict(after_margin_stats),
        "before_region_hard_ratio": float(before_margin_stats["region_hard_ratio"]),
        "after_region_hard_ratio": float(after_margin_stats["region_hard_ratio"]),
        "worst_region_margin_p50_delta": margin_p50_delta,
        "receiver_surface_worst_region_margin_p50_delta": margin_p50_delta,
        "receiver_surface_uint8_changed_pixels": int(receiver_uint8_changed_pixels_region),
        "receiver_surface_uint8_delta_abs_max": float(receiver_uint8_delta_abs_max),
        "receiver_surface_float_rgb_delta_linf": float(receiver_float_rgb_delta_linf),
        "receiver_surface_argmax_flipped_pixels": int(argmax_flipped_pixels_region),
        "accepted_step_count": int(accepted_step_count),
        "rejected_step_count": int(rejected_step_count),
        "blockers": [str(item) for item in blockers],
        "grad_norm_by_group": {str(name): float(value) for name, value in grad_norm_by_group.items()},
        "update_norm_by_group": {str(name): float(value) for name, value in update_norm_by_group.items()},
        "updated_parameter_names": sorted(str(name) for name in updated_parameter_names),
        "allowed_update_prefixes": list(ALLOWED_BIRTH_UPDATE_EXACT + ALLOWED_BIRTH_UPDATE_PREFIXES),
        "pose_guard": dict(pose_guard),
        # Receiver-motion vs target-birth disambiguation: the flips alias
        # above is churn telemetry; admission semantics live in transitions.
        "argmax_transitions": transition_counts or None,
        **transition_counts,
        # Exact nonlinear joint movement (batch-local authority) when a pose
        # teacher was available: 100*Δd_seg + (sqrt(10*d_pose') - sqrt(10*d_pose)).
        "exact_nonrate": dict(exact_nonrate) if exact_nonrate is not None else None,
        "candidate_frontier_telemetry": (
            dict(candidate_frontier_telemetry)
            if candidate_frontier_telemetry is not None
            else None
        ),
        # Frame0 composite compensation record (batch-local authority). When a
        # receiver-visible birth step held region progress but lost the pose
        # cap or the exact joint gate, a frame0-only (head_rgb_0.*) pose
        # compensation may have been attempted; SegNet reads frame1 only so the
        # seg term is structurally untouched by it. The composite (frame1 birth
        # + frame0 compensation) is admitted only when the exact nonrate score
        # strictly improves AND the pose cap is satisfied.
        "pose_compensation": pose_compensation_payload,
        "runtime_sidecar_bytes": int(runtime_sidecar_bytes),
        "human_visual_fidelity_objective": False,
        # Authority marker WITHOUT the canonical authority/readiness keys:
        # receipts travel inside substrate_artifact_metadata, where the
        # harness custody validator refuses score_claim/promotion_eligible/
        # etc. even as false-valued copies (single-custody-surface rule).
        "authority": "planning_control_false_authority",
    }
    return receipt
