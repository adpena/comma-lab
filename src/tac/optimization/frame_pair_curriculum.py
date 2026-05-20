# SPDX-License-Identifier: MIT
"""Score-aware frame/pair curriculum and masked correction knob planner.

The input is the per-frame master-gradient decomposition:
``frame_axis_l1[frame, axis(seg, pose, rate)]``.  This module converts that
array into two surfaces:

* SegNet-aligned frame weights: SegNet sees only the last frame of each
  non-overlapping seq_len=2 evaluator pair.
* PoseNet-aligned pair weights: PoseNet sees both frames in the pair.

It also emits a Photoshop-style adjustment-layer plan: masks plus small knobs
that can later become byte-neutral runtime rules, compact sidecars, or search
dimensions.  This is a planning/curriculum surface only; it makes no score or
promotion claim.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

Topology = Literal["non_overlapping", "sliding"]
SCHEMA = "ll_frame_pair_curriculum.v1"
AXES = ("seg", "pose", "rate")


class FramePairCurriculumError(ValueError):
    """Raised when curriculum inputs are malformed."""


@dataclass(frozen=True)
class CurriculumConfig:
    topology: Topology = "non_overlapping"
    top_k_frames: int = 16
    top_k_pairs: int = 8
    sampling_floor: float = 1e-12

    def validate(self) -> None:
        if self.topology not in {"non_overlapping", "sliding"}:
            raise FramePairCurriculumError(f"unsupported topology={self.topology!r}")
        if self.top_k_frames <= 0:
            raise FramePairCurriculumError("top_k_frames must be positive")
        if self.top_k_pairs <= 0:
            raise FramePairCurriculumError("top_k_pairs must be positive")
        if not math.isfinite(self.sampling_floor) or self.sampling_floor < 0.0:
            raise FramePairCurriculumError("sampling_floor must be finite and non-negative")


def load_frame_axis_npy(path: Path) -> Any:
    if np is None:
        raise RuntimeError("numpy required for frame/pair curriculum")
    return np.load(path, mmap_mode="r")


def build_frame_pair_curriculum(
    frame_axis_l1: Any,
    *,
    config: CurriculumConfig | None = None,
    decomposition_metadata: Mapping[str, Any] | None = None,
    response_plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build curriculum rows and masked correction knobs from frame L1 axes."""

    if np is None:
        raise RuntimeError("numpy required for frame/pair curriculum")
    cfg = config or CurriculumConfig()
    cfg.validate()
    arr = np.asarray(frame_axis_l1, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise FramePairCurriculumError(f"frame_axis_l1 must have shape (N_frames, 3); got {arr.shape}")
    if np.any(~np.isfinite(arr)) or np.any(arr < 0.0):
        raise FramePairCurriculumError("frame_axis_l1 must contain finite non-negative values")

    n_frames = int(arr.shape[0])
    if cfg.topology == "non_overlapping":
        if n_frames % 2 != 0:
            raise FramePairCurriculumError("non_overlapping topology requires an even frame count")
        n_pairs = n_frames // 2
    else:
        n_pairs = max(0, n_frames - 1)
    if n_pairs <= 0:
        raise FramePairCurriculumError("at least one evaluator pair is required")

    frame_total = arr.sum(axis=1)
    seg_prob = _normalize(arr[:, 0], floor=cfg.sampling_floor)
    pose_frame_prob = _normalize(arr[:, 1], floor=cfg.sampling_floor)
    total_frame_prob = _normalize(frame_total, floor=cfg.sampling_floor)
    frame_rows = _frame_rows(arr, seg_prob, pose_frame_prob, total_frame_prob, cfg.topology)
    pair_rows = _pair_rows(arr, cfg.topology, floor=cfg.sampling_floor)
    top_frames = sorted(frame_rows, key=lambda row: (-row["total_l1"], row["frame_index"]))[
        : min(cfg.top_k_frames, len(frame_rows))
    ]
    top_pairs = sorted(pair_rows, key=lambda row: (-row["total_l1"], row["pair_index"]))[
        : min(cfg.top_k_pairs, len(pair_rows))
    ]
    response_policy = _response_policy(response_plan)
    adjustment_layers = _adjustment_layers(top_frames, top_pairs, response_policy)
    return {
        "schema": SCHEMA,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "topology": cfg.topology,
        "n_frames": n_frames,
        "n_pairs": n_pairs,
        "axis_labels": list(AXES),
        "source_decomposition_schema": (
            None if decomposition_metadata is None else decomposition_metadata.get("schema")
        ),
        "scorer_input_topology": (
            {}
            if decomposition_metadata is None
            else dict(decomposition_metadata.get("scorer_input_topology") or {})
        ),
        "frame_axis_l1_sum": {
            axis: float(arr[:, idx].sum()) for idx, axis in enumerate(AXES)
        },
        "sampling_probability_checks": {
            "seg_frame_prob_sum": float(sum(row["seg_sampling_prob"] for row in frame_rows)),
            "pose_frame_prob_sum": float(sum(row["pose_frame_sampling_prob"] for row in frame_rows)),
            "total_frame_prob_sum": float(sum(row["total_sampling_prob"] for row in frame_rows)),
            "pair_pose_prob_sum": float(sum(row["pose_pair_sampling_prob"] for row in pair_rows)),
            "pair_total_prob_sum": float(sum(row["total_pair_sampling_prob"] for row in pair_rows)),
        },
        "response_policy": response_policy,
        "frame_rows": frame_rows,
        "pair_rows": pair_rows,
        "top_frames": top_frames,
        "top_pairs": top_pairs,
        "adjustment_layers": adjustment_layers,
    }


def _normalize(values: Any, *, floor: float) -> list[float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return []
    weighted = arr + float(floor)
    total = float(weighted.sum())
    if total <= 0.0:
        return [1.0 / float(arr.size)] * int(arr.size)
    return [float(value / total) for value in weighted]


def _pair_for_frame(frame_index: int, topology: Topology, n_frames: int) -> int | None:
    if topology == "non_overlapping":
        return frame_index // 2
    if frame_index == 0:
        return 0
    if frame_index == n_frames - 1:
        return frame_index - 1
    return None


def _frame_role(frame_index: int, topology: Topology, n_frames: int) -> str:
    if topology == "non_overlapping":
        return "pair_first_pose_only" if frame_index % 2 == 0 else "pair_last_segnet_and_pose"
    if frame_index == 0:
        return "sliding_first_pose_only"
    if frame_index == n_frames - 1:
        return "sliding_last_segnet_and_pose"
    return "sliding_middle_pose_twice_segnet_once"


def _frame_rows(
    arr: Any,
    seg_prob: list[float],
    pose_frame_prob: list[float],
    total_frame_prob: list[float],
    topology: Topology,
) -> list[dict[str, Any]]:
    n_frames = int(arr.shape[0])
    rows: list[dict[str, Any]] = []
    for frame_index in range(n_frames):
        seg = float(arr[frame_index, 0])
        pose = float(arr[frame_index, 1])
        rate = float(arr[frame_index, 2])
        total = seg + pose + rate
        rows.append(
            {
                "frame_index": frame_index,
                "pair_index": _pair_for_frame(frame_index, topology, n_frames),
                "frame_role": _frame_role(frame_index, topology, n_frames),
                "seg_l1": seg,
                "pose_l1": pose,
                "rate_l1": rate,
                "total_l1": total,
                "seg_sampling_prob": seg_prob[frame_index],
                "pose_frame_sampling_prob": pose_frame_prob[frame_index],
                "total_sampling_prob": total_frame_prob[frame_index],
                "recommended_training_targets": _frame_targets(seg, pose),
            }
        )
    return rows


def _pair_rows(arr: Any, topology: Topology, *, floor: float) -> list[dict[str, Any]]:
    n_frames = int(arr.shape[0])
    pairs = [(2 * idx, 2 * idx + 1) for idx in range(n_frames // 2)] if topology == "non_overlapping" else [
        (idx, idx + 1) for idx in range(max(0, n_frames - 1))
    ]
    pose_values = [float(arr[first, 1] + arr[last, 1]) for first, last in pairs]
    total_values: list[float] = []
    rows: list[dict[str, Any]] = []
    for pair_index, (first, last) in enumerate(pairs):
        seg = float(arr[last, 0])
        pose = float(arr[first, 1] + arr[last, 1])
        rate = float(arr[first, 2] + arr[last, 2])
        total_values.append(seg + pose + rate)
        rows.append(
            {
                "pair_index": pair_index,
                "first_frame": first,
                "last_frame": last,
                "seg_l1": seg,
                "pose_l1": pose,
                "rate_l1": rate,
                "total_l1": seg + pose + rate,
                "axis_mix": _axis_mix(seg=seg, pose=pose, rate=rate),
            }
        )
    pose_probs = _normalize(pose_values, floor=floor)
    total_probs = _normalize(total_values, floor=floor)
    for idx, row in enumerate(rows):
        row["pose_pair_sampling_prob"] = pose_probs[idx]
        row["total_pair_sampling_prob"] = total_probs[idx]
        row["recommended_correction_families"] = _pair_correction_families(row)
    return rows


def _axis_mix(*, seg: float, pose: float, rate: float) -> dict[str, float]:
    total = seg + pose + rate
    if total <= 0.0:
        return {"seg_share": 0.0, "pose_share": 0.0, "rate_share": 0.0}
    return {
        "seg_share": float(seg / total),
        "pose_share": float(pose / total),
        "rate_share": float(rate / total),
    }


def _frame_targets(seg: float, pose: float) -> list[str]:
    targets: list[str] = []
    if seg > 0.0:
        targets.append("segnet_frame_surrogate")
    if pose > 0.0:
        targets.append("posenet_pair_surrogate")
    if not targets:
        targets.append("low_priority_control")
    return targets


def _pair_correction_families(row: Mapping[str, Any]) -> list[str]:
    mix = row["axis_mix"]
    families: list[str] = []
    if float(mix["seg_share"]) >= 0.45:
        families.append("last_frame_boundary_mask_adjustment")
        families.append("last_frame_luma_chroma_knob")
    if float(mix["pose_share"]) >= 0.25:
        families.append("pair_global_motion_tone_knob")
        families.append("temporal_phase_or_blend_knob")
    if not families:
        families.append("rate_only_or_low_signal_control")
    return families


def _response_policy(response_plan: Mapping[str, Any] | None) -> dict[str, Any]:
    prohibitions = []
    if isinstance(response_plan, Mapping):
        raw = response_plan.get("prohibitions")
        if isinstance(raw, list):
            prohibitions = [item for item in raw if isinstance(item, Mapping)]
    blocked = any(
        str(item.get("rule")) == "do_not_widen_coordinate_sparse_residual_sidecar"
        for item in prohibitions
    )
    return {
        "score_claim": False,
        "charged_sparse_residual_widening_allowed": not blocked,
        "prohibitions": prohibitions,
        "default_knob_bias": (
            "prefer_byte_neutral_masked_runtime_knobs_before_charged_sparse_pixels"
            if blocked
            else "charged_sparse_pixels_allowed_after_break_even_gate"
        ),
    }


def _adjustment_layers(
    top_frames: list[Mapping[str, Any]],
    top_pairs: list[Mapping[str, Any]],
    response_policy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    layers: list[dict[str, Any]] = []
    for pair in top_pairs:
        pair_index = int(pair["pair_index"])
        mix = pair["axis_mix"]
        if float(mix["seg_share"]) >= 0.45:
            layers.append(
                _layer(
                    layer_id=f"pair_{pair_index:04d}_seg_boundary_last_frame",
                    target={"pair_index": pair_index, "frames": [int(pair["last_frame"])]},
                    primary_axis="seg",
                    mask_family="segnet_boundary_or_high_gradient_ring",
                    knobs=[
                        "luma_bias_i8",
                        "chroma_bias_i8x2",
                        "edge_sharpen_i4",
                        "argmax_margin_guard",
                    ],
                    rationale="SegNet sees the pair's last frame; use a local boundary/edge mask before raw residual pixels.",
                    response_policy=response_policy,
                )
            )
        if float(mix["pose_share"]) >= 0.25:
            layers.append(
                _layer(
                    layer_id=f"pair_{pair_index:04d}_pose_global_pair",
                    target={
                        "pair_index": pair_index,
                        "frames": [int(pair["first_frame"]), int(pair["last_frame"])],
                    },
                    primary_axis="pose",
                    mask_family="full_frame_low_frequency_or_horizon_weighted",
                    knobs=[
                        "temporal_blend_alpha_q4",
                        "global_luma_gain_q6",
                        "subpixel_shift_xy_small",
                        "chroma_rotation_q5",
                    ],
                    rationale="PoseNet consumes both frames; search coherent pair-level tone/phase knobs before per-pixel payload.",
                    response_policy=response_policy,
                )
            )
    high_seg_frames = [row for row in top_frames if float(row.get("seg_l1", 0.0)) > 0.0]
    if high_seg_frames:
        layers.append(
            _layer(
                layer_id="top_seg_frames_shared_adjustment_layer",
                target={"frames": [int(row["frame_index"]) for row in high_seg_frames[:8]]},
                primary_axis="seg",
                mask_family="shared_boundary_mask_palette",
                knobs=["per_frame_strength_q4", "class_boundary_dilate_erode_i2", "luma_bias_i8"],
                rationale="Shared masked layer for the hardest SegNet-visible frames; amortizes knobs across frames.",
                response_policy=response_policy,
            )
        )
    return _dedupe_layers(layers)


def _layer(
    *,
    layer_id: str,
    target: Mapping[str, Any],
    primary_axis: str,
    mask_family: str,
    knobs: list[str],
    rationale: str,
    response_policy: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "layer_id": layer_id,
        "target": dict(target),
        "primary_axis": primary_axis,
        "mask_family": mask_family,
        "knobs": knobs,
        "rationale": rationale,
        "byte_contract": {
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "preferred_first_materialization": response_policy.get("default_knob_bias"),
            "charged_sparse_residual_widening_allowed": bool(
                response_policy.get("charged_sparse_residual_widening_allowed")
            ),
        },
    }


def _dedupe_layers(layers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for layer in layers:
        layer_id = str(layer["layer_id"])
        if layer_id in seen:
            continue
        out.append(layer)
        seen.add(layer_id)
    return out


def render_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# LL Frame/Pair Curriculum and Masked Knob Plan",
        "",
        f"- Topology: `{payload['topology']}`",
        f"- Frames: {payload['n_frames']}",
        f"- Pairs: {payload['n_pairs']}",
        f"- Score claim: {payload['score_claim']}",
        f"- Knob bias: `{payload['response_policy']['default_knob_bias']}`",
        "",
        "## Top Frames",
        "",
    ]
    for row in payload["top_frames"]:
        lines.append(
            f"- frame `{row['frame_index']}` role=`{row['frame_role']}` "
            f"total={row['total_l1']:.6g} seg={row['seg_l1']:.6g} "
            f"pose={row['pose_l1']:.6g}"
        )
    lines.extend(["", "## Top Pairs", ""])
    for row in payload["top_pairs"]:
        lines.append(
            f"- pair `{row['pair_index']}` frames=({row['first_frame']},{row['last_frame']}) "
            f"total={row['total_l1']:.6g} seg_share={row['axis_mix']['seg_share']:.3f} "
            f"pose_share={row['axis_mix']['pose_share']:.3f}"
        )
    lines.extend(["", "## Adjustment Layers", ""])
    for layer in payload["adjustment_layers"]:
        lines.append(
            f"- `{layer['layer_id']}` axis=`{layer['primary_axis']}` "
            f"mask=`{layer['mask_family']}` knobs={', '.join(layer['knobs'])}"
        )
    lines.append("")
    return "\n".join(lines)
