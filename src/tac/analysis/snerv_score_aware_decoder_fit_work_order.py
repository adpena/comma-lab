# SPDX-License-Identifier: MIT
"""Build fail-closed SNeRV score-aware decoder-fit work orders.

SNeRV waterfill adjudication can leave an important signal in a JSON report:
rate is below the frontier, receiver replay is closed, but PoseNet/SegNet are
destroyed. That result should route to score-aware decoder fitting, not another
post-hoc packet tweak and not exact eval.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

SCHEMA = "snerv_score_aware_decoder_fit_work_order.v1"
EXPECTED_NEXT = "score_aware_stepmap_waterfill_and_decoder_fit_before_packaging"
TARGET_CLASSIFICATION = "rate_below_frontier_pose_or_seg_destroyed"
AXIS_TAG = "[macOS-CPU advisory]"


class SnervDecoderFitWorkOrderError(ValueError):
    """Raised when a SNeRV adjudication payload cannot produce a work order."""


@dataclass(frozen=True)
class SnervDecoderFitWorkOrder:
    """Machine-readable handoff from waterfill adjudication to decoder fitting."""

    schema: str
    source_path: str | None
    source_sha256: str | None
    axis_tag: str
    lane_id: str
    ready_for_local_decoder_fit_smoke: bool
    ready_for_exact_eval_dispatch: bool
    score_claim: bool
    promotion_eligible: bool
    selected_source_index: int | None
    selected_classification: str | None
    current_archive_bytes: int | None
    current_receiver_archive_sha256: str | None
    current_d_seg_linf: float | None
    current_d_pose_linf: float | None
    current_score_linf_advisory: float | None
    current_score_l2_advisory: float | None
    current_step_map_payload_bytes: int | None
    current_step_map_mode: str | None
    current_step_map_groups: tuple[dict[str, Any], ...]
    target_seg_ceiling: float
    target_pose_ceiling: float
    required_preconditions: tuple[str, ...]
    satisfied_preconditions: tuple[str, ...]
    blockers: tuple[str, ...]
    next_action: str
    recommended_smoke_commands: tuple[str, ...]
    escalation_after_smoke: tuple[str, ...]

    def as_jsonable(self) -> dict[str, Any]:
        return asdict(self)


def build_snerv_decoder_fit_work_order(
    adjudication_payload: dict[str, Any],
    *,
    source_path: str | None = None,
    source_sha256: str | None = None,
    lane_id: str = "lane_snerv_score_aware_decoder_fit_20260601",
) -> SnervDecoderFitWorkOrder:
    """Convert a SNeRV rate adjudication report into a local training work order."""

    if not isinstance(adjudication_payload, dict):
        raise SnervDecoderFitWorkOrderError("adjudication_payload must be a dict")
    rows = adjudication_payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise SnervDecoderFitWorkOrderError("adjudication payload has no rows")

    summary = adjudication_payload.get("summary")
    actionable_next = summary.get("actionable_next_code_move") if isinstance(summary, dict) else None
    target_rows = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("classification") == TARGET_CLASSIFICATION
        and row.get("receiver_archive_replay_verified") is True
        and row.get("step_map_accounting") == "charged_receiver_visible_payload"
    ]
    selected = min(
        target_rows,
        key=lambda row: (
            _optional_float(row.get("d_pose_linf")) is None,
            float("inf")
            if _optional_float(row.get("d_pose_linf")) is None
            else float(_optional_float(row.get("d_pose_linf"))),
            _optional_int(row.get("archive_bytes_charged")) or 10**18,
        ),
        default=None,
    )

    required = (
        "charged_receiver_visible_step_map_payload",
        "receiver_archive_replay_verified",
        "rate_below_frontier_but_distortion_destroyed",
        "waterfill_or_adaptive_step_map_groups_present",
        "adjudication_routes_to_decoder_fit",
    )
    satisfied: list[str] = []
    blockers: list[str] = []
    if selected is not None:
        satisfied.extend(required[:3])
        if selected.get("linf_steps_coder_groups"):
            satisfied.append("waterfill_or_adaptive_step_map_groups_present")
        else:
            blockers.append("step_map_group_metadata_missing")
        if actionable_next == EXPECTED_NEXT:
            satisfied.append("adjudication_routes_to_decoder_fit")
        else:
            blockers.append("adjudication_next_action_not_decoder_fit")
    else:
        blockers.append("no_replay_verified_low_rate_distortion_destroyed_row")
        if any(
            isinstance(row, dict)
            and row.get("step_map_accounting") == "missing_or_legacy_undercharged"
            for row in rows
        ):
            blockers.append("legacy_undercharged_step_map_payload")
        if any(
            isinstance(row, dict)
            and row.get("receiver_archive_replay_verified") is not True
            for row in rows
        ):
            blockers.append("receiver_archive_replay_missing")
        if actionable_next != EXPECTED_NEXT:
            blockers.append("adjudication_does_not_route_to_decoder_fit")

    ready = selected is not None and len(blockers) == 0
    levels = _optional_int(selected.get("levels")) if selected else None
    bits = _optional_float(selected.get("bits_per_coeff")) if selected else None
    if bits is None:
        bits = 5.0
    if levels is None:
        levels = 4
    smoke_commands = (
        (
            ".venv/bin/python tools/run_snerv_inverse_steg_advisory.py "
            f"--n-pairs 1 --levels {levels} --bits-per-coeff {bits:.3g} "
            "--step-map-coder-mode waterfill --step-map-waterfill-bits-per-coeff 6.0 "
            "--out .omx/research/snerv_score_aware_decoder_fit_after_smoke_<UTC>.json"
        ),
        (
            ".venv/bin/python tools/adjudicate_snerv_rate_sweep.py "
            ".omx/research/snerv_score_aware_decoder_fit_after_smoke_<UTC>.json "
            "--out .omx/research/snerv_score_aware_decoder_fit_after_adjudication_<UTC>.json"
        ),
    )
    next_action = (
        "run_local_score_aware_decoder_fit_smoke_with_waterfill_packet_in_loop"
        if ready
        else "repair_adjudication_or_receiver_replay_before_decoder_fit"
    )
    return SnervDecoderFitWorkOrder(
        schema=SCHEMA,
        source_path=source_path,
        source_sha256=source_sha256,
        axis_tag=AXIS_TAG,
        lane_id=lane_id,
        ready_for_local_decoder_fit_smoke=ready,
        ready_for_exact_eval_dispatch=False,
        score_claim=False,
        promotion_eligible=False,
        selected_source_index=_optional_int(selected.get("source_index")) if selected else None,
        selected_classification=selected.get("classification") if selected else None,
        current_archive_bytes=_optional_int(selected.get("archive_bytes_charged")) if selected else None,
        current_receiver_archive_sha256=(
            selected.get("receiver_archive_sha256") if selected else None
        ),
        current_d_seg_linf=_optional_float(selected.get("d_seg_linf")) if selected else None,
        current_d_pose_linf=_optional_float(selected.get("d_pose_linf")) if selected else None,
        current_score_linf_advisory=_optional_float(selected.get("score_linf_advisory"))
        if selected
        else None,
        current_score_l2_advisory=_optional_float(selected.get("score_l2_advisory"))
        if selected
        else None,
        current_step_map_payload_bytes=_optional_int(selected.get("linf_steps_payload_bytes"))
        if selected
        else None,
        current_step_map_mode=selected.get("linf_steps_coder_mode") if selected else None,
        current_step_map_groups=tuple(selected.get("linf_steps_coder_groups", ()))
        if selected
        else (),
        target_seg_ceiling=float(adjudication_payload.get("seg_preservation_ceiling", 0.02)),
        target_pose_ceiling=float(adjudication_payload.get("pose_preservation_ceiling", 0.10)),
        required_preconditions=required,
        satisfied_preconditions=tuple(satisfied),
        blockers=tuple(dict.fromkeys(blockers)),
        next_action=next_action,
        recommended_smoke_commands=smoke_commands if ready else (),
        escalation_after_smoke=(
            "if_pose_and_seg_within_ceilings_build_full600_receiver_replay_packet",
            "if_rate_grows_above_frontier_tune_waterfill_budget_before_packaging",
            "if_distortion_remains_destroyed_move_allocator_into_decoder_weight_training",
        ),
    )


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
