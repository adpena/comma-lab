# SPDX-License-Identifier: MIT
"""Pose-guarded SNeRV decoder-fit gate.

The closed-form scalar/component HF weighting sweeps showed an easy trap:
SegNet can improve while PoseNet gets much worse. This module makes the next
decoder-fit criterion executable. A candidate may only advance beyond local
advisory work if it improves SegNet/score while keeping PoseNet at or below the
least-squares waterfill control.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from tac.analysis.snerv_rate_adjudication import iter_snerv_candidate_rows

SCHEMA = "snerv_pose_guarded_decoder_gate.v1"
AXIS_TAG = "[macOS-CPU advisory]"
DEFAULT_BYTE_SLACK = 2048
DEFAULT_POSE_SLACK = 0.0
DEFAULT_SEG_CEILING = 0.02


class SnervPoseGuardedDecoderGateError(ValueError):
    """Raised when a SNeRV pose-guard payload cannot be built safely."""


@dataclass(frozen=True)
class SnervPoseGuardedRow:
    """One candidate row evaluated against the least-squares control."""

    source_index: int
    label: str
    source_artifact: str | None
    hf_decoder_fit_mode: str | None
    hf_decoder_saliency_component: str | None
    hf_decoder_saliency_gain: float | None
    archive_bytes: int
    receiver_archive_sha256: str | None
    receiver_archive_replay_verified: bool
    d_seg_linf: float | None
    d_pose_linf: float | None
    score_linf: float | None
    pose_delta_vs_control: float | None
    seg_delta_vs_control: float | None
    score_delta_vs_control: float | None
    passes_pose_guard: bool
    passes_seg_gate: bool
    passes_score_gate: bool
    passes_rate_gate: bool
    source_row_accepted: bool | None
    source_row_blockers: tuple[str, ...]
    accepted_for_local_continuation: bool
    blockers: tuple[str, ...]
    axis_tag: str = AXIS_TAG
    score_claim: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False

    def as_jsonable(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SnervPoseGuardedDecoderGate:
    """Machine-readable go/no-go gate for the next SNeRV decoder-fit family."""

    schema: str
    axis_tag: str
    source_paths: tuple[str, ...]
    baseline_label: str
    baseline_archive_bytes: int
    baseline_d_seg_linf: float
    baseline_d_pose_linf: float
    baseline_score_linf: float
    max_archive_bytes: int
    pose_slack: float
    seg_ceiling: float
    rows: tuple[SnervPoseGuardedRow, ...]
    accepted_rows: tuple[SnervPoseGuardedRow, ...]
    closed_form_scalar_weighting_no_go: bool
    verdict: str
    next_action: str
    blockers: tuple[str, ...]
    recommended_next_implementation: tuple[str, ...]
    score_claim: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False

    def as_jsonable(self) -> dict[str, Any]:
        d = asdict(self)
        d["rows"] = [row.as_jsonable() for row in self.rows]
        d["accepted_rows"] = [row.as_jsonable() for row in self.accepted_rows]
        return d


def build_snerv_pose_guarded_decoder_gate(
    payloads: list[dict[str, Any]],
    *,
    source_paths: tuple[str, ...] = (),
    byte_slack: int = DEFAULT_BYTE_SLACK,
    pose_slack: float = DEFAULT_POSE_SLACK,
    seg_ceiling: float = DEFAULT_SEG_CEILING,
) -> SnervPoseGuardedDecoderGate:
    """Build a false-authority decoder-fit gate from SNeRV advisory payloads."""

    if not payloads:
        raise SnervPoseGuardedDecoderGateError("at least one payload is required")
    if byte_slack < 0:
        raise SnervPoseGuardedDecoderGateError("byte_slack must be non-negative")
    if pose_slack < 0:
        raise SnervPoseGuardedDecoderGateError("pose_slack must be non-negative")
    if seg_ceiling <= 0:
        raise SnervPoseGuardedDecoderGateError("seg_ceiling must be positive")

    raw_rows: list[dict[str, Any]] = []
    for payload in payloads:
        raw_rows.extend(iter_snerv_candidate_rows(payload))
    if not raw_rows:
        raise SnervPoseGuardedDecoderGateError("no row-like SNeRV payloads found")

    baseline = _select_baseline(raw_rows)
    baseline_label = _row_label(baseline)
    baseline_archive = _required_int(
        baseline.get("archive_bytes_total", baseline.get("archive_bytes")),
        "baseline archive bytes",
    )
    baseline_seg = _required_float(
        baseline.get("d_seg_mean_linf", baseline.get("d_seg_linf")),
        "baseline d_seg",
    )
    baseline_pose = _required_float(
        baseline.get("d_pose_mean_linf", baseline.get("d_pose_linf")),
        "baseline d_pose",
    )
    baseline_score = _required_float(baseline.get("score_linf"), "baseline score")
    max_bytes = baseline_archive + int(byte_slack)

    evaluated = tuple(
        _evaluate_row(
            row,
            source_index=idx,
            baseline_seg=baseline_seg,
            baseline_pose=baseline_pose,
            baseline_score=baseline_score,
            max_bytes=max_bytes,
            pose_slack=pose_slack,
            seg_ceiling=seg_ceiling,
        )
        for idx, row in enumerate(raw_rows)
        if row is not baseline
    )
    accepted = tuple(row for row in evaluated if row.accepted_for_local_continuation)
    scalar_rows = [
        row
        for row in evaluated
        if row.hf_decoder_fit_mode == "score_weighted"
        or row.hf_decoder_saliency_component is not None
    ]
    closed_form_no_go = bool(scalar_rows) and not any(
        row.accepted_for_local_continuation for row in scalar_rows
    )
    blockers = []
    if not accepted:
        blockers.append("no_candidate_passes_pose_guarded_local_continuation_gate")
    if closed_form_no_go:
        blockers.append("closed_form_scalar_component_weighting_no_go")

    if accepted:
        verdict = "GO_LOCAL_CONTINUATION_ONLY"
        next_action = "run_bounded_non_exact_receiver_replay_on_best_local_candidate"
    else:
        verdict = "NO_GO_FOR_PROMOTION_OR_EXACT_EVAL"
        next_action = "implement_scorer_loop_or_nonlinear_qat_decoder_before_more_sweeps"

    return SnervPoseGuardedDecoderGate(
        schema=SCHEMA,
        axis_tag=AXIS_TAG,
        source_paths=tuple(source_paths),
        baseline_label=baseline_label,
        baseline_archive_bytes=baseline_archive,
        baseline_d_seg_linf=baseline_seg,
        baseline_d_pose_linf=baseline_pose,
        baseline_score_linf=baseline_score,
        max_archive_bytes=max_bytes,
        pose_slack=float(pose_slack),
        seg_ceiling=float(seg_ceiling),
        rows=evaluated,
        accepted_rows=accepted,
        closed_form_scalar_weighting_no_go=closed_form_no_go,
        verdict=verdict,
        next_action=next_action,
        blockers=tuple(blockers),
        recommended_next_implementation=(
            "train_decoder_weights_on_reconstructed_frame_scorer_loop",
            "try_nonlinear_or_learned_hf_decoder_qat_with_same_snar1_receiver_contract",
            "treat_posenet_d_pose_linf_as_hard_constraint_before_segnet_gain",
            "keep_least_squares_waterfill_as_control_until_candidate_passes_gate",
        ),
    )


def _evaluate_row(
    row: dict[str, Any],
    *,
    source_index: int,
    baseline_seg: float,
    baseline_pose: float,
    baseline_score: float,
    max_bytes: int,
    pose_slack: float,
    seg_ceiling: float,
) -> SnervPoseGuardedRow:
    archive = _required_int(row.get("archive_bytes_total", row.get("archive_bytes")), "archive bytes")
    d_seg = _optional_float(row.get("d_seg_mean_linf", row.get("d_seg_linf")))
    d_pose = _optional_float(row.get("d_pose_mean_linf", row.get("d_pose_linf")))
    score = _optional_float(row.get("score_linf"))
    replay = row.get("receiver_archive_replay_verified") is True
    pose_delta = None if d_pose is None else d_pose - baseline_pose
    seg_delta = None if d_seg is None else d_seg - baseline_seg
    score_delta = None if score is None else score - baseline_score
    passes_pose = d_pose is not None and d_pose <= baseline_pose + pose_slack
    passes_seg = d_seg is not None and d_seg <= seg_ceiling and d_seg < baseline_seg
    passes_score = score is not None and score < baseline_score
    passes_rate = archive <= max_bytes
    source_accepted = row.get("accepted")
    if not isinstance(source_accepted, bool):
        source_accepted = None
    source_artifact = _optional_str(row.get("source_artifact"))
    honor_source_contract = bool(
        source_accepted is not None
        or source_artifact == "snerv_scorer_loop_decoder_qat_smoke"
    )
    source_blockers = (
        _optional_str_tuple(row.get("blockers")) if honor_source_contract else ()
    )
    blockers = []
    if not replay:
        blockers.append("receiver_archive_replay_missing")
    if not passes_pose:
        blockers.append("pose_guard_failed")
    if not passes_seg:
        blockers.append("seg_gate_failed")
    if not passes_score:
        blockers.append("score_gate_failed")
    if not passes_rate:
        blockers.append("rate_gate_failed")
    if honor_source_contract and source_accepted is False:
        blockers.append("source_scorer_loop_rejected")
    blockers.extend(f"source:{blocker}" for blocker in source_blockers)
    accepted = (
        replay
        and passes_pose
        and passes_seg
        and passes_score
        and passes_rate
        and (not honor_source_contract or source_accepted is not False)
        and not source_blockers
    )
    return SnervPoseGuardedRow(
        source_index=source_index,
        label=_row_label(row),
        source_artifact=source_artifact,
        hf_decoder_fit_mode=_optional_str(row.get("hf_decoder_fit_mode")),
        hf_decoder_saliency_component=_optional_str(row.get("hf_decoder_saliency_component")),
        hf_decoder_saliency_gain=_optional_float(row.get("hf_decoder_saliency_gain")),
        archive_bytes=archive,
        receiver_archive_sha256=_optional_str(row.get("receiver_archive_sha256")),
        receiver_archive_replay_verified=replay,
        d_seg_linf=d_seg,
        d_pose_linf=d_pose,
        score_linf=score,
        pose_delta_vs_control=pose_delta,
        seg_delta_vs_control=seg_delta,
        score_delta_vs_control=score_delta,
        passes_pose_guard=passes_pose,
        passes_seg_gate=passes_seg,
        passes_score_gate=passes_score,
        passes_rate_gate=passes_rate,
        source_row_accepted=source_accepted,
        source_row_blockers=source_blockers,
        accepted_for_local_continuation=accepted,
        blockers=tuple(dict.fromkeys(blockers)),
    )


def _select_baseline(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [
        row
        for row in rows
        if row.get("sweep_label") == "least_squares_baseline_existing"
        or row.get("hf_decoder_fit_mode") in {None, "least_squares"}
    ]
    if not candidates:
        raise SnervPoseGuardedDecoderGateError("least-squares baseline row missing")
    return min(
        candidates,
        key=lambda row: (
            _optional_float(row.get("score_linf")) is None,
            float("inf")
            if _optional_float(row.get("score_linf")) is None
            else float(_optional_float(row.get("score_linf"))),
            _optional_int(row.get("archive_bytes_total", row.get("archive_bytes"))) or 10**18,
        ),
    )


def _row_label(row: dict[str, Any]) -> str:
    label = _optional_str(row.get("sweep_label"))
    if label:
        return label
    mode = _optional_str(row.get("hf_decoder_fit_mode")) or "least_squares"
    component = _optional_str(row.get("hf_decoder_saliency_component"))
    gain = _optional_float(row.get("hf_decoder_saliency_gain"))
    parts = [mode]
    if component:
        parts.append(component)
    if gain is not None:
        parts.append(f"gain_{gain:g}")
    return "_".join(parts)


def _required_int(value: Any, name: str) -> int:
    out = _optional_int(value)
    if out is None:
        raise SnervPoseGuardedDecoderGateError(f"missing integer: {name}")
    return out


def _required_float(value: Any, name: str) -> float:
    out = _optional_float(value)
    if out is None:
        raise SnervPoseGuardedDecoderGateError(f"missing float: {name}")
    return out


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


def _optional_str(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return value


def _optional_str_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    out = []
    for item in value:
        if isinstance(item, str) and item:
            out.append(item)
    return tuple(out)
