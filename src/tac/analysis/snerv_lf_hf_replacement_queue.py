# SPDX-License-Identifier: MIT
"""Queue-owned SNeRV LF/HF learned replacement planning.

This module consumes measured LF payload byte reports plus the current SNeRV LF
reroute/campaign handoff surfaces.  It deliberately stays false-authority: the
rows are local prototype work orders or explicit blockers, never score claims.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tac.substrates.hprc.archive_candidate import FALSE_AUTHORITY

SCHEMA = "snerv_lf_hf_replacement_queue.v1"
ROW_SCHEMA = "snerv_lf_hf_replacement_candidate_row.v1"
DEFAULT_LANE_ID = "lane_snerv_lf_hf_replacement_queue_20260605"
DEFAULT_QUEUE_ID = "snerv_lf_hf_replacement_queue.v1"
AXIS_TAG = "[planning/control:false-authority]"
DEFAULT_MIN_FREE_BYTES = 1_000_000_000
SSD_ROOTS = (
    Path("/Volumes/VertigoDataTier/pact"),
    Path("/Volumes/APDataStore/pact"),
)

QUEUE_FALSE_AUTHORITY = {
    **FALSE_AUTHORITY,
    "score_claim": False,
    "score_claim_valid": False,
    "frontier_score_claim": False,
    "promotion_eligible": False,
    "promotable": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "production_hardened_claim": False,
    "dispatch_attempted": False,
    "gpu_launched": False,
    "local_mlx_long_training_allowed": False,
    "dispatch_allowed": False,
    "exact_or_full_video_cuda_allowed": False,
}

_SOURCE_FORWARD_BLOCKERS = (
    "snerv_official_mfu_hfr_tub_export_not_bound",
    "snerv_official_mfu_hfr_tub_receiver_payload_not_bound",
    "snerv_official_mfu_hfr_tub_frame_producing_export_missing",
    "snerv_official_mfu_hfr_tub_receiver_payload_not_source_forward_authority",
    "snerv_official_mfu_hfr_tub_full_stack_source_forward_replay_missing",
)
_SOURCE_FORWARD_FRAME_REPLAY_CLOSED_BLOCKERS = (
    "snerv_official_mfu_hfr_tub_receiver_payload_not_bound",
    "snerv_official_mfu_hfr_tub_frame_producing_export_missing",
)
_SOURCE_FORWARD_QUEUE_FAMILIES = (
    "official_tub_lf_hf_decoder_replacement",
    "temporal_lf_predictor_gate",
)
_RENDERER_BLOCKERS = (
    "snerv_renderer_nondegenerate_smoke_missing",
    "snerv_renderer_nondegenerate_smoke_failed",
    "snerv_renderer_nondegenerate_smoke_min16_pairs_missing",
    "snerv_renderer_nondegenerate_export_value_domain_not_passed",
    "snerv_renderer_nondegenerate_receiver_reconstruction_not_verified",
    "snerv_scorer_input_distribution_guard_missing",
)
_SCORER_DOMAIN_CLOSED_BLOCKERS = (
    "snerv_scorer_input_distribution_guard_missing",
)
_SCORER_DOMAIN_REQUIRED_METRICS = (
    "snerv_posenet_yuv6_pair_distill",
    "snerv_segnet_last_frame_distill",
)
_SKIP_HIGH_BLOCKERS = (
    "snerv_official_skip_high_scalar_mean_requires_value_domain_xray_noncollapse",
    "snerv_renderer_nondegenerate_compact_skip_high_value_domain_not_passed",
    "snerv_renderer_nondegenerate_target_value_domain_not_passed",
)


class SnervLfHfReplacementQueueError(ValueError):
    """Raised when the LF/HF replacement queue cannot be built."""


def build_snerv_lf_hf_replacement_queue(
    *,
    lf_payload_reports: Sequence[Mapping[str, Any]] = (),
    reroute_queues: Sequence[Mapping[str, Any]] = (),
    campaign_plans: Sequence[Mapping[str, Any]] = (),
    source_forward_artifacts: Sequence[Mapping[str, Any]] = (),
    candidate_feedback_rows: Sequence[Mapping[str, Any]] = (),
    output_root: str | Path,
    lane_id: str = DEFAULT_LANE_ID,
    queue_id: str = DEFAULT_QUEUE_ID,
    generated_utc: str | None = None,
    min_free_bytes: int = DEFAULT_MIN_FREE_BYTES,
    allow_local_output: bool = False,
) -> dict[str, Any]:
    """Build a queue for learned SNeRV LF/HF replacement candidates.

    The builder is intentionally conservative.  Older LF byte reports remain
    useful as acquisition signal, but if the freshest SNAR2-era queue has no LF
    over-ceiling rows, the output records that as a blocker against re-enabling
    long training from LF dominance alone.
    """

    if not str(lane_id).strip():
        raise SnervLfHfReplacementQueueError("lane_id must be non-empty")
    if not str(queue_id).strip():
        raise SnervLfHfReplacementQueueError("queue_id must be non-empty")
    generated = generated_utc or datetime.now(UTC).isoformat()
    root = Path(output_root)
    storage_preflight = _storage_preflight(
        root,
        min_free_bytes=int(min_free_bytes),
        allow_local_output=bool(allow_local_output),
    )
    evidence_rows = [_lf_evidence_row(report, idx) for idx, report in enumerate(lf_payload_reports)]
    evidence_rows = [row for row in evidence_rows if row is not None]
    campaign_rows = _snerv_campaign_rows(campaign_plans)
    reroute_state = _reroute_state(reroute_queues)
    source_forward_state = _source_forward_state(source_forward_artifacts)
    scorer_domain_state = _scorer_domain_state(candidate_feedback_rows)
    current_state = _current_state(
        campaign_rows=campaign_rows,
        reroute_state=reroute_state,
        evidence_rows=evidence_rows,
        source_forward_state=source_forward_state,
        scorer_domain_state=scorer_domain_state,
    )
    selected_evidence = _selected_lf_evidence(evidence_rows)
    rows = _candidate_rows(
        campaign_rows=campaign_rows,
        selected_evidence=selected_evidence,
        current_state=current_state,
        output_root=root,
    )
    if not rows:
        rows = [
            _global_blocker_row(
                output_root=root,
                blocker="snerv_lf_hf_replacement_no_snerv_campaign_rows",
                selected_evidence=selected_evidence,
            )
        ]
    blocked_rows = [row for row in rows if row["blocked"]]
    executable_rows = [row for row in rows if row["command_argv"] and not row["blocked"]]
    blockers = _dedupe(
        [
            "snerv_lf_hf_replacement_queue_false_authority",
            *current_state.get("blockers", ()),
            *[blocker for row in rows for blocker in row.get("blockers", ())],
        ]
    )
    return {
        "schema": SCHEMA,
        "queue_id": str(queue_id),
        "lane_id": str(lane_id),
        "generated_utc": generated,
        "axis_tag": AXIS_TAG,
        "queue_kind": "planner_queue_not_training_queue",
        "allowed_use": (
            "local bounded LF/HF replacement prototype selection and blocker "
            "routing before any long training or exact eval"
        ),
        "forbidden_use": (
            "score claim, promotion, rank/kill decision, exact eval dispatch, "
            "or long-training re-enable without row blockers clearing"
        ),
        "storage_preflight": storage_preflight,
        "current_state": current_state,
        "source_forward_evidence": source_forward_state,
        "scorer_domain_evidence": scorer_domain_state,
        "lf_payload_evidence_rows": evidence_rows,
        "lf_payload_evidence_row_count": len(evidence_rows),
        "selected_lf_payload_evidence": selected_evidence,
        "queue_rows": rows,
        "queue_row_count": len(rows),
        "blocked_queue_row_count": len(blocked_rows),
        "local_executable_command_row_count": len(executable_rows),
        "learned_replacement_candidate_row_count": sum(
            1 for row in rows if row["candidate_class"] == "learned_lf_hf_replacement"
        ),
        "blocking_queue_row_ids": [row["queue_row_id"] for row in blocked_rows],
        "runnable_queue_row_ids": [row["queue_row_id"] for row in executable_rows],
        "blockers": blockers,
        **QUEUE_FALSE_AUTHORITY,
    }


def render_snerv_lf_hf_replacement_queue_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact handoff for operator review."""

    current = report.get("current_state", {}) if isinstance(report, Mapping) else {}
    selected = report.get("selected_lf_payload_evidence") if isinstance(report, Mapping) else None
    selected = selected if isinstance(selected, Mapping) else {}
    lines = [
        "# SNeRV LF/HF Replacement Queue",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- lane: `{report.get('lane_id')}`",
        f"- axis: `{report.get('axis_tag')}`",
        f"- queue rows: `{report.get('queue_row_count')}`",
        f"- runnable local rows: `{report.get('local_executable_command_row_count')}`",
        f"- current reroute rows: `{current.get('freshest_reroute_queue_row_count')}`",
        f"- current SNAR2 no-LF-overrun: `{current.get('freshest_queue_has_no_lf_over_ceiling_rows')}`",
        f"- LF dominance launch signal active: `{current.get('lf_dominance_launch_signal_active')}`",
        "- receiver payload frame replay proven: "
        f"`{_nested(current, ('source_forward_evidence', 'receiver_payload_frame_replay_proven'))}`",
        "- scorer domain tether proof passed: "
        f"`{_nested(current, ('scorer_domain_evidence', 'scorer_domain_tether_proof_passed'))}`",
        f"- selected LF evidence bytes: `{selected.get('lf_payload_bytes')}`",
        "",
        "## Candidate Rows",
    ]
    for row in report.get("queue_rows", []) if isinstance(report, Mapping) else []:
        if not isinstance(row, Mapping):
            continue
        lines.extend(
            [
                "",
                f"### `{row.get('queue_row_id')}`",
                f"- family: `{row.get('solution_family')}`",
                f"- action: `{row.get('planner_action')}`",
                f"- blocked: `{row.get('blocked')}`",
                f"- command: `{_shell_join(row.get('command_argv') or [])}`",
                "- blockers:",
            ]
        )
        blockers = [str(v) for v in row.get("blockers") or ()]
        lines.extend(f"  - `{blocker}`" for blocker in blockers)
    return "\n".join(lines) + "\n"


def _candidate_rows(
    *,
    campaign_rows: Sequence[Mapping[str, Any]],
    selected_evidence: Mapping[str, Any] | None,
    current_state: Mapping[str, Any],
    output_root: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source_rows = list(campaign_rows)
    if not source_rows:
        return rows
    for row in source_rows[:4]:
        rows.extend(
            [
                _candidate_row(
                    campaign_row=row,
                    selected_evidence=selected_evidence,
                    current_state=current_state,
                    output_root=output_root,
                    solution_family="official_tub_lf_hf_decoder_replacement",
                    planner_action="run_bounded_source_faithful_lf_hf_decoder_smoke",
                    learning_objective=(
                        "learn receiver-visible official MFU/HFR/TUB decoder "
                        "that reconstructs LF and generates HF under scorer "
                        "tether, replacing stored full LF grids only after "
                        "source-forward replay passes"
                    ),
                    static_blockers=(),
                    campaign_blocker_prefixes=_SOURCE_FORWARD_BLOCKERS + _RENDERER_BLOCKERS,
                    command_kind="bounded_snerv_training_smoke",
                    priority=10,
                ),
                _candidate_row(
                    campaign_row=row,
                    selected_evidence=selected_evidence,
                    current_state=current_state,
                    output_root=output_root,
                    solution_family="lf_conditioned_hf_residual_generator",
                    planner_action="probe_non_scalar_hf_generation_without_skip_high_collapse",
                    learning_objective=(
                        "keep a small LF carrier and learn HF residuals only "
                        "when SegNet/PoseNet price them; reject scalar/channel "
                        "mean skip-high collapse before replay"
                    ),
                    static_blockers=(
                        "snerv_hf_residual_generator_receiver_payload_not_implemented",
                    ),
                    campaign_blocker_prefixes=_SKIP_HIGH_BLOCKERS + _RENDERER_BLOCKERS,
                    command_kind="blocked_until_hf_payload_exists",
                    priority=20,
                ),
                _candidate_row(
                    campaign_row=row,
                    selected_evidence=selected_evidence,
                    current_state=current_state,
                    output_root=output_root,
                    solution_family="joint_lf_hf_factorized_codebook",
                    planner_action="build_score_tethered_joint_lf_hf_codebook_export",
                    learning_objective=(
                        "learn a byte-charged LF/HF factorized codebook with "
                        "NumPy receiver decode and section telemetry before "
                        "any full run"
                    ),
                    static_blockers=(
                        "snerv_joint_lf_hf_factorized_codebook_not_implemented",
                        "snerv_joint_lf_hf_codebook_numpy_receiver_missing",
                        "snerv_joint_lf_hf_codebook_section_byte_telemetry_missing",
                    ),
                    campaign_blocker_prefixes=_RENDERER_BLOCKERS,
                    command_kind="blocked_until_codebook_export_exists",
                    priority=30,
                ),
                _candidate_row(
                    campaign_row=row,
                    selected_evidence=selected_evidence,
                    current_state=current_state,
                    output_root=output_root,
                    solution_family="temporal_lf_predictor_gate",
                    planner_action="learn_temporal_lf_delta_predictor_with_receiver_gate",
                    learning_objective=(
                        "predict LF planes from temporal context and store only "
                        "a byte-charged correction stream when the official "
                        "source-forward path proves the predictor is consumed"
                    ),
                    static_blockers=(
                        "snerv_temporal_lf_predictor_gate_not_implemented",
                        "snerv_temporal_lf_predictor_correction_stream_not_byte_charged",
                    ),
                    campaign_blocker_prefixes=_SOURCE_FORWARD_BLOCKERS + _RENDERER_BLOCKERS,
                    command_kind="blocked_until_temporal_lf_gate_exists",
                    priority=40,
                ),
                _candidate_row(
                    campaign_row=row,
                    selected_evidence=selected_evidence,
                    current_state=current_state,
                    output_root=output_root,
                    solution_family="lf_super_resolution_from_tiny_anchor",
                    planner_action="store_tiny_lf_anchor_then_learn_receiver_super_resolution",
                    learning_objective=(
                        "store a deliberately tiny LF anchor and learn a NumPy "
                        "receiver super-resolution decoder whose HF errors are "
                        "priced by SegNet/PoseNet component telemetry"
                    ),
                    static_blockers=(
                        "snerv_lf_super_resolution_receiver_payload_not_implemented",
                        "snerv_lf_downsampled_anchor_component_deltas_missing",
                    ),
                    campaign_blocker_prefixes=_RENDERER_BLOCKERS,
                    command_kind="blocked_until_lf_super_resolution_export_exists",
                    priority=50,
                ),
                _candidate_row(
                    campaign_row=row,
                    selected_evidence=selected_evidence,
                    current_state=current_state,
                    output_root=output_root,
                    solution_family="score_tethered_spectral_band_allocator",
                    planner_action="learn_mfu_hfr_lf_hf_band_budget_from_scorer_telemetry",
                    learning_objective=(
                        "learn the LF/HF band split under scorer telemetry so "
                        "MFU/HFR controls actuate section bytes instead of only "
                        "nominal model-size tokens"
                    ),
                    static_blockers=(
                        "snerv_score_tethered_lf_hf_band_allocator_not_implemented",
                        "snerv_mfu_hfr_section_native_byte_telemetry_missing",
                    ),
                    campaign_blocker_prefixes=_RENDERER_BLOCKERS,
                    command_kind="blocked_until_band_allocator_export_exists",
                    priority=60,
                ),
                _candidate_row(
                    campaign_row=row,
                    selected_evidence=selected_evidence,
                    current_state=current_state,
                    output_root=output_root,
                    solution_family="entropy_modeled_lf_latent_hyperprior",
                    planner_action="replace_i64_lzma_lf_planes_with_learned_entropy_model",
                    learning_objective=(
                        "replace generic int64+LZMA LF storage with a learned "
                        "latent entropy model and deterministic NumPy decode, "
                        "then require receiver proof and component replay"
                    ),
                    static_blockers=(
                        "snerv_lf_latent_hyperprior_not_implemented",
                        "snerv_lf_latent_hyperprior_numpy_decoder_missing",
                        "snerv_lf_latent_hyperprior_receiver_replay_missing",
                    ),
                    campaign_blocker_prefixes=_RENDERER_BLOCKERS,
                    command_kind="blocked_until_lf_hyperprior_export_exists",
                    priority=70,
                ),
            ]
        )
    rows.sort(
        key=lambda row: (
            int(row.get("priority") or 999),
            str(row.get("source_campaign_row_id") or ""),
            str(row.get("solution_family") or ""),
        )
    )
    return rows


def _candidate_row(
    *,
    campaign_row: Mapping[str, Any],
    selected_evidence: Mapping[str, Any] | None,
    current_state: Mapping[str, Any],
    output_root: Path,
    solution_family: str,
    planner_action: str,
    learning_objective: str,
    static_blockers: Sequence[str],
    campaign_blocker_prefixes: Sequence[str],
    command_kind: str,
    priority: int,
) -> dict[str, Any]:
    source_row_id = str(campaign_row.get("row_id") or campaign_row.get("candidate_id") or "snerv")
    candidate_id = str(campaign_row.get("candidate_id") or "candidate")
    token = _stable_safe_token(f"{source_row_id}_{solution_family}")
    queue_row_id = f"snerv_lf_hf_replace_{token}"
    evidence_blockers: list[str] = []
    if selected_evidence is None:
        evidence_blockers.append("snerv_lf_hf_measured_lf_payload_report_missing")
    elif _positive_int(selected_evidence.get("lf_payload_bytes")) is None:
        evidence_blockers.append("snerv_lf_hf_selected_lf_payload_bytes_missing")
    current_blockers = [
        str(blocker)
        for blocker in current_state.get("blockers", ())
        if str(blocker)
    ]
    campaign_blockers = _campaign_blockers(campaign_row, campaign_blocker_prefixes)
    source_forward_closed = set(
        _nested(current_state, ("source_forward_evidence", "closed_campaign_blockers"))
        or ()
    )
    scorer_domain_closed = set(
        _nested(current_state, ("scorer_domain_evidence", "closed_campaign_blockers"))
        or ()
    )
    source_forward_extra_blockers: list[str] = []
    if solution_family in _SOURCE_FORWARD_QUEUE_FAMILIES:
        source_forward_extra_blockers = [
            str(blocker)
            for blocker in (
                _nested(current_state, ("source_forward_evidence", "queue_blockers"))
                or ()
            )
            if blocker
        ]
    campaign_blockers = [
        blocker
        for blocker in campaign_blockers
        if blocker not in source_forward_closed and blocker not in scorer_domain_closed
    ]
    blockers = _dedupe(
        [
            *static_blockers,
            *evidence_blockers,
            *current_blockers,
            *campaign_blockers,
            *source_forward_extra_blockers,
        ]
    )
    command: list[str] = []
    if (
        command_kind == "bounded_snerv_training_smoke"
        and not blockers
        and campaign_row.get("local_mlx_launch_command_ready") is True
    ):
        command = _bounded_snerv_smoke_command(
            campaign_row,
            queue_row_id=queue_row_id,
            output_root=output_root,
        )
        if not command:
            blockers = _dedupe([*blockers, "snerv_lf_hf_base_snerv_command_missing"])
    status = "local_bounded_smoke_ready_no_authority" if command and not blockers else "blocked_until_prerequisite_evidence"
    return {
        "schema": ROW_SCHEMA,
        "queue_row_id": queue_row_id,
        "lane_id": DEFAULT_LANE_ID,
        "source_campaign_row_id": source_row_id,
        "candidate_id": candidate_id,
        "candidate_class": "learned_lf_hf_replacement",
        "solution_family": solution_family,
        "planner_action": planner_action,
        "learning_objective": learning_objective,
        "priority": int(priority),
        "status": status,
        "blocked": bool(blockers),
        "blockers": blockers,
        "selected_lf_payload_evidence": selected_evidence,
        "measured_lf_payload_bytes": (
            None if selected_evidence is None else selected_evidence.get("lf_payload_bytes")
        ),
        "measured_raw_lf_bytes": (
            None if selected_evidence is None else selected_evidence.get("raw_lf_bytes")
        ),
        "hard_byte_ceiling": _positive_int(campaign_row.get("hard_byte_ceiling")),
        "source_campaign_status": {
            "implementation_status": campaign_row.get("implementation_status"),
            "local_mlx_launch_command_ready": bool(campaign_row.get("local_mlx_launch_command_ready")),
            "hard_byte_ceiling_satisfied_for_long_training": campaign_row.get("hard_byte_ceiling_satisfied_for_long_training"),
            "candidate_nominal_under_ceiling": campaign_row.get("candidate_nominal_under_ceiling"),
            "campaign_blockers": list(campaign_row.get("blockers") or ()),
        },
        "snar2_current_state": {
            "freshest_queue_has_no_lf_over_ceiling_rows": current_state.get(
                "freshest_queue_has_no_lf_over_ceiling_rows"
            ),
            "freshest_reroute_queue_row_count": current_state.get(
                "freshest_reroute_queue_row_count"
            ),
            "snar_header_minimization_report_count": current_state.get(
                "snar_header_minimization_report_count"
            ),
            "lf_dominance_launch_signal_active": current_state.get(
                "lf_dominance_launch_signal_active"
            ),
            "lf_dominance_signal_demoted": current_state.get(
                "lf_dominance_signal_demoted"
            ),
            "demoted_blockers": list(current_state.get("demoted_blockers") or ()),
        },
        "source_forward_evidence": current_state.get("source_forward_evidence"),
        "scorer_domain_evidence": current_state.get("scorer_domain_evidence"),
        "target_consumers": [
            "nerv_long_training_campaign_plan",
            "snerv_lf_over_ceiling_reroute_queue",
            "nerv_rate_allocator_queue",
            "cathedral_autopilot",
        ],
        "command_argv": command,
        "output_root": output_root.as_posix(),
        **QUEUE_FALSE_AUTHORITY,
    }


def _bounded_snerv_smoke_command(
    campaign_row: Mapping[str, Any],
    *,
    queue_row_id: str,
    output_root: Path,
) -> list[str]:
    command = [str(part) for part in campaign_row.get("command_argv") or ()]
    if not command:
        return []
    replacements = {
        "--planner-row-id": queue_row_id,
        "--num-pairs": "16",
        "--epochs": "128",
        "--snerv-score-aware-long-training-epochs": "128",
        "--snerv-score-aware-long-training-batch-pairs": "2",
        "--mlx-prefilter-scorer-batch-pairs": "1",
        "--mlx-prefilter-progress-every": "4",
        "--snerv-native-mlx-receiver-proof-timeout": "600",
        "--output-dir": (output_root / queue_row_id / "bounded_smoke").as_posix(),
        "--planner-row-queue-artifact": (output_root / "snerv_lf_hf_replacement_queue.json").as_posix(),
    }
    out: list[str] = []
    idx = 0
    while idx < len(command):
        token = command[idx]
        out.append(token)
        if token in replacements and idx + 1 < len(command):
            out.append(replacements[token])
            idx += 2
            continue
        idx += 1
    present = set(command)
    for flag, value in replacements.items():
        if flag not in present:
            out.extend([flag, value])
    return out


def _global_blocker_row(
    *,
    output_root: Path,
    blocker: str,
    selected_evidence: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema": ROW_SCHEMA,
        "queue_row_id": "snerv_lf_hf_replace_global_blocker",
        "lane_id": DEFAULT_LANE_ID,
        "source_campaign_row_id": None,
        "candidate_id": None,
        "candidate_class": "global_blocker",
        "solution_family": "lf_hf_replacement_queue_bootstrap",
        "planner_action": "attach_current_snerv_campaign_plan_before_candidate_emission",
        "learning_objective": "blocked before any prototype can be selected",
        "priority": 999,
        "status": "blocked_until_prerequisite_evidence",
        "blocked": True,
        "blockers": [str(blocker)],
        "selected_lf_payload_evidence": selected_evidence,
        "target_consumers": ["nerv_long_training_campaign_plan"],
        "command_argv": [],
        "output_root": output_root.as_posix(),
        **QUEUE_FALSE_AUTHORITY,
    }


def _lf_evidence_row(report: Mapping[str, Any], source_index: int) -> dict[str, Any] | None:
    if not isinstance(report, Mapping):
        return None
    schema = str(report.get("schema") or "")
    base = {
        "schema": "snerv_lf_hf_payload_evidence_ref.v1",
        "source_index": int(source_index),
        "source_schema": schema,
        "source_path": report.get("_source_path") or report.get("report_path"),
        "source_sha256": report.get("_source_sha256"),
        "authority": report.get("authority"),
        "axis_tag": report.get("axis_tag"),
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }
    if schema == "snerv_lf_payload_codec_sweep.v1":
        selected = report.get("selected_rate_only_row")
        selected = selected if isinstance(selected, Mapping) else {}
        return {
            **base,
            "evidence_kind": "lf_payload_codec_sweep",
            "plane_count": _positive_int(report.get("plane_count")),
            "plane_shapes": report.get("plane_shapes"),
            "raw_lf_bytes": _positive_int(report.get("raw_i64_bytes")),
            "lf_payload_bytes": _positive_int(selected.get("payload_bytes")),
            "selected_mode": selected.get("mode"),
            "baseline_payload_bytes": _positive_int(report.get("baseline_payload_bytes")),
            "blockers": _dedupe(report.get("blockers") or ()),
        }
    if schema == "snerv_lf_payload_archive_recode.v1":
        lf_payload = report.get("lf_payload")
        lf_payload = lf_payload if isinstance(lf_payload, Mapping) else {}
        return {
            **base,
            "evidence_kind": "receiver_packet_lf_recode",
            "plane_count": _positive_int(report.get("lf_plane_count")),
            "raw_lf_bytes": _positive_int(_nested(lf_payload, ("source_header", "raw_bytes"))),
            "lf_payload_bytes": _positive_int(lf_payload.get("source_bytes")),
            "candidate_lf_payload_bytes": _positive_int(lf_payload.get("candidate_bytes")),
            "candidate_packet_bytes": _positive_int(_nested(report, ("candidate_packet", "bytes"))),
            "candidate_packet_path": _nested(report, ("candidate_packet", "path")),
            "selected_mode": report.get("mode"),
            "receiver_contract_satisfied": report.get("receiver_contract_satisfied") is True,
            "blockers": _dedupe(report.get("blockers") or ()),
        }
    if schema == "snerv_lf_payload_recode_admission_plan.v1":
        selected = report.get("selected_row")
        selected = selected if isinstance(selected, Mapping) else {}
        return {
            **base,
            "evidence_kind": "lf_recode_admission_plan",
            "lf_payload_bytes": _positive_int(selected.get("lf_source_bytes")),
            "candidate_lf_payload_bytes": _positive_int(selected.get("lf_candidate_bytes")),
            "candidate_packet_bytes": _positive_int(selected.get("candidate_packet_bytes")),
            "candidate_packet_path": selected.get("candidate_packet_path"),
            "selected_mode": selected.get("mode"),
            "post_recode_over_waterline_bytes": _positive_int(
                selected.get("post_recode_over_waterline_bytes")
            ),
            "blockers": _dedupe(report.get("blockers") or ()),
        }
    if schema == "snerv_official_dummy_lf_payload_codec_sweep.v1":
        selected = report.get("selected_rate_only_row")
        selected = selected if isinstance(selected, Mapping) else {}
        return {
            **base,
            "evidence_kind": "official_dummy_lf_receiver_section",
            "plane_count": _positive_int(report.get("lf_plane_count")),
            "raw_lf_bytes": _positive_int(report.get("raw_i64_bytes")),
            "lf_payload_bytes": _positive_int(selected.get("receiver_section_total_bytes")),
            "selected_mode": selected.get("mode"),
            "blockers": _dedupe(report.get("blockers") or ()),
        }
    return None


def _selected_lf_evidence(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    valid = [row for row in rows if _positive_int(row.get("lf_payload_bytes")) is not None]
    if not valid:
        return None
    selected = max(
        valid,
        key=lambda row: (
            int(row.get("lf_payload_bytes") or 0),
            int(row.get("raw_lf_bytes") or 0),
            str(row.get("source_path") or ""),
        ),
    )
    return dict(selected)


def _source_forward_state(
    source_forward_artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    artifacts = [
        artifact
        for artifact in source_forward_artifacts
        if isinstance(artifact, Mapping)
        and artifact.get("schema") == "snerv_official_mfu_hfr_tub_forward_parity.v1"
    ]
    if not artifacts:
        return {
            "schema": "snerv_lf_hf_source_forward_evidence.v1",
            "artifact_count": 0,
            "selected_artifact_schema": None,
            "selected_artifact_generated_utc": None,
            "source_path": None,
            "source_sha256": None,
            "receiver_payload_frame_replay_proven": False,
            "receiver_runtime_decode_proven": False,
            "frame_producing_official_payload_replay_proven": False,
            "receiver_frame_decode_consumes_output2": False,
            "full_tub_source_forward_parity_proven": False,
            "closed_campaign_blockers": [],
            "queue_blockers": ["snerv_lf_hf_source_forward_artifact_missing"],
            "blockers": ["snerv_lf_hf_source_forward_artifact_missing"],
            **QUEUE_FALSE_AUTHORITY,
        }
    selected = max(
        artifacts,
        key=lambda artifact: (
            str(artifact.get("generated_utc") or ""),
            str(artifact.get("_source_path") or ""),
        ),
    )
    replay = selected.get("receiver_payload_frame_replay")
    replay = replay if isinstance(replay, Mapping) else {}
    receiver_runtime_decode = replay.get("receiver_runtime_decode_proven") is True
    frame_payload_replay = (
        replay.get("frame_producing_official_payload_replay_proven") is True
    )
    frame_replay_proven = receiver_runtime_decode and frame_payload_replay
    receiver_consumes_output2 = replay.get("receiver_frame_decode_consumes_output2") is True
    full_tub_parity = selected.get("full_tub_source_forward_parity_proven") is True
    source_authority = replay.get("source_forward_replay_authority") is True and full_tub_parity
    closed = list(_SOURCE_FORWARD_FRAME_REPLAY_CLOSED_BLOCKERS) if frame_replay_proven else []
    queue_blockers: list[str] = []
    if not frame_replay_proven:
        queue_blockers.extend(_SOURCE_FORWARD_FRAME_REPLAY_CLOSED_BLOCKERS)
    if not receiver_consumes_output2:
        queue_blockers.append("snerv_official_tub_output2_receiver_frame_decode_not_bound")
    if not source_authority:
        queue_blockers.extend(
            [
                "snerv_official_mfu_hfr_tub_receiver_payload_not_source_forward_authority",
                "snerv_official_mfu_hfr_tub_full_stack_source_forward_replay_missing",
            ]
        )
    return {
        "schema": "snerv_lf_hf_source_forward_evidence.v1",
        "artifact_count": len(artifacts),
        "selected_artifact_schema": selected.get("schema"),
        "selected_artifact_generated_utc": selected.get("generated_utc"),
        "source_path": selected.get("_source_path"),
        "source_sha256": selected.get("_source_sha256"),
        "receiver_payload_frame_replay_proven": frame_replay_proven,
        "receiver_runtime_decode_proven": receiver_runtime_decode,
        "frame_producing_official_payload_replay_proven": frame_payload_replay,
        "receiver_frame_decode_consumes_output2": receiver_consumes_output2,
        "full_tub_source_forward_parity_proven": full_tub_parity,
        "source_forward_replay_authority": source_authority,
        "decoded_frames_shape": replay.get("decoded_frames_shape"),
        "decoded_frames_sha256": replay.get("decoded_frames_sha256"),
        "payload_bytes": _positive_int(replay.get("payload_bytes")),
        "payload_sha256": replay.get("payload_sha256"),
        "closed_campaign_blockers": closed,
        "queue_blockers": _dedupe(queue_blockers),
        "blockers": _dedupe([*(selected.get("blockers") or ()), *queue_blockers]),
        **QUEUE_FALSE_AUTHORITY,
    }


def _scorer_domain_state(
    candidate_feedback_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = [
        row
        for row in candidate_feedback_rows
        if isinstance(row, Mapping) and row.get("schema") == "nerv_candidate_feedback_row.v1"
    ]
    if not rows:
        return {
            "schema": "snerv_lf_hf_scorer_domain_evidence.v1",
            "artifact_count": 0,
            "selected_artifact_schema": None,
            "selected_artifact_created_utc": None,
            "source_path": None,
            "source_sha256": None,
            "scorer_domain_tether_proof_passed": False,
            "required_metrics": list(_SCORER_DOMAIN_REQUIRED_METRICS),
            "metric_health": {},
            "missing_metrics": list(_SCORER_DOMAIN_REQUIRED_METRICS),
            "lambda_inactive_metrics": list(_SCORER_DOMAIN_REQUIRED_METRICS),
            "closed_campaign_blockers": [],
            "queue_blockers": ["snerv_scorer_input_distribution_guard_missing"],
            "blockers": ["snerv_lf_hf_scorer_domain_candidate_feedback_missing"],
            **QUEUE_FALSE_AUTHORITY,
        }
    selected = max(
        rows,
        key=lambda row: (
            str(row.get("created_utc") or row.get("generated_utc") or ""),
            str(row.get("_source_path") or row.get("source_report_path") or ""),
        ),
    )
    health = selected.get("snerv_scorer_domain_tether_health")
    health = health if isinstance(health, Mapping) else {}
    metric_health = health.get("metric_health")
    metric_health = metric_health if isinstance(metric_health, Mapping) else {}
    missing_metrics: list[str] = []
    lambda_inactive_metrics: list[str] = []
    for metric in _SCORER_DOMAIN_REQUIRED_METRICS:
        metric_row = metric_health.get(metric)
        metric_row = metric_row if isinstance(metric_row, Mapping) else {}
        if metric_row.get("metric_observed") is not True:
            missing_metrics.append(metric)
        if metric_row.get("lambda_active_observed") is not True:
            lambda_inactive_metrics.append(metric)
    explicit_blockers = _dedupe(
        [
            *(selected.get("snerv_scorer_domain_tether_blockers") or ()),
            *(health.get("blockers") or ()),
        ]
    )
    proof_passed = bool(
        selected.get("snerv_scorer_domain_tether_passed") is True
        and health.get("passed") is True
        and not missing_metrics
        and not lambda_inactive_metrics
        and not explicit_blockers
    )
    blockers: list[str] = []
    if not proof_passed:
        blockers.append("snerv_scorer_input_distribution_guard_missing")
    if missing_metrics:
        blockers.append("snerv_scorer_domain_tether_missing_telemetry")
    if lambda_inactive_metrics:
        blockers.append("snerv_scorer_domain_tether_lambda_inactive_telemetry")
    blockers.extend(explicit_blockers)
    return {
        "schema": "snerv_lf_hf_scorer_domain_evidence.v1",
        "artifact_count": len(rows),
        "selected_artifact_schema": selected.get("schema"),
        "selected_artifact_created_utc": selected.get("created_utc"),
        "source_path": selected.get("_source_path") or selected.get("source_report_path"),
        "source_sha256": selected.get("_source_sha256") or selected.get("source_report_sha256"),
        "candidate_id": selected.get("candidate_id"),
        "family": selected.get("family"),
        "scorer_domain_tether_proof_passed": proof_passed,
        "required_metrics": list(_SCORER_DOMAIN_REQUIRED_METRICS),
        "metric_health": {str(k): v for k, v in metric_health.items()},
        "missing_metrics": missing_metrics,
        "lambda_inactive_metrics": lambda_inactive_metrics,
        "closed_campaign_blockers": (
            list(_SCORER_DOMAIN_CLOSED_BLOCKERS) if proof_passed else []
        ),
        "queue_blockers": [] if proof_passed else ["snerv_scorer_input_distribution_guard_missing"],
        "blockers": _dedupe(blockers),
        **QUEUE_FALSE_AUTHORITY,
    }


def _snerv_campaign_rows(campaign_plans: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for plan in campaign_plans:
        for row in plan.get("campaign_rows") or ():
            if isinstance(row, Mapping) and row.get("family") == "snerv":
                rows.append(dict(row))
    rows.sort(
        key=lambda row: (
            0 if row.get("local_mlx_launch_command_ready") is True else 1,
            int(row.get("priority") or 999),
            str(row.get("candidate_id") or ""),
        )
    )
    return rows


def _reroute_state(reroute_queues: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    queues = [q for q in reroute_queues if isinstance(q, Mapping)]
    if not queues:
        return {
            "reroute_queue_count": 0,
            "freshest_reroute_queue_row_count": None,
            "freshest_queue_has_no_lf_over_ceiling_rows": None,
            "snar_header_minimization_report_count": 0,
            "all_reroute_queue_row_count": 0,
        }
    freshest = queues[-1]
    return {
        "reroute_queue_count": len(queues),
        "freshest_schema": freshest.get("schema"),
        "freshest_generated_utc": freshest.get("generated_utc"),
        "freshest_reroute_queue_row_count": _nonnegative_int(
            freshest.get("queue_row_count")
        ),
        "freshest_queue_has_no_lf_over_ceiling_rows": (
            _nonnegative_int(freshest.get("queue_row_count")) == 0
        ),
        "snar_header_minimization_report_count": _nonnegative_int(
            freshest.get("snar_header_minimization_report_count")
        )
        or 0,
        "all_reroute_queue_row_count": sum(
            int(q.get("queue_row_count") or 0) for q in queues
        ),
    }


def _current_state(
    *,
    campaign_rows: Sequence[Mapping[str, Any]],
    reroute_state: Mapping[str, Any],
    evidence_rows: Sequence[Mapping[str, Any]],
    source_forward_state: Mapping[str, Any],
    scorer_domain_state: Mapping[str, Any],
) -> dict[str, Any]:
    blockers: list[str] = []
    if not evidence_rows:
        blockers.append("snerv_lf_hf_measured_lf_payload_report_missing")
    if not campaign_rows:
        blockers.append("snerv_lf_hf_current_campaign_plan_missing")
    if reroute_state.get("freshest_queue_has_no_lf_over_ceiling_rows") is True:
        blockers.append("snerv_lf_hf_current_snar2_queue_has_no_lf_over_ceiling_rows")
    if reroute_state.get("reroute_queue_count") == 0:
        blockers.append("snerv_lf_hf_reroute_queue_missing")
    ready_rows = [
        row for row in campaign_rows if row.get("local_mlx_launch_command_ready") is True
    ]
    return {
        **dict(reroute_state),
        "snerv_campaign_row_count": len(campaign_rows),
        "snerv_local_mlx_launch_command_ready_row_count": len(ready_rows),
        "lf_payload_evidence_row_count": len(evidence_rows),
        "source_forward_evidence": dict(source_forward_state),
        "scorer_domain_evidence": dict(scorer_domain_state),
        "blockers": _dedupe(blockers),
    }


def _campaign_blockers(
    campaign_row: Mapping[str, Any],
    prefixes: Sequence[str],
) -> list[str]:
    source = [str(blocker) for blocker in campaign_row.get("blockers") or () if blocker]
    out: list[str] = []
    for blocker in source:
        if blocker in prefixes or any(blocker.startswith(prefix) for prefix in prefixes):
            out.append(blocker)
    return _dedupe(out)


def _storage_preflight(
    output_root: Path,
    *,
    min_free_bytes: int,
    allow_local_output: bool,
) -> dict[str, Any]:
    root = output_root.expanduser().resolve(strict=False)
    on_ssd = any(_is_relative_to(root, ssd_root) for ssd_root in SSD_ROOTS)
    free = shutil.disk_usage(_nearest_existing_parent(root)).free
    blockers: list[str] = []
    if not on_ssd and not allow_local_output:
        blockers.append("snerv_lf_hf_replacement_output_root_not_on_ssd_tier")
    if free < int(min_free_bytes):
        blockers.append("snerv_lf_hf_replacement_output_root_free_space_below_floor")
    if blockers:
        raise SnervLfHfReplacementQueueError(
            f"{root}: storage preflight blocked: {', '.join(blockers)}"
        )
    return {
        "schema": "snerv_lf_hf_replacement_storage_preflight.v1",
        "output_root": root.as_posix(),
        "ssd_tier": _ssd_tier(root),
        "free_bytes_before": int(free),
        "min_free_bytes": int(min_free_bytes),
        "allow_local_output": bool(allow_local_output),
        "blockers": [],
    }


def _ssd_tier(path: Path) -> str:
    for root in SSD_ROOTS:
        if _is_relative_to(path, root):
            return root.as_posix()
    return "local_or_unknown"


def _nearest_existing_parent(path: Path) -> Path:
    cursor = path
    while not cursor.exists() and cursor.parent != cursor:
        cursor = cursor.parent
    return cursor


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _nested(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _positive_int(value: Any) -> int | None:
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    return out if out > 0 else None


def _nonnegative_int(value: Any) -> int | None:
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    return out if out >= 0 else None


def _stable_safe_token(text: str, *, max_len: int = 120) -> str:
    clean = "".join(ch if ch.isalnum() else "_" for ch in str(text).lower())
    clean = "_".join(part for part in clean.split("_") if part)
    digest = hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:12]
    base = clean[: max(1, max_len - 13)].strip("_")
    return f"{base}_{digest}" if base else digest


def _dedupe(values: Sequence[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _shell_join(argv: Sequence[Any]) -> str:
    return " ".join(str(part) for part in argv)


def attach_source_identity(payload: Mapping[str, Any], path: str | Path) -> dict[str, Any]:
    """Return ``payload`` with source path and SHA-256 metadata attached."""

    source_path = Path(path)
    data = source_path.read_bytes()
    return {
        **dict(payload),
        "_source_path": source_path.as_posix(),
        "_source_sha256": hashlib.sha256(data).hexdigest(),
    }


def load_json_with_source_identity(path: str | Path) -> dict[str, Any]:
    """Load a JSON object and attach path/SHA metadata for custody."""

    source_path = Path(path)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SnervLfHfReplacementQueueError(f"{source_path}: JSON payload must be an object")
    return attach_source_identity(payload, source_path)
