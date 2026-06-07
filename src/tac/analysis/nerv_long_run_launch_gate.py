# SPDX-License-Identifier: MIT
"""Fail-closed launch gate for NeRV-family long MLX training runs.

Evidence for a long run accumulates across many producers (birth receipts,
pose-trust rows, survival receipts, parse-back manifests, SNeRV source-forward
proofs).  Nothing may consume that pile and say "approved" by vibes.  This
module is the single machine arbiter: it scans a run root for typed evidence,
walks the family ladder (L2 physical birth -> L3 pose-trusted -> L4 same-action
survival -> L5 representative/inflate), and emits a machine-readable verdict
whose only approval path is every required row present and consistent.

It also DEFINES the survival/hysteresis/coverage schemas before their
producers exist: the gate is the contract; producers conform to it:

* ``hi_nerv_target_region_birth_survival.v1`` - re-measurement of the same
  accepted birth (matching ``action_id``) on one surface in
  {``fakequant_mlx``, ``parseback_mlx``, ``inflated_torch_cpu``} with
  ``survived: bool`` and target-support stats re-measured on that surface.
* ``hi_nerv_target_region_birth_hysteresis.v1`` - same ``action_id`` after M
  further steps with ``passed: bool`` (no hard-won collapse, debt not above
  pre-birth).
* ``hi_nerv_representative_region_coverage.v1`` - hard-region-miner coverage
  row proving birth beyond the birth-at-init easy case.

L4 survival is proof ONLY when the same action identity is traced through
surfaces: survival rows must CARRY the live receipt's ``action_id``; a
re-solved birth under fakequant is a different experiment.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tac.analysis.action_effect import (
    ACTION_EFFECT_V1_SCHEMA,
    NormalizationScope,
    validate_action_effect_payload,
)
from tac.analysis.inverse_scorer_actions import (
    SIDECAR_FALLBACK_ACCEPTED,
    TARGET_REGION_WALL_NORMAL_LIFT_SCHEMA,
    WALL_NORMAL_BRANCH_RECEIPT_SCHEMA,
)
from tac.analysis.receiver_surface_metrics import receiver_surface_target_support_breakdown
from tac.analysis.snerv_source_forward_proof import (
    SNERV_SOURCE_FORWARD_PROOF_ACTION_EFFECT_SCHEMA,
    find_snerv_source_forward_proof_rows,
    validate_snerv_source_forward_proof_action_effect,
)
from tac.optimization.proxy_candidate_contract import (
    PROXY_FALSE_AUTHORITY_FIELDS,
    require_no_truthy_authority_fields,
)

NERV_LONG_RUN_LAUNCH_GATE_SCHEMA = "nerv_long_run_launch_gate.v1"
BIRTH_RECEIPT_SCHEMA = "hi_nerv_target_region_birth_receipt.v1"
BIRTH_SURVIVAL_SCHEMA = "hi_nerv_target_region_birth_survival.v1"
BIRTH_HYSTERESIS_SCHEMA = "hi_nerv_target_region_birth_hysteresis.v1"
REPRESENTATIVE_COVERAGE_SCHEMA = "hi_nerv_representative_region_coverage.v1"
SNERV_SOURCE_FORWARD_SCHEMA = "snerv_official_tub_source_forward_replay.v1"
ARCHIVE_PARSEBACK_SELECTION_CONTRACT_SCHEMA = "nerv_archive_parseback_selection_contract.v1"
SOURCE_QUALIFIED_METRICS_SCHEMA = "nerv_source_qualified_metrics.v1"
HI_NERV_SHORT_SCORER_SMOKE_READINESS_SCHEMA = "hi_nerv_short_scorer_smoke_readiness.v1"
HI_NERV_TARGET_REGION_ACTION_LOWERING_RACE_SCHEMA = "hi_nerv_target_region_action_lowering_race.v1"
EVALUATOR_ACTION_LOWERING_RACE_SCHEMA = "tac.evaluator_action_lowering_race.v1"
EVALUATOR_ACTION_LOWERING_TARGETS: tuple[str, ...] = (
    "backend_realization",
    "pair_local_latent_action",
    "frame0_pose_compensation",
    "frame1_seg_wall_crossing",
    "byte_priced_sidecar",
    "pose_compensated_composite",
    "snerv_source_state_action",
    "semantic_pose_primitive",
    "byte_entropy_rewrite",
)
_TARGET_REGION_ACTION_CODEC_SUPPORT_ENCODING = {
    "raw_v1": "explicit_yx_u16_coordinates",
    "zlib_wrapped_v1": "zlib_wrapped_raw_yx_u16_coordinates",
    "brotli_wrapped_v1": "brotli_wrapped_raw_yx_u16_coordinates",
    "split_brotli_v1": "brotli_split_yx_delta_streams",
    "tile_brotli_v1": "brotli_tile_bitmap_little_endian",
}
SUPPORTED_FAMILIES = ("hinerv", "hi_nerv", "snerv")
SURVIVAL_SURFACES_L4 = ("fakequant_mlx", "parseback_mlx")
SURVIVAL_SURFACE_L5 = "inflated_torch_cpu"
FRONTIER_POINTER_MAX_AGE_HOURS = 24.0
_MAX_EVIDENCE_FILE_BYTES = 64 * 1024 * 1024
HINERV_REQUIRED_FOUR_ARM_ACTION_KINDS = {
    "A": "birth_only",
    "B": "frame0_pose_target_only",
    "C": "independent_birth_plus_frame0_pose",
    "D": "joint_line_search_composite",
}


class NervLongRunLaunchGateError(ValueError):
    """Raised when launch-gate inputs are structurally invalid."""


def _read_json(path: Path) -> Any | None:
    try:
        if path.stat().st_size > _MAX_EVIDENCE_FILE_BYTES:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _iter_evidence_payloads(run_root: Path):
    for path in sorted(run_root.rglob("*.json")):
        payload = _read_json(path)
        if isinstance(payload, dict):
            yield path, payload
        # JSONL receipts: one payload per line.
    for path in sorted(run_root.rglob("*.jsonl")):
        try:
            if path.stat().st_size > _MAX_EVIDENCE_FILE_BYTES:
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    yield path, payload
        except OSError:
            continue


def _walk_schema(payload: Any, schema: str, hits: list[dict[str, Any]]) -> None:
    if isinstance(payload, Mapping):
        if str(payload.get("schema") or "") == schema:
            hits.append(dict(payload))
        for value in payload.values():
            _walk_schema(value, schema, hits)
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            _walk_schema(value, schema, hits)


def _collect_schema_rows(
    run_root: Path,
    schema: str,
    *,
    index: dict[str, list[str]],
    blockers: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path, payload in _iter_evidence_payloads(run_root):
        hits: list[dict[str, Any]] = []
        _walk_schema(payload, schema, hits)
        if not hits:
            continue
        try:
            require_no_truthy_authority_fields(payload, context=str(path))
        except Exception:
            blockers.append(f"evidence_truthy_authority:{path.name}")
            continue
        rows.extend(hits)
        index.setdefault(schema, []).append(path.as_posix())
    return rows


def _frontier_pointer_status(
    frontier_pointer: Path | None,
    *,
    now_utc: datetime,
) -> dict[str, Any]:
    if frontier_pointer is None or not frontier_pointer.is_file():
        return {"present": False, "fresh": False, "last_refreshed_utc": None}
    payload = _read_json(frontier_pointer)
    if not isinstance(payload, dict):
        return {"present": False, "fresh": False, "last_refreshed_utc": None}
    raw = str(payload.get("last_refreshed_utc") or "")
    try:
        refreshed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if refreshed.tzinfo is None:
            refreshed = refreshed.replace(tzinfo=UTC)
        age_hours = (now_utc - refreshed).total_seconds() / 3600.0
        fresh = math.isfinite(age_hours) and 0.0 <= age_hours <= (FRONTIER_POINTER_MAX_AGE_HOURS)
    except ValueError:
        refreshed = None
        fresh = False
    return {
        "present": True,
        "fresh": bool(fresh),
        "last_refreshed_utc": raw or None,
    }


def _accepted_live_birth(
    rows: list[dict[str, Any]],
    *,
    blockers: list[str],
) -> dict[str, Any] | None:
    for row in rows:
        if str(row.get("surface") or "live_mlx") != "live_mlx":
            continue
        if int(row.get("accepted_step_count") or 0) <= 0:
            continue
        if not _target_support_positive(
            row,
            blocker="live_birth_target_support_missing",
            not_positive_blocker="live_birth_target_support_not_positive",
            blockers=blockers,
        ):
            continue
        if not row.get("action_id"):
            continue
        return row
    return None


def _rejected_live_birth_receipt_blockers(rows: list[dict[str, Any]]) -> list[str]:
    blockers: list[str] = []
    for row in rows:
        if str(row.get("surface") or "live_mlx") != "live_mlx":
            continue
        if int(row.get("accepted_step_count") or 0) <= 0:
            blockers.append("live_birth_accepted_step_missing_or_zero")
        local_blockers: list[str] = []
        _target_support_positive(
            row,
            blocker="live_birth_target_support_missing",
            not_positive_blocker="live_birth_target_support_not_positive",
            blockers=local_blockers,
        )
        blockers.extend(local_blockers)
        if not row.get("action_id"):
            blockers.append("live_birth_action_id_missing")
        blockers.extend(str(blocker) for blocker in row.get("blockers") or [])
    return _dedupe(blockers)


def _positive_unclosed_masked_oracle_birth(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return a positive receiver-surface oracle that is still byte-unclosed."""

    for row in rows:
        telemetry = row.get("candidate_frontier_telemetry")
        if not isinstance(telemetry, Mapping):
            continue
        oracle = telemetry.get("masked_residual_oracle")
        if not isinstance(oracle, Mapping):
            continue
        if oracle.get("authority") != "receiver_surface_oracle_false_authority":
            continue
        if oracle.get("archive_closed") is True:
            continue
        if oracle.get("exact_accepted_before_archive_closure") is not True:
            continue
        if oracle.get("target_support_moved") is not True:
            continue
        out = dict(oracle)
        out.setdefault("action_id", row.get("action_id"))
        return out
    return None


def _target_region_action_archive_closure_status(
    run_root: Path,
    *,
    index: dict[str, list[str]],
) -> dict[str, Any]:
    """Find receiver-charged target-region action archive evidence.

    The masked oracle is produced before export, so its candidate blockers are
    necessarily conservative.  Retire only the blockers proven by later charged
    archive telemetry; parseback/inflate survival remain separate gates.
    """

    charged_meta_path: str | None = None
    selected_archive_path: str | None = None
    selected_archive_bytes: int | None = None
    candidate_blockers: list[str] = []
    for path, payload in _iter_evidence_payloads(run_root):
        stack = [payload]
        while stack:
            node = stack.pop()
            if isinstance(node, Mapping):
                node_archive_bytes = _archive_zip_bytes_from_node(node)
                target_actions = node.get("target_region_actions")
                if isinstance(target_actions, Mapping):
                    try:
                        payload_bytes = int(target_actions.get("payload_bytes") or 0)
                        base64_bytes = int(target_actions.get("base64_text_bytes") or 0)
                        meta_bytes = int(target_actions.get("charged_meta_json_bytes") or 0)
                    except (TypeError, ValueError):
                        payload_bytes = 0
                        base64_bytes = 0
                        meta_bytes = 0
                    if (
                        target_actions.get("schema") == "hi_nerv_target_region_archive_actions.v1"
                        and target_actions.get("charged_as_hiv1_meta_blob") is True
                        and target_actions.get("receiver_consumed") is True
                        and payload_bytes > 0
                        and base64_bytes > 0
                        and (meta_bytes > 0 or node_archive_bytes is not None)
                    ):
                        telemetry_blockers = _target_region_action_telemetry_blockers(
                            target_actions
                        )
                        if telemetry_blockers:
                            candidate_blockers.extend(telemetry_blockers)
                        else:
                            charged_meta_path = path.as_posix()
                            index.setdefault(
                                "hi_nerv_target_region_archive_actions.v1",
                                [],
                            ).append(path.as_posix())
                if node_archive_bytes is not None:
                    selected_archive_bytes = node_archive_bytes
                    raw_archive_path = node.get("selected_archive_path") or node.get(
                        "archive_path"
                    )
                    selected_archive_path = (
                        str(raw_archive_path) if raw_archive_path else path.as_posix()
                    )
                stack.extend(node.values())
            elif isinstance(node, (list, tuple)):
                stack.extend(node)
    return {
        "charged_meta_materialized": charged_meta_path is not None,
        "charged_meta_path": charged_meta_path,
        "archive_zip_bytes_measured": selected_archive_bytes is not None,
        "selected_archive_path": selected_archive_path,
        "selected_archive_bytes": selected_archive_bytes,
        "blockers": [] if charged_meta_path is not None else _dedupe(candidate_blockers),
    }


def _archive_zip_bytes_from_node(node: Mapping[str, Any]) -> int | None:
    for key in ("selected_archive_bytes", "archive_bytes"):
        try:
            archive_bytes = int(node.get(key) or 0)
        except (TypeError, ValueError):
            archive_bytes = 0
        if archive_bytes > 0:
            return archive_bytes
    return None


def _target_region_action_telemetry_blockers(
    target_actions: Mapping[str, Any],
) -> list[str]:
    blockers: list[str] = []
    codec = str(target_actions.get("payload_codec") or "").strip()
    expected_support_encoding = _TARGET_REGION_ACTION_CODEC_SUPPORT_ENCODING.get(codec)
    if not expected_support_encoding:
        blockers.append("target_region_action_payload_codec_missing_or_unknown")
    support_encoding = str(target_actions.get("support_encoding") or "").strip()
    if not support_encoding:
        blockers.append("target_region_action_support_encoding_missing")
    elif expected_support_encoding and support_encoding != expected_support_encoding:
        blockers.append("target_region_action_support_encoding_mismatch")
    try:
        support_encoded_bytes = int(target_actions.get("support_encoded_bytes") or 0)
    except (TypeError, ValueError):
        support_encoded_bytes = 0
    if support_encoded_bytes <= 0:
        blockers.append("target_region_action_support_encoded_bytes_missing")
    for field, blocker in (
        ("encoded_program_sha256", "target_region_action_encoded_program_sha256_missing"),
        ("decoded_support_sha256", "target_region_action_decoded_support_sha256_missing"),
        ("decoded_action_sha256", "target_region_action_decoded_action_sha256_missing"),
    ):
        value = str(target_actions.get(field) or "").strip()
        if not value:
            blockers.append(blocker)
    return blockers


def _target_region_action_survival_status(
    run_root: Path,
    *,
    action_id: str | None = None,
    support_sha256: str | None = None,
    index: dict[str, list[str]],
) -> dict[str, Any]:
    parseback_path: str | None = None
    inflate_path: str | None = None
    partial_parseback_path: str | None = None
    partial_inflate_path: str | None = None
    parseback_survived = False
    inflate_survived = False
    partial_parseback_survived = False
    partial_inflate_survived = False
    same_row_parseback_inflate_survived = False
    for path, payload in _iter_evidence_payloads(run_root):
        stack = [payload]
        while stack:
            node = stack.pop()
            if isinstance(node, Mapping):
                if str(node.get("schema") or "") == "hi_nerv_target_region_action_parseback_survival.v1":
                    index.setdefault(
                        "hi_nerv_target_region_action_parseback_survival.v1",
                        [],
                    ).append(path.as_posix())
                    row_action_id = str(node.get("action_id") or "")
                    if action_id and row_action_id != str(action_id):
                        stack.extend(node.values())
                        continue
                    row_support_sha256 = _target_region_action_support_sha256(node)
                    if support_sha256 and row_support_sha256 != str(support_sha256):
                        stack.extend(node.values())
                        continue
                    same_row_survived = (
                        node.get("survived") is True
                        and node.get("fakequant_survived") is True
                        and node.get("parseback_survived") is True
                        and node.get("inflate_survived") is True
                    )
                    if same_row_survived:
                        same_row_parseback_inflate_survived = True
                        parseback_survived = True
                        parseback_path = path.as_posix()
                        inflate_survived = True
                        inflate_path = path.as_posix()
                    else:
                        if (
                            node.get("survived") is True
                            and node.get("fakequant_survived") is True
                            and node.get("parseback_survived") is True
                        ):
                            parseback_survived = True
                            parseback_path = path.as_posix()
                            partial_parseback_survived = True
                            partial_parseback_path = path.as_posix()
                        if node.get("inflate_survived") is True:
                            inflate_survived = True
                            inflate_path = path.as_posix()
                            partial_inflate_survived = True
                            partial_inflate_path = path.as_posix()
                stack.extend(node.values())
            elif isinstance(node, (list, tuple)):
                stack.extend(node)
    return {
        "parseback_survived": parseback_survived,
        "parseback_path": parseback_path,
        "inflate_survived": inflate_survived,
        "inflate_path": inflate_path,
        "same_row_parseback_inflate_survived": same_row_parseback_inflate_survived,
        "partial_parseback_survived": partial_parseback_survived,
        "partial_parseback_path": partial_parseback_path,
        "partial_inflate_survived": partial_inflate_survived,
        "partial_inflate_path": partial_inflate_path,
    }


def _target_region_action_support_sha256(payload: Mapping[str, Any]) -> str:
    candidates: list[Any] = [
        payload.get("expected_support_sha256"),
        payload.get("support_sha256"),
    ]
    target_region_actions = payload.get("target_region_actions")
    if isinstance(target_region_actions, Mapping):
        candidates.extend(
            [
                target_region_actions.get("support_sha256"),
                target_region_actions.get("expected_support_sha256"),
            ]
        )
    telemetry = payload.get("target_region_action_section_telemetry")
    if isinstance(telemetry, Mapping):
        candidates.extend(
            [
                telemetry.get("support_sha256"),
                telemetry.get("expected_support_sha256"),
            ]
        )
    for value in candidates:
        if isinstance(value, str) and value:
            return value
    return ""


def _target_region_action_closure_blockers(
    oracle: Mapping[str, Any],
    *,
    archive_status: Mapping[str, Any],
    survival_status: Mapping[str, Any] | None = None,
) -> list[str]:
    blockers = [str(blocker) for blocker in oracle.get("blockers") or []]
    survival_status = survival_status if isinstance(survival_status, Mapping) else {}
    if archive_status.get("charged_meta_materialized"):
        blockers = [
            blocker
            for blocker in blockers
            if blocker
            not in {
                "hinerv_target_region_action_archive_meta_not_materialized",
                "target_region_action_archive_meta_not_materialized",
            }
        ]
    if archive_status.get("archive_zip_bytes_measured"):
        blockers = [
            blocker
            for blocker in blockers
            if blocker
            not in {
                "hinerv_target_region_action_archive_zip_byte_delta_missing",
                "target_region_action_archive_zip_byte_delta_missing",
            }
        ]
    blockers.extend(str(blocker) for blocker in archive_status.get("blockers") or [])
    if survival_status.get("parseback_survived"):
        blockers = [
            blocker
            for blocker in blockers
            if blocker
            not in {
                "hinerv_target_region_action_parseback_survival_missing",
                "target_region_action_parseback_survival_missing",
            }
        ]
    if survival_status.get("inflate_survived"):
        blockers = [
            blocker
            for blocker in blockers
            if blocker
            not in {
                "hinerv_target_region_action_inflate_survival_missing",
                "target_region_action_inflate_survival_missing",
            }
        ]
    if (
        (survival_status.get("partial_parseback_survived") or survival_status.get("partial_inflate_survived"))
        and not survival_status.get("same_row_parseback_inflate_survived")
    ):
        blockers.append("target_region_action_parseback_inflate_same_row_survival_missing")
    return blockers


def _pose_trusted(row: Mapping[str, Any]) -> bool:
    pose_guard = row.get("pose_guard") or {}
    nonrate = row.get("exact_nonrate") or {}
    delta = nonrate.get("delta_score_nonrate")
    accepted_delta = pose_guard.get("max_accepted_pose_output_delta_l2")
    pose_cap = pose_guard.get("max_pose_output_delta_l2")
    pose_cap_satisfied = (
        isinstance(accepted_delta, (int, float))
        and isinstance(pose_cap, (int, float))
        and float(accepted_delta) <= float(pose_cap)
    )
    return bool(
        pose_guard.get("available")
        and pose_guard.get("pose_input_contest_resolution")
        and pose_cap_satisfied
        and nonrate.get("pose_term_available")
        and isinstance(delta, (int, float))
        and float(delta) < 0.0
    )


def _pose_trusted_by_pair_local_servo_action_effect(
    rows: list[dict[str, Any]],
    *,
    action_id: str,
    blockers: list[str],
) -> bool:
    """Return true when the measured pair-local composite supplies pose trust.

    Older HiNeRV hard-birth receipts carried pose-guard fields directly on the
    live birth row.  The v6 PR95-like servo writes the stronger evidence as a
    same-action ActionEffect row: the frame-1 birth and frame-0 pose
    compensation are replayed as a composed action and admitted by exact
    nonlinear score.  Treat that as pose trust only when it proves both target
    support and Pose movement on the same action identity.
    """

    matches = _action_effect_rows_for_action(rows, action_id=action_id)
    saw_candidate = False
    candidate_blockers: list[str] = []
    for row in matches:
        if str(row.get("action_kind") or "") != "joint_line_search_composite":
            continue
        saw_candidate = True
        if str(row.get("exact_score_decision") or "") != "accept":
            candidate_blockers.append("pose_trust_pair_local_servo_exact_score_not_accepted")
            continue
        delta_nonrate = row.get("delta_score_nonrate")
        if not _finite_number(delta_nonrate) or float(delta_nonrate) >= 0.0:
            candidate_blockers.append("pose_trust_pair_local_servo_exact_nonrate_not_improved")
            continue
        old_pose = row.get("old_d_pose")
        new_pose = row.get("new_d_pose")
        pose_score_delta = row.get("pose_score_delta")
        pose_improved = (
            _finite_number(old_pose)
            and _finite_number(new_pose)
            and float(new_pose) < float(old_pose)
        ) or (_finite_number(pose_score_delta) and float(pose_score_delta) < 0.0)
        if not pose_improved:
            candidate_blockers.append("pose_trust_pair_local_servo_pose_not_improved")
            continue
        if not _positive_number(row.get("posenet_input_delta_linf_pair")):
            candidate_blockers.append("pose_trust_pair_local_servo_posenet_input_delta_missing")
            continue
        if not _target_support_positive(
            row,
            blocker="pose_trust_pair_local_servo_target_support_missing",
            not_positive_blocker="pose_trust_pair_local_servo_target_support_not_positive",
            blockers=candidate_blockers,
        ):
            continue
        target_to_wrong = row.get("target_to_wrong")
        if _finite_number(target_to_wrong) and float(target_to_wrong) > 0.0:
            candidate_blockers.append("pose_trust_pair_local_servo_target_spill_positive")
            continue
        if row.get("rejected_by_exact_score") is True:
            candidate_blockers.append("pose_trust_pair_local_servo_rejected_by_exact_score")
            continue
        if row.get("rejected_by_catastrophic_guard") is True:
            candidate_blockers.append("pose_trust_pair_local_servo_rejected_by_catastrophic_guard")
            continue
        return True

    if saw_candidate:
        blockers.extend(_dedupe(candidate_blockers))
    return False


def _survival_rows_for_action(
    rows: list[dict[str, Any]],
    *,
    action_id: str,
    surface: str,
    expected_support_sha256: str | None = None,
    blockers: list[str],
) -> dict[str, Any] | None:
    for row in rows:
        if str(row.get("surface") or "") != surface:
            continue
        if str(row.get("action_id") or "") != action_id:
            blockers.append("action_id_survival_mismatch")
            blockers.append(f"l4_survival_action_id_mismatch:{surface}")
            continue
        row_support_sha256 = _target_region_action_support_sha256(row)
        if expected_support_sha256:
            if not row_support_sha256:
                blockers.append(f"birth_survival_support_sha256_missing:{surface}")
                continue
            if row_support_sha256 != str(expected_support_sha256):
                blockers.append(f"birth_survival_support_sha256_mismatch:{surface}")
                continue
        if row.get("pose_compensation_required") is True and row.get("pose_compensation_survived") is not True:
            blockers.append(f"birth_survival_pose_compensation_not_survived:{surface}")
            continue
        if row.get("survived") is True:
            if _target_support_positive(
                row,
                blocker=f"birth_survival_target_support_missing:{surface}",
                not_positive_blocker=(
                    f"birth_survival_target_support_not_positive:{surface}"
                ),
                blockers=blockers,
            ):
                return row
            continue
        blockers.append(f"birth_not_survived:{surface}")
    return None


def _target_support_positive(
    row: Mapping[str, Any],
    *,
    blocker: str,
    not_positive_blocker: str,
    blockers: list[str],
) -> bool:
    surface = dict(row)
    transitions = row.get("argmax_transitions")
    if isinstance(transitions, Mapping):
        surface.update(transitions)
    support = receiver_surface_target_support_breakdown(surface)
    hard_won = support.get("receiver_surface_target_hard_won_count")
    net = support.get("receiver_surface_net_target_support_delta")
    if hard_won is None or net is None:
        blockers.append(blocker)
        return False
    if float(hard_won) <= 0.0 or float(net) <= 0.0:
        blockers.append(not_positive_blocker)
        return False
    return True


def _action_effect_rows_for_action(
    rows: list[dict[str, Any]],
    *,
    action_id: str,
) -> list[dict[str, Any]]:
    return [row for row in rows if str(row.get("action_id") or "") == action_id]


def _action_effect_support_sha256(row: Mapping[str, Any]) -> str:
    value = row.get("support_sha256")
    return value if isinstance(value, str) else ""


def _action_effect_has_receiver_survival(row: Mapping[str, Any]) -> bool:
    return (
        row.get("fakequant_survived") is True
        and row.get("parseback_survived") is True
        and row.get("inflate_survived") is True
    )


def _expected_hi_nerv_action_support_sha256(
    *,
    action_id: str,
    action_effect_rows: list[dict[str, Any]],
    wall_normal_branch_rows: list[dict[str, Any]],
    lowering_race_rows: list[dict[str, Any]],
    live_birth_row: Mapping[str, Any],
) -> str:
    candidates: list[str] = []
    live_support = _target_region_action_support_sha256(live_birth_row)
    if live_support:
        candidates.append(live_support)
    for row in _action_effect_rows_for_action(action_effect_rows, action_id=action_id):
        support_sha256 = _action_effect_support_sha256(row)
        if support_sha256:
            candidates.append(support_sha256)
    for row in wall_normal_branch_rows:
        action_ids = {str(value) for value in row.get("action_ids") or ()}
        if action_id not in action_ids:
            continue
        support_hashes = [
            str(value)
            for value in row.get("support_sha256s") or ()
            if isinstance(value, str) and value
        ]
        if row.get("same_support_sha256") is True and len(support_hashes) == 1:
            candidates.append(support_hashes[0])
    for row in lowering_race_rows:
        if str(row.get("action_id") or "") != action_id:
            continue
        support_sha256 = row.get("support_sha256")
        if isinstance(support_sha256, str) and support_sha256:
            candidates.append(support_sha256)
    unique = _dedupe(candidates)
    return unique[0] if len(unique) == 1 else ""


def _validated_action_effect_rows(
    rows: list[dict[str, Any]],
    *,
    blockers: list[str],
) -> list[dict[str, Any]]:
    valid: list[dict[str, Any]] = []
    for row in rows:
        status = validate_action_effect_payload(row)
        if status["passed"] is True:
            valid.append(row)
            continue
        row_id = str(row.get("action_id") or "unknown")
        for blocker in status["blockers"]:
            blocker = str(blocker)
            blockers.append(blocker)
            blockers.append(f"action_effect_invalid:{row_id}:{blocker}")
    return valid


def _require_hi_nerv_action_effect_evidence(
    rows: list[dict[str, Any]],
    *,
    action_id: str,
    expected_support_sha256: str | None = None,
    blockers: list[str],
) -> None:
    matches = _action_effect_rows_for_action(rows, action_id=action_id)
    if not matches:
        blockers.append("action_effect_missing")
        blockers.append(f"action_effect_v1_missing:{action_id}")
        return
    if any(not str(row.get("authority") or "").strip() for row in matches):
        blockers.append("action_effect_untyped_authority")
    if any(str(row.get("normalization_scope") or "") != NormalizationScope.BATCH_LOCAL.value for row in matches):
        blockers.append("normalization_scope_mismatch")
    if not any(row.get("fakequant_survived") is True for row in matches):
        blockers.append("action_effect_same_action_fakequant_survival_missing")
    if not any(row.get("parseback_survived") is True for row in matches):
        blockers.append("action_effect_same_action_parseback_survival_missing")
    if not any(row.get("inflate_survived") is True for row in matches):
        blockers.append("action_effect_same_action_inflate_survival_missing")
    receiver_rows = [row for row in matches if _action_effect_has_receiver_survival(row)]
    complete_receiver_rows = [
        row for row in receiver_rows if _action_effect_has_required_v6_fields(row)
    ]
    if not complete_receiver_rows:
        blockers.append("action_effect_same_action_same_support_receiver_survival_missing")
    proof_support_hashes = {
        _action_effect_support_sha256(row)
        for row in complete_receiver_rows
        if _action_effect_support_sha256(row)
    }
    if complete_receiver_rows and not proof_support_hashes:
        blockers.append("action_effect_same_action_same_support_sha256_missing")
    elif len(proof_support_hashes) > 1:
        blockers.append("action_effect_same_action_same_support_sha256_mismatch")
    if expected_support_sha256 and expected_support_sha256 not in proof_support_hashes:
        blockers.append("action_effect_same_action_same_support_sha256_mismatch")
        blockers.append(
            "action_effect_same_action_same_support_receiver_survival_missing"
        )
        complete_receiver_rows = [
            row
            for row in complete_receiver_rows
            if _action_effect_support_sha256(row) == expected_support_sha256
        ]
    best = next(
        iter(complete_receiver_rows),
        next((row for row in matches if _action_effect_has_required_v6_fields(row)), matches[0]),
    )
    for field in (
        "hard_won_count",
        "wrong_to_target",
        "net_target_support_delta",
        "uint8_changed_count_region",
        "seg_input_delta_linf_region",
        "posenet_input_delta_linf_pair",
    ):
        if not _positive_number(best.get(field)):
            blockers.append(f"action_effect_v1_field_missing_or_nonpositive:{field}")
    target_to_wrong = best.get("target_to_wrong")
    if (
        target_to_wrong is not None
        and _finite_number(target_to_wrong)
        and float(target_to_wrong) > 0.0
    ):
        blockers.append("action_effect_v1_spill_positive:target_to_wrong")
    delta_nonrate = best.get("delta_score_nonrate")
    if not _finite_number(delta_nonrate) or float(delta_nonrate) >= 0.0:
        blockers.append("action_effect_v1_exact_nonrate_not_improved")
    delta_total = best.get("delta_score_total")
    if delta_total is not None and (not _finite_number(delta_total) or float(delta_total) >= 0.0):
        blockers.append("action_effect_v1_exact_total_not_improved")
    for row in matches:
        old_bytes = row.get("old_bytes")
        new_bytes = row.get("new_bytes")
        if (
            _finite_number(old_bytes)
            and _finite_number(new_bytes)
            and int(new_bytes) != int(old_bytes)
            and not _finite_number(row.get("value_per_byte"))
        ):
            blockers.append("value_per_byte_missing")
            blockers.append(f"value_per_byte_ledger_missing:{action_id}")


def _require_hi_nerv_wall_normal_lift_evidence(
    rows: list[dict[str, Any]],
    *,
    action_id: str,
    blockers: list[str],
) -> None:
    matches = [row for row in rows if str(row.get("action_id") or "") == action_id]
    if not matches:
        blockers.append("target_region_wall_normal_lift_missing")
        blockers.append(f"target_region_wall_normal_lift_missing:{action_id}")
        return

    successful = False
    candidate_blockers: list[str] = []
    for row in matches:
        direct = row.get("direct_teacher")
        backend = row.get("backend_fit")
        direct = direct if isinstance(direct, Mapping) else {}
        backend = backend if isinstance(backend, Mapping) else {}
        selected = str(row.get("selected_next_operator") or "").strip()
        row_id = str(row.get("action_id") or action_id)
        if selected != "backend_fit_live":
            candidate_blockers.append(
                f"target_region_wall_normal_lift_selected_next_operator:{row_id}:{selected or 'missing'}"
            )
        if direct.get("crossed_target_wall") is not True:
            candidate_blockers.append(
                f"target_region_wall_normal_lift_direct_teacher_not_crossed:{row_id}"
            )
        if direct.get("teacher_is_true_wall_normal") is not True:
            candidate_blockers.append(
                f"target_region_wall_normal_lift_direct_teacher_not_true_wall_normal:{row_id}"
            )
        direct_decision = str(direct.get("exact_score_decision") or "").strip()
        if direct_decision not in {"accept", "accepted"}:
            candidate_blockers.append(
                f"target_region_wall_normal_lift_direct_teacher_exact_score_not_accepted:{row_id}"
            )
        direct_delta = direct.get("exact_delta_score_nonrate")
        if not _negative_number(direct_delta):
            candidate_blockers.append(
                f"target_region_wall_normal_lift_direct_teacher_nonnegative_delta:{row_id}"
            )
        if backend.get("realized_target_wall") is not True:
            candidate_blockers.append(
                f"target_region_wall_normal_lift_backend_not_realized:{row_id}"
            )
        backend_effect = backend.get("action_effect")
        backend_effect = backend_effect if isinstance(backend_effect, Mapping) else {}
        backend_effect_action_id = str(backend_effect.get("action_id") or "").strip()
        if backend_effect_action_id and backend_effect_action_id != row_id:
            candidate_blockers.append(
                f"target_region_wall_normal_lift_backend_action_id_mismatch:{row_id}:{backend_effect_action_id}"
            )
        if not _positive_number(backend.get("wrong_to_target_count")):
            candidate_blockers.append(
                f"target_region_wall_normal_lift_wrong_to_target_missing_or_nonpositive:{row_id}"
            )
        decision = str(backend.get("exact_score_decision") or "").strip()
        if decision not in {"accept", "accepted"}:
            candidate_blockers.append(
                f"target_region_wall_normal_lift_exact_score_not_accepted:{row_id}"
            )
        for blocker in row.get("blockers") or ():
            candidate_blockers.append(f"target_region_wall_normal_lift_blocker:{blocker}")
        if (
            selected == "backend_fit_live"
            and direct.get("crossed_target_wall") is True
            and direct.get("teacher_is_true_wall_normal") is True
            and direct_decision in {"accept", "accepted"}
            and _negative_number(direct_delta)
            and backend.get("realized_target_wall") is True
            and _positive_number(backend.get("wrong_to_target_count"))
            and (not backend_effect_action_id or backend_effect_action_id == row_id)
            and decision in {"accept", "accepted"}
            and not row.get("blockers")
        ):
            successful = True
    if not successful:
        blockers.append("target_region_wall_normal_lift_not_backend_realized")
        blockers.extend(_dedupe(candidate_blockers))


def _require_hi_nerv_wall_normal_branch_receipt(
    rows: list[dict[str, Any]],
    *,
    action_id: str,
    blockers: list[str],
) -> None:
    matches = [
        row
        for row in rows
        if action_id in {str(value) for value in row.get("action_ids") or ()}
        or str(row.get("first_failing_action_id") or "") == action_id
    ]
    if not matches:
        blockers.append("target_region_wall_normal_branch_receipt_missing")
        blockers.append(f"target_region_wall_normal_branch_receipt_missing:{action_id}")
        return

    candidate_blockers: list[str] = []
    for row in matches:
        row_action = str(row.get("first_failing_action_id") or action_id)
        if not _positive_number(row.get("branch_count")):
            candidate_blockers.append(
                f"target_region_wall_normal_branch_receipt_empty:{row_action}"
            )
        if row.get("same_action_id") is not True:
            candidate_blockers.append(
                f"target_region_wall_normal_branch_receipt_action_id_mismatch:{row_action}"
            )
        if row.get("same_support_sha256") is not True:
            candidate_blockers.append(
                f"target_region_wall_normal_branch_receipt_support_mismatch:{row_action}"
            )
        required = row.get("support_required_count")
        executable = row.get("support_executable_count")
        if _finite_number(required) and _finite_number(executable) and int(executable) < int(required):
            candidate_blockers.append(
                f"target_region_wall_normal_branch_receipt_support_not_executable:{row_action}"
            )
        first_failing = str(row.get("first_failing_surface") or "").strip()
        if first_failing not in {"", "none", SIDECAR_FALLBACK_ACCEPTED}:
            candidate_blockers.append(
                f"target_region_wall_normal_branch_receipt_failed_surface:{row_action}:{first_failing}"
            )
        for blocker in row.get("blockers") or ():
            candidate_blockers.append(f"target_region_wall_normal_branch_receipt_blocker:{blocker}")
        if not candidate_blockers:
            return

    blockers.append("target_region_wall_normal_branch_receipt_not_accepted")
    blockers.extend(_dedupe(candidate_blockers))


def _require_hi_nerv_four_arm_action_effect_evidence(
    rows: list[dict[str, Any]],
    *,
    action_id: str,
    blockers: list[str],
) -> None:
    matches = [
        row
        for row in _action_effect_rows_for_action(rows, action_id=action_id)
        if str(row.get("arm") or "").strip()
    ]
    if not matches:
        blockers.append("action_effect_four_arm_missing")
        for arm in HINERV_REQUIRED_FOUR_ARM_ACTION_KINDS:
            blockers.append(f"action_effect_four_arm_missing:{arm}")
        return

    by_arm: dict[str, dict[str, Any]] = {}
    for row in matches:
        arm = str(row.get("arm") or "").strip().upper()
        by_arm.setdefault(arm, row)

    for arm, expected_kind in HINERV_REQUIRED_FOUR_ARM_ACTION_KINDS.items():
        row = by_arm.get(arm)
        if row is None:
            blockers.append(f"action_effect_four_arm_missing:{arm}")
            continue
        if str(row.get("action_kind") or "") != expected_kind:
            blockers.append(f"action_effect_four_arm_kind_mismatch:{arm}")
        if not _finite_number(row.get("delta_score_nonrate")):
            blockers.append(f"action_effect_four_arm_delta_score_nonrate_missing:{arm}")
        if not _finite_number(row.get("old_d_seg")) or not _finite_number(row.get("new_d_seg")):
            blockers.append(f"action_effect_four_arm_seg_endpoint_missing:{arm}")
        if not _finite_number(row.get("old_d_pose")) or not _finite_number(row.get("new_d_pose")):
            blockers.append(f"action_effect_four_arm_pose_endpoint_missing:{arm}")
        exact_decision = str(row.get("exact_score_decision") or "")
        if exact_decision not in {"accept", "reject"}:
            blockers.append(f"action_effect_four_arm_exact_decision_missing:{arm}")
        if row.get("raw_cap_decision") in (None, ""):
            blockers.append(f"action_effect_four_arm_raw_cap_decision_missing:{arm}")
        if row.get("catastrophic_guard_decision") in (None, ""):
            blockers.append(f"action_effect_four_arm_catastrophic_decision_missing:{arm}")
        if row.get("would_accept_exact_score_if_raw_cap_disabled") is None:
            blockers.append(f"action_effect_four_arm_raw_cap_counterfactual_missing:{arm}")
        if row.get("rejected_by_raw_cap") is None:
            blockers.append(f"action_effect_four_arm_rejected_by_raw_cap_missing:{arm}")
        if row.get("rejected_by_exact_score") is None:
            blockers.append(f"action_effect_four_arm_rejected_by_exact_score_missing:{arm}")
        if row.get("rejected_by_catastrophic_guard") is None:
            blockers.append(f"action_effect_four_arm_rejected_by_catastrophic_guard_missing:{arm}")


def _require_hi_nerv_lowering_race_evidence(
    rows: list[dict[str, Any]],
    *,
    action_id: str,
    blockers: list[str],
) -> None:
    """Require a same-action evaluator-action lowering race before long runs.

    Four-arm ActionEffect rows prove the PR95-like ablation was measured.  The
    lowering race is the admission layer that decides whether the measured atom
    can actually survive as backend, sidecar, composite, or semantic primitive
    under exact nonlinear score.  Missing or failed race evidence means the
    smoke is still diagnostic, not a long-run launch proof.
    """

    matches = [row for row in rows if str(row.get("action_id") or "") == action_id]
    if not matches:
        blockers.append("evaluator_action_lowering_race_missing")
        blockers.append(f"evaluator_action_lowering_race_missing:{action_id}")
        return

    candidate_blockers: list[str] = []
    for row in matches:
        verdict = row.get("verdict")
        verdict = verdict if isinstance(verdict, Mapping) else {}
        row_id = str(row.get("action_id") or verdict.get("action_id") or action_id)
        schema = str(row.get("schema") or "")
        if schema == EVALUATOR_ACTION_LOWERING_RACE_SCHEMA:
            support_identity = row.get("support_identity")
            if not isinstance(support_identity, Mapping):
                candidate_blockers.append(
                    f"evaluator_action_lowering_race_support_identity_missing:{row_id}"
                )
                continue
            support_failure = str(support_identity.get("failure") or "").strip()
            if support_failure:
                candidate_blockers.append(
                    f"evaluator_action_lowering_race_support_identity_failed:{row_id}:{support_failure}"
                )
                continue
            if support_identity.get("all_candidates_same_support") is not True:
                candidate_blockers.append(
                    f"evaluator_action_lowering_race_support_mismatch:{row_id}"
                )
                continue
        candidate_count = _lowering_race_candidate_count(row)
        if (
            schema == EVALUATOR_ACTION_LOWERING_RACE_SCHEMA
            or candidate_count is not None
        ) and (candidate_count is None or candidate_count <= 0):
            candidate_blockers.append(
                f"evaluator_action_lowering_race_candidate_count_not_positive:{row_id}"
            )
            continue
        if schema in {
            EVALUATOR_ACTION_LOWERING_RACE_SCHEMA,
            HI_NERV_TARGET_REGION_ACTION_LOWERING_RACE_SCHEMA,
        }:
            target_accounting = row.get("target_accounting")
            if not isinstance(target_accounting, Mapping):
                candidate_blockers.append(
                    f"evaluator_action_lowering_race_target_accounting_missing:{row_id}"
                )
                continue
            missing_targets = _lowering_race_missing_targets(row)
            if missing_targets:
                candidate_blockers.append(
                    "evaluator_action_lowering_race_targets_missing:"
                    f"{row_id}:{','.join(missing_targets)}"
                )
                continue
        if row.get("same_support_as_direct_teacher") is False:
            candidate_blockers.append(
                f"evaluator_action_lowering_race_support_mismatch:{row_id}"
            )
            continue
        candidate_shape_blockers = _lowering_race_candidate_shape_blockers(row, row_id=row_id)
        if candidate_shape_blockers:
            candidate_blockers.extend(candidate_shape_blockers)
            continue
        best_lowering = str(row.get("best_lowering") or verdict.get("best_lowering") or "none")
        if best_lowering in {"", "none", "discard"}:
            candidate_blockers.append(
                f"evaluator_action_lowering_race_no_viable_lowering:{row_id}"
            )
        first_failing = str(
            row.get("first_failing_surface")
            or verdict.get("first_failing_surface")
            or ""
        )
        if first_failing not in {"", "none"}:
            candidate_blockers.append(
                f"evaluator_action_lowering_race_failed_surface:{row_id}:{first_failing}"
            )
            continue
        if best_lowering in {"", "none", "discard"}:
            continue
        completion_blockers = _lowering_race_completion_status_blockers(
            row,
            row_id=row_id,
            best_lowering=best_lowering,
        )
        if completion_blockers:
            candidate_blockers.extend(completion_blockers)
            continue
        authority = str(verdict.get("authority") or "").strip()
        if not authority or authority == "none":
            candidate_blockers.append(
                f"evaluator_action_lowering_race_authority_missing:{row_id}"
            )
            continue
        delta_total = verdict.get("delta_score_total")
        if not _negative_number(delta_total):
            candidate_blockers.append(
                f"evaluator_action_lowering_race_exact_total_not_improved:{row_id}"
            )
            continue
        delta_nonrate = verdict.get("delta_score_nonrate")
        if delta_nonrate is not None and not _finite_number(delta_nonrate):
            candidate_blockers.append(
                f"evaluator_action_lowering_race_nonrate_delta_invalid:{row_id}"
            )
            continue
        return

    blockers.append("evaluator_action_lowering_race_not_accepted")
    blockers.extend(_dedupe(candidate_blockers))


def _lowering_race_completion_status_blockers(
    row: Mapping[str, Any],
    *,
    row_id: str,
    best_lowering: str,
) -> list[str]:
    verdict = row.get("verdict") if isinstance(row.get("verdict"), Mapping) else {}
    backend_complete = row.get(
        "backend_realization_complete",
        verdict.get("backend_realization_complete"),
    )
    sidecar_complete = row.get(
        "sidecar_lowering_complete",
        verdict.get("sidecar_lowering_complete"),
    )
    blockers: list[str] = []
    if not isinstance(backend_complete, bool):
        blockers.append(
            f"evaluator_action_lowering_race_backend_realization_status_missing:{row_id}"
        )
    elif backend_complete is not True:
        blockers.append(
            f"evaluator_action_lowering_race_backend_realization_incomplete:{row_id}"
        )
    if best_lowering == "byte_priced_sidecar":
        if not isinstance(sidecar_complete, bool):
            blockers.append(
                f"evaluator_action_lowering_race_sidecar_lowering_status_missing:{row_id}"
            )
        elif sidecar_complete is not True:
            blockers.append(
                f"evaluator_action_lowering_race_sidecar_lowering_incomplete:{row_id}"
            )
    return blockers


def _lowering_race_candidate_shape_blockers(row: Mapping[str, Any], *, row_id: str) -> list[str]:
    candidates = row.get("lowering_candidates")
    if not isinstance(candidates, list):
        return []
    support_sha256 = str(row.get("support_sha256") or "").strip()
    out: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            out.append(f"evaluator_action_lowering_race_candidate_malformed:{row_id}")
            continue
        candidate_action_id = str(candidate.get("action_id") or "").strip()
        if candidate_action_id and candidate_action_id != row_id:
            out.append(f"evaluator_action_lowering_race_action_id_mismatch:{row_id}")
        if support_sha256:
            candidate_support = str(candidate.get("support_sha256") or "").strip()
            if candidate_support and candidate_support != support_sha256:
                out.append(f"evaluator_action_lowering_race_support_mismatch:{row_id}")
        target = str(candidate.get("lowering_target") or "").strip()
        if target == "byte_priced_sidecar":
            if not str(candidate.get("support_encoding") or "").strip():
                out.append(f"evaluator_action_lowering_race_selected_codec_missing:{row_id}")
            if candidate.get("support_encoded_bytes") is None:
                out.append(f"evaluator_action_lowering_race_byte_accounting_missing:{row_id}")
        if candidate.get("action_payload_bytes") is None or candidate.get("metadata_bytes") is None:
            out.append(f"evaluator_action_lowering_race_byte_accounting_missing:{row_id}")
        if candidate.get("viable") is True:
            authority = str(candidate.get("authority") or "").strip()
            if not authority or authority == "none":
                out.append(f"evaluator_action_lowering_race_authority_missing:{row_id}")
        failure = str(candidate.get("first_failing_surface") or "").strip()
        if failure == "BYTE_ACCOUNTING_MISSING":
            out.append(f"evaluator_action_lowering_race_byte_accounting_missing:{row_id}")
        elif failure == "PARSEBACK_FAILED":
            out.append(f"evaluator_action_lowering_race_parseback_missing:{row_id}")
        elif failure == "INFLATE_FAILED":
            out.append(f"evaluator_action_lowering_race_inflate_missing:{row_id}")
        elif failure == "SUPPORT_IDENTITY_MISMATCH":
            out.append(f"evaluator_action_lowering_race_support_mismatch:{row_id}")
    return _dedupe(out)


def _lowering_race_candidate_count(row: Mapping[str, Any]) -> int | None:
    count = row.get("candidate_count")
    if count is not None:
        if not _finite_number(count):
            return 0
        return int(count)
    candidates = row.get("lowering_candidates")
    if isinstance(candidates, list):
        return len(candidates)
    return None


def _lowering_race_missing_targets(row: Mapping[str, Any]) -> list[str]:
    accounting = row.get("target_accounting")
    if isinstance(accounting, Mapping):
        expected = accounting.get("expected_targets")
        declared_expected = (
            {str(value) for value in expected if str(value) in EVALUATOR_ACTION_LOWERING_TARGETS}
            if isinstance(expected, list)
            else set()
        )
        missing_from_expected = [
            target
            for target in EVALUATOR_ACTION_LOWERING_TARGETS
            if target not in declared_expected
        ] if declared_expected else []
        missing = accounting.get("missing_targets")
        if isinstance(missing, list):
            producer_missing = [
                str(value)
                for value in missing
                if str(value) in EVALUATOR_ACTION_LOWERING_TARGETS
            ]
            return _dedupe([*missing_from_expected, *producer_missing])
        if accounting.get("all_targets_accounted") is True:
            return missing_from_expected
    candidates = row.get("lowering_candidates")
    if not isinstance(candidates, list):
        return list(EVALUATOR_ACTION_LOWERING_TARGETS)
    present = {
        str(candidate.get("lowering_target") or "")
        for candidate in candidates
        if isinstance(candidate, Mapping)
    }
    return [
        target
        for target in EVALUATOR_ACTION_LOWERING_TARGETS
        if target not in present
    ]


def _representative_coverage_ok(
    rows: list[dict[str, Any]],
    *,
    blockers: list[str],
) -> bool:
    if not rows:
        blockers.append("representative_region_coverage_missing")
        return False
    candidate_blockers: list[str] = []
    for row in rows:
        row_blockers: list[str] = []
        if row.get("passed") is not True:
            row_blockers.append("representative_region_coverage_not_passed")
        region_classes = _positive_int_or_none(row.get("region_classes_covered"))
        distinct_classes = _positive_int_or_none(row.get("distinct_classes_accepted"))
        if region_classes is None:
            row_blockers.append("representative_region_coverage_region_classes_missing")
        if distinct_classes is None:
            row_blockers.append("representative_region_coverage_distinct_classes_missing")
        min_regions = _positive_int_or_none(row.get("min_distinct_class_size_buckets"))
        min_classes = _positive_int_or_none(row.get("min_distinct_classes"))
        if min_regions is not None and (region_classes is None or region_classes < min_regions):
            row_blockers.append("representative_region_coverage_region_classes_below_threshold")
        if min_classes is not None and (distinct_classes is None or distinct_classes < min_classes):
            row_blockers.append("representative_region_coverage_distinct_classes_below_threshold")
        accepted_count = row.get("accepted_count")
        accepted_count_int = _positive_int_or_none(accepted_count)
        if accepted_count_int is None:
            row_blockers.append("representative_region_coverage_accepted_count_not_positive")
        outcomes = row.get("outcomes")
        accepted_outcome_count = 0
        accepted_outcome_buckets: set[tuple[int, str]] = set()
        accepted_outcome_classes: set[int] = set()
        accepted_outcome_size_classes: set[str] = set()
        if not isinstance(outcomes, list) or not outcomes:
            row_blockers.append("representative_region_coverage_outcomes_missing")
        else:
            for outcome in outcomes:
                if not isinstance(outcome, Mapping):
                    row_blockers.append("representative_region_coverage_outcome_invalid")
                    continue
                if str(outcome.get("outcome") or "") != "birth_accepted":
                    continue
                class_index = _nonnegative_int_or_none(outcome.get("class_index"))
                size_class = str(outcome.get("size_class") or "").strip()
                if class_index is None or not size_class:
                    row_blockers.append("representative_region_coverage_accepted_outcome_missing_bucket")
                    continue
                accepted_outcome_count += 1
                accepted_outcome_buckets.add((class_index, size_class))
                accepted_outcome_classes.add(class_index)
                accepted_outcome_size_classes.add(size_class)
        if accepted_count_int is not None and accepted_outcome_count != accepted_count_int:
            row_blockers.append("representative_region_coverage_accepted_count_mismatch")
        if region_classes is not None and len(accepted_outcome_buckets) < region_classes:
            row_blockers.append("representative_region_coverage_outcomes_below_region_classes")
        if distinct_classes is not None and len(accepted_outcome_classes) < distinct_classes:
            row_blockers.append("representative_region_coverage_outcomes_below_distinct_classes")
        distinct_size_classes = _positive_int_or_none(row.get("distinct_size_classes_accepted"))
        if distinct_size_classes is not None and len(accepted_outcome_size_classes) < distinct_size_classes:
            row_blockers.append("representative_region_coverage_outcomes_below_distinct_size_classes")
        accepted_buckets = _coverage_bucket_pairs(row.get("accepted_class_size_buckets"))
        if not accepted_buckets:
            row_blockers.append("representative_region_coverage_accepted_buckets_missing")
        elif accepted_outcome_buckets and accepted_buckets != accepted_outcome_buckets:
            row_blockers.append("representative_region_coverage_accepted_buckets_mismatch")
        all_buckets = _coverage_bucket_pairs(row.get("all_class_size_buckets"))
        if not all_buckets:
            row_blockers.append("representative_region_coverage_all_buckets_missing")
        elif accepted_buckets and not accepted_buckets.issubset(all_buckets):
            row_blockers.append("representative_region_coverage_all_buckets_missing_accepted_bucket")
        if not row_blockers:
            return True
        candidate_blockers.extend(row_blockers)
    blockers.extend(_dedupe(candidate_blockers))
    return False


def _action_effect_has_required_v6_fields(row: Mapping[str, Any]) -> bool:
    return all(
        _positive_number(row.get(field))
        for field in (
            "hard_won_count",
            "wrong_to_target",
            "net_target_support_delta",
            "uint8_changed_count_region",
            "seg_input_delta_linf_region",
            "posenet_input_delta_linf_pair",
        )
    )


def _parseback_selection_contract_ok(rows: list[dict[str, Any]]) -> bool:
    for row in rows:
        if row.get("parseback_selection_required") is not True:
            continue
        if row.get("archive_parseback_axis_required") is not True:
            continue
        if row.get("live_only_improvement_is_false_authority") is not True:
            continue
        if row.get("fail_closed_on_axis_divergence") is not True:
            continue
        authority_order = row.get("selection_authority_order")
        if isinstance(authority_order, list) and "archive_parseback" in {str(value) for value in authority_order}:
            return True
    return False


def _source_qualified_metrics_ok(
    rows: list[dict[str, Any]],
    *,
    family: str,
) -> bool:
    accepted_family_values = {family, "hi_nerv" if family == "hinerv" else family}
    for row in rows:
        row_family = str(row.get("family") or family).strip().lower().replace("-", "_")
        if row_family not in accepted_family_values:
            continue
        if row.get("source_qualified") is not True:
            continue
        sources = [
            row.get("metric_source"),
            row.get("canonical_score_source"),
            row.get("seg_metric_source"),
            row.get("pose_metric_source"),
        ]
        if any(isinstance(value, str) and value.strip() for value in sources):
            return True
    return False


def _source_qualified_readiness_report_ok(
    rows: list[dict[str, Any]],
    *,
    family: str,
) -> bool:
    accepted_family_values = {family, "hi_nerv" if family == "hinerv" else family}
    for row in rows:
        row_family = str(row.get("family") or family).strip().lower().replace("-", "_")
        if row_family not in accepted_family_values:
            continue
        teacher = row.get("teacher_gate")
        receiver = row.get("receiver_surface_identity_gate")
        if not isinstance(teacher, Mapping) or not isinstance(receiver, Mapping):
            continue
        if bool(teacher.get("mock_scorer_teacher_allowed")):
            continue
        if bool(teacher.get("unscored_research_smoke_enabled")):
            continue
        seg_requested = bool(
            teacher.get("real_segnet_teacher_requested")
            or teacher.get("direct_live_segnet_requested")
        )
        pose_requested = bool(
            teacher.get("real_posenet_teacher_requested")
            or teacher.get("direct_live_posenet_requested")
        )
        if not (seg_requested and pose_requested):
            continue
        if not bool(receiver.get("archive_identity_present")):
            continue
        if not bool(receiver.get("direct_receiver_parseback_present")):
            continue
        if bool(receiver.get("archive_sha256_mismatch")):
            continue
        if not bool(receiver.get("candidate_cache_manifest_bound")):
            continue
        direct_live_segnet = row.get("direct_live_segnet_gate")
        if bool(teacher.get("direct_live_segnet_requested")) and not _mapping_has_finite_metric(
            direct_live_segnet
        ):
            continue
        direct_live_pose = row.get("direct_live_posenet_gate")
        generic_pose = row.get("posenet_distill_gate")
        if bool(teacher.get("direct_live_posenet_requested")):
            pose_has_metric = _mapping_has_finite_metric(direct_live_pose)
        else:
            pose_has_metric = _mapping_has_finite_metric(generic_pose)
        if not pose_has_metric:
            continue
        return True
    return False


def _mapping_has_finite_metric(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    metrics = value.get("metrics")
    if not isinstance(metrics, Mapping):
        return False
    return any(_finite_number(item) for item in metrics.values())


def _finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(parsed)


def _positive_number(value: Any) -> bool:
    return _finite_number(value) and float(value) > 0.0


def _negative_number(value: Any) -> bool:
    return _finite_number(value) and float(value) < 0.0


def _positive_int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _nonnegative_int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _coverage_bucket_pairs(value: Any) -> set[tuple[int, str]] | None:
    if not isinstance(value, list):
        return None
    pairs: set[tuple[int, str]] = set()
    for item in value:
        if not isinstance(item, list | tuple) or len(item) != 2:
            return None
        class_index = _nonnegative_int_or_none(item[0])
        size_class = str(item[1] or "").strip()
        if class_index is None or not size_class:
            return None
        pairs.add((class_index, size_class))
    return pairs


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def evaluate_nerv_long_run_launch_gate(
    *,
    family: str,
    run_root: str | Path,
    frontier_pointer: str | Path | None = None,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    """Return the fail-closed machine verdict for one family's long run."""

    family = _canonical_family(family)
    if family not in {"hinerv", "snerv"}:
        raise NervLongRunLaunchGateError(f"family must be one of {SUPPORTED_FAMILIES}; got {family!r}")
    root = Path(run_root).expanduser().resolve(strict=False)
    if not root.is_dir():
        raise NervLongRunLaunchGateError(f"run_root is not a directory: {root}")
    pointer_path = Path(frontier_pointer).expanduser().resolve(strict=False) if frontier_pointer is not None else None
    now = now_utc or datetime.now(UTC)

    blockers: list[str] = []
    warnings: list[str] = []
    evidence_index: dict[str, list[str]] = {}
    highest_level = "none"

    pointer = _frontier_pointer_status(pointer_path, now_utc=now)
    if not pointer["present"]:
        blockers.append("frontier_pointer_missing")
    elif not pointer["fresh"]:
        blockers.append("frontier_pointer_stale")

    if family == "hinerv":
        birth_rows = _collect_schema_rows(root, BIRTH_RECEIPT_SCHEMA, index=evidence_index, blockers=blockers)
        survival_rows = _collect_schema_rows(root, BIRTH_SURVIVAL_SCHEMA, index=evidence_index, blockers=blockers)
        hysteresis_rows = _collect_schema_rows(root, BIRTH_HYSTERESIS_SCHEMA, index=evidence_index, blockers=blockers)
        coverage_rows = _collect_schema_rows(
            root,
            REPRESENTATIVE_COVERAGE_SCHEMA,
            index=evidence_index,
            blockers=blockers,
        )
        action_effect_rows = _collect_schema_rows(root, ACTION_EFFECT_V1_SCHEMA, index=evidence_index, blockers=blockers)
        wall_normal_lift_rows = _collect_schema_rows(
            root,
            TARGET_REGION_WALL_NORMAL_LIFT_SCHEMA,
            index=evidence_index,
            blockers=blockers,
        )
        wall_normal_branch_rows = _collect_schema_rows(
            root,
            WALL_NORMAL_BRANCH_RECEIPT_SCHEMA,
            index=evidence_index,
            blockers=blockers,
        )
        parseback_contract_rows = _collect_schema_rows(
            root,
            ARCHIVE_PARSEBACK_SELECTION_CONTRACT_SCHEMA,
            index=evidence_index,
            blockers=blockers,
        )
        source_metric_rows = _collect_schema_rows(
            root,
            SOURCE_QUALIFIED_METRICS_SCHEMA,
            index=evidence_index,
            blockers=blockers,
        )
        readiness_rows = _collect_schema_rows(
            root,
            HI_NERV_SHORT_SCORER_SMOKE_READINESS_SCHEMA,
            index=evidence_index,
            blockers=blockers,
        )
        lowering_race_rows = _collect_schema_rows(
            root,
            HI_NERV_TARGET_REGION_ACTION_LOWERING_RACE_SCHEMA,
            index=evidence_index,
            blockers=blockers,
        )
        lowering_race_rows.extend(
            _collect_schema_rows(
                root,
                EVALUATOR_ACTION_LOWERING_RACE_SCHEMA,
                index=evidence_index,
                blockers=blockers,
            )
        )
        action_effect_rows = _validated_action_effect_rows(action_effect_rows, blockers=blockers)
        if not _parseback_selection_contract_ok(parseback_contract_rows):
            blockers.append("archive_parseback_selection_contract_missing")
        source_qualified = _source_qualified_metrics_ok(
            source_metric_rows,
            family=family,
        ) or _source_qualified_readiness_report_ok(
            readiness_rows,
            family=family,
        )
        if not source_qualified:
            blockers.append("source_qualified_metrics_missing")
        live = _accepted_live_birth(birth_rows, blockers=blockers)
        if live is None:
            unclosed_oracle = _positive_unclosed_masked_oracle_birth(birth_rows)
            if unclosed_oracle is None:
                if birth_rows:
                    blockers.append("real_video_birth_receipt_not_accepted")
                    blockers.extend(_rejected_live_birth_receipt_blockers(birth_rows))
                else:
                    blockers.append("real_video_birth_receipt_missing")
            else:
                archive_status = _target_region_action_archive_closure_status(
                    root,
                    index=evidence_index,
                )
                survival_status = _target_region_action_survival_status(
                    root,
                    action_id=str(unclosed_oracle.get("action_id") or ""),
                    support_sha256=_target_region_action_support_sha256(unclosed_oracle),
                    index=evidence_index,
                )
                blockers.append("real_video_birth_receipt_archive_unclosed")
                blockers.extend(
                    _target_region_action_closure_blockers(
                        unclosed_oracle,
                        archive_status=archive_status,
                        survival_status=survival_status,
                    )
                )
        else:
            highest_level = "L2"
            action_id = str(live.get("action_id"))
            expected_support_sha256 = _expected_hi_nerv_action_support_sha256(
                action_id=action_id,
                action_effect_rows=action_effect_rows,
                wall_normal_branch_rows=wall_normal_branch_rows,
                lowering_race_rows=lowering_race_rows,
                live_birth_row=live,
            )
            if _pose_trusted(live) or _pose_trusted_by_pair_local_servo_action_effect(
                action_effect_rows,
                action_id=action_id,
                blockers=blockers,
            ):
                highest_level = "L3"
            else:
                blockers.append("pose_trusted_birth_receipt_missing")
            _require_hi_nerv_action_effect_evidence(
                action_effect_rows,
                action_id=action_id,
                expected_support_sha256=expected_support_sha256,
                blockers=blockers,
            )
            _require_hi_nerv_wall_normal_lift_evidence(
                wall_normal_lift_rows,
                action_id=action_id,
                blockers=blockers,
            )
            _require_hi_nerv_wall_normal_branch_receipt(
                wall_normal_branch_rows,
                action_id=action_id,
                blockers=blockers,
            )
            _require_hi_nerv_four_arm_action_effect_evidence(
                action_effect_rows,
                action_id=action_id,
                blockers=blockers,
            )
            _require_hi_nerv_lowering_race_evidence(
                lowering_race_rows,
                action_id=action_id,
                blockers=blockers,
            )
            l4_ok = highest_level == "L3"
            for surface in SURVIVAL_SURFACES_L4:
                row = _survival_rows_for_action(
                    survival_rows,
                    action_id=action_id,
                    surface=surface,
                    expected_support_sha256=expected_support_sha256,
                    blockers=blockers,
                )
                if row is None:
                    blockers.append(f"birth_survival_receipt_missing:{surface}")
                    l4_ok = False
            hysteresis = next(
                (
                    row
                    for row in hysteresis_rows
                    if str(row.get("action_id") or "") == action_id and row.get("passed") is True
                ),
                None,
            )
            if hysteresis is None:
                blockers.append("hysteresis_receipt_missing")
                l4_ok = False
            if l4_ok:
                highest_level = "L4"
            l5_ok = l4_ok
            inflate_row = _survival_rows_for_action(
                survival_rows,
                action_id=action_id,
                surface=SURVIVAL_SURFACE_L5,
                expected_support_sha256=expected_support_sha256,
                blockers=blockers,
            )
            if inflate_row is None:
                blockers.append(f"birth_survival_receipt_missing:{SURVIVAL_SURFACE_L5}")
                l5_ok = False
            if not _representative_coverage_ok(coverage_rows, blockers=blockers):
                warnings.append("birth_at_init_only")
                l5_ok = False
            if l5_ok:
                highest_level = "L5"
            delta_bytes = int(live.get("runtime_sidecar_bytes") or 0)
            if delta_bytes != 0:
                blockers.append("value_per_byte_ledger_missing")
    else:  # snerv
        _collect_schema_rows(
            root,
            SNERV_SOURCE_FORWARD_SCHEMA,
            index=evidence_index,
            blockers=blockers,
        )
        proof_rows: list[dict[str, Any]] = []
        for path, payload in _iter_evidence_payloads(root):
            hits = find_snerv_source_forward_proof_rows(payload)
            if not hits:
                continue
            try:
                require_no_truthy_authority_fields(payload, context=str(path))
            except Exception:
                blockers.append(f"evidence_truthy_authority:{path.name}")
                continue
            proof_rows.extend(hits)
            evidence_index.setdefault(
                SNERV_SOURCE_FORWARD_PROOF_ACTION_EFFECT_SCHEMA,
                [],
            ).append(path.as_posix())
        valid_rows: list[dict[str, Any]] = []
        invalid_blockers: list[str] = []
        for row in proof_rows:
            status = validate_snerv_source_forward_proof_action_effect(row)
            if status["passed"] is True:
                valid_rows.append(row)
            else:
                invalid_blockers.extend(
                    f"snerv_source_forward_proof_invalid:{blocker}"
                    for blocker in status["blockers"]
                )
        if not valid_rows:
            blockers.append("snerv_full_source_forward_parity_missing")
            if proof_rows:
                blockers.extend(invalid_blockers)
        else:
            highest_level = "L3"
        bitflip = next(
            (
                row.get("destructive_payload_bit_flip")
                for row in valid_rows
                if isinstance(row.get("destructive_payload_bit_flip"), Mapping)
            ),
            None,
        )
        if bitflip is None or not bitflip.get("first_failed_tensor"):
            blockers.append("snerv_payload_bitflip_falsification_missing")
        elif valid_rows:
            highest_level = "L4"

    approved = not blockers and highest_level == ("L5" if family == "hinerv" else "L4")
    missing_evidence_keys = sorted(set(blockers))
    verdict = {
        "schema": NERV_LONG_RUN_LAUNCH_GATE_SCHEMA,
        "family": family,
        "run_root": root.as_posix(),
        "approved": bool(approved),
        "highest_level": highest_level,
        "blocking_evidence": missing_evidence_keys,
        "missing_evidence_keys": missing_evidence_keys,
        "non_blocking_warnings": sorted(set(warnings)),
        "evidence_index": evidence_index,
        "frontier_pointer": pointer,
        "evaluated_at_utc": now.isoformat(),
        "authority": "[planning/launch-gate]",
    }
    verdict.update(PROXY_FALSE_AUTHORITY_FIELDS)
    return verdict


def _canonical_family(value: Any) -> str:
    family = str(value or "").strip().lower().replace("-", "_")
    if family == "hi_nerv":
        return "hinerv"
    return family
