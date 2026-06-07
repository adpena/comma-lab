# SPDX-License-Identifier: MIT
"""Minimal evaluator-action lowering race over ActionEffect rows."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tac.analysis.action_commutator import build_commutator_ledger
from tac.analysis.action_effect import ActionEffect

LOWERING_RACE_SCHEMA = "tac.evaluator_action_lowering_race.v1"
LOWERING_VERDICT_SCHEMA = "tac.evaluator_action_lowering_verdict.v1"
LOWERING_CANDIDATE_SCHEMA = "tac.evaluator_action_lowering_candidate.v1"

LOWERING_TARGETS = (
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

DIRECT_TEACHER_NO_WALL_CROSS = "DIRECT_TEACHER_NO_WALL_CROSS"
BACKEND_REALIZATION_FAILED = "BACKEND_REALIZATION_FAILED"
SIDECAR_TOO_EXPENSIVE = "SIDECAR_TOO_EXPENSIVE"
COMPOSITE_NOT_MEASURED = "COMPOSITE_NOT_MEASURED"
SEMANTIC_PRIMITIVE_MISSING = "SEMANTIC_PRIMITIVE_MISSING"
PAIR_LOCAL_LATENT_NOT_MEASURED = "PAIR_LOCAL_LATENT_NOT_MEASURED"
FRAME0_POSE_COMPENSATION_NOT_MEASURED = "FRAME0_POSE_COMPENSATION_NOT_MEASURED"
FRAME1_SEG_WALL_CROSSING_NOT_MEASURED = "FRAME1_SEG_WALL_CROSSING_NOT_MEASURED"
SNERV_SOURCE_STATE_NOT_MEASURED = "SNERV_SOURCE_STATE_NOT_MEASURED"
BYTE_ENTROPY_REWRITE_NOT_MEASURED = "BYTE_ENTROPY_REWRITE_NOT_MEASURED"
DISCARD_NOT_DECLARED = "DISCARD_NOT_DECLARED"
PARSEBACK_FAILED = "PARSEBACK_FAILED"
INFLATE_FAILED = "INFLATE_FAILED"
FAKEQUANT_FAILED = "FAKEQUANT_FAILED"
EXACT_SCORE_REJECTED = "EXACT_SCORE_REJECTED"
BYTE_ACCOUNTING_MISSING = "BYTE_ACCOUNTING_MISSING"
ACTION_EFFECT_BLOCKED = "ACTION_EFFECT_BLOCKED"
INVALID_LOWERING_TARGET = "INVALID_LOWERING_TARGET"
SUPPORT_IDENTITY_MISSING = "SUPPORT_IDENTITY_MISSING"
SUPPORT_IDENTITY_MISMATCH = "SUPPORT_IDENTITY_MISMATCH"


@dataclass(frozen=True)
class LoweringVerdict:
    """Typed lowering-race verdict for one hard scorer atom/action."""

    action_id: str
    pair_id: int | None
    region_id: str | None
    support_sha256: str | None
    direct_teacher_status: str
    backend_status: str
    sidecar_status: str
    composite_status: str
    semantic_pose_status: str
    backend_realization_complete: bool
    sidecar_lowering_complete: bool
    best_lowering: str
    first_failing_surface: str
    authority: str
    promotion_eligible: bool
    delta_score_nonrate: float | None
    delta_score_total: float | None
    delta_bytes: int | None
    value_per_byte: float | None
    bytes_by_section: dict[str, int | None]
    exact_score_terms: dict[str, float | int | None]
    target_statuses: dict[str, str]
    next_blocker: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": LOWERING_VERDICT_SCHEMA,
            "action_id": self.action_id,
            "pair_id": self.pair_id,
            "region_id": self.region_id,
            "support_sha256": self.support_sha256,
            "direct_teacher_status": self.direct_teacher_status,
            "backend_status": self.backend_status,
            "sidecar_status": self.sidecar_status,
            "composite_status": self.composite_status,
            "semantic_pose_status": self.semantic_pose_status,
            "backend_realization_complete": self.backend_realization_complete,
            "sidecar_lowering_complete": self.sidecar_lowering_complete,
            "best_lowering": self.best_lowering,
            "first_failing_surface": self.first_failing_surface,
            "authority": self.authority,
            "promotion_eligible": self.promotion_eligible,
            "delta_score_nonrate": self.delta_score_nonrate,
            "delta_score_total": self.delta_score_total,
            "delta_bytes": self.delta_bytes,
            "value_per_byte": self.value_per_byte,
            "bytes_by_section": dict(self.bytes_by_section),
            "exact_score_terms": dict(self.exact_score_terms),
            "target_statuses": dict(self.target_statuses),
            "next_blocker": self.next_blocker,
        }


def build_lowering_race_report(
    *,
    action_id: str,
    action_effects: Iterable[ActionEffect | Mapping[str, Any]] = (),
    support_codec_report: Mapping[str, Any] | None = None,
    expected_support_sha256: str | None = None,
) -> dict[str, Any]:
    """Build the four-way lowering race for one action id."""

    race_inputs = [_coerce_race_input(row) for row in action_effects]
    if support_codec_report is not None:
        race_inputs.extend(_race_inputs_from_support_codec_report(support_codec_report))
    race_inputs = [row for row in race_inputs if row.effect.action_id == action_id]
    support_identity = _support_identity_status(
        [row.effect for row in race_inputs],
        expected_support_sha256=expected_support_sha256,
    )
    candidates = [
        _candidate_row(row, support_identity_failure=support_identity["failure"])
        for row in race_inputs
    ]
    grouped = {target: [row for row in candidates if row["lowering_target"] == target] for target in LOWERING_TARGETS}
    target_accounting = _target_accounting(grouped)
    statuses = {target: _target_status(target, rows) for target, rows in grouped.items()}
    viable = [row for row in candidates if row["viable"] is True]
    best = min(viable, key=lambda row: (float(row["delta_score_total"]), int(row["delta_bytes"] or 0)), default=None)
    first_failure = "none" if best is not None else _first_failure(statuses, candidates)
    next_blocker = _next_blocker(
        first_failure=first_failure,
        missing_targets=target_accounting["missing_targets"],
    )
    verdict = LoweringVerdict(
        action_id=action_id,
        pair_id=_first_pair_id([row.effect for row in race_inputs]),
        region_id=_first_region_id([row.effect for row in race_inputs]),
        support_sha256=_first_support_sha256([row.effect for row in race_inputs]),
        direct_teacher_status=_direct_teacher_status([row.effect for row in race_inputs]),
        backend_status=statuses["backend_realization"],
        sidecar_status=statuses["byte_priced_sidecar"],
        composite_status=statuses["pose_compensated_composite"],
        semantic_pose_status=statuses["semantic_pose_primitive"],
        backend_realization_complete=statuses["backend_realization"] == "accepted",
        sidecar_lowering_complete=statuses["byte_priced_sidecar"] == "accepted",
        best_lowering="discard" if best is None else str(best["lowering_target"]),
        first_failing_surface=first_failure,
        authority="none" if best is None else str(best["authority"]),
        promotion_eligible=False,
        delta_score_nonrate=None if best is None else _float_or_none(best.get("delta_score_nonrate")),
        delta_score_total=None if best is None else _float_or_none(best.get("delta_score_total")),
        delta_bytes=None if best is None else _int_or_none(best.get("delta_bytes")),
        value_per_byte=None if best is None else _float_or_none(best.get("value_per_byte")),
        bytes_by_section={} if best is None else _bytes_by_section(best),
        exact_score_terms={} if best is None else _exact_score_terms(best),
        target_statuses=statuses,
        next_blocker=next_blocker,
    )
    commutator_ledger = _commutator_ledger_for_candidates(candidates, race_inputs)
    return {
        "schema": LOWERING_RACE_SCHEMA,
        "action_id": action_id,
        "verdict": verdict.as_dict(),
        "backend_realization_complete": verdict.backend_realization_complete,
        "sidecar_lowering_complete": verdict.sidecar_lowering_complete,
        "lowering_candidates": candidates,
        "candidate_count": len(candidates),
        "target_accounting": target_accounting,
        "support_identity": support_identity,
        "commutator_ledger": commutator_ledger,
        "support_codec_summary": _support_codec_summary(support_codec_report),
        "promotion_eligible": False,
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }


def write_lowering_race_report(report: Mapping[str, Any], output_dir: str | Path) -> dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "evaluator_action_lowering_race_report.json"
    verdict_path = out_dir / "lowering_verdict.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    verdict = report.get("verdict") if isinstance(report.get("verdict"), Mapping) else {}
    verdict_path.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "report_path": report_path.as_posix(),
        "verdict_path": verdict_path.as_posix(),
    }


@dataclass(frozen=True)
class _RaceInput:
    effect: ActionEffect
    explicit_lowering_target: str | None = None
    explicit_lowering_target_raw: str | None = None


def _candidate_row(row: _RaceInput, *, support_identity_failure: str | None = None) -> dict[str, Any]:
    effect = row.effect
    byte_fields = _byte_fields(effect)
    explicit_target = _normalize_lowering_target(row.explicit_lowering_target)
    invalid_explicit_target = bool(
        row.explicit_lowering_target_raw and explicit_target is None
    )
    first_failure = _effect_failure(
        effect,
        byte_fields,
        invalid_explicit_target=invalid_explicit_target,
        support_identity_failure=support_identity_failure,
    )
    target = explicit_target if explicit_target is not None else _lowering_target(effect)
    viable = bool(first_failure == "ok" and effect.delta_score_total is not None and effect.delta_score_total < 0.0)
    return {
        "schema": LOWERING_CANDIDATE_SCHEMA,
        "action_id": effect.action_id,
        "lowering_target": target,
        "lowering_target_source": "explicit" if explicit_target is not None else "inferred",
        "explicit_lowering_target": row.explicit_lowering_target_raw,
        "action_kind": effect.action_kind,
        "family": effect.family,
        "authority": effect.authority,
        "normalization_scope": effect.normalization_scope,
        "support_sha256": effect.support_sha256,
        "support_encoding": effect.support_encoding,
        "support_encoded_bytes": effect.support_encoded_bytes,
        "action_payload_bytes": byte_fields["action_payload_bytes"],
        "metadata_bytes": byte_fields["metadata_bytes"],
        "old_bytes": effect.old_bytes,
        "new_bytes": effect.new_bytes,
        "delta_bytes": effect.delta_bytes,
        "delta_score_nonrate": effect.delta_score_nonrate,
        "delta_score_total": effect.delta_score_total,
        "value_per_byte": effect.value_per_byte,
        "wrong_to_target": effect.wrong_to_target,
        "target_to_wrong": effect.target_to_wrong,
        "wrong_to_wrong": effect.wrong_to_wrong,
        "argmax_changed": effect.argmax_changed_count_region,
        "pose_output_delta": effect.pose_output_l2_delta,
        "uint8_delta": effect.uint8_changed_count_region,
        "seg_input_delta": effect.seg_input_delta_linf_region,
        "pose_input_delta": effect.posenet_input_delta_linf_pair,
        "fakequant_survived": effect.fakequant_survived,
        "parseback_survived": effect.parseback_survived,
        "inflate_survived": effect.inflate_survived,
        "promotion_eligible": False,
        "viable": viable,
        "first_failing_surface": "none" if viable else first_failure,
        "blockers": list(effect.blockers),
    }


def _effect_failure(
    effect: ActionEffect,
    byte_fields: Mapping[str, int | None],
    *,
    invalid_explicit_target: bool,
    support_identity_failure: str | None,
) -> str:
    if invalid_explicit_target:
        return INVALID_LOWERING_TARGET
    if not effect.support_sha256:
        return SUPPORT_IDENTITY_MISSING
    if support_identity_failure:
        return support_identity_failure
    if effect.blockers:
        return str(effect.blockers[0] or ACTION_EFFECT_BLOCKED)
    if effect.support_encoded_bytes is None:
        return BYTE_ACCOUNTING_MISSING
    if byte_fields["action_payload_bytes"] is None or byte_fields["metadata_bytes"] is None:
        return BYTE_ACCOUNTING_MISSING
    if effect.parseback_survived is not True:
        return PARSEBACK_FAILED
    if effect.inflate_survived is not True:
        return INFLATE_FAILED
    if effect.fakequant_survived is not True:
        return FAKEQUANT_FAILED
    if effect.delta_score_total is None or effect.delta_score_total >= 0.0:
        return EXACT_SCORE_REJECTED
    return "ok"


def _support_identity_status(
    effects: Sequence[ActionEffect],
    *,
    expected_support_sha256: str | None,
) -> dict[str, Any]:
    expected = str(expected_support_sha256 or "").strip()
    support_hashes = _dedupe(effect.support_sha256 for effect in effects if effect.support_sha256)
    missing_count = sum(1 for effect in effects if not effect.support_sha256)
    blockers: list[str] = []
    failure: str | None = None
    if missing_count:
        blockers.append("lowering_race_support_sha256_missing")
        if not support_hashes:
            failure = SUPPORT_IDENTITY_MISSING
    if expected and any(support != expected for support in support_hashes):
        blockers.append("lowering_race_expected_support_sha256_mismatch")
        failure = SUPPORT_IDENTITY_MISMATCH
    if not expected and len(support_hashes) > 1:
        blockers.append("lowering_race_candidate_support_sha256_mismatch")
        failure = SUPPORT_IDENTITY_MISMATCH
    return {
        "schema": "tac.evaluator_action_lowering_race.support_identity.v1",
        "expected_support_sha256": expected or None,
        "support_sha256s": support_hashes,
        "missing_support_sha256_count": missing_count,
        "all_candidates_same_support": (
            missing_count == 0
            and bool(support_hashes)
            and (all(support == expected for support in support_hashes) if expected else len(support_hashes) == 1)
        ),
        "failure": failure,
        "blockers": blockers,
    }


def _target_accounting(
    grouped_candidates: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    present = [
        target
        for target in LOWERING_TARGETS
        if bool(grouped_candidates.get(target))
    ]
    missing = [target for target in LOWERING_TARGETS if target not in present]
    return {
        "schema": "tac.evaluator_action_lowering_race.target_accounting.v1",
        "expected_targets": list(LOWERING_TARGETS),
        "present_targets": present,
        "missing_targets": missing,
        "all_targets_accounted": not missing,
    }


def _target_status(target: str, rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return {
            "backend_realization": BACKEND_REALIZATION_FAILED,
            "pair_local_latent_action": PAIR_LOCAL_LATENT_NOT_MEASURED,
            "frame0_pose_compensation": FRAME0_POSE_COMPENSATION_NOT_MEASURED,
            "frame1_seg_wall_crossing": FRAME1_SEG_WALL_CROSSING_NOT_MEASURED,
            "byte_priced_sidecar": SIDECAR_TOO_EXPENSIVE,
            "pose_compensated_composite": COMPOSITE_NOT_MEASURED,
            "snerv_source_state_action": SNERV_SOURCE_STATE_NOT_MEASURED,
            "semantic_pose_primitive": SEMANTIC_PRIMITIVE_MISSING,
            "byte_entropy_rewrite": BYTE_ENTROPY_REWRITE_NOT_MEASURED,
        }[target]
    if any(row.get("viable") is True for row in rows):
        return "accepted"
    return str(rows[0].get("first_failing_surface") or EXACT_SCORE_REJECTED)


def _first_failure(statuses: Mapping[str, str], candidates: Sequence[Mapping[str, Any]]) -> str:
    for row in candidates:
        failure = str(row.get("first_failing_surface") or "")
        if failure and failure != "none":
            return failure
    for target in LOWERING_TARGETS:
        status = statuses[target]
        if status != "accepted":
            return status
    return "none"


def _lowering_target(effect: ActionEffect) -> str:
    text = " ".join(
        str(value or "")
        for value in (
            effect.action_kind,
            effect.inverse_source,
            effect.producer,
            effect.support_encoding,
            " ".join(effect.payload_sections),
        )
    ).lower()
    if "support_codec=" in text or effect.support_encoding in {
        "rle",
        "bitmap_bitset",
        "coordinate_list_u16",
        "tile_set_8x8_bitmap",
        "tile_set_16x16_bitmap",
        "target_region_action_coordinates_v1",
        "explicit_yx_u16_coordinates",
    }:
        return "byte_priced_sidecar"
    if (
        "frame0_pose_target_only" in text
        or "frame0_pose_compensation" in text
        or "pose_only_compensation" in text
    ):
        return "frame0_pose_compensation"
    if (
        "frame1_seg_wall_crossing" in text
        or "seg_wall_crossing" in text
        or "seg_frontier" in text
        or "wall_crossing" in text
    ):
        return "frame1_seg_wall_crossing"
    if (
        "independent_birth_plus_frame0_pose" in text
        or "joint_line_search_composite" in text
        or "frame0_pose_then_birth_composite" in text
        or "composite" in text
        or "pose_then" in text
        or "then_pose" in text
    ):
        return "pose_compensated_composite"
    if (
        "snerv" in text
        or "mfu" in text
        or "hfr" in text
        or " tub " in f" {text} "
        or "tub_" in text
        or "_tub" in text
        or "lf_hf" in text
    ):
        return "snerv_source_state_action"
    if (
        "byte_entropy" in text
        or "entropy_rewrite" in text
        or "byte_rewrite" in text
        or "range_coder" in text
        or "ans" in text
    ):
        return "byte_entropy_rewrite"
    if "discard" in text or "no_op" in text:
        return "discard"
    if (
        "semantic" in text
        or "latent_derived" in text
        or "semantic_pose" in text
        or "qrgb" in text
    ):
        return "semantic_pose_primitive"
    if (
        "backend" in text
        or "pair_local_latent" in text
        or "pair_latent" in text
        or "wall_normal" in text
        or "adapter" in text
        or "birth_only" in text
        or "target_region_birth" in text
    ):
        if "pair_local_latent" in text or "pair_latent" in text:
            return "pair_local_latent_action"
        return "backend_realization"
    return "byte_priced_sidecar"


def _direct_teacher_status(effects: Sequence[ActionEffect]) -> str:
    if any((effect.wrong_to_target or 0) > 0 or (effect.pose_output_l2_delta or 0.0) > 0.0 for effect in effects):
        return "surface_motion_observed"
    return DIRECT_TEACHER_NO_WALL_CROSS


def _byte_fields(effect: ActionEffect) -> dict[str, int | None]:
    action_payload = _section_int(effect.payload_sections, "action_payload_bytes")
    metadata = _section_int(effect.payload_sections, "metadata_bytes")
    return {
        "action_payload_bytes": action_payload,
        "metadata_bytes": metadata,
    }


def _section_int(sections: Sequence[str], key: str) -> int | None:
    prefix = f"{key}="
    for section in sections:
        text = str(section)
        if text.startswith(prefix):
            try:
                return int(text[len(prefix) :])
            except ValueError:
                return None
    return None


def _race_inputs_from_support_codec_report(report: Mapping[str, Any]) -> list[_RaceInput]:
    out: list[_RaceInput] = []
    for sub in report.get("reports", []) if isinstance(report.get("reports"), Sequence) else []:
        if isinstance(sub, Mapping) and isinstance(sub.get("selected_action_effect"), Mapping):
            out.append(_coerce_race_input(sub["selected_action_effect"]))
    if isinstance(report.get("selected_action_effect"), Mapping):
        out.append(_coerce_race_input(report["selected_action_effect"]))
    return out


def _support_codec_summary(report: Mapping[str, Any] | None) -> dict[str, Any]:
    if report is None:
        return {}
    summaries: list[dict[str, Any]] = []
    for sub in report.get("reports", []) if isinstance(report.get("reports"), Sequence) else []:
        if not isinstance(sub, Mapping):
            continue
        summaries.append(
            {
                "action_id": sub.get("action_id"),
                "selected_support_encoding": sub.get("selected_support_encoding"),
                "selected_total_cost_bytes": sub.get("selected_total_cost_bytes"),
            }
        )
    return {"schema": "tac.evaluator_action_lowering_race.support_codec_summary.v1", "reports": summaries}


def _coerce_race_input(row: ActionEffect | Mapping[str, Any]) -> _RaceInput:
    if isinstance(row, ActionEffect):
        explicit_raw = _explicit_lowering_target_from_effect(row)
        return _RaceInput(
            effect=row,
            explicit_lowering_target=explicit_raw,
            explicit_lowering_target_raw=explicit_raw,
        )
    explicit_raw = _explicit_lowering_target_from_mapping(row)
    return _RaceInput(
        effect=ActionEffect.from_dict(row),
        explicit_lowering_target=explicit_raw,
        explicit_lowering_target_raw=explicit_raw,
    )


def _explicit_lowering_target_from_mapping(row: Mapping[str, Any]) -> str | None:
    for key in ("lowering_target", "candidate_lowering_target"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    sections = row.get("payload_sections")
    return _explicit_lowering_target_from_sections(
        sections if isinstance(sections, Sequence) and not isinstance(sections, str | bytes) else ()
    )


def _explicit_lowering_target_from_effect(effect: ActionEffect) -> str | None:
    return _explicit_lowering_target_from_sections(effect.payload_sections)


def _explicit_lowering_target_from_sections(sections: Sequence[Any]) -> str | None:
    for section in sections:
        text = str(section).strip()
        if not text:
            continue
        for prefix in ("lowering_target=", "candidate_lowering_target="):
            if text.startswith(prefix):
                return text[len(prefix) :].strip()
    return None


def _normalize_lowering_target(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower().replace("-", "_")
    aliases = {
        "backend": "backend_realization",
        "pair_local_latent": "pair_local_latent_action",
        "frame0_pose": "frame0_pose_compensation",
        "frame1_seg": "frame1_seg_wall_crossing",
        "sidecar": "byte_priced_sidecar",
        "composite": "pose_compensated_composite",
        "semantic_pose": "semantic_pose_primitive",
        "byte_rewrite": "byte_entropy_rewrite",
        "snerv": "snerv_source_state_action",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in LOWERING_TARGETS else None


def _bytes_by_section(candidate: Mapping[str, Any]) -> dict[str, int | None]:
    return {
        "support_encoded_bytes": _int_or_none(candidate.get("support_encoded_bytes")),
        "action_payload_bytes": _int_or_none(candidate.get("action_payload_bytes")),
        "metadata_bytes": _int_or_none(candidate.get("metadata_bytes")),
        "delta_bytes": _int_or_none(candidate.get("delta_bytes")),
    }


def _exact_score_terms(candidate: Mapping[str, Any]) -> dict[str, float | int | None]:
    return {
        "delta_score_nonrate": _float_or_none(candidate.get("delta_score_nonrate")),
        "delta_score_total": _float_or_none(candidate.get("delta_score_total")),
        "delta_bytes": _int_or_none(candidate.get("delta_bytes")),
        "value_per_byte": _float_or_none(candidate.get("value_per_byte")),
    }


def _next_blocker(*, first_failure: str, missing_targets: Sequence[str]) -> str:
    if first_failure and first_failure != "none":
        return first_failure
    if missing_targets:
        return f"measure_lowering_target:{missing_targets[0]}"
    return "none"


def _commutator_ledger_for_candidates(
    candidates: Sequence[Mapping[str, Any]],
    race_inputs: Sequence[_RaceInput],
) -> dict[str, Any]:
    singles: list[ActionEffect] = []
    pairs: list[ActionEffect] = []
    for row, candidate in zip(race_inputs, candidates, strict=False):
        target = str(candidate.get("lowering_target") or _lowering_target(row.effect))
        if target == "pose_compensated_composite":
            pairs.append(row.effect)
        elif target != "discard":
            singles.append(row.effect)
    return build_commutator_ledger(
        singles,
        pairs,
        use_default_measurement_command=False,
        measurement_command_blockers=("lowering_race_measurement_command_not_bound",),
    )


def _first_pair_id(effects: Sequence[ActionEffect]) -> int | None:
    for effect in effects:
        if effect.pair_ids:
            return int(effect.pair_ids[0])
    return None


def _first_region_id(effects: Sequence[ActionEffect]) -> str | None:
    for effect in effects:
        if effect.region_ids:
            return str(effect.region_ids[0])
    return None


def _first_support_sha256(effects: Sequence[ActionEffect]) -> str | None:
    for effect in effects:
        if effect.support_sha256:
            return effect.support_sha256
    return None


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        return float(value)
    return None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    return None


def _dedupe(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


__all__ = [
    "ACTION_EFFECT_BLOCKED",
    "BACKEND_REALIZATION_FAILED",
    "BYTE_ACCOUNTING_MISSING",
    "BYTE_ENTROPY_REWRITE_NOT_MEASURED",
    "COMPOSITE_NOT_MEASURED",
    "DIRECT_TEACHER_NO_WALL_CROSS",
    "DISCARD_NOT_DECLARED",
    "EXACT_SCORE_REJECTED",
    "FAKEQUANT_FAILED",
    "FRAME0_POSE_COMPENSATION_NOT_MEASURED",
    "FRAME1_SEG_WALL_CROSSING_NOT_MEASURED",
    "INFLATE_FAILED",
    "INVALID_LOWERING_TARGET",
    "LOWERING_CANDIDATE_SCHEMA",
    "LOWERING_RACE_SCHEMA",
    "LOWERING_TARGETS",
    "LOWERING_VERDICT_SCHEMA",
    "PAIR_LOCAL_LATENT_NOT_MEASURED",
    "PARSEBACK_FAILED",
    "SEMANTIC_PRIMITIVE_MISSING",
    "SIDECAR_TOO_EXPENSIVE",
    "SNERV_SOURCE_STATE_NOT_MEASURED",
    "SUPPORT_IDENTITY_MISMATCH",
    "SUPPORT_IDENTITY_MISSING",
    "LoweringVerdict",
    "build_lowering_race_report",
    "write_lowering_race_report",
]
