# SPDX-License-Identifier: MIT
"""Minimal evaluator-action lowering race over ActionEffect rows."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tac.analysis.action_effect import ActionEffect

LOWERING_RACE_SCHEMA = "tac.evaluator_action_lowering_race.v1"
LOWERING_VERDICT_SCHEMA = "tac.evaluator_action_lowering_verdict.v1"
LOWERING_CANDIDATE_SCHEMA = "tac.evaluator_action_lowering_candidate.v1"

LOWERING_TARGETS = (
    "backend_realization",
    "byte_priced_sidecar",
    "pose_compensated_composite",
    "semantic_pose_primitive",
)

DIRECT_TEACHER_NO_WALL_CROSS = "DIRECT_TEACHER_NO_WALL_CROSS"
BACKEND_REALIZATION_FAILED = "BACKEND_REALIZATION_FAILED"
SIDECAR_TOO_EXPENSIVE = "SIDECAR_TOO_EXPENSIVE"
COMPOSITE_NOT_MEASURED = "COMPOSITE_NOT_MEASURED"
SEMANTIC_PRIMITIVE_MISSING = "SEMANTIC_PRIMITIVE_MISSING"
PARSEBACK_FAILED = "PARSEBACK_FAILED"
INFLATE_FAILED = "INFLATE_FAILED"
EXACT_SCORE_REJECTED = "EXACT_SCORE_REJECTED"
BYTE_ACCOUNTING_MISSING = "BYTE_ACCOUNTING_MISSING"


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
    best_lowering: str
    first_failing_surface: str
    authority: str
    promotion_eligible: bool
    delta_score_nonrate: float | None
    delta_score_total: float | None
    delta_bytes: int | None
    value_per_byte: float | None

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
            "best_lowering": self.best_lowering,
            "first_failing_surface": self.first_failing_surface,
            "authority": self.authority,
            "promotion_eligible": self.promotion_eligible,
            "delta_score_nonrate": self.delta_score_nonrate,
            "delta_score_total": self.delta_score_total,
            "delta_bytes": self.delta_bytes,
            "value_per_byte": self.value_per_byte,
        }


def build_lowering_race_report(
    *,
    action_id: str,
    action_effects: Iterable[ActionEffect | Mapping[str, Any]] = (),
    support_codec_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the four-way lowering race for one action id."""

    effects = [_coerce_effect(row) for row in action_effects]
    if support_codec_report is not None:
        effects.extend(_effects_from_support_codec_report(support_codec_report))
    effects = [effect for effect in effects if effect.action_id == action_id]
    candidates = [_candidate_row(effect) for effect in effects]
    grouped = {target: [row for row in candidates if row["lowering_target"] == target] for target in LOWERING_TARGETS}
    statuses = {target: _target_status(target, rows) for target, rows in grouped.items()}
    viable = [row for row in candidates if row["viable"] is True]
    best = min(viable, key=lambda row: (float(row["delta_score_total"]), int(row["delta_bytes"] or 0)), default=None)
    first_failure = "none" if best is not None else _first_failure(statuses, candidates)
    verdict = LoweringVerdict(
        action_id=action_id,
        pair_id=_first_pair_id(effects),
        region_id=_first_region_id(effects),
        support_sha256=_first_support_sha256(effects),
        direct_teacher_status=_direct_teacher_status(effects),
        backend_status=statuses["backend_realization"],
        sidecar_status=statuses["byte_priced_sidecar"],
        composite_status=statuses["pose_compensated_composite"],
        semantic_pose_status=statuses["semantic_pose_primitive"],
        best_lowering="none" if best is None else str(best["lowering_target"]),
        first_failing_surface=first_failure,
        authority="none" if best is None else str(best["authority"]),
        promotion_eligible=False,
        delta_score_nonrate=None if best is None else _float_or_none(best.get("delta_score_nonrate")),
        delta_score_total=None if best is None else _float_or_none(best.get("delta_score_total")),
        delta_bytes=None if best is None else _int_or_none(best.get("delta_bytes")),
        value_per_byte=None if best is None else _float_or_none(best.get("value_per_byte")),
    )
    return {
        "schema": LOWERING_RACE_SCHEMA,
        "action_id": action_id,
        "verdict": verdict.as_dict(),
        "lowering_candidates": candidates,
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


def _candidate_row(effect: ActionEffect) -> dict[str, Any]:
    byte_fields = _byte_fields(effect)
    first_failure = _effect_failure(effect, byte_fields)
    target = _lowering_target(effect)
    viable = bool(first_failure == "ok" and effect.delta_score_total is not None and effect.delta_score_total < 0.0)
    return {
        "schema": LOWERING_CANDIDATE_SCHEMA,
        "action_id": effect.action_id,
        "lowering_target": target,
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


def _effect_failure(effect: ActionEffect, byte_fields: Mapping[str, int | None]) -> str:
    if effect.support_encoded_bytes is None:
        return BYTE_ACCOUNTING_MISSING
    if byte_fields["action_payload_bytes"] is None or byte_fields["metadata_bytes"] is None:
        return BYTE_ACCOUNTING_MISSING
    if effect.parseback_survived is not True:
        return PARSEBACK_FAILED
    if effect.inflate_survived is not True:
        return INFLATE_FAILED
    if effect.delta_score_total is None or effect.delta_score_total >= 0.0:
        return EXACT_SCORE_REJECTED
    return "ok"


def _target_status(target: str, rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return {
            "backend_realization": BACKEND_REALIZATION_FAILED,
            "byte_priced_sidecar": SIDECAR_TOO_EXPENSIVE,
            "pose_compensated_composite": COMPOSITE_NOT_MEASURED,
            "semantic_pose_primitive": SEMANTIC_PRIMITIVE_MISSING,
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
    if "composite" in text or "pose_then" in text or "then_pose" in text:
        return "pose_compensated_composite"
    if "semantic" in text or "latent_derived" in text or "semantic_pose" in text:
        return "semantic_pose_primitive"
    if "backend" in text or "wall_normal" in text or "adapter" in text:
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
        "action_payload_bytes": 0 if action_payload is None and effect.support_encoded_bytes is not None else action_payload,
        "metadata_bytes": 0 if metadata is None and effect.support_encoded_bytes is not None else metadata,
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


def _effects_from_support_codec_report(report: Mapping[str, Any]) -> list[ActionEffect]:
    out: list[ActionEffect] = []
    for sub in report.get("reports", []) if isinstance(report.get("reports"), Sequence) else []:
        if isinstance(sub, Mapping) and isinstance(sub.get("selected_action_effect"), Mapping):
            out.append(ActionEffect.from_dict(sub["selected_action_effect"]))
    if isinstance(report.get("selected_action_effect"), Mapping):
        out.append(ActionEffect.from_dict(report["selected_action_effect"]))
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


def _coerce_effect(row: ActionEffect | Mapping[str, Any]) -> ActionEffect:
    return row if isinstance(row, ActionEffect) else ActionEffect.from_dict(row)


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


__all__ = [
    "BACKEND_REALIZATION_FAILED",
    "BYTE_ACCOUNTING_MISSING",
    "COMPOSITE_NOT_MEASURED",
    "DIRECT_TEACHER_NO_WALL_CROSS",
    "EXACT_SCORE_REJECTED",
    "INFLATE_FAILED",
    "LOWERING_CANDIDATE_SCHEMA",
    "LOWERING_RACE_SCHEMA",
    "LOWERING_TARGETS",
    "LOWERING_VERDICT_SCHEMA",
    "PARSEBACK_FAILED",
    "SEMANTIC_PRIMITIVE_MISSING",
    "SIDECAR_TOO_EXPENSIVE",
    "LoweringVerdict",
    "build_lowering_race_report",
    "write_lowering_race_report",
]
