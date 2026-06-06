# SPDX-License-Identifier: MIT
"""Typed evaluator-action effects for NeRV score-program compilation.

The contest objective is an evaluator quotient, not human video fidelity.  This
module gives HiNeRV, SNeRV, selector, sidecar, and byte-compiler actions one
shared score-unit record: exact Seg/Pose/rate movement, receiver-surface
survival, payload incidence, and noncommutative composition value.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from tac.optimization.proxy_candidate_contract import PROXY_FALSE_AUTHORITY_FIELDS
from tac.score_geometry import CONTEST_REFERENCE_BYTES, contest_score

ACTION_EFFECT_SCHEMA = "nerv_action_effect.v1"
ACTION_EFFECT_LEDGER_SCHEMA = "nerv_action_effect_ledger.v1"
ACTION_COMMUTATOR_ROW_SCHEMA = "nerv_action_commutator_row.v1"


@dataclass(frozen=True)
class ActionEffect:
    """Exact score movement for one receiver-visible evaluator action."""

    action_id: str
    family: str
    authority: str
    producer: str
    consumer: str | None
    affected_pairs: tuple[int, ...]
    affected_regions: tuple[str, ...]
    payload_sections: tuple[str, ...]
    old_d_seg: float
    new_d_seg: float
    old_d_pose: float
    new_d_pose: float
    old_bytes: int
    new_bytes: int
    receiver_surface: Mapping[str, float | int | bool | str]
    parseback_survived: bool
    fakequant_survived: bool
    inflate_survived: bool | None = None
    value_per_byte: float | None = None
    reference_bytes: int = CONTEST_REFERENCE_BYTES

    @property
    def delta_bytes(self) -> int:
        return int(self.new_bytes) - int(self.old_bytes)

    @property
    def delta_d_seg(self) -> float:
        return float(self.new_d_seg) - float(self.old_d_seg)

    @property
    def delta_d_pose(self) -> float:
        return float(self.new_d_pose) - float(self.old_d_pose)

    @property
    def delta_score_nonrate(self) -> float:
        return 100.0 * self.delta_d_seg + (
            math.sqrt(10.0 * float(self.new_d_pose))
            - math.sqrt(10.0 * float(self.old_d_pose))
        )

    @property
    def rate_score_delta(self) -> float:
        return 25.0 * self.delta_bytes / float(self.reference_bytes)

    @property
    def delta_score_total(self) -> float:
        return self.delta_score_nonrate + self.rate_score_delta

    @property
    def old_score(self) -> float:
        return contest_score(
            self.old_d_seg,
            self.old_d_pose,
            self.old_bytes,
            reference_bytes=self.reference_bytes,
        )

    @property
    def new_score(self) -> float:
        return contest_score(
            self.new_d_seg,
            self.new_d_pose,
            self.new_bytes,
            reference_bytes=self.reference_bytes,
        )

    def to_mapping(self, *, min_score_improvement: float = 0.0) -> dict[str, Any]:
        blockers = _action_effect_blockers(
            self,
            min_score_improvement=min_score_improvement,
        )
        score_admissible = _score_admissible(
            self,
            min_score_improvement=min_score_improvement,
        )
        payload = {
            "schema": ACTION_EFFECT_SCHEMA,
            **{
                key: value
                for key, value in asdict(self).items()
                if key != "receiver_surface"
            },
            "affected_pairs": list(self.affected_pairs),
            "affected_regions": list(self.affected_regions),
            "payload_sections": list(self.payload_sections),
            "receiver_surface": dict(self.receiver_surface),
            "old_score": self.old_score,
            "new_score": self.new_score,
            "delta_d_seg": self.delta_d_seg,
            "delta_d_pose": self.delta_d_pose,
            "delta_bytes": self.delta_bytes,
            "delta_score_nonrate": self.delta_score_nonrate,
            "rate_score_delta": self.rate_score_delta,
            "delta_score_total": self.delta_score_total,
            "byte_price": 25.0 / float(self.reference_bytes),
            "receiver_visible": _receiver_visible(self.receiver_surface),
            "byte_priced": _byte_priced(self),
            "score_admissible": score_admissible,
            "action_effect_admitted": not blockers,
            "blockers": blockers,
            "policy": {
                "exact_nonlinear_score_is_authority": True,
                "human_visual_fidelity_is_not_authority": True,
                "archive_bytes_are_charged_object": True,
                "composition_must_be_measured_not_assumed_additive": True,
            },
            **PROXY_FALSE_AUTHORITY_FIELDS,
        }
        payload["score_claim"] = False
        return payload


def build_action_effect(
    payload: Mapping[str, Any],
    *,
    min_score_improvement: float = 0.0,
    reference_bytes: int = CONTEST_REFERENCE_BYTES,
) -> dict[str, Any]:
    """Normalize a mapping into the canonical action-effect payload."""

    effect = ActionEffect(
        action_id=str(payload.get("action_id") or payload.get("id") or ""),
        family=_family(payload.get("family")),
        authority=str(payload.get("authority") or ""),
        producer=str(payload.get("producer") or ""),
        consumer=_optional_text(payload.get("consumer")),
        affected_pairs=tuple(_int_values(payload.get("affected_pairs"))),
        affected_regions=tuple(_text_values(payload.get("affected_regions"))),
        payload_sections=tuple(_text_values(payload.get("payload_sections"))),
        old_d_seg=_finite_float(payload.get("old_d_seg")),
        new_d_seg=_finite_float(payload.get("new_d_seg")),
        old_d_pose=_finite_float(payload.get("old_d_pose")),
        new_d_pose=_finite_float(payload.get("new_d_pose")),
        old_bytes=_nonnegative_int(payload.get("old_bytes")),
        new_bytes=_nonnegative_int(payload.get("new_bytes")),
        receiver_surface=_receiver_surface(payload.get("receiver_surface")),
        parseback_survived=payload.get("parseback_survived") is True,
        fakequant_survived=payload.get("fakequant_survived") is True,
        inflate_survived=(
            None
            if payload.get("inflate_survived") is None
            else payload.get("inflate_survived") is True
        ),
        value_per_byte=_optional_finite_float(payload.get("value_per_byte")),
        reference_bytes=int(reference_bytes),
    )
    return effect.to_mapping(min_score_improvement=min_score_improvement)


def action_effect_from_pair_local_servo(
    *,
    receipt: Mapping[str, Any],
    report: Mapping[str, Any],
    receiver_surface: Mapping[str, Any],
) -> dict[str, Any]:
    """Build an action effect from the pair-local NeRV servo report."""

    return build_action_effect(
        {
            "action_id": (
                receipt.get("action_id")
                or _nested_text(receipt, ("action_algebra_trace", "selected_action_id"))
                or receipt.get("actuator_id")
                or receipt.get("actuator_kind")
                or "pair_local_servo_action"
            ),
            "family": report.get("family") or receipt.get("family"),
            "authority": report.get("authority") or receipt.get("authority"),
            "producer": "nerv_pair_local_distortion_servo",
            "consumer": "nerv_long_training_campaign_admission",
            "affected_pairs": report.get("pair_ids") or receipt.get("pair_ids"),
            "affected_regions": _affected_regions(receipt),
            "payload_sections": receipt.get("payload_sections") or (),
            "old_d_seg": report.get("score_state_old_d_seg")
            or receipt.get("old_d_seg"),
            "new_d_seg": report.get("score_state_new_d_seg")
            or receipt.get("new_d_seg"),
            "old_d_pose": report.get("score_state_old_d_pose")
            or receipt.get("old_d_pose"),
            "new_d_pose": report.get("score_state_new_d_pose")
            or receipt.get("new_d_pose"),
            "old_bytes": receipt.get("old_archive_bytes"),
            "new_bytes": receipt.get("new_archive_bytes"),
            "receiver_surface": receiver_surface,
            "parseback_survived": _truthy_surface(
                report,
                ("surfaces", "parseback_survival"),
            ),
            "fakequant_survived": _truthy_surface(
                report,
                ("surfaces", "fakequant_survival"),
            ),
            "inflate_survived": _optional_truthy_surface(
                report,
                ("surfaces", "inflate_survival"),
            ),
            "value_per_byte": report.get("value_per_byte"),
        }
    )


def build_action_commutator_row(
    *,
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    composed: Mapping[str, Any],
) -> dict[str, Any]:
    """Measure non-additivity for ``second(first(base))``."""

    first_delta = _signed_finite_float(first.get("delta_score_total"))
    second_delta = _signed_finite_float(second.get("delta_score_total"))
    composed_delta = _signed_finite_float(composed.get("delta_score_total"))
    commutator_delta = composed_delta - first_delta - second_delta
    return {
        "schema": ACTION_COMMUTATOR_ROW_SCHEMA,
        "first_action_id": first.get("action_id"),
        "second_action_id": second.get("action_id"),
        "composed_action_id": composed.get("action_id"),
        "first_delta_score_total": first_delta,
        "second_delta_score_total": second_delta,
        "composed_delta_score_total": composed_delta,
        "commutator_delta_score_total": commutator_delta,
        "synergy_score_units": -commutator_delta,
        "macro_action_recommended": commutator_delta < 0.0,
        "policy": {
            "composition_value_is_measured_against_replayed_composition": True,
            "negative_commutator_means_superadditive_score_improvement": True,
        },
        **PROXY_FALSE_AUTHORITY_FIELDS,
    }


def build_action_effect_ledger(
    effects: Sequence[Mapping[str, Any]],
    *,
    commutators: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Return a canonical ledger for selector/menu/byte-compiler consumers."""

    normalized_effects = [
        dict(effect)
        for effect in effects
        if isinstance(effect, Mapping)
        and effect.get("schema") == ACTION_EFFECT_SCHEMA
    ]
    return {
        "schema": ACTION_EFFECT_LEDGER_SCHEMA,
        "effect_count": len(normalized_effects),
        "admitted_effect_count": sum(
            1 for effect in normalized_effects if effect.get("action_effect_admitted") is True
        ),
        "effects": normalized_effects,
        "commutators": [
            dict(row)
            for row in commutators
            if isinstance(row, Mapping)
            and row.get("schema") == ACTION_COMMUTATOR_ROW_SCHEMA
        ],
        "score_claim": False,
        **PROXY_FALSE_AUTHORITY_FIELDS,
    }


def _action_effect_blockers(
    effect: ActionEffect,
    *,
    min_score_improvement: float,
) -> list[str]:
    blockers: list[str] = []
    if not effect.action_id:
        blockers.append("action_effect_action_id_missing")
    if not effect.producer:
        blockers.append("action_effect_producer_missing")
    if effect.family not in {"hinerv", "snerv", "pact_nerv", "selector", "shared"}:
        blockers.append("action_effect_family_unknown")
    if not effect.authority:
        blockers.append("action_effect_authority_missing")
    if not _score_state_valid(effect):
        blockers.append("action_effect_score_state_invalid")
    if not _archive_byte_state_valid(effect):
        blockers.append("action_effect_archive_byte_state_invalid")
    if not _receiver_visible(effect.receiver_surface):
        blockers.append("action_effect_receiver_surface_motion_missing")
    if effect.fakequant_survived is not True:
        blockers.append("action_effect_fakequant_survival_missing")
    if effect.parseback_survived is not True:
        blockers.append("action_effect_parseback_survival_missing")
    if not _score_admissible(effect, min_score_improvement=min_score_improvement):
        blockers.append("action_effect_exact_score_delta_not_admissible")
    if not _byte_priced(effect):
        blockers.append("action_effect_byte_delta_not_priced")
    return _dedupe(blockers)


def _byte_priced(effect: ActionEffect) -> bool:
    if not _archive_byte_state_valid(effect):
        return False
    if effect.delta_bytes == 0:
        return True
    if effect.value_per_byte is None:
        return False
    return math.isfinite(float(effect.value_per_byte)) and float(effect.value_per_byte) >= 0.0


def _score_admissible(
    effect: ActionEffect,
    *,
    min_score_improvement: float,
) -> bool:
    if not _score_state_valid(effect):
        return False
    return effect.delta_score_total < -float(min_score_improvement)


def _score_state_valid(effect: ActionEffect) -> bool:
    return all(
        math.isfinite(float(value)) and float(value) >= 0.0
        for value in (
            effect.old_d_seg,
            effect.new_d_seg,
            effect.old_d_pose,
            effect.new_d_pose,
        )
    ) and math.isfinite(float(effect.delta_score_total))


def _archive_byte_state_valid(effect: ActionEffect) -> bool:
    return (
        int(effect.old_bytes) >= 0
        and int(effect.new_bytes) >= 0
        and int(effect.reference_bytes) > 0
    )


def _receiver_visible(surface: Mapping[str, Any]) -> bool:
    for key in (
        "uint8_changed_pixels",
        "receiver_surface_uint8_changed_pixels",
        "argmax_flipped_pixels",
        "receiver_surface_argmax_flipped_pixels",
        "pose_output_delta_l2",
        "receiver_surface_pose_output_delta",
        "segnet_input_delta_linf",
        "receiver_surface_segnet_input_delta_linf",
        "posenet_input_delta_linf",
        "receiver_surface_posenet_input_delta_linf",
    ):
        value = _optional_finite_float(surface.get(key))
        if value is not None and value > 0.0:
            return True
    return False


def _receiver_surface(value: Any) -> dict[str, float | int | bool | str]:
    if not isinstance(value, Mapping):
        return {}
    out: dict[str, float | int | bool | str] = {}
    for key, raw in value.items():
        if isinstance(raw, bool | int | float | str) and not (
            isinstance(raw, float) and not math.isfinite(raw)
        ):
            out[str(key)] = raw
    return out


def _affected_regions(receipt: Mapping[str, Any]) -> tuple[str, ...]:
    explicit = tuple(_text_values(receipt.get("affected_regions")))
    if explicit:
        return explicit
    debt = receipt.get("worst_scorer_debt")
    if isinstance(debt, Mapping):
        target = _optional_text(debt.get("target_id"))
        if target:
            return (target,)
    return ()


def _truthy_surface(payload: Mapping[str, Any], path: tuple[str, str]) -> bool:
    first = payload.get(path[0])
    if isinstance(first, Mapping):
        return first.get(path[1]) is True
    return False


def _optional_truthy_surface(
    payload: Mapping[str, Any],
    path: tuple[str, str],
) -> bool | None:
    first = payload.get(path[0])
    if not isinstance(first, Mapping) or path[1] not in first:
        return None
    return first.get(path[1]) is True


def _nested_text(payload: Mapping[str, Any], path: tuple[str, str]) -> str | None:
    first = payload.get(path[0])
    if not isinstance(first, Mapping):
        return None
    return _optional_text(first.get(path[1]))


def _family(value: Any) -> str:
    family = str(value or "shared").strip().lower().replace("-", "_")
    if family == "hi_nerv":
        return "hinerv"
    if family in {"hinerv", "snerv", "pact_nerv", "selector"}:
        return family
    return "shared"


def _finite_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) and out >= 0.0 else math.nan


def _optional_finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _signed_finite_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def _nonnegative_int(value: Any) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError):
        return -1
    return out if out >= 0 else -1


def _int_values(value: Any) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        value = [] if value is None else [value]
    out: list[int] = []
    for raw in value:
        try:
            item = int(raw)
        except (TypeError, ValueError):
            continue
        out.append(item)
    return out


def _text_values(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        value = [] if value is None else [value]
    return [str(item) for item in value if str(item)]


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _dedupe(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


__all__ = [
    "ACTION_COMMUTATOR_ROW_SCHEMA",
    "ACTION_EFFECT_LEDGER_SCHEMA",
    "ACTION_EFFECT_SCHEMA",
    "ActionEffect",
    "action_effect_from_pair_local_servo",
    "build_action_commutator_row",
    "build_action_effect",
    "build_action_effect_ledger",
]
