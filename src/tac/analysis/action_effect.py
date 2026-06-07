# SPDX-License-Identifier: MIT
"""Typed evaluator-action effects for NeRV score-program compilation.

The contest objective is an evaluator quotient, not human video fidelity.  This
module gives HiNeRV, SNeRV, selector, sidecar, and byte-compiler actions one
shared score-unit record: exact Seg/Pose/rate movement, receiver-surface
survival, payload incidence, and noncommutative composition value.

This file carries TWO surfaces:

* ``EvaluatorActionEffect`` (schema ``nerv_action_effect.v1``) — the original
  receiver-surface admission row consumed by
  ``tac.substrates._shared.mlx_score_aware.servo_lift`` via
  :func:`build_action_effect`.  It bundles receiver-surface visibility +
  custody-hash gating + admission blockers + the action-commutator/ledger
  builders.  Behavior is preserved byte-for-byte (only the class name changed
  from the former ``ActionEffect``; no caller imported that class by name).

* ``ActionEffect`` (schema ``tac.action_effect.v1``) — the THIN partner
  amendment #5 currency: ONE typed ledger row both score paths (HiNeRV birth +
  pair-local servo + PR110 selector) can produce, with the single shared
  scoring computation :func:`compute_delta_scores`, three tolerant
  constructors, and an fcntl-locked JSONL ledger.  THIN = no framework, no
  orchestration, no scoring monkey-patches; it delegates the one nonlinear
  formula to :func:`tac.score_geometry.contest_score` so neither path grows a
  second drifting objective.  It is an analysis-layer row: ``promotion_eligible``
  is structurally pinned ``False`` and validated in ``__post_init__``.

The two coexist deliberately (CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD" +
Catalog #110 APPEND-ONLY HISTORICAL_PROVENANCE): the landed evaluator-admission
surface keeps its consumers; the thin unified currency is the new cross-path
ledger row.
"""

from __future__ import annotations

import datetime as _dt
import fcntl
import hashlib
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from tac.analysis.receiver_surface_metrics import (
    normalize_receiver_surface,
    receiver_surface_receiver_visible,
    receiver_surface_scorer_visible,
)
from tac.exact_eval_custody import is_sha256_hex
from tac.optimization.proxy_candidate_contract import PROXY_FALSE_AUTHORITY_FIELDS
from tac.score_geometry import CONTEST_REFERENCE_BYTES, contest_score

ACTION_EFFECT_SCHEMA = "nerv_action_effect.v1"
ACTION_EFFECT_LEDGER_SCHEMA = "nerv_action_effect_ledger.v1"
ACTION_COMMUTATOR_ROW_SCHEMA = "nerv_action_commutator_row.v1"

# ── Partner amendment #5 thin-currency schema constants ───────────────────
ACTION_EFFECT_V1_SCHEMA = "tac.action_effect.v1"
# The non-promotable authority marker carried on a thin ActionEffect when no
# more specific surface authority is known.  This is the deliberate alternative
# to spreading the canonical false-authority keys (score_claim / promotable /
# rank_or_kill_eligible / ready_for_exact_eval_dispatch) into a row that may
# travel inside substrate_artifact_metadata, where the harness custody
# validator refuses those nested keys (single-custody-surface rule).
ACTION_EFFECT_PLANNING_AUTHORITY = "planning_control_false_authority"
# fcntl ledger lock timeout (a single append is <10ms; 30s is generous under
# heavy sibling fan-out contention).
_ACTION_EFFECT_LEDGER_LOCK_TIMEOUT_SECONDS = 30
_ACTION_EFFECT_V1_FORBIDDEN_SCORE_CLAIM_KEYS = frozenset(
    {
        "score_claim",
        "score_claim_valid",
        "frontier_score_claim",
        "rank_or_kill_eligible",
        "promotable",
        "ready_for_exact_eval_dispatch",
        "official_score",
        "official_score_claim",
    }
)
_ACTION_EFFECT_V1_DECISIONS = frozenset({"accept", "reject", "not_applicable"})
_ACTION_EFFECT_V1_TAINT_STATUSES = frozenset({"clean", "tainted", "remediated", "unknown"})
_ACTION_EFFECT_V1_INVERSE_SOURCES = frozenset(
    {
        "segnet_margin_gradient",
        "segnet_margin_vjp",
        "segnet_target_margin_vjp",
        "target_margin_vjp",
        "support_projected_segnet_margin_vjp",
        "posenet_yuv6_gradient",
        "frame0_pose_nullseg",
        "joint_seg_pose_projection",
        "integer_receiver_line_search",
        "masked_residual",
        "receiver_surface_masked_rgb_residual_on_support",
        "scorer_causal_pixel_synthesis",
        "source_rgb_residual_copy",
        "path_tube_support",
        "path_tube_segnet_margin_frontier",
        "frame0_pose_temporal_path",
        "selector_temporal_path",
        "manual_pr110_replay",
        "qrgb_basis",
        "commutator_macro_action",
        "menu_ilp_candidate",
    }
)
_ACTION_EFFECT_V1_FRAME_INDICES = frozenset({0, 1, "both"})
_ACTION_EFFECT_V1_FRAME_INCIDENCES = frozenset({"pose_only", "seg_pose_joint"})
_ACTION_EFFECT_V1_CANDIDATE_STATUSES = frozenset({"generated", "measured", "selected", "rejected", "composed"})


class ScoreAuthority(StrEnum):
    """Common ActionEffect authority surfaces.

    The IR stores authority as a string so exact CPU/CUDA lane tags can carry
    archive/runtime identity, but these enum values define the canonical local
    and contest surfaces producers should use when they are available.
    """

    LIVE_MLX = "live_mlx"
    BATCH_LOCAL_LIVE_MLX = "batch_local_live_mlx"
    ROUND_STE_MLX = "round_ste_mlx"
    FAKEQUANT_MLX = "fakequant_mlx"
    PARSEBACK_MLX = "parseback_mlx"
    INFLATE_TORCH_CPU = "inflate_torch_cpu"
    INFLATE_TORCH_CUDA = "inflate_torch_cuda"
    CONTEST_CPU = "contest_cpu"
    CONTEST_CUDA = "contest_cuda"
    RECEIVER_CLOSED_FRONTIER_RATE_ATTACK = "receiver_closed_frontier_rate_attack"
    PLANNING = ACTION_EFFECT_PLANNING_AUTHORITY


class NormalizationScope(StrEnum):
    """Score-unit normalization scope for an ActionEffect row."""

    BATCH_LOCAL = "batch_local"
    FULL_VIDEO_EQUIV_ESTIMATE = "full_video_equiv_estimate"
    FULL_VIDEO_EXACT = "full_video_exact"


@dataclass(frozen=True)
class ReceiverSurfaceDelta:
    """Receiver-surface movement attached to one ActionEffect action id."""

    float_rgb_delta_linf: float | None = None
    uint8_changed_pixels: int | None = None
    seg_input_delta_linf: float | None = None
    posenet_input_delta_linf: float | None = None
    seg_argmax_changed_pixels: int | None = None
    seg_target_hard_won_count: int | None = None
    seg_target_hard_lost_count: int | None = None
    seg_wrong_to_target_count: int | None = None
    seg_wrong_to_wrong_count: int | None = None
    pose_output_l2_delta: float | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None = None) -> ReceiverSurfaceDelta:
        row = _v1_mapping(payload)
        return cls(
            float_rgb_delta_linf=_v1_first_float(row, "float_rgb_delta_linf", "receiver_float_rgb_delta_linf"),
            uint8_changed_pixels=_v1_first_int(
                row,
                "uint8_changed_pixels",
                "uint8_changed_count_region",
                "receiver_uint8_changed_pixels_region",
            ),
            seg_input_delta_linf=_v1_first_float(
                row,
                "seg_input_delta_linf",
                "seg_input_delta_linf_region",
                "segnet_input_delta_linf",
                "segnet_input_delta_linf_region",
            ),
            posenet_input_delta_linf=_v1_first_float(
                row,
                "posenet_input_delta_linf",
                "posenet_input_delta_linf_pair",
                "pose_input_delta_linf_pair",
            ),
            seg_argmax_changed_pixels=_v1_first_int(
                row,
                "seg_argmax_changed_pixels",
                "argmax_flipped_pixels_region",
                "argmax_flipped_pixels",
            ),
            seg_target_hard_won_count=_v1_first_int(
                row,
                "seg_target_hard_won_count",
                "hard_won_count",
                "target_hard_won_count",
                "wrong_to_target_count",
            ),
            seg_target_hard_lost_count=_v1_first_int(
                row,
                "seg_target_hard_lost_count",
                "target_hard_lost_count",
                "target_to_wrong_count",
            ),
            seg_wrong_to_target_count=_v1_first_int(
                row,
                "seg_wrong_to_target_count",
                "wrong_to_target",
                "wrong_to_target_count",
            ),
            seg_wrong_to_wrong_count=_v1_first_int(
                row,
                "seg_wrong_to_wrong_count",
                "wrong_to_wrong",
                "wrong_to_wrong_count",
            ),
            pose_output_l2_delta=_v1_first_float(
                row,
                "pose_output_l2_delta",
                "max_accepted_pose_output_delta_l2",
                "pose_output_delta_l2",
            ),
        )

    def __post_init__(self) -> None:
        for name, value in (
            ("float_rgb_delta_linf", self.float_rgb_delta_linf),
            ("seg_input_delta_linf", self.seg_input_delta_linf),
            ("posenet_input_delta_linf", self.posenet_input_delta_linf),
            ("pose_output_l2_delta", self.pose_output_l2_delta),
        ):
            _v1_validate_distortion(name, value)
        for name, value in (
            ("uint8_changed_pixels", self.uint8_changed_pixels),
            ("seg_argmax_changed_pixels", self.seg_argmax_changed_pixels),
            ("seg_target_hard_won_count", self.seg_target_hard_won_count),
            ("seg_target_hard_lost_count", self.seg_target_hard_lost_count),
            ("seg_wrong_to_target_count", self.seg_wrong_to_target_count),
            ("seg_wrong_to_wrong_count", self.seg_wrong_to_wrong_count),
        ):
            _v1_validate_optional_nonnegative_int(name, value)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluatorActionEffect:
    """Exact score movement for one receiver-visible evaluator action.

    Schema ``nerv_action_effect.v1``.  This is the landed receiver-surface
    admission row (formerly named ``ActionEffect``); it is produced via
    :func:`build_action_effect` and consumed by ``servo_lift``.  Renamed to
    free the ``ActionEffect`` name for the thin partner-amendment-#5 currency;
    no external caller imported the class by name.
    """

    action_id: str
    family: str
    authority: str
    producer: str
    consumer: str | None
    affected_pairs: tuple[int, ...]
    affected_regions: tuple[str, ...]
    payload_sections: tuple[str, ...]
    state_custody: Mapping[str, str | int | bool]
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
            math.sqrt(10.0 * float(self.new_d_pose)) - math.sqrt(10.0 * float(self.old_d_pose))
        )

    @property
    def rate_score_delta(self) -> float:
        return 25.0 * self.delta_bytes / float(self.reference_bytes)

    @property
    def delta_score_total(self) -> float:
        return self.delta_score_nonrate + self.rate_score_delta

    @property
    def exact_value_per_byte(self) -> float | None:
        if not _score_state_valid(self) or not _archive_byte_state_valid(self):
            return None
        if self.delta_bytes == 0:
            return None
        return -self.delta_score_nonrate / abs(float(self.delta_bytes))

    @property
    def old_score(self) -> float:
        try:
            return contest_score(
                self.old_d_seg,
                self.old_d_pose,
                self.old_bytes,
                reference_bytes=self.reference_bytes,
            )
        except ValueError:
            return math.nan

    @property
    def new_score(self) -> float:
        try:
            return contest_score(
                self.new_d_seg,
                self.new_d_pose,
                self.new_bytes,
                reference_bytes=self.reference_bytes,
            )
        except ValueError:
            return math.nan

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
                if key not in {"receiver_surface", "state_custody", "value_per_byte"}
            },
            "affected_pairs": list(self.affected_pairs),
            "affected_regions": list(self.affected_regions),
            "payload_sections": list(self.payload_sections),
            "state_custody": dict(self.state_custody),
            "receiver_surface": dict(self.receiver_surface),
            "old_score": _finite_value_or_none(self.old_score),
            "new_score": _finite_value_or_none(self.new_score),
            "delta_d_seg": _finite_value_or_none(self.delta_d_seg),
            "delta_d_pose": _finite_value_or_none(self.delta_d_pose),
            "delta_bytes": self.delta_bytes,
            "delta_score_nonrate": _finite_value_or_none(self.delta_score_nonrate),
            "rate_score_delta": _finite_value_or_none(self.rate_score_delta),
            "delta_score_total": _finite_value_or_none(self.delta_score_total),
            "byte_price": 25.0 / float(self.reference_bytes),
            "value_per_byte": _finite_value_or_none(self.exact_value_per_byte),
            "reported_value_per_byte": _finite_value_or_none(self.value_per_byte),
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

    effect = EvaluatorActionEffect(
        action_id=str(payload.get("action_id") or payload.get("id") or ""),
        family=_family(payload.get("family")),
        authority=str(payload.get("authority") or ""),
        producer=str(payload.get("producer") or ""),
        consumer=_optional_text(payload.get("consumer")),
        affected_pairs=tuple(_int_values(payload.get("affected_pairs"))),
        affected_regions=tuple(_text_values(payload.get("affected_regions"))),
        payload_sections=tuple(_text_values(payload.get("payload_sections"))),
        state_custody=_state_custody(payload.get("state_custody"), payload=payload),
        old_d_seg=_finite_float(payload.get("old_d_seg")),
        new_d_seg=_finite_float(payload.get("new_d_seg")),
        old_d_pose=_finite_float(payload.get("old_d_pose")),
        new_d_pose=_finite_float(payload.get("new_d_pose")),
        old_bytes=_nonnegative_int(payload.get("old_bytes")),
        new_bytes=_nonnegative_int(payload.get("new_bytes")),
        receiver_surface=_receiver_surface(payload.get("receiver_surface")),
        parseback_survived=payload.get("parseback_survived") is True,
        fakequant_survived=payload.get("fakequant_survived") is True,
        inflate_survived=(None if payload.get("inflate_survived") is None else payload.get("inflate_survived") is True),
        value_per_byte=_optional_finite_float(payload.get("value_per_byte")),
        reference_bytes=int(reference_bytes),
    )
    return effect.to_mapping(min_score_improvement=min_score_improvement)


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
        if isinstance(effect, Mapping) and effect.get("schema") == ACTION_EFFECT_SCHEMA
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
            if isinstance(row, Mapping) and row.get("schema") == ACTION_COMMUTATOR_ROW_SCHEMA
        ],
        "score_claim": False,
        **PROXY_FALSE_AUTHORITY_FIELDS,
    }


def _action_effect_blockers(
    effect: EvaluatorActionEffect,
    *,
    min_score_improvement: float,
) -> list[str]:
    blockers: list[str] = []
    if not effect.action_id:
        blockers.append("action_effect_action_id_missing")
    if not effect.producer:
        blockers.append("action_effect_producer_missing")
    if not effect.consumer:
        blockers.append("action_effect_consumer_missing")
    if effect.family not in {"hinerv", "snerv", "pact_nerv", "selector", "shared"}:
        blockers.append("action_effect_family_unknown")
    if not effect.authority:
        blockers.append("action_effect_authority_missing")
    if not _score_state_valid(effect):
        blockers.append("action_effect_score_state_invalid")
    if not _archive_byte_state_valid(effect):
        blockers.append("action_effect_archive_byte_state_invalid")
    if not _state_custodied(effect.state_custody):
        blockers.append("action_effect_state_custody_hash_missing")
    if not _receiver_visible(effect.receiver_surface):
        blockers.append("action_effect_receiver_surface_motion_missing")
    if not _scorer_surface_visible(effect.receiver_surface):
        blockers.append("action_effect_scorer_surface_motion_missing")
    if effect.fakequant_survived is not True:
        blockers.append("action_effect_fakequant_survival_missing")
    if effect.parseback_survived is not True:
        blockers.append("action_effect_parseback_survival_missing")
    if not _score_admissible(effect, min_score_improvement=min_score_improvement):
        blockers.append("action_effect_exact_score_delta_not_admissible")
    if not _byte_priced(effect):
        blockers.append("action_effect_byte_delta_not_priced")
    return _dedupe(blockers)


def _byte_priced(effect: EvaluatorActionEffect) -> bool:
    if not _archive_byte_state_valid(effect):
        return False
    if effect.delta_bytes == 0:
        return True
    if not _archive_custodied(effect.state_custody):
        return False
    value = effect.exact_value_per_byte
    return value is not None and math.isfinite(float(value))


def _finite_value_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        return None
    return candidate if math.isfinite(candidate) else None


def _score_admissible(
    effect: EvaluatorActionEffect,
    *,
    min_score_improvement: float,
) -> bool:
    if not _score_state_valid(effect):
        return False
    return effect.delta_score_total < -float(min_score_improvement)


def _score_state_valid(effect: EvaluatorActionEffect) -> bool:
    return all(
        math.isfinite(float(value)) and float(value) >= 0.0
        for value in (
            effect.old_d_seg,
            effect.new_d_seg,
            effect.old_d_pose,
            effect.new_d_pose,
        )
    ) and math.isfinite(float(effect.delta_score_total))


def _archive_byte_state_valid(effect: EvaluatorActionEffect) -> bool:
    return int(effect.old_bytes) >= 0 and int(effect.new_bytes) >= 0 and int(effect.reference_bytes) > 0


def _state_custodied(custody: Mapping[str, Any]) -> bool:
    for key in (
        "archive_sha256",
        "candidate_archive_sha256",
        "source_archive_sha256",
        "payload_sha256",
        "runtime_tree_sha256",
        "section_tree_sha256",
    ):
        value = custody.get(key)
        if is_sha256_hex(value):
            return True
    return False


def _archive_custodied(custody: Mapping[str, Any]) -> bool:
    return any(
        is_sha256_hex(custody.get(key))
        for key in (
            "archive_sha256",
            "candidate_archive_sha256",
            "source_archive_sha256",
        )
    )


def _receiver_visible(surface: Mapping[str, Any]) -> bool:
    return receiver_surface_receiver_visible(surface)


def _scorer_surface_visible(surface: Mapping[str, Any]) -> bool:
    return receiver_surface_scorer_visible(surface)


def _receiver_surface(value: Any) -> dict[str, float | int | bool | str]:
    return normalize_receiver_surface(value)


def _state_custody(
    value: Any,
    *,
    payload: Mapping[str, Any],
) -> dict[str, str | int | bool]:
    out: dict[str, str | int | bool] = {}
    if isinstance(value, Mapping):
        for key, raw in value.items():
            if is_sha256_hex(raw):
                out[str(key)] = raw
    for key in (
        "archive_sha256",
        "candidate_archive_sha256",
        "source_archive_sha256",
        "payload_sha256",
        "runtime_tree_sha256",
        "section_tree_sha256",
    ):
        raw = payload.get(key)
        if key not in out and is_sha256_hex(raw):
            out[key] = raw
    return out


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


# ════════════════════════════════════════════════════════════════════════
# Partner amendment #5 — THIN unified currency (schema tac.action_effect.v1)
# ════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class DeltaScores:
    """Exact nonlinear before/after contest-score deltas for one action.

    ``delta_score_nonrate`` is seg+pose movement only::

        100*(new_d_seg - old_d_seg) + (sqrt(10*new_d_pose) - sqrt(10*old_d_pose))

    ``delta_score_total`` adds the linear rate term when both byte counts are
    supplied; ``None`` when bytes are unknown (a distortion-only observation).
    ``value_per_byte`` is ``-delta_total / abs(delta_bytes)`` (positive ⇒ the
    action saved score per byte touched, whether it spent or removed bytes);
    ``None`` when bytes do not change.
    """

    delta_d_seg: float | None
    delta_d_pose: float | None
    delta_bytes: int | None
    delta_score_nonrate: float | None
    delta_score_total: float | None
    value_per_byte: float | None


def compute_delta_scores(
    old_d_seg: float | None,
    new_d_seg: float | None,
    old_d_pose: float | None,
    new_d_pose: float | None,
    old_bytes: int | None,
    new_bytes: int | None,
    *,
    reference_bytes: int = CONTEST_REFERENCE_BYTES,
) -> DeltaScores:
    """Return exact nonlinear contest-score deltas for one action.

    This is the SINGLE shared scoring computation BOTH score paths route
    through.  It calls :func:`tac.score_geometry.contest_score` so the exact
    sqrt-curvature pose term and the linear rate term are never re-derived.

    Semantics:

    * The nonrate delta requires both seg endpoints AND both pose endpoints.
      If any of the four distortion endpoints is ``None`` the nonrate delta is
      ``None`` (we never fabricate a missing endpoint as zero).
    * The total delta requires the nonrate delta AND both byte counts.  When
      bytes are unknown the total is ``None`` (a distortion-only row).
    * ``value_per_byte`` is defined only when bytes change:
      ``-delta_score_total / abs(delta_bytes)`` so a positive value means the
      action reduced score (good) per absolute byte changed.

    Distortion inputs must be non-negative + finite; byte counts non-negative;
    ``reference_bytes`` positive.
    """

    if reference_bytes <= 0:
        raise ValueError("reference_bytes must be positive")
    _v1_validate_distortion("old_d_seg", old_d_seg)
    _v1_validate_distortion("new_d_seg", new_d_seg)
    _v1_validate_distortion("old_d_pose", old_d_pose)
    _v1_validate_distortion("new_d_pose", new_d_pose)
    _v1_validate_bytes("old_bytes", old_bytes)
    _v1_validate_bytes("new_bytes", new_bytes)

    have_all_distortion = (
        old_d_seg is not None and new_d_seg is not None and old_d_pose is not None and new_d_pose is not None
    )
    delta_d_seg = (new_d_seg - old_d_seg) if (old_d_seg is not None and new_d_seg is not None) else None
    delta_d_pose = (new_d_pose - old_d_pose) if (old_d_pose is not None and new_d_pose is not None) else None
    delta_bytes = (new_bytes - old_bytes) if (old_bytes is not None and new_bytes is not None) else None

    delta_score_nonrate: float | None = None
    if have_all_distortion:
        # 100*Δd_seg via the seg term; sqrt-curvature pose delta via the pose
        # term — both pulled from contest_score() so the curvature matches the
        # canonical objective EXACTLY (no second linearized formula).
        seg_before = contest_score(old_d_seg, 0.0, 0, reference_bytes=reference_bytes)
        seg_after = contest_score(new_d_seg, 0.0, 0, reference_bytes=reference_bytes)
        pose_before = contest_score(0.0, old_d_pose, 0, reference_bytes=reference_bytes)
        pose_after = contest_score(0.0, new_d_pose, 0, reference_bytes=reference_bytes)
        delta_score_nonrate = (seg_after - seg_before) + (pose_after - pose_before)

    # The total is defined whenever the byte endpoints are known: it is the
    # full nonrate+rate delta when distortion is also known, or the rate-only
    # delta for a byte-only observation (e.g. a PR110 selector candidate that
    # only changed archive bytes).  When bytes are unknown but distortion is
    # known the total is None (no rate term to add); a distortion-only row's
    # movement lives entirely in ``delta_score_nonrate``.
    delta_score_total: float | None = None
    if old_bytes is not None and new_bytes is not None:
        if have_all_distortion:
            before = contest_score(old_d_seg, old_d_pose, old_bytes, reference_bytes=reference_bytes)
            after = contest_score(new_d_seg, new_d_pose, new_bytes, reference_bytes=reference_bytes)
            delta_score_total = after - before
        else:
            # Byte-only observation: only the linear rate term moves.
            delta_score_total = 25.0 * (new_bytes - old_bytes) / reference_bytes

    vp: float | None = None
    if delta_score_total is not None and delta_bytes is not None and delta_bytes != 0:
        vp = value_per_byte(delta_score_total, delta_bytes)

    return DeltaScores(
        delta_d_seg=delta_d_seg,
        delta_d_pose=delta_d_pose,
        delta_bytes=delta_bytes,
        delta_score_nonrate=delta_score_nonrate,
        delta_score_total=delta_score_total,
        value_per_byte=vp,
    )


def exact_delta_score(
    old_d_seg: float | None,
    new_d_seg: float | None,
    old_d_pose: float | None,
    new_d_pose: float | None,
    old_archive_bytes: int | None,
    new_archive_bytes: int | None,
    *,
    reference_bytes: int = CONTEST_REFERENCE_BYTES,
) -> float | None:
    """Return the exact nonlinear total score delta for one before/after action."""

    return compute_delta_scores(
        old_d_seg,
        new_d_seg,
        old_d_pose,
        new_d_pose,
        old_archive_bytes,
        new_archive_bytes,
        reference_bytes=reference_bytes,
    ).delta_score_total


def value_per_byte(delta_score_total: float | None, delta_bytes: int | None) -> float | None:
    """Return score units saved per absolute byte touched."""

    if delta_score_total is None or delta_bytes is None or int(delta_bytes) == 0:
        return None
    return -float(delta_score_total) / abs(float(delta_bytes))


@dataclass(frozen=True)
class ActionEffect:
    """ONE typed score-effect ledger row shared by both contest score paths.

    Schema ``tac.action_effect.v1``.  Every field is exactly the partner thin-IR
    spec.  Construction validates that ``authority`` is a non-empty string and
    that ``promotion_eligible`` is ``False`` (this is an analysis-layer row; it
    can never self-declare promotion).
    """

    schema: str
    action_id: str
    family: str
    action_kind: str = "unspecified_action"
    inverse_source: str | None = None
    frame_index: int | str | None = None
    frame_incidence: str | None = None
    candidate_status: str | None = None
    authority: str = ACTION_EFFECT_PLANNING_AUTHORITY
    normalization_scope: str = NormalizationScope.BATCH_LOCAL.value
    producer: str = "unknown"
    consumer: str | None = None
    pair_ids: tuple[int, ...] = ()
    class_ids: tuple[int, ...] = ()
    region_ids: tuple[str, ...] = ()
    payload_sections: tuple[str, ...] = ()
    trained_groups: tuple[str, ...] = ()
    old_d_seg: float | None = None
    new_d_seg: float | None = None
    old_d_pose: float | None = None
    new_d_pose: float | None = None
    old_bytes: int | None = None
    new_bytes: int | None = None
    delta_bytes: int | None = None
    delta_score_nonrate: float | None = None
    delta_score_total: float | None = None
    value_per_byte: float | None = None
    receiver_surface: ReceiverSurfaceDelta = field(default_factory=ReceiverSurfaceDelta)
    exact_score_decision: str = "not_applicable"
    raw_cap_decision: str | None = None
    catastrophic_guard_decision: str | None = None
    would_accept_exact_score_if_raw_cap_disabled: bool | None = None
    would_accept_without_catastrophic_guard: bool | None = None
    rejected_by_raw_cap: bool | None = None
    rejected_by_exact_score: bool | None = None
    rejected_by_catastrophic_guard: bool | None = None
    parseback_survived: bool | None = None
    inflate_survived: bool | None = None
    restore_state_pass: bool | None = None
    artifact_ref: str | None = None
    archive_sha256: str | None = None
    payload_sha256: str | None = None
    base_state_sha256: str | None = None
    evaluator_hash: str | None = None
    dependency_hash: str | None = None
    taint_status: str = "unknown"
    fakequant_survived: bool | None = None
    hard_won_count: int | None = None
    wrong_to_target: int | None = None
    target_to_wrong: int | None = None
    wrong_to_wrong: int | None = None
    net_target_support_delta: int | None = None
    uint8_changed_count_region: int | None = None
    seg_input_delta_linf_region: float | None = None
    posenet_input_delta_linf_pair: float | None = None
    support_source: str | None = None
    support_cardinality: int | None = None
    support_sha256: str | None = None
    support_encoding: str | None = None
    support_encoded_bytes: int | None = None
    support_research_only: bool | None = None
    arm: str | None = None
    old_region_debt: float | None = None
    new_region_debt: float | None = None
    argmax_changed_count_region: int | None = None
    pose_output_l2_delta: float | None = None
    seg_score_delta: float | None = None
    pose_score_delta: float | None = None
    segnet_margin_delta: float | None = None
    fakequant_segnet_margin_delta: float | None = None
    parseback_segnet_margin_delta: float | None = None
    rejection_source: str | None = None
    blockers: tuple[str, ...] = ()
    interaction_or_commutator: float | None = None
    promotion_eligible: bool = False

    def __post_init__(self) -> None:
        if self.schema != ACTION_EFFECT_V1_SCHEMA:
            raise ValueError(f"schema must be {ACTION_EFFECT_V1_SCHEMA!r}; got {self.schema!r}")
        if not isinstance(self.action_id, str) or not self.action_id.strip():
            raise ValueError("action_id must be a non-empty string")
        if not isinstance(self.action_kind, str) or not self.action_kind.strip():
            raise ValueError("action_kind must be a non-empty string")
        if self.inverse_source is not None and self.inverse_source not in _ACTION_EFFECT_V1_INVERSE_SOURCES:
            raise ValueError(
                "inverse_source must be one of "
                f"{sorted(_ACTION_EFFECT_V1_INVERSE_SOURCES)}; got {self.inverse_source!r}"
            )
        if self.frame_index is not None and self.frame_index not in _ACTION_EFFECT_V1_FRAME_INDICES:
            raise ValueError("frame_index must be 0, 1, 'both', or None")
        if self.frame_incidence is not None and self.frame_incidence not in _ACTION_EFFECT_V1_FRAME_INCIDENCES:
            raise ValueError(
                "frame_incidence must be one of "
                f"{sorted(_ACTION_EFFECT_V1_FRAME_INCIDENCES)}; got {self.frame_incidence!r}"
            )
        if self.candidate_status is not None and self.candidate_status not in _ACTION_EFFECT_V1_CANDIDATE_STATUSES:
            raise ValueError(
                "candidate_status must be one of "
                f"{sorted(_ACTION_EFFECT_V1_CANDIDATE_STATUSES)}; got {self.candidate_status!r}"
            )
        if not isinstance(self.authority, str) or not self.authority.strip():
            raise ValueError("authority is REQUIRED and must be a non-empty string")
        if self.normalization_scope not in {item.value for item in NormalizationScope}:
            raise ValueError(
                "normalization_scope is REQUIRED and must be one of "
                f"{sorted(item.value for item in NormalizationScope)}; got {self.normalization_scope!r}"
            )
        if self.promotion_eligible is not False:
            raise ValueError(
                "ActionEffect is an analysis-layer ledger row; promotion_eligible "
                "must always be False in v1 (only the exact-eval custody surface promotes)"
            )
        if _v1_authority_is_batch_local(self.authority) and self.normalization_scope != NormalizationScope.BATCH_LOCAL.value:
            raise ValueError("normalization_scope_mismatch: batch-local authority requires batch_local scope")
        if self.normalization_scope == NormalizationScope.BATCH_LOCAL.value and _v1_authority_is_promotional(self.authority):
            raise ValueError("normalization_scope_mismatch: promotional authority cannot be batch_local")
        if self.exact_score_decision not in _ACTION_EFFECT_V1_DECISIONS:
            raise ValueError(f"exact_score_decision must be one of {sorted(_ACTION_EFFECT_V1_DECISIONS)}")
        if self.taint_status not in _ACTION_EFFECT_V1_TAINT_STATUSES:
            raise ValueError(f"taint_status must be one of {sorted(_ACTION_EFFECT_V1_TAINT_STATUSES)}")
        for name, value in (
            ("old_d_seg", self.old_d_seg),
            ("new_d_seg", self.new_d_seg),
            ("old_d_pose", self.old_d_pose),
            ("new_d_pose", self.new_d_pose),
        ):
            _v1_validate_distortion(name, value)
        for name, value in (("old_bytes", self.old_bytes), ("new_bytes", self.new_bytes)):
            _v1_validate_bytes(name, value)
        expected_delta_bytes = (
            self.new_bytes - self.old_bytes
            if self.old_bytes is not None and self.new_bytes is not None
            else None
        )
        if self.delta_bytes != expected_delta_bytes:
            raise ValueError(
                f"delta_bytes must be computed from old/new bytes; got {self.delta_bytes!r}, "
                f"expected {expected_delta_bytes!r}"
            )
        if self.delta_bytes not in (None, 0) and self.value_per_byte is None:
            raise ValueError("value_per_byte_missing")
        if not isinstance(self.receiver_surface, ReceiverSurfaceDelta):
            raise ValueError("receiver_surface must be a ReceiverSurfaceDelta")
        for name, value in (
            ("hard_won_count", self.hard_won_count),
            ("wrong_to_target", self.wrong_to_target),
            ("target_to_wrong", self.target_to_wrong),
            ("wrong_to_wrong", self.wrong_to_wrong),
            ("uint8_changed_count_region", self.uint8_changed_count_region),
            ("argmax_changed_count_region", self.argmax_changed_count_region),
            ("support_cardinality", self.support_cardinality),
            ("support_encoded_bytes", self.support_encoded_bytes),
        ):
            _v1_validate_optional_nonnegative_int(name, value)
        _v1_validate_optional_int("net_target_support_delta", self.net_target_support_delta)
        if self.support_sha256 is not None and not is_sha256_hex(self.support_sha256):
            raise ValueError("support_sha256 must be a SHA-256 hex string or None")
        for name, value in (
            ("seg_input_delta_linf_region", self.seg_input_delta_linf_region),
            ("posenet_input_delta_linf_pair", self.posenet_input_delta_linf_pair),
            ("old_region_debt", self.old_region_debt),
            ("new_region_debt", self.new_region_debt),
            ("pose_output_l2_delta", self.pose_output_l2_delta),
        ):
            _v1_validate_distortion(name, value)
        for name, value in (
            ("seg_score_delta", self.seg_score_delta),
            ("pose_score_delta", self.pose_score_delta),
            ("segnet_margin_delta", self.segnet_margin_delta),
            ("fakequant_segnet_margin_delta", self.fakequant_segnet_margin_delta),
            ("parseback_segnet_margin_delta", self.parseback_segnet_margin_delta),
            ("interaction_or_commutator", self.interaction_or_commutator),
        ):
            _v1_validate_optional_finite_float(name, value)
        if not isinstance(self.pair_ids, tuple):
            raise ValueError("pair_ids must be a tuple")
        if not isinstance(self.class_ids, tuple):
            raise ValueError("class_ids must be a tuple")
        if not isinstance(self.region_ids, tuple):
            raise ValueError("region_ids must be a tuple")
        if not isinstance(self.payload_sections, tuple):
            raise ValueError("payload_sections must be a tuple")
        if not isinstance(self.trained_groups, tuple):
            raise ValueError("trained_groups must be a tuple")
        if not isinstance(self.blockers, tuple):
            raise ValueError("blockers must be a tuple")
        for name, value in (
            ("archive_sha256", self.archive_sha256),
            ("payload_sha256", self.payload_sha256),
            ("base_state_sha256", self.base_state_sha256),
            ("evaluator_hash", self.evaluator_hash),
            ("dependency_hash", self.dependency_hash),
        ):
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{name} must be a string or None")

    # ── construction ──────────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        *,
        action_id: str,
        family: str,
        action_kind: str = "unspecified_action",
        inverse_source: str | None = None,
        frame_index: int | str | None = None,
        frame_incidence: str | None = None,
        candidate_status: str | None = None,
        authority: str,
        normalization_scope: str | None = None,
        producer: str,
        consumer: str | None = None,
        pair_ids: Sequence[int] = (),
        class_ids: Sequence[int] = (),
        region_ids: Sequence[str] = (),
        payload_sections: Sequence[str] = (),
        trained_groups: Sequence[str] = (),
        old_d_seg: float | None = None,
        new_d_seg: float | None = None,
        old_d_pose: float | None = None,
        new_d_pose: float | None = None,
        old_bytes: int | None = None,
        new_bytes: int | None = None,
        receiver_surface: ReceiverSurfaceDelta | Mapping[str, Any] | None = None,
        exact_score_decision: str = "not_applicable",
        raw_cap_decision: str | None = None,
        catastrophic_guard_decision: str | None = None,
        would_accept_exact_score_if_raw_cap_disabled: bool | None = None,
        would_accept_without_catastrophic_guard: bool | None = None,
        rejected_by_raw_cap: bool | None = None,
        rejected_by_exact_score: bool | None = None,
        rejected_by_catastrophic_guard: bool | None = None,
        parseback_survived: bool | None = None,
        inflate_survived: bool | None = None,
        restore_state_pass: bool | None = None,
        artifact_ref: str | None = None,
        archive_sha256: str | None = None,
        payload_sha256: str | None = None,
        base_state_sha256: str | None = None,
        evaluator_hash: str | None = None,
        dependency_hash: str | None = None,
        taint_status: str = "unknown",
        fakequant_survived: bool | None = None,
        hard_won_count: int | None = None,
        wrong_to_target: int | None = None,
        target_to_wrong: int | None = None,
        wrong_to_wrong: int | None = None,
        net_target_support_delta: int | None = None,
        uint8_changed_count_region: int | None = None,
        seg_input_delta_linf_region: float | None = None,
        posenet_input_delta_linf_pair: float | None = None,
        support_source: str | None = None,
        support_cardinality: int | None = None,
        support_sha256: str | None = None,
        support_encoding: str | None = None,
        support_encoded_bytes: int | None = None,
        support_research_only: bool | None = None,
        arm: str | None = None,
        old_region_debt: float | None = None,
        new_region_debt: float | None = None,
        argmax_changed_count_region: int | None = None,
        pose_output_l2_delta: float | None = None,
        seg_score_delta: float | None = None,
        pose_score_delta: float | None = None,
        segnet_margin_delta: float | None = None,
        fakequant_segnet_margin_delta: float | None = None,
        parseback_segnet_margin_delta: float | None = None,
        rejection_source: str | None = None,
        blockers: Sequence[str] = (),
        interaction_or_commutator: float | None = None,
        reference_bytes: int = CONTEST_REFERENCE_BYTES,
    ) -> ActionEffect:
        """Build an ActionEffect, computing the shared delta scores.

        Canonical constructor every ``from_*`` helper funnels through so the
        one scoring computation (:func:`compute_delta_scores`) is never
        duplicated.
        """

        normalized_authority = str(authority)
        normalized_scope = (
            str(normalization_scope)
            if normalization_scope is not None
            else _v1_default_normalization_scope(normalized_authority)
        )
        deltas = compute_delta_scores(
            old_d_seg,
            new_d_seg,
            old_d_pose,
            new_d_pose,
            old_bytes,
            new_bytes,
            reference_bytes=reference_bytes,
        )
        return cls(
            schema=ACTION_EFFECT_V1_SCHEMA,
            action_id=str(action_id),
            family=str(family),
            action_kind=str(action_kind),
            inverse_source=None if inverse_source is None else str(inverse_source),
            frame_index=_v1_frame_index_or_none(frame_index),
            frame_incidence=None if frame_incidence is None else str(frame_incidence),
            candidate_status=None if candidate_status is None else str(candidate_status),
            authority=normalized_authority,
            normalization_scope=normalized_scope,
            producer=str(producer),
            consumer=None if consumer is None else str(consumer),
            pair_ids=_v1_int_tuple(pair_ids),
            class_ids=_v1_int_tuple(class_ids),
            region_ids=_v1_str_tuple(region_ids),
            payload_sections=_v1_str_tuple(payload_sections),
            trained_groups=_v1_str_tuple(trained_groups),
            old_d_seg=old_d_seg,
            new_d_seg=new_d_seg,
            old_d_pose=old_d_pose,
            new_d_pose=new_d_pose,
            old_bytes=old_bytes,
            new_bytes=new_bytes,
            delta_bytes=deltas.delta_bytes,
            delta_score_nonrate=deltas.delta_score_nonrate,
            delta_score_total=deltas.delta_score_total,
            value_per_byte=deltas.value_per_byte,
            receiver_surface=(
                receiver_surface
                if isinstance(receiver_surface, ReceiverSurfaceDelta)
                else ReceiverSurfaceDelta.from_mapping(receiver_surface)
            ),
            exact_score_decision=_v1_normalize_decision(exact_score_decision),
            raw_cap_decision=None if raw_cap_decision is None else str(raw_cap_decision),
            catastrophic_guard_decision=(
                None if catastrophic_guard_decision is None else str(catastrophic_guard_decision)
            ),
            would_accept_exact_score_if_raw_cap_disabled=_v1_bool_or_none(
                would_accept_exact_score_if_raw_cap_disabled
            ),
            would_accept_without_catastrophic_guard=_v1_bool_or_none(
                would_accept_without_catastrophic_guard
            ),
            rejected_by_raw_cap=_v1_bool_or_none(rejected_by_raw_cap),
            rejected_by_exact_score=_v1_bool_or_none(rejected_by_exact_score),
            rejected_by_catastrophic_guard=_v1_bool_or_none(rejected_by_catastrophic_guard),
            parseback_survived=parseback_survived,
            inflate_survived=inflate_survived,
            restore_state_pass=restore_state_pass,
            artifact_ref=None if artifact_ref is None else str(artifact_ref),
            archive_sha256=None if archive_sha256 is None else str(archive_sha256),
            payload_sha256=None if payload_sha256 is None else str(payload_sha256),
            base_state_sha256=None if base_state_sha256 is None else str(base_state_sha256),
            evaluator_hash=None if evaluator_hash is None else str(evaluator_hash),
            dependency_hash=None if dependency_hash is None else str(dependency_hash),
            taint_status=str(taint_status),
            fakequant_survived=fakequant_survived,
            hard_won_count=hard_won_count,
            wrong_to_target=wrong_to_target,
            target_to_wrong=target_to_wrong,
            wrong_to_wrong=wrong_to_wrong,
            net_target_support_delta=net_target_support_delta,
            uint8_changed_count_region=uint8_changed_count_region,
            seg_input_delta_linf_region=seg_input_delta_linf_region,
            posenet_input_delta_linf_pair=posenet_input_delta_linf_pair,
            support_source=None if support_source is None else str(support_source),
            support_cardinality=support_cardinality,
            support_sha256=None if support_sha256 is None else str(support_sha256),
            support_encoding=None if support_encoding is None else str(support_encoding),
            support_encoded_bytes=support_encoded_bytes,
            support_research_only=_v1_bool_or_none(support_research_only),
            arm=None if arm is None else str(arm),
            old_region_debt=old_region_debt,
            new_region_debt=new_region_debt,
            argmax_changed_count_region=argmax_changed_count_region,
            pose_output_l2_delta=pose_output_l2_delta,
            seg_score_delta=seg_score_delta,
            pose_score_delta=pose_score_delta,
            segnet_margin_delta=segnet_margin_delta,
            fakequant_segnet_margin_delta=fakequant_segnet_margin_delta,
            parseback_segnet_margin_delta=parseback_segnet_margin_delta,
            rejection_source=None if rejection_source is None else str(rejection_source),
            blockers=_v1_str_tuple(blockers),
            interaction_or_commutator=interaction_or_commutator,
            promotion_eligible=False,
        )

    @classmethod
    def from_hinerv_birth_receipt(
        cls,
        receipt_or_payload: Mapping[str, Any],
        *,
        consumer: str | None = "nerv_long_training_campaign_admission",
        reference_bytes: int = CONTEST_REFERENCE_BYTES,
    ) -> ActionEffect:
        """Build an ActionEffect from a HiNeRV target-region birth receipt.

        Maps the landed ``hi_nerv_target_region_birth_receipt.v1`` schema:

        * ``action_id`` ← the receipt's stable ``action_id`` (CARRIED, not
          recomputed; falls back to ``actuator_id``).
        * exact nonrate distortion endpoints ← the ``exact_nonrate`` block.
          When the receipt has no pose teacher that block is absent → endpoints
          are ``None`` (a receiver-motion-only receipt) and the nonrate delta is
          ``None``.
        * ``region_ids`` ← the worst region as ``"b{batch}/c{class}/r{label}"``.
        * ``authority`` ← the receipt ``surface`` (e.g. ``fakequant_mlx`` /
          ``parseback_mlx`` / ``inflated_torch_cpu``), falling back to the
          receipt ``authority`` marker.
        * ``payload_sections`` ← ``updated_parameter_names``; ``trained_groups``
          carries the coarser action identity/training groups. Byte delta comes
          from ``runtime_sidecar_bytes`` (new-state-only unless explicit
          before/after).
        """

        if not isinstance(receipt_or_payload, Mapping):
            raise TypeError("birth receipt must be a mapping")
        receipt = receipt_or_payload

        action_id = _v1_first_text(receipt, "action_id", "actuator_id") or "hinerv_target_region_birth"
        surface = _v1_first_text(receipt, "surface")
        authority = _v1_hinerv_birth_authority(receipt, surface)
        normalization_scope = (
            _v1_first_text(receipt, "normalization_scope")
            or _v1_default_normalization_scope(authority)
        )
        action_kind = (
            _v1_first_text(receipt, "action_kind", "ablation_arm", "arm")
            or "target_region_birth"
        )

        worst = _v1_mapping(receipt.get("worst_region"))
        region_id = _v1_birth_region_id(worst)
        region_ids = (region_id,) if region_id else ()
        pair_ids = _v1_birth_pair_ids(receipt, worst)
        class_id = _v1_first_int(worst, "class_index")
        class_ids = () if class_id is None else (class_id,)

        exact = _v1_mapping(receipt.get("exact_nonrate"))
        old_d_seg = _v1_first_float(
            exact,
            "old_d_seg",
            "old_d_seg_batch",
            "d_seg_old",
            "old_segnet_distortion",
        )
        new_d_seg = _v1_first_float(
            exact,
            "new_d_seg",
            "new_d_seg_batch",
            "d_seg_new",
            "new_segnet_distortion",
        )
        old_d_pose = _v1_first_float(
            exact,
            "old_d_pose",
            "old_d_pose_batch",
            "d_pose_old",
            "old_posenet_distortion",
        )
        new_d_pose = _v1_first_float(
            exact,
            "new_d_pose",
            "new_d_pose_batch",
            "d_pose_new",
            "new_posenet_distortion",
        )

        old_bytes = _v1_first_int(receipt, "old_archive_bytes", "archive_bytes_old")
        new_bytes = _v1_first_int(receipt, "new_archive_bytes", "archive_bytes_new")
        sidecar = _v1_first_int(receipt, "runtime_sidecar_bytes")
        if new_bytes is None and old_bytes is not None and sidecar is not None:
            new_bytes = old_bytes + sidecar

        payload_sections = _v1_str_tuple(receipt.get("updated_parameter_names"))
        trained_groups = _v1_str_tuple(receipt.get("trained_groups"))
        if not trained_groups:
            trained_groups = _v1_str_tuple(receipt.get("action_trained_groups"))
        parseback_survived, inflate_survived = _v1_birth_survival_flags(receipt, surface)
        transitions = _v1_mapping(receipt.get("argmax_transitions"))
        if not transitions:
            transitions = _v1_mapping(receipt.get("region_argmax_transitions"))
        if not transitions:
            transitions = _v1_mapping(receipt.get("argmax_transition_counts"))
        transition_values = {**dict(receipt), **dict(transitions)}
        receiver_surface = _v1_mapping(receipt.get("receiver_surface"))
        pose_guard = _v1_mapping(receipt.get("pose_guard"))
        receiver_uint8_delta_abs_max = _v1_first_float(
            receipt,
            "receiver_uint8_delta_abs_max",
            "receiver_surface_uint8_delta_abs_max",
        )
        action_section_telemetry = _v1_mapping(
            receipt.get("target_region_action_section_telemetry")
        )
        support_identity = _v1_hinerv_birth_support_identity(
            receipt,
            action_section_telemetry=action_section_telemetry,
        )
        before_margin_stats = _v1_mapping(receipt.get("before_region_margin_stats"))
        after_margin_stats = _v1_mapping(receipt.get("after_region_margin_stats"))
        before_margin_p50 = _v1_first_float(before_margin_stats, "margin_p50")
        after_margin_p50 = _v1_first_float(after_margin_stats, "margin_p50")
        margin_p50_delta = _v1_first_float(
            receipt,
            "segnet_margin_delta",
            "target_margin_delta",
            "worst_region_margin_p50_delta",
            "receiver_surface_worst_region_margin_p50_delta",
        )
        if margin_p50_delta is None and before_margin_p50 is not None and after_margin_p50 is not None:
            margin_p50_delta = after_margin_p50 - before_margin_p50
        receiver_delta_linf = None if receiver_uint8_delta_abs_max is None else receiver_uint8_delta_abs_max / 255.0
        admission_decision = _v1_mapping(receipt.get("admission_decision"))
        if not admission_decision:
            telemetry = _v1_mapping(receipt.get("candidate_frontier_telemetry"))
            attempts = telemetry.get("attempts")
            if isinstance(attempts, Sequence) and not isinstance(attempts, (str, bytes)):
                attempt_rows = [item for item in attempts if isinstance(item, Mapping)]
                accepted_attempt = next(
                    (
                        item
                        for item in attempt_rows
                        if item.get("decision") in {
                            "accepted",
                            "accepted_with_frame0_pose_compensation",
                        }
                        or item.get("accepted") is True
                    ),
                    None,
                )
                if accepted_attempt is None:
                    accepted_attempt = next(
                        (
                            item
                            for item in attempt_rows
                            if item.get("exact_score_decision") == "accepted"
                            and item.get("catastrophic_guard_decision") == "satisfied"
                        ),
                        None,
                    )
                if accepted_attempt is None and attempt_rows:
                    accepted_attempt = attempt_rows[-1]
                if isinstance(accepted_attempt, Mapping):
                    admission_decision = dict(accepted_attempt)
        fakequant_survived = _v1_bool_or_none(receipt.get("fakequant_survived"))
        if fakequant_survived is None and surface == "fakequant_mlx":
            blockers = receipt.get("blockers")
            if isinstance(blockers, Sequence) and not isinstance(blockers, (str, bytes)) and len(blockers) == 0:
                fakequant_survived = True
        receipt_blockers = _v1_str_tuple(receipt.get("blockers") or ())
        raw_cap_decision = (
            _v1_first_text(admission_decision, "raw_cap_decision")
            or _v1_first_text(receipt, "raw_cap_decision")
            or ("not_applicable" if receipt_blockers else None)
        )
        catastrophic_guard_decision = (
            _v1_first_text(
                admission_decision,
                "catastrophic_guard_decision",
            )
            or _v1_first_text(receipt, "catastrophic_guard_decision")
            or ("not_applicable" if receipt_blockers else None)
        )
        would_accept_exact_score_if_raw_cap_disabled = _v1_bool_or_none(
            admission_decision.get("would_accept_exact_score_if_raw_cap_disabled")
            if admission_decision
            else receipt.get("would_accept_exact_score_if_raw_cap_disabled")
        )
        if would_accept_exact_score_if_raw_cap_disabled is None and receipt_blockers:
            would_accept_exact_score_if_raw_cap_disabled = False

        return cls.build(
            action_id=action_id,
            family="hinerv",
            action_kind=action_kind,
            authority=authority,
            normalization_scope=normalization_scope,
            producer="hinerv_target_region_birth",
            consumer=consumer,
            pair_ids=pair_ids,
            class_ids=class_ids,
            region_ids=region_ids,
            payload_sections=payload_sections,
            trained_groups=trained_groups,
            old_d_seg=old_d_seg,
            new_d_seg=new_d_seg,
            old_d_pose=old_d_pose,
            new_d_pose=new_d_pose,
            old_bytes=old_bytes,
            new_bytes=new_bytes,
            receiver_surface={
                **dict(receipt),
                **dict(receiver_surface),
                **dict(pose_guard),
                **dict(transition_values),
                "float_rgb_delta_linf": _v1_first_float(
                    receipt,
                    "receiver_float_rgb_delta_linf",
                    "float_rgb_delta_linf",
                ),
                "uint8_changed_pixels": _v1_first_int(
                    receipt,
                    "receiver_uint8_changed_pixels_region",
                    "receiver_surface_uint8_changed_pixels",
                    "uint8_changed_pixels_region",
                    "uint8_changed_count_region",
                ),
                "seg_input_delta_linf": _v1_first_float(
                    receiver_surface,
                    "seg_input_delta_linf_region",
                    "segnet_input_delta_linf_region",
                    "segnet_input_delta_linf",
                )
                or receiver_delta_linf,
                "posenet_input_delta_linf": _v1_first_float(
                    pose_guard,
                    "posenet_input_delta_linf_pair",
                    "pose_input_delta_linf_pair",
                )
                or receiver_delta_linf,
            },
            exact_score_decision=_v1_first_text(admission_decision, "exact_score_decision")
            or _v1_first_text(receipt, "exact_score_decision")
            or (
                "accept" if _v1_first_int(receipt, "accepted_step_count") and _v1_first_int(receipt, "accepted_step_count") > 0 else "reject"
            ),
            raw_cap_decision=_v1_first_text(admission_decision, "raw_cap_decision")
            or raw_cap_decision,
            catastrophic_guard_decision=catastrophic_guard_decision,
            would_accept_exact_score_if_raw_cap_disabled=would_accept_exact_score_if_raw_cap_disabled,
            would_accept_without_catastrophic_guard=_v1_bool_or_none(
                admission_decision.get("would_accept_without_catastrophic_guard")
                if admission_decision
                else receipt.get("would_accept_without_catastrophic_guard")
            ),
            rejected_by_raw_cap=_v1_bool_or_none(
                _v1_first_bool(
                    admission_decision,
                    "rejected_by_raw_cap",
                    "rejected_by_raw_pose_cap",
                )
                if admission_decision
                else _v1_first_bool(
                    receipt,
                    "rejected_by_raw_cap",
                    "rejected_by_raw_pose_cap",
                )
            ),
            rejected_by_exact_score=_v1_bool_or_none(
                _v1_first_bool(
                    admission_decision,
                    "rejected_by_exact_score",
                    "rejected_by_exact_delta_score",
                )
                if admission_decision
                else _v1_first_bool(
                    receipt,
                    "rejected_by_exact_score",
                    "rejected_by_exact_delta_score",
                )
            ),
            rejected_by_catastrophic_guard=_v1_bool_or_none(
                _v1_first_bool(
                    admission_decision,
                    "rejected_by_catastrophic_guard",
                    "rejected_by_catastrophic_pose_guard",
                )
                if admission_decision
                else _v1_first_bool(
                    receipt,
                    "rejected_by_catastrophic_guard",
                    "rejected_by_catastrophic_pose_guard",
                )
            ),
            parseback_survived=parseback_survived,
            inflate_survived=inflate_survived,
            restore_state_pass=_v1_first_bool(receipt, "restore_state_pass", "restore_state_passed"),
            artifact_ref=_v1_first_text(receipt, "artifact_ref", "artifact_path"),
            archive_sha256=_v1_first_text(receipt, "archive_sha256", "candidate_archive_sha256"),
            payload_sha256=_v1_first_text(receipt, "payload_sha256"),
            base_state_sha256=_v1_base_state_sha256(receipt),
            evaluator_hash=_v1_first_text(receipt, "evaluator_hash", "scorer_hash"),
            dependency_hash=_v1_first_text(receipt, "dependency_hash"),
            taint_status=_v1_first_text(receipt, "taint_status") or "unknown",
            fakequant_survived=fakequant_survived,
            hard_won_count=_v1_first_int(
                transition_values,
                "target_hard_won_count",
                "wrong_to_target_count",
                "receiver_surface_target_hard_won_count",
            ),
            wrong_to_target=_v1_first_int(
                transition_values,
                "wrong_to_target_count",
                "target_hard_won_count",
                "receiver_surface_wrong_to_target_count",
            ),
            target_to_wrong=_v1_first_int(
                transition_values,
                "target_to_wrong_count",
                "target_hard_lost_count",
                "receiver_surface_target_to_wrong_count",
            ),
            wrong_to_wrong=_v1_first_int(transition_values, "wrong_to_wrong_count"),
            net_target_support_delta=_v1_first_int(
                transition_values,
                "net_target_support_delta",
                "receiver_surface_net_target_support_delta",
            ),
            uint8_changed_count_region=_v1_first_int(
                receipt,
                "receiver_uint8_changed_pixels_region",
                "receiver_surface_uint8_changed_pixels",
                "uint8_changed_pixels_region",
                "uint8_changed_count_region",
            ),
            arm=_v1_first_text(admission_decision, "arm", "ablation_arm", "action_kind")
            or _v1_first_text(receipt, "arm", "ablation_arm", "action_kind"),
            old_region_debt=_v1_first_float(
                receipt,
                "old_region_debt",
                "raw_region_debt_old",
                "before_region_debt",
            ),
            new_region_debt=_v1_first_float(
                receipt,
                "new_region_debt",
                "raw_region_debt_new",
                "after_region_debt",
            ),
            argmax_changed_count_region=_v1_first_int(
                receipt,
                "argmax_changed_count_region",
                "argmax_flipped_pixels_region",
                "argmax_flipped_pixels",
            ),
            pose_output_l2_delta=_v1_first_float(
                admission_decision,
                "pose_output_l2_delta",
                "composite_pose_output_delta_l2",
                "pose_output_delta_l2",
            )
            or _v1_first_float(
                pose_guard,
                "max_accepted_pose_output_delta_l2",
                "pose_output_l2_delta",
                "pose_output_delta_l2",
            ),
            seg_score_delta=_v1_first_float(admission_decision, "seg_score_delta")
            or (None if old_d_seg is None or new_d_seg is None else 100.0 * (new_d_seg - old_d_seg)),
            pose_score_delta=_v1_first_float(admission_decision, "pose_score_delta")
            or _v1_pose_score_delta(old_d_pose, new_d_pose),
            segnet_margin_delta=margin_p50_delta,
            fakequant_segnet_margin_delta=_v1_first_float(
                receipt,
                "fakequant_segnet_margin_delta",
                "fakequant_worst_region_margin_p50_delta",
            ),
            parseback_segnet_margin_delta=_v1_first_float(
                receipt,
                "parseback_segnet_margin_delta",
                "parseback_worst_region_margin_p50_delta",
            ),
            rejection_source=_v1_first_text(admission_decision, "rejection_source")
            or _v1_first_text(receipt, "rejection_source"),
            blockers=receipt_blockers,
            interaction_or_commutator=_v1_first_float(
                receipt,
                "interaction_or_commutator",
                "commutator",
                "commutator_delta_score_nonrate",
            ),
            seg_input_delta_linf_region=_v1_first_float(
                receiver_surface,
                "seg_input_delta_linf_region",
                "segnet_input_delta_linf_region",
                "segnet_input_delta_linf",
            )
            or receiver_delta_linf,
            posenet_input_delta_linf_pair=_v1_first_float(
                pose_guard,
                "posenet_input_delta_linf_pair",
                "pose_input_delta_linf_pair",
            )
            or receiver_delta_linf,
            support_source=_v1_first_text(support_identity, "support_source"),
            support_cardinality=_v1_first_int(support_identity, "support_cardinality"),
            support_sha256=_v1_first_text(support_identity, "support_sha256"),
            support_encoding=_v1_first_text(support_identity, "support_encoding"),
            support_encoded_bytes=_v1_first_int(support_identity, "support_encoded_bytes"),
            support_research_only=(
                None
                if support_identity
                else (
                    True
                    if region_ids and action_kind in {"target_region_birth", "birth_only"}
                    else None
                )
            ),
            reference_bytes=reference_bytes,
        )

    @classmethod
    def from_hinerv_four_arm_ablation(
        cls,
        payload: Mapping[str, Any],
        *,
        consumer: str | None = "nerv_long_run_launch_gate",
        reference_bytes: int = CONTEST_REFERENCE_BYTES,
    ) -> tuple[ActionEffect, ...]:
        """Build one ActionEffect per measured HiNeRV v6 four-arm row.

        The renderer emits a bundle under ``four_arm_ablation`` on the live
        birth payload.  Each arm is receipt-shaped and is imported through the
        same HiNeRV birth constructor so score math, receiver-surface aliases,
        authority, and region ids cannot drift into a parallel adapter.
        """

        if not isinstance(payload, Mapping):
            raise TypeError("four-arm ablation payload must be a mapping")
        bundle = payload.get("four_arm_ablation") if "four_arm_ablation" in payload else payload
        bundle = _v1_mapping(bundle)
        arms = bundle.get("arms")
        if not isinstance(arms, Sequence) or isinstance(arms, (str, bytes)):
            return ()
        root_action_id = _v1_first_text(bundle, "action_id") or _v1_first_text(payload, "action_id")
        root_base_state = (
            bundle.get("parameter_group_sha256_before")
            if isinstance(bundle.get("parameter_group_sha256_before"), Mapping)
            else payload.get("parameter_group_sha256_before")
        )
        root_base_state_sha256 = _v1_base_state_sha256(bundle) or _v1_base_state_sha256(payload)
        out: list[ActionEffect] = []
        for raw in arms:
            if not isinstance(raw, Mapping):
                continue
            arm_row = dict(raw)
            if root_action_id and not arm_row.get("action_id"):
                arm_row["action_id"] = root_action_id
            if root_base_state_sha256 and not arm_row.get("base_state_sha256"):
                arm_row["base_state_sha256"] = root_base_state_sha256
            if root_base_state and not arm_row.get("parameter_group_sha256_before"):
                arm_row["parameter_group_sha256_before"] = root_base_state
            out.append(
                cls.from_hinerv_birth_receipt(
                    arm_row,
                    consumer=consumer,
                    reference_bytes=reference_bytes,
                )
            )
        return tuple(out)

    @classmethod
    def from_pair_local_admission(
        cls,
        admission: Mapping[str, Any],
        *,
        consumer: str | None = "nerv_long_training_campaign_admission",
        reference_bytes: int = CONTEST_REFERENCE_BYTES,
    ) -> ActionEffect:
        """Build an ActionEffect from a pair-local servo admission / receipt row.

        Consumes both the ``nerv_pair_local_distortion_servo_admission.v1`` shape
        produced by ``PairLocalServoAdmission.as_dict()`` AND the canonical servo
        *receipt* (``nerv_pair_local_distortion_servo_receipt.v1``) the long-run
        admission path uses.  Absolute distortion endpoints are recovered via the
        canonical servo alias vocabulary (``old_d_seg`` / ``old_segnet_distortion``
        / ``new_d_pose`` / ``new_posenet_distortion`` / ``old_archive_bytes`` …) —
        the SAME aliases the servo's own ``_score_state_from_mapping`` reads — so a
        receipt round-trips with real d's and the shared
        :func:`compute_delta_scores` reproduces the SAME numbers the servo kernel
        computed.

        A bare admission ``as_dict`` stores ONLY deltas (``delta_d_seg`` /
        ``delta_d_pose``) with no absolute anchor; those cannot be turned into
        absolutes without fabricating an endpoint, so for a deltas-only admission
        the distortion endpoints are ``None`` (the nonrate delta is ``None``,
        never invented).  The structural fields (action_id, pair_ids, authority,
        survival, byte delta) still round-trip.
        """

        if not isinstance(admission, Mapping):
            raise TypeError("admission must be a mapping")

        trace = _v1_mapping(admission.get("trace"))
        action_id = (
            _v1_first_text(admission, "actuator_id")
            or _v1_first_text(trace, "actuator_id")
            or "pair_local_servo_action"
        )
        family = _v1_first_text(admission, "family") or _v1_first_text(trace, "family") or "shared"
        authority = (
            _v1_first_text(admission, "authority")
            or _v1_first_text(admission, "axis_tag")
            or ACTION_EFFECT_PLANNING_AUTHORITY
        )

        pair_ids = _v1_int_tuple(
            admission.get("pair_ids")
            or admission.get("affected_pairs")
            or trace.get("pair_ids")
            or trace.get("affected_pairs")
            or ()
        )
        if not pair_ids:
            pair_index = _v1_first_int(admission, "pair_index")
            if pair_index is None:
                pair_index = _v1_first_int(trace, "pair_index")
            pair_ids = (pair_index,) if pair_index is not None else ()

        # Canonical servo score-state aliases (receipt surface).  A deltas-only
        # admission row carries none of these, leaving endpoints None.
        old_d_seg = _v1_first_float(admission, "old_d_seg", "d_seg_old", "old_segnet_distortion", "segnet_distortion_old")
        new_d_seg = _v1_first_float(admission, "new_d_seg", "d_seg_new", "new_segnet_distortion", "segnet_distortion_new")
        old_d_pose = _v1_first_float(
            admission, "old_d_pose", "d_pose_old", "old_posenet_distortion", "posenet_distortion_old"
        )
        new_d_pose = _v1_first_float(
            admission, "new_d_pose", "d_pose_new", "new_posenet_distortion", "posenet_distortion_new"
        )
        old_bytes, new_bytes = _v1_recover_byte_endpoints(admission)

        surfaces = _v1_mapping(admission.get("surfaces"))
        parseback_survived = _v1_bool_or_none(surfaces.get("parseback_survival"))
        inflate_survived = _v1_bool_or_none(surfaces.get("inflate_survival"))
        fakequant_survived = _v1_bool_or_none(surfaces.get("fakequant_survival"))
        if fakequant_survived is None:
            fakequant_survived = _v1_bool_or_none(admission.get("fakequant_survived"))
        transition_values = {**dict(trace), **dict(admission)}

        region_ids = _v1_str_tuple(admission.get("affected_regions"))
        payload_sections = _v1_str_tuple(admission.get("payload_sections"))

        return cls.build(
            action_id=action_id,
            family="hinerv" if family == "hi_nerv" else family,
            action_kind=_v1_first_text(admission, "action_kind") or "pair_local_servo",
            authority=authority,
            normalization_scope=(
                _v1_first_text(admission, "normalization_scope")
                or _v1_default_normalization_scope(authority)
            ),
            producer="nerv_pair_local_distortion_servo",
            consumer=consumer,
            pair_ids=pair_ids,
            region_ids=region_ids,
            payload_sections=payload_sections,
            old_d_seg=old_d_seg,
            new_d_seg=new_d_seg,
            old_d_pose=old_d_pose,
            new_d_pose=new_d_pose,
            old_bytes=old_bytes,
            new_bytes=new_bytes,
            receiver_surface={**dict(trace), **dict(admission), **dict(surfaces)},
            exact_score_decision=_v1_first_text(admission, "exact_score_decision") or "not_applicable",
            parseback_survived=parseback_survived,
            inflate_survived=inflate_survived,
            restore_state_pass=_v1_first_bool(admission, "restore_state_pass", "restore_state_passed"),
            artifact_ref=_v1_first_text(admission, "artifact_ref", "artifact_path"),
            archive_sha256=_v1_first_text(admission, "archive_sha256", "candidate_archive_sha256"),
            payload_sha256=_v1_first_text(admission, "payload_sha256"),
            evaluator_hash=_v1_first_text(admission, "evaluator_hash", "scorer_hash"),
            dependency_hash=_v1_first_text(admission, "dependency_hash"),
            taint_status=_v1_first_text(admission, "taint_status") or "unknown",
            fakequant_survived=fakequant_survived,
            hard_won_count=_v1_first_int(
                transition_values,
                "target_hard_won_count",
                "wrong_to_target_count",
                "receiver_surface_target_hard_won_count",
            ),
            wrong_to_target=_v1_first_int(transition_values, "wrong_to_target_count", "target_hard_won_count"),
            target_to_wrong=_v1_first_int(transition_values, "target_to_wrong_count", "target_hard_lost_count"),
            wrong_to_wrong=_v1_first_int(transition_values, "wrong_to_wrong_count"),
            net_target_support_delta=_v1_first_int(
                transition_values,
                "net_target_support_delta",
                "receiver_surface_net_target_support_delta",
            ),
            uint8_changed_count_region=_v1_first_int(
                transition_values,
                "uint8_changed_pixels",
                "receiver_uint8_changed_pixels_region",
                "uint8_changed_count_region",
            ),
            seg_input_delta_linf_region=_v1_first_float(
                transition_values,
                "segnet_input_delta_linf",
                "seg_input_delta_linf_region",
            ),
            posenet_input_delta_linf_pair=_v1_first_float(
                transition_values,
                "posenet_input_delta_linf_pair",
                "pose_input_delta_linf_pair",
            ),
            segnet_margin_delta=_v1_first_float(
                transition_values,
                "segnet_margin_delta",
                "target_margin_delta",
                "receiver_surface_worst_region_margin_p50_delta",
            ),
            fakequant_segnet_margin_delta=_v1_first_float(
                transition_values,
                "fakequant_segnet_margin_delta",
            ),
            parseback_segnet_margin_delta=_v1_first_float(
                transition_values,
                "parseback_segnet_margin_delta",
            ),
            reference_bytes=reference_bytes,
        )

    @classmethod
    def from_pr110_selector_row(
        cls,
        row: Mapping[str, Any],
        *,
        consumer: str | None = None,
        reference_bytes: int = CONTEST_REFERENCE_BYTES,
    ) -> ActionEffect:
        """Build an ActionEffect from a PR110 selector / lattice-atom row.

        Two real PR110 row shapes are mapped tolerantly:

        * **Selector candidate** (``summary.json`` / ``manifest.json``):
          ``candidate_id`` → ``action_id``; ``selected_pairs`` → ``pair_ids``;
          ``archive.bytes`` → ``new_bytes`` and
          ``archive.delta_bytes_vs_source_archive`` recovers ``old_bytes``;
          member ``name`` → ``payload_sections``.  These rows carry NO measured
          seg/pose distortion (the blockers say so), so distortion endpoints are
          ``None`` (a byte-only / rate observation).
        * **Lattice atom** (``..._lattice_typed_*.json`` ``atoms[]``):
          ``atom_id`` → ``action_id``; ``scope.pair_index`` → ``pair_ids``;
          ``score.seg_dist`` / ``score.pose_dist`` (or ``metadata.source_row``
          ``segnet_dist`` / ``posenet_dist``) → the candidate's measured
          (new-state) distortion.  A lattice atom is a SINGLE-state component
          observation (not a paired before/after), so the measured distortion
          lands on ``new_d_seg`` / ``new_d_pose`` with ``old_*`` left ``None``.

        Tolerant + schema-spec-driven: an unrecognized row still produces a
        valid ActionEffect (distortion ``None``, bytes from whatever is present)
        so the planner never crashes on a new PR110 variant.
        """

        if not isinstance(row, Mapping):
            raise TypeError("pr110 selector row must be a mapping")

        family = _v1_first_text(row, "family") or _v1_first_text(_v1_mapping(row.get("metadata")), "family") or "pr110"
        action_id = _v1_first_text(row, "candidate_id", "atom_id", "action_id") or "pr110_selector_action"
        authority = (
            _v1_first_text(row, "authority", "lane_tag", "axis_tag")
            or _v1_first_text(_v1_mapping(row.get("auth_eval")), "lane_tag", "evidence_grade", "score_axis")
            or ACTION_EFFECT_PLANNING_AUTHORITY
        )
        pair_ids = _v1_pr110_pair_ids(row)

        score_block = _v1_mapping(row.get("score"))
        source_row = _v1_mapping(_v1_mapping(row.get("metadata")).get("source_row"))
        old_d_seg = _v1_first_float(row, "old_d_seg", "d_seg_old", "old_segnet_distortion", "old_avg_segnet_dist")
        new_d_seg = _v1_first_float(row, "new_d_seg", "d_seg_new", "new_segnet_distortion", "new_avg_segnet_dist")
        if new_d_seg is None:
            new_d_seg = _v1_first_float(score_block, "seg_dist", "segnet_dist")
        if new_d_seg is None:
            new_d_seg = _v1_first_float(source_row, "segnet_dist", "seg_dist")
        old_d_pose = _v1_first_float(row, "old_d_pose", "d_pose_old", "old_posenet_distortion", "old_avg_posenet_dist")
        new_d_pose = _v1_first_float(row, "new_d_pose", "d_pose_new", "new_posenet_distortion", "new_avg_posenet_dist")
        if new_d_pose is None:
            new_d_pose = _v1_first_float(score_block, "pose_dist", "posenet_dist")
        if new_d_pose is None:
            new_d_pose = _v1_first_float(source_row, "posenet_dist", "pose_dist")

        old_bytes, new_bytes = _v1_pr110_byte_endpoints(row)
        payload_sections = _v1_pr110_payload_sections(row)

        return cls.build(
            action_id=action_id,
            family=family,
            action_kind=_v1_first_text(row, "action_kind") or (
                "selector_mode" if pair_ids else "selector_program"
            ),
            authority=_v1_normalize_auth_eval_authority(authority),
            normalization_scope=(
                _v1_first_text(row, "normalization_scope")
                or _v1_default_normalization_scope(authority)
            ),
            producer="pr110_frame_exploit_selector",
            consumer=consumer,
            pair_ids=pair_ids,
            region_ids=(),
            payload_sections=payload_sections,
            old_d_seg=old_d_seg,
            new_d_seg=new_d_seg,
            old_d_pose=old_d_pose,
            new_d_pose=new_d_pose,
            old_bytes=old_bytes,
            new_bytes=new_bytes,
            receiver_surface=row,
            exact_score_decision=_v1_first_text(row, "exact_score_decision") or "not_applicable",
            parseback_survived=_v1_first_bool(row, "parseback_survived", "official_inflate_control_passed"),
            inflate_survived=_v1_first_bool(row, "inflate_survived", "official_inflate_control_passed"),
            restore_state_pass=_v1_first_bool(row, "restore_state_pass", "restore_state_passed"),
            artifact_ref=_v1_first_text(row, "artifact_ref", "artifact_path", "summary_path"),
            archive_sha256=_v1_first_text(row, "archive_sha256", "candidate_archive_sha256"),
            payload_sha256=_v1_first_text(row, "payload_sha256"),
            evaluator_hash=_v1_first_text(row, "evaluator_hash"),
            dependency_hash=_v1_first_text(row, "dependency_hash"),
            taint_status=_v1_first_text(row, "taint_status") or "unknown",
            reference_bytes=reference_bytes,
        )

    @classmethod
    def from_frontier_rate_materializer(
        cls,
        materializer_row: Mapping[str, Any],
        *,
        auth_eval: Mapping[str, Any] | None = None,
        source_auth_eval: Mapping[str, Any] | None = None,
        consumer: str | None = "action_effect_commutator_ledger",
        reference_bytes: int = CONTEST_REFERENCE_BYTES,
    ) -> ActionEffect:
        """Build an ActionEffect from the final-rate materializer stack.

        This is the live frontier-rate path, not the historical PR110 selector
        path.  It consumes rows such as ``fp11_source_brotli_recode_manifest``
        and produces a byte-priced effect from source archive bytes to candidate
        archive bytes.  Exact CPU/CUDA auth-eval rows may tag the measurement
        authority and, when both before/after auth evals are supplied, provide
        nonrate endpoints.  A candidate-only exact eval is intentionally NOT
        treated as a before/after distortion delta.

        The source artifacts often carry their own promotion/score-claim
        booleans.  Those are custody metadata for their native surface and must
        not be copied into ``tac.action_effect.v1``; this constructor only
        extracts action, bytes, survival, payload scope, and optional
        before/after distortion endpoints.
        """

        if not isinstance(materializer_row, Mapping):
            raise TypeError("frontier-rate materializer row must be a mapping")
        row = materializer_row
        auth = _v1_mapping(auth_eval)
        source_auth = _v1_mapping(source_auth_eval)
        source_archive = _v1_mapping(row.get("source_archive"))
        candidate_archive = _v1_mapping(row.get("candidate_archive"))
        serialized_delta = _v1_mapping(row.get("serialized_archive_delta"))

        target_kind = (
            _v1_first_text(row, "target_kind", "operation_family", "materializer_id")
            or "frontier_rate_attack"
        )
        candidate_sha = (
            _v1_first_text(candidate_archive, "sha256")
            or _v1_first_text(row, "candidate_archive_sha256", "archive_sha256")
        )
        action_id = (
            _v1_first_text(row, "action_id", "candidate_id", "observation_id")
            or f"{target_kind}:{candidate_sha[:12] if candidate_sha else 'candidate'}"
        )

        old_bytes = (
            _v1_first_int(source_archive, "bytes")
            or _v1_first_int(row, "source_archive_bytes")
            or _v1_first_int(serialized_delta, "source_archive_bytes")
        )
        new_bytes = (
            _v1_first_int(candidate_archive, "bytes")
            or _v1_first_int(row, "candidate_archive_bytes", "archive_size_bytes")
            or _v1_first_int(serialized_delta, "candidate_archive_bytes")
            or _v1_first_int(auth, "archive_size_bytes")
        )

        old_d_seg = new_d_seg = old_d_pose = new_d_pose = None
        if auth and source_auth:
            old_d_seg = _v1_first_float(source_auth, "avg_segnet_dist", "d_seg", "segnet_dist")
            new_d_seg = _v1_first_float(auth, "avg_segnet_dist", "d_seg", "segnet_dist")
            old_d_pose = _v1_first_float(source_auth, "avg_posenet_dist", "d_pose", "posenet_dist")
            new_d_pose = _v1_first_float(auth, "avg_posenet_dist", "d_pose", "posenet_dist")

        parseback_survived = _v1_first_bool(
            row,
            "parseback_survived",
            "receiver_contract_satisfied",
            "receiver_proof_ready",
        )
        inflate_survived = _v1_first_bool(row, "inflate_survived", "runtime_consumption_proof_passed")
        if inflate_survived is None and auth:
            inflate_survived = True

        payload_sections = _v1_frontier_rate_payload_sections(row, target_kind)
        producer = (
            _v1_first_text(row, "materializer_id", "operation_family", "target_kind")
            or "frontier_final_rate_attack"
        )

        return cls.build(
            action_id=action_id,
            family="frontier_rate_attack",
            action_kind=_v1_first_text(row, "action_kind") or "frontier_rate_materializer",
            authority=_v1_frontier_rate_authority(auth, row),
            normalization_scope=(
                _v1_first_text(row, "normalization_scope")
                or _v1_default_normalization_scope(_v1_frontier_rate_authority(auth, row))
            ),
            producer=producer,
            consumer=consumer,
            pair_ids=(),
            region_ids=(),
            payload_sections=payload_sections,
            old_d_seg=old_d_seg,
            new_d_seg=new_d_seg,
            old_d_pose=old_d_pose,
            new_d_pose=new_d_pose,
            old_bytes=old_bytes,
            new_bytes=new_bytes,
            receiver_surface=row,
            exact_score_decision=_v1_first_text(row, "exact_score_decision") or "not_applicable",
            parseback_survived=parseback_survived,
            inflate_survived=inflate_survived,
            restore_state_pass=_v1_first_bool(row, "restore_state_pass", "restore_state_passed"),
            artifact_ref=_v1_first_text(row, "artifact_ref", "manifest_path"),
            archive_sha256=_v1_first_text(candidate_archive, "sha256") or _v1_first_text(row, "candidate_archive_sha256"),
            payload_sha256=_v1_first_text(row, "payload_sha256"),
            evaluator_hash=_v1_first_text(auth, "evaluator_hash", "runtime_content_sha256"),
            dependency_hash=_v1_first_text(auth, "dependency_hash", "runtime_tree_sha256"),
            taint_status=_v1_first_text(row, "taint_status") or "clean",
            fakequant_survived=None,
            reference_bytes=reference_bytes,
        )

    # ── serialization ─────────────────────────────────────────────────────

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON-serializable row (tuples → lists)."""

        payload = asdict(self)
        payload["pair_ids"] = list(self.pair_ids)
        payload["class_ids"] = list(self.class_ids)
        payload["region_ids"] = list(self.region_ids)
        payload["payload_sections"] = list(self.payload_sections)
        payload["trained_groups"] = list(self.trained_groups)
        payload["blockers"] = list(self.blockers)
        payload["receiver_surface"] = self.receiver_surface.as_dict()
        payload["old_archive_bytes"] = self.old_bytes
        payload["new_archive_bytes"] = self.new_bytes
        payload["restore_state_passed"] = self.restore_state_pass
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ActionEffect:
        """Rebuild an ActionEffect from its :meth:`as_dict` form."""

        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")
        _v1_forbid_score_claim_fields(payload)
        _v1_validate_survival_action_ids(payload)
        schema = str(payload.get("schema") or "")
        if schema != ACTION_EFFECT_V1_SCHEMA:
            raise ValueError(f"schema must be {ACTION_EFFECT_V1_SCHEMA!r}; got {schema!r}")
        return cls.build(
            action_id=str(payload["action_id"]),
            family=str(payload.get("family") or "shared"),
            action_kind=str(payload.get("action_kind") or "unknown_action"),
            inverse_source=None if payload.get("inverse_source") is None else str(payload["inverse_source"]),
            frame_index=_v1_frame_index_or_none(payload.get("frame_index")),
            frame_incidence=None if payload.get("frame_incidence") is None else str(payload["frame_incidence"]),
            candidate_status=None if payload.get("candidate_status") is None else str(payload["candidate_status"]),
            authority=str(payload.get("authority") or ""),
            normalization_scope=(
                None
                if not payload.get("normalization_scope")
                else str(payload.get("normalization_scope"))
            ),
            producer=str(payload.get("producer") or "unknown"),
            consumer=None if payload.get("consumer") is None else str(payload["consumer"]),
            pair_ids=_v1_int_tuple(payload.get("pair_ids") or ()),
            class_ids=_v1_int_tuple(payload.get("class_ids") or ()),
            region_ids=_v1_str_tuple(payload.get("region_ids") or ()),
            payload_sections=_v1_str_tuple(payload.get("payload_sections") or ()),
            trained_groups=_v1_str_tuple(payload.get("trained_groups") or ()),
            old_d_seg=_v1_first_float(payload, "old_d_seg"),
            new_d_seg=_v1_first_float(payload, "new_d_seg"),
            old_d_pose=_v1_first_float(payload, "old_d_pose"),
            new_d_pose=_v1_first_float(payload, "new_d_pose"),
            old_bytes=_v1_first_int(payload, "old_bytes", "old_archive_bytes"),
            new_bytes=_v1_first_int(payload, "new_bytes", "new_archive_bytes"),
            receiver_surface=_v1_mapping(payload.get("receiver_surface")),
            exact_score_decision=str(payload.get("exact_score_decision") or "not_applicable"),
            raw_cap_decision=None if payload.get("raw_cap_decision") is None else str(payload["raw_cap_decision"]),
            catastrophic_guard_decision=(
                None
                if payload.get("catastrophic_guard_decision") is None
                else str(payload["catastrophic_guard_decision"])
            ),
            would_accept_exact_score_if_raw_cap_disabled=_v1_bool_or_none(
                payload.get("would_accept_exact_score_if_raw_cap_disabled")
            ),
            would_accept_without_catastrophic_guard=_v1_bool_or_none(
                payload.get("would_accept_without_catastrophic_guard")
            ),
            rejected_by_raw_cap=_v1_bool_or_none(payload.get("rejected_by_raw_cap")),
            rejected_by_exact_score=_v1_bool_or_none(payload.get("rejected_by_exact_score")),
            rejected_by_catastrophic_guard=_v1_bool_or_none(payload.get("rejected_by_catastrophic_guard")),
            parseback_survived=_v1_bool_or_none(payload.get("parseback_survived")),
            inflate_survived=_v1_bool_or_none(payload.get("inflate_survived")),
            restore_state_pass=_v1_first_bool(payload, "restore_state_pass", "restore_state_passed"),
            artifact_ref=None if payload.get("artifact_ref") is None else str(payload["artifact_ref"]),
            archive_sha256=None if payload.get("archive_sha256") is None else str(payload["archive_sha256"]),
            payload_sha256=None if payload.get("payload_sha256") is None else str(payload["payload_sha256"]),
            base_state_sha256=(
                None if payload.get("base_state_sha256") is None else str(payload["base_state_sha256"])
            ),
            evaluator_hash=None if payload.get("evaluator_hash") is None else str(payload["evaluator_hash"]),
            dependency_hash=None if payload.get("dependency_hash") is None else str(payload["dependency_hash"]),
            taint_status=str(payload.get("taint_status") or "unknown"),
            fakequant_survived=_v1_bool_or_none(payload.get("fakequant_survived")),
            hard_won_count=_v1_int_or_none(payload.get("hard_won_count")),
            wrong_to_target=_v1_int_or_none(payload.get("wrong_to_target")),
            target_to_wrong=_v1_int_or_none(payload.get("target_to_wrong")),
            wrong_to_wrong=_v1_int_or_none(payload.get("wrong_to_wrong")),
            net_target_support_delta=_v1_int_or_none(payload.get("net_target_support_delta")),
            uint8_changed_count_region=_v1_int_or_none(payload.get("uint8_changed_count_region")),
            seg_input_delta_linf_region=_v1_float_or_none(payload.get("seg_input_delta_linf_region")),
            posenet_input_delta_linf_pair=_v1_float_or_none(payload.get("posenet_input_delta_linf_pair")),
            support_source=(
                None if payload.get("support_source") is None else str(payload["support_source"])
            ),
            support_cardinality=_v1_int_or_none(payload.get("support_cardinality")),
            support_sha256=None if payload.get("support_sha256") is None else str(payload["support_sha256"]),
            support_encoding=(
                None if payload.get("support_encoding") is None else str(payload["support_encoding"])
            ),
            support_encoded_bytes=_v1_int_or_none(payload.get("support_encoded_bytes")),
            support_research_only=_v1_bool_or_none(payload.get("support_research_only")),
            arm=None if payload.get("arm") is None else str(payload["arm"]),
            old_region_debt=_v1_float_or_none(payload.get("old_region_debt")),
            new_region_debt=_v1_float_or_none(payload.get("new_region_debt")),
            argmax_changed_count_region=_v1_int_or_none(payload.get("argmax_changed_count_region")),
            pose_output_l2_delta=_v1_float_or_none(payload.get("pose_output_l2_delta")),
            seg_score_delta=_v1_float_or_none(payload.get("seg_score_delta")),
            pose_score_delta=_v1_float_or_none(payload.get("pose_score_delta")),
            segnet_margin_delta=_v1_float_or_none(payload.get("segnet_margin_delta")),
            fakequant_segnet_margin_delta=_v1_float_or_none(
                payload.get("fakequant_segnet_margin_delta")
            ),
            parseback_segnet_margin_delta=_v1_float_or_none(
                payload.get("parseback_segnet_margin_delta")
            ),
            rejection_source=(
                None if payload.get("rejection_source") is None else str(payload["rejection_source"])
            ),
            blockers=_v1_str_tuple(payload.get("blockers") or ()),
            interaction_or_commutator=_v1_float_or_none(payload.get("interaction_or_commutator")),
        )


# ── JSONL ledger (fcntl-locked append, repo convention) ──────────────────


def append_action_effect(
    effect: ActionEffect,
    ledger_path: str | os.PathLike[str],
    *,
    lock_timeout_seconds: int = _ACTION_EFFECT_LEDGER_LOCK_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Append one ActionEffect row to a JSONL ledger under an fcntl lock.

    Mirrors the canonical fcntl-locked append convention (see
    ``tools/subagent_checkpoint.py``): a sibling ``.lock`` file is locked
    ``LOCK_EX`` for the open+write+fsync so concurrent appenders from sister
    agents serialize without lost rows (Catalog #131 bare-write discipline).
    A ``written_at_utc`` stamp is added on write.
    """

    if not isinstance(effect, ActionEffect):
        raise TypeError("effect must be an ActionEffect")
    path = Path(ledger_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    record = effect.as_dict()
    record["written_at_utc"] = _v1_now_iso()

    fh = _v1_acquire_ledger_lock(lock_path, lock_timeout_seconds)
    try:
        with open(path, "a", encoding="utf-8") as out:
            out.write(json.dumps(record, sort_keys=True) + "\n")
            out.flush()
            os.fsync(out.fileno())
    finally:
        _v1_release_ledger_lock(fh)
    return record


def read_action_effects(
    ledger_path: str | os.PathLike[str],
    *,
    action_id: str | None = None,
) -> list[ActionEffect]:
    """Read ActionEffect rows from a JSONL ledger (append order).

    Malformed lines are skipped.  ``action_id`` optionally filters to one
    action.
    """

    path = Path(ledger_path)
    if not path.exists():
        return []
    out: list[ActionEffect] = []
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            if action_id is not None and payload.get("action_id") != action_id:
                continue
            try:
                out.append(ActionEffect.from_dict(payload))
            except (KeyError, ValueError, TypeError):
                continue
    return out


def validate_action_effect_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one serialized ActionEffect row and return typed blockers."""

    blockers: list[str] = []
    if not isinstance(payload, Mapping):
        return {
            "schema": "tac.action_effect_validation.v1",
            "passed": False,
            "blockers": ["action_effect_malformed"],
        }
    if str(payload.get("schema") or "") != ACTION_EFFECT_V1_SCHEMA:
        blockers.append("action_effect_schema_invalid")
    if not str(payload.get("authority") or "").strip():
        blockers.append("action_effect_untyped_authority")
    if not str(payload.get("normalization_scope") or "").strip():
        blockers.append("normalization_scope_mismatch")
    for path in _v1_score_claim_field_paths(payload):
        blockers.append(f"action_effect_forbidden_score_authority:{path}")
    try:
        _v1_validate_survival_action_ids(payload)
    except ValueError:
        blockers.append("action_id_survival_mismatch")
    try:
        effect = ActionEffect.from_dict(payload)
    except ValueError as exc:
        blockers.extend(_v1_blockers_from_error(exc))
        effect = None
    except (KeyError, TypeError):
        blockers.append("action_effect_malformed")
        effect = None
    if effect is not None:
        if _v1_authority_is_batch_local(effect.authority) and effect.promotion_eligible is not False:
            blockers.append("local_row_used_for_promotion")
        if effect.normalization_scope == NormalizationScope.BATCH_LOCAL.value and _v1_authority_is_promotional(effect.authority):
            blockers.append("normalization_scope_mismatch")
        if effect.delta_bytes not in (None, 0) and effect.value_per_byte is None:
            blockers.append("value_per_byte_missing")
    blockers = _dedupe(blockers)
    return {
        "schema": "tac.action_effect_validation.v1",
        "passed": not blockers,
        "blockers": blockers,
        "action_id": payload.get("action_id"),
    }


# ── thin-currency internal helpers (prefixed _v1_ to avoid name collisions) ─


def _v1_validate_distortion(name: str, value: float | None) -> None:
    if value is None:
        return
    fval = float(value)
    if not math.isfinite(fval):
        raise ValueError(f"{name} must be finite; got {value!r}")
    if fval < 0.0:
        raise ValueError(f"{name} must be non-negative; got {value!r}")


def _v1_validate_bytes(name: str, value: int | None) -> None:
    if value is None:
        return
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an int; got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative; got {value!r}")


def _v1_validate_optional_nonnegative_int(name: str, value: int | None) -> None:
    if value is None:
        return
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an int; got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative; got {value!r}")


def _v1_validate_optional_int(name: str, value: int | None) -> None:
    if value is None:
        return
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an int; got {value!r}")


def _v1_validate_optional_finite_float(name: str, value: float | None) -> None:
    if value is None:
        return
    fval = float(value)
    if not math.isfinite(fval):
        raise ValueError(f"{name} must be finite; got {value!r}")


def _v1_now_iso() -> str:
    return _dt.datetime.now(tz=_dt.UTC).isoformat()


def _v1_acquire_ledger_lock(lock_path: Path, timeout_seconds: int):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.touch(exist_ok=True)
    fh = open(lock_path, "r+")  # noqa: SIM115 - caller owns lock lifetime
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fh
        except BlockingIOError:
            if time.monotonic() >= deadline:
                fh.close()
                raise TimeoutError(f"could not acquire {lock_path} within {timeout_seconds}s") from None
            time.sleep(0.05)


def _v1_release_ledger_lock(fh) -> None:
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    finally:
        fh.close()


def _v1_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _v1_hinerv_birth_support_identity(
    receipt: Mapping[str, Any],
    *,
    action_section_telemetry: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Extract the archive-executable target-region action support identity.

    HiNeRV birth receipts can carry support metadata at several receiver
    surfaces: root action-section telemetry, runner export selection,
    parse-back survival, and the wall-normal sidecar/direct-teacher receipt.
    ActionEffect rows are the DAG currency, so they must expose the same
    action/support identity without requiring downstream tools to re-walk the
    full smoke payload.
    """

    candidates: list[Mapping[str, Any]] = []

    def _append(value: Any) -> None:
        if isinstance(value, Mapping):
            candidates.append(value)

    def _append_wall_normal(value: Any) -> None:
        wall = _v1_mapping(value)
        if not wall:
            return
        _append(wall)
        _append(wall.get("sidecar_fallback"))
        direct = _v1_mapping(wall.get("direct_teacher"))
        _append(direct)
        _append(direct.get("action_effect"))

    _append(action_section_telemetry)
    _append(receipt.get("target_region_action_export_selection"))
    _append(receipt.get("target_region_action_parseback_survival"))
    _append_wall_normal(receipt.get("target_region_wall_normal_lift"))
    telemetry = _v1_mapping(receipt.get("candidate_frontier_telemetry"))
    _append_wall_normal(telemetry.get("target_region_wall_normal_lift"))
    masked = _v1_mapping(telemetry.get("masked_residual_oracle"))
    for key in ("best_wall_normal_candidate", "best_candidate", "selected_candidate"):
        candidate = _v1_mapping(masked.get(key))
        _append(candidate)
        _append(candidate.get("target_region_action_section_telemetry"))
        _append(candidate.get("direct_seg_wall_oracle"))

    support_sha256 = None
    support_cardinality = None
    support_encoding = None
    support_encoded_bytes = None
    support_source = None
    for row in candidates:
        if support_sha256 is None:
            support_sha256 = _v1_first_text(
                row,
                "support_sha256",
                "target_region_action_support_sha256",
                "archive_executable_support_sha256",
                "expected_support_sha256",
            )
        if support_cardinality is None:
            support_cardinality = _v1_first_int(
                row,
                "support_cardinality",
                "target_region_action_support_cardinality",
                "archive_executable_support_cardinality",
                "target_region_action_pixel_count",
                "total_action_pixels",
            )
        if support_encoding is None:
            support_encoding = _v1_first_text(
                row,
                "support_encoding",
                "target_region_action_support_encoding",
                "archive_executable_support_encoding",
            )
        if support_encoded_bytes is None:
            support_encoded_bytes = _v1_first_int(
                row,
                "support_encoded_bytes",
                "target_region_action_support_encoded_bytes",
                "archive_executable_support_encoded_bytes",
            )
        if support_source is None:
            support_source = _v1_first_text(row, "support_source")
    return {
        key: value
        for key, value in {
            "support_source": support_source,
            "support_cardinality": support_cardinality,
            "support_sha256": support_sha256,
            "support_encoding": support_encoding,
            "support_encoded_bytes": support_encoded_bytes,
        }.items()
        if value is not None
    }


def _v1_hinerv_birth_authority(
    receipt: Mapping[str, Any],
    surface: str | None,
) -> str:
    text = (
        str(surface).strip()
        if surface is not None and str(surface).strip()
        else _v1_first_text(receipt, "authority", "axis_tag")
    )
    if not text:
        return ACTION_EFFECT_PLANNING_AUTHORITY
    lowered = text.strip().lower()
    if lowered == "live_mlx":
        return ScoreAuthority.BATCH_LOCAL_LIVE_MLX.value
    if lowered in {"inflated_torch_cpu", "inflate_cpu", "torch_cpu"}:
        return ScoreAuthority.INFLATE_TORCH_CPU.value
    if lowered in {"inflated_torch_cuda", "inflate_cuda", "torch_cuda"}:
        return ScoreAuthority.INFLATE_TORCH_CUDA.value
    return text.strip()


def _v1_normalize_auth_eval_authority(authority: str | None) -> str:
    text = str(authority or "").strip()
    if not text:
        return ACTION_EFFECT_PLANNING_AUTHORITY
    lowered = text.lower()
    if lowered in {"contest_cpu", "[contest-cpu]", "contest-cpu"} or text == "contest-CPU":
        return ScoreAuthority.CONTEST_CPU.value
    if lowered in {"contest_cuda", "[contest-cuda]", "contest-cuda"} or text == "contest-CUDA":
        return ScoreAuthority.CONTEST_CUDA.value
    return text


def _v1_default_normalization_scope(authority: str | None) -> str:
    text = str(authority or "").strip()
    if _v1_authority_is_promotional(text):
        return NormalizationScope.FULL_VIDEO_EXACT.value
    if text == ScoreAuthority.RECEIVER_CLOSED_FRONTIER_RATE_ATTACK.value:
        return NormalizationScope.FULL_VIDEO_EQUIV_ESTIMATE.value
    return NormalizationScope.BATCH_LOCAL.value


def _v1_base_state_sha256(row: Mapping[str, Any]) -> str | None:
    direct = _v1_first_text(row, "base_state_sha256", "initial_state_sha256")
    if direct:
        return direct
    groups = row.get("parameter_group_sha256_before")
    if isinstance(groups, Mapping):
        normalized = {
            str(key): str(value)
            for key, value in groups.items()
            if str(key).strip() and str(value).strip()
        }
        if normalized:
            payload = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
            return hashlib.sha256(payload).hexdigest()
    return None


def _v1_authority_is_batch_local(authority: str | None) -> bool:
    text = str(authority or "").strip().lower()
    return text in {
        ACTION_EFFECT_PLANNING_AUTHORITY,
        ScoreAuthority.LIVE_MLX.value,
        ScoreAuthority.BATCH_LOCAL_LIVE_MLX.value,
        ScoreAuthority.ROUND_STE_MLX.value,
        ScoreAuthority.FAKEQUANT_MLX.value,
        ScoreAuthority.PARSEBACK_MLX.value,
    }


def _v1_authority_is_promotional(authority: str | None) -> bool:
    text = str(authority or "").strip().lower()
    return text in {
        ScoreAuthority.CONTEST_CPU.value,
        ScoreAuthority.CONTEST_CUDA.value,
        ScoreAuthority.INFLATE_TORCH_CPU.value,
        ScoreAuthority.INFLATE_TORCH_CUDA.value,
        "[contest-cpu]",
        "[contest-cuda]",
    } or text.startswith("[contest-")


def _v1_normalize_decision(value: str | None) -> str:
    text = str(value or "not_applicable").strip().lower()
    if text in {"accept", "accepted", "admit", "admitted", "pass", "passed"}:
        return "accept"
    if text in {"reject", "rejected", "refuse", "refused", "fail", "failed"}:
        return "reject"
    if text in {"not_applicable", "not-applicable", "none", "unknown", ""}:
        return "not_applicable"
    return text


def _v1_pose_score_delta(old_d_pose: float | None, new_d_pose: float | None) -> float | None:
    if old_d_pose is None or new_d_pose is None:
        return None
    _v1_validate_distortion("old_d_pose", old_d_pose)
    _v1_validate_distortion("new_d_pose", new_d_pose)
    return math.sqrt(10.0 * float(new_d_pose)) - math.sqrt(10.0 * float(old_d_pose))


def _v1_forbid_score_claim_fields(payload: Mapping[str, Any]) -> None:
    paths = list(_v1_score_claim_field_paths(payload))
    if paths:
        joined = ", ".join(paths[:8])
        if len(paths) > 8:
            joined += f", ... (+{len(paths) - 8} more)"
        raise ValueError(f"ActionEffect v1 rows must not carry score-claim fields: {joined}")


def _v1_score_claim_field_paths(value: Any, *, prefix: str = "$") -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, raw_value in value.items():
            key = str(raw_key)
            child = f"{prefix}.{key}"
            if key in _ACTION_EFFECT_V1_FORBIDDEN_SCORE_CLAIM_KEYS:
                paths.append(child)
            paths.extend(_v1_score_claim_field_paths(raw_value, prefix=child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            paths.extend(_v1_score_claim_field_paths(item, prefix=f"{prefix}[{index}]"))
    return paths


def _v1_validate_survival_action_ids(payload: Mapping[str, Any]) -> None:
    action_id = str(payload.get("action_id") or "").strip()
    if not action_id:
        return
    for key in (
        "fakequant_action_id",
        "fakequant_survival_action_id",
        "parseback_action_id",
        "parseback_survival_action_id",
        "inflate_action_id",
        "inflate_survival_action_id",
    ):
        value = payload.get(key)
        if value is not None and str(value).strip() != action_id:
            raise ValueError(f"action_id_survival_mismatch:{key}")
    for key in ("fakequant_survival", "parseback_survival", "inflate_survival"):
        nested = _v1_mapping(payload.get(key))
        value = nested.get("action_id")
        if value is not None and str(value).strip() != action_id:
            raise ValueError(f"action_id_survival_mismatch:{key}.action_id")


def _v1_blockers_from_error(exc: ValueError) -> list[str]:
    text = str(exc)
    blockers: list[str] = []
    if "authority" in text and "non-empty" in text:
        blockers.append("action_effect_untyped_authority")
    if "normalization_scope" in text:
        blockers.append("normalization_scope_mismatch")
    if "promotion_eligible" in text or "local_row_used_for_promotion" in text:
        blockers.append("local_row_used_for_promotion")
    if "value_per_byte_missing" in text:
        blockers.append("value_per_byte_missing")
    if "score-claim" in text or "official_score" in text:
        blockers.append("action_effect_forbidden_score_authority")
    if "action_id_survival_mismatch" in text:
        blockers.append("action_id_survival_mismatch")
    if not blockers:
        blockers.append(f"action_effect_malformed:{type(exc).__name__}")
    return blockers


def _v1_first_text(payload: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _v1_first_float(payload: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        out = _v1_float_or_none(payload.get(key))
        if out is not None:
            return out
    return None


def _v1_first_int(payload: Mapping[str, Any], *keys: str) -> int | None:
    for key in keys:
        out = _v1_int_or_none(payload.get(key))
        if out is not None:
            return out
    return None


def _v1_first_bool(payload: Mapping[str, Any], *keys: str) -> bool | None:
    for key in keys:
        out = _v1_bool_or_none(payload.get(key))
        if out is not None:
            return out
    return None


def _v1_float_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _v1_int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _v1_frame_index_or_none(value: Any) -> int | str | None:
    if value is None or isinstance(value, bool):
        return None
    if value == "both":
        return "both"
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return str(value)
    return parsed


def _v1_bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _v1_int_tuple(value: Any) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    out: list[int] = []
    for item in value:
        parsed = _v1_int_or_none(item)
        if parsed is not None:
            out.append(parsed)
    return tuple(out)


def _v1_str_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(str(item) for item in value if str(item).strip())


def _v1_birth_region_id(worst: Mapping[str, Any]) -> str | None:
    batch = _v1_first_int(worst, "batch_index")
    cls = _v1_first_int(worst, "class_index")
    label = _v1_first_int(worst, "region_label")
    if batch is None and cls is None and label is None:
        return None
    return f"b{batch}/c{cls}/r{label}"


def _v1_birth_pair_ids(receipt: Mapping[str, Any], worst: Mapping[str, Any]) -> tuple[int, ...]:
    pair_index = _v1_first_int(receipt, "pair_index")
    if pair_index is None:
        pair_index = _v1_first_int(worst, "batch_index")
    return (pair_index,) if pair_index is not None else ()


def _v1_birth_survival_flags(
    receipt: Mapping[str, Any],
    surface: str | None,
) -> tuple[bool | None, bool | None]:
    """Infer parse-back / inflate survival from an explicit flag or the surface.

    A birth receipt does not always carry boolean survival flags; when the
    receipt was traced THROUGH a parse-back or inflate surface and reported no
    blockers, that surface is the survival witness.  We never invent a True;
    explicit ``parseback_survived`` / ``inflate_survived`` keys win.
    """

    parseback = _v1_bool_or_none(receipt.get("parseback_survived"))
    inflate = _v1_bool_or_none(receipt.get("inflate_survived"))
    if parseback is not None or inflate is not None:
        return parseback, inflate
    blockers = receipt.get("blockers")
    no_blockers = isinstance(blockers, Sequence) and not isinstance(blockers, (str, bytes)) and len(blockers) == 0
    if surface == "parseback_mlx" and no_blockers:
        return True, inflate
    if surface == "inflated_torch_cpu" and no_blockers:
        return parseback, True
    return parseback, inflate


def _v1_recover_byte_endpoints(admission: Mapping[str, Any]) -> tuple[int | None, int | None]:
    old = _v1_first_int(admission, "old_bytes", "old_archive_bytes", "archive_bytes_old", "before_archive_bytes")
    new = _v1_first_int(admission, "new_bytes", "new_archive_bytes", "archive_bytes_new", "after_archive_bytes")
    if old is not None and new is not None:
        return old, new
    delta = _v1_first_int(admission, "delta_archive_bytes", "delta_bytes")
    if new is not None and old is None and delta is not None:
        return new - delta, new
    if old is not None and new is None and delta is not None:
        return old, old + delta
    return old, new


def _v1_pr110_pair_ids(row: Mapping[str, Any]) -> tuple[int, ...]:
    selected = _v1_int_tuple(row.get("selected_pairs"))
    if selected:
        return selected
    scope = _v1_mapping(row.get("scope"))
    pair_index = _v1_first_int(scope, "pair_index")
    if pair_index is None:
        pair_index = _v1_first_int(row, "pair_index", "pair")
    if pair_index is None:
        source_row = _v1_mapping(_v1_mapping(row.get("metadata")).get("source_row"))
        pair_index = _v1_first_int(source_row, "pair")
    return (pair_index,) if pair_index is not None else ()


def _v1_pr110_byte_endpoints(row: Mapping[str, Any]) -> tuple[int | None, int | None]:
    old = _v1_first_int(row, "old_bytes", "old_archive_bytes", "archive_bytes_old")
    new = _v1_first_int(row, "new_bytes", "new_archive_bytes", "archive_bytes_new")
    if old is not None or new is not None:
        return old, new
    archive = _v1_mapping(row.get("archive"))
    new_bytes = _v1_first_int(archive, "bytes")
    delta = _v1_first_int(archive, "delta_bytes_vs_source_archive")
    if new_bytes is not None and delta is not None:
        return new_bytes - delta, new_bytes
    # Lattice atom: budget.archive_delta_bytes is a pure delta with no absolute
    # anchor, so we cannot imply an absolute archive size; surface old=new=None
    # (the delta is recoverable from the raw row by a caller that wants it).
    if new_bytes is not None:
        return None, new_bytes
    return None, None


def _v1_pr110_payload_sections(row: Mapping[str, Any]) -> tuple[str, ...]:
    archive = _v1_mapping(row.get("archive"))
    members = archive.get("members")
    names: list[str] = []
    if isinstance(members, Sequence) and not isinstance(members, (str, bytes)):
        for member in members:
            if isinstance(member, Mapping):
                name = _v1_first_text(member, "name")
                if name:
                    names.append(name)
    if names:
        return tuple(names)
    metadata = _v1_mapping(row.get("metadata"))
    mode_id = _v1_first_text(metadata, "mode_id") or _v1_first_text(row, "mode_id")
    return (mode_id,) if mode_id else ()


def _v1_frontier_rate_payload_sections(row: Mapping[str, Any], target_kind: str) -> tuple[str, ...]:
    selected = _v1_str_tuple(row.get("selected_member_names"))
    if selected:
        return selected
    selected_member = _v1_first_text(row, "selected_member_name", "member_name")
    if selected_member:
        return (selected_member,)
    return (target_kind,) if target_kind else ()


def _v1_frontier_rate_authority(
    auth_eval: Mapping[str, Any],
    row: Mapping[str, Any],
) -> str:
    lane_tag = _v1_first_text(auth_eval, "lane_tag")
    if lane_tag:
        return f"{lane_tag} frontier_final_rate_attack"
    evidence_grade = _v1_first_text(auth_eval, "evidence_grade")
    if evidence_grade:
        grade = evidence_grade if evidence_grade.startswith("[") else f"[{evidence_grade}]"
        return f"{grade} frontier_final_rate_attack"
    score_axis = _v1_first_text(auth_eval, "score_axis")
    if score_axis == "contest_cpu":
        return "[contest-CPU] frontier_final_rate_attack"
    if score_axis == "contest_cuda":
        return "[contest-CUDA] frontier_final_rate_attack"
    if _v1_first_bool(row, "receiver_contract_satisfied", "receiver_proof_ready") is True:
        return "receiver_closed_frontier_rate_attack"
    return ACTION_EFFECT_PLANNING_AUTHORITY


__all__ = [
    "ACTION_COMMUTATOR_ROW_SCHEMA",
    "ACTION_EFFECT_LEDGER_SCHEMA",
    "ACTION_EFFECT_PLANNING_AUTHORITY",
    "ACTION_EFFECT_SCHEMA",
    "ACTION_EFFECT_V1_SCHEMA",
    "ActionEffect",
    "DeltaScores",
    "EvaluatorActionEffect",
    "NormalizationScope",
    "ReceiverSurfaceDelta",
    "ScoreAuthority",
    "append_action_effect",
    "build_action_commutator_row",
    "build_action_effect",
    "build_action_effect_ledger",
    "compute_delta_scores",
    "exact_delta_score",
    "read_action_effects",
    "validate_action_effect_payload",
    "value_per_byte",
]
