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
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
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
    ``value_per_byte`` is ``-delta_total / delta_bytes`` (positive ⇒ the action
    saved score per byte spent); ``None`` when bytes do not change.
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
      ``-delta_score_total / delta_bytes`` so a positive value means the action
      reduced score (good) per byte of archive change.

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

    value_per_byte: float | None = None
    if delta_score_total is not None and delta_bytes is not None and delta_bytes != 0:
        value_per_byte = -delta_score_total / float(delta_bytes)

    return DeltaScores(
        delta_d_seg=delta_d_seg,
        delta_d_pose=delta_d_pose,
        delta_bytes=delta_bytes,
        delta_score_nonrate=delta_score_nonrate,
        delta_score_total=delta_score_total,
        value_per_byte=value_per_byte,
    )


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
    authority: str
    producer: str
    consumer: str | None
    pair_ids: tuple[int, ...]
    region_ids: tuple[str, ...]
    payload_sections: tuple[str, ...]
    old_d_seg: float | None
    new_d_seg: float | None
    old_d_pose: float | None
    new_d_pose: float | None
    old_bytes: int | None
    new_bytes: int | None
    delta_score_nonrate: float | None
    delta_score_total: float | None
    value_per_byte: float | None
    parseback_survived: bool | None
    inflate_survived: bool | None
    promotion_eligible: bool = False

    def __post_init__(self) -> None:
        if self.schema != ACTION_EFFECT_V1_SCHEMA:
            raise ValueError(f"schema must be {ACTION_EFFECT_V1_SCHEMA!r}; got {self.schema!r}")
        if not isinstance(self.action_id, str) or not self.action_id.strip():
            raise ValueError("action_id must be a non-empty string")
        if not isinstance(self.authority, str) or not self.authority.strip():
            raise ValueError("authority is REQUIRED and must be a non-empty string")
        if self.promotion_eligible is not False:
            raise ValueError(
                "ActionEffect is an analysis-layer ledger row; promotion_eligible "
                "must always be False in v1 (only the exact-eval custody surface promotes)"
            )
        for name, value in (
            ("old_d_seg", self.old_d_seg),
            ("new_d_seg", self.new_d_seg),
            ("old_d_pose", self.old_d_pose),
            ("new_d_pose", self.new_d_pose),
        ):
            _v1_validate_distortion(name, value)
        for name, value in (("old_bytes", self.old_bytes), ("new_bytes", self.new_bytes)):
            _v1_validate_bytes(name, value)
        if not isinstance(self.pair_ids, tuple):
            raise ValueError("pair_ids must be a tuple")
        if not isinstance(self.region_ids, tuple):
            raise ValueError("region_ids must be a tuple")
        if not isinstance(self.payload_sections, tuple):
            raise ValueError("payload_sections must be a tuple")

    # ── construction ──────────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        *,
        action_id: str,
        family: str,
        authority: str,
        producer: str,
        consumer: str | None = None,
        pair_ids: Sequence[int] = (),
        region_ids: Sequence[str] = (),
        payload_sections: Sequence[str] = (),
        old_d_seg: float | None = None,
        new_d_seg: float | None = None,
        old_d_pose: float | None = None,
        new_d_pose: float | None = None,
        old_bytes: int | None = None,
        new_bytes: int | None = None,
        parseback_survived: bool | None = None,
        inflate_survived: bool | None = None,
        reference_bytes: int = CONTEST_REFERENCE_BYTES,
    ) -> ActionEffect:
        """Build an ActionEffect, computing the shared delta scores.

        Canonical constructor every ``from_*`` helper funnels through so the
        one scoring computation (:func:`compute_delta_scores`) is never
        duplicated.
        """

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
            authority=str(authority),
            producer=str(producer),
            consumer=None if consumer is None else str(consumer),
            pair_ids=_v1_int_tuple(pair_ids),
            region_ids=_v1_str_tuple(region_ids),
            payload_sections=_v1_str_tuple(payload_sections),
            old_d_seg=old_d_seg,
            new_d_seg=new_d_seg,
            old_d_pose=old_d_pose,
            new_d_pose=new_d_pose,
            old_bytes=old_bytes,
            new_bytes=new_bytes,
            delta_score_nonrate=deltas.delta_score_nonrate,
            delta_score_total=deltas.delta_score_total,
            value_per_byte=deltas.value_per_byte,
            parseback_survived=parseback_survived,
            inflate_survived=inflate_survived,
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
        * ``payload_sections`` ← ``updated_parameter_names``; byte delta from
          ``runtime_sidecar_bytes`` (new-state-only unless explicit before/after).
        """

        if not isinstance(receipt_or_payload, Mapping):
            raise TypeError("birth receipt must be a mapping")
        receipt = receipt_or_payload

        action_id = _v1_first_text(receipt, "action_id", "actuator_id") or "hinerv_target_region_birth"
        surface = _v1_first_text(receipt, "surface")
        authority = surface or _v1_first_text(receipt, "authority") or ACTION_EFFECT_PLANNING_AUTHORITY

        worst = _v1_mapping(receipt.get("worst_region"))
        region_id = _v1_birth_region_id(worst)
        region_ids = (region_id,) if region_id else ()
        pair_ids = _v1_birth_pair_ids(receipt, worst)

        exact = _v1_mapping(receipt.get("exact_nonrate"))
        old_d_seg = _v1_first_float(exact, "old_d_seg", "d_seg_old", "old_segnet_distortion")
        new_d_seg = _v1_first_float(exact, "new_d_seg", "d_seg_new", "new_segnet_distortion")
        old_d_pose = _v1_first_float(exact, "old_d_pose", "d_pose_old", "old_posenet_distortion")
        new_d_pose = _v1_first_float(exact, "new_d_pose", "d_pose_new", "new_posenet_distortion")

        old_bytes = _v1_first_int(receipt, "old_archive_bytes", "archive_bytes_old")
        new_bytes = _v1_first_int(receipt, "new_archive_bytes", "archive_bytes_new")
        sidecar = _v1_first_int(receipt, "runtime_sidecar_bytes")
        if new_bytes is None and old_bytes is not None and sidecar is not None:
            new_bytes = old_bytes + sidecar

        payload_sections = _v1_str_tuple(receipt.get("updated_parameter_names"))
        parseback_survived, inflate_survived = _v1_birth_survival_flags(receipt, surface)

        return cls.build(
            action_id=action_id,
            family="hinerv",
            authority=authority,
            producer="hinerv_target_region_birth",
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
            parseback_survived=parseback_survived,
            inflate_survived=inflate_survived,
            reference_bytes=reference_bytes,
        )

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

        pair_index = _v1_first_int(admission, "pair_index")
        if pair_index is None:
            pair_index = _v1_first_int(trace, "pair_index")
        pair_ids: tuple[int, ...] = (pair_index,) if pair_index is not None else ()

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

        region_ids = _v1_str_tuple(admission.get("affected_regions"))
        payload_sections = _v1_str_tuple(admission.get("payload_sections"))

        return cls.build(
            action_id=action_id,
            family="hinerv" if family == "hi_nerv" else family,
            authority=authority,
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
            parseback_survived=parseback_survived,
            inflate_survived=inflate_survived,
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
        pair_ids = _v1_pr110_pair_ids(row)

        score_block = _v1_mapping(row.get("score"))
        source_row = _v1_mapping(_v1_mapping(row.get("metadata")).get("source_row"))
        new_d_seg = _v1_first_float(score_block, "seg_dist", "segnet_dist")
        if new_d_seg is None:
            new_d_seg = _v1_first_float(source_row, "segnet_dist", "seg_dist")
        new_d_pose = _v1_first_float(score_block, "pose_dist", "posenet_dist")
        if new_d_pose is None:
            new_d_pose = _v1_first_float(source_row, "posenet_dist", "pose_dist")

        old_bytes, new_bytes = _v1_pr110_byte_endpoints(row)
        payload_sections = _v1_pr110_payload_sections(row)

        return cls.build(
            action_id=action_id,
            family=family,
            authority=ACTION_EFFECT_PLANNING_AUTHORITY,
            producer="pr110_frame_exploit_selector",
            consumer=consumer,
            pair_ids=pair_ids,
            region_ids=(),
            payload_sections=payload_sections,
            old_d_seg=None,
            new_d_seg=new_d_seg,
            old_d_pose=None,
            new_d_pose=new_d_pose,
            old_bytes=old_bytes,
            new_bytes=new_bytes,
            parseback_survived=None,
            inflate_survived=None,
            reference_bytes=reference_bytes,
        )

    # ── serialization ─────────────────────────────────────────────────────

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON-serializable row (tuples → lists)."""

        payload = asdict(self)
        payload["pair_ids"] = list(self.pair_ids)
        payload["region_ids"] = list(self.region_ids)
        payload["payload_sections"] = list(self.payload_sections)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ActionEffect:
        """Rebuild an ActionEffect from its :meth:`as_dict` form."""

        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")
        return cls(
            schema=str(payload.get("schema") or ACTION_EFFECT_V1_SCHEMA),
            action_id=str(payload["action_id"]),
            family=str(payload.get("family") or "shared"),
            authority=str(payload.get("authority") or ""),
            producer=str(payload.get("producer") or "unknown"),
            consumer=None if payload.get("consumer") is None else str(payload["consumer"]),
            pair_ids=_v1_int_tuple(payload.get("pair_ids") or ()),
            region_ids=_v1_str_tuple(payload.get("region_ids") or ()),
            payload_sections=_v1_str_tuple(payload.get("payload_sections") or ()),
            old_d_seg=_v1_float_or_none(payload.get("old_d_seg")),
            new_d_seg=_v1_float_or_none(payload.get("new_d_seg")),
            old_d_pose=_v1_float_or_none(payload.get("old_d_pose")),
            new_d_pose=_v1_float_or_none(payload.get("new_d_pose")),
            old_bytes=_v1_int_or_none(payload.get("old_bytes")),
            new_bytes=_v1_int_or_none(payload.get("new_bytes")),
            delta_score_nonrate=_v1_float_or_none(payload.get("delta_score_nonrate")),
            delta_score_total=_v1_float_or_none(payload.get("delta_score_total")),
            value_per_byte=_v1_float_or_none(payload.get("value_per_byte")),
            parseback_survived=_v1_bool_or_none(payload.get("parseback_survived")),
            inflate_survived=_v1_bool_or_none(payload.get("inflate_survived")),
            promotion_eligible=False,
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


__all__ = [
    "ACTION_COMMUTATOR_ROW_SCHEMA",
    "ACTION_EFFECT_LEDGER_SCHEMA",
    "ACTION_EFFECT_PLANNING_AUTHORITY",
    "ACTION_EFFECT_SCHEMA",
    "ACTION_EFFECT_V1_SCHEMA",
    "ActionEffect",
    "DeltaScores",
    "EvaluatorActionEffect",
    "append_action_effect",
    "build_action_commutator_row",
    "build_action_effect",
    "build_action_effect_ledger",
    "compute_delta_scores",
    "read_action_effects",
]
