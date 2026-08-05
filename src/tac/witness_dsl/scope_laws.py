# SPDX-License-Identifier: MIT
"""Scope-law resolver for dynamic-but-deterministic trainer constants.

``LawRef`` resolves constants at DSL compile time.  Scope laws are the adjacent
runtime surface: a sealed ticket declares the law and its input schema, then the
trainer resolves the value at the stage/window/gate where those inputs actually
exist.  Values are deterministic functions of the sealed config plus
deterministic telemetry, and each resolution row is suitable for checkpoint
metadata and telemetry.

Axis: apparatus.  ``score_claim=false``.  No scorer, archive, or frontier claim.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from tac.witness_dsl.lawref import InputRef, LawRef, lawref_to_declaration

ScopeLawTier = Literal["T2_SCOPE_LAW", "T3_LIVE_ADAPTED"]

T2_SCOPE_LAW: ScopeLawTier = "T2_SCOPE_LAW"
T3_LIVE_ADAPTED: ScopeLawTier = "T3_LIVE_ADAPTED"


class ScopeLawError(ValueError):
    """A scope-law declaration or evaluation is invalid."""


@dataclass(frozen=True)
class ScopeLawEvaluation:
    value: float | int | str | bool
    provenance: str
    explicit_override: bool = False


@dataclass(frozen=True)
class InertnessAlarm:
    """The alarm attached to a scope law when a declared law never resolves."""

    alarm_id: str
    positive_control: str
    refusal: str

    def to_dict(self) -> dict[str, str]:
        return {
            "alarm_id": self.alarm_id,
            "positive_control": self.positive_control,
            "refusal": self.refusal,
        }


@dataclass(frozen=True)
class ScopeLaw:
    name: str
    tier: ScopeLawTier
    input_fields: tuple[str, ...]
    evaluator: Callable[[Mapping[str, Any]], ScopeLawEvaluation]
    output_field: str
    provenance: str
    inertness_alarm: InertnessAlarm
    lawref: LawRef | None = None

    def resolve(self, inputs: Mapping[str, Any]) -> ScopeLawResolution:
        missing = [name for name in self.input_fields if name not in inputs]
        if missing:
            raise ScopeLawError(f"{self.name} missing required input field(s): {missing}")
        normalized_inputs = {name: _json_safe(inputs[name]) for name in self.input_fields}
        result = self.evaluator(normalized_inputs)
        row = ScopeLawResolution(
            name=self.name,
            law=self.name,
            tier=self.tier,
            inputs=normalized_inputs,
            resolved_value=_json_safe(result.value),
            provenance=result.provenance,
            inertness_alarm=self.inertness_alarm.to_dict(),
            explicit_override=bool(result.explicit_override),
            output_field=self.output_field,
        )
        return row.with_hash()

    def to_ticket_ref(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "name": self.name,
            "tier": self.tier,
            "inputs_schema": list(self.input_fields),
            "output_field": self.output_field,
            "provenance": self.provenance,
            "inertness_alarm": self.inertness_alarm.to_dict(),
        }
        if self.lawref is not None:
            row["lawref_declaration"] = lawref_to_declaration(self.lawref)
        return row


@dataclass(frozen=True)
class ScopeLawResolution:
    name: str
    law: str
    tier: ScopeLawTier
    inputs: Mapping[str, Any]
    resolved_value: Any
    provenance: str
    inertness_alarm: Mapping[str, Any]
    explicit_override: bool
    output_field: str
    resolution_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        row = {
            "name": self.name,
            "law": self.law,
            "tier": self.tier,
            "inputs": dict(self.inputs),
            "resolved_value": self.resolved_value,
            "provenance": self.provenance,
            "inertness_alarm": dict(self.inertness_alarm),
            "explicit_override": bool(self.explicit_override),
            "output_field": self.output_field,
            "resolution_hash": self.resolution_hash,
            "score_claim": False,
        }
        return row

    def with_hash(self) -> ScopeLawResolution:
        payload = self.to_dict()
        payload.pop("resolution_hash", None)
        digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
        return ScopeLawResolution(
            name=self.name,
            law=self.law,
            tier=self.tier,
            inputs=dict(self.inputs),
            resolved_value=self.resolved_value,
            provenance=self.provenance,
            inertness_alarm=dict(self.inertness_alarm),
            explicit_override=self.explicit_override,
            output_field=self.output_field,
            resolution_hash=digest,
        )


def _canonical_json(value: Any) -> str:
    return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"))


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(v) for v in value]
    return str(value)


def _finite_float(inputs: Mapping[str, Any], key: str) -> float:
    try:
        value = float(inputs[key])
    except (TypeError, ValueError) as exc:
        raise ScopeLawError(f"{key} must be a finite float, got {inputs.get(key)!r}") from exc
    if not math.isfinite(value):
        raise ScopeLawError(f"{key} must be finite, got {value!r}")
    return value


def _positive_int(inputs: Mapping[str, Any], key: str) -> int:
    try:
        value = int(inputs[key])
    except (TypeError, ValueError) as exc:
        raise ScopeLawError(f"{key} must be a positive int, got {inputs.get(key)!r}") from exc
    if value <= 0:
        raise ScopeLawError(f"{key} must be > 0, got {value!r}")
    return value


def _nonnegative_int(inputs: Mapping[str, Any], key: str) -> int:
    try:
        value = int(inputs[key])
    except (TypeError, ValueError) as exc:
        raise ScopeLawError(f"{key} must be a non-negative int, got {inputs.get(key)!r}") from exc
    if value < 0:
        raise ScopeLawError(f"{key} must be >= 0, got {value!r}")
    return value


def scope_law_geometry_hash(
    *, steps_per_epoch: int, horizon_epochs: int, window_epochs: int
) -> str:
    """Hash the geometry that makes a resolved-at-consumption value valid.

    This keeps runtime-derived values keyed by the run geometry that produced
    them, not by the flag name that requested them.  A resumed window with a
    different horizon/window/step geometry therefore gets a different key and
    must re-resolve.
    """

    payload = {
        "steps_per_epoch": _positive_int({"steps_per_epoch": steps_per_epoch}, "steps_per_epoch"),
        "horizon_epochs": _positive_int({"horizon_epochs": horizon_epochs}, "horizon_epochs"),
        "window_epochs": _positive_int({"window_epochs": window_epochs}, "window_epochs"),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def derive_ema_decay_from_updates(total_updates: int) -> tuple[float, str]:
    """Mirror the TR1 trainer's EMA LawRef path without importing MLX."""

    u_raw = int(total_updates)
    u_eff = max(u_raw, 8)
    try:
        from tac.canonical_equations.evaluators import eval_ema_decay_run_geometry

        decay = float(eval_ema_decay_run_geometry({
            "mode": "decay_from_warmup_fraction",
            "warmup_fraction": 0.5,
            "updates_per_run": u_eff,
        }))
        provenance = (
            "DERIVED ema_decay_run_geometry_v1 decay_from_warmup_fraction "
            f"phi=0.5 U={u_raw} -> {decay:.6f}"
        )
    except Exception as exc:
        decay = 1.0 - 2.0 / (0.5 * u_eff)
        provenance = (
            f"DERIVED closed-form d=1-2/(phi*U) phi=0.5 U={u_raw} -> "
            f"{decay:.6f} (LawRef evaluator import failed: {exc})"
        )
    ceiling = 1.0 - 2.0 / u_eff
    if decay > ceiling:
        decay = ceiling
        provenance += f"; run-geometry ceiling d<=1-2/U={ceiling:.6f} bound (no constant clamp)"
    else:
        provenance += f"; within run-geometry window (ceiling 1-2/U={ceiling:.6f}, no constant clamp)"
    return decay, provenance


def _eval_stage_ema_decay(inputs: Mapping[str, Any]) -> ScopeLawEvaluation:
    remaining_epochs = _positive_int(inputs, "remaining_epochs")
    steps_per_epoch = _positive_int(inputs, "steps_per_epoch")
    if not str(inputs["run_geometry_hash"]).strip():
        raise ScopeLawError("run_geometry_hash must be non-empty")
    updates = remaining_epochs * steps_per_epoch
    decay, provenance = derive_ema_decay_from_updates(updates)
    return ScopeLawEvaluation(
        value=decay,
        provenance=(
            f"JD3 stage-scoped window EMA: {provenance}; "
            f"run_geometry_hash={inputs['run_geometry_hash']}"
        ),
    )


def _eval_realized_hold_margin(inputs: Mapping[str, Any]) -> ScopeLawEvaluation:
    explicit = _finite_float(inputs, "explicit_margin")
    if explicit > 0.0:
        return ScopeLawEvaluation(
            value=explicit,
            provenance=f"EXPLICIT --jd1-realized-hold-margin {explicit}",
            explicit_override=True,
        )
    if explicit < 0.0:
        raise ScopeLawError("explicit_margin must be >= 0")
    sd = _finite_float(inputs, "realized_gate_dseg_per_pair_sd")
    ids = inputs["realized_gate_pair_ids"]
    if not isinstance(ids, Sequence) or isinstance(ids, (str, bytes, bytearray)):
        raise ScopeLawError("realized_gate_pair_ids must be a sequence")
    n_gate = len(ids)
    if n_gate <= 0:
        raise ScopeLawError("realized_gate_pair_ids must be non-empty")
    margin = sd / math.sqrt(float(n_gate))
    return ScopeLawEvaluation(
        value=margin,
        provenance=(
            "DERIVED first post-engagement realized gate uncertainty: "
            f"sd(per-pair d_seg)={sd:.12g}/sqrt(n_gate={n_gate}) -> {margin:.12g}"
        ),
    )


def _eval_realized_hold_floor_latch(inputs: Mapping[str, Any]) -> ScopeLawEvaluation:
    floor = _finite_float(inputs, "realized_gate_dseg_mean")
    return ScopeLawEvaluation(
        value=floor,
        provenance="DERIVED first post-engagement realized gate floor latch",
    )


def _eval_pose_retreat_bisection(inputs: Mapping[str, Any]) -> ScopeLawEvaluation:
    explicit = _finite_float(inputs, "explicit_pose_retreat")
    if explicit > 0.0:
        if explicit >= 1.0:
            raise ScopeLawError("explicit_pose_retreat must be 0.0 or in (0,1)")
        return ScopeLawEvaluation(
            value=explicit,
            provenance=f"EXPLICIT --jd1-realized-hold-pose-retreat {explicit}",
            explicit_override=True,
        )
    if explicit < 0.0:
        raise ScopeLawError("explicit_pose_retreat must be >= 0")
    return ScopeLawEvaluation(
        value=0.5,
        provenance="DERIVED bisection retreat of pose-pressure interval (0.5)",
    )


def _eval_max_retreats_a1(inputs: Mapping[str, Any]) -> ScopeLawEvaluation:
    explicit = int(inputs["explicit_max_retreats"])
    if explicit > 0:
        return ScopeLawEvaluation(
            value=explicit,
            provenance=f"EXPLICIT --jd1-realized-hold-max-retreats {explicit}",
            explicit_override=True,
        )
    if explicit < 0:
        raise ScopeLawError("explicit_max_retreats must be >= 0")
    a1_refuse = _positive_int(inputs, "a1_consecutive_refuse")
    return ScopeLawEvaluation(
        value=a1_refuse,
        provenance=(
            "DERIVED from A1_CONSECUTIVE_REFUSE: allow the same number of "
            "rollback+retreat events that would otherwise terminate the stage"
        ),
    )


def _eval_jd1_plateau_tail_average_ema(inputs: Mapping[str, Any]) -> ScopeLawEvaluation:
    updates_since_anchor = _nonnegative_int(inputs, "updates_since_anchor")
    live_weight = 1.0 / float(updates_since_anchor + 2)
    return ScopeLawEvaluation(
        value=live_weight,
        provenance=(
            "DERIVED JD1 plateau-tail EMA update: "
            "shadow_{n+1}=shadow_n+(live_n-shadow_n)/(n+2) after an anchor sample; "
            f"updates_since_anchor={updates_since_anchor} -> live_weight={live_weight:.12g}"
        ),
    )


def _ema_lawref() -> LawRef:
    return LawRef(
        equation_id="ema_decay_run_geometry_v1",
        inputs={
            "mode": InputRef.literal(
                2, "mode code 2 == decay_from_warmup_fraction for scope-law ticket custody"
            ),
            "warmup_fraction": InputRef.literal(
                0.5, "JD3 stage-window EMA law pins two-time-constant warmup at half window"
            ),
            "updates_per_run": InputRef.literal(
                1, "placeholder schema input; runtime supplies remaining_epochs*steps_per_epoch"
            ),
        },
        ladder_class="derived_live",
    )


def _jd1_tail_average_lawref() -> LawRef:
    return LawRef(
        equation_id="jd1_plateau_tail_average_ema_v1",
        inputs={
            "updates_since_anchor": InputRef.literal(
                0,
                "placeholder schema input; runtime supplies the settled-live update count",
            ),
        },
        ladder_class="derived_live",
    )


SCOPE_LAWS: dict[str, ScopeLaw] = {
    "jd3_stage_ema_decay": ScopeLaw(
        name="jd3_stage_ema_decay",
        tier=T2_SCOPE_LAW,
        input_fields=("remaining_epochs", "steps_per_epoch", "run_geometry_hash"),
        output_field="active_ema_decay",
        evaluator=_eval_stage_ema_decay,
        provenance="jd3 cure for parent-horizon EMA transfer: rederive at the current window",
        inertness_alarm=InertnessAlarm(
            alarm_id="jd3_stage_ema_decay_INERT",
            positive_control="declared --jd1-ema-stage-scope window but no stage EMA resolution row",
            refusal="block checkpoint selection until the stage-window EMA resolves",
        ),
        lawref=_ema_lawref(),
    ),
    "jd3_realized_hold_margin": ScopeLaw(
        name="jd3_realized_hold_margin",
        tier=T3_LIVE_ADAPTED,
        input_fields=(
            "explicit_margin",
            "realized_gate_dseg_per_pair_sd",
            "realized_gate_pair_ids",
        ),
        output_field="realized_hold.margin",
        evaluator=_eval_realized_hold_margin,
        provenance="first realized gate scatter sets the hold slack at the gate scope",
        inertness_alarm=InertnessAlarm(
            alarm_id="jd3_realized_hold_margin_INERT",
            positive_control="declared realized hold but first gate never resolved margin",
            refusal="block realized-hold adoption until margin resolution exists",
        ),
    ),
    "jd3_realized_hold_floor_latch": ScopeLaw(
        name="jd3_realized_hold_floor_latch",
        tier=T3_LIVE_ADAPTED,
        input_fields=("realized_gate_dseg_mean",),
        output_field="realized_hold.floor",
        evaluator=_eval_realized_hold_floor_latch,
        provenance="first realized gate latches the actual hold-space floor",
        inertness_alarm=InertnessAlarm(
            alarm_id="jd3_realized_hold_floor_INERT",
            positive_control="declared realized hold but no floor latch resolution row",
            refusal="block selection because the realized guard has no floor",
        ),
    ),
    "jd3_pose_retreat_bisection": ScopeLaw(
        name="jd3_pose_retreat_bisection",
        tier=T3_LIVE_ADAPTED,
        input_fields=("explicit_pose_retreat",),
        output_field="realized_hold.pose_retreat_factor",
        evaluator=_eval_pose_retreat_bisection,
        provenance="0.0 sentinel derives bisection retreat at the controller boundary",
        inertness_alarm=InertnessAlarm(
            alarm_id="jd3_pose_retreat_INERT",
            positive_control="declared realized hold but no pose-retreat resolution row",
            refusal="block rollback controller because retreat factor never resolved",
        ),
    ),
    "jd3_max_retreats_a1_policy": ScopeLaw(
        name="jd3_max_retreats_a1_policy",
        tier=T2_SCOPE_LAW,
        input_fields=("explicit_max_retreats", "a1_consecutive_refuse"),
        output_field="realized_hold.max_retreats",
        evaluator=_eval_max_retreats_a1,
        provenance="0 sentinel derives retreat budget from the A1 consecutive-refuse policy",
        inertness_alarm=InertnessAlarm(
            alarm_id="jd3_max_retreats_INERT",
            positive_control="declared realized hold but no max-retreats resolution row",
            refusal="block rollback controller because retreat budget never resolved",
        ),
    ),
    "jd1_plateau_tail_average_ema": ScopeLaw(
        name="jd1_plateau_tail_average_ema",
        tier=T3_LIVE_ADAPTED,
        input_fields=("updates_since_anchor",),
        output_field="ema_tail_live_weight",
        evaluator=_eval_jd1_plateau_tail_average_ema,
        provenance=(
            "dy2 plateau-tail average law: after explicit anchor, each settled live "
            "iterate enters the shipping shadow with weight 1/(n+2)"
        ),
        inertness_alarm=InertnessAlarm(
            alarm_id="jd1_plateau_tail_average_ema_INERT",
            positive_control=(
                "declared plateau_tail_average mode but no tail live-weight resolution row"
            ),
            refusal="block tail-average adoption because the EMA update law never resolved",
        ),
        lawref=_jd1_tail_average_lawref(),
    ),
}


def scope_law(name: str) -> ScopeLaw:
    try:
        return SCOPE_LAWS[name]
    except KeyError as exc:
        raise ScopeLawError(f"unknown scope law {name!r}") from exc


def resolve_scope_law(name: str, inputs: Mapping[str, Any]) -> dict[str, Any]:
    return scope_law(name).resolve(inputs).to_dict()


def ticket_scope_law_refs(names: Sequence[str]) -> list[dict[str, Any]]:
    return [scope_law(name).to_ticket_ref() for name in names]


def jd3_default_scope_law_refs() -> list[dict[str, Any]]:
    return ticket_scope_law_refs((
        "jd3_stage_ema_decay",
        "jd3_realized_hold_margin",
        "jd3_realized_hold_floor_latch",
        "jd3_pose_retreat_bisection",
        "jd3_max_retreats_a1_policy",
    ))


def jd1_tail_average_scope_law_refs() -> list[dict[str, Any]]:
    return ticket_scope_law_refs(("jd1_plateau_tail_average_ema",))


def validate_ticket_scope_laws(rows: Sequence[Mapping[str, Any]]) -> None:
    seen: set[str] = set()
    for row in rows:
        name = str(row.get("name", ""))
        if name in seen:
            raise ScopeLawError(f"duplicate scope law in ticket: {name}")
        seen.add(name)
        law = scope_law(name)
        if row.get("tier") != law.tier:
            raise ScopeLawError(f"{name}: ticket tier {row.get('tier')!r} != {law.tier!r}")
        if tuple(row.get("inputs_schema", ())) != law.input_fields:
            raise ScopeLawError(f"{name}: ticket inputs_schema differs from registry")
        if row.get("output_field") != law.output_field:
            raise ScopeLawError(f"{name}: ticket output_field differs from registry")
        if law.lawref is not None:
            expected = lawref_to_declaration(law.lawref)
            if row.get("lawref_declaration") != expected:
                raise ScopeLawError(f"{name}: ticket lawref_declaration differs from registry")
        elif "lawref_declaration" in row:
            raise ScopeLawError(f"{name}: ticket declares unexpected lawref_declaration")


def ticket_payload_hash(payload: Mapping[str, Any]) -> str:
    clean = {k: v for k, v in payload.items() if k != "ticket_hash"}
    return hashlib.sha256(_canonical_json(clean).encode("utf-8")).hexdigest()


def attach_scope_laws_to_ticket(payload: Mapping[str, Any],
                                scope_laws: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    validate_ticket_scope_laws(scope_laws)
    out = dict(payload)
    if scope_laws:
        out["scope_laws"] = [dict(row) for row in scope_laws]
    out["ticket_hash"] = ticket_payload_hash(out)
    return out


def inertness_violations(
    declared_scope_laws: Sequence[Mapping[str, Any]],
    resolved_scope_laws: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return declared laws that never resolved.

    This is the positive-control surface for the adaptive_eps_INERT class:
    a pinned/declared law with zero resolution events is an alarm, not a pass.
    """

    declared_names = {str(row.get("name", "")) for row in declared_scope_laws}
    resolved_names = {str(row.get("name", "")) for row in resolved_scope_laws}
    violations: list[dict[str, Any]] = []
    for name in sorted(declared_names - resolved_names):
        law = scope_law(name)
        violations.append({
            "name": name,
            "status": "INERT",
            "inertness_alarm": law.inertness_alarm.to_dict(),
            "score_claim": False,
        })
    return violations


__all__ = [
    "SCOPE_LAWS",
    "T2_SCOPE_LAW",
    "T3_LIVE_ADAPTED",
    "ScopeLaw",
    "ScopeLawError",
    "ScopeLawEvaluation",
    "ScopeLawResolution",
    "attach_scope_laws_to_ticket",
    "derive_ema_decay_from_updates",
    "inertness_violations",
    "jd1_tail_average_scope_law_refs",
    "jd3_default_scope_law_refs",
    "resolve_scope_law",
    "scope_law",
    "scope_law_geometry_hash",
    "ticket_payload_hash",
    "ticket_scope_law_refs",
    "validate_ticket_scope_laws",
]
