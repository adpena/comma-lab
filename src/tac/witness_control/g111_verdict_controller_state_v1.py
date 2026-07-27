# SPDX-License-Identifier: MIT
"""Pure replacement state for Fresh-G111 verdict controller side effects.

The native-v3 verdict worker/reducer already produces an immutable live effect.
This module consumes that effect without touching trainer objects, stdout, files,
or clocks.  It reduces every legacy verdict-cadence controller mutation into one
bounded, serializable state tree and returns deterministic publication intents.

The caller owns the external transaction:

1. adapt the committed live-reducer effect with :func:`adapt_live_reducer_effect`;
2. reduce it with :func:`reduce_controller_state`;
3. durably publish/ack the returned idempotent intents; and
4. install/checkpoint the returned state before the next trajectory decision.

This module is not launch authority and deliberately does not bypass the
trainer's native-v3 REFUSE guard.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any, Final

import numpy as np

from tac.contest_score import UNCOMPRESSED_SIZE_BYTES
from tac.witness_control.birth_completion import birth_complete, birth_persistence
from tac.witness_control.event_wirings import lane_nucleus_event
from tac.witness_control.g111_live_verdict_transaction_v1 import WORKER_PAYLOAD_SCHEMA
from tac.witness_control.g111_verdict_barrier_v1 import ImmutableVerdictResult
from tac.witness_curriculum.ladder_homotopy import (
    ARM_LANE,
    ARM_MOVABLE,
    perclass_lambda_proxy,
)

SCHEMA: Final = "tac.g111_verdict_controller_state.v1"
OBSERVATION_SCHEMA: Final = "tac.g111_verdict_controller_observation.v1"
INTENT_SCHEMA: Final = "tac.g111_verdict_controller_intent.v1"
SERIALIZED_STATE_SCHEMA: Final = "tac.g111_verdict_controller_state_arrays.v1"
STATE_RESULT_ID: Final = "g111-verdict-controller-state"

DEFAULT_HISTORY_LIMIT: Final = 128
DEFAULT_CLOSED_LOOP_LIMIT: Final = 128
DEFAULT_ANNULUS_LIMIT: Final = 16
DEFAULT_LABEL_FLOOR_LIMIT: Final = 128
DEFAULT_JOURNAL_LIMIT: Final = 64
DEFAULT_BIRTH_OBSERVATION_LIMIT: Final = 1

_EFFECT_KEYS: Final = frozenset(
    {
        "sequence",
        "result_id",
        "result_sha256",
        "best_intent_sequence",
        "payload",
    }
)
_PAYLOAD_KEYS: Final = frozenset(
    {
        "schema",
        "epoch",
        "seg_form",
        "ep_loss",
        "blob_bytes",
        "best_eligible",
        "closed_loop_enabled",
        "liveness",
        "scorer",
        "verdict",
        "live_gap",
    }
)
_OBSERVATION_KEYS: Final = frozenset(
    {
        "schema",
        "result_sequence",
        "result_id",
        "result_sha256",
        "epoch",
        "seg_form",
        "ep_loss",
        "blob_bytes",
        "d_seg",
        "d_pose",
        "implied_S",
        "closed_loop_enabled",
        "liveness",
        "live_gap",
        "nucleus_counts",
        "nucleus_stats",
        "annulus",
        "annulus_flip_frac",
        "per_class",
        "pose_gate",
    }
)
_STATE_KEYS: Final = frozenset(
    {
        "schema",
        "config",
        "journal_base_sequence",
        "next_sequence",
        "journal",
        "history",
        "closed_loop_verdicts",
        "nucleus_ready",
        "lane_sensor",
        "annulus_series",
        "label_floor_series",
        "last_d_pose",
        "ladder_costates",
        "birth_completion",
    }
)
_BIRTH_KEYS: Final = frozenset({"classes", "fired_epochs", "observations"})
_JOURNAL_KEYS: Final = frozenset({"sequence", "result_id", "result_sha256", "intents"})
_INTENT_KEYS: Final = frozenset(
    {
        "schema",
        "channel",
        "ordinal",
        "result_sequence",
        "result_id",
        "result_sha256",
        "intent_sha256",
        "idempotency_key",
        "payload",
    }
)


class G111VerdictControllerStateError(RuntimeError):
    """The replacement controller state/effect is malformed or out of order."""


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise G111VerdictControllerStateError(f"{name} must be a mapping")
    if any(type(key) is not str for key in value):
        raise G111VerdictControllerStateError(f"{name} keys must have exact str type")
    return value


def _keys(value: Mapping[str, Any], expected: frozenset[str], *, name: str) -> None:
    if set(value) != expected:
        raise G111VerdictControllerStateError(f"{name} fields {sorted(value)!r} != expected {sorted(expected)!r}")


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise G111VerdictControllerStateError(f"{name} must be an exact integer")
    result = int(value)
    if result < minimum:
        raise G111VerdictControllerStateError(f"{name} must be >= {minimum}")
    return result


def _number(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise G111VerdictControllerStateError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise G111VerdictControllerStateError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise G111VerdictControllerStateError(f"{name} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise G111VerdictControllerStateError(f"{name} must be <= {maximum}")
    return result


def _boolean(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise G111VerdictControllerStateError(f"{name} must be an exact bool")
    return bool(value)


def _string(value: object, *, name: str, nonempty: bool = True) -> str:
    if type(value) is not str or (nonempty and not value):
        suffix = "non-empty exact string" if nonempty else "exact string"
        raise G111VerdictControllerStateError(f"{name} must be a {suffix}")
    return value


def _sha256(value: object, *, name: str) -> str:
    result = _string(value, name=name)
    if len(result) != 64 or result != result.lower():
        raise G111VerdictControllerStateError(f"{name} must be a lowercase SHA-256 hex string")
    try:
        bytes.fromhex(result)
    except ValueError as exc:
        raise G111VerdictControllerStateError(f"{name} must be a lowercase SHA-256 hex string") from exc
    return result


def _clone(value: Any) -> Any:
    encoded = ImmutableVerdictResult.capture(
        submission_seq=0,
        result_id=STATE_RESULT_ID,
        payload={"value": value},
    )
    return encoded.payload["value"]


def _bounded_append(rows: list[Any], row: Any, limit: int) -> None:
    rows.append(row)
    del rows[:-limit]


def _utf8_array(value: str) -> np.ndarray:
    return np.frombuffer(value.encode("utf-8"), dtype=np.uint8).copy()


def _decode_utf8(value: object, *, name: str) -> str:
    array = np.asarray(value)
    if array.dtype != np.dtype(np.uint8) or array.ndim != 1:
        raise G111VerdictControllerStateError(f"{name} must be a one-dimensional uint8 array")
    try:
        return array.tobytes().decode("utf-8")
    except UnicodeError as exc:
        raise G111VerdictControllerStateError(f"{name} is not valid UTF-8") from exc


@dataclass(frozen=True, slots=True)
class G111VerdictControllerConfigV1:
    """All controller-affecting gates and bounds for one typed G111 config."""

    typed_config_sha256: str
    history_limit: int = DEFAULT_HISTORY_LIMIT
    closed_loop_limit: int = DEFAULT_CLOSED_LOOP_LIMIT
    annulus_limit: int = DEFAULT_ANNULUS_LIMIT
    label_floor_limit: int = DEFAULT_LABEL_FLOOR_LIMIT
    journal_limit: int = DEFAULT_JOURNAL_LIMIT
    birth_observation_limit: int = DEFAULT_BIRTH_OBSERVATION_LIMIT
    closed_loop_enabled: bool = True
    nucleus_guard_enabled: bool = True
    lane_sensor_enabled: bool = True
    annulus_sensor_enabled: bool = True
    label_floor_sensor_enabled: bool = False
    w_pose_law_enabled: bool = False
    ladder_enabled: bool = True
    birth_completion_enabled: bool = True
    require_live_d_pose: bool = True
    n_classes: int = 5
    nucleus_within_flip_thresh: float = 0.5
    nucleus_min_part_frac: float = 0.0
    lane_class: int = 1
    movable_class: int = 3
    birth_classes: tuple[int, ...] = (1, 3)
    birth_tau_persist: float = 0.8
    birth_area_band: float = 0.25
    birth_ramp_epochs: int = 50
    birth_post_level: float = 0.2

    def validate(self) -> None:
        _sha256(self.typed_config_sha256, name="typed_config_sha256")
        for name in (
            "history_limit",
            "closed_loop_limit",
            "annulus_limit",
            "label_floor_limit",
            "journal_limit",
            "birth_observation_limit",
        ):
            _integer(getattr(self, name), name=name, minimum=1)
        for name in (
            "closed_loop_enabled",
            "nucleus_guard_enabled",
            "lane_sensor_enabled",
            "annulus_sensor_enabled",
            "label_floor_sensor_enabled",
            "w_pose_law_enabled",
            "ladder_enabled",
            "birth_completion_enabled",
            "require_live_d_pose",
        ):
            _boolean(getattr(self, name), name=name)
        n_classes = _integer(self.n_classes, name="n_classes", minimum=1)
        _number(
            self.nucleus_within_flip_thresh,
            name="nucleus_within_flip_thresh",
            minimum=0.0,
            maximum=1.0,
        )
        _number(
            self.nucleus_min_part_frac,
            name="nucleus_min_part_frac",
            minimum=0.0,
            maximum=1.0,
        )
        lane_class = _integer(self.lane_class, name="lane_class")
        if lane_class >= n_classes:
            raise G111VerdictControllerStateError("lane_class exceeds n_classes")
        movable_class = _integer(self.movable_class, name="movable_class")
        if movable_class >= n_classes:
            raise G111VerdictControllerStateError("movable_class exceeds n_classes")
        if lane_class == movable_class:
            raise G111VerdictControllerStateError("lane_class and movable_class must differ")
        if not self.birth_classes or len(set(self.birth_classes)) != len(self.birth_classes):
            raise G111VerdictControllerStateError("birth_classes must be non-empty and unique")
        for cls in self.birth_classes:
            value = _integer(cls, name="birth class")
            if value >= n_classes:
                raise G111VerdictControllerStateError("birth class exceeds n_classes")
        _number(
            self.birth_tau_persist,
            name="birth_tau_persist",
            minimum=0.0,
            maximum=1.0,
        )
        _number(self.birth_area_band, name="birth_area_band", minimum=0.0)
        _integer(self.birth_ramp_epochs, name="birth_ramp_epochs", minimum=1)
        _number(
            self.birth_post_level,
            name="birth_post_level",
            minimum=0.0,
            maximum=1.0,
        )

    def to_mapping(self) -> dict[str, Any]:
        self.validate()
        value = asdict(self)
        value["birth_classes"] = list(self.birth_classes)
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> G111VerdictControllerConfigV1:
        raw = dict(_mapping(value, name="controller config"))
        expected = set(cls.__dataclass_fields__)
        if set(raw) != expected:
            raise G111VerdictControllerStateError("serialized controller config fields differ")
        classes = raw.get("birth_classes")
        if not isinstance(classes, list):
            raise G111VerdictControllerStateError("serialized birth_classes must be a list")
        raw["birth_classes"] = tuple(classes)
        try:
            result = cls(**raw)
        except TypeError as exc:
            raise G111VerdictControllerStateError("serialized controller config cannot be constructed") from exc
        result.validate()
        return result


def active_g111_controller_config_v1(*, typed_config_sha256: str) -> G111VerdictControllerConfigV1:
    """Bind active Fresh-G111 gates to the caller's current typed-DSL compile.

    The hash is deliberately not discovered or cached here. The checkpoint
    transaction must pass the compile hash it already owns.
    """

    config = G111VerdictControllerConfigV1(
        typed_config_sha256=typed_config_sha256,
        closed_loop_enabled=True,
        nucleus_guard_enabled=True,
        lane_sensor_enabled=True,
        annulus_sensor_enabled=True,
        label_floor_sensor_enabled=False,
        w_pose_law_enabled=False,
        ladder_enabled=True,
        birth_completion_enabled=True,
        require_live_d_pose=True,
        nucleus_within_flip_thresh=0.5,
        nucleus_min_part_frac=0.0,
        lane_class=1,
        movable_class=3,
        birth_classes=(1, 3),
        birth_tau_persist=0.8,
        birth_area_band=0.25,
        birth_ramp_epochs=50,
        birth_post_level=0.2,
    )
    config.validate()
    return config


def new_controller_state(
    config: G111VerdictControllerConfigV1,
    *,
    history: Sequence[Mapping[str, Any]] = (),
    closed_loop_verdicts: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Create detached bounded replacement state for a fresh controller."""

    cfg = config
    cfg.validate()
    state = {
        "schema": SCHEMA,
        "config": cfg.to_mapping(),
        "journal_base_sequence": 0,
        "next_sequence": 0,
        "journal": [],
        "history": [dict(row) for row in history][-cfg.history_limit :],
        "closed_loop_verdicts": [dict(row) for row in closed_loop_verdicts][-cfg.closed_loop_limit :],
        "nucleus_ready": True,
        "lane_sensor": None,
        "annulus_series": [],
        "label_floor_series": [],
        "last_d_pose": None,
        "ladder_costates": None,
        "birth_completion": {
            "classes": list(cfg.birth_classes),
            "fired_epochs": {},
            "observations": [],
        },
    }
    validate_controller_state(state)
    return state


def _validate_counts(
    counts: object,
    *,
    config: G111VerdictControllerConfigV1,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    value = _mapping(counts, name="nucleus_counts")
    expected = {"total_px", "pred_px", "gt_px", "wrong_px", "n_classes"}
    if set(value) != expected:
        raise G111VerdictControllerStateError("nucleus_counts fields differ")
    n_classes = _integer(value["n_classes"], name="nucleus_counts.n_classes")
    if n_classes != config.n_classes:
        raise G111VerdictControllerStateError("nucleus_counts.n_classes differs from config")
    total = _integer(value["total_px"], name="nucleus_counts.total_px", minimum=1)
    vectors: dict[str, list[int]] = {}
    for name in ("pred_px", "gt_px", "wrong_px"):
        raw = value[name]
        if not isinstance(raw, (list, tuple)) or len(raw) != n_classes:
            raise G111VerdictControllerStateError(f"nucleus_counts.{name} must contain exactly {n_classes} values")
        vectors[name] = [_integer(item, name=f"nucleus_counts.{name}[{index}]") for index, item in enumerate(raw)]
    if sum(vectors["pred_px"]) != total or sum(vectors["gt_px"]) != total:
        raise G111VerdictControllerStateError("nucleus count partitions must each sum to total_px")
    if any(wrong > gt for wrong, gt in zip(vectors["wrong_px"], vectors["gt_px"], strict=True)):
        raise G111VerdictControllerStateError("wrong_px cannot exceed gt_px")
    normalized = {
        "total_px": total,
        "pred_px": vectors["pred_px"],
        "gt_px": vectors["gt_px"],
        "wrong_px": vectors["wrong_px"],
        "n_classes": n_classes,
    }
    stats: dict[str, dict[str, Any]] = {}
    for cls in range(n_classes):
        gt = vectors["gt_px"][cls]
        stats[str(cls)] = {
            "part_frac": vectors["pred_px"][cls] / total,
            "within_flip": vectors["wrong_px"][cls] / gt if gt else 0.0,
            "gt_px": gt,
            "pred_px": vectors["pred_px"][cls],
            "gt_area": gt / total,
        }
    return normalized, stats


def _annulus_value(
    annulus: object,
    *,
    required: bool,
) -> tuple[dict[str, Any] | None, float | None]:
    if annulus is None:
        if required:
            raise G111VerdictControllerStateError("active annulus controller requires verdict.annulus")
        return None, None
    value = dict(_mapping(annulus, name="verdict.annulus"))
    if "error" in value:
        if set(value) != {"error"}:
            raise G111VerdictControllerStateError("annulus error payload must contain only error")
        _string(value["error"], name="verdict.annulus.error")
        return _clone(value), None
    threshold = _mapping(value.get("threshold"), name="verdict.annulus.threshold")
    fraction = _number(
        threshold.get("annulus_flip_frac"),
        name="verdict.annulus.threshold.annulus_flip_frac",
        minimum=0.0,
        maximum=1.0,
    )
    return _clone(value), fraction


def adapt_live_reducer_effect(
    effect: Mapping[str, Any],
    *,
    config: G111VerdictControllerConfigV1,
) -> dict[str, Any]:
    """Validate one committed live-reducer effect and extract sufficient state input."""

    cfg = config
    cfg.validate()
    value = _mapping(effect, name="live reducer effect")
    _keys(value, _EFFECT_KEYS, name="live reducer effect")
    sequence = _integer(value["sequence"], name="effect sequence")
    result_id = _string(value["result_id"], name="effect result_id")
    result_sha256 = _sha256(value["result_sha256"], name="effect result_sha256")
    best_sequence = value["best_intent_sequence"]
    if best_sequence is not None:
        _integer(best_sequence, name="effect best_intent_sequence")
    payload = _mapping(value["payload"], name="effect payload")
    _keys(payload, _PAYLOAD_KEYS, name="effect payload")
    if payload["schema"] != WORKER_PAYLOAD_SCHEMA:
        raise G111VerdictControllerStateError("effect worker payload schema differs")
    reconstructed = ImmutableVerdictResult.capture(
        submission_seq=sequence,
        result_id=result_id,
        payload=payload,
    )
    if reconstructed.result_sha256 != result_sha256:
        raise G111VerdictControllerStateError("effect result SHA-256 does not match canonical payload")

    epoch = _integer(payload["epoch"], name="payload epoch")
    seg_form = _string(payload["seg_form"], name="payload seg_form")
    ep_loss = _number(payload["ep_loss"], name="payload ep_loss")
    blob_bytes = _integer(payload["blob_bytes"], name="payload blob_bytes")
    closed_loop = _boolean(payload["closed_loop_enabled"], name="payload closed_loop_enabled")
    if closed_loop != cfg.closed_loop_enabled:
        raise G111VerdictControllerStateError("effect closed-loop gate differs from controller config")
    liveness = dict(_mapping(payload["liveness"], name="payload liveness"))
    _mapping(payload["scorer"], name="payload scorer")
    live_gap = dict(_mapping(payload["live_gap"], name="payload live_gap"))
    verdict = _mapping(payload["verdict"], name="payload verdict")
    d_seg = _number(verdict.get("d_seg"), name="verdict d_seg", minimum=0.0)
    d_pose_raw = verdict.get("d_pose")
    if d_pose_raw is None:
        if cfg.require_live_d_pose:
            raise G111VerdictControllerStateError("active G111 controller requires a live d_pose verdict")
        d_pose = None
        implied_s = None
    else:
        d_pose = _number(d_pose_raw, name="verdict d_pose", minimum=0.0)
        implied_s = 100.0 * d_seg + math.sqrt(10.0 * d_pose) + 25.0 * blob_bytes / float(UNCOMPRESSED_SIZE_BYTES)

    needs_counts = cfg.nucleus_guard_enabled or cfg.lane_sensor_enabled or cfg.birth_completion_enabled
    raw_counts = verdict.get("nucleus_counts")
    if raw_counts is None:
        if needs_counts:
            raise G111VerdictControllerStateError("active nucleus/birth controller requires verdict.nucleus_counts")
        counts = None
        stats = None
    else:
        counts, stats = _validate_counts(raw_counts, config=cfg)

    annulus, annulus_flip_frac = _annulus_value(
        verdict.get("annulus"),
        required=cfg.annulus_sensor_enabled,
    )
    per_class = verdict.get("per_class")
    if per_class is not None:
        per_class = _validate_per_class(per_class, config=cfg)
    elif cfg.ladder_enabled:
        raise G111VerdictControllerStateError(
            "active LADDER controller requires verdict.per_class"
        )
    pose_gate = verdict.get("_pose_gate_telemetry")
    if pose_gate is not None:
        pose_gate = _clone(_mapping(pose_gate, name="verdict._pose_gate_telemetry"))

    observation = {
        "schema": OBSERVATION_SCHEMA,
        "result_sequence": sequence,
        "result_id": result_id,
        "result_sha256": result_sha256,
        "epoch": epoch,
        "seg_form": seg_form,
        "ep_loss": ep_loss,
        "blob_bytes": blob_bytes,
        "d_seg": d_seg,
        "d_pose": d_pose,
        "implied_S": implied_s,
        "closed_loop_enabled": closed_loop,
        "liveness": _clone(liveness),
        "live_gap": _clone(live_gap),
        "nucleus_counts": counts,
        "nucleus_stats": stats,
        "annulus": annulus,
        "annulus_flip_frac": annulus_flip_frac,
        "per_class": per_class,
        "pose_gate": pose_gate,
    }
    _validate_observation(observation, config=cfg)
    return observation


def _validate_observation(
    observation: Mapping[str, Any],
    *,
    config: G111VerdictControllerConfigV1,
) -> None:
    value = _mapping(observation, name="controller observation")
    _keys(value, _OBSERVATION_KEYS, name="controller observation")
    if value["schema"] != OBSERVATION_SCHEMA:
        raise G111VerdictControllerStateError("controller observation schema differs")
    _integer(value["result_sequence"], name="observation result_sequence")
    _string(value["result_id"], name="observation result_id")
    _sha256(value["result_sha256"], name="observation result_sha256")
    _integer(value["epoch"], name="observation epoch")
    _string(value["seg_form"], name="observation seg_form")
    _number(value["ep_loss"], name="observation ep_loss")
    _integer(value["blob_bytes"], name="observation blob_bytes")
    _number(value["d_seg"], name="observation d_seg", minimum=0.0)
    if value["d_pose"] is None:
        if config.require_live_d_pose:
            raise G111VerdictControllerStateError("controller observation lacks required d_pose")
        if value["implied_S"] is not None:
            raise G111VerdictControllerStateError("observation implied_S requires d_pose")
    else:
        _number(value["d_pose"], name="observation d_pose", minimum=0.0)
        _number(value["implied_S"], name="observation implied_S", minimum=0.0)
    if (
        _boolean(
            value["closed_loop_enabled"],
            name="observation closed_loop_enabled",
        )
        != config.closed_loop_enabled
    ):
        raise G111VerdictControllerStateError("observation closed-loop gate differs from config")
    _mapping(value["liveness"], name="observation liveness")
    _mapping(value["live_gap"], name="observation live_gap")
    if value["nucleus_counts"] is not None:
        normalized, stats = _validate_counts(value["nucleus_counts"], config=config)
        if normalized != value["nucleus_counts"] or stats != value["nucleus_stats"]:
            raise G111VerdictControllerStateError("observation nucleus stats differ from exact count arithmetic")
    elif config.nucleus_guard_enabled or config.lane_sensor_enabled or config.birth_completion_enabled:
        raise G111VerdictControllerStateError("active nucleus/birth controller lacks counts")
    if value["annulus"] is not None:
        annulus, fraction = _annulus_value(value["annulus"], required=config.annulus_sensor_enabled)
        if annulus != value["annulus"] or fraction != value["annulus_flip_frac"]:
            raise G111VerdictControllerStateError("observation annulus value differs from metrics")
    elif config.annulus_sensor_enabled:
        raise G111VerdictControllerStateError("active annulus controller lacks metrics")
    if value["per_class"] is not None:
        normalized_per_class = _validate_per_class(
            value["per_class"],
            config=config,
        )
        if normalized_per_class != value["per_class"]:
            raise G111VerdictControllerStateError(
                "observation per_class differs from normalized vectors"
            )
    elif config.ladder_enabled:
        raise G111VerdictControllerStateError(
            "active LADDER controller lacks per_class vectors"
        )
    if value["pose_gate"] is not None:
        _mapping(value["pose_gate"], name="observation pose_gate")


def _validate_per_class(
    per_class: object,
    *,
    config: G111VerdictControllerConfigV1,
) -> dict[str, list[float]]:
    value = _mapping(per_class, name="per_class")
    expected = {"d_seg_by_class", "flip_share_by_class"}
    if set(value) != expected:
        raise G111VerdictControllerStateError("per_class fields differ")
    normalized: dict[str, list[float]] = {}
    for field in ("d_seg_by_class", "flip_share_by_class"):
        raw = value[field]
        if not isinstance(raw, (list, tuple)) or len(raw) != config.n_classes:
            raise G111VerdictControllerStateError(
                f"per_class.{field} must contain exactly {config.n_classes} values"
            )
        normalized[field] = [
            _number(
                item,
                name=f"per_class.{field}[{index}]",
                minimum=0.0,
                maximum=1.0,
            )
            for index, item in enumerate(raw)
        ]
    share_sum = sum(normalized["flip_share_by_class"])
    if not (math.isclose(share_sum, 0.0, abs_tol=1e-6) or math.isclose(share_sum, 1.0, abs_tol=5e-6)):
        raise G111VerdictControllerStateError(
            "per_class.flip_share_by_class must sum to zero or one"
        )
    return normalized


def _nucleus_decision(
    stats: Mapping[str, Any],
    config: G111VerdictControllerConfigV1,
) -> tuple[dict[str, bool], bool]:
    satisfied: dict[str, bool] = {}
    for cls in range(config.n_classes):
        row = _mapping(stats[str(cls)], name=f"nucleus_stats[{cls}]")
        if int(row["gt_px"]) == 0:
            satisfied[str(cls)] = True
        else:
            satisfied[str(cls)] = bool(
                float(row["part_frac"]) > config.nucleus_min_part_frac
                and float(row["within_flip"]) <= config.nucleus_within_flip_thresh
            )
    return satisfied, all(satisfied.values())


def _intent(
    *,
    channel: str,
    ordinal: int,
    observation: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    sequence = int(observation["result_sequence"])
    result_id = str(observation["result_id"])
    result_sha256 = str(observation["result_sha256"])
    normalized_payload = _clone(payload)
    digest = ImmutableVerdictResult.capture(
        submission_seq=sequence,
        result_id=f"{result_id}:{channel}:{ordinal}",
        payload={"payload": normalized_payload},
    ).result_sha256
    identity = f"{result_id}:{result_sha256}:{channel}:{ordinal}:{digest}"
    return {
        "schema": INTENT_SCHEMA,
        "channel": channel,
        "ordinal": ordinal,
        "result_sequence": sequence,
        "result_id": result_id,
        "result_sha256": result_sha256,
        "intent_sha256": digest,
        "idempotency_key": hashlib.sha256(identity.encode("utf-8")).hexdigest(),
        "payload": normalized_payload,
    }


@dataclass(frozen=True, slots=True)
class ControllerReductionV1:
    """One pure controller transition or exact retained-journal replay."""

    state: dict[str, Any]
    intents: tuple[dict[str, Any], ...]
    replayed: bool


def reduce_controller_state(
    state: Mapping[str, Any],
    observation: Mapping[str, Any],
) -> ControllerReductionV1:
    """Purely reduce one ordered observation into bounded controller state."""

    candidate = _clone(state)
    validate_controller_state(candidate)
    config = G111VerdictControllerConfigV1.from_mapping(candidate["config"])
    _validate_observation(observation, config=config)
    obs = _clone(observation)
    sequence = int(obs["result_sequence"])
    next_sequence = int(candidate["next_sequence"])
    base_sequence = int(candidate["journal_base_sequence"])

    if sequence < next_sequence:
        if sequence < base_sequence:
            raise G111VerdictControllerStateError("cannot replay an evicted controller result identity")
        journal_row = candidate["journal"][sequence - base_sequence]
        if journal_row["result_id"] != obs["result_id"] or journal_row["result_sha256"] != obs["result_sha256"]:
            raise G111VerdictControllerStateError("replayed sequence has a different result identity")
        return ControllerReductionV1(
            state=candidate,
            intents=tuple(_clone(journal_row["intents"])),
            replayed=True,
        )
    if sequence != next_sequence:
        raise G111VerdictControllerStateError(f"controller sequence {sequence} != expected {next_sequence}")
    if candidate["history"] and int(obs["epoch"]) <= int(candidate["history"][-1]["epoch"]):
        raise G111VerdictControllerStateError("new controller observation epoch must increase strictly")

    history_row: dict[str, Any] = {
        "epoch": int(obs["epoch"]),
        "d_seg": float(obs["d_seg"]),
    }
    if obs["d_pose"] is not None:
        history_row["d_pose"] = float(obs["d_pose"])
        history_row["implied_S"] = float(obs["implied_S"])
    history_row["blob_bytes"] = int(obs["blob_bytes"])
    _bounded_append(candidate["history"], history_row, config.history_limit)

    if config.closed_loop_enabled:
        _bounded_append(
            candidate["closed_loop_verdicts"],
            {
                "epoch": int(obs["epoch"]),
                "seg_form": str(obs["seg_form"]),
                "d_seg": float(obs["d_seg"]),
                "ep_loss": float(obs["ep_loss"]),
            },
            config.closed_loop_limit,
        )

    stats = obs["nucleus_stats"]
    satisfied: dict[str, bool] | None = None
    all_ok: bool | None = None
    lane_sensor: dict[str, Any] | None = None
    birth_rows: list[dict[str, Any]] = []
    if stats is not None:
        satisfied, all_ok = _nucleus_decision(stats, config)
        if config.nucleus_guard_enabled:
            candidate["nucleus_ready"] = bool(all_ok)
        if config.lane_sensor_enabled:
            lane_stats = stats[str(config.lane_class)]
            lane_event = lane_nucleus_event(
                float(lane_stats["part_frac"]),
                float(lane_stats["within_flip"]),
                within_flip_thresh=config.nucleus_within_flip_thresh,
                min_part_frac=config.nucleus_min_part_frac,
            )
            lane_sensor = {
                "epoch": int(obs["epoch"]),
                "event": lane_event,
            }
            candidate["lane_sensor"] = lane_sensor
        if config.birth_completion_enabled:
            birth = candidate["birth_completion"]
            birth_observation = {
                "epoch": int(obs["epoch"]),
                "stats": stats,
            }
            _bounded_append(
                birth["observations"],
                birth_observation,
                config.birth_observation_limit,
            )
            for cls in config.birth_classes:
                key = str(cls)
                if key in birth["fired_epochs"]:
                    continue
                row = stats[key]
                if birth_complete(
                    float(row["part_frac"]),
                    float(row["gt_area"]),
                    float(row["within_flip"]),
                    tau_persist=config.birth_tau_persist,
                    area_band=config.birth_area_band,
                ):
                    birth["fired_epochs"][key] = int(obs["epoch"])
                    birth_rows.append(
                        {
                            "stage": "birth_completion",
                            "event": "fired",
                            "class": cls,
                            "epoch": int(obs["epoch"]),
                            "part_frac": round(float(row["part_frac"]), 6),
                            "gt_area": round(float(row["gt_area"]), 6),
                            "persistence": round(
                                birth_persistence(float(row["within_flip"])),
                                4,
                            ),
                            "note": ("Morse-Smale birth-complete: hand off birth->boundary regime"),
                        }
                    )

    if config.annulus_sensor_enabled and obs["annulus_flip_frac"] is not None:
        _bounded_append(
            candidate["annulus_series"],
            {
                "epoch": int(obs["epoch"]),
                "annulus_flip_frac": float(obs["annulus_flip_frac"]),
            },
            config.annulus_limit,
        )
    if config.label_floor_sensor_enabled:
        _bounded_append(
            candidate["label_floor_series"],
            {
                "epoch": int(obs["epoch"]),
                "d_seg": float(obs["d_seg"]),
                "seg_form": str(obs["seg_form"]),
            },
            config.label_floor_limit,
        )
    if config.w_pose_law_enabled and obs["d_pose"] is not None and float(obs["d_pose"]) > 0.0:
        candidate["last_d_pose"] = float(obs["d_pose"])
    if config.ladder_enabled:
        per_class = obs["per_class"]
        if per_class is None:  # guarded by observation validation; retain fail-closed locality
            raise G111VerdictControllerStateError(
                "active LADDER controller lacks per_class vectors"
            )
        rates = per_class["d_seg_by_class"]
        shares = per_class["flip_share_by_class"]
        candidate["ladder_costates"] = {
            "epoch": int(obs["epoch"]),
            ARM_LANE: perclass_lambda_proxy(
                rates[config.lane_class],
                shares[config.lane_class],
            ),
            ARM_MOVABLE: perclass_lambda_proxy(
                rates[config.movable_class],
                shares[config.movable_class],
            ),
        }

    verdict_row = {
        "stage": "verdict",
        "epoch": int(obs["epoch"]),
        "seg_form": str(obs["seg_form"]),
        "d_seg": round(float(obs["d_seg"]), 6),
        "d_pose": (None if obs["d_pose"] is None else round(float(obs["d_pose"]), 6)),
        "blob_bytes": int(obs["blob_bytes"]),
        "implied_S": (None if obs["implied_S"] is None else round(float(obs["implied_S"]), 4)),
        "ep_loss": round(float(obs["ep_loss"]), 3),
        "pose_gate": obs["pose_gate"],
    }
    stdout_rows: list[dict[str, Any]] = [verdict_row]
    stdout_rows.extend(birth_rows)
    telemetry_payload = {
        "stage": "g111_verdict_controller_telemetry",
        "epoch": int(obs["epoch"]),
        "nucleus_counts": obs["nucleus_counts"],
        "nucleus_stats": stats,
        "nucleus_satisfied": satisfied,
        "nucleus_ready": all_ok,
        "lane_sensor": lane_sensor,
        "annulus": obs["annulus"],
        "annulus_flip_frac": obs["annulus_flip_frac"],
        "per_class": obs["per_class"],
        "pose_gate": obs["pose_gate"],
        "liveness": obs["liveness"],
        "live_gap": obs["live_gap"],
        "birth_fires": birth_rows,
    }
    causal_payload = {
        "stage": "g111_verdict_controller_boundary",
        "event": "verdict_reduced",
        "epoch": int(obs["epoch"]),
        "history_length": len(candidate["history"]),
        "closed_loop_length": len(candidate["closed_loop_verdicts"]),
        "nucleus_ready": bool(candidate["nucleus_ready"]),
        "lane_sensor": candidate["lane_sensor"],
        "annulus_series_length": len(candidate["annulus_series"]),
        "label_floor_series_length": len(candidate["label_floor_series"]),
        "last_d_pose": candidate["last_d_pose"],
        "ladder_costates": candidate["ladder_costates"],
        "birth_fired_epochs": candidate["birth_completion"]["fired_epochs"],
    }
    intents = (
        _intent(
            channel="stdout",
            ordinal=0,
            observation=obs,
            payload={"rows": stdout_rows},
        ),
        _intent(
            channel="telemetry",
            ordinal=1,
            observation=obs,
            payload=telemetry_payload,
        ),
        _intent(
            channel="causal",
            ordinal=2,
            observation=obs,
            payload=causal_payload,
        ),
    )

    candidate["journal"].append(
        {
            "sequence": sequence,
            "result_id": obs["result_id"],
            "result_sha256": obs["result_sha256"],
            "intents": list(intents),
        }
    )
    candidate["next_sequence"] = sequence + 1
    if len(candidate["journal"]) > config.journal_limit:
        drop = len(candidate["journal"]) - config.journal_limit
        del candidate["journal"][:drop]
        candidate["journal_base_sequence"] += drop
    validate_controller_state(candidate)
    return ControllerReductionV1(
        state=candidate,
        intents=tuple(_clone(intents)),
        replayed=False,
    )


def _validate_history(
    rows: object,
    *,
    name: str,
    limit: int,
    closed_loop: bool,
) -> None:
    if not isinstance(rows, list) or len(rows) > limit:
        raise G111VerdictControllerStateError(f"{name} must be a list bounded by {limit}")
    last_epoch = -1
    for raw in rows:
        row = _mapping(raw, name=f"{name} row")
        expected = {"epoch", "seg_form", "d_seg", "ep_loss"} if closed_loop else {"epoch", "d_seg", "blob_bytes"}
        if not closed_loop and "d_pose" in row:
            expected |= {"d_pose", "implied_S"}
        if set(row) != expected:
            raise G111VerdictControllerStateError(f"{name} row fields differ")
        epoch = _integer(row["epoch"], name=f"{name} epoch")
        if epoch <= last_epoch:
            raise G111VerdictControllerStateError(f"{name} epochs must increase strictly")
        last_epoch = epoch
        _number(row["d_seg"], name=f"{name} d_seg", minimum=0.0)
        if closed_loop:
            _string(row["seg_form"], name=f"{name} seg_form")
            _number(row["ep_loss"], name=f"{name} ep_loss")
        else:
            _integer(row["blob_bytes"], name=f"{name} blob_bytes")
            if "d_pose" in row:
                _number(row["d_pose"], name=f"{name} d_pose", minimum=0.0)
                _number(row["implied_S"], name=f"{name} implied_S", minimum=0.0)


def _validate_intent(intent: Mapping[str, Any]) -> None:
    value = _mapping(intent, name="controller intent")
    _keys(value, _INTENT_KEYS, name="controller intent")
    if value["schema"] != INTENT_SCHEMA:
        raise G111VerdictControllerStateError("controller intent schema differs")
    channel = _string(value["channel"], name="intent channel")
    if channel not in {"stdout", "telemetry", "causal"}:
        raise G111VerdictControllerStateError("controller intent channel differs")
    ordinal = _integer(value["ordinal"], name="intent ordinal")
    sequence = _integer(value["result_sequence"], name="intent result_sequence")
    result_id = _string(value["result_id"], name="intent result_id")
    result_sha256 = _sha256(value["result_sha256"], name="intent result_sha256")
    intent_sha256 = _sha256(value["intent_sha256"], name="intent intent_sha256")
    payload = _mapping(value["payload"], name="intent payload")
    expected_digest = ImmutableVerdictResult.capture(
        submission_seq=sequence,
        result_id=f"{result_id}:{channel}:{ordinal}",
        payload={"payload": payload},
    ).result_sha256
    if intent_sha256 != expected_digest:
        raise G111VerdictControllerStateError("intent SHA-256 differs from canonical payload")
    identity = f"{result_id}:{result_sha256}:{channel}:{ordinal}:{intent_sha256}"
    if value["idempotency_key"] != hashlib.sha256(identity.encode("utf-8")).hexdigest():
        raise G111VerdictControllerStateError("intent idempotency key differs from result identity")


def validate_controller_state(state: Mapping[str, Any]) -> None:
    """Validate the complete bounded replacement controller state."""

    value = _mapping(state, name="controller state")
    _keys(value, _STATE_KEYS, name="controller state")
    if value["schema"] != SCHEMA:
        raise G111VerdictControllerStateError("controller state schema differs")
    config = G111VerdictControllerConfigV1.from_mapping(value["config"])
    base = _integer(value["journal_base_sequence"], name="journal_base_sequence")
    next_sequence = _integer(value["next_sequence"], name="next_sequence")
    journal = value["journal"]
    if not isinstance(journal, list) or len(journal) > config.journal_limit or len(journal) != next_sequence - base:
        raise G111VerdictControllerStateError("controller journal differs from bounded base/next interval")
    for offset, raw in enumerate(journal):
        row = _mapping(raw, name="controller journal row")
        _keys(row, _JOURNAL_KEYS, name="controller journal row")
        sequence = _integer(row["sequence"], name="journal sequence")
        if sequence != base + offset:
            raise G111VerdictControllerStateError("controller journal is not contiguous")
        result_id = _string(row["result_id"], name="journal result_id")
        result_sha256 = _sha256(row["result_sha256"], name="journal result_sha256")
        intents = row["intents"]
        if not isinstance(intents, list) or len(intents) != 3:
            raise G111VerdictControllerStateError("journal row must retain exactly three intents")
        for ordinal, intent in enumerate(intents):
            _validate_intent(intent)
            if (
                int(intent["result_sequence"]) != sequence
                or intent["result_id"] != result_id
                or intent["result_sha256"] != result_sha256
                or int(intent["ordinal"]) != ordinal
            ):
                raise G111VerdictControllerStateError("journal intent identity differs from its result")
    _validate_history(
        value["history"],
        name="history",
        limit=config.history_limit,
        closed_loop=False,
    )
    _validate_history(
        value["closed_loop_verdicts"],
        name="closed_loop_verdicts",
        limit=config.closed_loop_limit,
        closed_loop=True,
    )
    _boolean(value["nucleus_ready"], name="nucleus_ready")
    lane_sensor = value["lane_sensor"]
    if lane_sensor is not None:
        lane = _mapping(lane_sensor, name="lane_sensor")
        if set(lane) != {"epoch", "event"}:
            raise G111VerdictControllerStateError("lane_sensor fields differ")
        _integer(lane["epoch"], name="lane_sensor epoch")
        event = _mapping(lane["event"], name="lane_sensor event")
        for field in (
            "fired",
            "born",
            "formed",
            "part_frac",
            "within_flip",
            "within_flip_thresh",
            "min_part_frac",
        ):
            if field not in event:
                raise G111VerdictControllerStateError(f"lane_sensor event lacks {field}")
    annulus_series = value["annulus_series"]
    if not isinstance(annulus_series, list) or len(annulus_series) > config.annulus_limit:
        raise G111VerdictControllerStateError("annulus_series exceeds its bound")
    last_epoch = -1
    for raw in annulus_series:
        row = _mapping(raw, name="annulus_series row")
        if set(row) != {"epoch", "annulus_flip_frac"}:
            raise G111VerdictControllerStateError("annulus_series row fields differ")
        epoch = _integer(row["epoch"], name="annulus_series epoch")
        if epoch <= last_epoch:
            raise G111VerdictControllerStateError("annulus_series epochs must increase strictly")
        last_epoch = epoch
        _number(
            row["annulus_flip_frac"],
            name="annulus_series fraction",
            minimum=0.0,
            maximum=1.0,
        )
    label_floor = value["label_floor_series"]
    if not isinstance(label_floor, list) or len(label_floor) > config.label_floor_limit:
        raise G111VerdictControllerStateError("label_floor_series exceeds its bound")
    last_epoch = -1
    for raw in label_floor:
        row = _mapping(raw, name="label_floor_series row")
        if set(row) != {"epoch", "d_seg", "seg_form"}:
            raise G111VerdictControllerStateError("label_floor_series row fields differ")
        epoch = _integer(row["epoch"], name="label_floor_series epoch")
        if epoch <= last_epoch:
            raise G111VerdictControllerStateError("label_floor_series epochs must increase strictly")
        last_epoch = epoch
        _number(row["d_seg"], name="label_floor_series d_seg", minimum=0.0)
        _string(row["seg_form"], name="label_floor_series seg_form")
    if value["last_d_pose"] is not None:
        _number(value["last_d_pose"], name="last_d_pose", minimum=0.0)
    ladder_costates = value["ladder_costates"]
    if ladder_costates is not None:
        ladder = _mapping(ladder_costates, name="ladder_costates")
        if set(ladder) != {"epoch", ARM_LANE, ARM_MOVABLE}:
            raise G111VerdictControllerStateError("ladder_costates fields differ")
        _integer(ladder["epoch"], name="ladder_costates epoch")
        _number(ladder[ARM_LANE], name="ladder lane costate", minimum=0.0)
        _number(ladder[ARM_MOVABLE], name="ladder movable costate", minimum=0.0)
    elif config.ladder_enabled and next_sequence > 0:
        raise G111VerdictControllerStateError(
            "active LADDER controller lacks reduced costates"
        )
    birth = _mapping(value["birth_completion"], name="birth_completion")
    _keys(birth, _BIRTH_KEYS, name="birth_completion")
    if birth["classes"] != list(config.birth_classes):
        raise G111VerdictControllerStateError("birth_completion classes differ from config")
    fired = _mapping(birth["fired_epochs"], name="birth fired_epochs")
    allowed = {str(cls) for cls in config.birth_classes}
    if not set(fired) <= allowed:
        raise G111VerdictControllerStateError("birth fired_epochs contains an unwatched class")
    for epoch in fired.values():
        _integer(epoch, name="birth fired epoch")
    observations = birth["observations"]
    if not isinstance(observations, list) or len(observations) > config.birth_observation_limit:
        raise G111VerdictControllerStateError("birth observations exceed their bound")
    for raw in observations:
        observation = _mapping(raw, name="birth observation")
        if set(observation) != {"epoch", "stats"}:
            raise G111VerdictControllerStateError("birth observation fields differ")
        _integer(observation["epoch"], name="birth observation epoch")
        stats = _mapping(observation["stats"], name="birth observation stats")
        if set(stats) != {str(cls) for cls in range(config.n_classes)}:
            raise G111VerdictControllerStateError("birth observation lacks exact per-class stats")


def state_arrays(
    state: Mapping[str, Any],
    *,
    prefix: str,
) -> Mapping[str, np.ndarray]:
    """Serialize replacement state as strict non-pickle arrays."""

    if type(prefix) is not str or prefix.strip() != prefix:
        raise G111VerdictControllerStateError("state prefix must be a canonical string")
    validate_controller_state(state)
    encoded = ImmutableVerdictResult.capture(
        submission_seq=0,
        result_id=STATE_RESULT_ID,
        payload={"state": state},
    )
    arrays = {
        f"{prefix}schema": _utf8_array(SERIALIZED_STATE_SCHEMA),
        f"{prefix}state_payload": np.frombuffer(encoded.payload_bytes, dtype=np.uint8).copy(),
        f"{prefix}state_sha256": _utf8_array(encoded.result_sha256),
    }
    return MappingProxyType(arrays)


def state_from_arrays(
    arrays: Mapping[str, Any],
    *,
    prefix: str,
    expected_config: G111VerdictControllerConfigV1 | None = None,
) -> dict[str, Any]:
    """Restore and validate replacement state, optionally binding exact config."""

    if type(prefix) is not str or prefix.strip() != prefix:
        raise G111VerdictControllerStateError("state prefix must be a canonical string")
    required = {
        f"{prefix}schema",
        f"{prefix}state_payload",
        f"{prefix}state_sha256",
    }
    missing = required - set(arrays)
    if missing:
        raise G111VerdictControllerStateError(f"serialized controller state lacks {sorted(missing)!r}")
    if _decode_utf8(arrays[f"{prefix}schema"], name="state schema") != SERIALIZED_STATE_SCHEMA:
        raise G111VerdictControllerStateError("serialized controller state schema differs")
    payload_array = np.asarray(arrays[f"{prefix}state_payload"])
    if payload_array.dtype != np.dtype(np.uint8) or payload_array.ndim != 1:
        raise G111VerdictControllerStateError("state payload must be a one-dimensional uint8 array")
    encoded = ImmutableVerdictResult(
        submission_seq=0,
        result_id=STATE_RESULT_ID,
        payload_bytes=payload_array.tobytes(),
        result_sha256=_decode_utf8(arrays[f"{prefix}state_sha256"], name="state SHA-256"),
    )
    encoded.validate()
    state = encoded.payload.get("state")
    if not isinstance(state, dict):
        raise G111VerdictControllerStateError("serialized controller state must decode to a mapping")
    validate_controller_state(state)
    if expected_config is not None:
        expected_config.validate()
        if state["config"] != expected_config.to_mapping():
            raise G111VerdictControllerStateError("restored controller config differs from expected config")
    return state


serialize_controller_state = state_arrays
restore_controller_state = state_from_arrays


__all__ = [
    "OBSERVATION_SCHEMA",
    "SCHEMA",
    "SERIALIZED_STATE_SCHEMA",
    "ControllerReductionV1",
    "G111VerdictControllerConfigV1",
    "G111VerdictControllerStateError",
    "active_g111_controller_config_v1",
    "adapt_live_reducer_effect",
    "new_controller_state",
    "reduce_controller_state",
    "restore_controller_state",
    "serialize_controller_state",
    "state_arrays",
    "state_from_arrays",
    "validate_controller_state",
]
