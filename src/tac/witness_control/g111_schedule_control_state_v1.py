# SPDX-License-Identifier: MIT
"""Fixed-capacity O3 schedule/control trajectory state for G111 native-v3.

The legacy trainer resume sidecar mixes model tensors, optimizer tensors,
lineage, and schedule/controller state in one flat namespace.  Native-v3 must
assign each physical leaf to exactly one atomic owner.  This module provides
the O3 boundary: the caller supplies only schedule/control arrays plus the
explicit next-update coordinate, and receives a fixed-shape non-pickle array
map suitable for a cold-derived checkpoint schema.

The payload is content addressed and canonically encoded through the same
NumPy-aware codec used by immutable verdict results.  Zero padding is checked
on reopen, so bytes beyond the declared payload cannot become a covert second
state channel.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Final

import numpy as np

from tac.witness_control.g111_verdict_barrier_v1 import ImmutableVerdictResult

SCHEMA: Final = "tac.g111_schedule_control_state.v1"
SERIALIZED_SCHEMA: Final = "tac.g111_schedule_control_state_arrays.v1"
STATE_RESULT_ID: Final = "g111-o3-schedule-control-state"
PAYLOAD_CAPACITY: Final = 8 * 1024 * 1024
_FORBIDDEN_ARRAY_PREFIXES: Final = (
    "liveP__",
    "emaP__",
    "optP__",
    "seedP__",
    "seedOptP__",
    "polyakM__",
    "__cfg_fresh_lineage_",
)
_PAYLOAD_FIELDS: Final = frozenset(
    {
        "schema",
        "typed_config_sha256",
        "coordinate",
        "control_scalars",
        "resume_control_arrays",
    }
)
_COORDINATE_FIELDS: Final = frozenset(
    {
        "completed_epoch",
        "next_epoch",
        "accepted_optimizer_steps",
        "stop_latched",
    }
)


class G111ScheduleControlStateError(RuntimeError):
    """O3 state is incomplete, malformed, or not canonically serialized."""


def _exact_nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise G111ScheduleControlStateError(f"{name} must be an exact integer")
    result = int(value)
    if result < 0:
        raise G111ScheduleControlStateError(f"{name} must be nonnegative")
    return result


def _exact_bool(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise G111ScheduleControlStateError(f"{name} must be an exact bool")
    return value


def _sha256(value: object, *, name: str) -> str:
    if type(value) is not str or len(value) != 64 or value != value.lower():
        raise G111ScheduleControlStateError(
            f"{name} must be 64 lowercase hexadecimal characters"
        )
    try:
        decoded = bytes.fromhex(value)
    except ValueError as exc:
        raise G111ScheduleControlStateError(
            f"{name} must be 64 lowercase hexadecimal characters"
        ) from exc
    if decoded.hex() != value:
        raise G111ScheduleControlStateError(
            f"{name} must be 64 lowercase hexadecimal characters"
        )
    return value


def _canonical_prefix(prefix: object) -> str:
    if type(prefix) is not str or not prefix or prefix.strip() != prefix:
        raise G111ScheduleControlStateError(
            "state prefix must be a non-empty canonical string"
        )
    return prefix


def _utf8_array(value: str) -> np.ndarray:
    return np.frombuffer(value.encode("utf-8"), dtype=np.uint8).copy()


def _decode_utf8(value: object, *, name: str) -> str:
    array = np.asarray(value)
    if array.dtype != np.dtype(np.uint8) or array.ndim != 1:
        raise G111ScheduleControlStateError(
            f"{name} must be a one-dimensional uint8 array"
        )
    try:
        return array.tobytes().decode("utf-8")
    except UnicodeError as exc:
        raise G111ScheduleControlStateError(
            f"{name} is not valid UTF-8"
        ) from exc


def _normalize_arrays(
    arrays: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    if not isinstance(arrays, Mapping):
        raise G111ScheduleControlStateError(
            "resume_control_arrays must be a mapping"
        )
    normalized: dict[str, np.ndarray] = {}
    for raw_key, raw_value in arrays.items():
        if type(raw_key) is not str or not raw_key or raw_key.strip() != raw_key:
            raise G111ScheduleControlStateError(
                "resume_control_arrays keys must be canonical strings"
            )
        if raw_key in normalized:
            raise G111ScheduleControlStateError(
                f"resume_control_arrays repeats {raw_key!r}"
            )
        if raw_key.startswith(_FORBIDDEN_ARRAY_PREFIXES):
            raise G111ScheduleControlStateError(
                f"O3 cannot own non-schedule array {raw_key!r}"
            )
        array = np.asarray(raw_value)
        if array.dtype.hasobject or array.dtype.fields is not None:
            raise G111ScheduleControlStateError(
                f"{raw_key}: object and structured arrays are forbidden"
            )
        if array.dtype.kind in "fc" and not bool(np.all(np.isfinite(array))):
            raise G111ScheduleControlStateError(
                f"{raw_key}: nonfinite schedule/control arrays are forbidden"
            )
        normalized[raw_key] = np.array(array, copy=True, order="C")
    return normalized


def _normalize_scalars(
    scalars: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(scalars, Mapping):
        raise G111ScheduleControlStateError("control_scalars must be a mapping")
    normalized: dict[str, Any] = {}
    for raw_key, value in scalars.items():
        if type(raw_key) is not str or not raw_key or raw_key.strip() != raw_key:
            raise G111ScheduleControlStateError(
                "control_scalars keys must be canonical strings"
            )
        if isinstance(value, np.generic):
            value = value.item()
        if value is not None and type(value) not in (bool, int, float, str):
            raise G111ScheduleControlStateError(
                f"control_scalars.{raw_key} has unsupported type "
                f"{type(value).__name__}"
            )
        if type(value) is float and not np.isfinite(value):
            raise G111ScheduleControlStateError(
                f"control_scalars.{raw_key} must be finite or None"
            )
        normalized[raw_key] = value
    return normalized


def new_state(
    *,
    typed_config_sha256: str,
    completed_epoch: int,
    next_epoch: int,
    accepted_optimizer_steps: int,
    stop_latched: bool,
    control_scalars: Mapping[str, Any],
    resume_control_arrays: Mapping[str, Any],
) -> dict[str, Any]:
    """Capture a detached, explicit O3 replacement state."""

    completed = _exact_nonnegative_int(
        completed_epoch,
        name="completed_epoch",
    )
    next_update = _exact_nonnegative_int(next_epoch, name="next_epoch")
    accepted = _exact_nonnegative_int(
        accepted_optimizer_steps,
        name="accepted_optimizer_steps",
    )
    stopped = _exact_bool(stop_latched, name="stop_latched")
    if stopped:
        if next_update != completed:
            raise G111ScheduleControlStateError(
                "stop-latched state must not advance next_epoch"
            )
    elif next_update != completed + 1:
        raise G111ScheduleControlStateError(
            "live state next_epoch must equal completed_epoch + 1"
        )
    state = {
        "schema": SCHEMA,
        "typed_config_sha256": _sha256(
            typed_config_sha256,
            name="typed_config_sha256",
        ),
        "coordinate": {
            "completed_epoch": completed,
            "next_epoch": next_update,
            "accepted_optimizer_steps": accepted,
            "stop_latched": stopped,
        },
        "control_scalars": _normalize_scalars(control_scalars),
        "resume_control_arrays": _normalize_arrays(resume_control_arrays),
    }
    validate_state(state)
    return state


def validate_state(state: Mapping[str, Any]) -> None:
    """Validate exact O3 schema and coordinate invariants."""

    if not isinstance(state, Mapping) or set(state) != _PAYLOAD_FIELDS:
        raise G111ScheduleControlStateError(
            "schedule/control state fields differ from the exact schema"
        )
    if state["schema"] != SCHEMA:
        raise G111ScheduleControlStateError(
            "schedule/control state schema differs"
        )
    _sha256(state["typed_config_sha256"], name="typed_config_sha256")
    coordinate = state["coordinate"]
    if not isinstance(coordinate, Mapping) or set(coordinate) != _COORDINATE_FIELDS:
        raise G111ScheduleControlStateError("coordinate fields differ")
    completed = _exact_nonnegative_int(
        coordinate["completed_epoch"],
        name="completed_epoch",
    )
    next_update = _exact_nonnegative_int(
        coordinate["next_epoch"],
        name="next_epoch",
    )
    _exact_nonnegative_int(
        coordinate["accepted_optimizer_steps"],
        name="accepted_optimizer_steps",
    )
    stopped = _exact_bool(coordinate["stop_latched"], name="stop_latched")
    if stopped and next_update != completed:
        raise G111ScheduleControlStateError(
            "stop-latched state must not advance next_epoch"
        )
    if not stopped and next_update != completed + 1:
        raise G111ScheduleControlStateError(
            "live state next_epoch must equal completed_epoch + 1"
        )
    _normalize_scalars(state["control_scalars"])
    _normalize_arrays(state["resume_control_arrays"])


def state_arrays(
    state: Mapping[str, Any],
    *,
    prefix: str,
) -> Mapping[str, np.ndarray]:
    """Serialize O3 into a fixed-shape non-pickle array map."""

    prefix = _canonical_prefix(prefix)
    validate_state(state)
    encoded = ImmutableVerdictResult.capture(
        submission_seq=0,
        result_id=STATE_RESULT_ID,
        payload={"state": state},
    )
    length = len(encoded.payload_bytes)
    if length > PAYLOAD_CAPACITY:
        raise G111ScheduleControlStateError(
            "serialized O3 state exceeds fixed capacity "
            f"{length} > {PAYLOAD_CAPACITY}"
        )
    payload = np.zeros(PAYLOAD_CAPACITY, dtype=np.uint8)
    payload[:length] = np.frombuffer(encoded.payload_bytes, dtype=np.uint8)
    return MappingProxyType(
        {
            f"{prefix}schema": _utf8_array(SERIALIZED_SCHEMA),
            f"{prefix}state_payload": payload,
            f"{prefix}state_payload_length": np.asarray(length, dtype=np.int64),
            f"{prefix}state_sha256": _utf8_array(encoded.result_sha256),
        }
    )


def state_from_arrays(
    arrays: Mapping[str, Any],
    *,
    prefix: str,
    expected_typed_config_sha256: str | None = None,
) -> dict[str, Any]:
    """Restore O3, checking padding, hash, schema, and optional config bind."""

    prefix = _canonical_prefix(prefix)
    required = {
        f"{prefix}schema",
        f"{prefix}state_payload",
        f"{prefix}state_payload_length",
        f"{prefix}state_sha256",
    }
    if not isinstance(arrays, Mapping) or not required.issubset(arrays):
        missing = required - set(arrays if isinstance(arrays, Mapping) else ())
        raise G111ScheduleControlStateError(
            f"serialized O3 state lacks {sorted(missing)!r}"
        )
    if _decode_utf8(arrays[f"{prefix}schema"], name="state schema") != SERIALIZED_SCHEMA:
        raise G111ScheduleControlStateError("serialized O3 schema differs")
    payload = np.asarray(arrays[f"{prefix}state_payload"])
    if payload.dtype != np.dtype(np.uint8) or payload.shape != (PAYLOAD_CAPACITY,):
        raise G111ScheduleControlStateError(
            "O3 payload must be the fixed-capacity uint8 slab"
        )
    raw_length = np.asarray(arrays[f"{prefix}state_payload_length"])
    if raw_length.dtype != np.dtype(np.int64) or raw_length.shape != ():
        raise G111ScheduleControlStateError(
            "O3 payload length must be an int64 scalar"
        )
    length = int(raw_length.item())
    if not 0 < length <= PAYLOAD_CAPACITY:
        raise G111ScheduleControlStateError(
            "O3 payload length lies outside fixed capacity"
        )
    if np.any(payload[length:]):
        raise G111ScheduleControlStateError(
            "O3 payload has nonzero bytes after its declared length"
        )
    encoded = ImmutableVerdictResult(
        submission_seq=0,
        result_id=STATE_RESULT_ID,
        payload_bytes=payload[:length].tobytes(),
        result_sha256=_decode_utf8(
            arrays[f"{prefix}state_sha256"],
            name="state SHA-256",
        ),
    )
    encoded.validate()
    state = encoded.payload.get("state")
    if not isinstance(state, dict):
        raise G111ScheduleControlStateError(
            "serialized O3 payload must decode to a state mapping"
        )
    validate_state(state)
    if expected_typed_config_sha256 is not None:
        expected = _sha256(
            expected_typed_config_sha256,
            name="expected_typed_config_sha256",
        )
        if state["typed_config_sha256"] != expected:
            raise G111ScheduleControlStateError(
                "restored O3 typed config differs from the active config"
            )
    return state


__all__ = [
    "PAYLOAD_CAPACITY",
    "SCHEMA",
    "SERIALIZED_SCHEMA",
    "G111ScheduleControlStateError",
    "new_state",
    "state_arrays",
    "state_from_arrays",
    "validate_state",
]
