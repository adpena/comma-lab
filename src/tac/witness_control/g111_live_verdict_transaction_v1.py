# SPDX-License-Identifier: MIT
"""Fresh-G111 live verdict reduction and main-thread effect publication.

This module is the trainer-facing layer above :mod:`g111_verdict_barrier_v1`.
Workers return scorer results only.  The reducer is a pure state transition that
records bounded O4 effect intents and O5 BEST intents.  O4 histories and sensors
belong exclusively to the replacement controller state; this transaction never
keeps a second copy.  External telemetry, controller mutation, and BEST file
publication happen later, on the creating thread, through
:class:`MainThreadVerdictEffectPublisher`.

The non-pickle checkpoint surface has a cold-stable array census and shapes.
O4 and O5 have disjoint key namespaces, explicit counts/base cursors, fixed-width
identities, and fixed-capacity canonical payload slabs.  A serialized checkpoint
retains only unacknowledged full effects plus the latest O5 artifact, so its
logical content is bounded by the verdict cadence rather than total run length.
Publication callbacks are allowed to be replayed after a crash before cursor
durability; they MUST therefore publish deterministic content under the supplied
result identity.  This module does not write files and is not launch authority.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

import numpy as np

from tac.witness_control.g111_verdict_barrier_v1 import (
    ImmutableVerdictResult,
    ResultIntegrityError,
)

SCHEMA: Final = "tac.g111_live_verdict_transaction.v1"
SNAPSHOT_SCHEMA: Final = "tac.g111_live_verdict_snapshot.v1"
WORKER_PAYLOAD_SCHEMA: Final = "tac.g111_live_verdict_worker_payload.v1"
SERIALIZED_STATE_SCHEMA: Final = "tac.g111_live_verdict_state_arrays.v2"
SERIALIZED_O4_SCHEMA: Final = f"{SERIALIZED_STATE_SCHEMA}.o4"
SERIALIZED_O5_SCHEMA: Final = f"{SERIALIZED_STATE_SCHEMA}.o5"
STATE_RESULT_ID: Final = "g111-live-verdict-state"

# Match the production native-v3 barrier's fixed 64-row identity journal.  The
# live scheduler normally permits only one in-flight verdict, but the checkpoint
# codec must not depend on that operational shortcut.  After compaction O5 can
# retain one acknowledged BEST tail in addition to one BEST per still-
# unacknowledged O4 row.  The independent byte slabs remain the hard memory
# bound and fail closed if unusually large pending snapshots do not fit.
SERIALIZED_MAX_EFFECT_ROWS: Final = 64
SERIALIZED_MAX_BEST_INTENT_ROWS: Final = SERIALIZED_MAX_EFFECT_ROWS + 1
SERIALIZED_RESULT_ID_CAPACITY: Final = 128
SERIALIZED_SHA256_BYTES: Final = 32
SERIALIZED_O4_PAYLOAD_CAPACITY: Final = 16 * 1024 * 1024
SERIALIZED_O5_ARTIFACT_CAPACITY: Final = 8 * 1024 * 1024

SERIALIZED_O4_FIELDS: Final[tuple[str, ...]] = (
    "schema",
    "effect_count",
    "effect_base_sequence",
    "next_effect_sequence",
    "effect_cursor",
    "effect_result_id_data",
    "effect_result_id_lengths",
    "effect_result_sha256",
    "effect_has_best_intent",
    "effect_best_intent_sequences",
    "effect_payload_offsets",
    "effect_payload_data",
)
SERIALIZED_O5_FIELDS: Final[tuple[str, ...]] = (
    "schema",
    "best_present",
    "best_d_seg",
    "best_epoch",
    "best_result_id_data",
    "best_result_id_length",
    "best_intent_sequence",
    "best_intent_count",
    "best_intent_base_sequence",
    "next_best_intent_sequence",
    "best_intent_cursor",
    "intent_effect_sequences",
    "intent_result_id_data",
    "intent_result_id_lengths",
    "intent_result_sha256",
    "intent_d_seg",
    "intent_epochs",
    "artifact_payload_offsets",
    "artifact_payload_data",
    "artifact_sha256",
)

_STATE_KEYS: Final = frozenset(
    {
        "schema",
        "effect_base_sequence",
        "next_effect_sequence",
        "effects",
        "history",
        "closed_loop_verdicts",
        "o5",
    }
)
_O5_KEYS: Final = frozenset(
    {
        "best_present",
        "best_d_seg",
        "best_epoch",
        "best_result_id",
        "best_intent_sequence",
        "best_intent_base_sequence",
        "next_best_intent_sequence",
        "best_intents",
    }
)
_BEST_INTENT_KEYS: Final = frozenset(
    {
        "intent_sequence",
        "effect_sequence",
        "result_id",
        "result_sha256",
        "d_seg",
        "epoch",
        "artifact",
    }
)
_BEST_ARTIFACT_KEYS: Final = frozenset(
    {
        "ema_np",
        "softmax_temp",
        "hosc_beta",
        "ema_updates",
    }
)
_SNAPSHOT_KEYS: Final = frozenset(
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


class LiveVerdictStateError(RuntimeError):
    """The explicit O4/O5 live-verdict state is malformed."""


class LiveVerdictMainThreadError(RuntimeError):
    """An effect publication was attempted off the creating thread."""


class LiveVerdictEffectPublicationError(RuntimeError):
    """A main-thread effect callback failed and poisoned later publication."""


def _exact_int(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise LiveVerdictStateError(f"{name} must be an exact integer")
    result = int(value)
    if result < minimum:
        raise LiveVerdictStateError(f"{name} must be >= {minimum}")
    return result


def _finite_float(value: object, *, name: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise LiveVerdictStateError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise LiveVerdictStateError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise LiveVerdictStateError(f"{name} must be >= {minimum}")
    return result


def _exact_mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LiveVerdictStateError(f"{name} must be a mapping")
    for key in value:
        if type(key) is not str:
            raise LiveVerdictStateError(f"{name} keys must have exact str type")
    return value


def _require_keys(
    value: Mapping[str, Any],
    expected: frozenset[str],
    *,
    name: str,
) -> None:
    if set(value) != expected:
        raise LiveVerdictStateError(
            f"{name} fields {sorted(value)!r} != expected {sorted(expected)!r}"
        )


def _utf8_array(value: str) -> np.ndarray:
    return np.frombuffer(value.encode("utf-8"), dtype=np.uint8).copy()


def _decode_utf8(value: object, *, name: str) -> str:
    array = np.asarray(value)
    if array.dtype != np.dtype(np.uint8) or array.ndim != 1:
        raise LiveVerdictStateError(f"{name} must be a one-dimensional uint8 array")
    try:
        return array.tobytes().decode("utf-8")
    except UnicodeError as exc:
        raise LiveVerdictStateError(f"{name} is not valid UTF-8") from exc


def _int64_scalar(value: int) -> np.ndarray:
    return np.asarray(int(value), dtype=np.int64)


def _read_int64_scalar(
    value: object,
    *,
    name: str,
    minimum: int = 0,
) -> int:
    array = np.asarray(value)
    if array.dtype != np.dtype(np.int64) or array.shape != ():
        raise LiveVerdictStateError(f"{name} must be an int64 scalar")
    return _exact_int(array.item(), name=name, minimum=minimum)


def _read_float64_scalar(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
) -> float:
    array = np.asarray(value)
    if array.dtype != np.dtype(np.float64) or array.shape != ():
        raise LiveVerdictStateError(f"{name} must be a float64 scalar")
    return _finite_float(array.item(), name=name, minimum=minimum)


def _read_bool_scalar(value: object, *, name: str) -> bool:
    array = np.asarray(value)
    if array.dtype != np.dtype(np.bool_) or array.shape != ():
        raise LiveVerdictStateError(f"{name} must be a bool scalar")
    return bool(array.item())


def _require_array(
    value: object,
    *,
    name: str,
    dtype: np.dtype[Any],
    shape: tuple[int, ...],
) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != dtype or array.shape != shape:
        raise LiveVerdictStateError(
            f"{name} must have dtype {dtype} and shape {shape}, "
            f"got dtype {array.dtype} and shape {array.shape}"
        )
    return array


def _require_zero(value: np.ndarray, *, name: str) -> None:
    if bool(np.any(value)):
        raise LiveVerdictStateError(f"{name} has nonzero unused padding")


def _encode_fixed_id_rows(
    values: list[str],
    *,
    rows: int,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    if len(values) > rows:
        raise LiveVerdictStateError(f"{name} row count exceeds fixed capacity")
    data = np.zeros((rows, SERIALIZED_RESULT_ID_CAPACITY), dtype=np.uint8)
    lengths = np.zeros(rows, dtype=np.int64)
    for index, value in enumerate(values):
        if type(value) is not str or not value or any(char.isspace() for char in value):
            raise LiveVerdictStateError(f"{name}[{index}] is not a canonical result ID")
        encoded = value.encode("utf-8")
        if len(encoded) > SERIALIZED_RESULT_ID_CAPACITY:
            raise LiveVerdictStateError(
                f"{name}[{index}] exceeds fixed UTF-8 capacity "
                f"{len(encoded)} > {SERIALIZED_RESULT_ID_CAPACITY}"
            )
        data[index, : len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)
        lengths[index] = len(encoded)
    return data, lengths


def _decode_fixed_id_rows(
    data_value: object,
    lengths_value: object,
    *,
    count: int,
    rows: int,
    name: str,
) -> list[str]:
    data = _require_array(
        data_value,
        name=f"{name} data",
        dtype=np.dtype(np.uint8),
        shape=(rows, SERIALIZED_RESULT_ID_CAPACITY),
    )
    lengths = _require_array(
        lengths_value,
        name=f"{name} lengths",
        dtype=np.dtype(np.int64),
        shape=(rows,),
    )
    values: list[str] = []
    for index in range(count):
        length = _exact_int(
            lengths[index],
            name=f"{name} length[{index}]",
            minimum=1,
        )
        if length > SERIALIZED_RESULT_ID_CAPACITY:
            raise LiveVerdictStateError(
                f"{name} length[{index}] exceeds fixed capacity"
            )
        _require_zero(data[index, length:], name=f"{name}[{index}] padding")
        try:
            value = data[index, :length].tobytes().decode("utf-8")
        except UnicodeError as exc:
            raise LiveVerdictStateError(
                f"{name}[{index}] is not valid UTF-8"
            ) from exc
        if not value or value.strip() != value or any(char.isspace() for char in value):
            raise LiveVerdictStateError(
                f"{name}[{index}] is not a canonical result ID"
            )
        values.append(value)
    _require_zero(data[count:], name=f"{name} unused data")
    _require_zero(lengths[count:], name=f"{name} unused lengths")
    return values


def _encode_fixed_id(value: str, *, allow_empty: bool, name: str) -> tuple[np.ndarray, np.ndarray]:
    if type(value) is not str:
        raise LiveVerdictStateError(f"{name} must be an exact string")
    if not value:
        if not allow_empty:
            raise LiveVerdictStateError(f"{name} must not be empty")
        return (
            np.zeros(SERIALIZED_RESULT_ID_CAPACITY, dtype=np.uint8),
            np.asarray(0, dtype=np.int64),
        )
    data, lengths = _encode_fixed_id_rows([value], rows=1, name=name)
    return data[0], np.asarray(lengths[0], dtype=np.int64)


def _decode_fixed_id(
    data_value: object,
    length_value: object,
    *,
    allow_empty: bool,
    name: str,
) -> str:
    data = _require_array(
        data_value,
        name=f"{name} data",
        dtype=np.dtype(np.uint8),
        shape=(SERIALIZED_RESULT_ID_CAPACITY,),
    )
    length = _read_int64_scalar(
        length_value,
        name=f"{name} length",
    )
    if length == 0:
        if not allow_empty:
            raise LiveVerdictStateError(f"{name} must not be empty")
        _require_zero(data, name=f"{name} empty data")
        return ""
    values = _decode_fixed_id_rows(
        data.reshape(1, SERIALIZED_RESULT_ID_CAPACITY),
        np.asarray([length], dtype=np.int64),
        count=1,
        rows=1,
        name=name,
    )
    return values[0]


def _encode_sha256_rows(values: list[str], *, rows: int, name: str) -> np.ndarray:
    if len(values) > rows:
        raise LiveVerdictStateError(f"{name} row count exceeds fixed capacity")
    data = np.zeros((rows, SERIALIZED_SHA256_BYTES), dtype=np.uint8)
    for index, value in enumerate(values):
        if type(value) is not str or len(value) != 2 * SERIALIZED_SHA256_BYTES:
            raise LiveVerdictStateError(f"{name}[{index}] is not a SHA-256 hex string")
        try:
            raw = bytes.fromhex(value)
        except ValueError as exc:
            raise LiveVerdictStateError(
                f"{name}[{index}] is not lowercase hexadecimal"
            ) from exc
        if value != value.lower():
            raise LiveVerdictStateError(
                f"{name}[{index}] is not lowercase hexadecimal"
            )
        data[index] = np.frombuffer(raw, dtype=np.uint8)
    return data


def _decode_sha256_rows(
    value: object,
    *,
    count: int,
    rows: int,
    name: str,
) -> list[str]:
    data = _require_array(
        value,
        name=name,
        dtype=np.dtype(np.uint8),
        shape=(rows, SERIALIZED_SHA256_BYTES),
    )
    _require_zero(data[count:], name=f"{name} unused rows")
    return [data[index].tobytes().hex() for index in range(count)]


def _pack_payload_rows(
    payloads: list[bytes],
    *,
    rows: int,
    capacity: int,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    if len(payloads) > rows:
        raise LiveVerdictStateError(f"{name} row count exceeds fixed capacity")
    offsets = np.zeros(rows + 1, dtype=np.int64)
    data = np.zeros(capacity, dtype=np.uint8)
    cursor = 0
    for index, payload in enumerate(payloads):
        if type(payload) is not bytes or not payload:
            raise LiveVerdictStateError(f"{name}[{index}] must be nonempty bytes")
        next_cursor = cursor + len(payload)
        if next_cursor > capacity:
            raise LiveVerdictStateError(
                f"{name} exceeds fixed byte capacity {next_cursor} > {capacity}"
            )
        data[cursor:next_cursor] = np.frombuffer(payload, dtype=np.uint8)
        offsets[index + 1] = next_cursor
        cursor = next_cursor
    return offsets, data


def _unpack_payload_rows(
    offsets_value: object,
    data_value: object,
    *,
    count: int,
    rows: int,
    capacity: int,
    name: str,
) -> list[bytes]:
    offsets = _require_array(
        offsets_value,
        name=f"{name} offsets",
        dtype=np.dtype(np.int64),
        shape=(rows + 1,),
    )
    data = _require_array(
        data_value,
        name=f"{name} data",
        dtype=np.dtype(np.uint8),
        shape=(capacity,),
    )
    if int(offsets[0]) != 0:
        raise LiveVerdictStateError(f"{name} offsets must begin at zero")
    payloads: list[bytes] = []
    previous = 0
    for index in range(count):
        end = _exact_int(
            offsets[index + 1],
            name=f"{name} offset[{index + 1}]",
            minimum=previous + 1,
        )
        if end > capacity:
            raise LiveVerdictStateError(f"{name} offset exceeds fixed capacity")
        payloads.append(data[previous:end].tobytes())
        previous = end
    _require_zero(offsets[count + 1 :], name=f"{name} unused offsets")
    _require_zero(data[previous:], name=f"{name} unused data")
    return payloads


def _expect_schema(value: object, *, expected: str, name: str) -> None:
    array = _require_array(
        value,
        name=name,
        dtype=np.dtype(np.uint8),
        shape=(len(expected.encode("utf-8")),),
    )
    if _decode_utf8(array, name=name) != expected:
        raise LiveVerdictStateError(f"{name} differs")


def new_reducer_state(
    *,
    history: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...] = (),
    closed_loop_verdicts: list[Mapping[str, Any]]
    | tuple[Mapping[str, Any], ...] = (),
    best_d_seg: float | None = None,
    best_epoch: int | None = None,
    best_result_id: str | None = None,
) -> dict[str, Any]:
    """Return detached explicit O4/O5 state for a fresh or restored reducer."""

    if history or closed_loop_verdicts:
        raise LiveVerdictStateError(
            "live-verdict transaction cannot restore controller-owned histories"
        )
    best_present = best_d_seg is not None or best_epoch is not None or best_result_id is not None
    if best_present:
        raise LiveVerdictStateError(
            "BEST restoration requires its retained artifact intent; use state_from_arrays"
        )
    else:
        best_value = 0.0
        best_ep = -1
        best_result_id = ""
    state = {
        "schema": SCHEMA,
        "effect_base_sequence": 0,
        "next_effect_sequence": 0,
        "effects": [],
        # Kept as empty compatibility fields until all callers consume the
        # replacement controller state directly.  They are never serialized
        # and validation forbids duplicate controller-owned history here.
        "history": [],
        "closed_loop_verdicts": [],
        "o5": {
            "best_present": bool(best_present),
            "best_d_seg": best_value,
            "best_epoch": best_ep,
            "best_result_id": best_result_id,
            "best_intent_sequence": -1,
            "best_intent_base_sequence": 0,
            "next_best_intent_sequence": 0,
            "best_intents": [],
        },
    }
    validate_reducer_state(state)
    return state


def build_worker_snapshot(
    *,
    epoch: int,
    seg_form: str,
    ep_loss: float,
    blob_bytes: int,
    best_eligible: bool,
    closed_loop_enabled: bool,
    liveness: Mapping[str, Any],
    scorer_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the string-keyed, real-shaped snapshot frozen by the transaction."""

    if type(seg_form) is not str or not seg_form:
        raise LiveVerdictStateError("seg_form must be a non-empty exact string")
    if type(best_eligible) is not bool or type(closed_loop_enabled) is not bool:
        raise LiveVerdictStateError("snapshot gates must be exact booleans")
    snapshot = {
        "schema": SNAPSHOT_SCHEMA,
        "epoch": _exact_int(epoch, name="epoch"),
        "seg_form": seg_form,
        "ep_loss": _finite_float(ep_loss, name="ep_loss"),
        "blob_bytes": _exact_int(blob_bytes, name="blob_bytes"),
        "best_eligible": best_eligible,
        "closed_loop_enabled": closed_loop_enabled,
        "liveness": dict(_exact_mapping(liveness, name="liveness")),
        "scorer": dict(_exact_mapping(scorer_snapshot, name="scorer_snapshot")),
    }
    _validate_snapshot(snapshot)
    return snapshot


def run_worker(
    submission_seq: int,
    snapshot: Mapping[str, Any],
    *,
    score_snapshot: Callable[[Mapping[str, Any]], Mapping[str, Any]],
) -> ImmutableVerdictResult:
    """Run only the caller's scorer and return one immutable result.

    ``score_snapshot`` must return ``{"verdict": mapping, "live_gap": mapping}``.
    It receives only the detached scorer subtree; no live trainer object is
    provided to the worker.
    """

    _validate_snapshot(snapshot)
    if not callable(score_snapshot):
        raise TypeError("score_snapshot must be callable")
    scored = _exact_mapping(
        score_snapshot(_exact_mapping(snapshot["scorer"], name="scorer")),
        name="scorer result",
    )
    if set(scored) != {"verdict", "live_gap"}:
        raise LiveVerdictStateError(
            "scorer result must contain exactly verdict and live_gap"
        )
    verdict = dict(_exact_mapping(scored["verdict"], name="verdict"))
    live_gap = dict(_exact_mapping(scored["live_gap"], name="live_gap"))
    payload = {
        "schema": WORKER_PAYLOAD_SCHEMA,
        "epoch": int(snapshot["epoch"]),
        "seg_form": str(snapshot["seg_form"]),
        "ep_loss": float(snapshot["ep_loss"]),
        "blob_bytes": int(snapshot["blob_bytes"]),
        "best_eligible": bool(snapshot["best_eligible"]),
        "closed_loop_enabled": bool(snapshot["closed_loop_enabled"]),
        "liveness": dict(snapshot["liveness"]),
        "scorer": dict(snapshot["scorer"]),
        "verdict": verdict,
        "live_gap": live_gap,
    }
    _validate_worker_payload(payload)
    return ImmutableVerdictResult.capture(
        submission_seq=submission_seq,
        result_id=f"g111-verdict-ep{int(snapshot['epoch'])}-seq{submission_seq}",
        payload=payload,
    )


def reduce_result(
    state: Mapping[str, Any],
    result: ImmutableVerdictResult,
) -> dict[str, Any]:
    """Purely append one ordered effect and update explicit O4/O5 state."""

    candidate = _clone_state(state)
    validate_reducer_state(candidate)
    result.validate()
    expected_sequence = int(candidate["next_effect_sequence"])
    if result.submission_seq != expected_sequence:
        raise LiveVerdictStateError(
            f"effect sequence {result.submission_seq} != expected {expected_sequence}"
        )
    payload = result.payload
    _validate_worker_payload(payload)
    verdict = _exact_mapping(payload["verdict"], name="verdict")
    d_seg = _finite_float(verdict.get("d_seg"), name="d_seg", minimum=0.0)
    o5 = candidate["o5"]
    best_intent_sequence: int | None = None
    if bool(payload["best_eligible"]) and (
        not bool(o5["best_present"])
        or d_seg < float(o5["best_d_seg"]) - 1e-12
    ):
        scorer = _exact_mapping(payload["scorer"], name="BEST scorer")
        liveness = _exact_mapping(payload["liveness"], name="BEST liveness")
        best_intent_sequence = int(o5["next_best_intent_sequence"])
        best_intent = {
            "intent_sequence": best_intent_sequence,
            "effect_sequence": result.submission_seq,
            "result_id": result.result_id,
            "result_sha256": result.result_sha256,
            "d_seg": d_seg,
            "epoch": int(payload["epoch"]),
            # Only the physical BEST writer's deploy inputs survive after the
            # full effect/scorer snapshot is acknowledged and compacted.
            "artifact": {
                "ema_np": scorer["ema_np"],
                "softmax_temp": scorer["softmax_temp"],
                "hosc_beta": scorer["hosc_beta"],
                "ema_updates": liveness.get("ema_updates"),
            },
        }
        o5["best_intents"].append(best_intent)
        o5["best_present"] = True
        o5["best_d_seg"] = d_seg
        o5["best_epoch"] = int(payload["epoch"])
        o5["best_result_id"] = result.result_id
        o5["best_intent_sequence"] = best_intent_sequence
        o5["next_best_intent_sequence"] = best_intent_sequence + 1

    candidate["effects"].append(
        {
            "sequence": result.submission_seq,
            "result_id": result.result_id,
            "result_sha256": result.result_sha256,
            "best_intent_sequence": best_intent_sequence,
            "payload": payload,
        }
    )
    candidate["next_effect_sequence"] = expected_sequence + 1
    validate_reducer_state(candidate)
    return candidate


def validate_reducer_state(state: Mapping[str, Any]) -> None:
    """Validate the complete explicit O4/O5 pure state tree."""

    value = _exact_mapping(state, name="reducer state")
    _require_keys(value, _STATE_KEYS, name="reducer state")
    if value["schema"] != SCHEMA:
        raise LiveVerdictStateError("live verdict reducer schema differs")
    effect_base = _exact_int(
        value["effect_base_sequence"],
        name="effect_base_sequence",
    )
    next_effect = _exact_int(
        value["next_effect_sequence"],
        name="next_effect_sequence",
    )
    effects = value["effects"]
    history = value["history"]
    closed_loop = value["closed_loop_verdicts"]
    if not isinstance(effects, list) or not isinstance(history, list) or not isinstance(closed_loop, list):
        raise LiveVerdictStateError("O4 effects and histories must be lists")
    if history or closed_loop:
        raise LiveVerdictStateError(
            "live-verdict transaction duplicates controller-owned history"
        )
    if effect_base > next_effect or len(effects) != next_effect - effect_base:
        raise LiveVerdictStateError(
            "effect list length differs from its global base/next cursors"
        )
    o5 = _exact_mapping(value["o5"], name="O5 state")
    _require_keys(o5, _O5_KEYS, name="O5 state")
    if type(o5["best_present"]) is not bool:
        raise LiveVerdictStateError("O5 best_present must be an exact bool")
    best_intents = o5["best_intents"]
    if not isinstance(best_intents, list):
        raise LiveVerdictStateError("O5 best_intents must be a list")
    next_best = _exact_int(
        o5["next_best_intent_sequence"],
        name="next_best_intent_sequence",
    )
    best_base = _exact_int(
        o5["best_intent_base_sequence"],
        name="best_intent_base_sequence",
    )
    if best_base > next_best or len(best_intents) != next_best - best_base:
        raise LiveVerdictStateError(
            "BEST intent list length differs from its global base/next cursors"
        )
    for offset, effect in enumerate(effects):
        sequence = effect_base + offset
        row = _exact_mapping(effect, name="effect")
        if set(row) != {
            "sequence",
            "result_id",
            "result_sha256",
            "best_intent_sequence",
            "payload",
        }:
            raise LiveVerdictStateError("effect fields differ")
        if _exact_int(row["sequence"], name="effect sequence") != sequence:
            raise LiveVerdictStateError("effects are not in exact sequence order")
        if type(row["result_id"]) is not str or type(row["result_sha256"]) is not str:
            raise LiveVerdictStateError("effect identity fields must be exact strings")
        effect_payload = _exact_mapping(row["payload"], name="effect payload")
        _validate_worker_payload(effect_payload)
        reconstructed = ImmutableVerdictResult.capture(
            submission_seq=sequence,
            result_id=row["result_id"],
            payload=effect_payload,
        )
        if reconstructed.result_sha256 != row["result_sha256"]:
            raise LiveVerdictStateError(
                "effect identity SHA-256 does not match its canonical payload"
            )
        best_seq = row["best_intent_sequence"]
        if best_seq is not None:
            best_index = _exact_int(best_seq, name="best intent sequence")
            if not best_base <= best_index < next_best:
                raise LiveVerdictStateError("effect references an absent BEST intent")
            intent = _exact_mapping(
                best_intents[best_index - best_base],
                name="BEST intent",
            )
            if (
                int(intent["effect_sequence"]) != sequence
                or intent["result_id"] != row["result_id"]
                or intent["result_sha256"] != row["result_sha256"]
            ):
                raise LiveVerdictStateError("effect and BEST intent identities differ")
    for offset, raw_intent in enumerate(best_intents):
        intent_sequence = best_base + offset
        intent = _exact_mapping(raw_intent, name="BEST intent")
        _require_keys(intent, _BEST_INTENT_KEYS, name="BEST intent")
        if (
            _exact_int(intent["intent_sequence"], name="BEST intent sequence")
            != intent_sequence
        ):
            raise LiveVerdictStateError("BEST intents are not in exact sequence order")
        _exact_int(intent["effect_sequence"], name="BEST effect sequence")
        if type(intent["result_id"]) is not str or not intent["result_id"]:
            raise LiveVerdictStateError("BEST result_id must be a non-empty exact string")
        if type(intent["result_sha256"]) is not str or len(intent["result_sha256"]) != 64:
            raise LiveVerdictStateError("BEST result_sha256 must be a SHA-256 hex string")
        try:
            bytes.fromhex(intent["result_sha256"])
        except ValueError as exc:
            raise LiveVerdictStateError(
                "BEST result_sha256 must be lowercase hexadecimal"
            ) from exc
        if intent["result_sha256"] != intent["result_sha256"].lower():
            raise LiveVerdictStateError(
                "BEST result_sha256 must be lowercase hexadecimal"
            )
        _finite_float(intent["d_seg"], name="BEST intent d_seg", minimum=0.0)
        _exact_int(intent["epoch"], name="BEST intent epoch")
        artifact = _exact_mapping(intent["artifact"], name="BEST artifact")
        _require_keys(artifact, _BEST_ARTIFACT_KEYS, name="BEST artifact")
        _exact_mapping(artifact["ema_np"], name="BEST artifact ema_np")
        _finite_float(
            artifact["softmax_temp"],
            name="BEST artifact softmax_temp",
        )
        _finite_float(artifact["hosc_beta"], name="BEST artifact hosc_beta")
        if artifact["ema_updates"] is not None:
            _exact_int(artifact["ema_updates"], name="BEST artifact ema_updates")
    if bool(o5["best_present"]):
        _finite_float(o5["best_d_seg"], name="O5 best_d_seg", minimum=0.0)
        _exact_int(o5["best_epoch"], name="O5 best_epoch")
        if type(o5["best_result_id"]) is not str or not o5["best_result_id"]:
            raise LiveVerdictStateError("present O5 BEST requires a result ID")
        last_best = _exact_int(
            o5["best_intent_sequence"],
            name="O5 best_intent_sequence",
        )
        if not best_intents or last_best != next_best - 1:
            raise LiveVerdictStateError(
                "O5 BEST pointer does not name a retained intent tail"
            )
        tail = _exact_mapping(best_intents[-1], name="O5 BEST intent tail")
        if (
            float(o5["best_d_seg"]) != float(tail["d_seg"])
            or int(o5["best_epoch"]) != int(tail["epoch"])
            or o5["best_result_id"] != tail["result_id"]
            or last_best != int(tail["intent_sequence"])
        ):
            raise LiveVerdictStateError(
                "O5 BEST summary differs from the retained BEST intent tail"
            )
    else:
        if (
            float(o5["best_d_seg"]) != 0.0
            or int(o5["best_epoch"]) != -1
            or o5["best_result_id"] != ""
            or int(o5["best_intent_sequence"]) != -1
            or best_intents
            or best_base != next_best
        ):
            raise LiveVerdictStateError("absent O5 BEST carries state")


def _validate_snapshot(snapshot: Mapping[str, Any]) -> None:
    value = _exact_mapping(snapshot, name="worker snapshot")
    _require_keys(value, _SNAPSHOT_KEYS, name="worker snapshot")
    if value["schema"] != SNAPSHOT_SCHEMA:
        raise LiveVerdictStateError("worker snapshot schema differs")
    _exact_int(value["epoch"], name="snapshot epoch")
    if type(value["seg_form"]) is not str or not value["seg_form"]:
        raise LiveVerdictStateError("snapshot seg_form must be a non-empty string")
    _finite_float(value["ep_loss"], name="snapshot ep_loss")
    _exact_int(value["blob_bytes"], name="snapshot blob_bytes")
    if type(value["best_eligible"]) is not bool or type(value["closed_loop_enabled"]) is not bool:
        raise LiveVerdictStateError("snapshot gates must be exact booleans")
    _exact_mapping(value["liveness"], name="snapshot liveness")
    scorer = _exact_mapping(value["scorer"], name="snapshot scorer")
    required_scorer = {
        "ema_np",
        "softmax_temp",
        "hosc_beta",
        "dir",
        "pose_verdict_index",
        "pose_gate_engaged_epoch",
    }
    if not required_scorer <= set(scorer):
        raise LiveVerdictStateError(
            f"snapshot scorer lacks {sorted(required_scorer - set(scorer))!r}"
        )
    _exact_int(scorer["pose_verdict_index"], name="pose_verdict_index")
    _exact_int(
        scorer["pose_gate_engaged_epoch"],
        name="pose_gate_engaged_epoch",
        minimum=-1,
    )
    _finite_float(scorer["softmax_temp"], name="softmax_temp")
    _finite_float(scorer["hosc_beta"], name="hosc_beta")
    _exact_mapping(scorer["ema_np"], name="snapshot ema_np")
    if scorer["dir"] is not None and not isinstance(scorer["dir"], (list, tuple)):
        raise LiveVerdictStateError(
            "self-orient snapshot must be an ordered list/tuple, never an int-key mapping"
        )


def _validate_worker_payload(payload: Mapping[str, Any]) -> None:
    value = _exact_mapping(payload, name="worker payload")
    _require_keys(value, _PAYLOAD_KEYS, name="worker payload")
    if value["schema"] != WORKER_PAYLOAD_SCHEMA:
        raise LiveVerdictStateError("worker payload schema differs")
    _exact_int(value["epoch"], name="payload epoch")
    if type(value["seg_form"]) is not str or not value["seg_form"]:
        raise LiveVerdictStateError("payload seg_form must be a non-empty string")
    _finite_float(value["ep_loss"], name="payload ep_loss")
    _exact_int(value["blob_bytes"], name="payload blob_bytes")
    if type(value["best_eligible"]) is not bool or type(value["closed_loop_enabled"]) is not bool:
        raise LiveVerdictStateError("payload gates must be exact booleans")
    _exact_mapping(value["liveness"], name="payload liveness")
    _exact_mapping(value["scorer"], name="payload scorer")
    verdict = _exact_mapping(value["verdict"], name="payload verdict")
    _finite_float(verdict.get("d_seg"), name="payload d_seg", minimum=0.0)
    if verdict.get("d_pose") is not None:
        _finite_float(verdict["d_pose"], name="payload d_pose", minimum=0.0)
    _exact_mapping(value["live_gap"], name="payload live_gap")


def _clone_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Clone through the canonical immutable-result codec."""

    encoded = ImmutableVerdictResult.capture(
        submission_seq=0,
        result_id=STATE_RESULT_ID,
        payload={"state": state},
    )
    decoded = encoded.payload["state"]
    if not isinstance(decoded, dict):
        raise LiveVerdictStateError("canonical reducer state did not decode to a mapping")
    return decoded


@dataclass(frozen=True, slots=True)
class PublisherCursor:
    """Serializable main-thread publication coordinate."""

    next_effect_sequence: int = 0
    next_best_intent_sequence: int = 0

    def validate(self) -> None:
        _exact_int(self.next_effect_sequence, name="publisher effect cursor")
        _exact_int(self.next_best_intent_sequence, name="publisher BEST cursor")


class MainThreadVerdictEffectPublisher:
    """Publish reduced effects once per process on the creating thread.

    A crash can happen after a callback's physical write and before the cursor
    is checkpointed.  The restored cursor then correctly replays that callback.
    Callers MUST use ``result_id``/``result_sha256`` as an idempotency key and
    overwrite deterministic bytes; literal process-lifetime exactly-once I/O is
    impossible without an external transactional store.
    """

    def __init__(self, *, cursor: PublisherCursor | None = None) -> None:
        self._main_thread_ident = threading.get_ident()
        self._cursor = cursor or PublisherCursor()
        self._cursor.validate()

    @property
    def cursor(self) -> PublisherCursor:
        return self._cursor

    def publish_pending(
        self,
        state: Mapping[str, Any],
        *,
        publish_effect: Callable[[Mapping[str, Any]], None],
        publish_best: Callable[[Mapping[str, Any], Mapping[str, Any]], None],
    ) -> int:
        if threading.get_ident() != self._main_thread_ident:
            raise LiveVerdictMainThreadError(
                "live verdict effects must publish on the creating thread"
            )
        validate_reducer_state(state)
        if not callable(publish_effect) or not callable(publish_best):
            raise TypeError("effect and BEST publishers must be callable")
        effects = state["effects"]
        best_intents = state["o5"]["best_intents"]
        effect_cursor = self._cursor.next_effect_sequence
        best_cursor = self._cursor.next_best_intent_sequence
        effect_base = int(state["effect_base_sequence"])
        best_base = int(state["o5"]["best_intent_base_sequence"])
        if (
            not effect_base <= effect_cursor <= int(state["next_effect_sequence"])
            or not best_base
            <= best_cursor
            <= int(state["o5"]["next_best_intent_sequence"])
        ):
            raise LiveVerdictStateError(
                "publisher cursor is outside the retained reducer state"
            )
        published = 0
        while effect_cursor < int(state["next_effect_sequence"]):
            effect = effects[effect_cursor - effect_base]
            if int(effect["sequence"]) != effect_cursor:
                raise LiveVerdictStateError("publisher effect sequence is not contiguous")
            try:
                publish_effect(effect)
                best_sequence = effect["best_intent_sequence"]
                if best_sequence is not None:
                    if int(best_sequence) != best_cursor:
                        raise LiveVerdictStateError(
                            "publisher BEST cursor is not aligned to the effect"
                        )
                    publish_best(effect, best_intents[best_cursor - best_base])
                    best_cursor += 1
            except BaseException as exc:
                error = LiveVerdictEffectPublicationError(
                    f"main-thread publication failed at effect sequence {effect_cursor}"
                )
                error.__cause__ = exc
                raise error from exc
            effect_cursor += 1
            self._cursor = PublisherCursor(effect_cursor, best_cursor)
            published += 1
        if best_cursor != int(state["o5"]["next_best_intent_sequence"]):
            raise LiveVerdictStateError("unreachable BEST intents remain after effect publication")
        return published


def compact_acknowledged_state(
    state: Mapping[str, Any],
    cursor: PublisherCursor,
) -> dict[str, Any]:
    """Drop acknowledged full effects while retaining the latest O5 artifact.

    The replacement controller owns durable O4 history.  Every full scorer
    snapshot before ``cursor.next_effect_sequence`` has already published and
    is removed here.  The most recent BEST intent is retained even after
    acknowledgement so the O5 summary remains tied to its exact artifact.
    """

    candidate = _clone_state(state)
    validate_reducer_state(candidate)
    cursor.validate()
    effect_base = int(candidate["effect_base_sequence"])
    next_effect = int(candidate["next_effect_sequence"])
    best_base = int(candidate["o5"]["best_intent_base_sequence"])
    next_best = int(candidate["o5"]["next_best_intent_sequence"])
    if (
        not effect_base <= cursor.next_effect_sequence <= next_effect
        or not best_base <= cursor.next_best_intent_sequence <= next_best
    ):
        raise LiveVerdictStateError(
            "cannot compact outside the retained publication interval"
        )

    effect_drop = cursor.next_effect_sequence - effect_base
    candidate["effects"] = candidate["effects"][effect_drop:]
    candidate["effect_base_sequence"] = cursor.next_effect_sequence

    # Acknowledged non-tail BEST intents are dead.  Keep the latest forever
    # (O(model), not O(verdicts*model)) so validation can bind O5 summary and
    # deploy artifact rather than trusting a detached scalar minimum.
    retain_best_from = cursor.next_best_intent_sequence
    if bool(candidate["o5"]["best_present"]):
        retain_best_from = min(retain_best_from, next_best - 1)
    best_drop = retain_best_from - best_base
    candidate["o5"]["best_intents"] = candidate["o5"]["best_intents"][
        best_drop:
    ]
    candidate["o5"]["best_intent_base_sequence"] = retain_best_from
    validate_reducer_state(candidate)
    return candidate


def state_arrays(
    state: Mapping[str, Any],
    cursor: PublisherCursor,
    *,
    prefix: str,
) -> Mapping[str, np.ndarray]:
    """Serialize O4/O5 into disjoint fixed-census, fixed-shape arrays."""

    if type(prefix) is not str or prefix.strip() != prefix:
        raise LiveVerdictStateError("state prefix must be a canonical string")
    validate_reducer_state(state)
    cursor.validate()
    if (
        cursor.next_effect_sequence < int(state["effect_base_sequence"])
        or cursor.next_effect_sequence > int(state["next_effect_sequence"])
        or cursor.next_best_intent_sequence
        < int(state["o5"]["best_intent_base_sequence"])
        or cursor.next_best_intent_sequence
        > int(state["o5"]["next_best_intent_sequence"])
    ):
        raise LiveVerdictStateError("publication cursor exceeds reducer state")
    compacted = compact_acknowledged_state(state, cursor)
    effects = compacted["effects"]
    best_intents = compacted["o5"]["best_intents"]
    effect_count = len(effects)
    best_intent_count = len(best_intents)
    if effect_count > SERIALIZED_MAX_EFFECT_ROWS:
        raise LiveVerdictStateError(
            "serialized O4 effect count exceeds fixed capacity "
            f"{effect_count} > {SERIALIZED_MAX_EFFECT_ROWS}"
        )
    if best_intent_count > SERIALIZED_MAX_BEST_INTENT_ROWS:
        raise LiveVerdictStateError(
            "serialized O5 BEST-intent count exceeds fixed capacity "
            f"{best_intent_count} > {SERIALIZED_MAX_BEST_INTENT_ROWS}"
        )

    effect_ids = [str(effect["result_id"]) for effect in effects]
    effect_hashes = [str(effect["result_sha256"]) for effect in effects]
    effect_id_data, effect_id_lengths = _encode_fixed_id_rows(
        effect_ids,
        rows=SERIALIZED_MAX_EFFECT_ROWS,
        name="O4 effect result IDs",
    )
    effect_sha256 = _encode_sha256_rows(
        effect_hashes,
        rows=SERIALIZED_MAX_EFFECT_ROWS,
        name="O4 effect result SHA-256",
    )
    effect_has_best = np.zeros(SERIALIZED_MAX_EFFECT_ROWS, dtype=np.bool_)
    effect_best_sequences = np.zeros(SERIALIZED_MAX_EFFECT_ROWS, dtype=np.int64)
    effect_payload_bytes: list[bytes] = []
    for index, effect in enumerate(effects):
        sequence = int(effect["sequence"])
        encoded_effect = ImmutableVerdictResult.capture(
            submission_seq=sequence,
            result_id=effect_ids[index],
            payload=_exact_mapping(effect["payload"], name="effect payload"),
        )
        if encoded_effect.result_sha256 != effect_hashes[index]:
            raise LiveVerdictStateError(
                "O4 effect identity SHA-256 differs before serialization"
            )
        effect_payload_bytes.append(encoded_effect.payload_bytes)
        best_sequence = effect["best_intent_sequence"]
        if best_sequence is not None:
            effect_has_best[index] = True
            effect_best_sequences[index] = _exact_int(
                best_sequence,
                name="effect BEST intent sequence",
            )
    effect_payload_offsets, effect_payload_data = _pack_payload_rows(
        effect_payload_bytes,
        rows=SERIALIZED_MAX_EFFECT_ROWS,
        capacity=SERIALIZED_O4_PAYLOAD_CAPACITY,
        name="O4 effect payloads",
    )

    o5 = compacted["o5"]
    best_result_id_data, best_result_id_length = _encode_fixed_id(
        str(o5["best_result_id"]),
        allow_empty=not bool(o5["best_present"]),
        name="O5 BEST result ID",
    )
    intent_ids = [str(intent["result_id"]) for intent in best_intents]
    intent_hashes = [str(intent["result_sha256"]) for intent in best_intents]
    intent_id_data, intent_id_lengths = _encode_fixed_id_rows(
        intent_ids,
        rows=SERIALIZED_MAX_BEST_INTENT_ROWS,
        name="O5 intent result IDs",
    )
    intent_sha256 = _encode_sha256_rows(
        intent_hashes,
        rows=SERIALIZED_MAX_BEST_INTENT_ROWS,
        name="O5 intent result SHA-256",
    )
    intent_effect_sequences = np.zeros(
        SERIALIZED_MAX_BEST_INTENT_ROWS,
        dtype=np.int64,
    )
    intent_d_seg = np.zeros(SERIALIZED_MAX_BEST_INTENT_ROWS, dtype=np.float64)
    intent_epochs = np.zeros(SERIALIZED_MAX_BEST_INTENT_ROWS, dtype=np.int64)
    artifact_payload_bytes: list[bytes] = []
    artifact_hashes: list[str] = []
    for index, intent in enumerate(best_intents):
        intent_sequence = int(intent["intent_sequence"])
        intent_effect_sequences[index] = int(intent["effect_sequence"])
        intent_d_seg[index] = float(intent["d_seg"])
        intent_epochs[index] = int(intent["epoch"])
        encoded_artifact = ImmutableVerdictResult.capture(
            submission_seq=intent_sequence,
            result_id=f"{intent_ids[index]}:o5-artifact",
            payload={
                "artifact": _exact_mapping(
                    intent["artifact"],
                    name="BEST artifact",
                )
            },
        )
        artifact_payload_bytes.append(encoded_artifact.payload_bytes)
        artifact_hashes.append(encoded_artifact.result_sha256)
    artifact_payload_offsets, artifact_payload_data = _pack_payload_rows(
        artifact_payload_bytes,
        rows=SERIALIZED_MAX_BEST_INTENT_ROWS,
        capacity=SERIALIZED_O5_ARTIFACT_CAPACITY,
        name="O5 artifact payloads",
    )
    artifact_sha256 = _encode_sha256_rows(
        artifact_hashes,
        rows=SERIALIZED_MAX_BEST_INTENT_ROWS,
        name="O5 artifact SHA-256",
    )

    arrays = {
        f"{prefix}o4_schema": _utf8_array(SERIALIZED_O4_SCHEMA),
        f"{prefix}o4_effect_count": _int64_scalar(effect_count),
        f"{prefix}o4_effect_base_sequence": _int64_scalar(
            compacted["effect_base_sequence"]
        ),
        f"{prefix}o4_next_effect_sequence": _int64_scalar(
            compacted["next_effect_sequence"]
        ),
        f"{prefix}o4_effect_cursor": _int64_scalar(
            cursor.next_effect_sequence
        ),
        f"{prefix}o4_effect_result_id_data": effect_id_data,
        f"{prefix}o4_effect_result_id_lengths": effect_id_lengths,
        f"{prefix}o4_effect_result_sha256": effect_sha256,
        f"{prefix}o4_effect_has_best_intent": effect_has_best,
        f"{prefix}o4_effect_best_intent_sequences": effect_best_sequences,
        f"{prefix}o4_effect_payload_offsets": effect_payload_offsets,
        f"{prefix}o4_effect_payload_data": effect_payload_data,
        f"{prefix}o5_schema": _utf8_array(SERIALIZED_O5_SCHEMA),
        f"{prefix}o5_best_present": np.asarray(
            bool(o5["best_present"]),
            dtype=np.bool_,
        ),
        f"{prefix}o5_best_d_seg": np.asarray(
            float(o5["best_d_seg"]),
            dtype=np.float64,
        ),
        f"{prefix}o5_best_epoch": np.asarray(
            int(o5["best_epoch"]),
            dtype=np.int64,
        ),
        f"{prefix}o5_best_result_id_data": best_result_id_data,
        f"{prefix}o5_best_result_id_length": best_result_id_length,
        f"{prefix}o5_best_intent_sequence": np.asarray(
            int(o5["best_intent_sequence"]),
            dtype=np.int64,
        ),
        f"{prefix}o5_best_intent_count": _int64_scalar(best_intent_count),
        f"{prefix}o5_best_intent_base_sequence": _int64_scalar(
            o5["best_intent_base_sequence"]
        ),
        f"{prefix}o5_next_best_intent_sequence": _int64_scalar(
            o5["next_best_intent_sequence"]
        ),
        f"{prefix}o5_best_intent_cursor": _int64_scalar(
            cursor.next_best_intent_sequence
        ),
        f"{prefix}o5_intent_effect_sequences": intent_effect_sequences,
        f"{prefix}o5_intent_result_id_data": intent_id_data,
        f"{prefix}o5_intent_result_id_lengths": intent_id_lengths,
        f"{prefix}o5_intent_result_sha256": intent_sha256,
        f"{prefix}o5_intent_d_seg": intent_d_seg,
        f"{prefix}o5_intent_epochs": intent_epochs,
        f"{prefix}o5_artifact_payload_offsets": artifact_payload_offsets,
        f"{prefix}o5_artifact_payload_data": artifact_payload_data,
        f"{prefix}o5_artifact_sha256": artifact_sha256,
    }
    return MappingProxyType(arrays)


def state_from_arrays(
    arrays: Mapping[str, Any],
    *,
    prefix: str,
) -> tuple[dict[str, Any], PublisherCursor]:
    """Restore and validate fixed-census O4/O5 reducer/publication state."""

    if type(prefix) is not str or prefix.strip() != prefix:
        raise LiveVerdictStateError("state prefix must be a canonical string")
    required = {
        *(f"{prefix}o4_{field}" for field in SERIALIZED_O4_FIELDS),
        *(f"{prefix}o5_{field}" for field in SERIALIZED_O5_FIELDS),
    }
    actual = {key for key in arrays if key.startswith(prefix)}
    if actual != required:
        raise LiveVerdictStateError(
            "serialized live-verdict array census differs; "
            f"missing={sorted(required - actual)}, "
            f"unknown={sorted(actual - required)}"
        )
    _expect_schema(
        arrays[f"{prefix}o4_schema"],
        expected=SERIALIZED_O4_SCHEMA,
        name="serialized O4 schema",
    )
    _expect_schema(
        arrays[f"{prefix}o5_schema"],
        expected=SERIALIZED_O5_SCHEMA,
        name="serialized O5 schema",
    )

    effect_count = _read_int64_scalar(
        arrays[f"{prefix}o4_effect_count"],
        name="O4 effect count",
    )
    if effect_count > SERIALIZED_MAX_EFFECT_ROWS:
        raise LiveVerdictStateError("O4 effect count exceeds fixed row capacity")
    effect_base = _read_int64_scalar(
        arrays[f"{prefix}o4_effect_base_sequence"],
        name="O4 effect base sequence",
    )
    next_effect = _read_int64_scalar(
        arrays[f"{prefix}o4_next_effect_sequence"],
        name="O4 next effect sequence",
    )
    effect_cursor = _read_int64_scalar(
        arrays[f"{prefix}o4_effect_cursor"],
        name="O4 effect cursor",
    )
    if (
        effect_base > next_effect
        or effect_count != next_effect - effect_base
        or not effect_base <= effect_cursor <= next_effect
    ):
        raise LiveVerdictStateError(
            "O4 count/base/next/cursor coordinates differ"
        )
    effect_ids = _decode_fixed_id_rows(
        arrays[f"{prefix}o4_effect_result_id_data"],
        arrays[f"{prefix}o4_effect_result_id_lengths"],
        count=effect_count,
        rows=SERIALIZED_MAX_EFFECT_ROWS,
        name="O4 effect result IDs",
    )
    effect_hashes = _decode_sha256_rows(
        arrays[f"{prefix}o4_effect_result_sha256"],
        count=effect_count,
        rows=SERIALIZED_MAX_EFFECT_ROWS,
        name="O4 effect result SHA-256",
    )
    effect_has_best = _require_array(
        arrays[f"{prefix}o4_effect_has_best_intent"],
        name="O4 effect BEST-presence vector",
        dtype=np.dtype(np.bool_),
        shape=(SERIALIZED_MAX_EFFECT_ROWS,),
    )
    effect_best_sequences = _require_array(
        arrays[f"{prefix}o4_effect_best_intent_sequences"],
        name="O4 effect BEST-sequence vector",
        dtype=np.dtype(np.int64),
        shape=(SERIALIZED_MAX_EFFECT_ROWS,),
    )
    _require_zero(
        effect_has_best[effect_count:],
        name="O4 effect BEST-presence unused rows",
    )
    _require_zero(
        effect_best_sequences[effect_count:],
        name="O4 effect BEST-sequence unused rows",
    )
    effect_payloads = _unpack_payload_rows(
        arrays[f"{prefix}o4_effect_payload_offsets"],
        arrays[f"{prefix}o4_effect_payload_data"],
        count=effect_count,
        rows=SERIALIZED_MAX_EFFECT_ROWS,
        capacity=SERIALIZED_O4_PAYLOAD_CAPACITY,
        name="O4 effect payloads",
    )

    best_present = _read_bool_scalar(
        arrays[f"{prefix}o5_best_present"],
        name="O5 best_present",
    )
    best_d_seg = _read_float64_scalar(
        arrays[f"{prefix}o5_best_d_seg"],
        name="O5 best_d_seg",
        minimum=0.0,
    )
    best_epoch = _read_int64_scalar(
        arrays[f"{prefix}o5_best_epoch"],
        name="O5 best_epoch",
        minimum=-1,
    )
    best_result_id = _decode_fixed_id(
        arrays[f"{prefix}o5_best_result_id_data"],
        arrays[f"{prefix}o5_best_result_id_length"],
        allow_empty=not best_present,
        name="O5 BEST result ID",
    )
    best_intent_sequence = _read_int64_scalar(
        arrays[f"{prefix}o5_best_intent_sequence"],
        name="O5 best intent sequence",
        minimum=-1,
    )
    best_intent_count = _read_int64_scalar(
        arrays[f"{prefix}o5_best_intent_count"],
        name="O5 best intent count",
    )
    if best_intent_count > SERIALIZED_MAX_BEST_INTENT_ROWS:
        raise LiveVerdictStateError(
            "O5 BEST-intent count exceeds fixed row capacity"
        )
    best_base = _read_int64_scalar(
        arrays[f"{prefix}o5_best_intent_base_sequence"],
        name="O5 best intent base sequence",
    )
    next_best = _read_int64_scalar(
        arrays[f"{prefix}o5_next_best_intent_sequence"],
        name="O5 next best intent sequence",
    )
    best_cursor = _read_int64_scalar(
        arrays[f"{prefix}o5_best_intent_cursor"],
        name="O5 best intent cursor",
    )
    if (
        best_base > next_best
        or best_intent_count != next_best - best_base
        or not best_base <= best_cursor <= next_best
    ):
        raise LiveVerdictStateError(
            "O5 count/base/next/cursor coordinates differ"
        )
    intent_effect_sequences = _require_array(
        arrays[f"{prefix}o5_intent_effect_sequences"],
        name="O5 intent effect sequences",
        dtype=np.dtype(np.int64),
        shape=(SERIALIZED_MAX_BEST_INTENT_ROWS,),
    )
    intent_ids = _decode_fixed_id_rows(
        arrays[f"{prefix}o5_intent_result_id_data"],
        arrays[f"{prefix}o5_intent_result_id_lengths"],
        count=best_intent_count,
        rows=SERIALIZED_MAX_BEST_INTENT_ROWS,
        name="O5 intent result IDs",
    )
    intent_hashes = _decode_sha256_rows(
        arrays[f"{prefix}o5_intent_result_sha256"],
        count=best_intent_count,
        rows=SERIALIZED_MAX_BEST_INTENT_ROWS,
        name="O5 intent result SHA-256",
    )
    intent_d_seg = _require_array(
        arrays[f"{prefix}o5_intent_d_seg"],
        name="O5 intent d_seg",
        dtype=np.dtype(np.float64),
        shape=(SERIALIZED_MAX_BEST_INTENT_ROWS,),
    )
    intent_epochs = _require_array(
        arrays[f"{prefix}o5_intent_epochs"],
        name="O5 intent epochs",
        dtype=np.dtype(np.int64),
        shape=(SERIALIZED_MAX_BEST_INTENT_ROWS,),
    )
    _require_zero(
        intent_effect_sequences[best_intent_count:],
        name="O5 intent effect-sequence unused rows",
    )
    _require_zero(
        intent_d_seg[best_intent_count:],
        name="O5 intent d_seg unused rows",
    )
    _require_zero(
        intent_epochs[best_intent_count:],
        name="O5 intent epoch unused rows",
    )
    artifact_payloads = _unpack_payload_rows(
        arrays[f"{prefix}o5_artifact_payload_offsets"],
        arrays[f"{prefix}o5_artifact_payload_data"],
        count=best_intent_count,
        rows=SERIALIZED_MAX_BEST_INTENT_ROWS,
        capacity=SERIALIZED_O5_ARTIFACT_CAPACITY,
        name="O5 artifact payloads",
    )
    artifact_hashes = _decode_sha256_rows(
        arrays[f"{prefix}o5_artifact_sha256"],
        count=best_intent_count,
        rows=SERIALIZED_MAX_BEST_INTENT_ROWS,
        name="O5 artifact SHA-256",
    )

    effects: list[dict[str, Any]] = []
    for index in range(effect_count):
        sequence = effect_base + index
        if not bool(effect_has_best[index]) and int(effect_best_sequences[index]) != 0:
            raise LiveVerdictStateError(
                "O4 effect without BEST carries a BEST sequence"
            )
        best_sequence = (
            _exact_int(
                effect_best_sequences[index],
                name="O4 effect BEST intent sequence",
            )
            if bool(effect_has_best[index])
            else None
        )
        encoded_effect = ImmutableVerdictResult(
            submission_seq=sequence,
            result_id=effect_ids[index],
            payload_bytes=effect_payloads[index],
            result_sha256=effect_hashes[index],
        )
        try:
            encoded_effect.validate()
        except ResultIntegrityError as exc:
            raise LiveVerdictStateError(
                "serialized O4 effect payload or identity is invalid"
            ) from exc
        effects.append(
            {
                "sequence": sequence,
                "result_id": effect_ids[index],
                "result_sha256": effect_hashes[index],
                "best_intent_sequence": best_sequence,
                "payload": encoded_effect.payload,
            }
        )

    best_intents: list[dict[str, Any]] = []
    for index in range(best_intent_count):
        intent_sequence = best_base + index
        effect_sequence = _exact_int(
            intent_effect_sequences[index],
            name="O5 intent effect sequence",
        )
        d_seg = _finite_float(
            intent_d_seg[index],
            name="O5 intent d_seg",
            minimum=0.0,
        )
        epoch = _exact_int(intent_epochs[index], name="O5 intent epoch")
        encoded_artifact = ImmutableVerdictResult(
            submission_seq=intent_sequence,
            result_id=f"{intent_ids[index]}:o5-artifact",
            payload_bytes=artifact_payloads[index],
            result_sha256=artifact_hashes[index],
        )
        try:
            encoded_artifact.validate()
        except ResultIntegrityError as exc:
            raise LiveVerdictStateError(
                "serialized O5 artifact payload or identity is invalid"
            ) from exc
        artifact_wrapper = encoded_artifact.payload
        if set(artifact_wrapper) != {"artifact"}:
            raise LiveVerdictStateError(
                "serialized O5 artifact wrapper fields differ"
            )
        artifact = artifact_wrapper["artifact"]
        if not isinstance(artifact, dict):
            raise LiveVerdictStateError(
                "serialized O5 artifact must decode to a mapping"
            )
        best_intents.append(
            {
                "intent_sequence": intent_sequence,
                "effect_sequence": effect_sequence,
                "result_id": intent_ids[index],
                "result_sha256": intent_hashes[index],
                "d_seg": d_seg,
                "epoch": epoch,
                "artifact": artifact,
            }
        )

    state = {
        "schema": SCHEMA,
        "effect_base_sequence": effect_base,
        "next_effect_sequence": next_effect,
        "effects": effects,
        "history": [],
        "closed_loop_verdicts": [],
        "o5": {
            "best_present": best_present,
            "best_d_seg": best_d_seg,
            "best_epoch": best_epoch,
            "best_result_id": best_result_id,
            "best_intent_sequence": best_intent_sequence,
            "best_intent_base_sequence": best_base,
            "next_best_intent_sequence": next_best,
            "best_intents": best_intents,
        },
    }
    cursor = PublisherCursor(
        next_effect_sequence=effect_cursor,
        next_best_intent_sequence=best_cursor,
    )
    validate_reducer_state(state)
    cursor.validate()
    return state, cursor


__all__ = [
    "SCHEMA",
    "SERIALIZED_MAX_BEST_INTENT_ROWS",
    "SERIALIZED_MAX_EFFECT_ROWS",
    "SERIALIZED_O4_FIELDS",
    "SERIALIZED_O4_PAYLOAD_CAPACITY",
    "SERIALIZED_O5_ARTIFACT_CAPACITY",
    "SERIALIZED_O5_FIELDS",
    "SERIALIZED_RESULT_ID_CAPACITY",
    "SERIALIZED_STATE_SCHEMA",
    "SNAPSHOT_SCHEMA",
    "WORKER_PAYLOAD_SCHEMA",
    "LiveVerdictEffectPublicationError",
    "LiveVerdictMainThreadError",
    "LiveVerdictStateError",
    "MainThreadVerdictEffectPublisher",
    "PublisherCursor",
    "build_worker_snapshot",
    "compact_acknowledged_state",
    "new_reducer_state",
    "reduce_result",
    "run_worker",
    "state_arrays",
    "state_from_arrays",
    "validate_reducer_state",
]
