# SPDX-License-Identifier: MIT
"""Deterministic async-verdict reduction and strict checkpoint quiescence.

This module is deliberately trainer-independent.  A worker receives a detached,
immutable NumPy tree and must return :class:`ImmutableVerdictResult`; it never
touches controller state or a journal.  The thread that creates
:class:`QuiescentVerdictTransaction` is the only thread allowed to reduce
results or enter a checkpoint.  Future callbacks may only latch cancellation or
worker exceptions; they never inspect successful payloads or reduce state.

Lock order is:

1. the logical checkpoint barrier / submission gate;
2. the reducer lock;
3. caller-owned controller locks, if any.

The core never acquires a caller-owned lock.  Reducers therefore operate on a
detached candidate state and return the complete replacement state.  A reducer
must not publish external side effects: successful return is the transaction's
single publication point.

Native-v3 checkpoints use a strict barrier:

* stop new submissions;
* join every submitted worker;
* reduce results in exact submission order;
* publish and acknowledge external effects through one typed main-thread
  transition;
* require no pending work and equal submit/apply cursors;
* capture the post-ack reducer state;
* keep the barrier active while the caller publishes the rest of the
  checkpoint;
* release on success or failure.

Legacy ``__cl_pend_*`` payloads are intentionally rejected by the native-v3
state opener.  The bounded journal stores sequence, result ID, and content hash
only; controller and sensor histories remain ordinary O3/O4 state owned by the
caller's reducer.

External effect callbacks must use a deterministic result identity as their
idempotency key.  A process can crash after the physical effect but before the
acknowledged replacement state is durable, so replay from the previous complete
checkpoint is intentional.  Any callback/validation failure poisons the same
transaction permanently; it can never be retried in-process.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import threading
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import CancelledError, Executor, Future
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import pairwise
from types import MappingProxyType
from typing import Any, Final

import numpy as np

SCHEMA: Final = "tac.g111_verdict_barrier.v1"
LEGACY_PENDING_PREFIX: Final = "__cl_pend_"
SHA256_HEX_LENGTH: Final = 64
_RESULT_DOMAIN: Final = b"tac.g111.immutable-verdict-result.v1\0"
_STATE_FIELDS: Final = frozenset(
    {
        "schema",
        "next_submit_seq",
        "next_apply_seq",
        "pending_count",
        "last_applied_result_id",
        "last_applied_result_sha256",
        "journal_limit",
        "journal_sequences",
        "journal_result_id_data",
        "journal_result_id_offsets",
        "journal_result_sha256",
    }
)


class VerdictTransactionError(RuntimeError):
    """The verdict transaction failed closed."""


class FatalVerdictTransactionError(VerdictTransactionError):
    """A fatal error requiring restart from the last complete checkpoint."""


class MainThreadViolationError(VerdictTransactionError):
    """A reducer or checkpoint operation ran outside the creating thread."""


class SubmissionBlockedError(VerdictTransactionError):
    """A submission was attempted while the checkpoint barrier was active."""


class TransactionPoisonedError(VerdictTransactionError):
    """A prior fatal worker/reducer failure made all further use unsafe."""


class WorkerExecutionError(FatalVerdictTransactionError):
    """A verdict worker raised instead of returning a complete result."""


class WorkerCancelledError(FatalVerdictTransactionError):
    """A submitted verdict worker was cancelled before completion."""


class WorkerResultTypeError(FatalVerdictTransactionError):
    """A worker returned a mutable or otherwise unsupported result."""


class ResultIntegrityError(FatalVerdictTransactionError):
    """An immutable result's identity or canonical content hash is invalid."""


class SequenceGapError(FatalVerdictTransactionError):
    """A result arrived after the exact next sequence."""


class DuplicateSequenceError(FatalVerdictTransactionError):
    """A sequence at or before the applied cursor was offered again."""


class DuplicateResultError(FatalVerdictTransactionError):
    """A result ID already present in the bounded applied journal reappeared."""


class ReducerApplicationError(FatalVerdictTransactionError):
    """The pure reducer failed before its replacement state was published."""


class ExternalEffectTransitionError(FatalVerdictTransactionError):
    """Main-thread external-effect publication/acknowledgement failed."""


class CheckpointPreparationError(FatalVerdictTransactionError):
    """Checkpoint preparation failed before a post-ack capture was yielded."""


class CheckpointPublicationError(FatalVerdictTransactionError):
    """Caller checkpoint publication failed while the barrier was held."""


class NonQuiescentCheckpointError(VerdictTransactionError):
    """A checkpoint capture did not satisfy strict native-v3 quiescence."""


class NativeV3PendingPayloadError(VerdictTransactionError):
    """A legacy pending-verdict key appeared in native-v3 state."""


class CanonicalStateError(VerdictTransactionError):
    """Canonical barrier state is missing, malformed, or internally unequal."""


def _validate_nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative, got {result}")
    return result


def _readonly_array(value: np.ndarray) -> np.ndarray:
    """Copy into an immutable ``bytes`` owner, not merely a cleared write flag."""

    source = np.ascontiguousarray(value)
    immutable_buffer = source.tobytes(order="C")
    return np.frombuffer(immutable_buffer, dtype=source.dtype).reshape(source.shape)


def _clone_numpy_tree(value: Any) -> Any:
    """Return a detached mutable copy of a pure Python/NumPy state tree."""

    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("object arrays are forbidden in verdict transaction state")
        return np.array(value, copy=True, order="C")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        cloned: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("verdict transaction mapping keys must be strings")
            cloned[key] = _clone_numpy_tree(item)
        return cloned
    if isinstance(value, list):
        return [_clone_numpy_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_numpy_tree(item) for item in value)
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    raise TypeError(f"verdict transaction state must be a pure Python/NumPy tree; got {type(value).__name__}")


def _freeze_numpy_tree(value: Any) -> Any:
    """Detach and recursively make a worker snapshot immutable."""

    cloned = _clone_numpy_tree(value)

    def freeze(item: Any) -> Any:
        if isinstance(item, np.ndarray):
            return _readonly_array(item)
        if isinstance(item, dict):
            return MappingProxyType({key: freeze(child) for key, child in item.items()})
        if isinstance(item, list):
            return tuple(freeze(child) for child in item)
        if isinstance(item, tuple):
            return tuple(freeze(child) for child in item)
        return item

    return freeze(cloned)


def _encode_payload_value(value: Any) -> dict[str, Any]:
    """Encode one result-payload value into a canonical JSON object."""

    if isinstance(value, np.generic):
        value = np.asarray(value)
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject or value.dtype.fields is not None:
            raise TypeError("object and structured arrays are forbidden in verdict results")
        array = np.ascontiguousarray(value)
        if array.dtype.kind in {"f", "c"} and not bool(np.all(np.isfinite(array))):
            raise ValueError("verdict result arrays must be finite")
        return {
            "type": "ndarray",
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "data": base64.b64encode(array.tobytes(order="C")).decode("ascii"),
        }
    if isinstance(value, Mapping):
        items: list[list[Any]] = []
        for key in sorted(value):
            if not isinstance(key, str):
                raise TypeError("verdict result mapping keys must be strings")
            items.append([key, _encode_payload_value(value[key])])
        return {"type": "mapping", "items": items}
    if isinstance(value, list):
        return {
            "type": "list",
            "items": [_encode_payload_value(item) for item in value],
        }
    if isinstance(value, tuple):
        return {
            "type": "tuple",
            "items": [_encode_payload_value(item) for item in value],
        }
    if isinstance(value, bytes):
        return {
            "type": "bytes",
            "data": base64.b64encode(value).decode("ascii"),
        }
    if value is None:
        return {"type": "none"}
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, int):
        return {"type": "int", "value": str(value)}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("verdict result floats must be finite")
        return {"type": "float", "value": value.hex()}
    if isinstance(value, str):
        return {"type": "str", "value": value}
    raise TypeError(f"unsupported verdict result payload type {type(value).__name__}")


def _require_encoded_fields(
    encoded: Mapping[str, Any],
    *,
    expected: frozenset[str],
    kind: str,
) -> None:
    actual = set(encoded)
    if actual != expected:
        raise ResultIntegrityError(
            f"canonical verdict {kind} fields {sorted(actual)!r} != expected {sorted(expected)!r}"
        )


def _decode_payload_value(encoded: object) -> Any:
    """Strict recursive decoder; every malformed boundary has one typed error."""

    if not isinstance(encoded, Mapping):
        raise ResultIntegrityError(f"canonical verdict payload node must be a mapping, got {type(encoded).__name__}")
    kind = encoded.get("type")
    if not isinstance(kind, str):
        raise ResultIntegrityError("canonical verdict payload node type must be a string")
    try:
        if kind == "ndarray":
            _require_encoded_fields(
                encoded,
                expected=frozenset({"type", "dtype", "shape", "data"}),
                kind=kind,
            )
            if not isinstance(encoded["dtype"], str):
                raise ResultIntegrityError("canonical ndarray dtype must be a string")
            shape_value = encoded["shape"]
            if not isinstance(shape_value, list) or any(
                isinstance(dim, bool) or not isinstance(dim, int) for dim in shape_value
            ):
                raise ResultIntegrityError("canonical ndarray shape must be a list of integers")
            if not isinstance(encoded["data"], str):
                raise ResultIntegrityError("canonical ndarray data must be a string")
            dtype = np.dtype(encoded["dtype"])
            if dtype.hasobject or dtype.fields is not None:
                raise ResultIntegrityError("canonical verdict result contains a forbidden dtype")
            shape = tuple(shape_value)
            if any(dim < 0 for dim in shape):
                raise ResultIntegrityError("canonical verdict result contains a negative dimension")
            raw = base64.b64decode(encoded["data"], validate=True)
            expected_bytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
            if len(raw) != expected_bytes:
                raise ResultIntegrityError(
                    f"canonical verdict result byte length {len(raw)} != expected {expected_bytes}"
                )
            return np.frombuffer(raw, dtype=dtype).copy().reshape(shape)
        if kind == "mapping":
            _require_encoded_fields(
                encoded,
                expected=frozenset({"type", "items"}),
                kind=kind,
            )
            items = encoded["items"]
            if not isinstance(items, list):
                raise ResultIntegrityError("canonical verdict mapping items must be a list")
            result: dict[str, Any] = {}
            previous: str | None = None
            for pair in items:
                if not isinstance(pair, list) or len(pair) != 2 or not isinstance(pair[0], str):
                    raise ResultIntegrityError("canonical verdict mapping entry is malformed")
                key = pair[0]
                if previous is not None and key <= previous:
                    raise ResultIntegrityError("canonical verdict mapping keys are not strictly ordered")
                previous = key
                result[key] = _decode_payload_value(pair[1])
            return result
        if kind in {"list", "tuple"}:
            _require_encoded_fields(
                encoded,
                expected=frozenset({"type", "items"}),
                kind=kind,
            )
            items = encoded["items"]
            if not isinstance(items, list):
                raise ResultIntegrityError(f"canonical verdict {kind} items must be a list")
            decoded = [_decode_payload_value(item) for item in items]
            return decoded if kind == "list" else tuple(decoded)
        if kind == "bytes":
            _require_encoded_fields(
                encoded,
                expected=frozenset({"type", "data"}),
                kind=kind,
            )
            if not isinstance(encoded["data"], str):
                raise ResultIntegrityError("canonical bytes data must be a string")
            return base64.b64decode(encoded["data"], validate=True)
        if kind == "none":
            _require_encoded_fields(
                encoded,
                expected=frozenset({"type"}),
                kind=kind,
            )
            return None
        if kind == "bool":
            _require_encoded_fields(
                encoded,
                expected=frozenset({"type", "value"}),
                kind=kind,
            )
            if not isinstance(encoded["value"], bool):
                raise ResultIntegrityError("canonical bool value must be boolean")
            return encoded["value"]
        if kind == "int":
            _require_encoded_fields(
                encoded,
                expected=frozenset({"type", "value"}),
                kind=kind,
            )
            value = encoded["value"]
            if not isinstance(value, str):
                raise ResultIntegrityError("canonical int value must be a string")
            decoded_int = int(value)
            if str(decoded_int) != value:
                raise ResultIntegrityError("canonical int value is not normalized")
            return decoded_int
        if kind == "float":
            _require_encoded_fields(
                encoded,
                expected=frozenset({"type", "value"}),
                kind=kind,
            )
            value = encoded["value"]
            if not isinstance(value, str):
                raise ResultIntegrityError("canonical float value must be a string")
            decoded_float = float.fromhex(value)
            if not math.isfinite(decoded_float) or decoded_float.hex() != value:
                raise ResultIntegrityError("canonical verdict result float is not normalized and finite")
            return decoded_float
        if kind == "str":
            _require_encoded_fields(
                encoded,
                expected=frozenset({"type", "value"}),
                kind=kind,
            )
            if not isinstance(encoded["value"], str):
                raise ResultIntegrityError("canonical str value must be a string")
            return encoded["value"]
    except ResultIntegrityError:
        raise
    except (KeyError, TypeError, ValueError, UnicodeError) as exc:
        raise ResultIntegrityError(f"canonical verdict {kind!r} payload node is malformed") from exc
    raise ResultIntegrityError(f"unknown canonical verdict result payload type {kind!r}")


def _canonical_payload_bytes(payload: Mapping[str, Any]) -> bytes:
    encoded = _encode_payload_value(payload)
    return json.dumps(
        encoded,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _result_sha256(*, submission_seq: int, result_id: str, payload_bytes: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(_RESULT_DOMAIN)
    digest.update(str(submission_seq).encode("ascii"))
    digest.update(b"\0")
    digest.update(result_id.encode("utf-8"))
    digest.update(b"\0")
    digest.update(payload_bytes)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ImmutableVerdictResult:
    """Content-addressed worker result with no live mutable payload alias."""

    submission_seq: int
    result_id: str
    payload_bytes: bytes
    result_sha256: str

    @classmethod
    def capture(
        cls,
        *,
        submission_seq: int,
        result_id: str,
        payload: Mapping[str, Any],
    ) -> ImmutableVerdictResult:
        sequence = _validate_nonnegative_int(submission_seq, name="submission_seq")
        if type(result_id) is not str or not result_id:
            raise ValueError("result_id must have exact non-empty str type")
        if any(character.isspace() for character in result_id):
            raise ValueError("result_id must not contain whitespace")
        if not isinstance(payload, Mapping):
            raise TypeError(f"payload must be a mapping, got {type(payload).__name__}")
        payload_bytes = _canonical_payload_bytes(payload)
        return cls(
            submission_seq=sequence,
            result_id=result_id,
            payload_bytes=payload_bytes,
            result_sha256=_result_sha256(
                submission_seq=sequence,
                result_id=result_id,
                payload_bytes=payload_bytes,
            ),
        )

    def validate(self) -> None:
        try:
            sequence = _validate_nonnegative_int(
                self.submission_seq,
                name="submission_seq",
            )
        except (TypeError, ValueError) as exc:
            raise ResultIntegrityError("submission_seq is not a nonnegative integer") from exc
        if type(self.result_id) is not str or not self.result_id:
            raise ResultIntegrityError("result_id must have exact non-empty str type")
        if any(character.isspace() for character in self.result_id):
            raise ResultIntegrityError("result_id must not contain whitespace")
        if not isinstance(self.payload_bytes, bytes):
            raise ResultIntegrityError("payload_bytes must be immutable bytes")
        if type(self.result_sha256) is not str:
            raise ResultIntegrityError("result_sha256 must have exact str type")
        try:
            decoded = json.loads(self.payload_bytes.decode("utf-8"))
            payload = _decode_payload_value(decoded)
            if not isinstance(payload, dict):
                raise ResultIntegrityError("canonical verdict result payload must be a mapping")
            if _canonical_payload_bytes(payload) != self.payload_bytes:
                raise ResultIntegrityError("verdict result payload is not in canonical form")
        except ResultIntegrityError:
            raise
        except (
            AttributeError,
            KeyError,
            OverflowError,
            RecursionError,
            TypeError,
            ValueError,
            UnicodeError,
            json.JSONDecodeError,
        ) as exc:
            raise ResultIntegrityError("canonical verdict result payload is invalid") from exc
        expected = _result_sha256(
            submission_seq=sequence,
            result_id=self.result_id,
            payload_bytes=self.payload_bytes,
        )
        if self.result_sha256 != expected:
            raise ResultIntegrityError(f"verdict result SHA mismatch: {self.result_sha256!r} != {expected!r}")

    @property
    def payload(self) -> dict[str, Any]:
        """Return a detached mutable decoding; the result itself remains immutable."""

        self.validate()
        decoded = _decode_payload_value(json.loads(self.payload_bytes.decode("utf-8")))
        if not isinstance(decoded, dict):  # guarded by validate; keeps the type contract explicit.
            raise ResultIntegrityError("canonical verdict result payload must be a mapping")
        return decoded


@dataclass(frozen=True, slots=True)
class AppliedVerdictRow:
    """Bounded exactly-once journal identity."""

    submission_seq: int
    result_id: str
    result_sha256: str

    def validate(self) -> None:
        try:
            _validate_nonnegative_int(self.submission_seq, name="submission_seq")
        except (TypeError, ValueError) as exc:
            raise CanonicalStateError("applied submission sequence must be a nonnegative integer") from exc
        if type(self.result_id) is not str or not self.result_id:
            raise CanonicalStateError("applied result ID must have exact non-empty str type")
        if any(character.isspace() for character in self.result_id):
            raise CanonicalStateError("applied result ID must not contain whitespace")
        if type(self.result_sha256) is not str:
            raise CanonicalStateError("applied result SHA-256 must have exact str type")
        if len(self.result_sha256) != SHA256_HEX_LENGTH:
            raise CanonicalStateError("applied result SHA-256 must have 64 hex characters")
        try:
            decoded_sha256 = bytes.fromhex(self.result_sha256)
        except ValueError as exc:
            raise CanonicalStateError("applied result SHA-256 is not hexadecimal") from exc
        if decoded_sha256.hex() != self.result_sha256:
            raise CanonicalStateError("applied result SHA-256 must use canonical lowercase hexadecimal")


Reducer = Callable[[Any, ImmutableVerdictResult], Any]
Worker = Callable[[int, Mapping[str, Any]], ImmutableVerdictResult]
ExternalEffectTransition = Callable[[Any], Any]
ReducerStateValidator = Callable[[Any], None]


class DeterministicVerdictReducer:
    """Pure replacement-state reducer with an exact monotone apply cursor."""

    def __init__(
        self,
        *,
        reducer: Reducer,
        initial_state: Any,
        max_journal_rows: int,
    ) -> None:
        if not callable(reducer):
            raise TypeError("reducer must be callable")
        limit = _validate_nonnegative_int(
            max_journal_rows,
            name="max_journal_rows",
        )
        if limit < 1:
            raise ValueError("max_journal_rows must be at least one")
        self._reducer = reducer
        self._state = _clone_numpy_tree(initial_state)
        self._max_journal_rows = limit
        self._next_apply_seq = 0
        self._journal: list[AppliedVerdictRow] = []
        self._lock = threading.RLock()
        self._poisoned_error: FatalVerdictTransactionError | None = None
        self._application_active = False
        self._external_transition_active = False

    @classmethod
    def restore(
        cls,
        *,
        reducer: Reducer,
        restored_state: Any,
        max_journal_rows: int,
        next_apply_seq: int,
        journal: tuple[AppliedVerdictRow, ...],
    ) -> DeterministicVerdictReducer:
        instance = cls(
            reducer=reducer,
            initial_state=restored_state,
            max_journal_rows=max_journal_rows,
        )
        instance._next_apply_seq = _validate_nonnegative_int(
            next_apply_seq,
            name="next_apply_seq",
        )
        instance._journal = list(journal)
        instance._validate_journal()
        return instance

    @property
    def next_apply_seq(self) -> int:
        with self._lock:
            return self._next_apply_seq

    @property
    def state(self) -> Any:
        with self._lock:
            return _clone_numpy_tree(self._state)

    @property
    def journal(self) -> tuple[AppliedVerdictRow, ...]:
        with self._lock:
            return tuple(self._journal)

    @property
    def max_journal_rows(self) -> int:
        return self._max_journal_rows

    @property
    def poisoned(self) -> bool:
        with self._lock:
            return self._poisoned_error is not None

    def assert_healthy(self) -> None:
        with self._lock:
            if self._poisoned_error is not None:
                raise TransactionPoisonedError(
                    "verdict reducer is poisoned by a prior application failure"
                ) from self._poisoned_error

    def _validate_journal(self) -> None:
        if len(self._journal) > self._max_journal_rows:
            raise CanonicalStateError("applied verdict journal exceeds its bound")
        expected_start = self._next_apply_seq - len(self._journal)
        if expected_start < 0:
            raise CanonicalStateError("applied verdict journal is longer than the apply cursor")
        seen_ids: set[str] = set()
        for offset, row in enumerate(self._journal):
            row.validate()
            expected_seq = expected_start + offset
            if row.submission_seq != expected_seq:
                raise CanonicalStateError(f"journal sequence {row.submission_seq} != expected {expected_seq}")
            if row.result_id in seen_ids:
                raise CanonicalStateError(f"duplicate applied result ID in journal: {row.result_id!r}")
            seen_ids.add(row.result_id)

    def apply(self, result: ImmutableVerdictResult) -> None:
        """Apply one exact next result, publishing state only after success."""

        with self._lock:
            self.assert_healthy()
            if self._application_active or self._external_transition_active:
                failure = ExternalEffectTransitionError("reducer application re-entry is forbidden")
                self._poisoned_error = failure
                raise failure
            try:
                if not isinstance(result, ImmutableVerdictResult):
                    raise WorkerResultTypeError(
                        f"worker must return ImmutableVerdictResult, got {type(result).__name__}"
                    )
                result.validate()
                if result.submission_seq < self._next_apply_seq:
                    raise DuplicateSequenceError(
                        f"result sequence {result.submission_seq} is before next_apply_seq {self._next_apply_seq}"
                    )
                if result.submission_seq > self._next_apply_seq:
                    raise SequenceGapError(
                        f"result sequence {result.submission_seq} leaves a gap "
                        f"before next_apply_seq {self._next_apply_seq}"
                    )
                if any(row.result_id == result.result_id for row in self._journal):
                    raise DuplicateResultError(f"result ID {result.result_id!r} already applied")
            except FatalVerdictTransactionError as exc:
                self._poisoned_error = exc
                raise

            candidate = _clone_numpy_tree(self._state)
            self._application_active = True
            try:
                replacement = self._reducer(candidate, result)
                replacement = _clone_numpy_tree(replacement)
                self.assert_healthy()
            except BaseException as exc:
                failure = ReducerApplicationError(f"reducer failed for submission sequence {result.submission_seq}")
                self._poisoned_error = failure
                raise failure from exc
            finally:
                self._application_active = False

            self._state = replacement
            self._next_apply_seq += 1
            self._journal.append(
                AppliedVerdictRow(
                    submission_seq=result.submission_seq,
                    result_id=result.result_id,
                    result_sha256=result.result_sha256,
                )
            )
            if len(self._journal) > self._max_journal_rows:
                del self._journal[: len(self._journal) - self._max_journal_rows]
            self._validate_journal()

    def apply_external_effect_transition(
        self,
        *,
        transition: ExternalEffectTransition,
        validate_replacement: ReducerStateValidator,
    ) -> Any:
        """Publish effects and atomically replace state with their acknowledgement.

        ``transition`` receives a detached candidate state.  It may perform
        deterministic, idempotent external effects and must return the complete
        acknowledged replacement state.  The validator is mandatory and runs
        both before any effect and after the returned state has been detached.

        A failure at any point keeps the pre-transition reducer state and
        permanently poisons the reducer.  Physical effects that happened before
        a failure may replay only after restart from a complete checkpoint, so
        callers must key them by the result identity carried in reducer state.
        """

        if not callable(transition):
            raise TypeError("external-effect transition must be callable")
        if not callable(validate_replacement):
            raise TypeError("replacement-state validator must be callable")
        with self._lock:
            self.assert_healthy()
            if self._application_active or self._external_transition_active:
                failure = ExternalEffectTransitionError("external-effect transition re-entry is forbidden")
                self._poisoned_error = failure
                raise failure
            candidate = _clone_numpy_tree(self._state)
            self._external_transition_active = True
            try:
                validate_replacement(_clone_numpy_tree(candidate))
                replacement = transition(candidate)
                replacement = _clone_numpy_tree(replacement)
                validate_replacement(_clone_numpy_tree(replacement))
                self.assert_healthy()
            except BaseException as exc:
                failure = ExternalEffectTransitionError("external-effect publication/acknowledgement transition failed")
                self._poisoned_error = failure
                raise failure from exc
            finally:
                self._external_transition_active = False
            self._state = replacement
            return _clone_numpy_tree(self._state)


def _utf8_array(value: str) -> np.ndarray:
    return _readonly_array(np.frombuffer(value.encode("utf-8"), dtype=np.uint8))


def _decode_utf8_array(value: np.ndarray, *, name: str) -> str:
    array = np.asarray(value)
    if array.dtype != np.dtype(np.uint8) or array.ndim != 1:
        raise CanonicalStateError(f"{name} must be a one-dimensional uint8 array")
    try:
        return array.tobytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CanonicalStateError(f"{name} is not valid UTF-8") from exc


def _int64_scalar(value: int) -> np.ndarray:
    return _readonly_array(np.asarray([value], dtype=np.int64))


def _read_int64_scalar(value: np.ndarray, *, name: str) -> int:
    array = np.asarray(value)
    if array.dtype != np.dtype(np.int64) or array.shape != (1,):
        raise CanonicalStateError(f"{name} must have dtype int64 and shape (1,)")
    return _validate_nonnegative_int(array[0], name=name)


def _pack_result_ids(rows: tuple[AppliedVerdictRow, ...]) -> tuple[np.ndarray, np.ndarray]:
    encoded = [row.result_id.encode("utf-8") for row in rows]
    offsets = np.zeros(len(encoded) + 1, dtype=np.int64)
    for index, item in enumerate(encoded, start=1):
        offsets[index] = offsets[index - 1] + len(item)
    data = np.frombuffer(b"".join(encoded), dtype=np.uint8).copy()
    return _readonly_array(data), _readonly_array(offsets)


def _unpack_result_ids(data: np.ndarray, offsets: np.ndarray) -> tuple[str, ...]:
    data_array = np.asarray(data)
    offsets_array = np.asarray(offsets)
    if data_array.dtype != np.dtype(np.uint8) or data_array.ndim != 1:
        raise CanonicalStateError("journal_result_id_data must be a one-dimensional uint8 array")
    if offsets_array.dtype != np.dtype(np.int64) or offsets_array.ndim != 1:
        raise CanonicalStateError("journal_result_id_offsets must be a one-dimensional int64 array")
    if offsets_array.size < 1 or int(offsets_array[0]) != 0:
        raise CanonicalStateError("journal result-ID offsets must start at zero")
    if np.any(np.diff(offsets_array) < 0):
        raise CanonicalStateError("journal result-ID offsets must be monotone")
    if int(offsets_array[-1]) != int(data_array.size):
        raise CanonicalStateError("journal result-ID offsets do not close over the data array")
    result: list[str] = []
    raw = data_array.tobytes()
    for start, stop in pairwise(offsets_array):
        try:
            value = raw[int(start) : int(stop)].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise CanonicalStateError("journal result ID is not valid UTF-8") from exc
        if not value:
            raise CanonicalStateError("journal result IDs must be non-empty")
        result.append(value)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class VerdictCheckpointCapture:
    """Detached state yielded while the logical checkpoint barrier remains active."""

    next_submit_seq: int
    next_apply_seq: int
    pending_count: int
    last_applied_result_id: str | None
    last_applied_result_sha256: str | None
    journal_limit: int
    journal: tuple[AppliedVerdictRow, ...]
    _reducer_state: Any

    @property
    def reducer_state(self) -> Any:
        return _clone_numpy_tree(self._reducer_state)

    def _validate(self) -> None:
        try:
            next_submit_seq = _validate_nonnegative_int(
                self.next_submit_seq,
                name="next_submit_seq",
            )
            next_apply_seq = _validate_nonnegative_int(
                self.next_apply_seq,
                name="next_apply_seq",
            )
            pending_count = _validate_nonnegative_int(
                self.pending_count,
                name="pending_count",
            )
            journal_limit = _validate_nonnegative_int(
                self.journal_limit,
                name="journal_limit",
            )
        except (TypeError, ValueError) as exc:
            raise CanonicalStateError("checkpoint capture cursors and limits must be nonnegative integers") from exc
        if pending_count != 0 or next_submit_seq != next_apply_seq:
            raise NonQuiescentCheckpointError("checkpoint capture requires pending_count=0 and equal cursors")
        if journal_limit < 1:
            raise CanonicalStateError("journal_limit must be at least one")
        if not isinstance(self.journal, tuple):
            raise CanonicalStateError("checkpoint journal must be an immutable tuple")
        if len(self.journal) != min(next_apply_seq, journal_limit):
            raise CanonicalStateError("bounded journal length does not equal min(next_apply_seq, journal_limit)")
        expected_start = next_apply_seq - len(self.journal)
        for offset, row in enumerate(self.journal):
            if not isinstance(row, AppliedVerdictRow):
                raise CanonicalStateError("checkpoint journal rows must be AppliedVerdictRow values")
            row.validate()
            if row.submission_seq != expected_start + offset:
                raise CanonicalStateError("checkpoint journal sequences are not a contiguous cursor suffix")
        if self.last_applied_result_id is not None and type(self.last_applied_result_id) is not str:
            raise CanonicalStateError("last-applied result ID must have exact str type when present")
        if self.last_applied_result_sha256 is not None and type(self.last_applied_result_sha256) is not str:
            raise CanonicalStateError("last-applied result SHA-256 must have exact str type when present")
        expected_id = self.journal[-1].result_id if self.journal else None
        expected_sha = self.journal[-1].result_sha256 if self.journal else None
        if self.last_applied_result_id != expected_id:
            raise CanonicalStateError("last-applied result ID does not equal the bounded journal tail")
        if self.last_applied_result_sha256 != expected_sha:
            raise CanonicalStateError("last-applied result SHA-256 does not equal the bounded journal tail")

    def numpy_state(self, *, prefix: str = "") -> Mapping[str, np.ndarray]:
        """Return eleven immutable native-v3 arrays.

        Six fields bind directly through ``BarrierStateBinding``; five more
        carry the bounded applied-result journal and its configured limit.
        """

        if not isinstance(prefix, str) or prefix.strip() != prefix:
            raise CanonicalStateError(f"barrier prefix must be a canonical string, got {prefix!r}")
        self._validate()
        id_data, id_offsets = _pack_result_ids(self.journal)
        if self.journal:
            sha_rows = np.stack(
                [np.frombuffer(bytes.fromhex(row.result_sha256), dtype=np.uint8) for row in self.journal],
                axis=0,
            )
        else:
            sha_rows = np.empty((0, 32), dtype=np.uint8)
        arrays = {
            f"{prefix}schema": _utf8_array(SCHEMA),
            f"{prefix}next_submit_seq": _int64_scalar(self.next_submit_seq),
            f"{prefix}next_apply_seq": _int64_scalar(self.next_apply_seq),
            f"{prefix}pending_count": _int64_scalar(self.pending_count),
            f"{prefix}last_applied_result_id": _utf8_array(self.last_applied_result_id or ""),
            f"{prefix}last_applied_result_sha256": _utf8_array(self.last_applied_result_sha256 or ""),
            f"{prefix}journal_limit": _int64_scalar(self.journal_limit),
            f"{prefix}journal_sequences": _readonly_array(
                np.asarray(
                    [row.submission_seq for row in self.journal],
                    dtype=np.int64,
                )
            ),
            f"{prefix}journal_result_id_data": id_data,
            f"{prefix}journal_result_id_offsets": id_offsets,
            f"{prefix}journal_result_sha256": _readonly_array(sha_rows),
        }
        return MappingProxyType(arrays)


class QuiescentVerdictTransaction:
    """Own monotonically sequenced submissions and strict checkpoint draining."""

    def __init__(
        self,
        *,
        reducer: Reducer,
        initial_state: Any,
        max_journal_rows: int = 64,
    ) -> None:
        self._main_thread_ident = threading.get_ident()
        self._gate = threading.Condition(threading.RLock())
        self._checkpoint_active = False
        self._external_transition_active = False
        self._next_submit_seq = 0
        self._pending: dict[int, Future[ImmutableVerdictResult]] = {}
        self._reducer = DeterministicVerdictReducer(
            reducer=reducer,
            initial_state=initial_state,
            max_journal_rows=max_journal_rows,
        )
        self._fatal_error: FatalVerdictTransactionError | None = None
        self._unobserved_async_fatal: FatalVerdictTransactionError | None = None

    @classmethod
    def from_numpy_state(
        cls,
        arrays: Mapping[str, np.ndarray],
        *,
        reducer: Reducer,
        restored_reducer_state: Any,
        prefix: str = "",
    ) -> QuiescentVerdictTransaction:
        """Open one strict quiescent native-v3 state.

        The caller restores O3/controller state separately and passes its
        detached staged value as ``restored_reducer_state``.
        """

        if not isinstance(prefix, str) or prefix.strip() != prefix:
            raise CanonicalStateError(f"barrier prefix must be a canonical string, got {prefix!r}")
        for key in arrays:
            if not isinstance(key, str):
                raise CanonicalStateError(f"canonical verdict barrier key must be a string, got {key!r}")
            if LEGACY_PENDING_PREFIX in key:
                raise NativeV3PendingPayloadError(f"native-v3 forbids legacy pending payload key {key!r}")
        required = {f"{prefix}{field}" for field in _STATE_FIELDS}
        missing = required - set(arrays)
        if missing:
            raise CanonicalStateError(f"canonical verdict barrier state is missing keys {sorted(missing)!r}")
        schema = _decode_utf8_array(
            arrays[f"{prefix}schema"],
            name=f"{prefix}schema",
        )
        if schema != SCHEMA:
            raise CanonicalStateError(f"verdict barrier schema {schema!r} != {SCHEMA!r}")
        next_submit_seq = _read_int64_scalar(
            arrays[f"{prefix}next_submit_seq"],
            name=f"{prefix}next_submit_seq",
        )
        next_apply_seq = _read_int64_scalar(
            arrays[f"{prefix}next_apply_seq"],
            name=f"{prefix}next_apply_seq",
        )
        pending_count = _read_int64_scalar(
            arrays[f"{prefix}pending_count"],
            name=f"{prefix}pending_count",
        )
        if pending_count != 0 or next_submit_seq != next_apply_seq:
            raise NonQuiescentCheckpointError(
                "native-v3 restore requires pending_count=0 and equal submit/apply cursors"
            )
        journal_limit = _read_int64_scalar(
            arrays[f"{prefix}journal_limit"],
            name=f"{prefix}journal_limit",
        )
        if journal_limit < 1:
            raise CanonicalStateError("journal_limit must be at least one")
        sequences = np.asarray(arrays[f"{prefix}journal_sequences"])
        if sequences.dtype != np.dtype(np.int64) or sequences.ndim != 1:
            raise CanonicalStateError("journal_sequences must be a one-dimensional int64 array")
        result_ids = _unpack_result_ids(
            arrays[f"{prefix}journal_result_id_data"],
            arrays[f"{prefix}journal_result_id_offsets"],
        )
        hashes = np.asarray(arrays[f"{prefix}journal_result_sha256"])
        if hashes.dtype != np.dtype(np.uint8) or hashes.shape != (len(sequences), 32):
            raise CanonicalStateError("journal_result_sha256 must have dtype uint8 and shape (N, 32)")
        if len(result_ids) != len(sequences):
            raise CanonicalStateError("journal result-ID count does not equal sequence count")
        journal = tuple(
            AppliedVerdictRow(
                submission_seq=int(sequence),
                result_id=result_ids[index],
                result_sha256=hashes[index].tobytes().hex(),
            )
            for index, sequence in enumerate(sequences)
        )
        if len(journal) != min(next_apply_seq, journal_limit):
            raise CanonicalStateError("bounded journal length does not equal min(next_apply_seq, journal_limit)")
        last_id = _decode_utf8_array(
            arrays[f"{prefix}last_applied_result_id"],
            name=f"{prefix}last_applied_result_id",
        )
        last_sha = _decode_utf8_array(
            arrays[f"{prefix}last_applied_result_sha256"],
            name=f"{prefix}last_applied_result_sha256",
        )
        expected_last_id = journal[-1].result_id if journal else ""
        expected_last_sha = journal[-1].result_sha256 if journal else ""
        if last_id != expected_last_id or last_sha != expected_last_sha:
            raise CanonicalStateError("last-applied identity does not equal the bounded journal tail")

        instance = cls.__new__(cls)
        instance._main_thread_ident = threading.get_ident()
        instance._gate = threading.Condition(threading.RLock())
        instance._checkpoint_active = False
        instance._external_transition_active = False
        instance._next_submit_seq = next_submit_seq
        instance._pending = {}
        instance._reducer = DeterministicVerdictReducer.restore(
            reducer=reducer,
            restored_state=restored_reducer_state,
            max_journal_rows=journal_limit,
            next_apply_seq=next_apply_seq,
            journal=journal,
        )
        instance._fatal_error = None
        instance._unobserved_async_fatal = None
        return instance

    def _assert_main_thread(self) -> None:
        if threading.get_ident() != self._main_thread_ident:
            raise MainThreadViolationError("verdict reduction/checkpoint operations must run on the creating thread")

    def _assert_healthy(self) -> None:
        with self._gate:
            fatal_error = self._fatal_error
            raise_original = fatal_error is not None and self._unobserved_async_fatal is fatal_error
            if raise_original:
                self._unobserved_async_fatal = None
            futures = tuple(self._pending.values()) if fatal_error is not None else ()
            if fatal_error is not None:
                self._pending.clear()
        for future in futures:
            future.cancel()
        if fatal_error is not None:
            if raise_original:
                raise fatal_error
            raise TransactionPoisonedError(
                "verdict transaction is fatally poisoned; restart from the last complete checkpoint"
            ) from fatal_error
        self._reducer.assert_healthy()

    def assert_healthy(self) -> None:
        """Thread-safe fail-fast gate for state mutations outside this object.

        A worker cancellation/exception is latched by its future callback before
        the main thread next polls or reduces results.  Call this immediately
        before any optimizer/controller mutation that is conditional on the
        verdict transaction remaining healthy.
        """

        self._assert_healthy()

    def _poison(
        self,
        error: FatalVerdictTransactionError,
        *,
        cause: BaseException | None = None,
    ) -> FatalVerdictTransactionError:
        """Latch the first fatal error and discard all unusable pending futures."""

        if cause is not None and error.__cause__ is None:
            error.__cause__ = cause
        with self._gate:
            if self._fatal_error is None:
                self._fatal_error = error
            canonical_error = self._fatal_error
            if self._unobserved_async_fatal is canonical_error:
                self._unobserved_async_fatal = None
            futures = tuple(self._pending.values())
            self._pending.clear()
        for future in futures:
            future.cancel()
        return canonical_error

    def _latch_async_worker_failure(
        self,
        error: FatalVerdictTransactionError,
        *,
        cause: BaseException,
    ) -> None:
        """Latch a callback-observed worker fatal without reducing any result."""

        if error.__cause__ is None:
            error.__cause__ = cause
        with self._gate:
            if self._fatal_error is None:
                self._fatal_error = error
                self._unobserved_async_fatal = error
            futures = tuple(self._pending.values())
            self._pending.clear()
        for future in futures:
            if not future.done():
                future.cancel()

    def _observe_worker_completion(
        self,
        sequence: int,
        future: Future[ImmutableVerdictResult],
    ) -> None:
        """Future callback: eagerly poison only cancellation or execution error."""

        if future.cancelled():
            cause: BaseException = CancelledError()
            error: FatalVerdictTransactionError = WorkerCancelledError(
                f"verdict worker sequence {sequence} was cancelled"
            )
            self._latch_async_worker_failure(error, cause=cause)
            return
        try:
            worker_error = future.exception()
        except CancelledError as exc:
            error = WorkerCancelledError(f"verdict worker sequence {sequence} was cancelled")
            self._latch_async_worker_failure(error, cause=exc)
            return
        except BaseException as exc:
            error = WorkerExecutionError(
                f"verdict worker sequence {sequence} completion inspection raised {type(exc).__name__}"
            )
            self._latch_async_worker_failure(error, cause=exc)
            return
        if worker_error is None:
            return
        error = WorkerExecutionError(f"verdict worker sequence {sequence} raised {type(worker_error).__name__}")
        self._latch_async_worker_failure(error, cause=worker_error)

    @property
    def checkpoint_active(self) -> bool:
        with self._gate:
            return self._checkpoint_active

    @property
    def next_submit_seq(self) -> int:
        with self._gate:
            return self._next_submit_seq

    @property
    def next_apply_seq(self) -> int:
        return self._reducer.next_apply_seq

    @property
    def pending_count(self) -> int:
        with self._gate:
            return len(self._pending)

    @property
    def reducer_state(self) -> Any:
        return self._reducer.state

    @property
    def journal(self) -> tuple[AppliedVerdictRow, ...]:
        return self._reducer.journal

    @property
    def poisoned(self) -> bool:
        with self._gate:
            transaction_poisoned = self._fatal_error is not None
        return transaction_poisoned or self._reducer.poisoned

    @property
    def fatal_error(self) -> FatalVerdictTransactionError | None:
        """Return the latched typed failure for diagnostics; never clears it."""

        with self._gate:
            return self._fatal_error

    def submit(
        self,
        executor: Executor,
        worker: Worker,
        snapshot: Mapping[str, Any],
    ) -> int:
        """Capture ``snapshot`` and submit one sequence-owned immutable worker."""

        with self._gate:
            # Check the checkpoint gate before thread identity so an accidental
            # worker-side submission gets the precise barrier refusal.
            if self._checkpoint_active:
                raise SubmissionBlockedError("new verdict submissions are blocked by the checkpoint barrier")
            if self._external_transition_active:
                raise SubmissionBlockedError("new verdict submissions are blocked by the external-effect transition")
            self._assert_main_thread()
            self._assert_healthy()
            if not callable(worker):
                raise TypeError("worker must be callable")
            frozen_snapshot = _freeze_numpy_tree(snapshot)
            if not isinstance(frozen_snapshot, Mapping):
                raise TypeError("worker snapshot must be a mapping")
            sequence = self._next_submit_seq
            future = executor.submit(worker, sequence, frozen_snapshot)
            self._pending[sequence] = future
            self._next_submit_seq += 1
            future.add_done_callback(
                lambda completed, owned_sequence=sequence: self._observe_worker_completion(
                    owned_sequence,
                    completed,
                )
            )
            return sequence

    def _future_for_next_apply(self) -> Future[ImmutableVerdictResult] | None:
        with self._gate:
            sequence = self._reducer.next_apply_seq
            if sequence == self._next_submit_seq:
                return None
            future = self._pending.get(sequence)
            if future is None:
                error = SequenceGapError(f"pending future for exact next sequence {sequence} is absent")
                raise self._poison(error)
            return future

    def _apply_future(
        self,
        future: Future[ImmutableVerdictResult],
    ) -> None:
        expected_sequence = self._reducer.next_apply_seq
        with self._gate:
            current = self._pending.get(expected_sequence)
            if current is not future:
                error = SequenceGapError(f"pending future identity drift at sequence {expected_sequence}")
                raise self._poison(error)
        try:
            result = future.result()
        except CancelledError as exc:
            error = WorkerCancelledError(f"verdict worker sequence {expected_sequence} was cancelled")
            canonical_error = self._poison(error, cause=exc)
            raise canonical_error from canonical_error.__cause__
        except BaseException as exc:
            error = WorkerExecutionError(f"verdict worker sequence {expected_sequence} raised {type(exc).__name__}")
            canonical_error = self._poison(error, cause=exc)
            raise canonical_error from canonical_error.__cause__
        with self._gate:
            self._assert_healthy()
            try:
                if not isinstance(result, ImmutableVerdictResult):
                    raise WorkerResultTypeError(
                        f"worker must return ImmutableVerdictResult, got {type(result).__name__}"
                    )
                self._reducer.apply(result)
            except FatalVerdictTransactionError as exc:
                canonical_error = self._poison(exc)
                raise canonical_error from canonical_error.__cause__
            except Exception as exc:
                error = ReducerApplicationError(
                    f"unexpected reducer failure for submission sequence {expected_sequence}"
                )
                canonical_error = self._poison(error, cause=exc)
                raise canonical_error from canonical_error.__cause__
            current = self._pending.get(expected_sequence)
            if current is not future:
                error = SequenceGapError(
                    f"pending future identity drift after reduction at sequence {expected_sequence}"
                )
                raise self._poison(error)
            del self._pending[expected_sequence]
        self._assert_healthy()

    def apply_completed(self) -> int:
        """Apply the contiguous completed prefix without waiting for workers."""

        self._assert_main_thread()
        self._assert_healthy()
        with self._gate:
            if self._checkpoint_active:
                raise SubmissionBlockedError("standalone reduction is blocked while checkpointing")
            if self._external_transition_active:
                raise SubmissionBlockedError("standalone reduction is blocked during the external-effect transition")
        applied = 0
        while True:
            future = self._future_for_next_apply()
            if future is None or not future.done():
                break
            self._apply_future(future)
            applied += 1
        return applied

    def _drain_all_in_order(self) -> None:
        while True:
            future = self._future_for_next_apply()
            if future is None:
                return
            self._apply_future(future)

    def _capture(self) -> VerdictCheckpointCapture:
        with self._gate:
            pending_count = len(self._pending)
            next_submit_seq = self._next_submit_seq
        next_apply_seq = self._reducer.next_apply_seq
        if pending_count != 0 or next_submit_seq != next_apply_seq:
            raise NonQuiescentCheckpointError(
                "strict checkpoint requires pending_count=0 and next_submit_seq==next_apply_seq"
            )
        journal = self._reducer.journal
        last = journal[-1] if journal else None
        return VerdictCheckpointCapture(
            next_submit_seq=next_submit_seq,
            next_apply_seq=next_apply_seq,
            pending_count=pending_count,
            last_applied_result_id=(last.result_id if last is not None else None),
            last_applied_result_sha256=(last.result_sha256 if last is not None else None),
            journal_limit=self._reducer.max_journal_rows,
            journal=journal,
            _reducer_state=self._reducer.state,
        )

    def _apply_external_effect_transition(
        self,
        *,
        transition: ExternalEffectTransition,
        validate_replacement: ReducerStateValidator,
        checkpoint_owned: bool,
    ) -> Any:
        """Run one non-reentrant main-thread effect/ack transition."""

        self._assert_main_thread()
        self._assert_healthy()
        if not callable(transition):
            raise TypeError("external-effect transition must be callable")
        if not callable(validate_replacement):
            raise TypeError("replacement-state validator must be callable")
        with self._gate:
            if self._external_transition_active:
                raise NonQuiescentCheckpointError("an external-effect transition is already active")
            if checkpoint_owned:
                if not self._checkpoint_active:
                    raise NonQuiescentCheckpointError(
                        "checkpoint-owned external-effect transition lacks an active barrier"
                    )
            elif self._checkpoint_active:
                raise SubmissionBlockedError("standalone external-effect transition is blocked while checkpointing")
            self._external_transition_active = True
        try:
            try:
                replacement = self._reducer.apply_external_effect_transition(
                    transition=transition,
                    validate_replacement=validate_replacement,
                )
            except FatalVerdictTransactionError as exc:
                canonical_error = self._poison(exc)
                raise canonical_error from canonical_error.__cause__
            self._assert_healthy()
            return replacement
        finally:
            with self._gate:
                self._external_transition_active = False
                self._gate.notify_all()

    def apply_external_effect_transition(
        self,
        *,
        transition: ExternalEffectTransition,
        validate_replacement: ReducerStateValidator,
    ) -> Any:
        """Reduce the completed prefix, publish its effects, and acknowledge it.

        This is the ordinary main-loop transition.  It never waits for an
        unfinished worker; :meth:`prepared_checkpoint` is the strict drain-all
        variant.  The returned state is detached from transaction ownership.
        """

        self._assert_main_thread()
        self._assert_healthy()
        if not callable(transition):
            raise TypeError("external-effect transition must be callable")
        if not callable(validate_replacement):
            raise TypeError("replacement-state validator must be callable")
        self.apply_completed()
        return self._apply_external_effect_transition(
            transition=transition,
            validate_replacement=validate_replacement,
            checkpoint_owned=False,
        )

    @contextmanager
    def _checkpoint_barrier(self) -> Iterator[None]:
        """Activate and hold the single logical checkpoint barrier."""

        with self._gate:
            if self._checkpoint_active:
                raise NonQuiescentCheckpointError("checkpoint barrier is already active")
            if self._external_transition_active:
                raise NonQuiescentCheckpointError("checkpoint cannot start during an external-effect transition")
            self._checkpoint_active = True
        try:
            yield
        finally:
            with self._gate:
                self._checkpoint_active = False
                self._gate.notify_all()

    def poison_checkpoint_publication(self, cause: BaseException) -> None:
        """Latch and raise a typed fatal caller-publication failure.

        Call this from an exception handler inside a checkpoint publication
        body.  It raises :class:`CheckpointPublicationError`; the surrounding
        context manager then releases the barrier without clearing the poison.
        """

        self._assert_main_thread()
        self._assert_healthy()
        if not isinstance(cause, BaseException):
            raise TypeError("checkpoint publication cause must be an exception")
        with self._gate:
            if not self._checkpoint_active:
                raise NonQuiescentCheckpointError(
                    "checkpoint publication can be poisoned only while its barrier is active"
                )
            if self._external_transition_active:
                raise NonQuiescentCheckpointError("checkpoint publication cannot start during effect preparation")
        error = CheckpointPublicationError("caller checkpoint publication failed while the barrier was active")
        raise self._poison(error, cause=cause)

    @contextmanager
    def checkpoint(self) -> Iterator[VerdictCheckpointCapture]:
        """Join, reduce, and hold the logical barrier through caller publication."""

        self._assert_main_thread()
        self._assert_healthy()
        with self._checkpoint_barrier():
            self._drain_all_in_order()
            yield self._capture()

    @contextmanager
    def prepared_checkpoint(
        self,
        *,
        transition: ExternalEffectTransition,
        validate_replacement: ReducerStateValidator,
    ) -> Iterator[VerdictCheckpointCapture]:
        """Drain, publish/ack effects, then yield one post-ack capture.

        The barrier remains active through the caller's checkpoint writes.
        Any exception from preparation or the caller body is latched as a typed
        fatal error on this transaction.  Callers that catch their own write
        exception may explicitly invoke :meth:`poison_checkpoint_publication`;
        an uncaught body exception is wrapped and poisoned automatically.
        """

        self._assert_main_thread()
        self._assert_healthy()
        if not callable(transition):
            raise TypeError("external-effect transition must be callable")
        if not callable(validate_replacement):
            raise TypeError("replacement-state validator must be callable")
        with self._checkpoint_barrier():
            try:
                self._drain_all_in_order()
                self._apply_external_effect_transition(
                    transition=transition,
                    validate_replacement=validate_replacement,
                    checkpoint_owned=True,
                )
                capture = self._capture()
            except BaseException as exc:
                if self.fatal_error is not None:
                    raise
                error = CheckpointPreparationError("checkpoint preparation failed before post-ack capture")
                canonical_error = self._poison(error, cause=exc)
                raise canonical_error from canonical_error.__cause__
            try:
                yield capture
            except BaseException as exc:
                if self.fatal_error is not None:
                    raise
                error = CheckpointPublicationError("caller checkpoint publication failed while the barrier was active")
                canonical_error = self._poison(error, cause=exc)
                raise canonical_error from canonical_error.__cause__
