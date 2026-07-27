# SPDX-License-Identifier: MIT
"""Typed post-checkpoint G111 verdict obligation.

Native-v3 checkpoints are strictly quiescent and never serialize an in-flight
worker or scorer snapshot.  At an evaluation/checkpoint collision, the trainer
instead commits one fixed-size obligation derived from the checkpoint
coordinate and immutable configuration:

1. finish epoch ``e`` and decide using only earlier verdicts;
2. arm and serialize the obligation while the verdict transaction is
   quiescent;
3. atomically publish checkpoint ``e``;
4. reconstruct the worker snapshot from the restored O1-O5 state and dispatch
   verdict ``e``;
5. forbid the next optimizer step until the obligation is discharged.

After a crash, reopening checkpoint ``e`` recreates the same obligation and the
same transaction cursors.  The scorer snapshot is reconstructed, never stored.
If dispatch becomes indeterminate in-process, the state machine is poisoned and
requires restart from the last complete checkpoint.

This module is trainer-independent.  It writes no files, starts no workers, and
does not authorize a G111 launch.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

import numpy as np

SCHEMA: Final = "tac.g111_post_checkpoint_verdict_obligation.v1"
WORKER_SNAPSHOT_SCHEMA: Final = "tac.g111_live_verdict_snapshot.v1"
SHA256_HEX_LENGTH: Final = 64
MAX_STAGE_UTF8_BYTES: Final = 256
MAX_BOUNDARY_KIND_UTF8_BYTES: Final = 128

_STATE_SUFFIXES: Final = frozenset(
    {
        "schema",
        "present",
        "checkpoint_epoch",
        "submission_seq",
        "stage",
        "boundary_kind",
        "config_sha256",
        "obligation_id",
    }
)
_OBLIGATION_ID_DOMAIN: Final = b"tac.g111.post-checkpoint-verdict-obligation.v1\0"


class VerdictObligationError(RuntimeError):
    """The post-checkpoint verdict obligation failed closed."""


class MalformedVerdictObligationError(VerdictObligationError):
    """Serialized or reconstructed obligation state is malformed."""


class DuplicateVerdictObligationError(VerdictObligationError):
    """An obligation was armed or discharged more than once in-process."""


class StaleVerdictObligationError(VerdictObligationError):
    """An obligation does not match the restored coordinate or cursors."""


class VerdictObligationOwedError(VerdictObligationError):
    """An optimizer step was attempted before the obligation was discharged."""


class VerdictObligationDispatchError(VerdictObligationError):
    """Snapshot reconstruction or dispatch failed and poisoned this process."""


class VerdictObligationPoisonedError(VerdictObligationError):
    """A prior indeterminate dispatch requires restart from checkpoint."""


class FinalCheckpointNotReadyError(VerdictObligationError):
    """The final verdict was not joined before the final checkpoint."""


def _exact_nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise MalformedVerdictObligationError(f"{name} must be an exact integer")
    result = int(value)
    if result < 0:
        raise MalformedVerdictObligationError(f"{name} must be nonnegative")
    return result


def _exact_text(
    value: object,
    *,
    name: str,
    max_utf8_bytes: int,
    allow_empty: bool = False,
) -> str:
    if type(value) is not str:
        raise MalformedVerdictObligationError(f"{name} must have exact str type")
    encoded = value.encode("utf-8")
    if not allow_empty and not encoded:
        raise MalformedVerdictObligationError(f"{name} must be non-empty")
    if len(encoded) > max_utf8_bytes:
        raise MalformedVerdictObligationError(
            f"{name} exceeds {max_utf8_bytes} UTF-8 bytes"
        )
    return value


def _sha256_text(value: object, *, name: str, allow_empty: bool = False) -> str:
    text = _exact_text(
        value,
        name=name,
        max_utf8_bytes=SHA256_HEX_LENGTH,
        allow_empty=allow_empty,
    )
    if allow_empty and not text:
        return text
    if len(text) != SHA256_HEX_LENGTH or text != text.lower():
        raise MalformedVerdictObligationError(
            f"{name} must be 64 lowercase hexadecimal characters"
        )
    try:
        bytes.fromhex(text)
    except ValueError as exc:
        raise MalformedVerdictObligationError(
            f"{name} must be 64 lowercase hexadecimal characters"
        ) from exc
    return text


def _utf8_array(value: str) -> np.ndarray:
    return np.frombuffer(value.encode("utf-8"), dtype=np.uint8).copy()


def _decode_utf8_array(value: object, *, name: str) -> str:
    array = np.asarray(value)
    if array.dtype != np.dtype(np.uint8) or array.ndim != 1:
        raise MalformedVerdictObligationError(
            f"{name} must be a one-dimensional uint8 array"
        )
    try:
        return array.tobytes().decode("utf-8")
    except UnicodeError as exc:
        raise MalformedVerdictObligationError(
            f"{name} is not valid UTF-8"
        ) from exc


def _int64_scalar(value: int) -> np.ndarray:
    return np.asarray(value, dtype=np.int64)


def _read_int64_scalar(value: object, *, name: str) -> int:
    array = np.asarray(value)
    if array.dtype != np.dtype(np.int64) or array.shape != ():
        raise MalformedVerdictObligationError(
            f"{name} must be an int64 scalar"
        )
    return int(array.item())


def _uint8_scalar(value: int) -> np.ndarray:
    return np.asarray(value, dtype=np.uint8)


def _read_present(value: object) -> bool:
    array = np.asarray(value)
    if array.dtype != np.dtype(np.uint8) or array.shape != ():
        raise MalformedVerdictObligationError(
            "present must be a uint8 scalar"
        )
    raw = int(array.item())
    if raw not in (0, 1):
        raise MalformedVerdictObligationError("present must be exactly 0 or 1")
    return bool(raw)


def _canonical_obligation_bytes(
    *,
    checkpoint_epoch: int,
    submission_seq: int,
    stage: str,
    boundary_kind: str,
    config_sha256: str,
) -> bytes:
    return json.dumps(
        {
            "boundary_kind": boundary_kind,
            "checkpoint_epoch": checkpoint_epoch,
            "config_sha256": config_sha256,
            "schema": SCHEMA,
            "stage": stage,
            "submission_seq": submission_seq,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _obligation_id(
    *,
    checkpoint_epoch: int,
    submission_seq: int,
    stage: str,
    boundary_kind: str,
    config_sha256: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(_OBLIGATION_ID_DOMAIN)
    digest.update(
        _canonical_obligation_bytes(
            checkpoint_epoch=checkpoint_epoch,
            submission_seq=submission_seq,
            stage=stage,
            boundary_kind=boundary_kind,
            config_sha256=config_sha256,
        )
    )
    return digest.hexdigest()


def _readonly_array(value: np.ndarray) -> np.ndarray:
    source = np.ascontiguousarray(value)
    return np.frombuffer(source.tobytes(order="C"), dtype=source.dtype).reshape(
        source.shape
    )


def _freeze_snapshot_tree(value: Any) -> Any:
    """Detach the reconstructed snapshot without serializing it."""

    if isinstance(value, np.ndarray):
        if value.dtype.hasobject or value.dtype.fields is not None:
            raise MalformedVerdictObligationError(
                "reconstructed snapshot forbids object and structured arrays"
            )
        return _readonly_array(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, child in value.items():
            if type(key) is not str:
                raise MalformedVerdictObligationError(
                    "reconstructed snapshot mapping keys must have exact str type"
                )
            frozen[key] = _freeze_snapshot_tree(child)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_snapshot_tree(child) for child in value)
    if value is None or type(value) in (bool, int, float, str, bytes):
        return value
    raise MalformedVerdictObligationError(
        "reconstructed snapshot must be a pure Python/NumPy tree; "
        f"got {type(value).__name__}"
    )


@dataclass(frozen=True)
class VerdictObligationRecord:
    """One fixed-size schedule coordinate; never a scorer payload."""

    checkpoint_epoch: int
    submission_seq: int
    stage: str
    boundary_kind: str
    config_sha256: str
    obligation_id: str

    @classmethod
    def create(
        cls,
        *,
        checkpoint_epoch: int,
        submission_seq: int,
        stage: str,
        boundary_kind: str,
        config_sha256: str,
    ) -> VerdictObligationRecord:
        epoch = _exact_nonnegative_int(
            checkpoint_epoch,
            name="checkpoint_epoch",
        )
        sequence = _exact_nonnegative_int(submission_seq, name="submission_seq")
        stage_text = _exact_text(
            stage,
            name="stage",
            max_utf8_bytes=MAX_STAGE_UTF8_BYTES,
        )
        boundary_text = _exact_text(
            boundary_kind,
            name="boundary_kind",
            max_utf8_bytes=MAX_BOUNDARY_KIND_UTF8_BYTES,
        )
        config_hash = _sha256_text(config_sha256, name="config_sha256")
        return cls(
            checkpoint_epoch=epoch,
            submission_seq=sequence,
            stage=stage_text,
            boundary_kind=boundary_text,
            config_sha256=config_hash,
            obligation_id=_obligation_id(
                checkpoint_epoch=epoch,
                submission_seq=sequence,
                stage=stage_text,
                boundary_kind=boundary_text,
                config_sha256=config_hash,
            ),
        )

    def validate(self) -> None:
        expected = VerdictObligationRecord.create(
            checkpoint_epoch=self.checkpoint_epoch,
            submission_seq=self.submission_seq,
            stage=self.stage,
            boundary_kind=self.boundary_kind,
            config_sha256=self.config_sha256,
        )
        supplied_id = _sha256_text(self.obligation_id, name="obligation_id")
        if supplied_id != expected.obligation_id:
            raise MalformedVerdictObligationError(
                "obligation_id differs from its canonical coordinate/config"
            )


class PostCheckpointVerdictObligation:
    """Main-thread state machine for one owed post-checkpoint submission."""

    def __init__(self, record: VerdictObligationRecord | None = None) -> None:
        if record is not None:
            record.validate()
        self._record = record
        self._creator_thread = threading.get_ident()
        self._last_discharged_id: str | None = None
        self._poison_cause: BaseException | None = None

    @property
    def owed(self) -> bool:
        return self._record is not None

    @property
    def poisoned(self) -> bool:
        return self._poison_cause is not None

    @property
    def record(self) -> VerdictObligationRecord | None:
        return self._record

    def _assert_main_thread(self) -> None:
        if threading.get_ident() != self._creator_thread:
            raise VerdictObligationError(
                "post-checkpoint obligation escaped its creating thread"
            )

    def _assert_healthy(self) -> None:
        if self._poison_cause is not None:
            raise VerdictObligationPoisonedError(
                "post-checkpoint verdict dispatch is poisoned; restart from "
                "the last complete checkpoint"
            ) from self._poison_cause

    def arm(
        self,
        *,
        checkpoint_epoch: int,
        submission_seq: int,
        stage: str,
        boundary_kind: str,
        config_sha256: str,
        next_submit_seq: int,
        next_apply_seq: int,
    ) -> VerdictObligationRecord:
        """Arm one obligation while the verdict transaction is quiescent."""

        self._assert_main_thread()
        self._assert_healthy()
        if self._record is not None:
            raise DuplicateVerdictObligationError(
                "a post-checkpoint verdict obligation is already owed"
            )
        sequence = _exact_nonnegative_int(submission_seq, name="submission_seq")
        submit_cursor = _exact_nonnegative_int(
            next_submit_seq,
            name="next_submit_seq",
        )
        apply_cursor = _exact_nonnegative_int(
            next_apply_seq,
            name="next_apply_seq",
        )
        if submit_cursor != apply_cursor or submit_cursor != sequence:
            raise StaleVerdictObligationError(
                "obligation can arm only at a quiescent exact submit/apply cursor"
            )
        record = VerdictObligationRecord.create(
            checkpoint_epoch=checkpoint_epoch,
            submission_seq=sequence,
            stage=stage,
            boundary_kind=boundary_kind,
            config_sha256=config_sha256,
        )
        self._record = record
        self._last_discharged_id = None
        return record

    def assert_optimizer_step_allowed(self) -> None:
        """Refuse training mutation while a restored obligation is owed."""

        self._assert_main_thread()
        self._assert_healthy()
        if self._record is not None:
            raise VerdictObligationOwedError(
                "optimizer step refused until the post-checkpoint verdict "
                f"obligation {self._record.obligation_id} is discharged"
            )

    def discharge(
        self,
        *,
        restored_checkpoint_epoch: int,
        restored_stage: str,
        restored_boundary_kind: str,
        restored_config_sha256: str,
        next_submit_seq: Callable[[], int],
        next_apply_seq: Callable[[], int],
        reconstruct_snapshot: Callable[
            [VerdictObligationRecord], Mapping[str, Any]
        ],
        submit: Callable[[Mapping[str, Any]], int],
    ) -> int:
        """Reconstruct from restored state and submit exactly once in-process."""

        self._assert_main_thread()
        self._assert_healthy()
        record = self._record
        if record is None:
            if self._last_discharged_id is not None:
                raise DuplicateVerdictObligationError(
                    "post-checkpoint verdict obligation was already discharged "
                    f"in this process: {self._last_discharged_id}"
                )
            raise DuplicateVerdictObligationError(
                "no post-checkpoint verdict obligation is owed"
            )
        if not callable(next_submit_seq) or not callable(next_apply_seq):
            raise TypeError("transaction cursor readers must be callable")
        if not callable(reconstruct_snapshot) or not callable(submit):
            raise TypeError("snapshot reconstruction and submit must be callable")

        restored_epoch = _exact_nonnegative_int(
            restored_checkpoint_epoch,
            name="restored_checkpoint_epoch",
        )
        restored_stage_text = _exact_text(
            restored_stage,
            name="restored_stage",
            max_utf8_bytes=MAX_STAGE_UTF8_BYTES,
        )
        restored_boundary_text = _exact_text(
            restored_boundary_kind,
            name="restored_boundary_kind",
            max_utf8_bytes=MAX_BOUNDARY_KIND_UTF8_BYTES,
        )
        restored_config_hash = _sha256_text(
            restored_config_sha256,
            name="restored_config_sha256",
        )
        if (
            restored_epoch != record.checkpoint_epoch
            or restored_stage_text != record.stage
            or restored_boundary_text != record.boundary_kind
            or restored_config_hash != record.config_sha256
        ):
            raise StaleVerdictObligationError(
                "obligation differs from the restored checkpoint coordinate/config"
            )

        submit_before = _exact_nonnegative_int(
            next_submit_seq(),
            name="next_submit_seq",
        )
        apply_before = _exact_nonnegative_int(
            next_apply_seq(),
            name="next_apply_seq",
        )
        if submit_before != apply_before or submit_before != record.submission_seq:
            raise StaleVerdictObligationError(
                "obligation submission sequence differs from restored "
                "quiescent transaction cursors"
            )

        try:
            raw_snapshot = reconstruct_snapshot(record)
            if not isinstance(raw_snapshot, Mapping):
                raise MalformedVerdictObligationError(
                    "reconstructed worker snapshot must be a mapping"
                )
            snapshot = _freeze_snapshot_tree(raw_snapshot)
            if snapshot.get("schema") != WORKER_SNAPSHOT_SCHEMA:
                raise MalformedVerdictObligationError(
                    "reconstructed worker snapshot schema differs"
                )
            snapshot_epoch = _exact_nonnegative_int(
                snapshot.get("epoch"),
                name="reconstructed snapshot epoch",
            )
            if snapshot_epoch != record.checkpoint_epoch:
                raise StaleVerdictObligationError(
                    "reconstructed worker snapshot epoch differs from obligation"
                )
            returned_sequence = _exact_nonnegative_int(
                submit(snapshot),
                name="submit return sequence",
            )
            submit_after = _exact_nonnegative_int(
                next_submit_seq(),
                name="next_submit_seq after dispatch",
            )
            apply_after = _exact_nonnegative_int(
                next_apply_seq(),
                name="next_apply_seq after dispatch",
            )
            if (
                returned_sequence != record.submission_seq
                or submit_after != record.submission_seq + 1
                or apply_after != record.submission_seq
            ):
                raise StaleVerdictObligationError(
                    "dispatch did not perform exactly one submit at the "
                    "obligation sequence"
                )
        except Exception as exc:
            self._poison_cause = exc
            raise VerdictObligationDispatchError(
                "post-checkpoint verdict reconstruction/dispatch became "
                "indeterminate; restart from the last complete checkpoint"
            ) from exc

        self._record = None
        self._last_discharged_id = record.obligation_id
        return returned_sequence

    def assert_final_checkpoint_ready(
        self,
        *,
        next_submit_seq: int,
        next_apply_seq: int,
    ) -> None:
        """Require final verdict discharge and join before final checkpoint."""

        self._assert_main_thread()
        self._assert_healthy()
        if self._record is not None:
            raise FinalCheckpointNotReadyError(
                "final checkpoint refused while a verdict obligation is owed"
            )
        submit_cursor = _exact_nonnegative_int(
            next_submit_seq,
            name="next_submit_seq",
        )
        apply_cursor = _exact_nonnegative_int(
            next_apply_seq,
            name="next_apply_seq",
        )
        if submit_cursor != apply_cursor:
            raise FinalCheckpointNotReadyError(
                "final checkpoint refused until the final verdict is joined "
                "and applied"
            )

    def numpy_state(self, *, prefix: str = "") -> Mapping[str, np.ndarray]:
        """Return the fixed typed checkpoint state; never a worker payload."""

        self._assert_main_thread()
        self._assert_healthy()
        if type(prefix) is not str:
            raise TypeError("prefix must have exact str type")
        record = self._record
        arrays = {
            f"{prefix}schema": _utf8_array(SCHEMA),
            f"{prefix}present": _uint8_scalar(1 if record is not None else 0),
            f"{prefix}checkpoint_epoch": _int64_scalar(
                record.checkpoint_epoch if record is not None else -1
            ),
            f"{prefix}submission_seq": _int64_scalar(
                record.submission_seq if record is not None else -1
            ),
            f"{prefix}stage": _utf8_array(
                record.stage if record is not None else ""
            ),
            f"{prefix}boundary_kind": _utf8_array(
                record.boundary_kind if record is not None else ""
            ),
            f"{prefix}config_sha256": _utf8_array(
                record.config_sha256 if record is not None else ""
            ),
            f"{prefix}obligation_id": _utf8_array(
                record.obligation_id if record is not None else ""
            ),
        }
        for key, array in arrays.items():
            if array.dtype.hasobject or array.dtype.fields is not None:
                raise AssertionError(f"{key} unexpectedly contains object data")
        return MappingProxyType(arrays)

    @classmethod
    def from_numpy_state(
        cls,
        arrays: Mapping[str, object],
        *,
        prefix: str = "",
    ) -> PostCheckpointVerdictObligation:
        """Parse strict canonical state from a complete checkpoint mapping."""

        if not isinstance(arrays, Mapping):
            raise MalformedVerdictObligationError(
                "obligation state arrays must be a mapping"
            )
        if type(prefix) is not str:
            raise TypeError("prefix must have exact str type")
        matched: dict[str, object] = {}
        for key, value in arrays.items():
            if type(key) is not str:
                raise MalformedVerdictObligationError(
                    "obligation state keys must have exact str type"
                )
            if key.startswith(prefix):
                matched[key[len(prefix) :]] = value
        if set(matched) != _STATE_SUFFIXES:
            raise MalformedVerdictObligationError(
                "obligation state fields differ from the canonical keyset"
            )
        schema = _decode_utf8_array(matched["schema"], name="schema")
        if schema != SCHEMA:
            raise MalformedVerdictObligationError(
                "post-checkpoint obligation schema differs"
            )
        present = _read_present(matched["present"])
        epoch = _read_int64_scalar(
            matched["checkpoint_epoch"],
            name="checkpoint_epoch",
        )
        sequence = _read_int64_scalar(
            matched["submission_seq"],
            name="submission_seq",
        )
        stage = _decode_utf8_array(matched["stage"], name="stage")
        boundary_kind = _decode_utf8_array(
            matched["boundary_kind"],
            name="boundary_kind",
        )
        config_sha256 = _decode_utf8_array(
            matched["config_sha256"],
            name="config_sha256",
        )
        obligation_id = _decode_utf8_array(
            matched["obligation_id"],
            name="obligation_id",
        )
        if not present:
            if (
                epoch != -1
                or sequence != -1
                or stage
                or boundary_kind
                or config_sha256
                or obligation_id
            ):
                raise MalformedVerdictObligationError(
                    "absent obligation must use canonical empty sentinels"
                )
            return cls()
        if epoch < 0 or sequence < 0:
            raise MalformedVerdictObligationError(
                "present obligation cursors must be nonnegative"
            )
        record = VerdictObligationRecord(
            checkpoint_epoch=epoch,
            submission_seq=sequence,
            stage=stage,
            boundary_kind=boundary_kind,
            config_sha256=config_sha256,
            obligation_id=obligation_id,
        )
        record.validate()
        return cls(record)
