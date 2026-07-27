# SPDX-License-Identifier: MIT
"""Typed, fail-closed native-v3 trajectory transaction primitives.

This module owns the format-independent part of
``g111_trajectory_transaction.v2``.  It deliberately does not know about the
trainer or mutate live objects.  A caller supplies two independent views:

* a :class:`TransactionManifest` captured from the serialized arrays; and
* an :class:`ExpectedTransactionSchema` derived from the current DSL and
  freshly constructed model, optimizers, controllers, and journals.

Validation stages private, read-only NumPy copies and proves exact reverse
coverage.  Prefixes are never treated as completeness evidence.  Publication
into live trainer objects is an integration-unit responsibility and may happen
only after :func:`validate_transaction` and the caller's cross-invariants pass.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

import numpy as np

SCHEMA: Final = "g111_trajectory_transaction.v2"
MANIFEST_KEY: Final = "__g111_trajectory_transaction_v2_manifest"
PENDING_VERDICT_PREFIX: Final = "__cl_pend_"
VERDICT_BARRIER_SCHEMA: Final = "tac.g111_verdict_barrier.v1"

CURRENT_TRAIN_STATE: Final = "current_train_state"
ROLLBACK_SAVEPOINT: Final = "rollback_savepoint"
SCHEDULE_CONTROL_STATE: Final = "schedule_control_state"
VERDICT_TRANSACTION: Final = "verdict_transaction"
CAUSAL_SELECTION_STATE: Final = "causal_selection_state"
LINEAGE_ENVELOPE: Final = "lineage_envelope"

ATOMIC_OWNERS: Final[tuple[str, ...]] = (
    CURRENT_TRAIN_STATE,
    ROLLBACK_SAVEPOINT,
    SCHEDULE_CONTROL_STATE,
    VERDICT_TRANSACTION,
    CAUSAL_SELECTION_STATE,
    LINEAGE_ENVELOPE,
)
RESTORABLE_STATE_OWNERS: Final[tuple[str, ...]] = ATOMIC_OWNERS[:5]

PRIMARY_MODEL_DOMAIN: Final = "primary_model_ema_optimizer_family"
PROTECTED_SEED_DOMAIN: Final = "protected_seed_optimizer_support"
FRESH_LINEAGE_DOMAIN: Final = "fresh_root_physical_lineage"
RNG_STREAMS_DOMAIN: Final = "rng_streams"
EVENT_GATES_DOMAIN: Final = "event_gates_duplicate_booleans"
STAGE_TRANSITION_DOMAIN: Final = "stage_transition_rewarmup"
SPIKE_ROLLBACK_DOMAIN: Final = "spike_rollback_last_good_snapshot"
LADDER_DOMAIN: Final = "ladder"
TAIL_INPUTS_DOMAIN: Final = "tail_verdict_inputs"
VERDICT_JOURNAL_DOMAIN: Final = "verdict_journal_sensor_histories"
PENDING_REDUCER_DOMAIN: Final = "pending_verdict_reducer_boundary"
JACOBIAN_BASIN_DOMAIN: Final = "jacobian_basin"
POLYAK_DOMAIN: Final = "polyak_atomic_state"
BEST_STAGE_DOMAIN: Final = "best_stage_bookkeeping"

SEMANTIC_DOMAINS: Final[tuple[str, ...]] = (
    PRIMARY_MODEL_DOMAIN,
    PROTECTED_SEED_DOMAIN,
    FRESH_LINEAGE_DOMAIN,
    RNG_STREAMS_DOMAIN,
    EVENT_GATES_DOMAIN,
    STAGE_TRANSITION_DOMAIN,
    SPIKE_ROLLBACK_DOMAIN,
    LADDER_DOMAIN,
    TAIL_INPUTS_DOMAIN,
    VERDICT_JOURNAL_DOMAIN,
    PENDING_REDUCER_DOMAIN,
    JACOBIAN_BASIN_DOMAIN,
    POLYAK_DOMAIN,
    BEST_STAGE_DOMAIN,
)

CANONICAL_DOMAIN_COVERAGE: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        PRIMARY_MODEL_DOMAIN: (CURRENT_TRAIN_STATE,),
        PROTECTED_SEED_DOMAIN: (CURRENT_TRAIN_STATE,),
        FRESH_LINEAGE_DOMAIN: (LINEAGE_ENVELOPE,),
        RNG_STREAMS_DOMAIN: (SCHEDULE_CONTROL_STATE,),
        EVENT_GATES_DOMAIN: (SCHEDULE_CONTROL_STATE,),
        STAGE_TRANSITION_DOMAIN: (SCHEDULE_CONTROL_STATE,),
        SPIKE_ROLLBACK_DOMAIN: (CURRENT_TRAIN_STATE, ROLLBACK_SAVEPOINT),
        LADDER_DOMAIN: (SCHEDULE_CONTROL_STATE,),
        TAIL_INPUTS_DOMAIN: (SCHEDULE_CONTROL_STATE, VERDICT_TRANSACTION),
        VERDICT_JOURNAL_DOMAIN: (VERDICT_TRANSACTION,),
        PENDING_REDUCER_DOMAIN: (VERDICT_TRANSACTION,),
        JACOBIAN_BASIN_DOMAIN: (SCHEDULE_CONTROL_STATE, VERDICT_TRANSACTION),
        POLYAK_DOMAIN: (CURRENT_TRAIN_STATE,),
        BEST_STAGE_DOMAIN: (CAUSAL_SELECTION_STATE,),
    }
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class TransactionValidationError(ValueError):
    """A native-v3 transaction failed before any live-state publication."""


def _fail(message: str) -> None:
    raise TransactionValidationError(message)


def _canonical_dtype(dtype: str | np.dtype[Any]) -> str:
    try:
        value = np.dtype(dtype)
    except TypeError as exc:
        raise TransactionValidationError(f"invalid dtype {dtype!r}") from exc
    if value.hasobject or value.fields is not None:
        _fail(
            "object and structured arrays are forbidden because native-v3 requires pickle-free canonical scalar dtypes"
        )
    return value.str


def _canonical_key(value: object, *, what: str = "key") -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        _fail(f"{what} must be a non-empty canonical string, got {value!r}")
    return value


def _exact_integer(value: object, *, what: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        _fail(f"{what} must be an exact integer, got {value!r}")
    return int(value)


def _canonical_shape(shape: object) -> tuple[int, ...]:
    if isinstance(shape, (str, bytes, bytearray, Mapping)):
        _fail(f"shape must be an iterable of exact integers, got {shape!r}")
    try:
        dimensions = tuple(shape)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TransactionValidationError(f"shape must be an iterable of exact integers, got {shape!r}") from exc
    result = tuple(_exact_integer(item, what="shape dimension") for item in dimensions)
    if any(item < 0 for item in result):
        _fail(f"shape dimensions must be non-negative, got {result}")
    return result


def _canonical_array(value: np.ndarray | object, *, key: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.hasobject or array.dtype.fields is not None:
        _fail(f"{key!r}: object and structured arrays are forbidden because native-v3 requires canonical scalar dtypes")
    original_shape = array.shape
    contiguous = np.ascontiguousarray(array)
    immutable_buffer = bytes(contiguous.tobytes(order="C"))
    result = np.frombuffer(immutable_buffer, dtype=contiguous.dtype).reshape(original_shape)
    return result


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def _decode_utf8_uint8(array: np.ndarray, *, key: str) -> str:
    value = np.asarray(array)
    if value.dtype != np.dtype(np.uint8) or value.ndim != 1:
        _fail(f"{key!r}: serialized UTF-8 field must be one-dimensional uint8")
    try:
        return value.tobytes(order="C").decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TransactionValidationError(f"{key!r}: serialized field is not valid UTF-8") from exc


def _read_nonnegative_int64_vector_scalar(
    array: np.ndarray,
    *,
    key: str,
) -> int:
    value = np.asarray(array)
    if value.dtype != np.dtype(np.int64) or value.shape != (1,):
        _fail(f"{key!r}: serialized cursor must have dtype int64 and shape (1,)")
    result = int(value[0])
    if result < 0:
        _fail(f"{key!r}: serialized cursor must be non-negative")
    return result


@dataclass(frozen=True, slots=True)
class EntrySpec:
    """Independent expected dtype/shape contract for one serialized key."""

    key: str
    dtype: str
    shape: tuple[int, ...]
    finite: bool = True
    allow_empty: bool = False

    def __post_init__(self) -> None:
        if self.finite is not True and self.finite is not False:
            _fail(f"{self.key!r}: finite must be an exact bool")
        if self.allow_empty is not True and self.allow_empty is not False:
            _fail(f"{self.key!r}: allow_empty must be an exact bool")
        object.__setattr__(self, "key", _canonical_key(self.key))
        object.__setattr__(self, "dtype", _canonical_dtype(self.dtype))
        object.__setattr__(self, "shape", _canonical_shape(self.shape))
        if not self.allow_empty and any(dimension == 0 for dimension in self.shape):
            _fail(f"{self.key!r}: empty expected shapes require allow_empty=True")


@dataclass(frozen=True, slots=True)
class EntryDescriptor:
    """Canonical serialized-entry descriptor committed by the manifest."""

    key: str
    owner: str
    dtype: str
    shape: tuple[int, ...]
    nbytes: int
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", _canonical_key(self.key))
        object.__setattr__(self, "owner", _canonical_key(self.owner, what="owner"))
        object.__setattr__(self, "dtype", _canonical_dtype(self.dtype))
        object.__setattr__(self, "shape", _canonical_shape(self.shape))
        object.__setattr__(
            self,
            "nbytes",
            _exact_integer(self.nbytes, what=f"{self.key!r}: nbytes"),
        )
        if self.nbytes < 0:
            _fail(f"{self.key!r}: nbytes must be non-negative")
        if not isinstance(self.sha256, str) or not _SHA256_RE.fullmatch(self.sha256):
            _fail(f"{self.key!r}: sha256 must be 64 lowercase hexadecimal characters")

    @classmethod
    def from_array(cls, key: str, owner: str, array: np.ndarray) -> EntryDescriptor:
        value = np.asarray(array)
        if value.dtype.hasobject or value.dtype.fields is not None:
            _fail(
                f"{key!r}: object and structured arrays are forbidden because "
                "native-v3 requires canonical scalar dtypes"
            )
        return cls(
            key=key,
            owner=owner,
            dtype=value.dtype.str,
            shape=tuple(int(item) for item in value.shape),
            nbytes=int(value.nbytes),
            sha256=_array_sha256(value),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "owner": self.owner,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "nbytes": self.nbytes,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, object]) -> EntryDescriptor:
        _require_exact_fields(
            row,
            {"key", "owner", "dtype", "shape", "nbytes", "sha256"},
            "entry descriptor",
        )
        shape = row["shape"]
        if not isinstance(shape, list):
            _fail("entry descriptor shape must be a JSON list")
        return cls(
            key=row["key"],  # type: ignore[arg-type]
            owner=row["owner"],  # type: ignore[arg-type]
            dtype=row["dtype"],  # type: ignore[arg-type]
            shape=tuple(shape),  # type: ignore[arg-type]
            nbytes=row["nbytes"],  # type: ignore[arg-type]
            sha256=row["sha256"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class OwnerClaim:
    owner: str
    keys: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner", _canonical_key(self.owner, what="owner"))
        keys = tuple(_canonical_key(key) for key in self.keys)
        if tuple(sorted(keys)) != keys:
            _fail(f"owner {self.owner!r}: claimed keys are not canonically sorted")
        if len(set(keys)) != len(keys):
            _fail(f"owner {self.owner!r}: duplicate key inside owner claim")
        object.__setattr__(self, "keys", keys)


@dataclass(frozen=True, slots=True)
class OwnerActivity:
    owner: str
    active: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner", _canonical_key(self.owner, what="owner"))
        if not isinstance(self.active, bool):
            _fail(f"owner {self.owner!r}: activity must be boolean")


@dataclass(frozen=True, slots=True)
class DomainCoverage:
    domain: str
    owners: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "domain", _canonical_key(self.domain, what="domain"))
        owners = tuple(_canonical_key(owner, what="owner") for owner in self.owners)
        if not owners:
            _fail(f"domain {self.domain!r}: coverage must not be empty")
        if len(set(owners)) != len(owners):
            _fail(f"domain {self.domain!r}: duplicate owner")
        object.__setattr__(self, "owners", owners)


@dataclass(frozen=True, slots=True)
class TransactionManifest:
    """Captured transaction declaration; never used as its own expected schema."""

    schema: str
    entries: tuple[EntryDescriptor, ...]
    owner_claims: tuple[OwnerClaim, ...]
    activity: tuple[OwnerActivity, ...]
    domain_coverage: tuple[DomainCoverage, ...]
    derived_lineage_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.schema != SCHEMA:
            _fail(f"schema mismatch: expected {SCHEMA!r}, got {self.schema!r}")
        entries = tuple(self.entries)
        claims = tuple(self.owner_claims)
        activity = tuple(self.activity)
        coverage = tuple(self.domain_coverage)
        derived = tuple(_canonical_key(key) for key in self.derived_lineage_keys)
        if tuple(sorted(derived)) != derived or len(set(derived)) != len(derived):
            _fail("derived lineage keys must be unique and canonically sorted")
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "owner_claims", claims)
        object.__setattr__(self, "activity", activity)
        object.__setattr__(self, "domain_coverage", coverage)
        object.__setattr__(self, "derived_lineage_keys", derived)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "entries": [entry.as_dict() for entry in self.entries],
            "owner_claims": [{"owner": claim.owner, "keys": list(claim.keys)} for claim in self.owner_claims],
            "activity": [{"owner": activity.owner, "active": activity.active} for activity in self.activity],
            "domain_coverage": [
                {"domain": coverage.domain, "owners": list(coverage.owners)} for coverage in self.domain_coverage
            ],
            "derived_lineage_keys": list(self.derived_lineage_keys),
        }

    def to_json(self) -> str:
        return json.dumps(
            self.as_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @classmethod
    def from_json(cls, payload: str) -> TransactionManifest:
        try:
            row = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise TransactionValidationError("manifest is not valid JSON") from exc
        if not isinstance(row, dict):
            _fail("manifest root must be a JSON object")
        _require_exact_fields(
            row,
            {
                "schema",
                "entries",
                "owner_claims",
                "activity",
                "domain_coverage",
                "derived_lineage_keys",
            },
            "manifest",
        )
        for field in (
            "entries",
            "owner_claims",
            "activity",
            "domain_coverage",
            "derived_lineage_keys",
        ):
            if not isinstance(row[field], list):
                _fail(f"manifest field {field!r} must be a JSON list")
        parsed_entries: list[EntryDescriptor] = []
        for item in row["entries"]:
            if not isinstance(item, dict):
                _fail("entry descriptor must be a JSON object")
            parsed_entries.append(EntryDescriptor.from_dict(item))
        claims: list[OwnerClaim] = []
        for item in row["owner_claims"]:
            if not isinstance(item, dict):
                _fail("owner claim must be a JSON object")
            _require_exact_fields(item, {"owner", "keys"}, "owner claim")
            if not isinstance(item["keys"], list):
                _fail("owner claim keys must be a JSON list")
            claims.append(OwnerClaim(item["owner"], tuple(item["keys"])))
        activities: list[OwnerActivity] = []
        for item in row["activity"]:
            if not isinstance(item, dict):
                _fail("owner activity must be a JSON object")
            _require_exact_fields(item, {"owner", "active"}, "owner activity")
            activities.append(OwnerActivity(item["owner"], item["active"]))
        domains: list[DomainCoverage] = []
        for item in row["domain_coverage"]:
            if not isinstance(item, dict):
                _fail("domain coverage must be a JSON object")
            _require_exact_fields(item, {"domain", "owners"}, "domain coverage")
            if not isinstance(item["owners"], list):
                _fail("domain coverage owners must be a JSON list")
            domains.append(DomainCoverage(item["domain"], tuple(item["owners"])))
        return cls(
            schema=row["schema"],
            entries=tuple(parsed_entries),
            owner_claims=tuple(claims),
            activity=tuple(activities),
            domain_coverage=tuple(domains),
            derived_lineage_keys=tuple(row["derived_lineage_keys"]),
        )


@dataclass(frozen=True, slots=True)
class ExpectedOwnerSchema:
    """Expected owner topology constructed independently of checkpoint claims.

    Native-v3's inactive rule is exact: ``active=False`` means the owner has
    zero required leaves and its manifest claim must be empty.  ``permitted``
    may still describe the inventory that would be legal if the owner were
    active; it grants no inactive ownership.
    """

    owner: str
    active: bool
    required: tuple[EntrySpec, ...]
    permitted: tuple[EntrySpec, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner", _canonical_key(self.owner, what="owner"))
        if not isinstance(self.active, bool):
            _fail(f"owner {self.owner!r}: expected activity must be boolean")
        required = _spec_tuple(self.required, owner=self.owner, label="required")
        permitted = _spec_tuple(self.permitted, owner=self.owner, label="permitted")
        required_keys = {spec.key for spec in required}
        permitted_keys = {spec.key for spec in permitted}
        if not required_keys <= permitted_keys:
            _fail(
                f"owner {self.owner!r}: required keys absent from permitted schema: "
                f"{sorted(required_keys - permitted_keys)}"
            )
        if not self.active and required_keys:
            _fail(f"inactive owner {self.owner!r} must declare zero required keys; got {sorted(required_keys)}")
        object.__setattr__(self, "required", required)
        object.__setattr__(self, "permitted", permitted)


@dataclass(frozen=True, slots=True)
class ParallelHistorySpec:
    """Invariant contract for one bounded structure-of-arrays history."""

    name: str
    keys: tuple[str, ...]
    sequence_key: str
    max_length: int
    next_sequence_key: str | None = None
    require_contiguous: bool = False
    allow_empty: bool = True

    def __post_init__(self) -> None:
        if self.require_contiguous is not True and self.require_contiguous is not False:
            _fail(f"history {self.name!r}: require_contiguous must be an exact bool")
        if self.allow_empty is not True and self.allow_empty is not False:
            _fail(f"history {self.name!r}: allow_empty must be an exact bool")
        object.__setattr__(self, "name", _canonical_key(self.name, what="history name"))
        keys = tuple(_canonical_key(key) for key in self.keys)
        if not keys or len(set(keys)) != len(keys):
            _fail(f"history {self.name!r}: keys must be non-empty and unique")
        if self.sequence_key not in keys:
            _fail(f"history {self.name!r}: sequence_key must be one of keys")
        object.__setattr__(self, "keys", keys)
        object.__setattr__(
            self,
            "sequence_key",
            _canonical_key(self.sequence_key, what="sequence key"),
        )
        if self.next_sequence_key is not None:
            object.__setattr__(
                self,
                "next_sequence_key",
                _canonical_key(self.next_sequence_key, what="next sequence key"),
            )
        object.__setattr__(
            self,
            "max_length",
            _exact_integer(
                self.max_length,
                what=f"history {self.name!r}: max_length",
            ),
        )
        if self.max_length <= 0:
            _fail(f"history {self.name!r}: max_length must be positive")


@dataclass(frozen=True, slots=True)
class ExpectedTransactionSchema:
    """Independent authority for activity, key topology, domains, and histories."""

    owners: tuple[ExpectedOwnerSchema, ...]
    domain_coverage: tuple[DomainCoverage, ...]
    derived_lineage: tuple[EntrySpec, ...] = ()
    histories: tuple[ParallelHistorySpec, ...] = ()

    def __post_init__(self) -> None:
        owners = tuple(self.owners)
        domains = tuple(self.domain_coverage)
        derived = _spec_tuple(
            self.derived_lineage,
            owner=LINEAGE_ENVELOPE,
            label="derived lineage",
        )
        histories = tuple(self.histories)
        owner_names = tuple(owner.owner for owner in owners)
        if owner_names != ATOMIC_OWNERS:
            _fail(f"expected owner schema must contain the six atomic owners in canonical order; got {owner_names}")
        domain_map = _domain_map(domains, label="expected")
        if tuple(domain_map) != SEMANTIC_DOMAINS:
            _fail(
                "expected schema must contain all fourteen semantic domains in canonical "
                f"order; got {tuple(domain_map)}"
            )
        if domain_map != dict(CANONICAL_DOMAIN_COVERAGE):
            _fail("expected semantic-domain coverage differs from the canonical matrix")
        all_specs: dict[str, str] = {}
        for owner in owners:
            for spec in owner.permitted:
                previous = all_specs.setdefault(spec.key, owner.owner)
                if previous != owner.owner:
                    _fail(f"expected key {spec.key!r} is permitted by multiple owners: {previous!r}, {owner.owner!r}")
        for spec in derived:
            if spec.key in all_specs:
                _fail(f"derived lineage key {spec.key!r} also appears in an owner schema")
            all_specs[spec.key] = "derived_lineage"
        verdict_schema = owners[ATOMIC_OWNERS.index(VERDICT_TRANSACTION)]
        if not verdict_schema.active and histories:
            _fail("inactive verdict_transaction must declare zero history schemas")
        object.__setattr__(self, "owners", owners)
        object.__setattr__(self, "domain_coverage", domains)
        object.__setattr__(self, "derived_lineage", derived)
        object.__setattr__(self, "histories", histories)


@dataclass(frozen=True, slots=True)
class QuiescentBarrierState:
    """Canonical O4 reducer/barrier coordinate at a native-v3 checkpoint."""

    next_submit_seq: int
    next_apply_seq: int
    pending_count: int
    last_applied_result_id: str = ""
    last_applied_result_sha256: str = ""

    def validate(self) -> None:
        for name in ("next_submit_seq", "next_apply_seq", "pending_count"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                _fail(f"barrier {name} must be a non-negative integer")
        if self.pending_count != 0:
            _fail("native-v3 checkpoint barrier requires pending_count == 0")
        if self.next_submit_seq != self.next_apply_seq:
            _fail("native-v3 checkpoint barrier requires next_submit_seq == next_apply_seq")
        if type(self.last_applied_result_id) is not str:
            _fail("last applied result ID must have exact str type")
        if type(self.last_applied_result_sha256) is not str:
            _fail("last applied result SHA-256 must have exact str type")
        if self.last_applied_result_id and any(character.isspace() for character in self.last_applied_result_id):
            _fail("last applied result ID is not canonical")
        has_id = bool(self.last_applied_result_id)
        has_hash = bool(self.last_applied_result_sha256)
        if has_id != has_hash:
            _fail("last applied result ID and SHA-256 must be present together")
        if has_hash and not _SHA256_RE.fullmatch(self.last_applied_result_sha256):
            _fail("last applied result SHA-256 is not canonical")
        if self.next_apply_seq == 0 and has_id:
            _fail("empty verdict journal cursor forbids a last-applied identity")
        if self.next_apply_seq > 0 and not has_id:
            _fail("non-empty verdict journal cursor requires a last-applied identity")


@dataclass(frozen=True, slots=True)
class BarrierStateBinding:
    """Bind O4 quiescence to exact serialized checkpoint fields.

    The six core keys match ``VerdictCheckpointCapture.numpy_state``. Validation
    parses the checkpoint bytes through this binding; a detached
    :class:`QuiescentBarrierState` is only an optional expected value and can
    never satisfy the gate by itself.
    """

    schema_key: str
    next_submit_seq_key: str
    next_apply_seq_key: str
    pending_count_key: str
    last_applied_result_id_key: str
    last_applied_result_sha256_key: str
    expected_schema: str = VERDICT_BARRIER_SCHEMA

    def __post_init__(self) -> None:
        key_fields = (
            "schema_key",
            "next_submit_seq_key",
            "next_apply_seq_key",
            "pending_count_key",
            "last_applied_result_id_key",
            "last_applied_result_sha256_key",
        )
        keys = tuple(_canonical_key(getattr(self, field), what=f"barrier {field}") for field in key_fields)
        if len(set(keys)) != len(keys):
            _fail("barrier serialized-key binding contains duplicate keys")
        for field, key in zip(key_fields, keys, strict=True):
            object.__setattr__(self, field, key)
        object.__setattr__(
            self,
            "expected_schema",
            _canonical_key(self.expected_schema, what="barrier expected schema"),
        )

    @classmethod
    def from_prefix(cls, prefix: str = "") -> BarrierStateBinding:
        if not isinstance(prefix, str) or prefix.strip() != prefix:
            _fail(f"barrier prefix must be a canonical string, got {prefix!r}")
        return cls(
            schema_key=f"{prefix}schema",
            next_submit_seq_key=f"{prefix}next_submit_seq",
            next_apply_seq_key=f"{prefix}next_apply_seq",
            pending_count_key=f"{prefix}pending_count",
            last_applied_result_id_key=f"{prefix}last_applied_result_id",
            last_applied_result_sha256_key=(f"{prefix}last_applied_result_sha256"),
        )

    @property
    def serialized_keys(self) -> tuple[str, ...]:
        return (
            self.schema_key,
            self.next_submit_seq_key,
            self.next_apply_seq_key,
            self.pending_count_key,
            self.last_applied_result_id_key,
            self.last_applied_result_sha256_key,
        )

    def parse(
        self,
        arrays: Mapping[str, np.ndarray],
    ) -> QuiescentBarrierState:
        missing = sorted(set(self.serialized_keys) - set(arrays))
        if missing:
            _fail(f"serialized verdict barrier is missing keys: {missing}")
        schema = _decode_utf8_uint8(
            arrays[self.schema_key],
            key=self.schema_key,
        )
        if schema != self.expected_schema:
            _fail(f"serialized verdict barrier schema {schema!r} != {self.expected_schema!r}")
        state = QuiescentBarrierState(
            next_submit_seq=_read_nonnegative_int64_vector_scalar(
                arrays[self.next_submit_seq_key],
                key=self.next_submit_seq_key,
            ),
            next_apply_seq=_read_nonnegative_int64_vector_scalar(
                arrays[self.next_apply_seq_key],
                key=self.next_apply_seq_key,
            ),
            pending_count=_read_nonnegative_int64_vector_scalar(
                arrays[self.pending_count_key],
                key=self.pending_count_key,
            ),
            last_applied_result_id=_decode_utf8_uint8(
                arrays[self.last_applied_result_id_key],
                key=self.last_applied_result_id_key,
            ),
            last_applied_result_sha256=_decode_utf8_uint8(
                arrays[self.last_applied_result_sha256_key],
                key=self.last_applied_result_sha256_key,
            ),
        )
        state.validate()
        return state


@dataclass(frozen=True, slots=True)
class StagedTransaction:
    """Validated private arrays safe to hand to a later staged restore."""

    arrays: Mapping[str, np.ndarray]
    manifest: TransactionManifest
    semantic_hash: str
    owner_semantic_hashes: Mapping[str, str]
    barrier_state: QuiescentBarrierState | None


InvariantValidator = Callable[[Mapping[str, np.ndarray]], None]


def canonical_domain_coverage() -> tuple[DomainCoverage, ...]:
    """Return the immutable fourteen-domain coverage matrix."""

    return tuple(DomainCoverage(domain, CANONICAL_DOMAIN_COVERAGE[domain]) for domain in SEMANTIC_DOMAINS)


def build_manifest(
    arrays: Mapping[str, np.ndarray | object],
    *,
    owner_claims: Mapping[str, Iterable[str]],
    activity: Mapping[str, bool],
    domain_coverage: Mapping[str, Sequence[str]],
    derived_lineage_keys: Iterable[str] = (),
) -> TransactionManifest:
    """Describe a captured array mapping without consulting expected topology."""

    normalized_claims = _normalize_owner_claims(owner_claims)
    normalized_activity = _normalize_activity(activity)
    normalized_domains = _normalize_domains(domain_coverage)
    derived = tuple(sorted(_canonical_key(key) for key in derived_lineage_keys))
    if len(set(derived)) != len(derived):
        _fail("derived lineage key list contains duplicates")
    staged = stage_arrays(arrays)
    payload_keys = set(staged) - {MANIFEST_KEY}
    claimed_by: dict[str, str] = {}
    for owner, keys in normalized_claims.items():
        for key in keys:
            if key in claimed_by:
                _fail(f"key {key!r} multiply owned by {claimed_by[key]!r} and {owner!r}")
            claimed_by[key] = owner
    overlap = set(derived) & set(claimed_by)
    if overlap:
        _fail(f"derived lineage keys cannot also be owner-claimed: {sorted(overlap)}")
    expected_payload = set(claimed_by) | set(derived)
    if payload_keys != expected_payload:
        missing = sorted(expected_payload - payload_keys)
        unowned = sorted(payload_keys - expected_payload)
        _fail(f"manifest construction reverse coverage mismatch; missing={missing}, unowned={unowned}")
    entries = tuple(
        EntryDescriptor.from_array(
            key,
            LINEAGE_ENVELOPE if key in derived else claimed_by[key],
            staged[key],
        )
        for key in sorted(payload_keys)
    )
    return TransactionManifest(
        schema=SCHEMA,
        entries=entries,
        owner_claims=tuple(OwnerClaim(owner, normalized_claims[owner]) for owner in ATOMIC_OWNERS),
        activity=tuple(OwnerActivity(owner, normalized_activity[owner]) for owner in ATOMIC_OWNERS),
        domain_coverage=tuple(DomainCoverage(domain, normalized_domains[domain]) for domain in SEMANTIC_DOMAINS),
        derived_lineage_keys=derived,
    )


def manifest_array(manifest: TransactionManifest) -> np.ndarray:
    """Encode a manifest as pickle-free canonical UTF-8 bytes."""

    return np.frombuffer(manifest.to_json().encode("utf-8"), dtype=np.uint8).copy()


def manifest_from_array(array: np.ndarray | object) -> TransactionManifest:
    """Decode a canonical uint8 manifest array."""

    value = np.asarray(array)
    if value.dtype != np.dtype(np.uint8) or value.ndim != 1:
        _fail("manifest array must be one-dimensional uint8")
    try:
        payload = value.tobytes(order="C").decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TransactionValidationError("manifest array is not valid UTF-8") from exc
    manifest = TransactionManifest.from_json(payload)
    if not np.array_equal(value, manifest_array(manifest)):
        _fail("manifest array is not canonical JSON serialization")
    return manifest


def stage_arrays(
    arrays: Mapping[str, np.ndarray | object],
) -> Mapping[str, np.ndarray]:
    """Deep-copy arrays into an immutable, key-sorted staging mapping."""

    if not isinstance(arrays, Mapping):
        raise TypeError("arrays must be a mapping")
    staged: dict[str, np.ndarray] = {}
    items = [(_canonical_key(raw_key), value) for raw_key, value in arrays.items()]
    for key, value in sorted(items, key=lambda item: item[0]):
        if key in staged:
            _fail(f"duplicate staged key {key!r}")
        staged[key] = _canonical_array(value, key=key)
    return MappingProxyType(staged)


def load_npz_staging(path: str | Path) -> Mapping[str, np.ndarray]:
    """Load an NPZ with pickle disabled and return immutable private arrays."""

    try:
        with np.load(Path(path), allow_pickle=False) as archive:
            names = tuple(archive.files)
            if len(names) != len(set(names)):
                duplicates = sorted(name for name in set(names) if names.count(name) > 1)
                _fail(f"NPZ contains duplicate member names: {duplicates}")
            raw = {key: archive[key] for key in names}
    except TransactionValidationError:
        raise
    except ValueError as exc:
        raise TransactionValidationError("NPZ contains an object array or otherwise requires pickle") from exc
    return stage_arrays(raw)


def validate_transaction(
    arrays: Mapping[str, np.ndarray | object],
    manifest: TransactionManifest,
    expected: ExpectedTransactionSchema,
    *,
    barrier_binding: BarrierStateBinding | None = None,
    expected_barrier_state: QuiescentBarrierState | None = None,
    invariant_validators: Iterable[InvariantValidator] = (),
) -> StagedTransaction:
    """Validate and stage a complete native-v3 transaction without mutation."""

    staged = stage_arrays(arrays)
    if MANIFEST_KEY in staged:
        embedded = manifest_from_array(staged[MANIFEST_KEY])
        if embedded.to_json() != manifest.to_json():
            _fail("embedded manifest differs from supplied manifest")
    payload_keys = set(staged) - {MANIFEST_KEY}
    pending = sorted(key for key in payload_keys if PENDING_VERDICT_PREFIX in key)
    if pending:
        _fail(f"native-v3 forbids pending-verdict payloads: {pending}")

    entries = _unique_entries(manifest.entries)
    if set(entries) != payload_keys:
        missing = sorted(payload_keys - set(entries))
        extra = sorted(set(entries) - payload_keys)
        _fail(f"entry descriptor coverage mismatch; undescribed={missing}, missing={extra}")

    claims = _claim_map(manifest.owner_claims)
    activity = _activity_map(manifest.activity)
    domains = _domain_map(manifest.domain_coverage, label="manifest")
    expected_domains = _domain_map(expected.domain_coverage, label="expected")
    if domains != expected_domains:
        _fail("semantic-domain coverage mismatch")

    expected_owner_map = {owner.owner: owner for owner in expected.owners}
    expected_activity = {owner: expected_owner_map[owner].active for owner in ATOMIC_OWNERS}
    if activity != expected_activity:
        _fail(f"owner activity drift; manifest={activity}, expected={expected_activity}")
    verdict_active = activity[VERDICT_TRANSACTION]
    if verdict_active and barrier_binding is None:
        _fail("active verdict_transaction requires a serialized BarrierStateBinding")
    if not verdict_active and (barrier_binding is not None or expected_barrier_state is not None):
        _fail(
            "inactive verdict_transaction forbids barrier binding/state because "
            "inactive owners must serialize zero state"
        )

    derived = set(manifest.derived_lineage_keys)
    expected_derived = {spec.key: spec for spec in expected.derived_lineage}
    if derived != set(expected_derived):
        _fail(f"derived lineage keyset mismatch; manifest={sorted(derived)}, expected={sorted(expected_derived)}")

    claimed_by: dict[str, str] = {}
    for owner in ATOMIC_OWNERS:
        owner_schema = expected_owner_map[owner]
        claimed = set(claims[owner])
        if not activity[owner] and claimed:
            _fail(f"inactive owner {owner!r} claims keys: {sorted(claimed)}")
        for key in claimed:
            previous = claimed_by.setdefault(key, owner)
            if previous != owner:
                _fail(f"key {key!r} multiply owned by {previous!r} and {owner!r}")
        required = {spec.key for spec in owner_schema.required}
        permitted = {spec.key for spec in owner_schema.permitted}
        missing = sorted(required - claimed)
        extra = sorted(claimed - permitted)
        if missing or extra:
            _fail(f"owner {owner!r} schema mismatch; missing={missing}, extra={extra}")

    overlap = derived & set(claimed_by)
    if overlap:
        _fail(f"derived lineage keys are multiply owned: {sorted(overlap)}")
    reverse_keys = payload_keys - derived
    if set(claimed_by) != reverse_keys:
        unowned = sorted(reverse_keys - set(claimed_by))
        phantom = sorted(set(claimed_by) - reverse_keys)
        _fail(f"exact reverse coverage failed; unowned={unowned}, phantom={phantom}")
    if barrier_binding is not None:
        unowned_barrier_keys = sorted(set(barrier_binding.serialized_keys) - set(claims[VERDICT_TRANSACTION]))
        if unowned_barrier_keys:
            _fail(f"serialized verdict barrier keys are not all owned by verdict_transaction: {unowned_barrier_keys}")

    expected_specs: dict[str, EntrySpec] = {}
    for owner_schema in expected.owners:
        expected_specs.update({spec.key: spec for spec in owner_schema.permitted})
    expected_specs.update(expected_derived)
    for key in sorted(payload_keys):
        descriptor = entries[key]
        expected_owner = LINEAGE_ENVELOPE if key in derived else claimed_by[key]
        if descriptor.owner != expected_owner:
            _fail(f"{key!r}: descriptor owner {descriptor.owner!r} differs from exact owner {expected_owner!r}")
        spec = expected_specs.get(key)
        if spec is None:
            _fail(f"{key!r}: unknown or prefix-impostor serialized key")
        _validate_entry(staged[key], descriptor, spec)

    _validate_histories(staged, expected.histories)
    parsed_barrier_state = barrier_binding.parse(staged) if barrier_binding is not None else None
    if expected_barrier_state is not None:
        expected_barrier_state.validate()
        if parsed_barrier_state != expected_barrier_state:
            _fail("serialized verdict barrier differs from supplied expected barrier state")
    for index, validator in enumerate(tuple(invariant_validators)):
        if not callable(validator):
            raise TypeError(f"invariant validator {index} is not callable")
        try:
            validator(staged)
        except TransactionValidationError:
            raise
        except Exception as exc:
            raise TransactionValidationError(f"cross-invariant validator {index} failed: {exc}") from exc

    owner_hashes = canonical_owner_semantic_hashes(manifest)
    return StagedTransaction(
        arrays=staged,
        manifest=manifest,
        semantic_hash=canonical_semantic_hash(manifest),
        owner_semantic_hashes=MappingProxyType(owner_hashes),
        barrier_state=parsed_barrier_state,
    )


def canonical_semantic_hash(manifest: TransactionManifest) -> str:
    """Hash the canonical typed manifest, including every entry content hash."""

    return hashlib.sha256(manifest.to_json().encode("utf-8")).hexdigest()


def canonical_owner_semantic_hashes(
    manifest: TransactionManifest,
) -> dict[str, str]:
    """Compute deterministic per-owner semantic hashes from exact claims."""

    entries = _unique_entries(manifest.entries)
    claims = _claim_map(manifest.owner_claims)
    derived = set(manifest.derived_lineage_keys)
    result: dict[str, str] = {}
    for owner in ATOMIC_OWNERS:
        keys = set(claims[owner])
        if owner == LINEAGE_ENVELOPE:
            keys |= derived
        rows = [entries[key].as_dict() for key in sorted(keys)]
        payload = json.dumps(
            {"owner": owner, "entries": rows},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        result[owner] = hashlib.sha256(payload).hexdigest()
    return result


def verify_canonical_reserialization(
    reference: StagedTransaction,
    reserialized_arrays: Mapping[str, np.ndarray | object],
    expected: ExpectedTransactionSchema,
    *,
    barrier_binding: BarrierStateBinding | None = None,
    invariant_validators: Iterable[InvariantValidator] = (),
) -> StagedTransaction:
    """Revalidate a staged restore and require equal O1-O5 semantic hashes.

    The reserialization manifest is rebuilt from the reference ownership and
    coverage declarations, not copied with stale entry hashes.  This detects a
    changed value even when dtype/shape/key topology remains identical.
    """

    claims = _claim_map(reference.manifest.owner_claims)
    activity = _activity_map(reference.manifest.activity)
    domains = _domain_map(reference.manifest.domain_coverage, label="reference")
    rebuilt = build_manifest(
        reserialized_arrays,
        owner_claims=claims,
        activity=activity,
        domain_coverage=domains,
        derived_lineage_keys=reference.manifest.derived_lineage_keys,
    )
    actual = validate_transaction(
        reserialized_arrays,
        rebuilt,
        expected,
        barrier_binding=barrier_binding,
        expected_barrier_state=reference.barrier_state,
        invariant_validators=invariant_validators,
    )
    for owner in RESTORABLE_STATE_OWNERS:
        if actual.owner_semantic_hashes[owner] != reference.owner_semantic_hashes[owner]:
            _fail(f"canonical reserialization changed semantic hash for owner {owner!r}")
    return actual


def require_matching_topology(
    arrays: Mapping[str, np.ndarray],
    *,
    key_pairs: Iterable[tuple[str, str]],
    complete_left_keys: Iterable[str],
    complete_right_keys: Iterable[str],
    label: str,
) -> None:
    """Validate explicitly paired EMA/savepoint/Polyak leaves.

    Pairing is supplied by the caller's freshly constructed topology.  The
    helper never sorts two unrelated namespaces and guesses correspondence.
    The independently supplied complete inventories prevent a caller from
    validating only a convenient matching subset.
    """

    pairs: list[tuple[str, str]] = []
    for index, pair in enumerate(key_pairs):
        if not isinstance(pair, tuple) or len(pair) != 2:
            _fail(f"{label}: topology pair {index} must be a 2-tuple")
        pairs.append(
            (
                _canonical_key(pair[0], what="left topology key"),
                _canonical_key(pair[1], what="right topology key"),
            )
        )
    if not pairs:
        _fail(f"{label}: topology pairing must not be empty")
    left = tuple(pair[0] for pair in pairs)
    right = tuple(pair[1] for pair in pairs)
    if len(left) != len(set(left)) or len(right) != len(set(right)):
        _fail(f"{label}: topology pairing reuses a left or right leaf")

    complete_left = tuple(_canonical_key(key, what="complete left topology key") for key in complete_left_keys)
    complete_right = tuple(_canonical_key(key, what="complete right topology key") for key in complete_right_keys)
    if not complete_left or not complete_right:
        _fail(f"{label}: complete topology inventories must not be empty")
    if len(complete_left) != len(set(complete_left)):
        _fail(f"{label}: complete left topology inventory contains duplicates")
    if len(complete_right) != len(set(complete_right)):
        _fail(f"{label}: complete right topology inventory contains duplicates")
    if set(left) != set(complete_left) or set(right) != set(complete_right):
        missing_left = sorted(set(complete_left) - set(left))
        extra_left = sorted(set(left) - set(complete_left))
        missing_right = sorted(set(complete_right) - set(right))
        extra_right = sorted(set(right) - set(complete_right))
        _fail(
            f"{label}: topology pairing does not exhaust complete inventories; "
            f"missing_left={missing_left}, extra_left={extra_left}, "
            f"missing_right={missing_right}, extra_right={extra_right}"
        )
    for left_key, right_key in pairs:
        if left_key not in arrays or right_key not in arrays:
            _fail(f"{label}: topology references a missing serialized key")
        left_array = arrays[left_key]
        right_array = arrays[right_key]
        if left_array.dtype.str != right_array.dtype.str:
            _fail(f"{label}: dtype differs for {left_key!r} and {right_key!r}")
        if left_array.shape != right_array.shape:
            _fail(f"{label}: shape differs for {left_key!r} and {right_key!r}")


def _validate_entry(
    array: np.ndarray,
    descriptor: EntryDescriptor,
    spec: EntrySpec,
) -> None:
    if array.dtype.str != descriptor.dtype:
        _fail(f"{descriptor.key!r}: dtype mismatch against descriptor ({array.dtype.str} != {descriptor.dtype})")
    if tuple(array.shape) != descriptor.shape:
        _fail(f"{descriptor.key!r}: shape mismatch against descriptor ({array.shape} != {descriptor.shape})")
    if int(array.nbytes) != descriptor.nbytes:
        _fail(f"{descriptor.key!r}: byte-length mismatch ({array.nbytes} != {descriptor.nbytes})")
    actual_hash = _array_sha256(array)
    if actual_hash != descriptor.sha256:
        _fail(f"{descriptor.key!r}: SHA-256 content hash mismatch")
    if array.dtype.str != spec.dtype:
        _fail(f"{descriptor.key!r}: dtype drift ({array.dtype.str} != expected {spec.dtype})")
    if tuple(array.shape) != spec.shape:
        _fail(f"{descriptor.key!r}: shape drift ({array.shape} != expected {spec.shape})")
    if not spec.allow_empty and array.size == 0:
        _fail(f"{descriptor.key!r}: empty/dummy array is forbidden")
    if spec.finite and array.dtype.kind in "fc" and not np.isfinite(array).all():
        _fail(f"{descriptor.key!r}: non-finite values are forbidden")


def _validate_histories(
    arrays: Mapping[str, np.ndarray],
    histories: Sequence[ParallelHistorySpec],
) -> None:
    names: set[str] = set()
    for history in histories:
        if history.name in names:
            _fail(f"duplicate history specification {history.name!r}")
        names.add(history.name)
        missing = [key for key in history.keys if key not in arrays]
        if missing:
            _fail(f"history {history.name!r}: missing columns {missing}")
        lengths: dict[str, int] = {}
        for key in history.keys:
            array = arrays[key]
            if array.ndim != 1:
                _fail(f"history {history.name!r}: column {key!r} must be 1-D")
            lengths[key] = int(array.shape[0])
        if len(set(lengths.values())) != 1:
            _fail(f"history {history.name!r}: unequal parallel lengths {lengths}")
        length = next(iter(lengths.values()))
        if length == 0 and not history.allow_empty:
            _fail(f"history {history.name!r}: empty history is forbidden")
        if length > history.max_length:
            _fail(f"history {history.name!r}: length {length} exceeds bound {history.max_length}")
        sequence = arrays[history.sequence_key]
        if sequence.dtype.kind not in "iu":
            _fail(f"history {history.name!r}: sequence column must be integral")
        sequence64 = sequence.astype(np.int64, copy=False)
        if np.any(sequence64 < 0):
            _fail(f"history {history.name!r}: negative sequence number")
        if length > 1:
            deltas = np.diff(sequence64)
            if np.any(deltas <= 0):
                _fail(f"history {history.name!r}: sequence is unordered or duplicated")
            if history.require_contiguous and np.any(deltas != 1):
                _fail(f"history {history.name!r}: sequence is not contiguous")
        if history.next_sequence_key is not None:
            if history.next_sequence_key not in arrays:
                _fail(f"history {history.name!r}: missing next-sequence scalar {history.next_sequence_key!r}")
            next_array = arrays[history.next_sequence_key]
            if next_array.shape != () or next_array.dtype.kind not in "iu":
                _fail(f"history {history.name!r}: next-sequence must be an integral scalar")
            next_sequence = int(next_array)
            if next_sequence < 0:
                _fail(f"history {history.name!r}: next sequence is negative")
            expected_length = min(next_sequence, history.max_length)
            if length != expected_length:
                _fail(
                    f"history {history.name!r}: invalid bounded-history "
                    f"truncation; length={length}, expected={expected_length}"
                )
            if length:
                expected_start = next_sequence - length
                expected_sequence = np.arange(expected_start, next_sequence, dtype=np.int64)
                if not np.array_equal(sequence64, expected_sequence):
                    _fail(f"history {history.name!r}: rows are not the canonical bounded suffix")


def _require_exact_fields(
    row: Mapping[str, object],
    expected: set[str],
    label: str,
) -> None:
    actual = set(row)
    if actual != expected:
        _fail(f"{label} fields differ; missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}")


def _spec_tuple(
    specs: Iterable[EntrySpec],
    *,
    owner: str,
    label: str,
) -> tuple[EntrySpec, ...]:
    result = tuple(specs)
    if any(not isinstance(spec, EntrySpec) for spec in result):
        raise TypeError(f"owner {owner!r}: {label} entries must be EntrySpec")
    keys = tuple(spec.key for spec in result)
    if tuple(sorted(keys)) != keys:
        _fail(f"owner {owner!r}: {label} specs are not canonically sorted")
    if len(set(keys)) != len(keys):
        _fail(f"owner {owner!r}: duplicate {label} key")
    return result


def _normalize_owner_claims(
    claims: Mapping[str, Iterable[str]],
) -> dict[str, tuple[str, ...]]:
    if set(claims) != set(ATOMIC_OWNERS):
        _fail(
            "owner claims must contain exactly the six atomic owners; "
            f"missing={sorted(set(ATOMIC_OWNERS) - set(claims))}, "
            f"unknown={sorted(set(claims) - set(ATOMIC_OWNERS))}"
        )
    return {owner: tuple(sorted(_canonical_key(key) for key in claims[owner])) for owner in ATOMIC_OWNERS}


def _normalize_activity(activity: Mapping[str, bool]) -> dict[str, bool]:
    if set(activity) != set(ATOMIC_OWNERS):
        _fail(
            "owner activity must contain exactly the six atomic owners; "
            f"missing={sorted(set(ATOMIC_OWNERS) - set(activity))}, "
            f"unknown={sorted(set(activity) - set(ATOMIC_OWNERS))}"
        )
    result: dict[str, bool] = {}
    for owner in ATOMIC_OWNERS:
        value = activity[owner]
        if not isinstance(value, bool):
            _fail(f"owner {owner!r}: activity must be boolean")
        result[owner] = value
    return result


def _normalize_domains(
    domains: Mapping[str, Sequence[str]],
) -> dict[str, tuple[str, ...]]:
    if set(domains) != set(SEMANTIC_DOMAINS):
        _fail(
            "domain coverage must contain all fourteen domains; "
            f"missing={sorted(set(SEMANTIC_DOMAINS) - set(domains))}, "
            f"unknown={sorted(set(domains) - set(SEMANTIC_DOMAINS))}"
        )
    result: dict[str, tuple[str, ...]] = {}
    for domain in SEMANTIC_DOMAINS:
        owners = tuple(domains[domain])
        coverage = DomainCoverage(domain, owners)
        unknown = set(coverage.owners) - set(ATOMIC_OWNERS)
        if unknown:
            _fail(f"domain {domain!r}: unknown owners {sorted(unknown)}")
        result[domain] = coverage.owners
    return result


def _claim_map(claims: Sequence[OwnerClaim]) -> dict[str, tuple[str, ...]]:
    owners = tuple(claim.owner for claim in claims)
    if owners != ATOMIC_OWNERS:
        _fail(f"manifest owner claims have missing, duplicate, unknown, or reordered owners: {owners}")
    return {claim.owner: claim.keys for claim in claims}


def _activity_map(
    activity: Sequence[OwnerActivity],
) -> dict[str, bool]:
    owners = tuple(row.owner for row in activity)
    if owners != ATOMIC_OWNERS:
        _fail(f"manifest owner activity has missing, duplicate, unknown, or reordered owners: {owners}")
    return {row.owner: row.active for row in activity}


def _domain_map(
    domains: Sequence[DomainCoverage],
    *,
    label: str,
) -> dict[str, tuple[str, ...]]:
    names = tuple(domain.domain for domain in domains)
    if names != SEMANTIC_DOMAINS:
        _fail(f"{label} domain coverage has missing, duplicate, unknown, or reordered domains: {names}")
    result = {domain.domain: domain.owners for domain in domains}
    for domain, owners in result.items():
        unknown = set(owners) - set(ATOMIC_OWNERS)
        if unknown:
            _fail(f"{label} domain {domain!r}: unknown owners {sorted(unknown)}")
    return result


def _unique_entries(
    entries: Sequence[EntryDescriptor],
) -> dict[str, EntryDescriptor]:
    keys = tuple(entry.key for entry in entries)
    if tuple(sorted(keys)) != keys:
        _fail("entry descriptors are not canonically key-sorted")
    if len(set(keys)) != len(keys):
        _fail("entry descriptor collision: duplicate serialized key")
    return {entry.key: entry for entry in entries}
