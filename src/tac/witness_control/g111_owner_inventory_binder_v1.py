# SPDX-License-Identifier: MIT
"""Real-array O1--O6 inventory binding for the fresh G111 producer.

The trainer currently emits several *flat* array maps: deploy, resume, the
native-v3 verdict barrier, controller/reducer state, and lineage inputs.  Raw
keys overlap between those maps (for example ``__cfg_n_hidden``), so this
module qualifies every leaf by its physical source before assigning exactly
one of the six canonical trajectory owners.

The expected topology is deliberately built from a separately constructed
fresh runtime plus an explicit owner inventory.  Captured checkpoint arrays
can never define their own expected subset.  This module is pure with respect
to trainer state: it stages immutable replacements for one later publication;
it does not mutate a model, optimizer, controller, BEST pointer, or lineage
tip.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

import numpy as np

from tac.witness_control.g111_live_verdict_transaction_v1 import (
    state_from_arrays as open_live_verdict_state,
)
from tac.witness_control.g111_verdict_barrier_v1 import (
    QuiescentVerdictTransaction,
)
from tac.witness_control.trajectory_transaction_v2 import (
    ATOMIC_OWNERS,
    LINEAGE_ENVELOPE,
    MANIFEST_KEY,
    VERDICT_TRANSACTION,
    BarrierStateBinding,
    EntrySpec,
    ExpectedOwnerSchema,
    ExpectedTransactionSchema,
    StagedTransaction,
    TransactionValidationError,
    build_manifest,
    canonical_domain_coverage,
    load_npz_staging,
    manifest_array,
    manifest_from_array,
    validate_transaction,
    verify_canonical_reserialization,
)

DEPLOY_SOURCE: Final = "deploy"
RESUME_SOURCE: Final = "resume"
BARRIER_SOURCE: Final = "barrier"
CONTROLLER_SOURCE: Final = "controller"
LINEAGE_SOURCE: Final = "lineage"
RUNTIME_SOURCES: Final[tuple[str, ...]] = (
    DEPLOY_SOURCE,
    RESUME_SOURCE,
    BARRIER_SOURCE,
    CONTROLLER_SOURCE,
    LINEAGE_SOURCE,
)

CURRENT_BARRIER_PREFIX: Final = "__g111_v3_verdict__"
CURRENT_LIVE_STATE_PREFIX: Final = "__g111_live__"
_BARRIER_FIELDS: Final[tuple[str, ...]] = (
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
)
_LIVE_STATE_FIELDS: Final[tuple[str, ...]] = (
    "schema",
    "state_payload",
    "state_sha256",
    "effect_cursor",
    "best_intent_cursor",
)


def _fail(message: str) -> None:
    raise TransactionValidationError(message)


def _canonical_string(value: object, *, label: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        _fail(f"{label} must be an exact non-empty canonical string")
    return value


@dataclass(frozen=True, slots=True, order=True)
class RuntimeLeafRef:
    """One raw key at one physical trainer array boundary."""

    source: str
    key: str

    def __post_init__(self) -> None:
        source = _canonical_string(self.source, label="runtime source")
        key = _canonical_string(self.key, label="runtime key")
        if source not in RUNTIME_SOURCES:
            _fail(f"unknown runtime source {source!r}")
        if key == MANIFEST_KEY:
            _fail(f"runtime key {key!r} is reserved for native-v3")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "key", key)

    @property
    def qualified_key(self) -> str:
        return f"{self.source}.{self.key}"


@dataclass(frozen=True, slots=True)
class G111RuntimeArrays:
    """The five actual flat array maps at the G111 checkpoint boundary."""

    deploy: Mapping[str, Any]
    resume: Mapping[str, Any]
    barrier: Mapping[str, Any]
    controller: Mapping[str, Any]
    lineage: Mapping[str, Any]

    def by_source(self) -> Mapping[str, Mapping[str, Any]]:
        return MappingProxyType(
            {
                DEPLOY_SOURCE: self.deploy,
                RESUME_SOURCE: self.resume,
                BARRIER_SOURCE: self.barrier,
                CONTROLLER_SOURCE: self.controller,
                LINEAGE_SOURCE: self.lineage,
            }
        )


@dataclass(frozen=True, slots=True)
class FreshG111RuntimeSchema:
    """Independent expected topology derived from newly constructed runtime."""

    expected: ExpectedTransactionSchema
    owner_refs: Mapping[str, tuple[RuntimeLeafRef, ...]]
    derived_lineage_refs: tuple[RuntimeLeafRef, ...]
    refs_by_source: Mapping[str, tuple[RuntimeLeafRef, ...]]
    barrier_binding: BarrierStateBinding
    barrier_prefix: str
    live_state_prefix: str
    require_live_state: bool


@dataclass(frozen=True, slots=True)
class G111BoundInventory:
    """Built and validated native-v3 transaction over real source arrays."""

    checkpoint_arrays: Mapping[str, np.ndarray]
    staged: StagedTransaction
    schema: FreshG111RuntimeSchema


@dataclass(frozen=True, slots=True)
class G111OwnerReplacement:
    """Detached source-shaped arrays for one later atomic trainer publication."""

    owner: str
    deploy: Mapping[str, np.ndarray]
    resume: Mapping[str, np.ndarray]
    barrier: Mapping[str, np.ndarray]
    controller: Mapping[str, np.ndarray]
    lineage: Mapping[str, np.ndarray]


def _normalize_runtime_arrays(
    runtime: G111RuntimeArrays,
) -> dict[str, dict[str, np.ndarray]]:
    if not isinstance(runtime, G111RuntimeArrays):
        raise TypeError("runtime must be G111RuntimeArrays")
    normalized: dict[str, dict[str, np.ndarray]] = {}
    for source, values in runtime.by_source().items():
        if not isinstance(values, Mapping):
            _fail(f"{source} arrays must be a mapping")
        source_arrays: dict[str, np.ndarray] = {}
        for raw_key, raw_value in values.items():
            key = _canonical_string(raw_key, label=f"{source} key")
            if key in source_arrays:
                _fail(f"{source} contains duplicate key {key!r}")
            array = np.asarray(raw_value)
            if array.dtype.hasobject or array.dtype.fields is not None:
                _fail(f"{source}.{key}: object and structured arrays are forbidden")
            source_arrays[key] = array
        normalized[source] = source_arrays
    return normalized


def _entry_spec(ref: RuntimeLeafRef, array: np.ndarray) -> EntrySpec:
    return EntrySpec(
        key=ref.qualified_key,
        dtype=array.dtype.str,
        shape=tuple(int(value) for value in array.shape),
        finite=array.dtype.kind in "fc",
        allow_empty=array.size == 0,
    )


def _normalize_owner_refs(
    owner_refs: Mapping[str, Iterable[RuntimeLeafRef]],
) -> dict[str, tuple[RuntimeLeafRef, ...]]:
    if not isinstance(owner_refs, Mapping) or tuple(owner_refs) != ATOMIC_OWNERS:
        _fail("owner inventory must contain O1--O6 in canonical order")
    normalized: dict[str, tuple[RuntimeLeafRef, ...]] = {}
    seen: dict[RuntimeLeafRef, str] = {}
    for owner in ATOMIC_OWNERS:
        refs = tuple(sorted(owner_refs[owner]))
        if not refs:
            _fail(f"active current-G111 owner {owner!r} has no runtime leaves")
        if len(set(refs)) != len(refs):
            _fail(f"owner {owner!r} repeats a runtime leaf")
        for ref in refs:
            if not isinstance(ref, RuntimeLeafRef):
                raise TypeError("owner inventory entries must be RuntimeLeafRef")
            previous = seen.setdefault(ref, owner)
            if previous != owner:
                _fail(f"runtime leaf {ref.source}.{ref.key} overlaps owners {previous!r} and {owner!r}")
        normalized[owner] = refs
    return normalized


def _require_current_barrier(
    arrays: Mapping[str, np.ndarray],
    *,
    prefix: str,
) -> None:
    expected = {f"{prefix}{field}" for field in _BARRIER_FIELDS}
    actual = set(arrays)
    if actual != expected:
        _fail(
            "current G111 O4 barrier census differs; "
            f"missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}"
        )


def _require_current_live_state(
    arrays: Mapping[str, np.ndarray],
    *,
    prefix: str,
) -> None:
    expected = {f"{prefix}{field}" for field in _LIVE_STATE_FIELDS}
    actual = {key for key in arrays if key.startswith(prefix)}
    if actual != expected:
        _fail(
            "current G111 live reducer/controller census differs; "
            f"missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}"
        )


def build_fresh_g111_runtime_schema(
    fresh_runtime: G111RuntimeArrays,
    *,
    owner_refs: Mapping[str, Iterable[RuntimeLeafRef]],
    derived_lineage_refs: Iterable[RuntimeLeafRef] = (),
    barrier_prefix: str = CURRENT_BARRIER_PREFIX,
    live_state_prefix: str = CURRENT_LIVE_STATE_PREFIX,
    require_live_state: bool = True,
) -> FreshG111RuntimeSchema:
    """Build independent topology from fresh objects, never checkpoint claims.

    ``owner_refs`` is the runtime adapter inventory.  It must cover every leaf
    in ``fresh_runtime`` exactly once, except explicitly derived O6 leaves.
    All six current-G111 owners are active and therefore must be nonempty.
    """

    barrier_prefix = _canonical_string(barrier_prefix, label="barrier prefix")
    live_state_prefix = _canonical_string(live_state_prefix, label="live-state prefix")
    if type(require_live_state) is not bool:
        _fail("require_live_state must be an exact bool")
    fresh = _normalize_runtime_arrays(fresh_runtime)
    owners = _normalize_owner_refs(owner_refs)
    derived = tuple(sorted(derived_lineage_refs))
    if len(set(derived)) != len(derived):
        _fail("derived lineage inventory repeats a runtime leaf")
    for ref in derived:
        if not isinstance(ref, RuntimeLeafRef):
            raise TypeError("derived lineage entries must be RuntimeLeafRef")
        if ref.source != LINEAGE_SOURCE:
            _fail("derived lineage leaves must come from the lineage source")

    owner_by_ref = {ref: owner for owner, refs in owners.items() for ref in refs}
    overlap = set(derived) & set(owner_by_ref)
    if overlap:
        _fail(f"derived lineage leaves overlap owner claims: {[ref.qualified_key for ref in sorted(overlap)]}")
    expected_refs = {RuntimeLeafRef(source, key) for source, arrays in fresh.items() for key in arrays}
    inventoried_refs = set(owner_by_ref) | set(derived)
    if inventoried_refs != expected_refs:
        _fail(
            "fresh runtime reverse coverage failed; "
            f"missing={[ref.qualified_key for ref in sorted(expected_refs - inventoried_refs)]}, "
            f"unknown={[ref.qualified_key for ref in sorted(inventoried_refs - expected_refs)]}"
        )
    for ref in expected_refs:
        if ref.source == BARRIER_SOURCE and owner_by_ref.get(ref) != VERDICT_TRANSACTION:
            _fail("every physical barrier leaf must be owned by O4 verdict_transaction")
        if ref.source == LINEAGE_SOURCE and (ref not in derived and owner_by_ref.get(ref) != LINEAGE_ENVELOPE):
            _fail("every non-derived lineage leaf must be owned by O6 lineage_envelope")

    _require_current_barrier(fresh[BARRIER_SOURCE], prefix=barrier_prefix)
    if require_live_state:
        _require_current_live_state(fresh[CONTROLLER_SOURCE], prefix=live_state_prefix)

    expected_owners: list[ExpectedOwnerSchema] = []
    for owner in ATOMIC_OWNERS:
        specs = tuple(_entry_spec(ref, fresh[ref.source][ref.key]) for ref in owners[owner])
        expected_owners.append(
            ExpectedOwnerSchema(
                owner=owner,
                active=True,
                required=specs,
                permitted=specs,
            )
        )
    expected = ExpectedTransactionSchema(
        owners=tuple(expected_owners),
        domain_coverage=canonical_domain_coverage(),
        derived_lineage=tuple(_entry_spec(ref, fresh[ref.source][ref.key]) for ref in derived),
    )
    refs_by_source = MappingProxyType(
        {source: tuple(sorted(ref for ref in expected_refs if ref.source == source)) for source in RUNTIME_SOURCES}
    )
    return FreshG111RuntimeSchema(
        expected=expected,
        owner_refs=MappingProxyType(owners),
        derived_lineage_refs=derived,
        refs_by_source=refs_by_source,
        barrier_binding=BarrierStateBinding.from_prefix(f"{BARRIER_SOURCE}.{barrier_prefix}"),
        barrier_prefix=barrier_prefix,
        live_state_prefix=live_state_prefix,
        require_live_state=require_live_state,
    )


def _qualify_runtime(
    runtime: G111RuntimeArrays,
    schema: FreshG111RuntimeSchema,
) -> dict[str, np.ndarray]:
    actual = _normalize_runtime_arrays(runtime)
    for source in RUNTIME_SOURCES:
        expected_keys = {ref.key for ref in schema.refs_by_source[source]}
        actual_keys = set(actual[source])
        if actual_keys != expected_keys:
            _fail(
                f"{source} runtime census differs from fresh topology; "
                f"missing={sorted(expected_keys - actual_keys)}, "
                f"unknown={sorted(actual_keys - expected_keys)}"
            )
    return {
        ref.qualified_key: actual[ref.source][ref.key]
        for source in RUNTIME_SOURCES
        for ref in schema.refs_by_source[source]
    }


def _runtime_invariants(
    schema: FreshG111RuntimeSchema,
) -> tuple[Any, ...]:
    def validate_barrier(arrays: Mapping[str, np.ndarray]) -> None:
        QuiescentVerdictTransaction.from_numpy_state(
            arrays,
            reducer=lambda state, _result: state,
            restored_reducer_state={},
            prefix=f"{BARRIER_SOURCE}.{schema.barrier_prefix}",
        )

    validators: list[Any] = [validate_barrier]
    if schema.require_live_state:

        def validate_live_state(arrays: Mapping[str, np.ndarray]) -> None:
            open_live_verdict_state(
                arrays,
                prefix=f"{CONTROLLER_SOURCE}.{schema.live_state_prefix}",
            )

        validators.append(validate_live_state)
    return tuple(validators)


def bind_g111_owner_inventory(
    runtime: G111RuntimeArrays,
    *,
    fresh_schema: FreshG111RuntimeSchema,
) -> G111BoundInventory:
    """Partition real leaves, build a manifest, and stage native-v3."""

    if not isinstance(fresh_schema, FreshG111RuntimeSchema):
        raise TypeError("fresh_schema must be FreshG111RuntimeSchema")
    payload = _qualify_runtime(runtime, fresh_schema)
    owner_claims = {
        owner: tuple(ref.qualified_key for ref in fresh_schema.owner_refs[owner]) for owner in ATOMIC_OWNERS
    }
    activity = dict.fromkeys(ATOMIC_OWNERS, True)
    manifest = build_manifest(
        payload,
        owner_claims=owner_claims,
        activity=activity,
        domain_coverage={row.domain: row.owners for row in canonical_domain_coverage()},
        derived_lineage_keys=tuple(ref.qualified_key for ref in fresh_schema.derived_lineage_refs),
    )
    checkpoint_arrays = {
        **payload,
        MANIFEST_KEY: manifest_array(manifest),
    }
    staged = validate_transaction(
        checkpoint_arrays,
        manifest,
        fresh_schema.expected,
        barrier_binding=fresh_schema.barrier_binding,
        invariant_validators=_runtime_invariants(fresh_schema),
    )
    return G111BoundInventory(
        checkpoint_arrays=staged.arrays,
        staged=staged,
        schema=fresh_schema,
    )


def reopen_g111_owner_inventory(
    path: str | Path,
    *,
    fresh_schema: FreshG111RuntimeSchema,
) -> G111BoundInventory:
    """Physically reopen and fully validate an existing native-v3 NPZ."""

    arrays = load_npz_staging(path)
    if MANIFEST_KEY not in arrays:
        _fail("physical native-v3 NPZ lacks its manifest")
    manifest = manifest_from_array(arrays[MANIFEST_KEY])
    staged = validate_transaction(
        arrays,
        manifest,
        fresh_schema.expected,
        barrier_binding=fresh_schema.barrier_binding,
        invariant_validators=_runtime_invariants(fresh_schema),
    )
    return G111BoundInventory(
        checkpoint_arrays=arrays,
        staged=staged,
        schema=fresh_schema,
    )


def write_and_reopen_g111_owner_inventory(
    path: str | Path,
    bound: G111BoundInventory,
) -> G111BoundInventory:
    """Atomically write exact staged arrays, then parse them back from NPZ."""

    if not isinstance(bound, G111BoundInventory):
        raise TypeError("bound must be G111BoundInventory")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.g111-tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez(handle, **bound.checkpoint_arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return reopen_g111_owner_inventory(
        destination,
        fresh_schema=bound.schema,
    )


def verify_g111_canonical_reserialization(
    reference: G111BoundInventory,
    runtime: G111RuntimeArrays,
) -> StagedTransaction:
    """Require a fresh capture to preserve all O1--O5 semantic hashes."""

    if not isinstance(reference, G111BoundInventory):
        raise TypeError("reference must be G111BoundInventory")
    payload = _qualify_runtime(runtime, reference.schema)
    return verify_canonical_reserialization(
        reference.staged,
        payload,
        reference.schema.expected,
        barrier_binding=reference.schema.barrier_binding,
        invariant_validators=_runtime_invariants(reference.schema),
    )


def staged_g111_owner_replacements(
    bound: G111BoundInventory,
) -> Mapping[str, G111OwnerReplacement]:
    """Return detached O1--O6 replacements for exactly one later publisher."""

    if not isinstance(bound, G111BoundInventory):
        raise TypeError("bound must be G111BoundInventory")
    claims = {claim.owner: set(claim.keys) for claim in bound.staged.manifest.owner_claims}
    claims[LINEAGE_ENVELOPE].update(bound.staged.manifest.derived_lineage_keys)
    replacements: dict[str, G111OwnerReplacement] = {}
    for owner in ATOMIC_OWNERS:
        by_source: dict[str, dict[str, np.ndarray]] = {source: {} for source in RUNTIME_SOURCES}
        for ref in (
            *bound.schema.owner_refs[owner],
            *(bound.schema.derived_lineage_refs if owner == LINEAGE_ENVELOPE else ()),
        ):
            if ref.qualified_key not in claims[owner]:
                _fail(f"staged owner {owner!r} lost leaf {ref.qualified_key!r}")
            by_source[ref.source][ref.key] = bound.staged.arrays[ref.qualified_key]
        frozen = {source: MappingProxyType(values) for source, values in by_source.items()}
        replacements[owner] = G111OwnerReplacement(
            owner=owner,
            deploy=frozen[DEPLOY_SOURCE],
            resume=frozen[RESUME_SOURCE],
            barrier=frozen[BARRIER_SOURCE],
            controller=frozen[CONTROLLER_SOURCE],
            lineage=frozen[LINEAGE_SOURCE],
        )
    return MappingProxyType(replacements)


__all__ = [
    "BARRIER_SOURCE",
    "CONTROLLER_SOURCE",
    "CURRENT_BARRIER_PREFIX",
    "CURRENT_LIVE_STATE_PREFIX",
    "DEPLOY_SOURCE",
    "LINEAGE_SOURCE",
    "RESUME_SOURCE",
    "RUNTIME_SOURCES",
    "FreshG111RuntimeSchema",
    "G111BoundInventory",
    "G111OwnerReplacement",
    "G111RuntimeArrays",
    "RuntimeLeafRef",
    "bind_g111_owner_inventory",
    "build_fresh_g111_runtime_schema",
    "reopen_g111_owner_inventory",
    "staged_g111_owner_replacements",
    "verify_g111_canonical_reserialization",
    "write_and_reopen_g111_owner_inventory",
]
