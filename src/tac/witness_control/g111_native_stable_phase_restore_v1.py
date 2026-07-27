# SPDX-License-Identifier: MIT
"""Stable-AdamW restore transaction for a validated G111 native-v3 checkpoint.

The physical opener proves bytes and descriptors.  The owner-inventory binder
proves that those descriptors match a separately constructed fresh runtime.
Neither surface publishes live trainer state.  This module closes the next
boundary for the topology-stable part of the G111 schedule:

* semantically reopen O2 rollback state against an independent topology/config;
* semantically reopen O3 schedule state against the active typed config;
* require the checkpoint and schedule to still be in the AdamW phase;
* reject any film-polar topology/config until a topology-changing restore
  contract exists;
* reconstruct O4 with the caller's real verdict reducer and reopen O5 state;
* decode the fixed-capacity O6 lineage envelope;
* freeze detached O1--O6 replacement arrays; and
* invoke exactly one caller-supplied publication callback after all validation.

This module deliberately refuses Muon and film-polar checkpoints.  Reusing an
AdamW optimizer tree after either topology change would be a silent partial
restore, not crash-faithful continuation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

import numpy as np

from tac.witness_control.g111_live_verdict_transaction_v1 import (
    PublisherCursor,
)
from tac.witness_control.g111_live_verdict_transaction_v1 import (
    state_from_arrays as open_live_verdict_state,
)
from tac.witness_control.g111_owner_inventory_binder_v1 import (
    CURRENT_BARRIER_PREFIX,
    CURRENT_LIVE_STATE_PREFIX,
    G111BoundInventory,
    G111OwnerReplacement,
    staged_g111_owner_replacements,
)
from tac.witness_control.g111_rollback_trajectory_state_v1 import (
    G111RollbackTrajectoryConfigV1,
    G111RollbackTrajectoryStateV1,
    RollbackSavepointTopologyV1,
)
from tac.witness_control.g111_rollback_trajectory_state_v1 import (
    state_from_arrays as open_rollback_state,
)
from tac.witness_control.g111_schedule_control_state_v1 import (
    G111ScheduleControlStateError,
)
from tac.witness_control.g111_schedule_control_state_v1 import (
    state_from_arrays as open_schedule_state,
)
from tac.witness_control.g111_verdict_barrier_v1 import (
    ImmutableVerdictResult,
    QuiescentVerdictTransaction,
    ResultIntegrityError,
)
from tac.witness_control.trajectory_transaction_v2 import (
    ATOMIC_OWNERS,
    CAUSAL_SELECTION_STATE,
    CURRENT_TRAIN_STATE,
    LINEAGE_ENVELOPE,
    ROLLBACK_SAVEPOINT,
    SCHEDULE_CONTROL_STATE,
    VERDICT_TRANSACTION,
    TransactionValidationError,
    canonical_semantic_hash,
)

SUPPORTED_OPTIMIZER_FAMILY: Final = "adamw"
SCHEDULE_PREFIX: Final = "__g111_o3__"
LINEAGE_ARRAY_SCHEMA: Final = "tac.g111_lineage_envelope_arrays.v1"
LINEAGE_RESULT_ID: Final = "g111-o6-lineage-envelope"
LINEAGE_PAYLOAD_CAPACITY: Final = 64 * 1024
LINEAGE_KEYS: Final = frozenset(
    {
        "__g111_o6__schema",
        "__g111_o6__payload",
        "__g111_o6__payload_length",
        "__g111_o6__sha256",
    }
)


class G111StablePhaseRestoreError(TransactionValidationError):
    """Native-v3 state is not a complete topology-stable AdamW restore."""


def _fail(message: str) -> None:
    raise G111StablePhaseRestoreError(message)


def _canonical_sha256(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{name} must be a canonical lowercase SHA-256")
    return value


def _read_exact_scalar(value: object, *, name: str) -> object:
    array = np.asarray(value)
    if array.shape not in ((), (1,)) or array.dtype.hasobject or array.dtype.fields is not None:
        _fail(f"{name} must be one scalar or one-element vector value")
    result = array.reshape(()).item()
    return result.item() if isinstance(result, np.generic) else result


def _read_exact_string(value: object, *, name: str) -> str:
    result = _read_exact_scalar(value, name=name)
    if type(result) is not str or not result or result.strip() != result:
        _fail(f"{name} must be an exact non-empty canonical string")
    return result


def _read_exact_binary_flag(value: object, *, name: str) -> bool:
    result = _read_exact_scalar(value, name=name)
    if isinstance(result, (bool, np.bool_)):
        return bool(result)
    if isinstance(result, (int, np.integer)) and not isinstance(result, (bool, np.bool_)) and int(result) in (0, 1):
        return bool(result)
    _fail(f"{name} must be an exact binary flag")


def _freeze_arrays(
    values: Mapping[str, Any],
) -> Mapping[str, np.ndarray]:
    frozen: dict[str, np.ndarray] = {}
    for key in sorted(values):
        if type(key) is not str or not key or key.strip() != key:
            _fail("replacement array keys must be exact canonical strings")
        array = np.array(np.asarray(values[key]), copy=True, order="C")
        if array.dtype.hasobject or array.dtype.fields is not None:
            _fail(f"{key}: object and structured arrays are forbidden")
        array.setflags(write=False)
        frozen[key] = array
    return MappingProxyType(frozen)


def _freeze_replacement(
    replacement: G111OwnerReplacement,
) -> G111OwnerReplacement:
    return G111OwnerReplacement(
        owner=replacement.owner,
        deploy=_freeze_arrays(replacement.deploy),
        resume=_freeze_arrays(replacement.resume),
        barrier=_freeze_arrays(replacement.barrier),
        controller=_freeze_arrays(replacement.controller),
        lineage=_freeze_arrays(replacement.lineage),
    )


def _tree_keys(
    replacement: G111OwnerReplacement,
    *,
    prefix: str,
) -> frozenset[str]:
    return frozenset(key[len(prefix) :] for key in replacement.resume if key.startswith(prefix))


def _require_o1_topology(
    replacement: G111OwnerReplacement,
    *,
    topology: RollbackSavepointTopologyV1,
) -> None:
    if not replacement.deploy:
        _fail("O1 current_train_state has no deploy arrays")
    prefixes = {
        "live": "liveP__",
        "ema": "emaP__",
        "opt": "optP__",
        "seed": "seedP__",
        "seed_opt": "seedOptP__",
    }
    for tree_name, prefix in prefixes.items():
        actual = _tree_keys(replacement, prefix=prefix)
        expected_specs = {spec.key: spec for spec in topology.tree(tree_name)}
        expected = frozenset(expected_specs)
        if actual != expected:
            _fail(
                f"O1 {tree_name} keyset differs from independent rollback topology; "
                f"missing={sorted(expected - actual)}, "
                f"unknown={sorted(actual - expected)}"
            )
        for key, spec in expected_specs.items():
            array = replacement.resume[f"{prefix}{key}"]
            if array.dtype.str != spec.dtype or tuple(array.shape) != spec.shape:
                _fail(
                    f"O1 {tree_name}.{key} dtype/shape differs from independent "
                    f"rollback topology; ({array.dtype.str},{array.shape}) != "
                    f"({spec.dtype},{spec.shape})"
                )
    if not _tree_keys(replacement, prefix="optP__"):
        _fail("stable AdamW restore requires nonempty optimizer moments")
    if topology.tree("film_polar"):
        _fail(
            "film-polar rollback topology is unsupported by the stable AdamW "
            "restore; use a topology-changing restore contract"
        )


def _open_lineage_envelope(
    replacement: G111OwnerReplacement,
) -> Mapping[str, Any]:
    arrays = replacement.lineage
    if set(arrays) != LINEAGE_KEYS:
        _fail(
            "O6 lineage envelope key census differs; "
            f"missing={sorted(LINEAGE_KEYS - set(arrays))}, "
            f"unknown={sorted(set(arrays) - LINEAGE_KEYS)}"
        )
    schema = np.asarray(arrays["__g111_o6__schema"])
    if schema.dtype != np.dtype(np.uint8) or schema.ndim != 1:
        _fail("O6 lineage schema must be a one-dimensional uint8 array")
    try:
        schema_value = schema.tobytes(order="C").decode("utf-8")
    except UnicodeDecodeError as exc:
        raise G111StablePhaseRestoreError("O6 lineage schema is not valid UTF-8") from exc
    if schema_value != LINEAGE_ARRAY_SCHEMA:
        _fail("O6 lineage array schema differs")

    payload = np.asarray(arrays["__g111_o6__payload"])
    if payload.dtype != np.dtype(np.uint8) or payload.shape != (LINEAGE_PAYLOAD_CAPACITY,):
        _fail("O6 lineage payload has the wrong fixed dtype or capacity")
    length_array = np.asarray(arrays["__g111_o6__payload_length"])
    if length_array.dtype != np.dtype(np.int64) or length_array.shape != ():
        _fail("O6 lineage payload length must be an int64 scalar")
    length = int(length_array.item())
    if not 0 < length <= LINEAGE_PAYLOAD_CAPACITY:
        _fail("O6 lineage payload length lies outside fixed capacity")
    if bool(np.any(payload[length:])):
        _fail("O6 lineage payload has nonzero bytes after its declared length")

    sha_array = np.asarray(arrays["__g111_o6__sha256"])
    if sha_array.dtype != np.dtype(np.uint8) or sha_array.ndim != 1:
        _fail("O6 lineage SHA-256 must be a one-dimensional uint8 array")
    try:
        sha256 = sha_array.tobytes(order="C").decode("ascii")
    except UnicodeDecodeError as exc:
        raise G111StablePhaseRestoreError("O6 lineage SHA-256 is not ASCII") from exc
    _canonical_sha256(sha256, name="O6 lineage SHA-256")
    encoded = ImmutableVerdictResult(
        submission_seq=0,
        result_id=LINEAGE_RESULT_ID,
        payload_bytes=payload[:length].tobytes(order="C"),
        result_sha256=sha256,
    )
    try:
        encoded.validate()
    except ResultIntegrityError as exc:
        raise G111StablePhaseRestoreError("O6 lineage payload identity or SHA-256 differs") from exc
    wrapper = encoded.payload
    if set(wrapper) != {"lineage"} or not isinstance(wrapper["lineage"], Mapping):
        _fail("O6 payload must contain exactly one lineage mapping")
    return MappingProxyType(dict(wrapper["lineage"]))


@dataclass(frozen=True, slots=True)
class G111StableAdamWRestorePlan:
    """Fully staged native-v3 replacements ready for one live publication."""

    checkpoint_semantic_sha256: str
    typed_config_sha256: str
    optimizer_family: str
    stage: str
    completed_epoch: int
    next_epoch: int
    accepted_optimizer_steps: int
    owner_replacements: Mapping[str, G111OwnerReplacement]
    rollback_state: G111RollbackTrajectoryStateV1
    schedule_state: Mapping[str, Any]
    verdict_transaction: QuiescentVerdictTransaction
    verdict_reducer_state: Mapping[str, Any]
    publisher_cursor: PublisherCursor
    lineage_state: Mapping[str, Any]

    def __post_init__(self) -> None:
        if tuple(self.owner_replacements) != ATOMIC_OWNERS:
            _fail("restore plan owner replacements are not in O1--O6 order")
        object.__setattr__(
            self,
            "owner_replacements",
            MappingProxyType(dict(self.owner_replacements)),
        )


def stage_g111_stable_adamw_restore(
    bound: G111BoundInventory,
    *,
    expected_typed_config_sha256: str,
    rollback_config: G111RollbackTrajectoryConfigV1,
    rollback_topology: RollbackSavepointTopologyV1,
    verdict_reducer: Callable[[Any, ImmutableVerdictResult], Any],
) -> G111StableAdamWRestorePlan:
    """Validate and stage one topology-stable AdamW native-v3 restore.

    The expected config/topology and reducer come from freshly constructed
    runtime objects; checkpoint bytes cannot define their own admissibility.
    """

    if not isinstance(bound, G111BoundInventory):
        raise TypeError("bound must be a G111BoundInventory")
    if not isinstance(rollback_config, G111RollbackTrajectoryConfigV1):
        raise TypeError("rollback_config must be G111RollbackTrajectoryConfigV1")
    if not isinstance(rollback_topology, RollbackSavepointTopologyV1):
        raise TypeError("rollback_topology must be RollbackSavepointTopologyV1")
    if not callable(verdict_reducer):
        raise TypeError("verdict_reducer must be callable")
    typed_config_sha256 = _canonical_sha256(
        expected_typed_config_sha256,
        name="expected typed config SHA-256",
    )
    activity = tuple((row.owner, row.active) for row in bound.staged.manifest.activity)
    if activity != tuple((owner, True) for owner in ATOMIC_OWNERS):
        _fail("stable restore requires all six native-v3 owners active")

    replacements = MappingProxyType(
        {
            owner: _freeze_replacement(replacement)
            for owner, replacement in staged_g111_owner_replacements(bound).items()
        }
    )
    o1 = replacements[CURRENT_TRAIN_STATE]
    _require_o1_topology(o1, topology=rollback_topology)

    o2 = replacements[ROLLBACK_SAVEPOINT]
    rollback_state = open_rollback_state(
        o2.controller,
        expected_config=rollback_config,
        topology=rollback_topology,
    )

    o3 = replacements[SCHEDULE_CONTROL_STATE]
    try:
        schedule_state = open_schedule_state(
            o3.controller,
            prefix=SCHEDULE_PREFIX,
            expected_typed_config_sha256=typed_config_sha256,
        )
    except G111ScheduleControlStateError as exc:
        raise G111StablePhaseRestoreError(str(exc)) from exc
    control_scalars = schedule_state["control_scalars"]
    if control_scalars.get("muon_switched") is not False:
        _fail("Muon or ambiguous optimizer phase is unsupported by the stable AdamW restore")
    resume_control = schedule_state["resume_control_arrays"]
    if "__resume_primary_optimizer_family" not in resume_control:
        _fail("O3 lacks the primary optimizer family")
    optimizer_family = _read_exact_string(
        resume_control["__resume_primary_optimizer_family"],
        name="O3 primary optimizer family",
    )
    if optimizer_family != SUPPORTED_OPTIMIZER_FAMILY:
        _fail(f"optimizer family {optimizer_family!r} is unsupported; stable restore requires 'adamw'")
    if "__resume_stage" not in resume_control:
        _fail("O3 lacks the checkpoint stage")
    stage = _read_exact_string(
        resume_control["__resume_stage"],
        name="O3 checkpoint stage",
    )
    if "__cfg_film_polar_chart_spel" not in resume_control:
        _fail("O3 lacks explicit film-polar configuration custody")
    if _read_exact_binary_flag(
        resume_control["__cfg_film_polar_chart_spel"],
        name="O3 film-polar configuration",
    ):
        _fail(
            "film-polar configuration is unsupported by the stable AdamW "
            "restore; use a topology-changing restore contract"
        )

    live_arrays = {
        **replacements[VERDICT_TRANSACTION].controller,
        **replacements[CAUSAL_SELECTION_STATE].controller,
    }
    reducer_state, publisher_cursor = open_live_verdict_state(
        live_arrays,
        prefix=CURRENT_LIVE_STATE_PREFIX,
    )
    verdict_transaction = QuiescentVerdictTransaction.from_numpy_state(
        replacements[VERDICT_TRANSACTION].barrier,
        reducer=verdict_reducer,
        restored_reducer_state=reducer_state,
        prefix=CURRENT_BARRIER_PREFIX,
    )
    if verdict_transaction.next_submit_seq != verdict_transaction.next_apply_seq:
        _fail("checkpointed O4 verdict transaction is not quiescent")
    if verdict_transaction.next_apply_seq != int(reducer_state["next_effect_sequence"]):
        _fail("O4 barrier apply cursor differs from the live reducer effect cursor")

    lineage_state = _open_lineage_envelope(replacements[LINEAGE_ENVELOPE])
    coordinate = schedule_state["coordinate"]
    return G111StableAdamWRestorePlan(
        checkpoint_semantic_sha256=canonical_semantic_hash(bound.staged.manifest),
        typed_config_sha256=typed_config_sha256,
        optimizer_family=optimizer_family,
        stage=stage,
        completed_epoch=int(coordinate["completed_epoch"]),
        next_epoch=int(coordinate["next_epoch"]),
        accepted_optimizer_steps=int(coordinate["accepted_optimizer_steps"]),
        owner_replacements=replacements,
        rollback_state=rollback_state,
        schedule_state=MappingProxyType(dict(schedule_state)),
        verdict_transaction=verdict_transaction,
        verdict_reducer_state=MappingProxyType(reducer_state),
        publisher_cursor=publisher_cursor,
        lineage_state=lineage_state,
    )


def restore_g111_stable_adamw(
    bound: G111BoundInventory,
    *,
    expected_typed_config_sha256: str,
    rollback_config: G111RollbackTrajectoryConfigV1,
    rollback_topology: RollbackSavepointTopologyV1,
    verdict_reducer: Callable[[Any, ImmutableVerdictResult], Any],
    publish: Callable[[G111StableAdamWRestorePlan], Any],
) -> Any:
    """Stage every owner, then invoke one live publication callback.

    Validation and rehydration of replacement objects complete before
    ``publish`` is called.  The callback is the sole trainer-specific state-swap
    boundary; this helper never mutates a model, optimizer, controller, BEST
    pointer, or lineage tip itself.
    """

    if not callable(publish):
        raise TypeError("publish must be callable")
    plan = stage_g111_stable_adamw_restore(
        bound,
        expected_typed_config_sha256=expected_typed_config_sha256,
        rollback_config=rollback_config,
        rollback_topology=rollback_topology,
        verdict_reducer=verdict_reducer,
    )
    return publish(plan)


__all__ = [
    "SUPPORTED_OPTIMIZER_FAMILY",
    "G111StableAdamWRestorePlan",
    "G111StablePhaseRestoreError",
    "restore_g111_stable_adamw",
    "stage_g111_stable_adamw_restore",
]
