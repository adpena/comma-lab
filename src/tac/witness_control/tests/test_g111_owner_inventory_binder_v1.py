# SPDX-License-Identifier: MIT
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from tac.witness_control.g111_live_verdict_transaction_v1 import (
    PublisherCursor,
    new_reducer_state,
)
from tac.witness_control.g111_live_verdict_transaction_v1 import (
    state_arrays as live_state_arrays,
)
from tac.witness_control.g111_owner_inventory_binder_v1 import (
    BARRIER_SOURCE,
    CONTROLLER_SOURCE,
    CURRENT_BARRIER_PREFIX,
    CURRENT_LIVE_STATE_PREFIX,
    DEPLOY_SOURCE,
    LINEAGE_SOURCE,
    RESUME_SOURCE,
    G111RuntimeArrays,
    RuntimeLeafRef,
    bind_g111_owner_inventory,
    build_fresh_g111_runtime_schema,
    staged_g111_owner_replacements,
    verify_g111_canonical_reserialization,
    write_and_reopen_g111_owner_inventory,
)
from tac.witness_control.g111_verdict_barrier_v1 import (
    QuiescentVerdictTransaction,
)
from tac.witness_control.trajectory_transaction_v2 import (
    ATOMIC_OWNERS,
    CAUSAL_SELECTION_STATE,
    CURRENT_TRAIN_STATE,
    LINEAGE_ENVELOPE,
    MANIFEST_KEY,
    RESTORABLE_STATE_OWNERS,
    ROLLBACK_SAVEPOINT,
    SCHEDULE_CONTROL_STATE,
    VERDICT_TRANSACTION,
    TransactionValidationError,
)

REPO = Path(__file__).resolve().parents[4]
TRAINER = REPO / "experiments" / "train_levelset_witness_realized_through_R_mlx.py"

_O1_RESUME_EXACT = frozenset(
    {
        "__resume_epoch",
        "__resume_has_opt",
        "__resume_primary_optimizer_family",
        "__resume_stage",
        "__resume_seed_state_manifest_json",
        "__resume_seed_support_count",
        "__resume_seed_support_geometry_sha256",
        "__resume_has_seed",
        "__resume_active_trainable_components_json",
    }
)
_O6_RESUME_EXACT = frozenset(
    {
        "__basis_taper_unfolded",
        "__resume_semantic_schema",
    }
)
_O1_PREFIXES = (
    "liveP__",
    "emaP__",
    "optP__",
    "seedP__",
    "seedOptP__",
    "polyakM__",
)


def _resume_builder_literal_keys() -> frozenset[str]:
    """Read the real current builder without importing its MLX hot module."""

    tree = ast.parse(TRAINER.read_text())
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_build_resume_state_arrays"
    )
    keys: set[str] = set()
    for node in ast.walk(function):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == "out"
                and isinstance(target.slice, ast.Constant)
                and isinstance(target.slice.value, str)
            ):
                keys.add(target.slice.value)
    return frozenset(keys)


def _barrier_arrays() -> dict[str, np.ndarray]:
    transaction = QuiescentVerdictTransaction(
        reducer=lambda state, _result: state,
        initial_state={"real_controller_state": 0},
        max_journal_rows=64,
    )
    with transaction.checkpoint() as capture:
        return dict(capture.numpy_state(prefix=CURRENT_BARRIER_PREFIX))


def _controller_arrays() -> dict[str, np.ndarray]:
    controller = dict(
        live_state_arrays(
            new_reducer_state(history=[], closed_loop_verdicts=[]),
            PublisherCursor(),
            prefix=CURRENT_LIVE_STATE_PREFIX,
        )
    )
    controller.update(
        {
            # O2 mirrors the real rollback topology names frozen in the G111
            # inventory memo, including the distinct rbLiveP namespace.
            "__g111_rollback__count": np.asarray(0, np.int64),
            "__g111_rollback__savepoint_sha256": np.asarray("a" * 64),
            "rbLiveP__trunk.in_proj.weight": np.arange(6, dtype=np.float32).reshape(2, 3),
            # O3 is the pure replacement surface for later trainer wiring.
            "__g111_control__boundary_transition_count": np.asarray(3, np.int64),
            "__g111_control__last_boundary_epoch": np.asarray(8, np.int64),
            "__g111_control__tail_cycle": np.asarray(2, np.int64),
            # O5 is causal selection, not a toy owner sentinel.
            "__g111_selection__best_present": np.asarray(1, np.int8),
            "__g111_selection__best_metric": np.asarray(0.0012, np.float64),
            "__g111_selection__best_epoch": np.asarray(8, np.int64),
            "__g111_selection__best_result_id": np.asarray("verdict-8"),
            "__g111_selection__deploy_sha256": np.asarray("b" * 64),
            "__g111_selection__deploy_bytes": np.asarray(111840, np.int64),
        }
    )
    return controller


def _runtime() -> G111RuntimeArrays:
    resume = {
        # Actual _build_resume_state_arrays namespaces.
        "liveP__trunk.in_proj.weight": np.arange(6, dtype=np.float32).reshape(2, 3),
        "emaP__trunk.in_proj.weight": np.arange(6, dtype=np.float32).reshape(2, 3),
        "optP__trunk.in_proj.weight.m": np.ones((2, 3), np.float32),
        "seedP__residual.values": np.ones((3, 3), np.float32),
        "seedOptP__residual.values.m": np.ones((3, 3), np.float32),
        "polyakM__trunk.in_proj.weight": np.arange(6, dtype=np.float64).reshape(2, 3),
        "__resume_epoch": np.asarray(8, np.int64),
        "__resume_has_opt": np.asarray(1, np.int8),
        "__resume_primary_optimizer_family": np.asarray("adamw"),
        "__resume_semantic_schema": np.asarray("levelset_full_state.v3"),
        "__resume_stage": np.asarray("ce"),
        "__resume_seed_state_manifest_json": np.asarray('{"component":"island_seed"}'),
        "__resume_seed_support_count": np.asarray(3, np.int64),
        "__resume_seed_support_geometry_sha256": np.asarray("c" * 64),
        "__resume_has_seed": np.asarray(1, np.int8),
        "__resume_active_trainable_components_json": np.asarray('["primary_model","island_seed"]'),
        # Actual current registry/controller prefixes.
        "__rng_np_keys": np.arange(8, dtype=np.uint32),
        "__rng_np_pos": np.asarray(3, np.int64),
        "__mg_fired": np.asarray(1, np.int8),
        "__evt_stage_idx": np.asarray(1, np.int64),
        "__ta_step": np.asarray(4, np.int64),
        "__bc_fired_class": np.asarray([1, 2], np.int64),
        "__posegate_failures": np.asarray(0, np.int64),
        "__pvg_count": np.asarray([2], np.int64),
        "__raterolling_ep": np.asarray([6, 7], np.int64),
        "__dtp_event_mark_epoch": np.asarray(7, np.int64),
        "__recent_losses": np.asarray([0.3, 0.2], np.float64),
        "__hardness_prob": np.asarray([0.2, 0.8], np.float64),
    }
    # Census every fixed real key currently emitted by
    # _build_resume_state_arrays. Values are small, typed stand-ins; names and
    # source boundaries are the production ones.
    for key in _resume_builder_literal_keys():
        resume.setdefault(key, np.asarray(1, np.int64))

    return G111RuntimeArrays(
        deploy={
            "trunk.in_proj.weight": np.arange(6, dtype=np.float32).reshape(2, 3),
            "pose_carrier.dxi": np.zeros((2, 6), np.float32),
            "__cfg_n_hidden": np.asarray(2, np.int64),
            "__cfg_hidden_dim": np.asarray(96, np.int64),
            "__cfg_g109_target_projection_sha256": np.asarray("d" * 64),
            "__cfg_fresh_lineage_root_sha256": np.asarray("e" * 64),
        },
        resume=resume,
        barrier=_barrier_arrays(),
        controller=_controller_arrays(),
        lineage={
            "__cfg_fresh_lineage_schema": np.asarray("tac.fresh_producer_lineage.v1"),
            "__cfg_fresh_lineage_parent_checkpoint_id_sha256": np.asarray("f" * 64),
            "__cfg_fresh_lineage_state_sha256": np.asarray("0" * 64),
        },
    )


def _owner_refs(
    runtime: G111RuntimeArrays,
) -> tuple[dict[str, tuple[RuntimeLeafRef, ...]], tuple[RuntimeLeafRef, ...]]:
    by_source = runtime.by_source()
    resume_o1 = {key for key in by_source[RESUME_SOURCE] if key in _O1_RESUME_EXACT or key.startswith(_O1_PREFIXES)}
    resume_o6 = {
        key
        for key in by_source[RESUME_SOURCE]
        if key in _O6_RESUME_EXACT or key.startswith("__cfg_")
    }
    resume_o3 = set(by_source[RESUME_SOURCE]) - resume_o1 - resume_o6
    deploy_o6 = {key for key in by_source[DEPLOY_SOURCE] if key.startswith("__")}
    deploy_o5 = set(by_source[DEPLOY_SOURCE]) - deploy_o6
    controller = set(by_source[CONTROLLER_SOURCE])
    controller_o2 = {key for key in controller if key.startswith("__g111_rollback__") or key.startswith("rbLiveP__")}
    controller_o3 = {key for key in controller if key.startswith("__g111_control__")}
    controller_o5 = {key for key in controller if key.startswith("__g111_selection__")}
    controller_o4 = controller - controller_o2 - controller_o3 - controller_o5
    derived = (RuntimeLeafRef(LINEAGE_SOURCE, "__cfg_fresh_lineage_state_sha256"),)
    refs = {
        CURRENT_TRAIN_STATE: tuple(RuntimeLeafRef(RESUME_SOURCE, key) for key in resume_o1),
        ROLLBACK_SAVEPOINT: tuple(RuntimeLeafRef(CONTROLLER_SOURCE, key) for key in controller_o2),
        SCHEDULE_CONTROL_STATE: (
            *(RuntimeLeafRef(RESUME_SOURCE, key) for key in resume_o3),
            *(RuntimeLeafRef(CONTROLLER_SOURCE, key) for key in controller_o3),
        ),
        VERDICT_TRANSACTION: (
            *(RuntimeLeafRef(BARRIER_SOURCE, key) for key in by_source[BARRIER_SOURCE]),
            *(RuntimeLeafRef(CONTROLLER_SOURCE, key) for key in controller_o4),
        ),
        CAUSAL_SELECTION_STATE: (
            *(RuntimeLeafRef(DEPLOY_SOURCE, key) for key in deploy_o5),
            *(RuntimeLeafRef(CONTROLLER_SOURCE, key) for key in controller_o5),
        ),
        LINEAGE_ENVELOPE: (
            *(RuntimeLeafRef(DEPLOY_SOURCE, key) for key in deploy_o6),
            *(RuntimeLeafRef(RESUME_SOURCE, key) for key in resume_o6),
            *(
                RuntimeLeafRef(LINEAGE_SOURCE, key)
                for key in by_source[LINEAGE_SOURCE]
                if key != "__cfg_fresh_lineage_state_sha256"
            ),
        ),
    }
    return refs, derived


def _schema(runtime: G111RuntimeArrays):
    refs, derived = _owner_refs(runtime)
    return build_fresh_g111_runtime_schema(
        runtime,
        owner_refs=refs,
        derived_lineage_refs=derived,
    )


def _copy_runtime(runtime: G111RuntimeArrays) -> G111RuntimeArrays:
    return G111RuntimeArrays(
        **{
            source: {key: np.asarray(value).copy() for key, value in arrays.items()}
            for source, arrays in runtime.by_source().items()
        }
    )


def test_current_g111_active_config_census_is_owned_exactly_once() -> None:
    runtime = _runtime()
    literal_keys = _resume_builder_literal_keys()
    assert len(literal_keys) >= 130
    assert literal_keys <= set(runtime.resume)
    assert {
        "liveP__trunk.in_proj.weight",
        "emaP__trunk.in_proj.weight",
        "optP__trunk.in_proj.weight.m",
        "seedP__residual.values",
        "seedOptP__residual.values.m",
        "polyakM__trunk.in_proj.weight",
    } <= set(runtime.resume)

    bound = bind_g111_owner_inventory(runtime, fresh_schema=_schema(runtime))
    claims = {claim.owner: set(claim.keys) for claim in bound.staged.manifest.owner_claims}
    assert tuple(claims) == ATOMIC_OWNERS
    assert all(claims[owner] for owner in ATOMIC_OWNERS)
    assert set().union(*claims.values()) | set(bound.staged.manifest.derived_lineage_keys) == set(
        bound.checkpoint_arrays
    ) - {MANIFEST_KEY}

    replacements = staged_g111_owner_replacements(bound)
    assert tuple(replacements) == ATOMIC_OWNERS
    assert "liveP__trunk.in_proj.weight" in replacements[CURRENT_TRAIN_STATE].resume
    assert "__rng_np_keys" in replacements[SCHEDULE_CONTROL_STATE].resume
    assert f"{CURRENT_BARRIER_PREFIX}journal_sequences" in replacements[VERDICT_TRANSACTION].barrier
    assert "trunk.in_proj.weight" in replacements[CAUSAL_SELECTION_STATE].deploy
    assert "__cfg_n_hidden" in replacements[LINEAGE_ENVELOPE].resume
    assert not replacements[CURRENT_TRAIN_STATE].resume["liveP__trunk.in_proj.weight"].flags.writeable


def test_unknown_captured_key_is_refused_against_fresh_schema() -> None:
    fresh = _runtime()
    captured = _copy_runtime(fresh)
    captured.resume["__cfg_unknown_future_actuator"] = np.asarray(1, np.int8)
    with pytest.raises(TransactionValidationError, match=r"unknown=.*actuator"):
        bind_g111_owner_inventory(captured, fresh_schema=_schema(fresh))


def test_missing_active_key_is_refused_against_fresh_schema() -> None:
    fresh = _runtime()
    captured = _copy_runtime(fresh)
    del captured.resume["seedOptP__residual.values.m"]
    with pytest.raises(TransactionValidationError, match=r"missing=.*seedOptP"):
        bind_g111_owner_inventory(captured, fresh_schema=_schema(fresh))


def test_legacy_pending_key_is_refused_even_when_inventory_names_it() -> None:
    runtime = _runtime()
    runtime.resume["nested.__cl_pend_epoch"] = np.asarray(8, np.int64)
    refs, derived = _owner_refs(runtime)
    schema = build_fresh_g111_runtime_schema(
        runtime,
        owner_refs=refs,
        derived_lineage_refs=derived,
    )
    with pytest.raises(TransactionValidationError, match="pending-verdict"):
        bind_g111_owner_inventory(runtime, fresh_schema=schema)


def test_absent_active_owner_is_refused_by_fresh_runtime_schema() -> None:
    runtime = _runtime()
    refs, derived = _owner_refs(runtime)
    refs[ROLLBACK_SAVEPOINT] = ()
    with pytest.raises(TransactionValidationError, match="has no runtime leaves"):
        build_fresh_g111_runtime_schema(
            runtime,
            owner_refs=refs,
            derived_lineage_refs=derived,
        )


def test_overlap_between_owners_is_refused() -> None:
    runtime = _runtime()
    refs, derived = _owner_refs(runtime)
    collision = refs[CURRENT_TRAIN_STATE][0]
    refs[ROLLBACK_SAVEPOINT] = (
        *refs[ROLLBACK_SAVEPOINT],
        collision,
    )
    with pytest.raises(TransactionValidationError, match="overlaps owners"):
        build_fresh_g111_runtime_schema(
            runtime,
            owner_refs=refs,
            derived_lineage_refs=derived,
        )


def test_physical_npz_parseback_reopens_same_native_v3_transaction(
    tmp_path: Path,
) -> None:
    runtime = _runtime()
    bound = bind_g111_owner_inventory(runtime, fresh_schema=_schema(runtime))
    path = tmp_path / "g111_native_v3.npz"
    reopened = write_and_reopen_g111_owner_inventory(path, bound)
    assert path.is_file()
    assert reopened.staged.semantic_hash == bound.staged.semantic_hash
    assert reopened.staged.owner_semantic_hashes == bound.staged.owner_semantic_hashes
    assert reopened.staged.barrier_state is not None
    assert reopened.staged.barrier_state.pending_count == 0


def test_canonical_reserialization_equality_and_value_drift() -> None:
    runtime = _runtime()
    bound = bind_g111_owner_inventory(runtime, fresh_schema=_schema(runtime))
    equal = verify_g111_canonical_reserialization(bound, _copy_runtime(runtime))
    for owner in RESTORABLE_STATE_OWNERS:
        assert equal.owner_semantic_hashes[owner] == bound.staged.owner_semantic_hashes[owner]

    changed = _copy_runtime(runtime)
    changed.resume["liveP__trunk.in_proj.weight"][0, 0] += 1
    with pytest.raises(
        TransactionValidationError,
        match=r"changed semantic hash.*current_train_state",
    ):
        verify_g111_canonical_reserialization(bound, changed)
