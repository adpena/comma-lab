# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np
import pytest

from tac.witness_control.g111_live_verdict_transaction_v1 import (
    PublisherCursor,
    new_reducer_state,
)
from tac.witness_control.g111_live_verdict_transaction_v1 import (
    state_arrays as live_state_arrays,
)
from tac.witness_control.g111_native_stable_phase_restore_v1 import (
    G111StablePhaseRestoreError,
    restore_g111_stable_adamw,
    stage_g111_stable_adamw_restore,
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
)
from tac.witness_control.g111_rollback_trajectory_state_v1 import (
    TREE_PREFIXES,
    build_rollback_savepoint_topology,
    capture_rollback_trajectory_state,
    config_for_topology,
)
from tac.witness_control.g111_rollback_trajectory_state_v1 import (
    state_arrays as rollback_state_arrays,
)
from tac.witness_control.g111_schedule_control_state_v1 import (
    new_state as new_schedule_state,
)
from tac.witness_control.g111_schedule_control_state_v1 import (
    state_arrays as schedule_state_arrays,
)
from tac.witness_control.g111_verdict_barrier_v1 import (
    ImmutableVerdictResult,
    QuiescentVerdictTransaction,
)
from tac.witness_control.trajectory_transaction_v2 import (
    ATOMIC_OWNERS,
    CAUSAL_SELECTION_STATE,
    CURRENT_TRAIN_STATE,
    LINEAGE_ENVELOPE,
    MANIFEST_KEY,
    ROLLBACK_SAVEPOINT,
    SCHEDULE_CONTROL_STATE,
    VERDICT_TRANSACTION,
)

TYPED_CONFIG_SHA256 = "a" * 64


def _reducer(state, result):
    return {
        **state,
        "seen": [*state.get("seen", []), result.result_id],
    }


def _lineage_arrays() -> dict[str, np.ndarray]:
    encoded = ImmutableVerdictResult.capture(
        submission_seq=0,
        result_id="g111-o6-lineage-envelope",
        payload={
            "lineage": {
                "__cfg_fresh_lineage_checkpoint_id_sha256": np.asarray("b" * 64),
                "__cfg_fresh_lineage_epoch": np.asarray(0, np.int64),
            }
        },
    )
    payload = np.zeros(64 * 1024, dtype=np.uint8)
    payload[: len(encoded.payload_bytes)] = np.frombuffer(encoded.payload_bytes, dtype=np.uint8)
    return {
        "__g111_o6__schema": np.frombuffer(b"tac.g111_lineage_envelope_arrays.v1", dtype=np.uint8).copy(),
        "__g111_o6__payload": payload,
        "__g111_o6__payload_length": np.asarray(len(encoded.payload_bytes), np.int64),
        "__g111_o6__sha256": np.frombuffer(encoded.result_sha256.encode("ascii"), dtype=np.uint8).copy(),
    }


def _fixture(
    *,
    optimizer_family: str = "adamw",
    muon_switched: bool = False,
    film_polar_config: bool = False,
    film_polar_topology: bool = False,
    nested_phase_arrays: bool = False,
):
    live = {"weight": np.arange(6, dtype=np.float32).reshape(2, 3)}
    ema = {"weight": np.arange(6, dtype=np.float32).reshape(2, 3)}
    opt = {
        "weight.m": np.ones((2, 3), np.float32),
        "weight.v": np.full((2, 3), 2.0, np.float32),
    }
    film = {"phase": np.zeros((2, 3), np.float32)} if film_polar_topology else {}
    topology = build_rollback_savepoint_topology(
        live=live,
        ema=ema,
        opt=opt,
        film_polar=film,
        seed={},
        seed_opt={},
        seed_support_geometry_sha256=None,
    )
    rollback_config = config_for_topology(
        typed_config_sha256=TYPED_CONFIG_SHA256,
        topology=topology,
        mode="rollback",
        window=4,
        frac=0.5,
        lr_cut=0.5,
        max_rollbacks=2,
        recent_losses_capacity=3,
    )
    rollback_state = capture_rollback_trajectory_state(
        config=rollback_config,
        topology=topology,
        rollbacks=0,
        events=(),
        lr_scale=1.0,
        ep_spikes=0,
        ep_batches=0,
        recent_losses=(),
    )
    o2 = dict(rollback_state_arrays(rollback_state, topology=topology))

    schedule = new_schedule_state(
        typed_config_sha256=TYPED_CONFIG_SHA256,
        completed_epoch=0,
        next_epoch=1,
        accepted_optimizer_steps=0,
        stop_latched=False,
        control_scalars={"muon_switched": muon_switched},
        resume_control_arrays={
            "__resume_primary_optimizer_family": np.asarray(
                [[optimizer_family]] if nested_phase_arrays else [optimizer_family]
            ),
            "__resume_stage": np.asarray(["stageColdRoot"]),
            "__resume_semantic_schema": np.asarray(["levelset_full_state.v3"]),
            "__cfg_film_polar_chart_spel": np.asarray(
                [[int(film_polar_config)]] if nested_phase_arrays else [int(film_polar_config)],
                np.int64,
            ),
        },
    )
    o3 = dict(schedule_state_arrays(schedule, prefix="__g111_o3__"))

    transaction = QuiescentVerdictTransaction(
        reducer=_reducer,
        initial_state=new_reducer_state(),
        max_journal_rows=64,
    )
    with transaction.checkpoint() as capture:
        barrier = dict(capture.numpy_state(prefix=CURRENT_BARRIER_PREFIX))
    live_state = dict(
        live_state_arrays(
            capture.reducer_state,
            PublisherCursor(),
            prefix=CURRENT_LIVE_STATE_PREFIX,
        )
    )
    controller = {**o2, **o3, **live_state}
    resume = {
        **{f"liveP__{key}": value for key, value in live.items()},
        **{f"emaP__{key}": value for key, value in ema.items()},
        **{f"optP__{key}": value for key, value in opt.items()},
    }
    runtime = G111RuntimeArrays(
        deploy={"weight": ema["weight"], "__cfg_model": np.asarray(1)},
        resume=resume,
        barrier=barrier,
        controller=controller,
        lineage=_lineage_arrays(),
    )
    rollback_prefixes = tuple(TREE_PREFIXES.values())
    o4_live = {key for key in live_state if key.startswith(f"{CURRENT_LIVE_STATE_PREFIX}o4_")}
    o5_live = set(live_state) - o4_live
    refs = {
        CURRENT_TRAIN_STATE: (
            *(RuntimeLeafRef(DEPLOY_SOURCE, key) for key in runtime.deploy),
            *(RuntimeLeafRef(RESUME_SOURCE, key) for key in runtime.resume),
        ),
        ROLLBACK_SAVEPOINT: tuple(
            RuntimeLeafRef(CONTROLLER_SOURCE, key)
            for key in o2
            if key.startswith("__g111_rollback__") or key.startswith(rollback_prefixes)
        ),
        SCHEDULE_CONTROL_STATE: tuple(RuntimeLeafRef(CONTROLLER_SOURCE, key) for key in o3),
        VERDICT_TRANSACTION: (
            *(RuntimeLeafRef(BARRIER_SOURCE, key) for key in barrier),
            *(RuntimeLeafRef(CONTROLLER_SOURCE, key) for key in o4_live),
        ),
        CAUSAL_SELECTION_STATE: tuple(RuntimeLeafRef(CONTROLLER_SOURCE, key) for key in o5_live),
        LINEAGE_ENVELOPE: tuple(RuntimeLeafRef(LINEAGE_SOURCE, key) for key in runtime.lineage),
    }
    schema = build_fresh_g111_runtime_schema(runtime, owner_refs=refs)
    bound = bind_g111_owner_inventory(runtime, fresh_schema=schema)
    return bound, rollback_config, topology


def _stage(bound, rollback_config, topology, **overrides):
    return stage_g111_stable_adamw_restore(
        bound,
        expected_typed_config_sha256=overrides.pop("expected_typed_config_sha256", TYPED_CONFIG_SHA256),
        rollback_config=rollback_config,
        rollback_topology=topology,
        verdict_reducer=overrides.pop("verdict_reducer", _reducer),
        **overrides,
    )


def test_real_o1_o6_stable_adamw_state_is_rehydrated():
    bound, rollback_config, topology = _fixture()
    plan = _stage(bound, rollback_config, topology)
    assert tuple(plan.owner_replacements) == ATOMIC_OWNERS
    assert plan.optimizer_family == "adamw"
    assert plan.stage == "stageColdRoot"
    assert plan.completed_epoch == 0
    assert plan.next_epoch == 1
    assert plan.accepted_optimizer_steps == 0
    assert plan.verdict_transaction.next_submit_seq == 0
    assert plan.publisher_cursor.next_effect_sequence == 0
    assert plan.lineage_state["__cfg_fresh_lineage_checkpoint_id_sha256"].item() == "b" * 64


def test_all_replacement_arrays_are_detached_and_read_only():
    bound, rollback_config, topology = _fixture()
    plan = _stage(bound, rollback_config, topology)
    for replacement in plan.owner_replacements.values():
        for source in (
            replacement.deploy,
            replacement.resume,
            replacement.barrier,
            replacement.controller,
            replacement.lineage,
        ):
            assert all(not value.flags.writeable for value in source.values())


def test_one_publication_callback_receives_complete_plan_and_mutates_runtime():
    bound, rollback_config, topology = _fixture()
    runtime = {"generation": 0, "plan": None}
    calls = []

    def publish(plan):
        calls.append(plan)
        runtime.update(generation=1, plan=plan)
        return "published"

    result = restore_g111_stable_adamw(
        bound,
        expected_typed_config_sha256=TYPED_CONFIG_SHA256,
        rollback_config=rollback_config,
        rollback_topology=topology,
        verdict_reducer=_reducer,
        publish=publish,
    )
    assert result == "published"
    assert len(calls) == 1
    assert runtime["generation"] == 1
    assert runtime["plan"] is calls[0]


@pytest.mark.parametrize(
    ("optimizer_family", "muon_switched", "match"),
    (
        ("muon_multioptimizer", True, "Muon"),
        ("muon_multioptimizer", False, "optimizer family"),
        ("adamw", True, "Muon"),
    ),
)
def test_muon_or_inconsistent_optimizer_phase_is_refused(optimizer_family, muon_switched, match):
    bound, rollback_config, topology = _fixture(
        optimizer_family=optimizer_family,
        muon_switched=muon_switched,
    )
    with pytest.raises(G111StablePhaseRestoreError, match=match):
        _stage(bound, rollback_config, topology)


def test_muon_phase_is_refused_before_publication():
    bound, rollback_config, topology = _fixture(
        optimizer_family="muon_multioptimizer",
        muon_switched=True,
    )
    calls = []
    with pytest.raises(G111StablePhaseRestoreError, match="Muon"):
        restore_g111_stable_adamw(
            bound,
            expected_typed_config_sha256=TYPED_CONFIG_SHA256,
            rollback_config=rollback_config,
            rollback_topology=topology,
            verdict_reducer=_reducer,
            publish=calls.append,
        )
    assert calls == []


def test_nested_phase_control_array_is_refused():
    bound, rollback_config, topology = _fixture(nested_phase_arrays=True)
    with pytest.raises(G111StablePhaseRestoreError, match="one scalar or one-element vector"):
        _stage(bound, rollback_config, topology)


@pytest.mark.parametrize(
    ("film_polar_config", "film_polar_topology"),
    ((True, False), (False, True), (True, True)),
)
def test_film_polar_config_or_topology_is_refused(film_polar_config, film_polar_topology):
    bound, rollback_config, topology = _fixture(
        film_polar_config=film_polar_config,
        film_polar_topology=film_polar_topology,
    )
    with pytest.raises(G111StablePhaseRestoreError, match="film-polar"):
        _stage(bound, rollback_config, topology)


def test_wrong_typed_config_is_refused_before_publication():
    bound, rollback_config, topology = _fixture()
    calls = []
    with pytest.raises(G111StablePhaseRestoreError, match="typed config"):
        restore_g111_stable_adamw(
            bound,
            expected_typed_config_sha256="c" * 64,
            rollback_config=rollback_config,
            rollback_topology=topology,
            verdict_reducer=_reducer,
            publish=calls.append,
        )
    assert calls == []


def test_noncanonical_typed_config_is_refused():
    bound, rollback_config, topology = _fixture()
    with pytest.raises(G111StablePhaseRestoreError, match="canonical lowercase SHA-256"):
        _stage(
            bound,
            rollback_config,
            topology,
            expected_typed_config_sha256="A" * 64,
        )


def test_missing_o1_optimizer_leaf_is_refused_against_fresh_topology():
    bound, rollback_config, topology = _fixture()
    foreign_topology = build_rollback_savepoint_topology(
        live={"weight": np.arange(6, dtype=np.float32).reshape(2, 3)},
        ema={"weight": np.arange(6, dtype=np.float32).reshape(2, 3)},
        opt={"weight.m": np.ones((2, 3), np.float32)},
        film_polar={},
        seed={},
        seed_opt={},
        seed_support_geometry_sha256=None,
    )
    foreign_config = config_for_topology(
        typed_config_sha256=TYPED_CONFIG_SHA256,
        topology=foreign_topology,
        mode="rollback",
        window=4,
        frac=0.5,
        lr_cut=0.5,
        max_rollbacks=2,
        recent_losses_capacity=3,
    )
    with pytest.raises(G111StablePhaseRestoreError, match=r"unknown=.*weight.v"):
        _stage(bound, foreign_config, foreign_topology)


def test_o1_leaf_dtype_shape_is_refused_against_independent_topology():
    bound, _, _ = _fixture()
    foreign_topology = build_rollback_savepoint_topology(
        live={"weight": np.arange(6, dtype=np.float64).reshape(2, 3)},
        ema={"weight": np.arange(6, dtype=np.float64).reshape(2, 3)},
        opt={
            "weight.m": np.ones((2, 3), np.float32),
            "weight.v": np.full((2, 3), 2.0, np.float32),
        },
        film_polar={},
        seed={},
        seed_opt={},
        seed_support_geometry_sha256=None,
    )
    foreign_config = config_for_topology(
        typed_config_sha256=TYPED_CONFIG_SHA256,
        topology=foreign_topology,
        mode="rollback",
        window=4,
        frac=0.5,
        lr_cut=0.5,
        max_rollbacks=2,
        recent_losses_capacity=3,
    )
    with pytest.raises(G111StablePhaseRestoreError, match="dtype/shape differs"):
        _stage(bound, foreign_config, foreign_topology)


def test_tampered_o6_padding_is_refused_without_publication():
    bound, rollback_config, topology = _fixture()
    arrays = {key: np.asarray(value).copy() for key, value in bound.checkpoint_arrays.items()}
    payload_key = "lineage.__g111_o6__payload"
    arrays[payload_key][-1] = 1
    from tac.witness_control.trajectory_transaction_v2 import (
        build_manifest,
        manifest_array,
    )

    claims = {claim.owner: claim.keys for claim in bound.staged.manifest.owner_claims}
    activity = {row.owner: row.active for row in bound.staged.manifest.activity}
    manifest = build_manifest(
        {key: value for key, value in arrays.items() if key != MANIFEST_KEY},
        owner_claims=claims,
        activity=activity,
        domain_coverage={row.domain: row.owners for row in bound.staged.manifest.domain_coverage},
    )
    arrays[MANIFEST_KEY] = manifest_array(manifest)
    from tac.witness_control.trajectory_transaction_v2 import (
        validate_transaction,
    )

    rebound = type(bound)(
        checkpoint_arrays=arrays,
        staged=validate_transaction(
            arrays,
            manifest,
            bound.schema.expected,
            barrier_binding=bound.schema.barrier_binding,
        ),
        schema=bound.schema,
    )
    calls = []
    with pytest.raises(G111StablePhaseRestoreError, match="nonzero bytes"):
        restore_g111_stable_adamw(
            rebound,
            expected_typed_config_sha256=TYPED_CONFIG_SHA256,
            rollback_config=rollback_config,
            rollback_topology=topology,
            verdict_reducer=_reducer,
            publish=calls.append,
        )
    assert calls == []


@pytest.mark.parametrize("bad", (None, 1, "not-callable"))
def test_noncallable_publisher_is_refused(bad):
    bound, rollback_config, topology = _fixture()
    with pytest.raises(TypeError, match="publish must be callable"):
        restore_g111_stable_adamw(
            bound,
            expected_typed_config_sha256=TYPED_CONFIG_SHA256,
            rollback_config=rollback_config,
            rollback_topology=topology,
            verdict_reducer=_reducer,
            publish=bad,
        )


def test_wrong_bound_type_is_refused():
    _, rollback_config, topology = _fixture()
    with pytest.raises(TypeError, match="G111BoundInventory"):
        _stage(object(), rollback_config, topology)


def test_wrong_rollback_config_type_is_refused():
    bound, _, topology = _fixture()
    with pytest.raises(TypeError, match="rollback_config"):
        _stage(bound, object(), topology)


def test_wrong_rollback_topology_type_is_refused():
    bound, rollback_config, _ = _fixture()
    with pytest.raises(TypeError, match="rollback_topology"):
        _stage(bound, rollback_config, object())


def test_noncallable_verdict_reducer_is_refused():
    bound, rollback_config, topology = _fixture()
    with pytest.raises(TypeError, match="verdict_reducer"):
        _stage(
            bound,
            rollback_config,
            topology,
            verdict_reducer=None,
        )
