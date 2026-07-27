# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tac.witness_control.g111_rollback_trajectory_state_v1 import (
    DEFAULT_PREFIX,
    OBSERVATION_ONLY_EXCLUDED_FIELDS,
    TREE_NAMES,
    TREE_PREFIXES,
    G111RollbackTrajectoryConfigV1,
    G111RollbackTrajectoryStateError,
    build_rollback_savepoint_topology,
    capture_rollback_trajectory_state,
    config_for_topology,
    cross_validate_rollback_against_o1,
    state_arrays,
    state_from_arrays,
)


def _sha(label: str) -> str:
    import hashlib

    return hashlib.sha256(label.encode()).hexdigest()


def _trees() -> dict[str, dict[str, np.ndarray]]:
    live = {
        "film.weight": np.arange(6, dtype=np.float32).reshape(2, 3),
        "trunk.bias": np.asarray([0.25, -0.5], dtype=np.float32),
    }
    return {
        "live": live,
        "ema": {key: value + np.float32(0.125) for key, value in live.items()},
        "opt": {
            "optimizers.0.state.film.weight.m": np.full((2, 3), 0.1, np.float32),
            "optimizers.0.state.film.weight.v": np.full((2, 3), 0.2, np.float32),
            "step": np.asarray(17, np.int64),
        },
        "film_polar": {
            "__film_polar_Q": np.eye(3, dtype=np.float32),
            "__film_polar_momentum": np.arange(6, dtype=np.float32).reshape(2, 3),
        },
        "seed": {"residual": np.arange(8, dtype=np.float32).reshape(2, 2, 2)},
        "seed_opt": {
            "state.residual.m": np.full((2, 2, 2), 0.3, np.float32),
            "state.residual.v": np.full((2, 2, 2), 0.4, np.float32),
            "step": np.asarray(9, np.int64),
        },
    }


def _topology():
    trees = _trees()
    return build_rollback_savepoint_topology(
        **trees,
        seed_support_geometry_sha256=_sha("seed-support"),
    )


def _config(topology=None) -> G111RollbackTrajectoryConfigV1:
    topology = _topology() if topology is None else topology
    return config_for_topology(
        typed_config_sha256=_sha("typed-config"),
        topology=topology,
        mode="rollback",
        window=4,
        frac=0.5,
        lr_cut=0.5,
        max_rollbacks=8,
        recent_losses_capacity=5,
    )


def _state(*, savepoint: bool = True):
    topology = _topology()
    config = _config(topology)
    return (
        capture_rollback_trajectory_state(
            config=config,
            topology=topology,
            rollbacks=2,
            events=(True, False, True),
            lr_scale=0.25,
            ep_spikes=3,
            ep_batches=7,
            recent_losses=(0.9, 0.8, 0.7),
            savepoint=_trees() if savepoint else None,
            snap_epoch=11 if savepoint else None,
            completed_optimizer_steps=123 if savepoint else None,
        ),
        config,
        topology,
    )


def _mutable(arrays):
    return {key: np.asarray(value).copy() for key, value in arrays.items()}


def _resign(arrays: dict[str, np.ndarray]) -> None:
    # Tests that target semantic padding need to get past the independent
    # aggregate integrity check.  Reimplementing the digest is intentionally
    # avoided: serialize a decoded semantic state is the sole public signer.
    from tac.witness_control import g111_rollback_trajectory_state_v1 as subject

    digest_key = f"{DEFAULT_PREFIX}state_sha256"
    arrays[digest_key] = np.frombuffer(
        subject._semantic_digest(arrays, digest_key=digest_key).encode(),
        dtype=np.uint8,
    ).copy()


def test_roundtrip_preserves_decision_complete_guard_and_savepoint() -> None:
    state, config, topology = _state()
    arrays = state_arrays(state, topology=topology)
    restored = state_from_arrays(
        arrays,
        expected_config=config,
        topology=topology,
    )

    assert restored.rollbacks == 2
    assert restored.events == (True, False, True)
    assert restored.lr_scale == 0.25
    assert restored.ep_spikes == 3
    assert restored.ep_batches == 7
    assert restored.recent_losses == (0.9, 0.8, 0.7)
    assert restored.savepoint.present
    assert restored.savepoint.snap_epoch == 11
    assert restored.savepoint.completed_optimizer_steps == 123
    for tree_name in TREE_NAMES:
        expected_tree = _trees()[tree_name]
        actual_tree = restored.savepoint.tree(tree_name)
        assert set(actual_tree) == set(expected_tree)
        for key, value in expected_tree.items():
            np.testing.assert_array_equal(actual_tree[key], value)
    cross_validate_rollback_against_o1(
        restored,
        expected_config=config,
        fresh_o1_topology=topology,
    )


def test_cold_and_live_arrays_have_identical_keys_dtypes_and_shapes() -> None:
    live, _, topology = _state(savepoint=True)
    cold, _, _ = _state(savepoint=False)
    live_arrays = state_arrays(live, topology=topology)
    cold_arrays = state_arrays(cold, topology=topology)

    assert set(cold_arrays) == set(live_arrays)
    assert {key: (value.dtype.str, value.shape) for key, value in cold_arrays.items()} == {
        key: (value.dtype.str, value.shape) for key, value in live_arrays.items()
    }
    for tree_prefix in TREE_PREFIXES.values():
        for key, value in cold_arrays.items():
            if key.startswith(tree_prefix):
                assert not np.any(value)
    assert int(cold_arrays[f"{DEFAULT_PREFIX}snap_epoch"]) == -1
    assert int(cold_arrays[f"{DEFAULT_PREFIX}completed_optimizer_steps"]) == 0


def test_cold_npz_is_pickle_free_and_restores_without_hidden_state(
    tmp_path: Path,
) -> None:
    cold, config, topology = _state(savepoint=False)
    arrays = state_arrays(cold, topology=topology)
    path = tmp_path / "o2.npz"
    np.savez(path, **arrays)
    with np.load(path, allow_pickle=False) as archive:
        reopened = {key: np.asarray(archive[key]) for key in archive.files}
    restored = state_from_arrays(
        reopened,
        expected_config=config,
        topology=topology,
    )
    assert not restored.savepoint.present
    assert all(not restored.savepoint.tree(name) for name in TREE_NAMES)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("typed_config_sha256", _sha("different-config")),
        ("o1_topology_sha256", _sha("different-topology")),
        ("seed_support_geometry_sha256", _sha("different-support")),
    ],
)
def test_config_topology_and_support_tampering_refuses(
    field: str,
    replacement: str,
) -> None:
    state, config, topology = _state()
    arrays = _mutable(state_arrays(state, topology=topology))
    arrays[f"{DEFAULT_PREFIX}{field}"] = np.frombuffer(replacement.encode(), dtype=np.uint8).copy()
    _resign(arrays)
    with pytest.raises(G111RollbackTrajectoryStateError):
        state_from_arrays(
            arrays,
            expected_config=config,
            topology=topology,
        )


def test_savepoint_value_tampering_refuses_before_publication() -> None:
    state, config, topology = _state()
    arrays = _mutable(state_arrays(state, topology=topology))
    arrays["rbLiveP__film.weight"][0, 0] += np.float32(1.0)
    with pytest.raises(
        G111RollbackTrajectoryStateError,
        match="SHA-256 differs",
    ):
        state_from_arrays(
            arrays,
            expected_config=config,
            topology=topology,
        )


def test_absent_savepoint_nonzero_hidden_leaf_refuses_even_if_resigned() -> None:
    state, config, topology = _state(savepoint=False)
    arrays = _mutable(state_arrays(state, topology=topology))
    arrays["rbLiveP__film.weight"][0, 0] = np.float32(1.0)
    _resign(arrays)
    with pytest.raises(
        G111RollbackTrajectoryStateError,
        match="nonzero hidden leaves",
    ):
        state_from_arrays(
            arrays,
            expected_config=config,
            topology=topology,
        )


@pytest.mark.parametrize(
    ("field", "index", "value", "message"),
    [
        ("events", 3, 1, "nonzero padding"),
        ("recent_losses", 4, 1.0, "nonzero padding"),
    ],
)
def test_fixed_capacity_padding_must_be_zero(
    field: str,
    index: int,
    value: float,
    message: str,
) -> None:
    state, config, topology = _state()
    arrays = _mutable(state_arrays(state, topology=topology))
    arrays[f"{DEFAULT_PREFIX}{field}"][index] = value
    _resign(arrays)
    with pytest.raises(G111RollbackTrajectoryStateError, match=message):
        state_from_arrays(
            arrays,
            expected_config=config,
            topology=topology,
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"events": (True, False, True, False, True)},
        {"recent_losses": (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)},
        {"rollbacks": 9},
        {"ep_spikes": 8, "ep_batches": 7},
    ],
)
def test_capture_refuses_overflow_and_impossible_counters(overrides) -> None:
    topology = _topology()
    config = _config(topology)
    kwargs = {
        "config": config,
        "topology": topology,
        "rollbacks": 2,
        "events": (True, False),
        "lr_scale": 0.25,
        "ep_spikes": 3,
        "ep_batches": 7,
        "recent_losses": (0.9, 0.8),
    }
    kwargs.update(overrides)
    with pytest.raises(G111RollbackTrajectoryStateError):
        capture_rollback_trajectory_state(**kwargs)


def test_topology_requires_exact_o1_live_ema_and_seed_support() -> None:
    trees = _trees()
    bad_ema = dict(trees["ema"])
    bad_ema["film.weight"] = np.zeros((3, 2), np.float32)
    with pytest.raises(
        G111RollbackTrajectoryStateError,
        match="live and EMA topologies",
    ):
        build_rollback_savepoint_topology(
            live=trees["live"],
            ema=bad_ema,
            opt=trees["opt"],
            film_polar=trees["film_polar"],
            seed=trees["seed"],
            seed_opt=trees["seed_opt"],
            seed_support_geometry_sha256=_sha("seed-support"),
        )
    with pytest.raises(
        G111RollbackTrajectoryStateError,
        match="requires its exact O1 support",
    ):
        build_rollback_savepoint_topology(**trees)


def test_cross_validation_rejects_different_fresh_o1_topology() -> None:
    state, config, topology = _state()
    trees = _trees()
    trees["opt"] = dict(trees["opt"])
    trees["opt"]["new_moment"] = np.asarray([1.0], np.float32)
    different = build_rollback_savepoint_topology(
        **trees,
        seed_support_geometry_sha256=_sha("seed-support"),
    )
    with pytest.raises(
        G111RollbackTrajectoryStateError,
        match="fresh O1 topology",
    ):
        cross_validate_rollback_against_o1(
            state,
            expected_config=config,
            fresh_o1_topology=different,
        )
    assert topology.sha256 != different.sha256


def test_object_dtype_and_unknown_owned_leaf_are_forbidden() -> None:
    trees = _trees()
    trees["opt"] = {"bad": np.asarray([{"pickle": True}], dtype=object)}
    with pytest.raises(
        G111RollbackTrajectoryStateError,
        match="object or structured",
    ):
        build_rollback_savepoint_topology(
            **trees,
            seed_support_geometry_sha256=_sha("seed-support"),
        )

    state, config, topology = _state()
    arrays = _mutable(state_arrays(state, topology=topology))
    arrays["rbLiveP__unknown"] = np.asarray([0.0], np.float32)
    with pytest.raises(
        G111RollbackTrajectoryStateError,
        match="unknown",
    ):
        state_from_arrays(
            arrays,
            expected_config=config,
            topology=topology,
        )


def test_exhausted_warned_is_explicitly_observation_only_and_unserialized() -> None:
    state, _, topology = _state()
    arrays = state_arrays(state, topology=topology)
    assert OBSERVATION_ONLY_EXCLUDED_FIELDS == ("exhausted_warned",)
    assert all("exhausted_warned" not in key for key in arrays)


def test_legacy_mode_refuses_rollback_only_state() -> None:
    topology = _topology()
    config = config_for_topology(
        typed_config_sha256=_sha("legacy-config"),
        topology=topology,
        mode="legacy",
        window=4,
        frac=0.5,
        lr_cut=0.5,
        max_rollbacks=8,
        recent_losses_capacity=5,
    )
    cold = capture_rollback_trajectory_state(
        config=config,
        topology=topology,
        rollbacks=0,
        events=(),
        lr_scale=1.0,
        ep_spikes=0,
        ep_batches=0,
        recent_losses=(0.9, 0.8),
    )
    assert not cold.savepoint.present

    with pytest.raises(
        G111RollbackTrajectoryStateError,
        match="legacy mode cannot carry rollback-only",
    ):
        capture_rollback_trajectory_state(
            config=config,
            topology=topology,
            rollbacks=0,
            events=(False,),
            lr_scale=1.0,
            ep_spikes=0,
            ep_batches=0,
            recent_losses=(0.9, 0.8),
        )
