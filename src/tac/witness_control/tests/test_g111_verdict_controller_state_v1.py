from __future__ import annotations

import copy
import hashlib
from dataclasses import replace

import numpy as np
import pytest

from tac.witness_control.g111_live_verdict_transaction_v1 import (
    PublisherCursor,
    build_worker_snapshot,
    compact_acknowledged_state,
    new_reducer_state,
    reduce_result,
    run_worker,
)
from tac.witness_control.g111_verdict_barrier_v1 import ImmutableVerdictResult
from tac.witness_control.g111_verdict_controller_state_v1 import (
    G111VerdictControllerStateError,
    active_g111_controller_config_v1,
    adapt_live_reducer_effect,
    new_controller_state,
    reduce_controller_state,
    state_arrays,
    state_from_arrays,
    validate_controller_state,
)

CURRENT_TYPED_CONFIG_SHA256 = "9d05a943edfddc3eb9ff87910b5334305dd164977bcd976cdc30192cdc874be4"


def _config():
    return active_g111_controller_config_v1(typed_config_sha256=CURRENT_TYPED_CONFIG_SHA256)


def _scorer_snapshot(*, epoch: int, sequence: int) -> dict:
    return {
        "epoch": epoch,
        "ema_np": {
            "code": np.arange(24, dtype=np.float32).reshape(6, 4) + sequence,
            "layer.weight": np.arange(12, dtype=np.float32).reshape(3, 4),
        },
        "softmax_temp": 0.25,
        "hosc_beta": 2.0,
        # Exact active G111 has self-orient OFF.
        "dir": None,
        "pose_verdict_index": sequence,
        "pose_gate_engaged_epoch": -1,
        "expected_d_seg": 0.009 - sequence * 1e-5,
        "expected_d_pose": 0.0016 + sequence * 1e-7,
        "expected_annulus": 0.2 + sequence * 1e-6,
    }


def _score(scorer) -> dict:
    # Five-class, real-shaped sufficient counts. Classes 1 and 3 satisfy the
    # configured Morse-Smale birth predicate on the first observation.
    counts = {
        "total_px": 1_000,
        "pred_px": [600, 100, 100, 100, 100],
        "gt_px": [600, 100, 100, 100, 100],
        "wrong_px": [60, 10, 10, 10, 10],
        "n_classes": 5,
    }
    d_seg = float(scorer["expected_d_seg"])
    d_pose = float(scorer["expected_d_pose"])
    annulus_frac = float(scorer["expected_annulus"])
    return {
        "verdict": {
            "d_seg": d_seg,
            "d_pose": d_pose,
            "nucleus_counts": counts,
            "annulus": {
                "threshold": {
                    "annulus_flip_frac": annulus_frac,
                    "annulus_px": 324_000,
                },
                "bottom_k": {
                    "annulus_flip_frac": annulus_frac + 0.01,
                    "annulus_px": 128_000,
                },
            },
            "per_class": {
                "d_seg_by_class": [0.004, 0.002, 0.001, 0.0015, 0.0005],
                "flip_share_by_class": [0.6, 0.1, 0.1, 0.1, 0.1],
            },
            "_pose_gate_telemetry": {
                "pose_verdict_index": int(scorer["pose_verdict_index"]),
                "d_pose_live": True,
            },
        },
        "live_gap": {
            "d_seg_live": d_seg + 0.0001,
            "d_pose_live": d_pose + 0.0001,
        },
    }


def _snapshot(*, sequence: int, epoch: int) -> dict:
    return build_worker_snapshot(
        epoch=epoch,
        seg_form="tau_softplus",
        ep_loss=1.25 / epoch,
        blob_bytes=48_000 + sequence,
        best_eligible=False,
        closed_loop_enabled=True,
        liveness={
            "ema_updates": epoch * 20,
            "frac": 1.0,
            "stepped": True,
            "acc": 3,
            "skip": 0,
            "ep_tot": 3,
        },
        scorer_snapshot=_scorer_snapshot(epoch=epoch, sequence=sequence),
    )


def _live_effects(count: int) -> list[dict]:
    """Produce actual effects through the committed live reducer, then compact."""

    state = new_reducer_state()
    effects: list[dict] = []
    for sequence in range(count):
        result = run_worker(
            sequence,
            _snapshot(sequence=sequence, epoch=(sequence + 1) * 25),
            score_snapshot=_score,
        )
        state = reduce_result(state, result)
        effects.append(copy.deepcopy(state["effects"][-1]))
        state = compact_acknowledged_state(
            state,
            PublisherCursor(
                next_effect_sequence=sequence + 1,
                next_best_intent_sequence=0,
            ),
        )
    return effects


def _rehash_effect(effect: dict) -> None:
    effect["result_sha256"] = ImmutableVerdictResult.capture(
        submission_seq=effect["sequence"],
        result_id=effect["result_id"],
        payload=effect["payload"],
    ).result_sha256


def test_active_config_reduces_every_live_controller_surface_to_state_and_intents() -> None:
    effect = _live_effects(1)[0]
    config = _config()
    observation = adapt_live_reducer_effect(effect, config=config)
    reduced = reduce_controller_state(new_controller_state(config), observation)

    assert not reduced.replayed
    validate_controller_state(reduced.state)
    assert reduced.state["history"] == [
        {
            "epoch": 25,
            "d_seg": 0.009,
            "d_pose": 0.0016,
            "implied_S": observation["implied_S"],
            "blob_bytes": 48_000,
        }
    ]
    assert reduced.state["closed_loop_verdicts"] == [
        {
            "epoch": 25,
            "seg_form": "tau_softplus",
            "d_seg": 0.009,
            "ep_loss": 0.05,
        }
    ]
    assert reduced.state["nucleus_ready"] is True
    assert reduced.state["lane_sensor"]["epoch"] == 25
    assert reduced.state["lane_sensor"]["event"]["fired"] is True
    assert reduced.state["annulus_series"] == [{"epoch": 25, "annulus_flip_frac": 0.2}]
    # These schema-supported controller mutations are OFF in the exact active
    # config and therefore remain byte-neutral.
    assert reduced.state["label_floor_series"] == []
    assert reduced.state["last_d_pose"] is None
    assert reduced.state["ladder_costates"] == {
        "epoch": 25,
        "lane": pytest.approx(0.0002),
        "movable": pytest.approx(0.00015),
    }
    birth = reduced.state["birth_completion"]
    assert birth["fired_epochs"] == {"1": 25, "3": 25}
    assert len(birth["observations"]) == 1
    assert birth["observations"][0]["stats"]["1"]["gt_area"] == 0.1

    assert [intent["channel"] for intent in reduced.intents] == [
        "stdout",
        "telemetry",
        "causal",
    ]
    assert len({intent["idempotency_key"] for intent in reduced.intents}) == 3
    assert all(
        intent["result_id"] == effect["result_id"] and intent["result_sha256"] == effect["result_sha256"]
        for intent in reduced.intents
    )


def test_schema_models_label_floor_and_last_pose_when_their_typed_gates_are_on() -> None:
    config = replace(
        _config(),
        typed_config_sha256=hashlib.sha256(b"g111-controller-all-schema-surfaces").hexdigest(),
        label_floor_sensor_enabled=True,
        w_pose_law_enabled=True,
    )
    effect = _live_effects(1)[0]
    observation = adapt_live_reducer_effect(effect, config=config)
    reduced = reduce_controller_state(new_controller_state(config), observation)

    assert reduced.state["label_floor_series"] == [{"epoch": 25, "d_seg": 0.009, "seg_form": "tau_softplus"}]
    assert reduced.state["last_d_pose"] == 0.0016


def test_serialized_crash_replay_returns_identical_intents_and_continuation() -> None:
    effects = _live_effects(2)
    config = _config()
    cold_arrays = state_arrays(new_controller_state(config), prefix="g111_ctrl__")
    observations = [adapt_live_reducer_effect(effect, config=config) for effect in effects]
    first = reduce_controller_state(new_controller_state(config), observations[0])
    arrays = state_arrays(first.state, prefix="g111_ctrl__")
    assert set(arrays) == set(cold_arrays)
    assert all(arrays[key].shape == cold_arrays[key].shape for key in arrays)
    restored = state_from_arrays(
        arrays,
        prefix="g111_ctrl__",
        expected_config=config,
    )

    replay = reduce_controller_state(restored, observations[0])
    assert replay.replayed
    assert replay.state == first.state
    assert replay.intents == first.intents

    continuous = reduce_controller_state(first.state, observations[1])
    resumed = reduce_controller_state(replay.state, observations[1])
    assert resumed.state == continuous.state
    assert resumed.intents == continuous.intents
    assert np.array_equal(
        state_arrays(resumed.state, prefix="x__")["x__state_payload"],
        state_arrays(continuous.state, prefix="x__")["x__state_payload"],
    )


def test_all_trajectory_series_and_replay_journal_are_bounded() -> None:
    config = replace(
        _config(),
        typed_config_sha256=hashlib.sha256(b"g111-controller-bounds-all-on").hexdigest(),
        label_floor_sensor_enabled=True,
        w_pose_law_enabled=True,
    )
    state = new_controller_state(config)
    for effect in _live_effects(130):
        observation = adapt_live_reducer_effect(effect, config=config)
        state = reduce_controller_state(state, observation).state

    assert len(state["history"]) == config.history_limit == 128
    assert len(state["closed_loop_verdicts"]) == config.closed_loop_limit == 128
    assert len(state["annulus_series"]) == config.annulus_limit == 16
    assert len(state["label_floor_series"]) == config.label_floor_limit == 128
    assert len(state["journal"]) == config.journal_limit == 64
    assert state["journal_base_sequence"] == 66
    assert state["next_sequence"] == 130
    assert len(state["birth_completion"]["observations"]) == 1
    assert state["history"][0]["epoch"] == 75
    assert state["annulus_series"][0]["epoch"] == 2_875
    assert state["last_d_pose"] == pytest.approx(0.0016 + 129e-7)
    assert state["ladder_costates"] == {
        "epoch": 3_250,
        "lane": pytest.approx(0.0002),
        "movable": pytest.approx(0.00015),
    }
    with pytest.raises(
        G111VerdictControllerStateError,
        match="evicted controller result identity",
    ):
        reduce_controller_state(
            state,
            adapt_live_reducer_effect(_live_effects(1)[0], config=config),
        )


@pytest.mark.parametrize("mutation", ["sha", "counts", "annulus", "d_seg"])
def test_malformed_effect_is_refused_before_state_mutation(mutation: str) -> None:
    config = _config()
    effect = _live_effects(1)[0]
    if mutation == "sha":
        effect["result_sha256"] = "0" * 64
    elif mutation == "counts":
        effect["payload"]["verdict"]["nucleus_counts"]["pred_px"][0] -= 1
        _rehash_effect(effect)
    elif mutation == "annulus":
        effect["payload"]["verdict"]["annulus"] = {"threshold": {}}
        _rehash_effect(effect)
    else:
        effect["payload"]["verdict"]["d_seg"] = -0.01
        _rehash_effect(effect)

    state = new_controller_state(config)
    before = copy.deepcopy(state)
    with pytest.raises(G111VerdictControllerStateError):
        observation = adapt_live_reducer_effect(effect, config=config)
        reduce_controller_state(state, observation)
    assert state == before


def test_restore_refuses_tamper_and_wrong_config() -> None:
    config = _config()
    effect = _live_effects(1)[0]
    state = reduce_controller_state(
        new_controller_state(config),
        adapt_live_reducer_effect(effect, config=config),
    ).state
    arrays = dict(state_arrays(state, prefix="ctrl__"))

    wrong_config = replace(
        config,
        typed_config_sha256=hashlib.sha256(b"wrong-g111-config").hexdigest(),
    )
    with pytest.raises(
        G111VerdictControllerStateError,
        match="differs from expected config",
    ):
        state_from_arrays(
            arrays,
            prefix="ctrl__",
            expected_config=wrong_config,
        )

    arrays["ctrl__state_payload"] = arrays["ctrl__state_payload"].copy()
    arrays["ctrl__state_payload"][10] ^= np.uint8(1)
    with pytest.raises(Exception, match=r"SHA-256|canonical"):
        state_from_arrays(arrays, prefix="ctrl__")

    arrays = dict(state_arrays(state, prefix="ctrl__"))
    payload_length = int(arrays["ctrl__state_payload_length"])
    arrays["ctrl__state_payload"] = arrays["ctrl__state_payload"].copy()
    arrays["ctrl__state_payload"][payload_length] = np.uint8(1)
    with pytest.raises(G111VerdictControllerStateError, match="nonzero bytes"):
        state_from_arrays(arrays, prefix="ctrl__")


def test_active_ladder_refuses_missing_or_nonpartitioned_per_class_signal() -> None:
    config = _config()
    for mutation in ("missing", "bad_share"):
        effect = _live_effects(1)[0]
        if mutation == "missing":
            effect["payload"]["verdict"].pop("per_class")
        else:
            effect["payload"]["verdict"]["per_class"]["flip_share_by_class"] = [
                0.6,
                0.1,
                0.1,
                0.1,
                0.0,
            ]
        _rehash_effect(effect)
        with pytest.raises(G111VerdictControllerStateError, match="per_class"):
            adapt_live_reducer_effect(effect, config=config)
