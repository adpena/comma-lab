from __future__ import annotations

import inspect
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from tac.witness_control.g111_live_verdict_transaction_v1 import (
    SERIALIZED_MAX_BEST_INTENT_ROWS,
    SERIALIZED_MAX_EFFECT_ROWS,
    SERIALIZED_O4_FIELDS,
    SERIALIZED_O5_FIELDS,
    LiveVerdictEffectPublicationError,
    LiveVerdictStateError,
    MainThreadVerdictEffectPublisher,
    PublisherCursor,
    build_worker_snapshot,
    compact_acknowledged_state,
    new_reducer_state,
    reduce_result,
    run_worker,
    state_arrays,
    state_from_arrays,
    validate_reducer_state,
)
from tac.witness_control.g111_verdict_barrier_v1 import (
    ExternalEffectTransitionError,
    QuiescentVerdictTransaction,
    TransactionPoisonedError,
    WorkerExecutionError,
)


def _scorer_snapshot(*, epoch: int, pose_index: int) -> dict:
    return {
        "epoch": epoch,
        "ema_np": {
            "code": np.arange(24, dtype=np.float32).reshape(6, 4),
            "layer.weight": np.arange(12, dtype=np.float32).reshape(3, 4),
        },
        "softmax_temp": 0.25,
        "hosc_beta": 2.0,
        # Real self-orient shape: positional tuple, never an int-key mapping.
        "dir": (
            np.arange(8, dtype=np.float32).reshape(4, 2),
            np.arange(8, 16, dtype=np.float32).reshape(4, 2),
            np.arange(16, 24, dtype=np.float32).reshape(4, 2),
        ),
        "pose_verdict_index": pose_index,
        "pose_gate_engaged_epoch": -1,
    }


def _snapshot(
    *,
    epoch: int,
    d_seg: float,
    d_pose: float | None = 0.002,
    best_eligible: bool = True,
) -> dict:
    scorer = _scorer_snapshot(epoch=epoch, pose_index=epoch + 10)
    scorer["expected_d_seg"] = d_seg
    scorer["expected_d_pose"] = d_pose
    return build_worker_snapshot(
        epoch=epoch,
        seg_form="tau_softplus",
        ep_loss=1.25 / epoch,
        blob_bytes=48_000 + epoch,
        best_eligible=best_eligible,
        closed_loop_enabled=True,
        liveness={
            "ema_updates": epoch * 20,
            "frac": 1.0,
            "stepped": True,
            "acc": 3,
            "skip": 0,
            "ep_tot": 3,
        },
        scorer_snapshot=scorer,
    )


def _score(scorer) -> dict:
    return {
        "verdict": {
            "d_seg": float(scorer["expected_d_seg"]),
            "d_pose": scorer["expected_d_pose"],
            "_pose_gate_telemetry": {
                "pose_verdict_index": int(scorer["pose_verdict_index"]),
                "d_pose_live": scorer["expected_d_pose"] is not None,
            },
        },
        "live_gap": {"d_seg_live": float(scorer["expected_d_seg"]) + 0.001},
    }


def _worker(sequence: int, snapshot) -> object:
    return run_worker(sequence, snapshot, score_snapshot=_score)


def _new_transaction() -> QuiescentVerdictTransaction:
    return QuiescentVerdictTransaction(
        reducer=reduce_result,
        initial_state=new_reducer_state(),
        max_journal_rows=8,
    )


def test_worker_is_data_only_and_main_thread_publishes_before_decide() -> None:
    main_ident = threading.get_ident()
    worker_idents: list[int] = []
    physical: list[tuple[str, str, int]] = []
    transaction = _new_transaction()
    publisher = MainThreadVerdictEffectPublisher()

    def score_with_thread_sentinel(scorer) -> dict:
        worker_idents.append(threading.get_ident())
        return _score(scorer)

    def worker(sequence: int, snapshot):
        return run_worker(
            sequence,
            snapshot,
            score_snapshot=score_with_thread_sentinel,
        )

    with ThreadPoolExecutor(max_workers=1) as executor:
        assert transaction.submit(executor, worker, _snapshot(epoch=1, d_seg=0.02)) == 0
        with transaction.checkpoint() as capture:
            # Reduction is pure: no telemetry/controller/BEST callback fired.
            assert physical == []
            # The replacement controller is the sole durable O4 history owner.
            assert capture.reducer_state["history"] == []
            assert capture.reducer_state["closed_loop_verdicts"] == []
            publisher.publish_pending(
                capture.reducer_state,
                publish_effect=lambda effect: physical.append(
                    ("effect", effect["result_id"], threading.get_ident())
                ),
                publish_best=lambda effect, intent: physical.append(
                    ("best", intent["result_id"], threading.get_ident())
                ),
            )
        physical.append(("decide", "epoch-2", threading.get_ident()))

    assert worker_idents and all(ident != main_ident for ident in worker_idents)
    assert [kind for kind, _, _ in physical] == ["effect", "best", "decide"]
    assert all(ident == main_ident for _, _, ident in physical)


def test_cold_and_live_array_schemas_match_and_separate_o4_from_o5() -> None:
    prefix = "fixed__"
    cold = state_arrays(new_reducer_state(), PublisherCursor(), prefix=prefix)
    live_state = reduce_result(
        new_reducer_state(),
        run_worker(0, _snapshot(epoch=1, d_seg=0.02), score_snapshot=_score),
    )
    live = state_arrays(live_state, PublisherCursor(), prefix=prefix)

    assert set(cold) == set(live)
    assert {
        key: (array.dtype.str, array.shape)
        for key, array in cold.items()
    } == {
        key: (array.dtype.str, array.shape)
        for key, array in live.items()
    }
    o4 = {f"{prefix}o4_{field}" for field in SERIALIZED_O4_FIELDS}
    o5 = {f"{prefix}o5_{field}" for field in SERIALIZED_O5_FIELDS}
    assert set(live) == o4 | o5
    assert o4.isdisjoint(o5)
    assert all(key.startswith(f"{prefix}o4_") for key in o4)
    assert all(key.startswith(f"{prefix}o5_") for key in o5)
    assert not any(key.endswith("state_payload") for key in live)
    assert int(cold[f"{prefix}o4_effect_count"]) == 0
    assert int(live[f"{prefix}o4_effect_count"]) == 1
    assert int(cold[f"{prefix}o5_best_intent_count"]) == 0
    assert int(live[f"{prefix}o5_best_intent_count"]) == 1


def test_checkpoint_compacts_acknowledged_full_snapshots_but_retains_best_tail() -> None:
    state = new_reducer_state()
    publisher = MainThreadVerdictEffectPublisher()
    for sequence, d_seg in enumerate((0.03, 0.02, 0.01)):
        state = reduce_result(
            state,
            run_worker(sequence, _snapshot(epoch=sequence + 1, d_seg=d_seg), score_snapshot=_score),
        )
        publisher.publish_pending(
            state,
            publish_effect=lambda effect: None,
            publish_best=lambda effect, intent: None,
        )

    compacted = compact_acknowledged_state(state, publisher.cursor)
    assert compacted["effect_base_sequence"] == 3
    assert compacted["effects"] == []
    assert compacted["o5"]["best_intent_base_sequence"] == 2
    assert len(compacted["o5"]["best_intents"]) == 1
    assert compacted["o5"]["best_intents"][0]["d_seg"] == 0.01

    restored, cursor = state_from_arrays(
        state_arrays(state, publisher.cursor, prefix="live__"),
        prefix="live__",
    )
    validate_reducer_state(restored)
    assert restored["effect_base_sequence"] == compacted["effect_base_sequence"]
    assert restored["effects"] == []
    assert restored["history"] == []
    assert restored["closed_loop_verdicts"] == []
    assert restored["o5"]["best_d_seg"] == compacted["o5"]["best_d_seg"]
    assert np.array_equal(
        restored["o5"]["best_intents"][0]["artifact"]["ema_np"]["code"],
        compacted["o5"]["best_intents"][0]["artifact"]["ema_np"]["code"],
    )
    assert cursor == PublisherCursor(3, 3)


def test_nonzero_base_parseback_is_canonically_stable_and_replays_tail() -> None:
    state = new_reducer_state()
    for sequence, d_seg in enumerate((0.03, 0.04, 0.02)):
        state = reduce_result(
            state,
            run_worker(
                sequence,
                _snapshot(epoch=sequence + 1, d_seg=d_seg),
                score_snapshot=_score,
            ),
        )
    prefix = "tail__"
    arrays = state_arrays(state, PublisherCursor(1, 1), prefix=prefix)
    restored, cursor = state_from_arrays(arrays, prefix=prefix)
    reserialized = state_arrays(restored, cursor, prefix=prefix)
    assert set(reserialized) == set(arrays)
    assert all(
        np.array_equal(reserialized[key], arrays[key])
        for key in arrays
    )

    published: list[tuple[str, int]] = []
    publisher = MainThreadVerdictEffectPublisher(cursor=cursor)
    assert publisher.publish_pending(
        restored,
        publish_effect=lambda effect: published.append(
            ("effect", int(effect["sequence"]))
        ),
        publish_best=lambda effect, intent: published.append(
            ("best", int(intent["intent_sequence"]))
        ),
    ) == 2
    assert published == [("effect", 1), ("effect", 2), ("best", 1)]
    assert publisher.cursor == PublisherCursor(3, 2)


def test_crash_after_best_write_replays_same_content_addressed_bytes() -> None:
    state = reduce_result(
        new_reducer_state(),
        run_worker(0, _snapshot(epoch=1, d_seg=0.02), score_snapshot=_score),
    )
    publisher = MainThreadVerdictEffectPublisher()
    physical: dict[str, bytes] = {}
    fail_once = [True]

    def publish_effect(effect) -> None:
        key = f"effect:{effect['result_id']}"
        content = str(effect["result_sha256"]).encode()
        assert physical.setdefault(key, content) == content

    def publish_best(effect, intent) -> None:
        key = f"best:{intent['result_id']}"
        content = json.dumps(
            {
                "result_id": intent["result_id"],
                "result_sha256": intent["result_sha256"],
                "d_seg": intent["d_seg"],
                "epoch": intent["epoch"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        assert physical.setdefault(key, content) == content
        if fail_once[0]:
            fail_once[0] = False
            raise OSError("simulated crash after deterministic BEST replace")

    with pytest.raises(LiveVerdictEffectPublicationError):
        publisher.publish_pending(
            state,
            publish_effect=publish_effect,
            publish_best=publish_best,
        )
    # The durable cursor was still zero at the crash cut.  Restore therefore
    # replays both callbacks; content-addressed replacement is byte-identical.
    restored_state, restored_cursor = state_from_arrays(
        state_arrays(state, PublisherCursor(), prefix="crash__"),
        prefix="crash__",
    )
    before = dict(physical)
    restored_publisher = MainThreadVerdictEffectPublisher(cursor=restored_cursor)
    restored_publisher.publish_pending(
        restored_state,
        publish_effect=publish_effect,
        publish_best=publish_best,
    )
    assert physical == before
    assert restored_publisher.cursor == PublisherCursor(1, 1)


def test_worse_result_after_resume_does_not_emit_best_intent() -> None:
    first = reduce_result(
        new_reducer_state(),
        run_worker(0, _snapshot(epoch=1, d_seg=0.01), score_snapshot=_score),
    )
    publisher = MainThreadVerdictEffectPublisher()
    publisher.publish_pending(
        first,
        publish_effect=lambda effect: None,
        publish_best=lambda effect, intent: None,
    )
    resumed, _ = state_from_arrays(
        state_arrays(first, publisher.cursor, prefix="resume__"),
        prefix="resume__",
    )
    after_worse = reduce_result(
        resumed,
        run_worker(1, _snapshot(epoch=2, d_seg=0.02), score_snapshot=_score),
    )
    assert after_worse["o5"]["next_best_intent_sequence"] == 1
    assert after_worse["effects"][-1]["best_intent_sequence"] is None
    assert after_worse["o5"]["best_d_seg"] == 0.01


def test_live_transaction_refuses_duplicate_controller_history() -> None:
    with pytest.raises(
        LiveVerdictStateError,
        match="controller-owned histories",
    ):
        new_reducer_state(history=[{"epoch": 1}])

    state = new_reducer_state()
    state["history"].append({"epoch": 1})
    with pytest.raises(
        LiveVerdictStateError,
        match="duplicates controller-owned history",
    ):
        validate_reducer_state(state)


def test_fixed_row_capacities_fail_closed() -> None:
    state = new_reducer_state()
    for sequence in range(SERIALIZED_MAX_EFFECT_ROWS + 1):
        state = reduce_result(
            state,
            run_worker(
                sequence,
                _snapshot(epoch=sequence + 1, d_seg=0.02 + sequence * 0.001),
                score_snapshot=_score,
            ),
        )
    with pytest.raises(
        LiveVerdictStateError,
        match="O4 effect count exceeds fixed capacity",
    ):
        state_arrays(state, PublisherCursor(), prefix="effect_overflow__")

    state = new_reducer_state()
    for sequence in range(SERIALIZED_MAX_BEST_INTENT_ROWS + 1):
        state = reduce_result(
            state,
            run_worker(
                sequence,
                _snapshot(epoch=sequence + 1, d_seg=1.0 / (sequence + 1)),
                score_snapshot=_score,
            ),
        )
    # Model the independently valid coordinate where effects have been
    # acknowledged but their O5 publications have not.  The fixed codec must
    # still refuse rather than truncate the retained physical artifacts.
    state["effects"] = []
    state["effect_base_sequence"] = state["next_effect_sequence"]
    validate_reducer_state(state)
    with pytest.raises(
        LiveVerdictStateError,
        match="O5 BEST-intent count exceeds fixed capacity",
    ):
        state_arrays(
            state,
            PublisherCursor(
                next_effect_sequence=state["next_effect_sequence"],
                next_best_intent_sequence=0,
            ),
            prefix="best_overflow__",
        )


def test_fixed_arrays_refuse_payload_identity_padding_and_census_tamper() -> None:
    prefix = "tamper__"
    state = reduce_result(
        new_reducer_state(),
        run_worker(0, _snapshot(epoch=1, d_seg=0.02), score_snapshot=_score),
    )

    unknown = dict(state_arrays(state, PublisherCursor(), prefix=prefix))
    unknown[f"{prefix}mixed_owner_payload"] = np.zeros(1, dtype=np.uint8)
    with pytest.raises(LiveVerdictStateError, match="array census differs"):
        state_from_arrays(unknown, prefix=prefix)

    payload = dict(state_arrays(state, PublisherCursor(), prefix=prefix))
    payload[f"{prefix}o4_effect_payload_data"] = payload[
        f"{prefix}o4_effect_payload_data"
    ].copy()
    payload[f"{prefix}o4_effect_payload_data"][10] ^= np.uint8(1)
    with pytest.raises(
        LiveVerdictStateError,
        match="O4 effect payload or identity is invalid",
    ):
        state_from_arrays(payload, prefix=prefix)

    padding = dict(state_arrays(state, PublisherCursor(), prefix=prefix))
    used = int(padding[f"{prefix}o5_artifact_payload_offsets"][1])
    padding[f"{prefix}o5_artifact_payload_data"] = padding[
        f"{prefix}o5_artifact_payload_data"
    ].copy()
    padding[f"{prefix}o5_artifact_payload_data"][used] = np.uint8(1)
    with pytest.raises(LiveVerdictStateError, match="unused data"):
        state_from_arrays(padding, prefix=prefix)

    identity = dict(state_arrays(state, PublisherCursor(), prefix=prefix))
    identity[f"{prefix}o4_effect_result_id_data"] = identity[
        f"{prefix}o4_effect_result_id_data"
    ].copy()
    result_id_length = int(identity[f"{prefix}o4_effect_result_id_lengths"][0])
    identity[f"{prefix}o4_effect_result_id_data"][0, result_id_length] = np.uint8(1)
    with pytest.raises(LiveVerdictStateError, match="padding"):
        state_from_arrays(identity, prefix=prefix)

    digest = dict(state_arrays(state, PublisherCursor(), prefix=prefix))
    digest[f"{prefix}o5_artifact_sha256"] = digest[
        f"{prefix}o5_artifact_sha256"
    ].copy()
    digest[f"{prefix}o5_artifact_sha256"][0, 0] ^= np.uint8(1)
    with pytest.raises(
        LiveVerdictStateError,
        match="O5 artifact payload or identity is invalid",
    ):
        state_from_arrays(digest, prefix=prefix)


def test_worker_failure_poison_refuses_later_submit_and_checkpoint() -> None:
    transaction = _new_transaction()

    def bad_worker(sequence: int, snapshot):
        raise RuntimeError("real scorer failed")

    with ThreadPoolExecutor(max_workers=1) as executor:
        transaction.submit(executor, bad_worker, _snapshot(epoch=1, d_seg=0.02))
        with pytest.raises(WorkerExecutionError), transaction.checkpoint():
            raise AssertionError("unreachable")
        with pytest.raises(TransactionPoisonedError):
            transaction.submit(executor, _worker, _snapshot(epoch=2, d_seg=0.01))
        with pytest.raises(TransactionPoisonedError), transaction.checkpoint():
            raise AssertionError("unreachable")


def test_publication_failure_poisons_same_transaction_and_future_submit() -> None:
    transaction = _new_transaction()
    publisher = MainThreadVerdictEffectPublisher()

    def transition(state):
        publisher.publish_pending(
            state,
            publish_effect=lambda effect: None,
            publish_best=lambda effect, intent: (_ for _ in ()).throw(
                OSError("BEST replace failed")
            ),
        )
        return compact_acknowledged_state(state, publisher.cursor)

    with ThreadPoolExecutor(max_workers=1) as executor:
        transaction.submit(executor, _worker, _snapshot(epoch=1, d_seg=0.02))
        with pytest.raises(ExternalEffectTransitionError):
            transaction.apply_external_effect_transition(
                transition=transition,
                validate_replacement=validate_reducer_state,
            )
        assert transaction.poisoned
        with pytest.raises(TransactionPoisonedError):
            transaction.submit(executor, _worker, _snapshot(epoch=2, d_seg=0.01))


def test_validation_binds_best_summary_and_effect_payload_hash() -> None:
    state = reduce_result(
        new_reducer_state(),
        run_worker(0, _snapshot(epoch=1, d_seg=0.02), score_snapshot=_score),
    )
    bad_best = compact_acknowledged_state(state, PublisherCursor())
    bad_best["o5"]["best_d_seg"] = 0.001
    with pytest.raises(LiveVerdictStateError, match="summary differs"):
        validate_reducer_state(bad_best)

    bad_effect = compact_acknowledged_state(state, PublisherCursor())
    bad_effect["effects"][0]["payload"]["verdict"]["d_seg"] = 0.03
    with pytest.raises(LiveVerdictStateError, match="canonical payload"):
        validate_reducer_state(bad_effect)


def test_real_trainer_source_wires_native_order_and_keeps_admission_blocked() -> None:
    from experiments import train_levelset_witness_realized_through_R_mlx as trainer

    source = inspect.getsource(trainer.run_train)
    assert "_reserve_pose_verdict_index_main()" in source
    assert "run_g111_live_verdict_worker" in source
    assert "reduce_g111_live_verdict_result" in source
    assert "transaction.apply_external_effect_transition(" in source
    assert "transaction.prepared_checkpoint(" in source
    assert "transaction.poison_checkpoint_publication(" in source
    assert "_g111_live_transaction.assert_healthy()" in source
    assert "_publish_g111_live_completed(wait=True)" in source
    assert "tuple(dir_feats_per_pair[pi].copy() for pi in vpairs)" in source
    assert "_require_g111_native_v3_launch_gate(None)" in source
    assert "checkpoint publication remains blocked" in source
    pure_scorer = source[
        source.index("def _g111_pure_verdict_from_snapshot") :
        source.index("history: list[dict[str, Any]]")
    ]
    assert "_verdict_v(" not in pure_scorer
    assert "_pose_verdict_count" not in pure_scorer
    assert "print(" not in pure_scorer
    worker_scorer = source[
        source.index("def _score_g111_live_snapshot") :
        source.index("def _run_g111_live_worker")
    ]
    assert "_g111_pure_verdict_from_snapshot" in worker_scorer
    assert "verdict = _verdict_from_snapshot(" not in worker_scorer
    native_effect = source[
        source.index("def _publish_g111_live_effect") :
        source.index("def _publish_g111_live_best")
    ]
    assert "reduce_g111_controller_state" in native_effect
    assert "intent['idempotency_key']" in native_effect
    assert "_g111_controller_state[\"value\"] = candidate" in native_effect
    assert "_emit_verdict_row(" not in native_effect
    native_best = source[
        source.index("def _publish_g111_live_best") :
        source.index("def _apply_g111_live_effect_transition")
    ]
    assert '"levelset_witness_ema_BEST_"' in native_best
    assert "f\"{str(intent['result_sha256'])}.npz\"" in native_best
    assert 'out_dir / "levelset_witness_ema_BEST.npz"' not in native_best

    decision_window = source[source.index("if args.async_verdict:") :]
    assert decision_window.index("_join_async_verdict()") < decision_window.index(
        "_cl_decide(ep)"
    )
    assert decision_window.index("_cl_decide(ep)") < decision_window.index(
        "_schedule_async_verdict(ep, seg_form, ep_loss)"
    )
