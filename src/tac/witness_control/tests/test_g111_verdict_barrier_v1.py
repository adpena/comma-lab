from __future__ import annotations

import base64
import json
import threading
import time
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pytest

from tac.witness_control.g111_verdict_barrier_v1 import (
    RESULT_ID_MAX_UTF8_BYTES,
    AppliedVerdictRow,
    CanonicalStateError,
    CheckpointPublicationError,
    DeterministicVerdictReducer,
    DuplicateResultError,
    DuplicateSequenceError,
    ExternalEffectTransitionError,
    ImmutableVerdictResult,
    NativeV3PendingPayloadError,
    NonQuiescentCheckpointError,
    QuiescentVerdictTransaction,
    ReducerApplicationError,
    ResultIntegrityError,
    SequenceGapError,
    SubmissionBlockedError,
    TransactionPoisonedError,
    WorkerCancelledError,
    WorkerExecutionError,
    WorkerResultTypeError,
)
from tac.witness_control.trajectory_transaction_v2 import BarrierStateBinding


def _append_reducer(state: dict, result: ImmutableVerdictResult) -> dict:
    state["rows"].append(
        {
            "seq": result.submission_seq,
            "result_id": result.result_id,
            "value": int(result.payload["value"]),
        }
    )
    return state


def _result(sequence: int, value: int, *, result_id: str | None = None) -> ImmutableVerdictResult:
    return ImmutableVerdictResult.capture(
        submission_seq=sequence,
        result_id=result_id or f"result-{sequence}",
        payload={"value": value},
    )


def _immediate_worker(
    sequence: int,
    snapshot,
) -> ImmutableVerdictResult:
    return _result(sequence, int(snapshot["value"]))


def _validate_ack_state(state: dict) -> None:
    if set(state) != {"rows", "acked"}:
        raise ValueError("ack state has unexpected fields")
    if not isinstance(state["rows"], list):
        raise TypeError("ack state rows must be a list")
    if (
        isinstance(state["acked"], bool)
        or not isinstance(state["acked"], int)
        or not 0 <= state["acked"] <= len(state["rows"])
    ):
        raise ValueError("ack cursor is outside the retained rows")


def _idempotent_effect_transition(
    published: list[tuple[str, int]],
):
    def transition(state: dict) -> dict:
        for row in state["rows"][state["acked"] :]:
            identity = (str(row["result_id"]), int(row["value"]))
            if identity not in published:
                published.append(identity)
        state["acked"] = len(state["rows"])
        return state

    return transition


def _assert_all_later_actions_refuse(
    transaction: QuiescentVerdictTransaction,
) -> None:
    fatal_error = transaction.fatal_error
    assert fatal_error is not None
    with (
        ThreadPoolExecutor(max_workers=1) as executor,
        pytest.raises(TransactionPoisonedError) as submit_error,
    ):
        transaction.submit(executor, _immediate_worker, {"value": 99})
    assert submit_error.value.__cause__ is fatal_error

    with pytest.raises(TransactionPoisonedError) as apply_error:
        transaction.apply_completed()
    assert apply_error.value.__cause__ is fatal_error

    with pytest.raises(TransactionPoisonedError) as transition_error:
        transaction.apply_external_effect_transition(
            transition=lambda state: state,
            validate_replacement=lambda state: None,
        )
    assert transition_error.value.__cause__ is fatal_error

    with (
        pytest.raises(TransactionPoisonedError) as checkpoint_error,
        transaction.checkpoint(),
    ):
        raise AssertionError("unreachable")
    assert checkpoint_error.value.__cause__ is fatal_error

    with (
        pytest.raises(TransactionPoisonedError) as prepared_error,
        transaction.prepared_checkpoint(
            transition=lambda state: state,
            validate_replacement=lambda state: None,
        ),
    ):
        raise AssertionError("unreachable")
    assert prepared_error.value.__cause__ is fatal_error


def test_active_worker_checkpoint_blocks_submission_joins_and_applies_once() -> None:
    transaction = QuiescentVerdictTransaction(
        reducer=_append_reducer,
        initial_state={"rows": []},
        max_journal_rows=4,
    )
    worker_started = threading.Event()
    worker_release = threading.Event()
    blocked_submission = threading.Event()

    with ThreadPoolExecutor(max_workers=2) as executor:

        def worker(sequence: int, snapshot) -> ImmutableVerdictResult:
            worker_started.set()
            assert worker_release.wait(timeout=5.0)
            try:
                transaction.submit(executor, _immediate_worker, {"value": 99})
            except SubmissionBlockedError:
                blocked_submission.set()
            return _result(sequence, int(snapshot["value"]))

        assert transaction.submit(executor, worker, {"value": 7}) == 0
        assert worker_started.wait(timeout=5.0)

        def release_after_barrier() -> None:
            deadline = time.monotonic() + 5.0
            while not transaction.checkpoint_active:
                if time.monotonic() >= deadline:
                    raise AssertionError("checkpoint barrier did not activate")
                time.sleep(0.001)
            worker_release.set()

        releaser = threading.Thread(target=release_after_barrier)
        releaser.start()
        with transaction.checkpoint() as capture:
            assert transaction.checkpoint_active is True
            assert blocked_submission.wait(timeout=5.0)
            assert capture.pending_count == 0
            assert capture.next_submit_seq == capture.next_apply_seq == 1
            assert capture.reducer_state == {"rows": [{"seq": 0, "result_id": "result-0", "value": 7}]}
        releaser.join(timeout=5.0)
        assert not releaser.is_alive()

    assert transaction.checkpoint_active is False
    assert transaction.pending_count == 0
    assert [row.submission_seq for row in transaction.journal] == [0]


def test_apply_completed_then_checkpoint_does_not_apply_result_twice() -> None:
    applied: list[int] = []

    def reducer(state: dict, result: ImmutableVerdictResult) -> dict:
        applied.append(result.submission_seq)
        return _append_reducer(state, result)

    transaction = QuiescentVerdictTransaction(
        reducer=reducer,
        initial_state={"rows": []},
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        transaction.submit(executor, _immediate_worker, {"value": 4})
        executor.shutdown(wait=True)
        assert transaction.apply_completed() == 1
        assert transaction.apply_completed() == 0
        with transaction.checkpoint() as capture:
            assert capture.next_apply_seq == 1

    assert applied == [0]
    assert transaction.reducer_state["rows"] == [{"seq": 0, "result_id": "result-0", "value": 4}]


def test_reducer_rejects_gap_duplicate_sequence_and_duplicate_result_id() -> None:
    gap_reducer = DeterministicVerdictReducer(
        reducer=_append_reducer,
        initial_state={"rows": []},
        max_journal_rows=4,
    )
    with pytest.raises(SequenceGapError, match="leaves a gap"):
        gap_reducer.apply(_result(1, 1))
    with pytest.raises(TransactionPoisonedError):
        gap_reducer.apply(_result(0, 10))

    sequence_reducer = DeterministicVerdictReducer(
        reducer=_append_reducer,
        initial_state={"rows": []},
        max_journal_rows=4,
    )
    sequence_reducer.apply(_result(0, 10))
    with pytest.raises(DuplicateSequenceError, match="before next_apply_seq"):
        sequence_reducer.apply(_result(0, 10))
    with pytest.raises(TransactionPoisonedError):
        sequence_reducer.apply(_result(1, 11))

    identity_reducer = DeterministicVerdictReducer(
        reducer=_append_reducer,
        initial_state={"rows": []},
        max_journal_rows=4,
    )
    identity_reducer.apply(_result(0, 10, result_id="same-id"))
    with pytest.raises(DuplicateResultError, match="already applied"):
        identity_reducer.apply(_result(1, 11, result_id="same-id"))
    with pytest.raises(TransactionPoisonedError):
        identity_reducer.apply(_result(1, 11))

    assert gap_reducer.next_apply_seq == 0
    assert sequence_reducer.next_apply_seq == 1
    assert identity_reducer.next_apply_seq == 1
    assert identity_reducer.state["rows"] == [{"seq": 0, "result_id": "same-id", "value": 10}]


def test_completed_workers_are_reduced_in_submission_order() -> None:
    order: list[int] = []
    release_zero = threading.Event()
    one_done = threading.Event()

    def reducer(state: dict, result: ImmutableVerdictResult) -> dict:
        order.append(result.submission_seq)
        return _append_reducer(state, result)

    transaction = QuiescentVerdictTransaction(
        reducer=reducer,
        initial_state={"rows": []},
    )
    with ThreadPoolExecutor(max_workers=2) as executor:

        def worker(sequence: int, snapshot) -> ImmutableVerdictResult:
            if sequence == 0:
                assert release_zero.wait(timeout=5.0)
            else:
                one_done.set()
            return _result(sequence, int(snapshot["value"]))

        transaction.submit(executor, worker, {"value": 0})
        transaction.submit(executor, worker, {"value": 1})
        assert one_done.wait(timeout=5.0)
        assert transaction.apply_completed() == 0
        release_zero.set()
        with transaction.checkpoint() as capture:
            assert capture.next_apply_seq == 2

    assert order == [0, 1]


def test_result_integrity_hash_is_checked() -> None:
    reducer = DeterministicVerdictReducer(
        reducer=_append_reducer,
        initial_state={"rows": []},
        max_journal_rows=4,
    )
    corrupted = replace(_result(0, 3), result_sha256="0" * 64)
    with pytest.raises(ResultIntegrityError, match="SHA mismatch"):
        reducer.apply(corrupted)
    assert reducer.next_apply_seq == 0
    assert reducer.poisoned is True
    with pytest.raises(TransactionPoisonedError):
        reducer.apply(_result(0, 3))


def test_nan_ndarray_payload_is_typed_integrity_failure_and_permanent_poison() -> None:
    nan_data = base64.b64encode(np.asarray([np.nan], dtype=np.float64).tobytes(order="C")).decode("ascii")
    payload_bytes = json.dumps(
        {
            "items": [
                [
                    "value",
                    {
                        "data": nan_data,
                        "dtype": np.dtype(np.float64).str,
                        "shape": [1],
                        "type": "ndarray",
                    },
                ]
            ],
            "type": "mapping",
        },
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    malformed = replace(_result(0, 3), payload_bytes=payload_bytes)
    reducer = DeterministicVerdictReducer(
        reducer=_append_reducer,
        initial_state={"rows": []},
        max_journal_rows=4,
    )

    with pytest.raises(ResultIntegrityError) as caught:
        reducer.apply(malformed)
    assert isinstance(caught.value.__cause__, ValueError)
    assert reducer.poisoned is True
    assert reducer.next_apply_seq == 0

    with pytest.raises(TransactionPoisonedError) as later:
        reducer.apply(_result(0, 3))
    assert later.value.__cause__ is caught.value


def test_reducer_exception_poisons_without_publishing_or_advancing() -> None:
    original_state = {"rows": [], "array": np.asarray([1, 2], dtype=np.int64)}

    class ReducerSentinelError(BaseException):
        pass

    def broken_reducer(state: dict, result: ImmutableVerdictResult) -> dict:
        state["rows"].append(result.result_id)
        state["array"][0] = 99
        raise ReducerSentinelError("injected reducer failure")

    transaction = QuiescentVerdictTransaction(
        reducer=broken_reducer,
        initial_state=original_state,
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        transaction.submit(executor, _immediate_worker, {"value": 1})
        with (
            pytest.raises(ReducerApplicationError) as caught,
            transaction.checkpoint(),
        ):
            raise AssertionError("unreachable")

    assert isinstance(caught.value.__cause__, ReducerSentinelError)
    assert transaction.poisoned is True
    assert transaction.fatal_error is caught.value
    assert transaction.next_apply_seq == 0
    assert transaction.pending_count == 0
    state = transaction.reducer_state
    assert state["rows"] == []
    np.testing.assert_array_equal(state["array"], np.asarray([1, 2]))
    assert transaction.checkpoint_active is False
    _assert_all_later_actions_refuse(transaction)


def test_worker_exception_is_typed_fatal_and_preserves_original_cause() -> None:
    class WorkerSentinelError(Exception):
        pass

    def broken_worker(sequence: int, snapshot) -> ImmutableVerdictResult:
        del sequence, snapshot
        raise WorkerSentinelError("injected worker failure")

    transaction = QuiescentVerdictTransaction(
        reducer=_append_reducer,
        initial_state={"rows": []},
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        transaction.submit(executor, broken_worker, {"value": 1})
        with (
            pytest.raises(WorkerExecutionError) as caught,
            transaction.checkpoint(),
        ):
            raise AssertionError("unreachable")

    assert isinstance(caught.value.__cause__, WorkerSentinelError)
    assert transaction.fatal_error is caught.value
    assert transaction.pending_count == 0
    assert transaction.checkpoint_active is False
    _assert_all_later_actions_refuse(transaction)


def test_worker_done_callback_poison_is_visible_before_main_thread_apply() -> None:
    class WorkerSentinelError(Exception):
        pass

    worker_release = threading.Event()

    def broken_worker(sequence: int, snapshot) -> ImmutableVerdictResult:
        del sequence, snapshot
        assert worker_release.wait(timeout=5.0)
        raise WorkerSentinelError("injected asynchronous worker failure")

    transaction = QuiescentVerdictTransaction(
        reducer=_append_reducer,
        initial_state={"rows": []},
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        transaction.submit(executor, broken_worker, {"value": 1})
        worker_release.set()
        deadline = time.monotonic() + 5.0
        while not transaction.poisoned:
            if time.monotonic() >= deadline:
                raise AssertionError("worker done callback did not poison transaction")
            time.sleep(0.001)

        with pytest.raises(WorkerExecutionError) as caught:
            transaction.assert_healthy()

    assert isinstance(caught.value.__cause__, WorkerSentinelError)
    assert transaction.fatal_error is caught.value
    assert transaction.next_apply_seq == 0
    assert transaction.pending_count == 0
    assert transaction.reducer_state == {"rows": []}
    _assert_all_later_actions_refuse(transaction)


def test_successful_worker_done_callback_does_not_poison_before_ordered_apply() -> None:
    transaction = QuiescentVerdictTransaction(
        reducer=_append_reducer,
        initial_state={"rows": []},
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        transaction.submit(executor, _immediate_worker, {"value": 41})
        executor.shutdown(wait=True)
        transaction.assert_healthy()
        assert transaction.poisoned is False
        assert transaction.apply_completed() == 1

    assert transaction.reducer_state["rows"][0]["value"] == 41


def test_successful_invalid_result_is_not_callback_poisoned_before_validation() -> None:
    def invalid_worker(sequence: int, snapshot) -> dict:
        return {"sequence": sequence, "value": snapshot["value"]}

    transaction = QuiescentVerdictTransaction(
        reducer=_append_reducer,
        initial_state={"rows": []},
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        transaction.submit(executor, invalid_worker, {"value": 43})
        executor.shutdown(wait=True)
        transaction.assert_healthy()
        with pytest.raises(WorkerResultTypeError) as caught, transaction.checkpoint():
            raise AssertionError("unreachable")

    assert transaction.fatal_error is caught.value
    assert transaction.next_apply_seq == 0
    _assert_all_later_actions_refuse(transaction)


def test_later_worker_failure_poison_preempts_blocked_earlier_result() -> None:
    class LaterWorkerError(Exception):
        pass

    first_started = threading.Event()
    first_release = threading.Event()

    def blocked_first(sequence: int, snapshot) -> ImmutableVerdictResult:
        first_started.set()
        assert first_release.wait(timeout=5.0)
        return _result(sequence, int(snapshot["value"]))

    def broken_later(sequence: int, snapshot) -> ImmutableVerdictResult:
        del sequence, snapshot
        raise LaterWorkerError("later scorer failed before ordered reduction")

    transaction = QuiescentVerdictTransaction(
        reducer=_append_reducer,
        initial_state={"rows": []},
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        transaction.submit(executor, blocked_first, {"value": 47})
        assert first_started.wait(timeout=5.0)
        transaction.submit(executor, broken_later, {"value": 49})
        deadline = time.monotonic() + 5.0
        while not transaction.poisoned:
            if time.monotonic() >= deadline:
                raise AssertionError("later worker failure was not eagerly latched")
            time.sleep(0.001)

        with pytest.raises(WorkerExecutionError) as caught:
            transaction.assert_healthy()
        first_release.set()

    assert isinstance(caught.value.__cause__, LaterWorkerError)
    assert transaction.fatal_error is caught.value
    assert transaction.next_apply_seq == 0
    assert transaction.pending_count == 0
    assert transaction.reducer_state == {"rows": []}
    _assert_all_later_actions_refuse(transaction)


def test_worker_failure_cancels_n600_queue_without_recursive_callbacks() -> None:
    class QueueHeadError(Exception):
        pass

    head_release = threading.Event()

    def queued_worker(sequence: int, snapshot) -> ImmutableVerdictResult:
        if sequence == 0:
            assert head_release.wait(timeout=5.0)
            raise QueueHeadError("queue head failed")
        return _result(sequence, int(snapshot["value"]))

    transaction = QuiescentVerdictTransaction(
        reducer=_append_reducer,
        initial_state={"rows": []},
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        for value in range(600):
            transaction.submit(executor, queued_worker, {"value": value})
        assert transaction.pending_count == 600
        head_release.set()
        deadline = time.monotonic() + 5.0
        while not transaction.poisoned:
            if time.monotonic() >= deadline:
                raise AssertionError("queue-head failure was not eagerly latched")
            time.sleep(0.001)
        with pytest.raises(WorkerExecutionError) as caught:
            transaction.assert_healthy()

    assert isinstance(caught.value.__cause__, QueueHeadError)
    assert transaction.pending_count == 0
    assert transaction.next_apply_seq == 0
    _assert_all_later_actions_refuse(transaction)


def test_cancelled_worker_is_typed_fatal() -> None:
    class CancelledExecutor:
        def submit(self, worker, *args):
            del worker, args
            future: Future[ImmutableVerdictResult] = Future()
            assert future.cancel()
            return future

    transaction = QuiescentVerdictTransaction(
        reducer=_append_reducer,
        initial_state={"rows": []},
    )
    transaction.submit(CancelledExecutor(), _immediate_worker, {"value": 1})
    with (
        pytest.raises(WorkerCancelledError) as caught,
        transaction.checkpoint(),
    ):
        raise AssertionError("unreachable")

    assert isinstance(caught.value.__cause__, BaseException)
    assert transaction.fatal_error is caught.value
    assert transaction.pending_count == 0
    assert transaction.checkpoint_active is False
    _assert_all_later_actions_refuse(transaction)


@pytest.mark.parametrize(
    ("worker", "expected_error"),
    [
        (
            lambda sequence, snapshot: {
                "sequence": sequence,
                "value": snapshot["value"],
            },
            WorkerResultTypeError,
        ),
        (
            lambda sequence, snapshot: replace(
                _result(sequence, int(snapshot["value"])),
                result_sha256="0" * 64,
            ),
            ResultIntegrityError,
        ),
        (
            lambda sequence, snapshot: _result(
                sequence + 1,
                int(snapshot["value"]),
            ),
            SequenceGapError,
        ),
    ],
    ids=["wrong-result-type", "result-integrity", "sequence-mismatch"],
)
def test_invalid_worker_result_fatally_poisons_transaction(
    worker,
    expected_error: type[Exception],
) -> None:
    transaction = QuiescentVerdictTransaction(
        reducer=_append_reducer,
        initial_state={"rows": []},
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        transaction.submit(executor, worker, {"value": 1})
        with pytest.raises(expected_error) as caught, transaction.checkpoint():
            raise AssertionError("unreachable")

    assert transaction.fatal_error is caught.value
    assert transaction.pending_count == 0
    assert transaction.next_apply_seq == 0
    assert transaction.checkpoint_active is False
    _assert_all_later_actions_refuse(transaction)


def test_duplicate_worker_result_identity_fatally_poisons_transaction() -> None:
    transaction = QuiescentVerdictTransaction(
        reducer=_append_reducer,
        initial_state={"rows": []},
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        transaction.submit(
            executor,
            lambda sequence, snapshot: _result(
                sequence,
                int(snapshot["value"]),
                result_id="duplicate",
            ),
            {"value": 1},
        )
        with transaction.checkpoint():
            pass
        transaction.submit(
            executor,
            lambda sequence, snapshot: _result(
                sequence,
                int(snapshot["value"]),
                result_id="duplicate",
            ),
            {"value": 2},
        )
        with (
            pytest.raises(DuplicateResultError) as caught,
            transaction.checkpoint(),
        ):
            raise AssertionError("unreachable")

    assert transaction.fatal_error is caught.value
    assert transaction.pending_count == 0
    assert transaction.next_apply_seq == 1
    assert transaction.checkpoint_active is False
    _assert_all_later_actions_refuse(transaction)


def test_worker_snapshot_and_result_capture_have_no_mutable_alias() -> None:
    transaction = QuiescentVerdictTransaction(
        reducer=_append_reducer,
        initial_state={"rows": []},
    )
    source = {
        "value": np.asarray([5], dtype=np.int64),
        "nested": {"tag": "before"},
    }
    worker_started = threading.Event()
    worker_release = threading.Event()

    with ThreadPoolExecutor(max_workers=1) as executor:

        def worker(sequence: int, snapshot) -> ImmutableVerdictResult:
            worker_started.set()
            assert worker_release.wait(timeout=5.0)
            assert snapshot["value"].flags.writeable is False
            with pytest.raises(ValueError):
                snapshot["value"].setflags(write=True)
            payload = {
                "value": int(snapshot["value"][0]),
                "tag": snapshot["nested"]["tag"],
            }
            result = ImmutableVerdictResult.capture(
                submission_seq=sequence,
                result_id=f"result-{sequence}",
                payload=payload,
            )
            payload["value"] = 999
            return result

        transaction.submit(executor, worker, source)
        assert worker_started.wait(timeout=5.0)
        source["value"][0] = 44
        source["nested"]["tag"] = "after"
        worker_release.set()
        with transaction.checkpoint() as capture:
            row = capture.reducer_state["rows"][0]
            assert row["value"] == 5
            arrays = capture.numpy_state()
            with pytest.raises(ValueError, match="read-only"):
                arrays["next_submit_seq"][0] = 10
            for array in arrays.values():
                assert array.flags.writeable is False
                with pytest.raises(ValueError):
                    array.setflags(write=True)


def test_result_capture_and_decode_reject_non_mapping_payloads_with_typed_errors() -> None:
    with pytest.raises(TypeError, match="payload must be a mapping"):
        ImmutableVerdictResult.capture(
            submission_seq=0,
            result_id="bad-payload",
            payload=[],  # type: ignore[arg-type]
        )

    valid = _result(0, 1)
    malformed_payloads = [
        b"[]",
        json.dumps(
            {"type": "mapping", "items": [["nested", []]]},
            separators=(",", ":"),
        ).encode(),
        json.dumps(
            {
                "type": "mapping",
                "items": [
                    [
                        "nested",
                        {"type": "list", "items": [1]},
                    ]
                ],
            },
            separators=(",", ":"),
        ).encode(),
        json.dumps(
            {"type": "mapping", "items": {"not": "a-list"}},
            separators=(",", ":"),
        ).encode(),
    ]
    for payload_bytes in malformed_payloads:
        malformed = replace(valid, payload_bytes=payload_bytes)
        with pytest.raises(ResultIntegrityError):
            malformed.validate()


@pytest.mark.parametrize(
    "result",
    [
        replace(_result(0, 1), result_id=7),
        replace(_result(0, 1), result_id="not canonical"),
        replace(_result(0, 1), result_sha256=b"not-a-string"),
        replace(_result(0, 1), submission_seq="0"),
    ],
)
def test_immutable_result_identity_types_fail_with_result_integrity_error(
    result: ImmutableVerdictResult,
) -> None:
    with pytest.raises(ResultIntegrityError):
        result.validate()


@pytest.mark.parametrize(
    "row",
    [
        AppliedVerdictRow(
            submission_seq=0,
            result_id=7,
            result_sha256="0" * 64,
        ),
        AppliedVerdictRow(
            submission_seq=0,
            result_id="result",
            result_sha256=b"not-a-string",
        ),
        AppliedVerdictRow(
            submission_seq=0,
            result_id="not canonical",
            result_sha256="0" * 64,
        ),
        AppliedVerdictRow(
            submission_seq=0,
            result_id="result",
            result_sha256="A" * 64,
        ),
        AppliedVerdictRow(
            submission_seq=0,
            result_id="result",
            result_sha256=("0" * 62) + "  ",
        ),
        AppliedVerdictRow(
            submission_seq="0",
            result_id="result",
            result_sha256="0" * 64,
        ),
    ],
)
def test_applied_result_identity_types_fail_with_canonical_state_error(
    row: AppliedVerdictRow,
) -> None:
    with pytest.raises(CanonicalStateError):
        row.validate()


def test_checkpoint_barrier_releases_when_publication_body_fails() -> None:
    transaction = QuiescentVerdictTransaction(
        reducer=_append_reducer,
        initial_state={"rows": []},
    )

    with (
        pytest.raises(OSError, match="injected publish failure"),
        transaction.checkpoint(),
    ):
        assert transaction.checkpoint_active is True
        raise OSError("injected publish failure")

    assert transaction.checkpoint_active is False
    with ThreadPoolExecutor(max_workers=1) as executor:
        assert transaction.submit(executor, _immediate_worker, {"value": 8}) == 0
        with transaction.checkpoint():
            pass
    assert transaction.next_apply_seq == 1


def test_external_effect_transition_acknowledges_once_without_reapplying_reducer() -> None:
    applied: list[int] = []
    published: list[tuple[str, int]] = []

    def reducer(state: dict, result: ImmutableVerdictResult) -> dict:
        applied.append(result.submission_seq)
        return _append_reducer(state, result)

    transaction = QuiescentVerdictTransaction(
        reducer=reducer,
        initial_state={"rows": [], "acked": 0},
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        transaction.submit(executor, _immediate_worker, {"value": 10})
        transaction.submit(executor, _immediate_worker, {"value": 11})
        executor.shutdown(wait=True)
        replacement = transaction.apply_external_effect_transition(
            transition=_idempotent_effect_transition(published),
            validate_replacement=_validate_ack_state,
        )

    assert applied == [0, 1]
    assert published == [("result-0", 10), ("result-1", 11)]
    assert replacement["acked"] == 2
    replacement["acked"] = 0
    assert transaction.reducer_state["acked"] == 2

    transaction.apply_external_effect_transition(
        transition=_idempotent_effect_transition(published),
        validate_replacement=_validate_ack_state,
    )
    assert applied == [0, 1]
    assert published == [("result-0", 10), ("result-1", 11)]
    assert transaction.next_apply_seq == transaction.next_submit_seq == 2


def test_reducer_external_transition_reentrant_apply_fails_without_publication() -> None:
    reducer = DeterministicVerdictReducer(
        reducer=_append_reducer,
        initial_state={"rows": [{"result_id": "seed"}], "acked": 0},
        max_journal_rows=4,
    )

    def reentrant_transition(state: dict) -> dict:
        state["acked"] = 1
        reducer.apply(_result(0, 5))
        return state

    with pytest.raises(ExternalEffectTransitionError) as caught:
        reducer.apply_external_effect_transition(
            transition=reentrant_transition,
            validate_replacement=_validate_ack_state,
        )

    assert isinstance(caught.value.__cause__, ExternalEffectTransitionError)
    assert reducer.poisoned is True
    assert reducer.next_apply_seq == 0
    assert reducer.journal == ()
    assert reducer.state == {"rows": [{"result_id": "seed"}], "acked": 0}


def test_reducer_external_transition_reentrant_transition_fails_without_publication() -> None:
    reducer = DeterministicVerdictReducer(
        reducer=_append_reducer,
        initial_state={"rows": [], "acked": 0},
        max_journal_rows=4,
    )

    def reentrant_transition(state: dict) -> dict:
        state["acked"] = 1
        reducer.apply_external_effect_transition(
            transition=lambda nested_state: nested_state,
            validate_replacement=_validate_ack_state,
        )
        return state

    with pytest.raises(ExternalEffectTransitionError) as caught:
        reducer.apply_external_effect_transition(
            transition=reentrant_transition,
            validate_replacement=_validate_ack_state,
        )

    assert isinstance(caught.value.__cause__, ExternalEffectTransitionError)
    assert reducer.poisoned is True
    assert reducer.next_apply_seq == 0
    assert reducer.journal == ()
    assert reducer.state == {"rows": [], "acked": 0}


def test_reducer_apply_reentrant_apply_fails_without_publication() -> None:
    reducer: DeterministicVerdictReducer

    def reentrant_reducer(state: dict, result: ImmutableVerdictResult) -> dict:
        state["rows"].append(result.result_id)
        reducer.apply(_result(result.submission_seq, 59, result_id="nested"))
        return state

    reducer = DeterministicVerdictReducer(
        reducer=reentrant_reducer,
        initial_state={"rows": [], "acked": 0},
        max_journal_rows=4,
    )
    with pytest.raises(ReducerApplicationError) as caught:
        reducer.apply(_result(0, 53))

    assert isinstance(caught.value.__cause__, ExternalEffectTransitionError)
    assert reducer.poisoned is True
    assert reducer.next_apply_seq == 0
    assert reducer.journal == ()
    assert reducer.state == {"rows": [], "acked": 0}


def test_reducer_apply_reentrant_external_transition_fails_without_publication() -> None:
    reducer: DeterministicVerdictReducer

    def reentrant_reducer(state: dict, result: ImmutableVerdictResult) -> dict:
        state["rows"].append(result.result_id)
        reducer.apply_external_effect_transition(
            transition=lambda nested_state: nested_state,
            validate_replacement=_validate_ack_state,
        )
        return state

    reducer = DeterministicVerdictReducer(
        reducer=reentrant_reducer,
        initial_state={"rows": [], "acked": 0},
        max_journal_rows=4,
    )
    with pytest.raises(ReducerApplicationError) as caught:
        reducer.apply(_result(0, 61))

    assert isinstance(caught.value.__cause__, ExternalEffectTransitionError)
    assert reducer.poisoned is True
    assert reducer.next_apply_seq == 0
    assert reducer.journal == ()
    assert reducer.state == {"rows": [], "acked": 0}


def test_caught_nested_reentry_still_prevents_outer_effect_commit() -> None:
    reducer = DeterministicVerdictReducer(
        reducer=_append_reducer,
        initial_state={"rows": [{"result_id": "seed"}], "acked": 0},
        max_journal_rows=4,
    )

    def catches_nested_failure(state: dict) -> dict:
        try:
            reducer.apply(_result(0, 67))
        except ExternalEffectTransitionError:
            pass
        state["acked"] = 1
        return state

    with pytest.raises(ExternalEffectTransitionError) as caught:
        reducer.apply_external_effect_transition(
            transition=catches_nested_failure,
            validate_replacement=_validate_ack_state,
        )

    assert isinstance(caught.value.__cause__, TransactionPoisonedError)
    assert reducer.poisoned is True
    assert reducer.next_apply_seq == 0
    assert reducer.journal == ()
    assert reducer.state == {"rows": [{"result_id": "seed"}], "acked": 0}


def test_external_effect_validator_receives_only_discarded_copies() -> None:
    transaction = QuiescentVerdictTransaction(
        reducer=_append_reducer,
        initial_state={"rows": [], "acked": 0},
    )

    def mutating_validator(state: dict) -> None:
        _validate_ack_state(state)
        state["validator_mutation"] = True

    def transition(state: dict) -> dict:
        assert "validator_mutation" not in state
        return state

    replacement = transaction.apply_external_effect_transition(
        transition=transition,
        validate_replacement=mutating_validator,
    )
    assert "validator_mutation" not in replacement
    assert "validator_mutation" not in transaction.reducer_state


def test_external_effect_failure_keeps_preack_state_and_permanently_poisons() -> None:
    physical_effects: list[str] = []
    transaction = QuiescentVerdictTransaction(
        reducer=_append_reducer,
        initial_state={"rows": [], "acked": 0},
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        transaction.submit(executor, _immediate_worker, {"value": 7})
        executor.shutdown(wait=True)

        def fail_after_physical_effect(state: dict) -> dict:
            physical_effects.append(state["rows"][0]["result_id"])
            state["acked"] = 1
            raise OSError("injected external-effect failure")

        with pytest.raises(ExternalEffectTransitionError) as caught:
            transaction.apply_external_effect_transition(
                transition=fail_after_physical_effect,
                validate_replacement=_validate_ack_state,
            )

    assert isinstance(caught.value.__cause__, OSError)
    assert transaction.fatal_error is caught.value
    assert transaction.reducer_state["acked"] == 0
    assert transaction.next_apply_seq == 1
    assert physical_effects == ["result-0"]
    _assert_all_later_actions_refuse(transaction)


def test_external_effect_invalid_replacement_keeps_preack_state_and_poisons() -> None:
    physical_effects: list[str] = []
    transaction = QuiescentVerdictTransaction(
        reducer=_append_reducer,
        initial_state={"rows": [], "acked": 0},
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        transaction.submit(executor, _immediate_worker, {"value": 17})
        executor.shutdown(wait=True)

        def invalid_ack_after_physical_effect(state: dict) -> dict:
            physical_effects.append(state["rows"][0]["result_id"])
            state["acked"] = len(state["rows"]) + 1
            return state

        with pytest.raises(ExternalEffectTransitionError) as caught:
            transaction.apply_external_effect_transition(
                transition=invalid_ack_after_physical_effect,
                validate_replacement=_validate_ack_state,
            )

    assert isinstance(caught.value.__cause__, ValueError)
    assert transaction.fatal_error is caught.value
    assert transaction.reducer_state["acked"] == 0
    assert physical_effects == ["result-0"]
    _assert_all_later_actions_refuse(transaction)


def test_prepared_checkpoint_drains_then_captures_post_ack_state_once() -> None:
    applied: list[int] = []
    published: list[tuple[str, int]] = []
    worker_started = threading.Event()
    worker_release = threading.Event()

    def reducer(state: dict, result: ImmutableVerdictResult) -> dict:
        applied.append(result.submission_seq)
        return _append_reducer(state, result)

    transaction = QuiescentVerdictTransaction(
        reducer=reducer,
        initial_state={"rows": [], "acked": 0},
    )
    with ThreadPoolExecutor(max_workers=1) as executor:

        def worker(sequence: int, snapshot) -> ImmutableVerdictResult:
            worker_started.set()
            assert worker_release.wait(timeout=5.0)
            return _result(sequence, int(snapshot["value"]))

        transaction.submit(executor, worker, {"value": 23})
        assert worker_started.wait(timeout=5.0)

        def release_after_barrier() -> None:
            deadline = time.monotonic() + 5.0
            while not transaction.checkpoint_active:
                if time.monotonic() >= deadline:
                    raise AssertionError("prepared checkpoint barrier did not activate")
                time.sleep(0.001)
            worker_release.set()

        releaser = threading.Thread(target=release_after_barrier)
        releaser.start()
        with transaction.prepared_checkpoint(
            transition=_idempotent_effect_transition(published),
            validate_replacement=_validate_ack_state,
        ) as capture:
            assert transaction.checkpoint_active is True
            assert capture.next_submit_seq == capture.next_apply_seq == 1
            assert capture.pending_count == 0
            assert capture.reducer_state["acked"] == 1
            assert published == [("result-0", 23)]
        releaser.join(timeout=5.0)
        assert not releaser.is_alive()

    assert applied == [0]
    assert transaction.reducer_state["acked"] == 1
    assert transaction.checkpoint_active is False
    with transaction.prepared_checkpoint(
        transition=_idempotent_effect_transition(published),
        validate_replacement=_validate_ack_state,
    ):
        pass
    assert applied == [0]
    assert published == [("result-0", 23)]


def test_prepared_checkpoint_restore_starts_from_durable_post_ack_state() -> None:
    published: list[tuple[str, int]] = []
    transaction = QuiescentVerdictTransaction(
        reducer=_append_reducer,
        initial_state={"rows": [], "acked": 0},
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        transaction.submit(executor, _immediate_worker, {"value": 31})
        with transaction.prepared_checkpoint(
            transition=_idempotent_effect_transition(published),
            validate_replacement=_validate_ack_state,
        ) as capture:
            arrays = capture.numpy_state(prefix="verdict.")
            durable_reducer_state = capture.reducer_state

    restored = QuiescentVerdictTransaction.from_numpy_state(
        arrays,
        reducer=_append_reducer,
        restored_reducer_state=durable_reducer_state,
        prefix="verdict.",
    )
    restored.apply_external_effect_transition(
        transition=_idempotent_effect_transition(published),
        validate_replacement=_validate_ack_state,
    )

    assert restored.reducer_state["acked"] == 1
    assert published == [("result-0", 31)]


def test_prepared_checkpoint_effect_failure_poisons_before_capture() -> None:
    transaction = QuiescentVerdictTransaction(
        reducer=_append_reducer,
        initial_state={"rows": [], "acked": 0},
    )
    yielded = False

    def broken_transition(state: dict) -> dict:
        del state
        raise RuntimeError("injected preparation failure")

    with (
        pytest.raises(ExternalEffectTransitionError) as caught,
        transaction.prepared_checkpoint(
            transition=broken_transition,
            validate_replacement=_validate_ack_state,
        ),
    ):
        yielded = True

    assert yielded is False
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert transaction.fatal_error is caught.value
    assert transaction.checkpoint_active is False
    _assert_all_later_actions_refuse(transaction)


def test_explicit_checkpoint_publication_poison_latches_same_transaction() -> None:
    transaction = QuiescentVerdictTransaction(
        reducer=_append_reducer,
        initial_state={"rows": [], "acked": 0},
    )
    publish_failure = OSError("injected checkpoint file publication failure")

    with (
        pytest.raises(CheckpointPublicationError) as caught,
        transaction.prepared_checkpoint(
            transition=lambda state: state,
            validate_replacement=_validate_ack_state,
        ) as capture,
    ):
        assert capture.reducer_state["acked"] == 0
        try:
            raise publish_failure
        except OSError as exc:
            transaction.poison_checkpoint_publication(exc)

    assert caught.value.__cause__ is publish_failure
    assert transaction.fatal_error is caught.value
    assert transaction.checkpoint_active is False
    _assert_all_later_actions_refuse(transaction)


def test_uncaught_prepared_checkpoint_body_failure_auto_poisons() -> None:
    transaction = QuiescentVerdictTransaction(
        reducer=_append_reducer,
        initial_state={"rows": [], "acked": 0},
    )
    publication_failure = OSError("injected uncaught checkpoint publication failure")

    with (
        pytest.raises(CheckpointPublicationError) as caught,
        transaction.prepared_checkpoint(
            transition=lambda state: state,
            validate_replacement=_validate_ack_state,
        ),
    ):
        raise publication_failure

    assert caught.value.__cause__ is publication_failure
    assert transaction.fatal_error is caught.value
    assert transaction.checkpoint_active is False
    _assert_all_later_actions_refuse(transaction)


def test_bounded_canonical_state_round_trip_and_next_sequence() -> None:
    transaction = QuiescentVerdictTransaction(
        reducer=_append_reducer,
        initial_state={"rows": []},
        max_journal_rows=2,
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        for value in range(3):
            transaction.submit(executor, _immediate_worker, {"value": value})
        with transaction.checkpoint() as capture:
            arrays = capture.numpy_state(prefix="vtx.")
            restored_state = capture.reducer_state

    bound_state = BarrierStateBinding.from_prefix("vtx.").parse(arrays)
    assert bound_state.next_submit_seq == bound_state.next_apply_seq == 3
    assert bound_state.pending_count == 0
    assert bound_state.last_applied_result_id == "result-2"

    restored = QuiescentVerdictTransaction.from_numpy_state(
        arrays,
        reducer=_append_reducer,
        restored_reducer_state=restored_state,
        prefix="vtx.",
    )
    assert restored.next_submit_seq == restored.next_apply_seq == 3
    assert [row.submission_seq for row in restored.journal] == [1, 2]
    assert [row["value"] for row in restored.reducer_state["rows"]] == [0, 1, 2]

    with ThreadPoolExecutor(max_workers=1) as executor:
        assert restored.submit(executor, _immediate_worker, {"value": 3}) == 3
        with restored.checkpoint() as capture:
            assert capture.next_apply_seq == 4
            assert [row.submission_seq for row in capture.journal] == [2, 3]


def test_cold_and_saturated_captures_have_identical_fixed_array_contract_and_round_trip() -> None:
    cold = QuiescentVerdictTransaction(
        reducer=_append_reducer,
        initial_state={"rows": []},
        max_journal_rows=2,
    )
    with cold.checkpoint() as capture:
        cold_arrays = capture.numpy_state(prefix="vtx.")
        cold_reducer_state = capture.reducer_state

    saturated = QuiescentVerdictTransaction(
        reducer=_append_reducer,
        initial_state={"rows": []},
        max_journal_rows=2,
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        saturated.submit(executor, _immediate_worker, {"value": 10})
        saturated.submit(executor, _immediate_worker, {"value": 11})
        with saturated.checkpoint() as capture:
            saturated_arrays = capture.numpy_state(prefix="vtx.")
            saturated_reducer_state = capture.reducer_state

    def contract(arrays: Mapping[str, np.ndarray]) -> dict[str, tuple[str, tuple[int, ...]]]:
        return {key: (value.dtype.str, value.shape) for key, value in arrays.items()}

    assert contract(cold_arrays) == contract(saturated_arrays)
    assert cold_arrays["vtx.journal_sequences"].shape == (2,)
    assert cold_arrays["vtx.journal_result_id_data"].shape == (2 * RESULT_ID_MAX_UTF8_BYTES,)
    assert cold_arrays["vtx.journal_result_id_offsets"].shape == (3,)
    assert cold_arrays["vtx.journal_result_sha256"].shape == (2, 32)
    assert cold_arrays["vtx.last_applied_result_id"].shape == (RESULT_ID_MAX_UTF8_BYTES,)
    assert cold_arrays["vtx.last_applied_result_sha256"].shape == (64,)

    cold_bound = BarrierStateBinding.from_prefix("vtx.").parse(cold_arrays)
    assert cold_bound.next_apply_seq == 0
    assert cold_bound.last_applied_result_id == ""
    assert cold_bound.last_applied_result_sha256 == ""

    for arrays, reducer_state, expected_sequences in (
        (cold_arrays, cold_reducer_state, []),
        (saturated_arrays, saturated_reducer_state, [0, 1]),
    ):
        restored = QuiescentVerdictTransaction.from_numpy_state(
            arrays,
            reducer=_append_reducer,
            restored_reducer_state=reducer_state,
            prefix="vtx.",
        )
        assert [row.submission_seq for row in restored.journal] == expected_sequences
        with restored.checkpoint() as recapture:
            roundtrip = recapture.numpy_state(prefix="vtx.")
        assert set(roundtrip) == set(arrays)
        for key in arrays:
            assert np.array_equal(roundtrip[key], arrays[key]), key


def test_result_id_utf8_width_is_code_bounded_before_journal_capture() -> None:
    exact_width = "x" * RESULT_ID_MAX_UTF8_BYTES
    assert _result(0, 1, result_id=exact_width).result_id == exact_width

    with pytest.raises(ValueError, match="UTF-8 encoding exceeds"):
        _result(0, 1, result_id="é" * (RESULT_ID_MAX_UTF8_BYTES // 2 + 1))

    with pytest.raises(ValueError, match="valid UTF-8"):
        _result(0, 1, result_id="\ud800")

    with pytest.raises(CanonicalStateError, match="UTF-8 encoding exceeds"):
        AppliedVerdictRow(
            submission_seq=0,
            result_id="x" * (RESULT_ID_MAX_UTF8_BYTES + 1),
            result_sha256="a" * 64,
        ).validate()


def test_fixed_capture_refuses_cursor_capacity_that_cannot_serialize_as_int64() -> None:
    transaction = QuiescentVerdictTransaction(
        reducer=_append_reducer,
        initial_state={"rows": []},
        max_journal_rows=int(np.iinfo(np.int64).max) + 1,
    )
    with (
        pytest.raises(CanonicalStateError, match="exceeds int64 capacity"),
        transaction.checkpoint() as capture,
    ):
        capture.numpy_state()


@pytest.mark.parametrize(
    ("key", "index", "match"),
    [
        ("last_applied_result_id", 7, "noncanonical nonzero padding"),
        ("last_applied_result_sha256", 7, "noncanonical nonzero padding"),
        ("journal_sequences", 1, "noncanonical nonzero padding"),
        ("journal_result_id_data", 7, "noncanonical nonzero padding"),
        ("journal_result_id_offsets", 1, "noncanonical padding"),
        ("journal_result_sha256", (1, 0), "noncanonical nonzero padding"),
    ],
)
def test_cold_fixed_capacity_restore_rejects_noncanonical_padding(
    key: str,
    index: int | tuple[int, int],
    match: str,
) -> None:
    transaction = QuiescentVerdictTransaction(
        reducer=_append_reducer,
        initial_state={"rows": []},
        max_journal_rows=2,
    )
    with transaction.checkpoint() as capture:
        arrays = dict(capture.numpy_state())
        reducer_state = capture.reducer_state

    tampered = np.array(arrays[key], copy=True)
    tampered[index] = 1
    arrays[key] = tampered
    with pytest.raises(CanonicalStateError, match=match):
        QuiescentVerdictTransaction.from_numpy_state(
            arrays,
            reducer=_append_reducer,
            restored_reducer_state=reducer_state,
        )


def test_fixed_capacity_restore_rejects_active_result_id_width_overflow() -> None:
    transaction = QuiescentVerdictTransaction(
        reducer=_append_reducer,
        initial_state={"rows": []},
        max_journal_rows=2,
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        transaction.submit(executor, _immediate_worker, {"value": 1})
        with transaction.checkpoint() as capture:
            arrays = dict(capture.numpy_state())
            reducer_state = capture.reducer_state

    offsets = np.array(arrays["journal_result_id_offsets"], copy=True)
    offsets[1:] = RESULT_ID_MAX_UTF8_BYTES + 1
    arrays["journal_result_id_offsets"] = offsets
    with pytest.raises(CanonicalStateError, match="exceeds its active fixed-capacity"):
        QuiescentVerdictTransaction.from_numpy_state(
            arrays,
            reducer=_append_reducer,
            restored_reducer_state=reducer_state,
        )


def test_native_v3_restore_rejects_legacy_pending_and_nonquiescent_state() -> None:
    transaction = QuiescentVerdictTransaction(
        reducer=_append_reducer,
        initial_state={"rows": []},
    )
    with transaction.checkpoint() as capture:
        arrays = dict(capture.numpy_state())
        restored_state = capture.reducer_state

    legacy = dict(arrays)
    legacy["__cl_pend_epoch"] = np.asarray([3], dtype=np.int64)
    with pytest.raises(NativeV3PendingPayloadError):
        QuiescentVerdictTransaction.from_numpy_state(
            legacy,
            reducer=_append_reducer,
            restored_reducer_state=restored_state,
        )

    with transaction.checkpoint() as prefixed_capture:
        prefixed = dict(prefixed_capture.numpy_state(prefix="vtx."))
        prefixed_state = prefixed_capture.reducer_state

    for key in (
        "__cl_pend_epoch",
        "vtx.__cl_pend_epoch",
        "ns/__cl_pend_shadow",
        "ns:__cl_pend_shadow",
        r"ns\__cl_pend_shadow",
        "prefix__cl_pend_shadow",
    ):
        legacy_prefixed = dict(prefixed)
        legacy_prefixed[key] = np.asarray([3], dtype=np.int64)
        with pytest.raises(NativeV3PendingPayloadError):
            QuiescentVerdictTransaction.from_numpy_state(
                legacy_prefixed,
                reducer=_append_reducer,
                restored_reducer_state=prefixed_state,
                prefix="vtx.",
            )

    safe_extra = dict(prefixed)
    safe_extra["ns/__cl_pending_shadow"] = np.asarray([3], dtype=np.int64)
    safe_restore = QuiescentVerdictTransaction.from_numpy_state(
        safe_extra,
        reducer=_append_reducer,
        restored_reducer_state=prefixed_state,
        prefix="vtx.",
    )
    assert safe_restore.next_submit_seq == safe_restore.next_apply_seq == 0

    nonquiescent = dict(arrays)
    nonquiescent["pending_count"] = np.asarray([1], dtype=np.int64)
    with pytest.raises(NonQuiescentCheckpointError):
        QuiescentVerdictTransaction.from_numpy_state(
            nonquiescent,
            reducer=_append_reducer,
            restored_reducer_state=restored_state,
        )


def test_canonical_restore_rejects_journal_tail_identity_drift() -> None:
    transaction = QuiescentVerdictTransaction(
        reducer=_append_reducer,
        initial_state={"rows": []},
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        transaction.submit(executor, _immediate_worker, {"value": 1})
        with transaction.checkpoint() as capture:
            arrays = dict(capture.numpy_state())
            restored_state = capture.reducer_state

    wrong_id = np.zeros(RESULT_ID_MAX_UTF8_BYTES, dtype=np.uint8)
    wrong_id[: len(b"wrong-id")] = np.frombuffer(b"wrong-id", dtype=np.uint8)
    arrays["last_applied_result_id"] = wrong_id
    with pytest.raises(CanonicalStateError, match="journal tail"):
        QuiescentVerdictTransaction.from_numpy_state(
            arrays,
            reducer=_append_reducer,
            restored_reducer_state=restored_state,
        )
