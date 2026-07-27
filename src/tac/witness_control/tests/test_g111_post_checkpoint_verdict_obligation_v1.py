from __future__ import annotations

import hashlib
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from tac.witness_control.g111_post_checkpoint_verdict_obligation_v1 import (
    DuplicateVerdictObligationError,
    FinalCheckpointNotReadyError,
    MalformedVerdictObligationError,
    PostCheckpointVerdictObligation,
    StaleVerdictObligationError,
    VerdictObligationDispatchError,
    VerdictObligationOwedError,
    VerdictObligationPoisonedError,
)
from tac.witness_control.g111_verdict_barrier_v1 import (
    ImmutableVerdictResult,
    QuiescentVerdictTransaction,
)

CONFIG_SHA = hashlib.sha256(b"g111-test-config").hexdigest()
PREFIX = "__g111_post_checkpoint_verdict__"


class _FakeTransaction:
    def __init__(self, sequence: int = 0) -> None:
        self.next_submit_seq = sequence
        self.next_apply_seq = sequence
        self.snapshots: list[Mapping[str, object]] = []

    def submit(self, snapshot: Mapping[str, object]) -> int:
        sequence = self.next_submit_seq
        self.snapshots.append(snapshot)
        self.next_submit_seq += 1
        return sequence

    def join_and_apply(self) -> None:
        self.next_apply_seq = self.next_submit_seq


def _restored_o1_o5(epoch: int) -> dict[str, object]:
    return {
        "epoch": epoch,
        "stage": "tau_softplus",
        "ep_loss": 0.125,
        "ema_code": np.arange(12, dtype=np.float32).reshape(3, 4),
        "liveness": {"ema_updates": 80, "accepted": 4},
    }


def _reconstructor(restored: Mapping[str, object]):
    def reconstruct(record):
        assert record.stage == restored["stage"]
        return {
            "schema": "tac.g111_live_verdict_snapshot.v1",
            "epoch": int(restored["epoch"]),
            "seg_form": str(restored["stage"]),
            "ep_loss": float(restored["ep_loss"]),
            "blob_bytes": 4096,
            "best_eligible": True,
            "closed_loop_enabled": True,
            "liveness": dict(restored["liveness"]),
            "scorer": {
                "ema_np": {"code": np.asarray(restored["ema_code"])},
                "pose_verdict_index": record.submission_seq,
            },
        }

    return reconstruct


def _arm(
    state: PostCheckpointVerdictObligation,
    transaction: _FakeTransaction,
    *,
    epoch: int = 25,
) -> None:
    state.arm(
        checkpoint_epoch=epoch,
        submission_seq=transaction.next_submit_seq,
        stage="tau_softplus",
        boundary_kind="intra_stage",
        config_sha256=CONFIG_SHA,
        next_submit_seq=transaction.next_submit_seq,
        next_apply_seq=transaction.next_apply_seq,
    )


def _discharge(
    state: PostCheckpointVerdictObligation,
    transaction: _FakeTransaction,
    restored: Mapping[str, object],
) -> int:
    return state.discharge(
        restored_checkpoint_epoch=int(restored["epoch"]),
        restored_stage=str(restored["stage"]),
        restored_boundary_kind="intra_stage",
        restored_config_sha256=CONFIG_SHA,
        next_submit_seq=lambda: transaction.next_submit_seq,
        next_apply_seq=lambda: transaction.next_apply_seq,
        reconstruct_snapshot=_reconstructor(restored),
        submit=transaction.submit,
    )


def _snapshot_hash(sequence: int, snapshot: Mapping[str, object]) -> str:
    return ImmutableVerdictResult.capture(
        submission_seq=sequence,
        result_id=f"snapshot-{sequence}",
        payload=snapshot,
    ).result_sha256


def test_fixed_typed_state_has_no_pending_payload_and_roundtrips_canonically() -> None:
    transaction = _FakeTransaction(sequence=7)
    state = PostCheckpointVerdictObligation()
    cold_arrays = state.numpy_state(prefix=PREFIX)
    _arm(state, transaction)

    arrays = state.numpy_state(prefix=PREFIX)
    assert set(cold_arrays) == set(arrays)
    assert all(
        cold_arrays[key].dtype == arrays[key].dtype
        and cold_arrays[key].shape == arrays[key].shape
        for key in arrays
    )
    assert set(arrays) == {
        f"{PREFIX}schema",
        f"{PREFIX}present",
        f"{PREFIX}checkpoint_epoch",
        f"{PREFIX}submission_seq",
        f"{PREFIX}stage",
        f"{PREFIX}boundary_kind",
        f"{PREFIX}config_sha256",
        f"{PREFIX}obligation_id",
    }
    assert not any("__cl_pend_" in key for key in arrays)
    assert not any(
        key.endswith(("snapshot", "scorer", "ema_np", "worker_payload"))
        for key in arrays
    )
    assert all(
        not value.dtype.hasobject and value.dtype.fields is None
        for value in arrays.values()
    )
    assert sum(value.nbytes for value in arrays.values()) < 1024

    restored = PostCheckpointVerdictObligation.from_numpy_state(
        arrays,
        prefix=PREFIX,
    )
    reserialized = restored.numpy_state(prefix=PREFIX)
    assert set(reserialized) == set(arrays)
    for key in arrays:
        assert arrays[key].dtype == reserialized[key].dtype
        assert arrays[key].shape == reserialized[key].shape
        assert np.array_equal(arrays[key], reserialized[key])

    tampered = dict(arrays)
    tampered[f"{PREFIX}stage"] = tampered[f"{PREFIX}stage"].copy()
    tampered[f"{PREFIX}stage"][-1] = np.uint8(1)
    with pytest.raises(MalformedVerdictObligationError, match="padding"):
        PostCheckpointVerdictObligation.from_numpy_state(
            tampered,
            prefix=PREFIX,
        )


def test_checkpoint_then_dispatch_blocks_optimizer_and_submits_exactly_once() -> None:
    transaction = _FakeTransaction()
    state = PostCheckpointVerdictObligation()
    _arm(state, transaction)
    restored = _restored_o1_o5(25)

    with pytest.raises(VerdictObligationOwedError):
        state.assert_optimizer_step_allowed()
    assert _discharge(state, transaction, restored) == 0
    state.assert_optimizer_step_allowed()
    assert transaction.next_submit_seq == 1
    assert transaction.next_apply_seq == 0
    assert len(transaction.snapshots) == 1
    assert transaction.snapshots[0]["epoch"] == 25

    restored["ema_code"][0, 0] = -999.0
    frozen_code = transaction.snapshots[0]["scorer"]["ema_np"]["code"]
    assert float(frozen_code[0, 0]) == 0.0
    assert frozen_code.flags.writeable is False

    with pytest.raises(DuplicateVerdictObligationError, match="already discharged"):
        _discharge(state, transaction, restored)
    assert len(transaction.snapshots) == 1


def test_real_transaction_decides_previous_then_checkpoints_then_submits() -> None:
    def worker(sequence: int, snapshot) -> ImmutableVerdictResult:
        return ImmutableVerdictResult.capture(
            submission_seq=sequence,
            result_id=f"verdict-{sequence}",
            payload={"epoch": int(snapshot["epoch"])},
        )

    def reducer(state, result: ImmutableVerdictResult):
        return {"epochs": [*state["epochs"], int(result.payload["epoch"])]}

    transaction = QuiescentVerdictTransaction(
        reducer=reducer,
        initial_state={"epochs": []},
        max_journal_rows=8,
    )
    obligation = PostCheckpointVerdictObligation()
    order: list[str] = []

    with ThreadPoolExecutor(max_workers=1) as executor:
        previous_snapshot = {
            "schema": "tac.g111_live_verdict_snapshot.v1",
            "epoch": 0,
        }
        transaction.submit(executor, worker, previous_snapshot)
        with transaction.checkpoint():
            pass
        order.append("join_previous")
        assert transaction.reducer_state["epochs"] == [0]

        order.append("decide_previous")
        decision_rows = tuple(transaction.reducer_state["epochs"])
        obligation.arm(
            checkpoint_epoch=25,
            submission_seq=transaction.next_submit_seq,
            stage="tau_softplus",
            boundary_kind="intra_stage",
            config_sha256=CONFIG_SHA,
            next_submit_seq=transaction.next_submit_seq,
            next_apply_seq=transaction.next_apply_seq,
        )
        with transaction.checkpoint():
            obligation.numpy_state(prefix=PREFIX)
            assert transaction.pending_count == 0
            order.append("checkpoint")

        restored = _restored_o1_o5(25)
        obligation.discharge(
            restored_checkpoint_epoch=25,
            restored_stage="tau_softplus",
            restored_boundary_kind="intra_stage",
            restored_config_sha256=CONFIG_SHA,
            next_submit_seq=lambda: transaction.next_submit_seq,
            next_apply_seq=lambda: transaction.next_apply_seq,
            reconstruct_snapshot=_reconstructor(restored),
            submit=lambda snapshot: transaction.submit(executor, worker, snapshot),
        )
        order.append("submit_current")
        assert transaction.reducer_state["epochs"] == [0]
        with transaction.checkpoint():
            pass
        order.append("join_final")

    assert decision_rows == (0,)
    assert transaction.reducer_state["epochs"] == [0, 25]
    assert order == [
        "join_previous",
        "decide_previous",
        "checkpoint",
        "submit_current",
        "join_final",
    ]
    obligation.assert_final_checkpoint_ready(
        next_submit_seq=transaction.next_submit_seq,
        next_apply_seq=transaction.next_apply_seq,
    )


def test_crash_after_checkpoint_before_dispatch_reconstructs_same_snapshot() -> None:
    continuous_tx = _FakeTransaction(sequence=3)
    continuous = PostCheckpointVerdictObligation()
    _arm(continuous, continuous_tx, epoch=50)
    checkpoint_arrays = {
        key: np.array(value, copy=True)
        for key, value in continuous.numpy_state(prefix=PREFIX).items()
    }

    continuous_restored = _restored_o1_o5(50)
    assert _discharge(continuous, continuous_tx, continuous_restored) == 3
    continuous_hash = _snapshot_hash(3, continuous_tx.snapshots[0])

    resumed = PostCheckpointVerdictObligation.from_numpy_state(
        checkpoint_arrays,
        prefix=PREFIX,
    )
    resumed_tx = _FakeTransaction(sequence=3)
    resumed_restored = _restored_o1_o5(50)
    assert _discharge(resumed, resumed_tx, resumed_restored) == 3
    resumed_hash = _snapshot_hash(3, resumed_tx.snapshots[0])

    assert resumed.record is None
    assert continuous_hash == resumed_hash
    assert np.array_equal(
        continuous_tx.snapshots[0]["scorer"]["ema_np"]["code"],
        resumed_tx.snapshots[0]["scorer"]["ema_np"]["code"],
    )


def test_crash_during_dispatch_poison_requires_checkpoint_restart() -> None:
    transaction = _FakeTransaction(sequence=2)
    state = PostCheckpointVerdictObligation()
    _arm(state, transaction, epoch=75)
    checkpoint_arrays = {
        key: np.array(value, copy=True)
        for key, value in state.numpy_state(prefix=PREFIX).items()
    }
    restored = _restored_o1_o5(75)
    attempted_hashes: list[str] = []

    def crash_after_submit(snapshot: Mapping[str, object]) -> int:
        sequence = transaction.submit(snapshot)
        attempted_hashes.append(_snapshot_hash(sequence, snapshot))
        raise OSError("simulated process crash during dispatch")

    with pytest.raises(VerdictObligationDispatchError, match="indeterminate"):
        state.discharge(
            restored_checkpoint_epoch=75,
            restored_stage="tau_softplus",
            restored_boundary_kind="intra_stage",
            restored_config_sha256=CONFIG_SHA,
            next_submit_seq=lambda: transaction.next_submit_seq,
            next_apply_seq=lambda: transaction.next_apply_seq,
            reconstruct_snapshot=_reconstructor(restored),
            submit=crash_after_submit,
        )
    assert state.poisoned
    with pytest.raises(VerdictObligationPoisonedError):
        state.assert_optimizer_step_allowed()
    with pytest.raises(VerdictObligationPoisonedError):
        _discharge(state, transaction, restored)

    resumed = PostCheckpointVerdictObligation.from_numpy_state(
        checkpoint_arrays,
        prefix=PREFIX,
    )
    resumed_tx = _FakeTransaction(sequence=2)
    assert _discharge(resumed, resumed_tx, _restored_o1_o5(75)) == 2
    assert attempted_hashes == [_snapshot_hash(2, resumed_tx.snapshots[0])]


def test_rejects_duplicate_malformed_stale_and_wrong_coordinate() -> None:
    transaction = _FakeTransaction(sequence=5)
    state = PostCheckpointVerdictObligation()
    _arm(state, transaction)
    with pytest.raises(DuplicateVerdictObligationError):
        _arm(state, transaction)

    restored = _restored_o1_o5(25)
    transaction.next_submit_seq = 6
    transaction.next_apply_seq = 6
    with pytest.raises(StaleVerdictObligationError, match="cursors"):
        _discharge(state, transaction, restored)
    assert not state.poisoned

    transaction.next_submit_seq = 5
    transaction.next_apply_seq = 5
    wrong_epoch = _restored_o1_o5(26)
    with pytest.raises(StaleVerdictObligationError, match="coordinate/config"):
        _discharge(state, transaction, wrong_epoch)
    with pytest.raises(StaleVerdictObligationError, match="coordinate/config"):
        state.discharge(
            restored_checkpoint_epoch=25,
            restored_stage="tau_softplus",
            restored_boundary_kind="intra_stage",
            restored_config_sha256="0" * 64,
            next_submit_seq=lambda: transaction.next_submit_seq,
            next_apply_seq=lambda: transaction.next_apply_seq,
            reconstruct_snapshot=_reconstructor(restored),
            submit=transaction.submit,
        )
    assert not state.poisoned

    arrays = dict(state.numpy_state(prefix=PREFIX))
    arrays[f"{PREFIX}submission_seq"] = np.asarray(5, dtype=object)
    with pytest.raises(MalformedVerdictObligationError, match="int64 scalar"):
        PostCheckpointVerdictObligation.from_numpy_state(arrays, prefix=PREFIX)

    absent = PostCheckpointVerdictObligation().numpy_state(prefix=PREFIX)
    bad_absent = dict(absent)
    bad_absent[f"{PREFIX}stage"] = bad_absent[f"{PREFIX}stage"].copy()
    bad_absent[f"{PREFIX}stage"][:9] = np.frombuffer(
        b"not-empty",
        dtype=np.uint8,
    )
    with pytest.raises(MalformedVerdictObligationError, match="empty sentinels"):
        PostCheckpointVerdictObligation.from_numpy_state(
            bad_absent,
            prefix=PREFIX,
        )

    extra = dict(absent)
    extra[f"{PREFIX}snapshot"] = np.empty(0, dtype=np.uint8)
    with pytest.raises(MalformedVerdictObligationError, match="keyset"):
        PostCheckpointVerdictObligation.from_numpy_state(extra, prefix=PREFIX)


def test_final_epoch_joins_verdict_before_final_checkpoint() -> None:
    transaction = _FakeTransaction()
    state = PostCheckpointVerdictObligation()
    _arm(state, transaction, epoch=100)
    _discharge(state, transaction, _restored_o1_o5(100))

    with pytest.raises(FinalCheckpointNotReadyError, match="joined"):
        state.assert_final_checkpoint_ready(
            next_submit_seq=transaction.next_submit_seq,
            next_apply_seq=transaction.next_apply_seq,
        )
    transaction.join_and_apply()
    state.assert_final_checkpoint_ready(
        next_submit_seq=transaction.next_submit_seq,
        next_apply_seq=transaction.next_apply_seq,
    )
