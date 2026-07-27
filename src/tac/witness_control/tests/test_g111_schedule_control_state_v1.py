from __future__ import annotations

import numpy as np
import pytest

from tac.witness_control.g111_schedule_control_state_v1 import (
    G111ScheduleControlStateError,
    new_state,
    state_arrays,
    state_from_arrays,
)

CFG = "ab" * 32
PREFIX = "__g111_o3__"


def _state(epoch: int = 0):
    return new_state(
        typed_config_sha256=CFG,
        completed_epoch=epoch,
        next_epoch=epoch + 1,
        accepted_optimizer_steps=epoch * 3,
        stop_latched=False,
        control_scalars={
            "prev_seg_form": "ce",
            "muon_switched": False,
            "last_boundary_epoch": None,
        },
        resume_control_arrays={
            "__resume_epoch": np.asarray(epoch, dtype=np.int64),
            "__evt_loss_tail": np.arange(epoch + 1, dtype=np.float32),
        },
    )


def test_roundtrip_preserves_arrays_and_binds_config() -> None:
    restored = state_from_arrays(
        state_arrays(_state(4), prefix=PREFIX),
        prefix=PREFIX,
        expected_typed_config_sha256=CFG,
    )
    assert restored["coordinate"] == {
        "accepted_optimizer_steps": 12,
        "completed_epoch": 4,
        "next_epoch": 5,
        "stop_latched": False,
    }
    np.testing.assert_array_equal(
        restored["resume_control_arrays"]["__evt_loss_tail"],
        np.arange(5, dtype=np.float32),
    )


def test_cold_and_live_array_topology_is_fixed() -> None:
    cold = state_arrays(_state(0), prefix=PREFIX)
    live = state_arrays(_state(200), prefix=PREFIX)
    assert set(cold) == set(live)
    assert {
        key: (value.dtype.str, value.shape) for key, value in cold.items()
    } == {
        key: (value.dtype.str, value.shape) for key, value in live.items()
    }


def test_rejects_non_o3_resume_arrays() -> None:
    with pytest.raises(G111ScheduleControlStateError, match="cannot own"):
        new_state(
            typed_config_sha256=CFG,
            completed_epoch=0,
            next_epoch=1,
            accepted_optimizer_steps=0,
            stop_latched=False,
            control_scalars={},
            resume_control_arrays={"liveP__code": np.zeros(1, np.float32)},
        )


def test_rejects_coordinate_skip_and_stop_advance() -> None:
    with pytest.raises(G111ScheduleControlStateError, match="completed_epoch"):
        new_state(
            typed_config_sha256=CFG,
            completed_epoch=2,
            next_epoch=4,
            accepted_optimizer_steps=1,
            stop_latched=False,
            control_scalars={},
            resume_control_arrays={},
        )
    with pytest.raises(G111ScheduleControlStateError, match="must not advance"):
        new_state(
            typed_config_sha256=CFG,
            completed_epoch=2,
            next_epoch=3,
            accepted_optimizer_steps=1,
            stop_latched=True,
            control_scalars={},
            resume_control_arrays={},
        )


def test_rejects_nonfinite_and_noncanonical_padding() -> None:
    with pytest.raises(G111ScheduleControlStateError, match="finite or None"):
        new_state(
            typed_config_sha256=CFG,
            completed_epoch=0,
            next_epoch=1,
            accepted_optimizer_steps=0,
            stop_latched=False,
            control_scalars={"bad": float("inf")},
            resume_control_arrays={},
        )
    arrays = {key: np.array(value, copy=True) for key, value in state_arrays(_state(), prefix=PREFIX).items()}
    length = int(arrays[f"{PREFIX}state_payload_length"])
    arrays[f"{PREFIX}state_payload"][length] = 1
    with pytest.raises(G111ScheduleControlStateError, match="nonzero bytes"):
        state_from_arrays(arrays, prefix=PREFIX)


def test_rejects_wrong_active_config() -> None:
    with pytest.raises(G111ScheduleControlStateError, match="active config"):
        state_from_arrays(
            state_arrays(_state(), prefix=PREFIX),
            prefix=PREFIX,
            expected_typed_config_sha256="cd" * 32,
        )
