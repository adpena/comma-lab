from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tools import probe_resumability_537_real_smoke as probe


def test_array_delta_reports_bit_identity_and_numeric_distance():
    left = {
        "liveP__w": np.asarray([1.0, 2.0], dtype=np.float32),
        "__resume_stage": np.asarray("unify_tau"),
    }
    same = probe._array_delta(left, {key: value.copy() for key, value in left.items()})
    assert same["bit_identical"] is True
    assert same["max_abs"] == 0.0

    changed = probe._array_delta(
        left,
        {
            "liveP__w": np.asarray([1.0, 2.25], dtype=np.float32),
            "__resume_stage": np.asarray("unify_tau"),
        },
    )
    assert changed["bit_identical"] is False
    assert changed["max_abs"] == 0.25


def test_gpu_verdict_uses_exact_measured_control_floor_without_multiplier():
    control_a = {"liveP__w": np.asarray([0.0], dtype=np.float32)}
    control_b = {"liveP__w": np.asarray([0.5], dtype=np.float32)}
    resumed_inside = {"liveP__w": np.asarray([0.25], dtype=np.float32)}
    resumed_outside = {"liveP__w": np.asarray([1.1], dtype=np.float32)}

    floor = probe._array_delta(control_a, control_b)
    inside = probe._within_measured_floor(
        floor,
        probe._array_delta(control_a, resumed_inside),
        probe._array_delta(control_b, resumed_inside),
    )
    outside = probe._within_measured_floor(
        floor,
        probe._array_delta(control_a, resumed_outside),
        probe._array_delta(control_b, resumed_outside),
    )

    assert floor["max_abs"] == 0.5
    assert inside["pass"] is True
    assert outside["pass"] is False
    assert outside["per_key"]["liveP__w"] == {
        "measured_control_floor_max_abs": 0.5,
        "resumed_nearest_control_max_abs": pytest.approx(0.6),
        "pass": False,
    }


def test_final_pair_check_excludes_periodic_pair(tmp_path: Path):
    true_pair = (
        tmp_path / "levelset_ckpt_stage_unify_tau_ep4.npz",
        tmp_path / "levelset_resume_stage_unify_tau_ep4.npz",
    )
    periodic_pair = (
        tmp_path / "levelset_periodic_ema_stage_unify_tau_ep4.npz",
        tmp_path / "levelset_periodic_resume_stage_unify_tau_ep4.npz",
    )
    for path in true_pair + periodic_pair:
        path.write_bytes(b"checkpoint")

    ema, resume = probe._final_stage_pair(tmp_path, 4)
    assert ema == [true_pair[0]]
    assert resume == [true_pair[1]]


def test_failure_receipt_does_not_reuse_environment_blocker_after_execution(tmp_path: Path):
    (tmp_path / "resumed").mkdir()
    (tmp_path / "resumed" / "levelset_resume_state.npz").write_bytes(b"preserved")

    failure = probe._failure_classification(
        RuntimeError("resume arm failed rc=1: interpreter path disappeared"), tmp_path,
    )
    assert failure["status"] == "EXECUTED_PROOF_ERROR"
    assert "RuntimeError: resume arm failed" in failure["exact_blocker"]
    assert failure["checkpoint_files"] == ["resumed/levelset_resume_state.npz"]
    assert failure["crash_epoch"] is None
    assert failure["final_pair_preserved"] is False


def test_no_metal_status_requires_no_checkpoint_progress(tmp_path: Path):
    failure = probe._failure_classification(
        RuntimeError("[metal::load_device] No Metal device available"), tmp_path,
    )
    assert failure["status"] == "BLOCKED_ENVIRONMENT_NO_METAL_DEVICE"
    assert failure["checkpoint_files"] == []


def test_failure_receipt_resolves_preserved_crash_epoch_and_final_pair(tmp_path: Path):
    crash = tmp_path / "crash"
    resumed = tmp_path / "resumed"
    crash.mkdir()
    resumed.mkdir()
    np.savez(
        crash / "levelset_periodic_resume_stage_unify_tau_ep3.npz",
        __resume_epoch=np.asarray(3, dtype=np.int64),
    )
    (resumed / "levelset_ckpt_stage_unify_tau_ep4.npz").write_bytes(b"ema")
    (resumed / "levelset_resume_stage_unify_tau_ep4.npz").write_bytes(b"resume")

    failure = probe._failure_classification(RuntimeError("late proof failure"), tmp_path)

    assert failure["status"] == "EXECUTED_PROOF_ERROR"
    assert failure["crash_epoch"] == 3
    assert failure["final_pair_preserved"] is True
    assert failure["final_pair_by_arm"] == {"resumed": True}
