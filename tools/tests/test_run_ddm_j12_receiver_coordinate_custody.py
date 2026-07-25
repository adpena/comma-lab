from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools import run_ddm_j12_receiver_coordinate_custody as j12


def _checkpoint_payload(step: int) -> bytes:
    values = np.arange(3, dtype=np.float32)
    return j12._canonical_npz(
        {
            "ema": values + 4,
            "first_moment": values + 8,
            "second_moment": values + 12,
            "step": np.asarray([step], dtype="<i8"),
            "theta": values,
        }
    )


def _write_checkpoint(root: Path, step: int, typed_hash: str) -> None:
    path = root / "05_conditional_smoke" / "checkpoints" / f"step_{step:03d}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _checkpoint_payload(step)
    path.write_bytes(payload)
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "typed_config_hash": typed_hash,
                "step": step,
                "npz_sha256": j12._sha256(payload),
                "telemetry": [{"step": step}],
            }
        ),
        encoding="utf-8",
    )


def test_canonical_npz_is_key_order_independent() -> None:
    first = j12._canonical_npz(
        {
            "z": np.asarray([3, 4], dtype=np.int64),
            "a": np.asarray([1, 2], dtype=np.float32),
        }
    )
    second = j12._canonical_npz(
        {
            "a": np.asarray([1, 2], dtype=np.float32),
            "z": np.asarray([3, 4], dtype=np.int64),
        }
    )
    assert first == second


def test_conditional_state_resume_loads_latest_complete_checkpoint(
    tmp_path: Path,
) -> None:
    typed_hash = "a" * 64
    for step in range(1, 5):
        _write_checkpoint(tmp_path, step, typed_hash)

    state, telemetry = j12._load_conditional_state(
        tmp_path,
        parameter_count=3,
        typed_hash=typed_hash,
    )

    assert state.step == 4
    np.testing.assert_array_equal(state.theta, np.arange(3, dtype=np.float32))
    assert telemetry == [{"step": 4}]


def test_conditional_state_resume_refuses_checkpoint_gap(tmp_path: Path) -> None:
    typed_hash = "b" * 64
    _write_checkpoint(tmp_path, 2, typed_hash)

    with pytest.raises(j12.J12Error, match="contain a gap"):
        j12._load_conditional_state(
            tmp_path,
            parameter_count=3,
            typed_hash=typed_hash,
        )


def test_conditional_state_resume_refuses_hash_drift(tmp_path: Path) -> None:
    typed_hash = "c" * 64
    _write_checkpoint(tmp_path, 1, typed_hash)
    path = tmp_path / "05_conditional_smoke/checkpoints/step_001.npz"
    path.write_bytes(path.read_bytes() + b"drift")

    with pytest.raises(j12.J12Error, match="checkpoint custody differs"):
        j12._load_conditional_state(
            tmp_path,
            parameter_count=3,
            typed_hash=typed_hash,
        )


def test_exact_price_uses_joint_objective_sign_only() -> None:
    baseline = {
        "d_seg": 0.1,
        "d_pose": 25.0,
        "archive_bytes": 1000,
    }
    endpoint = {
        "d_seg": 0.1,
        "d_pose": 16.0,
        "archive_bytes": 1100,
    }

    result = j12._price(baseline, endpoint)

    assert result["joint_delta"] < 0
    assert result["accepted"] is True
    assert result["acceptance_authority"] == "strict_realized_joint_delta_s_lt_zero"
