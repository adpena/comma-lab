from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from experiments import ddm_pk3_frame0_pose_overlay_runtime as overlay
from experiments import ddm_pk3_frame0_pose_representation as pk3
from tac.payload_retention_gate import check_no_measure_and_discard_payload


def test_pose_overlay_roundtrip_and_exact_endpoints() -> None:
    controls = np.zeros((4, overlay.DIMENSIONS), dtype=np.int32)
    controls[0, 0] = -7
    controls[-1, 0] = 7
    controls[1, 3] = 2
    payload = overlay.encode_pose_overlay(controls)
    assert len(payload) == 6 + controls.size // 2
    assert np.array_equal(overlay.decode_pose_overlay(payload), controls)
    expanded = overlay.expand_pose_controls(controls)
    assert expanded.shape == (overlay.PAIR_COUNT, overlay.DIMENSIONS)
    assert expanded[0, 0] == -7
    assert expanded[-1, 0] == 7


def test_pose_overlay_rejects_reserved_nibble_and_trailing_bytes() -> None:
    controls = np.zeros((2, overlay.DIMENSIONS), dtype=np.int32)
    payload = bytearray(overlay.encode_pose_overlay(controls))
    payload[6] = 0xF0
    with pytest.raises(overlay.Frame0PoseOverlayError, match="reserved"):
        overlay.decode_pose_overlay(bytes(payload))
    with pytest.raises(overlay.Frame0PoseOverlayError, match="length"):
        overlay.decode_pose_overlay(overlay.encode_pose_overlay(controls) + b"\0")


def test_pose_overlay_applies_only_real_int12_lattice() -> None:
    base = np.zeros((overlay.PAIR_COUNT, overlay.DIMENSIONS), dtype=np.int32)
    controls = np.ones((2, overlay.DIMENSIONS), dtype=np.int32)
    result = overlay.apply_compensation_overlay(
        base, overlay.encode_pose_overlay(controls)
    )
    assert np.array_equal(result, np.ones_like(base))
    base[0, 0] = 2047
    with pytest.raises(overlay.Frame0PoseOverlayError, match="outside signed-int12"):
        overlay.apply_compensation_overlay(base, overlay.encode_pose_overlay(controls))


def test_temporal_weights_are_partition_of_unity() -> None:
    for pair in (0, 1, 299, 599):
        weights = pk3.temporal_weights(pair, 8)
        assert weights.shape == (8,)
        assert np.count_nonzero(weights) <= 2
        assert float(weights.sum()) == pytest.approx(1.0)
    assert np.array_equal(pk3.temporal_weights(0, 8), np.eye(8)[0])
    assert np.array_equal(pk3.temporal_weights(599, 8), np.eye(8)[-1])


def test_fit_controls_reduces_a_synthetic_real_lattice_model() -> None:
    pairs = np.asarray([0, 599], dtype=np.int16)
    jacobians = np.zeros((2, 6, 12), dtype=np.float64)
    jacobians[:, 0, 0] = 1.0
    errors = np.zeros((2, 6), dtype=np.float64)
    errors[0, 0] = 3.0
    errors[1, 0] = -3.0
    controls = pk3.fit_controls(
        pairs, jacobians, errors, knots=2, ridge=1e-8, gain=1.0
    )
    expanded = overlay.expand_pose_controls(controls)
    assert pk3.modeled_dpose(pairs, jacobians, errors, expanded) == 0.0
    assert controls[0, 0] == -3
    assert controls[1, 0] == 3


def test_pk3_runners_do_not_measure_and_discard_payloads() -> None:
    findings = check_no_measure_and_discard_payload(
        repo_root=pk3.REPO,
        roots=[
            Path("experiments/ddm_pk3_frame0_pose_overlay_runtime.py"),
            Path("experiments/ddm_pk3_frame0_pose_representation.py"),
        ],
    )
    assert findings == []
