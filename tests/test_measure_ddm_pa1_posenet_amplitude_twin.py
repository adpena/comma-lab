from __future__ import annotations

import numpy as np

from tac.canonical_equations.ddm_pa1_posenet_amplitude_twin_laws_20260723 import (
    E2_DPOSE,
    E2_DSEG,
    FRAME0_SCORER_DPOSE,
    FRAME0_SCORER_DSEG,
    JOINT_SCORER_DPOSE,
    JOINT_SCORER_DSEG,
    amplitude_gap_is_small,
    joint_score_delta,
    receiver_survives,
    target_rate_partition,
)
from tools.measure_ddm_pa1_posenet_amplitude_twin import (
    CHANNEL_NAMES,
    _delta,
    _encode_affine,
    _encode_frame0_affine,
    _fit_affine,
    _inverse_yuv6,
    _moment_state,
    _moment_update,
)


def _moments(values: np.ndarray) -> dict:
    import torch

    state = _moment_state(values.shape[1])
    _moment_update(state, torch.from_numpy(values))
    return state


def test_fp16_affine_payload_prices_exactly_24_and_48_bytes() -> None:
    gain = np.ones((1, len(CHANNEL_NAMES)), dtype=np.float32)
    bias = np.zeros_like(gain)
    assert len(_encode_affine(gain, bias)) == 48
    assert len(_encode_frame0_affine(gain[0], bias[0])) == 24


def test_fit_affine_matches_channel_moments_after_fp16_quantization() -> None:
    rng = np.random.default_rng(7)
    source = rng.normal(80.0, 8.0, size=(5, 12, 4, 6)).astype(np.float32)
    target = source * np.linspace(1.0, 2.1, 12, dtype=np.float32)[None, :, None, None]
    target += np.arange(12, dtype=np.float32)[None, :, None, None]
    gain, bias = _fit_affine(_moments(source), _moments(target))
    realized = source * gain[None, :, None, None] + bias[None, :, None, None]
    np.testing.assert_allclose(
        realized.mean(axis=(0, 2, 3)),
        target.mean(axis=(0, 2, 3)),
        rtol=3e-3,
        atol=3e-2,
    )
    np.testing.assert_allclose(
        realized.var(axis=(0, 2, 3)),
        target.var(axis=(0, 2, 3)),
        rtol=3e-3,
        atol=3e-2,
    )


def test_inverse_yuv6_places_all_four_luma_parities() -> None:
    import torch

    yuv = torch.full((1, 1, 6, 2, 3), 128.0)
    yuv[:, :, 0] = 50.0
    yuv[:, :, 1] = 60.0
    yuv[:, :, 2] = 70.0
    yuv[:, :, 3] = 80.0
    rgb = _inverse_yuv6(yuv)
    expected = torch.empty((4, 6))
    expected[0::2, 0::2] = 50.0
    expected[1::2, 0::2] = 60.0
    expected[0::2, 1::2] = 70.0
    expected[1::2, 1::2] = 80.0
    for channel in range(3):
        torch.testing.assert_close(rgb[0, 0, channel], expected)


def test_joint_delta_prices_pose_seg_and_exact_bytes() -> None:
    baseline = {"d_pose": 4.0, "d_seg": 0.02, "pose_term": np.sqrt(40.0)}
    arm = {
        "d_pose": 3.0,
        "d_seg": 0.021,
        "pose_term": np.sqrt(30.0),
        "counted_delta_bytes": 48,
    }
    row = _delta(baseline, arm)
    assert row["delta_d_pose"] == -1.0
    np.testing.assert_allclose(row["delta_d_seg"], 0.001, rtol=0.0, atol=1e-15)
    assert row["delta_bytes"] == 48
    assert row["delta_rate_term"] == 25.0 * 48 / 37_545_489
    np.testing.assert_allclose(
        row["joint_delta_s"],
        0.1 + np.sqrt(30.0) - np.sqrt(40.0) + 25.0 * 48 / 37_545_489,
        rtol=0.0,
        atol=1e-15,
    )


def test_measured_frame0_and_joint_laws_keep_joint_price_honest() -> None:
    frame0 = joint_score_delta(
        E2_DSEG,
        E2_DPOSE,
        FRAME0_SCORER_DSEG,
        FRAME0_SCORER_DPOSE,
        0,
    )
    joint = joint_score_delta(
        E2_DSEG,
        E2_DPOSE,
        JOINT_SCORER_DSEG,
        JOINT_SCORER_DPOSE,
        0,
    )
    np.testing.assert_allclose(frame0, -1.9167666862136272, atol=1e-14)
    np.testing.assert_allclose(joint, 0.6045627713754421, atol=1e-14)
    assert frame0 < 0.0
    assert joint > 0.0


def test_falsifier_and_rate_partition_are_explicit() -> None:
    assert not amplitude_gap_is_small(
        0.20144544838533338,
        0.28378726561837414,
        0.1,
    )
    assert (
        target_rate_partition(
            receiver_effective=True,
            target_uses_video_derived_fact=False,
        )
        == "FREE"
    )
    assert (
        target_rate_partition(
            receiver_effective=True,
            target_uses_video_derived_fact=True,
        )
        == "COUNTED"
    )
    assert (
        target_rate_partition(
            receiver_effective=False,
            target_uses_video_derived_fact=False,
        )
        == "NULL"
    )


def test_receiver_survival_requires_batches_frame1_and_free_payload() -> None:
    rows = [f"{index:064x}" for index in range(38)]
    frame1 = "f" * 64
    assert receiver_survives(
        source_batch_sha256=rows,
        packaged_batch_sha256=list(rows),
        source_frame1_sha256=frame1,
        packaged_frame1_sha256=frame1,
        amplitude_payload_bytes=0,
    )
    assert not receiver_survives(
        source_batch_sha256=rows,
        packaged_batch_sha256=[*rows[:-1], "e" * 64],
        source_frame1_sha256=frame1,
        packaged_frame1_sha256=frame1,
        amplitude_payload_bytes=0,
    )
    assert not receiver_survives(
        source_batch_sha256=rows,
        packaged_batch_sha256=list(rows),
        source_frame1_sha256=frame1,
        packaged_frame1_sha256="e" * 64,
        amplitude_payload_bytes=0,
    )
    assert not receiver_survives(
        source_batch_sha256=rows,
        packaged_batch_sha256=list(rows),
        source_frame1_sha256=frame1,
        packaged_frame1_sha256=frame1,
        amplitude_payload_bytes=1,
    )
