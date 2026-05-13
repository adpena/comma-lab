"""Tests for tac.openpilot_seeding V2 fixes (Lane OS-V2).

V2 fixes 8 issues identified in the V1 audit:

1. Correct YUV420 planar preprocessing (4 quarter-Y + U + V), not 4xY-replicate.
2. features_buffer carried across inference calls (RNN-like recurrent state).
3. Calibration mode flag {none, linear, mlp}; default 'none'.
4. Auto-detect pose head + openpilot version pin.
5. Crop-to-FOV before resize.
6. Fallback to BASELINE poses by default (not lane-mark).
7. Percentage-based fx threshold (6%, not 50px).
8. openpilot version pin (covered by fix 4).
"""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

from tac.openpilot_seeding import (
    OPENPILOT_SUPERCOMBO_URL,
    SUPERCOMBO_FEATURES_BUFFER_NAMES,
    SUPERCOMBO_INPUT_NAME,
    SUPERCOMBO_INPUT_SHAPE,
    SUPERCOMBO_POSE_HEAD_END,
    SUPERCOMBO_POSE_HEAD_START,
    SUPERCOMBO_VERSION_PIN,
    SupercomboUnavailable,
    _auto_detect_pose_head_indices,
    _crop_to_road_fov,
    _frames_to_supercombo_yuv,
    _frames_to_supercombo_yuv_v1,
    fallback_seed_from_baseline,
    fit_calibration_mlp,
    infer_pose_from_video,
    seed_pose_tto,
)


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "experiments" / "seed_poses_from_openpilot.py"


# ─────────────────────────────────────────────────────────────────────
# Fix 1: correct YUV420 planar layout (4 quarter-Y + U + V)
# ─────────────────────────────────────────────────────────────────────


def test_correct_yuv420_layout() -> None:
    """V2 default produces YUV420 planar layout, not V1's Y-replicated layout.

    V1: per-frame channels were [Y, Y, Y, Y, U, V] (Y replicated 4 times).
    V2: per-frame channels are [Y[0::2,0::2], Y[0::2,1::2], Y[1::2,0::2],
        Y[1::2,1::2], U, V] (4 quarter-Y planes from full-res Y + chroma).

    To distinguish, we feed a frame with a horizontal gradient in R only.
    Under V1's 4xY-replicate, channels 0-3 are bit-identical. Under V2's
    quarter-Y layout, channels 0-3 are spatially shifted (different 2x2
    sub-pixel positions of the full-res Y), so they differ at the per-pixel
    level even though their global statistics are similar.
    """
    h, w = 256, 384
    # Horizontal gradient in R, zero in G/B; this gives a strong horizontal
    # gradient in Y (Y = 0.299 R + ...), so adjacent columns differ.
    f0 = torch.zeros(h, w, 3, dtype=torch.uint8)
    f1 = torch.zeros(h, w, 3, dtype=torch.uint8)
    grad = torch.linspace(0, 255, w).round().to(torch.uint8)  # (w,)
    f1[:, :, 0] = grad.unsqueeze(0).expand(h, -1)

    out_v2 = _frames_to_supercombo_yuv(f1, f0, fov_crop=False)
    out_v1 = _frames_to_supercombo_yuv_v1(f1, f0)

    assert out_v2.shape == (1, 12, 128, 256)
    assert out_v1.shape == (1, 12, 128, 256)

    # Channels 6..9 are the second-frame Y planes in both layouts.
    # In V1 they should all be identical (Y replicated). In V2 they should
    # differ (quarter-Y subsampling at different 2x2 phases).
    v1_y_planes = [out_v1[0, c] for c in range(6, 10)]
    v2_y_planes = [out_v2[0, c] for c in range(6, 10)]

    # V1: all four Y planes identical.
    for c in range(1, 4):
        assert torch.allclose(v1_y_planes[0], v1_y_planes[c]), (
            "V1 layout: all 4 Y channels should be identical"
        )

    # V2: the four quarter-Y planes should differ from each other (they're
    # 2x2 sub-pixel phases of the same Y signal). Pick the most-likely-to-
    # differ pair: y0 (top-left) vs y1 (top-right) — differ by one column.
    # Bilinear-smoothed Y has a gradient ~1 step per pixel; quarter-Y planes
    # at half-res are offset by 1 source-pixel column → diff is very small in
    # absolute units (1/255 ≈ 0.004) but strictly nonzero where V1 is exactly
    # zero by construction.
    v2_diff_total = (v2_y_planes[0] - v2_y_planes[1]).abs().sum().item()
    v1_diff_total = (v1_y_planes[0] - v1_y_planes[1]).abs().sum().item()
    assert v1_diff_total == 0.0, (
        "V1 layout: Y channels are byte-identical (Y replicated)"
    )
    assert v2_diff_total > 0.0, (
        f"V2 quarter-Y planes y0 and y1 should differ under a horizontal "
        f"gradient (got total abs diff = {v2_diff_total:.6f})"
    )
    # And the cross-layout difference should be large (V2 is fundamentally
    # different from V1 channel content).
    cross_diff = (v2_y_planes[0] - v1_y_planes[0]).abs().mean().item()
    assert cross_diff > 1e-4, (
        f"V2 quarter-Y plane should differ from V1 replicated-Y plane "
        f"(got mean abs diff = {cross_diff:.6f})"
    )


def test_yuv_per_frame_channel_count_unchanged() -> None:
    """Fix 1 must not change the (1, 12, 128, 256) input contract."""
    f = torch.zeros(384, 512, 3, dtype=torch.uint8)
    out = _frames_to_supercombo_yuv(f, f, fov_crop=False)
    assert out.shape == SUPERCOMBO_INPUT_SHAPE


# ─────────────────────────────────────────────────────────────────────
# Fix 2: features_buffer recurrent state propagation
# ─────────────────────────────────────────────────────────────────────


class _RecurrentFakeSupercombo:
    """Fake session that has a features_buffer input AND output.

    Each ``run()`` call returns:
      - out[0]: (1, 6504) flat tensor with the pose head filled with the
        current sum of features_buffer (so we can verify that the buffer
        was actually fed in).
      - out[1]: (1, 99, 128) features_buffer output = previous_buffer + 1
        (so we can verify the propagation step-by-step).
    """

    def __init__(self) -> None:
        self.last_features_buffer_in: Any | None = None
        self.call_count = 0

    def get_inputs(self) -> list[Any]:
        class _Spec:
            def __init__(self, name: str, shape: tuple[int, ...]) -> None:
                self.name = name
                self.shape = shape

        return [
            _Spec(SUPERCOMBO_INPUT_NAME, SUPERCOMBO_INPUT_SHAPE),
            _Spec("desire", (1, 100, 8)),
            _Spec("features_buffer", (1, 99, 128)),
        ]

    def get_outputs(self) -> list[Any]:
        class _Spec:
            def __init__(self, name: str, shape: tuple[int, ...]) -> None:
                self.name = name
                self.shape = shape

        return [
            _Spec("outputs", (1, 6504)),
            _Spec("features_buffer", (1, 99, 128)),
        ]

    def run(self, _output_names: list[str] | None, feed: dict[str, Any]) -> list[Any]:
        import numpy as np

        fb_in = feed["features_buffer"]
        self.last_features_buffer_in = np.asarray(fb_in).copy()
        # Output[0]: pose head encodes the sum of features_buffer so we can
        # verify it was non-zero on the second call.
        out0 = np.zeros((1, 6504), dtype=np.float32)
        marker = float(np.asarray(fb_in).sum())
        out0[0, SUPERCOMBO_POSE_HEAD_START:SUPERCOMBO_POSE_HEAD_END] = (
            np.array([marker, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        )
        # Output[1]: features_buffer = previous + 1 (so it's strictly nonzero
        # after the first call).
        fb_out = np.asarray(fb_in, dtype=np.float32) + 1.0
        self.call_count += 1
        return [out0, fb_out]


@pytest.fixture()
def tiny_video(tmp_path: Path) -> Path:
    """Build a tiny 4-frame video for the recurrent-state tests."""
    av = pytest.importorskip("av")
    import numpy as np

    video_path = tmp_path / "tiny.mkv"
    container = av.open(str(video_path), mode="w")
    stream = container.add_stream("h264", rate=20)
    stream.width = 64
    stream.height = 48
    stream.pix_fmt = "yuv420p"
    for i in range(4):
        arr = np.full((48, 64, 3), i * 30, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)
    container.close()
    return video_path


def test_features_buffer_propagated(tiny_video: Path) -> None:
    """V2 propagates features_buffer between inferences (RNN state carry-over)."""
    sess = _RecurrentFakeSupercombo()
    poses = infer_pose_from_video(
        sess, tiny_video, n_frames=4, device=torch.device("cpu"),
        propagate_features_buffer=True,
    )
    assert poses.shape == (2, 6)

    # First call: features_buffer in = zeros → marker (sum) = 0.0.
    assert poses[0, 0].item() == pytest.approx(0.0), (
        f"first-call marker should be 0 (zero buffer), got {poses[0, 0].item()}"
    )
    # Second call: features_buffer in = (zeros + 1) shape (1, 99, 128) =
    # 99*128 = 12672. Marker should equal 12672.
    expected_marker = 99 * 128
    assert poses[1, 0].item() == pytest.approx(expected_marker), (
        f"second-call marker should be {expected_marker} after propagation, "
        f"got {poses[1, 0].item()}"
    )


def test_features_buffer_disabled_when_flag_false(tiny_video: Path) -> None:
    """propagate_features_buffer=False returns to V1 zero-state-every-call."""
    sess = _RecurrentFakeSupercombo()
    poses = infer_pose_from_video(
        sess, tiny_video, n_frames=4, device=torch.device("cpu"),
        propagate_features_buffer=False,
    )
    # Both calls have zero features_buffer → marker = 0 for both.
    assert poses[0, 0].item() == pytest.approx(0.0)
    assert poses[1, 0].item() == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────
# Fix 3: calibration mode {none, linear, mlp}
# ─────────────────────────────────────────────────────────────────────


def test_seed_pose_tto_mode_none_returns_raw() -> None:
    """V2 default mode='none' returns raw poses unchanged (TTO learns scale)."""
    raw = torch.randn(50, 6) + 1.5
    baseline = torch.randn(50, 6) + 31.0
    out = seed_pose_tto(raw, baseline_poses=baseline, mode="none")
    assert torch.allclose(out, raw)


def test_seed_pose_tto_mode_linear_matches_v1() -> None:
    """mode='linear' produces V1's per-dim affine result."""
    torch.manual_seed(7)
    raw = torch.randn(200, 6) * 0.3
    raw[:, 0] += 1.5
    baseline = torch.randn(200, 6) * 1.265
    baseline[:, 0] += 31.295

    out = seed_pose_tto(raw, baseline_poses=baseline, mode="linear")
    # Per-dim mean and std should match baseline.
    for d in range(6):
        assert abs(out[:, d].mean().item() - baseline[:, d].mean().item()) < 0.01
        assert abs(out[:, d].std().item() - baseline[:, d].std().item()) < 0.01


def test_seed_pose_tto_mode_mlp_fits_baseline() -> None:
    """mode='mlp' fits a small MLP that drives MSE down vs the raw poses."""
    torch.manual_seed(13)
    raw = torch.randn(200, 6) * 0.3
    raw[:, 0] += 1.5
    baseline = torch.randn(200, 6) * 1.265
    baseline[:, 0] += 31.295

    raw_mse = torch.nn.functional.mse_loss(raw, baseline).item()
    out = seed_pose_tto(raw, baseline_poses=baseline, mode="mlp")
    mlp_mse = torch.nn.functional.mse_loss(out, baseline).item()
    assert mlp_mse < raw_mse, (
        f"mlp calibration should reduce MSE vs baseline "
        f"(raw {raw_mse:.4f} vs mlp {mlp_mse:.4f})"
    )


def test_seed_pose_tto_mode_invalid() -> None:
    raw = torch.randn(10, 6)
    with pytest.raises(ValueError, match="invalid mode"):
        seed_pose_tto(raw, mode="bogus")


def test_seed_pose_tto_v1_compat_scale_to_match_true() -> None:
    """scale_to_match=True (V1 alias) → mode='linear'."""
    torch.manual_seed(17)
    raw = torch.randn(100, 6) * 0.3
    baseline = torch.randn(100, 6) * 2.0 + 5.0
    out_alias = seed_pose_tto(raw, baseline_poses=baseline, scale_to_match=True)
    out_explicit = seed_pose_tto(raw, baseline_poses=baseline, mode="linear")
    assert torch.allclose(out_alias, out_explicit)


def test_seed_pose_tto_v1_compat_scale_to_match_false() -> None:
    """scale_to_match=False (V1 alias) → mode='none' → raw passthrough."""
    raw = torch.randn(100, 6) + 1.5
    baseline = torch.randn(100, 6) + 31.0
    out = seed_pose_tto(raw, baseline_poses=baseline, scale_to_match=False)
    assert torch.allclose(out, raw)


def test_fit_calibration_mlp_shape_and_eval() -> None:
    raw = torch.randn(100, 6)
    baseline = torch.randn(100, 6) + 5.0
    mlp = fit_calibration_mlp(raw, baseline, hidden=4, epochs=10)
    out = mlp(raw)
    assert out.shape == raw.shape
    # MLP must be in eval() mode after fit.
    assert not mlp.training


# ─────────────────────────────────────────────────────────────────────
# Fix 4: auto-detect pose head + version pin
# ─────────────────────────────────────────────────────────────────────


def test_supercombo_version_pin_format() -> None:
    """Version pin must be a v0.X.Y tag, not 'master'."""
    assert SUPERCOMBO_VERSION_PIN.startswith("v"), (
        f"version pin should be a release tag like v0.9.7, got "
        f"{SUPERCOMBO_VERSION_PIN!r}"
    )
    assert SUPERCOMBO_VERSION_PIN != "master", (
        "version pin must be a specific release, not master"
    )
    assert OPENPILOT_SUPERCOMBO_URL.endswith("supercombo.onnx")
    assert SUPERCOMBO_VERSION_PIN in OPENPILOT_SUPERCOMBO_URL, (
        "URL must interpolate the version pin"
    )


def test_pose_head_auto_detection_falls_back_to_canonical() -> None:
    """Auto-detect returns the canonical [5755:5761] when no named head exists."""

    class _NoNamedPoseHead:
        def get_outputs(self) -> list[Any]:
            class _Spec:
                def __init__(self, name: str, shape: tuple[int, ...]) -> None:
                    self.name = name
                    self.shape = shape
            return [_Spec("outputs", (1, 6504))]

    sess = _NoNamedPoseHead()
    start, end = _auto_detect_pose_head_indices(sess)
    assert (start, end) == (
        SUPERCOMBO_POSE_HEAD_START, SUPERCOMBO_POSE_HEAD_END,
    )
    assert end - start == 6


def test_pose_head_auto_detection_named_head() -> None:
    """If a 'pose' output with a 6-dim shape is present, auto-detect returns canonical (informational match)."""

    class _NamedPoseHead:
        def get_outputs(self) -> list[Any]:
            class _Spec:
                def __init__(self, name: str, shape: tuple[int, ...]) -> None:
                    self.name = name
                    self.shape = shape
            return [
                _Spec("outputs", (1, 6504)),
                _Spec("pose", (1, 6)),
            ]

    sess = _NamedPoseHead()
    start, end = _auto_detect_pose_head_indices(sess)
    assert end - start == 6


def test_pose_head_auto_detection_with_sample_output() -> None:
    """A sample output with finite values at the canonical slice validates it."""
    import numpy as np

    class _Bare:
        def get_outputs(self) -> list[Any]:
            return []

    sample = np.zeros((1, 6504), dtype=np.float32)
    sample[0, SUPERCOMBO_POSE_HEAD_START:SUPERCOMBO_POSE_HEAD_END] = (
        np.arange(6, dtype=np.float32)
    )
    start, end = _auto_detect_pose_head_indices(_Bare(), sample_output=sample)
    assert (start, end) == (SUPERCOMBO_POSE_HEAD_START, SUPERCOMBO_POSE_HEAD_END)


def test_features_buffer_names_includes_canonical() -> None:
    """The detection list must include the canonical 'features_buffer'."""
    assert "features_buffer" in SUPERCOMBO_FEATURES_BUFFER_NAMES


# ─────────────────────────────────────────────────────────────────────
# Fix 5: road FOV crop before resize
# ─────────────────────────────────────────────────────────────────────


def test_road_fov_crop_default_fractions() -> None:
    """Default crop drops top 30% (sky) and bottom 5% (dashboard)."""
    pair = torch.randn(2, 3, 100, 200)
    cropped = _crop_to_road_fov(pair)
    # Default: top=0.30, bottom=0.95 → height 65% → 65 rows.
    assert cropped.shape == (2, 3, 65, 200), (
        f"expected (2, 3, 65, 200), got {tuple(cropped.shape)}"
    )


def test_road_fov_crop_preserves_road_band() -> None:
    """Mid-band content survives the crop; sky/dashboard rows do not."""
    pair = torch.zeros(1, 1, 100, 50)
    pair[0, 0, 50, :] = 99.0  # mid-frame row should survive
    pair[0, 0, 5, :] = 88.0   # top sky row should be dropped
    pair[0, 0, 99, :] = 77.0  # bottom dashboard row should be dropped

    cropped = _crop_to_road_fov(pair)
    assert (cropped == 99.0).any(), "mid-frame content should survive"
    assert not (cropped == 88.0).any(), "top sky content should be cropped"
    assert not (cropped == 77.0).any(), "bottom dashboard content should be cropped"


def test_road_fov_crop_invalid_fractions() -> None:
    pair = torch.zeros(2, 3, 100, 200)
    with pytest.raises(ValueError, match="invalid FOV crop"):
        _crop_to_road_fov(pair, fov_top_frac=0.5, fov_bottom_frac=0.4)


def test_road_fov_crop_4d_required() -> None:
    pair = torch.zeros(3, 100, 200)
    with pytest.raises(ValueError, match="4D"):
        _crop_to_road_fov(pair)


def test_yuv_with_fov_crop_still_produces_supercombo_shape() -> None:
    """fov_crop=True still yields (1, 12, 128, 256)."""
    h, w = 384, 512
    f0 = torch.zeros(h, w, 3, dtype=torch.uint8)
    f1 = torch.full((h, w, 3), 128, dtype=torch.uint8)
    out = _frames_to_supercombo_yuv(f1, f0, fov_crop=True)
    assert out.shape == SUPERCOMBO_INPUT_SHAPE


# ─────────────────────────────────────────────────────────────────────
# Fix 6: fallback to BASELINE poses (default), not lane_mark
# ─────────────────────────────────────────────────────────────────────


def test_fallback_seed_from_baseline_loads_pt(tmp_path: Path) -> None:
    """fallback_seed_from_baseline loads a (N, 6) tensor from disk."""
    poses = torch.randn(600, 6) + 31.0
    p = tmp_path / "baseline.pt"
    torch.save(poses, p)
    out = fallback_seed_from_baseline(p)
    assert out.shape == (600, 6)
    assert torch.allclose(out, poses)


def test_fallback_seed_from_baseline_truncates(tmp_path: Path) -> None:
    """If baseline has more pairs than requested, slice."""
    poses = torch.randn(800, 6)
    p = tmp_path / "baseline.pt"
    torch.save(poses, p)
    out = fallback_seed_from_baseline(p, n_pairs=600)
    assert out.shape == (600, 6)
    assert torch.allclose(out, poses[:600])


def test_fallback_seed_from_baseline_pads(tmp_path: Path) -> None:
    """If baseline has fewer pairs than requested, pad by repeating last pose."""
    poses = torch.randn(500, 6)
    p = tmp_path / "baseline.pt"
    torch.save(poses, p)
    out = fallback_seed_from_baseline(p, n_pairs=600)
    assert out.shape == (600, 6)
    assert torch.allclose(out[:500], poses)
    # Padded rows = last row of original.
    for i in range(500, 600):
        assert torch.allclose(out[i], poses[-1])


def test_fallback_seed_from_baseline_missing_file(tmp_path: Path) -> None:
    with pytest.raises(SupercomboUnavailable, match="not found"):
        fallback_seed_from_baseline(tmp_path / "does_not_exist.pt")


def test_fallback_seed_from_baseline_wrong_shape(tmp_path: Path) -> None:
    bad = torch.randn(600, 5)
    p = tmp_path / "bad.pt"
    torch.save(bad, p)
    with pytest.raises(SupercomboUnavailable, match=r"\(N, 6\)"):
        fallback_seed_from_baseline(p)


# Standalone-tool fallback flag tests — verify --fallback-mode reaches the
# right code path.


def test_standalone_default_fallback_mode_baseline(tmp_path: Path) -> None:
    """--fallback-mode default = baseline; loads --baseline-poses."""
    baseline = torch.randn(600, 6) + 31.0
    bp = tmp_path / "baseline.pt"
    torch.save(baseline, bp)
    output = tmp_path / "seed.pt"
    fake_supercombo = tmp_path / "missing.onnx"

    result = subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--supercombo-path", str(fake_supercombo),
            "--output", str(output),
            "--device", "cpu",
            "--n-frames", "1200",
            "--baseline-poses", str(bp),
            "--allow-fallback",
            # No --fallback-mode → default=baseline.
        ],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, (
        f"baseline fallback failed: stdout={result.stdout!r} "
        f"stderr={result.stderr!r}"
    )
    assert output.exists()
    assert "fallback_kind=baseline" in result.stdout


def test_standalone_lane_mark_mode_requires_masks(tmp_path: Path) -> None:
    """--fallback-mode=lane_mark without --masks exits 1."""
    output = tmp_path / "seed.pt"
    fake_supercombo = tmp_path / "missing.onnx"
    result = subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--supercombo-path", str(fake_supercombo),
            "--output", str(output),
            "--device", "cpu",
            "--allow-fallback",
            "--fallback-mode", "lane_mark",
        ],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 1


# ─────────────────────────────────────────────────────────────────────
# Fix 7: percentage-based fx threshold (6%, not 50px)
# ─────────────────────────────────────────────────────────────────────


def test_fx_percentage_threshold_no_warn_within_6pct(
    tiny_video: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """fx within 6% of 910 (i.e. > 855) should NOT warn."""

    class _Pose:
        call_count = 0

        def get_inputs(self) -> list[Any]:
            class _S:
                def __init__(self, n: str, sh: tuple[int, ...]) -> None:
                    self.name = n
                    self.shape = sh
            return [_S(SUPERCOMBO_INPUT_NAME, SUPERCOMBO_INPUT_SHAPE)]

        def get_outputs(self) -> list[Any]:
            class _S:
                def __init__(self, n: str, sh: tuple[int, ...]) -> None:
                    self.name = n
                    self.shape = sh
            return [_S("outputs", (1, 6504))]

        def run(self, _names: list[str] | None, _feed: dict[str, Any]) -> list[Any]:
            import numpy as np
            out = np.zeros((1, 6504), dtype=np.float32)
            return [out]

    caplog.set_level(logging.WARNING, logger="tac.openpilot_seeding")
    # 910 * (1 - 0.05) = 864.5 — within 6% percentage threshold; V1's 50-px
    # threshold would flag this (910 - 864.5 = 45.5 < 50, also no warn under
    # V1; pick a value V1 warns on: 910 - 860 = 50 → V1 warns at ">50";
    # under V2 6%: 50/910 ≈ 5.5% → NO warn).
    fx_within_pct = 860.0
    infer_pose_from_video(
        _Pose(), tiny_video, n_frames=2, device=torch.device("cpu"),
        fx=fx_within_pct,
    )
    msgs = [r.message for r in caplog.records]
    assert not any(">6%" in m for m in msgs), (
        f"fx={fx_within_pct} within 6% should not warn; got: {msgs}"
    )


def test_fx_percentage_threshold_warns_outside_6pct(
    tiny_video: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """fx > 6% off should warn."""

    class _Pose:
        def get_inputs(self) -> list[Any]:
            class _S:
                def __init__(self, n: str, sh: tuple[int, ...]) -> None:
                    self.name = n
                    self.shape = sh
            return [_S(SUPERCOMBO_INPUT_NAME, SUPERCOMBO_INPUT_SHAPE)]

        def get_outputs(self) -> list[Any]:
            return []

        def run(self, _names: list[str] | None, _feed: dict[str, Any]) -> list[Any]:
            import numpy as np
            return [np.zeros((1, 6504), dtype=np.float32)]

    caplog.set_level(logging.WARNING, logger="tac.openpilot_seeding")
    fx_off = 700.0  # |700 - 910| / 910 = 23% — well above 6%.
    infer_pose_from_video(
        _Pose(), tiny_video, n_frames=2, device=torch.device("cpu"),
        fx=fx_off,
    )
    msgs = [r.message for r in caplog.records]
    assert any(">6%" in m for m in msgs), (
        f"fx={fx_off} (23% off) should trigger 6% warning; got: {msgs}"
    )


# ─────────────────────────────────────────────────────────────────────
# Sanity: V1 standalone test_fallback_produces_correct_shape still works
# (regression guard — V2 must not break the existing V1 fallback path).
# ─────────────────────────────────────────────────────────────────────


def test_standalone_argparse_has_v2_flags(tmp_path: Path) -> None:
    """V2 flags must be exposed by argparse (preflight_arity contract)."""
    text = SCRIPT.read_text()
    for flag in (
        "--scale-to-match-mode",
        "--fallback-mode",
        "--fov-crop",
        "--no-fov-crop",
        "--legacy-v1-yuv",
        "--no-features-buffer",
    ):
        assert f'"{flag}"' in text, f"V2 standalone tool missing flag {flag}"
