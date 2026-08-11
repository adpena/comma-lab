# SPDX-License-Identifier: MIT
"""Tests for ``tac.differentiable_eval_roundtrip``.

Coverage: PR #95 / PR #106 binary-forensics replication of (a) eval_roundtrip
baked into training inner loop and (b) autograd-preserving rgb_to_yuv6.

References:
  - Source dossier: ``.omx/research/hnerv_leaderboard_binary_forensics_dossier_20260509.md``
  - PR #95 oracle: ``experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/source/submissions/belt_and_suspenders/src/data.py:51-81``
  - PR #95 oracle: ``experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/source/submissions/belt_and_suspenders/src/stages/common.py:179-194``
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

# Ensure upstream is importable for the parity tests.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_UPSTREAM = _REPO_ROOT / "upstream"
if _UPSTREAM.is_dir() and str(_UPSTREAM) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM))

from tac.differentiable_eval_roundtrip import (  # noqa: E402
    CAMERA_HW,
    SCORER_HW,
    CameraLiftKernel,
    EvalRoundTripOrdering,
    Yuv6PatchToken,
    Yuv6RoutingMode,
    apply_camera_uint8_lift_during_training,
    apply_eval_roundtrip_during_training,
    assert_yuv6_forward_equivalence_to_upstream,
    differentiable_rgb_to_yuv6,
    patch_upstream_yuv6_globally,
    unpatch_upstream_yuv6,
)


def _upstream_available() -> bool:
    try:
        import frame_utils  # noqa: F401
    except ImportError:
        return False
    return True


_HAS_UPSTREAM = _upstream_available()


@pytest.fixture(scope="session")
def retained_real_work_rgb() -> torch.Tensor:
    """One real contest frame, decoded only by the canonical retained-source path.

    This fixture intentionally does not use random/synthetic pixels for HR2's
    receiver tests.  The material positive-control runner retains this decoded
    camera frame and every derived output on the SSD; the unit suite only uses
    the same canonical source frame as a behavior fixture.
    """
    if not _HAS_UPSTREAM:
        pytest.skip("upstream frame_utils unavailable")
    try:
        import av
        from frame_utils import yuv420_to_rgb
    except ImportError:
        pytest.skip("PyAV/canonical yuv420_to_rgb unavailable")
    video = _UPSTREAM / "videos" / "0.mkv"
    if not video.is_file():
        pytest.skip("retained upstream video unavailable")
    with av.open(str(video)) as container:
        frame = next(container.decode(container.streams.video[0]))
        camera_hwc = yuv420_to_rgb(frame)
    camera = camera_hwc.permute(2, 0, 1).unsqueeze(0).float()
    return F.interpolate(camera, size=SCORER_HW, mode="bilinear", align_corners=False)


def _explicit_camera_lift(
    work_rgb: torch.Tensor,
    *,
    kernel: CameraLiftKernel,
) -> torch.Tensor:
    lifted = F.interpolate(
        work_rgb,
        size=CAMERA_HW,
        mode=kernel.value,
        align_corners=False,
    )
    return lifted.clamp(0.0, 255.0).round()


def _explicit_camera_uint8_ste_roundtrip(
    work_rgb: torch.Tensor,
    *,
    kernel: CameraLiftKernel,
) -> torch.Tensor:
    lifted = F.interpolate(
        work_rgb,
        size=CAMERA_HW,
        mode=kernel.value,
        align_corners=False,
    )
    clamped = lifted.clamp(0.0, 255.0)
    camera_ste = clamped + (clamped.round() - clamped).detach()
    return F.interpolate(camera_ste, size=SCORER_HW, mode="bilinear", align_corners=False)


def _tensor_max_relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    scale = torch.maximum(actual.abs().max(), expected.abs().max()).clamp_min(1e-12)
    return float((actual - expected).abs().max() / scale)


# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #


def test_camera_hw_constants_match_pr95_recipe():
    """PR #95 source uses (874, 1164) as camera resolution."""
    assert CAMERA_HW == (874, 1164)


def test_scorer_hw_constants_match_pr95_recipe():
    """PR #95 source uses (384, 512) as scorer resolution."""
    assert SCORER_HW == (384, 512)


# --------------------------------------------------------------------------- #
# differentiable_rgb_to_yuv6 — forward equivalence                             #
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not _HAS_UPSTREAM, reason="upstream frame_utils unavailable")
def test_yuv6_zero_error_vs_upstream_random_inputs():
    """Random RGB inputs: bit-equivalent to upstream rgb_to_yuv6."""
    result = assert_yuv6_forward_equivalence_to_upstream(num_samples=10, atol=1e-6)
    assert result["passed"] is True
    assert result["max_abs_error"] == 0.0  # BT.601 coefficients are exact rationals.


@pytest.mark.skipif(not _HAS_UPSTREAM, reason="upstream frame_utils unavailable")
def test_yuv6_zero_error_vs_upstream_realistic_uint8_inputs():
    """uint8-quantized RGB at scorer res: bit-equivalent to upstream."""
    import frame_utils

    torch.manual_seed(20260509)
    # Match the contest layout: full scorer res, integer-valued floats.
    rgb = torch.randint(0, 256, (2, 3, 384, 512), dtype=torch.float32)
    with torch.no_grad():
        upstream = frame_utils.rgb_to_yuv6(rgb)
    local = differentiable_rgb_to_yuv6(rgb)
    torch.testing.assert_close(local, upstream, atol=0.0, rtol=0.0)


def test_yuv6_bt601_coefficients_pure_red():
    """Pure red pixel should produce Y = 255 * 0.299 (BT.601 coefficient)."""
    red = torch.zeros(1, 3, 4, 4)
    red[:, 0, :, :] = 255.0
    out = differentiable_rgb_to_yuv6(red)
    expected_y = 255.0 * 0.299
    assert abs(out[0, 0, 0, 0].item() - expected_y) < 1e-4


def test_yuv6_bt601_coefficients_pure_green():
    """Pure green pixel should produce Y = 255 * 0.587."""
    green = torch.zeros(1, 3, 4, 4)
    green[:, 1, :, :] = 255.0
    out = differentiable_rgb_to_yuv6(green)
    expected_y = 255.0 * 0.587
    assert abs(out[0, 0, 0, 0].item() - expected_y) < 1e-4


def test_yuv6_bt601_coefficients_pure_blue():
    """Pure blue pixel should produce Y = 255 * 0.114."""
    blue = torch.zeros(1, 3, 4, 4)
    blue[:, 2, :, :] = 255.0
    out = differentiable_rgb_to_yuv6(blue)
    expected_y = 255.0 * 0.114
    assert abs(out[0, 0, 0, 0].item() - expected_y) < 1e-4


def test_yuv6_output_shape_collapses_2x2():
    """Output spatial dims should be (H//2, W//2); 6 channels stacked."""
    rgb = torch.rand(2, 3, 64, 96) * 255.0
    out = differentiable_rgb_to_yuv6(rgb)
    assert out.shape == (2, 6, 32, 48)


def test_yuv6_supports_higher_dim_batches():
    """The (..., 3, H, W) signature should support 4D and 5D batches."""
    rgb_5d = torch.rand(2, 3, 3, 32, 32) * 255.0
    out = differentiable_rgb_to_yuv6(rgb_5d)
    assert out.shape == (2, 3, 6, 16, 16)


# --------------------------------------------------------------------------- #
# differentiable_rgb_to_yuv6 — autograd                                        #
# --------------------------------------------------------------------------- #


def test_yuv6_gradient_flows_through_rgb_input():
    """Backward through differentiable_rgb_to_yuv6 must yield non-zero RGB grads.

    This is the WHOLE POINT of the patch — without it, pose loss gradient is
    zero through the YUV6 op. Aaron's PR #95 quote: "pose plateaued at 142
    across 2500+ epochs" without this fix.
    """
    rgb = (torch.rand(1, 3, 16, 16) * 255.0).requires_grad_(True)
    out = differentiable_rgb_to_yuv6(rgb)
    out.sum().backward()
    assert rgb.grad is not None
    assert rgb.grad.abs().sum().item() > 0.0


def test_yuv6_gradient_nonzero_in_unclamped_region():
    """Gradients must be non-zero in the 'middle' (non-saturated) region.

    The clamp produces zero gradient at 0/255 boundaries; in the middle region
    (e.g. rgb=128) the gradient should pass through with the BT.601 coefficient.
    """
    rgb = torch.full((1, 3, 4, 4), 128.0, requires_grad=True)
    out = differentiable_rgb_to_yuv6(rgb)
    # Sum of Y outputs only; should give non-zero RGB grad with kYR/G/B sign.
    out[:, :4].sum().backward()
    assert rgb.grad is not None
    assert (rgb.grad[0, 0] != 0.0).any().item()  # R channel
    assert (rgb.grad[0, 1] != 0.0).any().item()  # G channel
    assert (rgb.grad[0, 2] != 0.0).any().item()  # B channel


# --------------------------------------------------------------------------- #
# differentiable_rgb_to_yuv6 — input validation                                #
# --------------------------------------------------------------------------- #


def test_yuv6_rejects_too_few_dims():
    with pytest.raises(ValueError, match="requires.*3, H, W"):
        differentiable_rgb_to_yuv6(torch.rand(8, 8))


def test_yuv6_rejects_wrong_channel_count():
    with pytest.raises(ValueError, match="3 channels"):
        differentiable_rgb_to_yuv6(torch.rand(2, 4, 16, 16))


# --------------------------------------------------------------------------- #
# apply_eval_roundtrip_during_training — PR #95 recipe semantics               #
# --------------------------------------------------------------------------- #


def test_eval_roundtrip_default_preserves_input_shape():
    """Default invocation must return the same shape as input."""
    rgb = torch.rand(2, 2, 3, 384, 512) * 255.0
    out = apply_eval_roundtrip_during_training(rgb)
    assert out.shape == rgb.shape


def test_eval_roundtrip_4d_batch_layout():
    """NCHW input (N, 3, H, W) is supported."""
    rgb = torch.rand(8, 3, 384, 512) * 255.0
    out = apply_eval_roundtrip_during_training(rgb)
    assert out.shape == (8, 3, 384, 512)


def test_eval_roundtrip_clamps_to_uint8_range():
    """Out-of-range inputs must be clamped to [0, 255] (PR #95 step 3)."""
    rgb = torch.full((1, 3, 64, 64), 1000.0)  # way out of [0, 255]
    out = apply_eval_roundtrip_during_training(rgb, simulate_resize=False)
    assert out.max().item() <= 255.0 + 1e-4
    assert out.min().item() >= 0.0 - 1e-4


def test_eval_roundtrip_round_is_integer_valued():
    """STE-round forward output must be integer-valued (uint8 simulation)."""
    rgb = torch.rand(1, 3, 64, 64) * 255.0
    out = apply_eval_roundtrip_during_training(rgb, simulate_resize=False)
    # Forward path is clamp + round, so output must equal its own round().
    torch.testing.assert_close(out, out.round(), atol=1e-4, rtol=0.0)


def test_eval_roundtrip_ste_gradient_passes_through_round():
    """STE round must allow gradient to flow back to input.

    Without STE, gradient through ``round()`` is zero everywhere. With STE,
    the backward pass is identity through the round step.
    """
    rgb = (torch.rand(1, 3, 32, 32) * 255.0).requires_grad_(True)
    out = apply_eval_roundtrip_during_training(rgb, simulate_resize=False)
    out.sum().backward()
    assert rgb.grad is not None
    # Gradient should be 1.0 wherever the input is in (0, 255).
    in_range = (rgb.detach() > 0.5) & (rgb.detach() < 254.5)
    assert in_range.any().item()
    assert rgb.grad[in_range].abs().min().item() > 0.0


def test_eval_roundtrip_resize_uses_bicubic_up_bilinear_down():
    """PR #95 uses bicubic UP, bilinear DOWN (asymmetric ladder).

    We cannot easily inspect F.interpolate calls, but we can verify that
    ``simulate_resize=False`` is functionally distinct from ``True`` on
    a non-trivial input.
    """
    torch.manual_seed(0)
    rgb = torch.rand(1, 3, 64, 64) * 255.0
    out_with = apply_eval_roundtrip_during_training(rgb, simulate_resize=True)
    out_without = apply_eval_roundtrip_during_training(rgb, simulate_resize=False)
    # The resize roundtrip should perturb the output.
    assert (out_with - out_without).abs().mean().item() > 0.01


def test_eval_roundtrip_no_simulate_uint8_skips_clamp_round():
    """``simulate_uint8=False`` returns the resize-only result, no clamp/round."""
    rgb = torch.full((1, 3, 8, 8), 1000.0)
    out = apply_eval_roundtrip_during_training(
        rgb, simulate_uint8=False, simulate_resize=False
    )
    # No clamp -> values stay at 1000.
    assert out.max().item() == 1000.0


def test_eval_roundtrip_ste_round_off_disables_round_only():
    """``ste_round=False`` keeps clamp but skips round."""
    rgb = torch.rand(1, 3, 32, 32) * 255.0 + 0.3  # non-integer
    out = apply_eval_roundtrip_during_training(
        rgb, ste_round=False, simulate_resize=False
    )
    # Should NOT be integer-valued (round skipped) but should be clamped.
    assert (out - out.round()).abs().max().item() > 0.01


def test_eval_roundtrip_idempotent_on_already_roundtripped_input():
    """Applying the roundtrip twice on the same input should be near-stable.

    The second pass should not radically change the output (already uint8-ish).
    """
    torch.manual_seed(42)
    rgb = torch.rand(1, 3, 64, 64) * 255.0
    out1 = apply_eval_roundtrip_during_training(rgb)
    out2 = apply_eval_roundtrip_during_training(out1)
    # Resampling adds some drift but shouldn't be huge.
    assert (out1 - out2).abs().mean().item() < 5.0


def test_eval_roundtrip_deterministic_for_same_input():
    """Same input must produce the same output (no hidden randomness)."""
    torch.manual_seed(0)
    rgb = torch.rand(1, 3, 64, 64) * 255.0
    out1 = apply_eval_roundtrip_during_training(rgb)
    out2 = apply_eval_roundtrip_during_training(rgb)
    torch.testing.assert_close(out1, out2)


# --------------------------------------------------------------------------- #
# HR2 typed camera-uint8 ordering — real-frame forward/backward controls       #
# --------------------------------------------------------------------------- #


def test_hr2_ordering_enum_is_closed_and_typed():
    assert {member.value for member in EvalRoundTripOrdering} == {
        "legacy_scorer_uint8",
        "camera_uint8",
    }


def test_hr2_lift_kernel_enum_represents_candidate_and_incumbent():
    assert {member.value for member in CameraLiftKernel} == {"bicubic", "bilinear"}


def test_hr2_invalid_ordering_refuses_on_real_frame(retained_real_work_rgb):
    with pytest.raises(ValueError, match="ordering must be one of"):
        apply_eval_roundtrip_during_training(
            retained_real_work_rgb,
            ordering="after_everything_somewhere",
        )


def test_hr2_invalid_lift_kernel_refuses_on_real_frame(retained_real_work_rgb):
    with pytest.raises(ValueError, match="lift_kernel must be one of"):
        apply_eval_roundtrip_during_training(
            retained_real_work_rgb,
            lift_kernel="nearest",
        )


def test_hr2_default_is_byte_identical_to_explicit_legacy_real_frame(
    retained_real_work_rgb,
):
    default = apply_eval_roundtrip_during_training(retained_real_work_rgb)
    explicit = apply_eval_roundtrip_during_training(
        retained_real_work_rgb,
        ordering=EvalRoundTripOrdering.LEGACY_SCORER_UINT8,
        lift_kernel=CameraLiftKernel.BICUBIC,
    )
    torch.testing.assert_close(default, explicit, atol=0.0, rtol=0.0)


def test_hr2_default_matches_pre_hr2_expression_exactly_on_real_frame(
    retained_real_work_rgb,
):
    up = F.interpolate(
        retained_real_work_rgb,
        size=CAMERA_HW,
        mode="bicubic",
        align_corners=False,
    )
    expected = F.interpolate(
        up,
        size=SCORER_HW,
        mode="bilinear",
        align_corners=False,
    ).clamp(0.0, 255.0).round()
    actual = apply_eval_roundtrip_during_training(retained_real_work_rgb)
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("kernel", list(CameraLiftKernel))
def test_hr2_training_camera_bytes_equal_public_receiver_expression_real_frame(
    retained_real_work_rgb,
    kernel,
):
    actual = apply_camera_uint8_lift_during_training(
        retained_real_work_rgb,
        lift_kernel=kernel,
    )
    expected = _explicit_camera_lift(retained_real_work_rgb, kernel=kernel)
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    assert torch.equal(actual.to(torch.uint8), expected.to(torch.uint8))


def test_hr2_camera_lift_returns_camera_shape_on_real_frame(retained_real_work_rgb):
    actual = apply_camera_uint8_lift_during_training(retained_real_work_rgb)
    assert actual.shape == (1, 3, *CAMERA_HW)


def test_hr2_camera_lift_forward_is_integer_valued_on_real_frame(
    retained_real_work_rgb,
):
    actual = apply_camera_uint8_lift_during_training(retained_real_work_rgb)
    torch.testing.assert_close(actual, actual.round(), atol=0.0, rtol=0.0)


@pytest.mark.parametrize("kernel", list(CameraLiftKernel))
def test_hr2_camera_uint8_roundtrip_matches_independent_forward_real_frame(
    retained_real_work_rgb,
    kernel,
):
    actual = apply_eval_roundtrip_during_training(
        retained_real_work_rgb,
        ordering=EvalRoundTripOrdering.CAMERA_UINT8,
        lift_kernel=kernel,
    )
    expected = _explicit_camera_uint8_ste_roundtrip(
        retained_real_work_rgb,
        kernel=kernel,
    )
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


def test_hr2_camera_uint8_downsample_remains_float_not_fake_second_round(
    retained_real_work_rgb,
):
    actual = apply_eval_roundtrip_during_training(
        retained_real_work_rgb,
        ordering=EvalRoundTripOrdering.CAMERA_UINT8,
    )
    assert (actual - actual.round()).abs().max().item() > 0.0


def test_hr2_orderings_are_materially_distinct_on_real_frame(retained_real_work_rgb):
    legacy = apply_eval_roundtrip_during_training(retained_real_work_rgb)
    camera = apply_eval_roundtrip_during_training(
        retained_real_work_rgb,
        ordering=EvalRoundTripOrdering.CAMERA_UINT8,
    )
    assert torch.count_nonzero(legacy != camera).item() > 0


def test_hr2_prior_law_rgb_channel_argmax_diff_is_nonzero_on_real_frame(
    retained_real_work_rgb,
):
    """Scorer-free prior-law proxy only; this is NOT a SegNet conclusion."""
    legacy = apply_eval_roundtrip_during_training(retained_real_work_rgb)
    camera = apply_eval_roundtrip_during_training(
        retained_real_work_rgb,
        ordering=EvalRoundTripOrdering.CAMERA_UINT8,
    )
    rgb_argmax_changes = torch.count_nonzero(legacy.argmax(1) != camera.argmax(1))
    assert rgb_argmax_changes.item() > 0


def test_hr2_camera_uint8_mode_is_deterministic_on_real_frame(retained_real_work_rgb):
    first = apply_eval_roundtrip_during_training(
        retained_real_work_rgb,
        ordering=EvalRoundTripOrdering.CAMERA_UINT8,
    )
    second = apply_eval_roundtrip_during_training(
        retained_real_work_rgb,
        ordering=EvalRoundTripOrdering.CAMERA_UINT8,
    )
    torch.testing.assert_close(first, second, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("kernel", list(CameraLiftKernel))
def test_hr2_per_tensor_gradient_max_relative_error_vs_cpu_torch_real_frame(
    retained_real_work_rgb,
    kernel,
):
    """#903 control: compare the full input-gradient tensor, never just loss."""
    actual_input = retained_real_work_rgb.detach().clone().requires_grad_(True)
    reference_input = retained_real_work_rgb.detach().clone().requires_grad_(True)
    weight = torch.linspace(
        0.25,
        1.25,
        SCORER_HW[0] * SCORER_HW[1],
        dtype=actual_input.dtype,
    ).reshape(1, 1, *SCORER_HW)
    actual = apply_eval_roundtrip_during_training(
        actual_input,
        ordering=EvalRoundTripOrdering.CAMERA_UINT8,
        lift_kernel=kernel,
    )
    reference = _explicit_camera_uint8_ste_roundtrip(
        reference_input,
        kernel=kernel,
    )
    (actual * weight).sum().backward()
    (reference * weight).sum().backward()
    assert actual_input.grad is not None
    assert reference_input.grad is not None
    max_rel_error = _tensor_max_relative_error(actual_input.grad, reference_input.grad)
    assert max_rel_error <= 1e-7


def test_hr2_gradient_control_would_detect_a_wrong_lift_kernel(retained_real_work_rgb):
    bicubic_input = retained_real_work_rgb.detach().clone().requires_grad_(True)
    bilinear_input = retained_real_work_rgb.detach().clone().requires_grad_(True)
    bicubic = apply_eval_roundtrip_during_training(
        bicubic_input,
        ordering=EvalRoundTripOrdering.CAMERA_UINT8,
        lift_kernel=CameraLiftKernel.BICUBIC,
    )
    bilinear = apply_eval_roundtrip_during_training(
        bilinear_input,
        ordering=EvalRoundTripOrdering.CAMERA_UINT8,
        lift_kernel=CameraLiftKernel.BILINEAR,
    )
    weight = torch.linspace(0.5, 1.5, bicubic.numel(), dtype=bicubic.dtype).reshape_as(bicubic)
    (bicubic * weight).sum().backward()
    (bilinear * weight).sum().backward()
    assert bicubic_input.grad is not None and bilinear_input.grad is not None
    assert _tensor_max_relative_error(bicubic_input.grad, bilinear_input.grad) > 1e-5


def test_hr2_camera_uint8_backward_is_nonzero_on_real_frame(retained_real_work_rgb):
    work = retained_real_work_rgb.detach().clone().requires_grad_(True)
    output = apply_eval_roundtrip_during_training(
        work,
        ordering=EvalRoundTripOrdering.CAMERA_UINT8,
    )
    output.square().mean().backward()
    assert work.grad is not None
    assert torch.count_nonzero(work.grad).item() > 0


def test_hr2_camera_lift_preserves_leading_pair_dimension_real_frame(
    retained_real_work_rgb,
):
    pair = torch.stack([retained_real_work_rgb[0], retained_real_work_rgb[0]], dim=0).unsqueeze(0)
    camera = apply_camera_uint8_lift_during_training(pair)
    scorer = apply_eval_roundtrip_during_training(
        pair,
        ordering=EvalRoundTripOrdering.CAMERA_UINT8,
    )
    assert camera.shape == (1, 2, 3, *CAMERA_HW)
    assert scorer.shape == pair.shape


def test_hr2_camera_mode_no_resize_quantizes_exactly_once_on_real_frame(
    retained_real_work_rgb,
):
    actual = apply_eval_roundtrip_during_training(
        retained_real_work_rgb,
        ordering=EvalRoundTripOrdering.CAMERA_UINT8,
        simulate_resize=False,
    )
    expected = retained_real_work_rgb.clamp(0.0, 255.0).round()
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


# --------------------------------------------------------------------------- #
# apply_eval_roundtrip_during_training — input validation                       #
# --------------------------------------------------------------------------- #


def test_roundtrip_rejects_wrong_channel_count():
    with pytest.raises(ValueError, match="3 channels"):
        apply_eval_roundtrip_during_training(torch.rand(2, 4, 16, 16))


def test_roundtrip_rejects_too_few_dims():
    with pytest.raises(ValueError, match="requires.*3, H, W"):
        apply_eval_roundtrip_during_training(torch.rand(16, 16))


def test_roundtrip_rejects_non_floating_input():
    with pytest.raises(ValueError, match="float tensor"):
        apply_eval_roundtrip_during_training(
            torch.randint(0, 255, (1, 3, 16, 16), dtype=torch.uint8)
        )


# --------------------------------------------------------------------------- #
# Global monkey-patch                                                           #
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not _HAS_UPSTREAM, reason="upstream frame_utils unavailable")
def test_global_patch_overwrites_frame_utils_rgb_to_yuv6():
    """patch_upstream_yuv6_globally swaps frame_utils.rgb_to_yuv6 with our differentiable version."""
    import frame_utils

    original = frame_utils.rgb_to_yuv6
    token = patch_upstream_yuv6_globally()
    try:
        assert frame_utils.rgb_to_yuv6 is differentiable_rgb_to_yuv6
        assert token.frame_utils_was_patched is True
    finally:
        unpatch_upstream_yuv6(token)
    assert frame_utils.rgb_to_yuv6 is original


@pytest.mark.skipif(not _HAS_UPSTREAM, reason="upstream frame_utils unavailable")
def test_global_patch_idempotent_token_returned():
    """Re-patching is a no-op (token reflects the current state)."""
    token1 = patch_upstream_yuv6_globally()
    try:
        token2 = patch_upstream_yuv6_globally()
        try:
            # Both tokens should record patched-state correctly; second
            # token's `*_was_patched` should be False (already was the
            # differentiable version, so no swap).
            assert token2.frame_utils_was_patched is False
        finally:
            unpatch_upstream_yuv6(token2)
    finally:
        unpatch_upstream_yuv6(token1)


@pytest.mark.skipif(not _HAS_UPSTREAM, reason="upstream frame_utils unavailable")
def test_global_patch_token_revert_is_safe_when_nothing_was_patched():
    """unpatch with a no-op token must not raise."""
    token = Yuv6PatchToken(
        frame_utils_orig=None,
        modules_orig=None,
        frame_utils_was_patched=False,
        modules_was_patched=False,
    )
    unpatch_upstream_yuv6(token)  # should not raise


# --------------------------------------------------------------------------- #
# Routing mode                                                                  #
# --------------------------------------------------------------------------- #


def test_yuv6_routing_mode_enum_values():
    """Routing mode enum must include both PR #95 and tac-routing options + AUTO."""
    assert Yuv6RoutingMode.MONKEY_PATCH_GLOBAL.value == "monkey_patch_global"
    assert Yuv6RoutingMode.TAC_DIFFERENTIABLE_ROUTING.value == "tac_differentiable_routing"
    assert Yuv6RoutingMode.AUTO.value == "auto"


def test_yuv6_routing_mode_string_constructible():
    """CLI flags must be parseable as enum values."""
    assert Yuv6RoutingMode("monkey_patch_global") is Yuv6RoutingMode.MONKEY_PATCH_GLOBAL
    assert Yuv6RoutingMode("tac_differentiable_routing") is Yuv6RoutingMode.TAC_DIFFERENTIABLE_ROUTING
    assert Yuv6RoutingMode("auto") is Yuv6RoutingMode.AUTO
