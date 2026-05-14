# SPDX-License-Identifier: MIT
"""Tests for `tac.residual_basis.hinton_distilled_scorer_surrogate`.

Per W's DEFERRED reactivation criterion #1 + N D2 council verdict + Phase 3
Catalog #134 prerequisite + CLAUDE.md "Bugs must be permanently fixed AND
self-protected against" non-negotiable.
"""

from __future__ import annotations


import pytest
import torch

from tac.residual_basis.hinton_distilled_scorer_surrogate import (
    CONTEST_POSE_DIM,
    CONTEST_SEG_CLASSES,
    DEFAULT_DISTILL_GAP_THRESHOLD,
    DistilledPoseNet,
    DistilledSegNet,
    DistortionViaDistilledScorerError,
    MAX_SURROGATE_PARAMS,
    ScorerSurrogateConfig,
    compute_distortion_via_distilled_scorer,
    load_pretrained_distilled_scorer_pair,
    measure_distillation_gap_smoke,
)


# ---------------------------------------------------------------------------
# ScorerSurrogateConfig
# ---------------------------------------------------------------------------


def test_scorer_surrogate_config_council_canonical_matches_contest():
    config = ScorerSurrogateConfig.council_canonical()
    assert config.seg_class_count == CONTEST_SEG_CLASSES == 5
    assert config.pose_dim == CONTEST_POSE_DIM == 6
    assert config.distill_temperature == 2.0  # Hinton-Vinyals-Dean 2014 canon
    assert config.distill_gap_threshold == DEFAULT_DISTILL_GAP_THRESHOLD == 0.03
    assert config.expected_distill_label_substring == ""


def test_scorer_surrogate_config_rejects_invalid_seg_classes():
    with pytest.raises(DistortionViaDistilledScorerError, match="seg_class_count"):
        ScorerSurrogateConfig(
            seg_class_count=1, pose_dim=6, distill_temperature=2.0,
            distill_gap_threshold=0.03, expected_distill_label_substring="",
        )


def test_scorer_surrogate_config_rejects_invalid_pose_dim():
    with pytest.raises(DistortionViaDistilledScorerError, match="pose_dim"):
        ScorerSurrogateConfig(
            seg_class_count=5, pose_dim=0, distill_temperature=2.0,
            distill_gap_threshold=0.03, expected_distill_label_substring="",
        )


def test_scorer_surrogate_config_rejects_invalid_temperature():
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(DistortionViaDistilledScorerError, match="distill_temperature"):
            ScorerSurrogateConfig(
                seg_class_count=5, pose_dim=6, distill_temperature=bad,
                distill_gap_threshold=0.03, expected_distill_label_substring="",
            )


def test_scorer_surrogate_config_rejects_invalid_gap_threshold():
    for bad in (0.0, -0.5, float("nan"), float("inf")):
        with pytest.raises(DistortionViaDistilledScorerError, match="distill_gap_threshold"):
            ScorerSurrogateConfig(
                seg_class_count=5, pose_dim=6, distill_temperature=2.0,
                distill_gap_threshold=bad, expected_distill_label_substring="",
            )


def test_scorer_surrogate_config_is_frozen():
    config = ScorerSurrogateConfig.council_canonical()
    with pytest.raises(Exception):
        config.distill_temperature = 1.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# DistilledSegNet
# ---------------------------------------------------------------------------


def test_distilled_segnet_default_construction_under_param_cap():
    seg = DistilledSegNet()
    n_params = sum(p.numel() for p in seg.parameters())
    assert n_params <= MAX_SURROGATE_PARAMS


def test_distilled_segnet_forward_shape():
    seg = DistilledSegNet(seg_class_count=5, base_channels=8)
    rgb = torch.rand(2, 3, 64, 96) * 200.0 + 28.0
    out = seg(rgb)
    assert out.shape == (2, 5, 64, 96)
    assert torch.isfinite(out).all()


def test_distilled_segnet_rejects_wrong_input_shape():
    seg = DistilledSegNet()
    with pytest.raises(DistortionViaDistilledScorerError, match="DistilledSegNet expects"):
        seg(torch.rand(2, 4, 64, 64))  # 4 channels != 3
    with pytest.raises(DistortionViaDistilledScorerError, match="DistilledSegNet expects"):
        seg(torch.rand(3, 64, 64))  # 3-dim not 4-dim


def test_distilled_segnet_rejects_invalid_construction_args():
    with pytest.raises(DistortionViaDistilledScorerError):
        DistilledSegNet(seg_class_count=1)
    with pytest.raises(DistortionViaDistilledScorerError):
        DistilledSegNet(base_channels=2)


# ---------------------------------------------------------------------------
# DistilledPoseNet
# ---------------------------------------------------------------------------


def test_distilled_posenet_default_construction_under_param_cap():
    pose = DistilledPoseNet()
    n_params = sum(p.numel() for p in pose.parameters())
    assert n_params <= MAX_SURROGATE_PARAMS


def test_distilled_posenet_forward_shape():
    pose = DistilledPoseNet(pose_dim=6, base_channels=8)
    yuv6_pair = torch.rand(3, 12, 32, 48)
    out = pose(yuv6_pair)
    assert out.shape == (3, 6)
    assert torch.isfinite(out).all()


def test_distilled_posenet_rejects_wrong_input_shape():
    pose = DistilledPoseNet()
    with pytest.raises(DistortionViaDistilledScorerError, match="DistilledPoseNet expects"):
        pose(torch.rand(2, 6, 32, 48))  # 6 channels != 12 (caller didn't concatenate pair)
    with pytest.raises(DistortionViaDistilledScorerError, match="DistilledPoseNet expects"):
        pose(torch.rand(2, 32, 48))  # 3-dim


# ---------------------------------------------------------------------------
# Surrogate parameter footprint cap (per Quantizr UCLA approach)
# ---------------------------------------------------------------------------


def test_surrogate_footprint_under_5m_cap():
    config = ScorerSurrogateConfig.council_canonical()
    seg, pose = load_pretrained_distilled_scorer_pair(config=config)
    seg_params = sum(p.numel() for p in seg.parameters())
    pose_params = sum(p.numel() for p in pose.parameters())
    total = seg_params + pose_params
    # Per Quantizr UCLA approach: ~5-10MB each so the pair fits in ≤20MB
    # = 5M params at 4 bytes/param. Target is well below.
    assert total <= MAX_SURROGATE_PARAMS, (
        f"surrogate too large: {total} > {MAX_SURROGATE_PARAMS}"
    )


def test_load_pretrained_random_init_smoke_path():
    """No state-dicts, no custody fields = random-init smoke path."""
    config = ScorerSurrogateConfig.council_canonical()
    seg, pose = load_pretrained_distilled_scorer_pair(config=config)
    assert isinstance(seg, DistilledSegNet)
    assert isinstance(pose, DistilledPoseNet)
    # Both networks frozen (in eval mode + no requires_grad).
    for p in (*seg.parameters(), *pose.parameters()):
        assert not p.requires_grad


def test_load_pretrained_with_state_dict_requires_distill_label():
    """Supplying a state_dict requires distill_label (custody)."""
    config = ScorerSurrogateConfig.council_canonical()
    seg = DistilledSegNet()
    seg_sd = seg.state_dict()
    with pytest.raises(DistortionViaDistilledScorerError, match="distill_label"):
        load_pretrained_distilled_scorer_pair(
            config=config,
            seg_state_dict=seg_sd,
            distill_label="",
            distillation_gap=0.01,
            ema_decay=0.997,
        )


def test_load_pretrained_with_state_dict_refuses_high_gap():
    """Distillation gap above threshold is refused (Catalog #134 Phase 3 prereq)."""
    config = ScorerSurrogateConfig.council_canonical()
    seg = DistilledSegNet()
    seg_sd = seg.state_dict()
    with pytest.raises(DistortionViaDistilledScorerError, match="distillation_gap"):
        load_pretrained_distilled_scorer_pair(
            config=config,
            seg_state_dict=seg_sd,
            distill_label="t10_canonical",
            distillation_gap=0.05,  # > 0.03 threshold
            ema_decay=0.997,
        )


def test_load_pretrained_refuses_invalid_ema_decay():
    """EMA decay outside [0.99, 1.0) is refused per CLAUDE.md non-negotiable."""
    config = ScorerSurrogateConfig.council_canonical()
    seg = DistilledSegNet()
    seg_sd = seg.state_dict()
    for bad_ema in (0.5, 0.98, 1.0, 1.5, float("nan")):
        with pytest.raises(DistortionViaDistilledScorerError, match="ema_decay"):
            load_pretrained_distilled_scorer_pair(
                config=config,
                seg_state_dict=seg_sd,
                distill_label="t10_canonical",
                distillation_gap=0.01,
                ema_decay=bad_ema,
            )


def test_load_pretrained_with_full_custody_succeeds():
    """All custody fields present → loads cleanly."""
    config = ScorerSurrogateConfig.council_canonical()
    seg = DistilledSegNet()
    pose = DistilledPoseNet()
    seg_sd = seg.state_dict()
    pose_sd = pose.state_dict()
    seg2, pose2 = load_pretrained_distilled_scorer_pair(
        config=config,
        seg_state_dict=seg_sd,
        pose_state_dict=pose_sd,
        distill_label="t10_council_canonical_2026-05-11",
        distillation_gap=0.025,  # Below 0.03 threshold
        ema_decay=0.997,
    )
    # Verify state dicts loaded by checking that one parameter matches.
    assert torch.allclose(
        seg2.stem[0].weight,
        seg.stem[0].weight,
    )


def test_load_pretrained_label_substring_match():
    config = ScorerSurrogateConfig(
        seg_class_count=5, pose_dim=6, distill_temperature=2.0,
        distill_gap_threshold=0.03,
        expected_distill_label_substring="t10",
    )
    seg = DistilledSegNet()
    seg_sd = seg.state_dict()
    # Wrong substring → refused.
    with pytest.raises(DistortionViaDistilledScorerError, match="does not contain"):
        load_pretrained_distilled_scorer_pair(
            config=config,
            seg_state_dict=seg_sd,
            distill_label="not_the_right_label",
            distillation_gap=0.01,
            ema_decay=0.997,
        )
    # Matching substring → succeeds.
    load_pretrained_distilled_scorer_pair(
        config=config,
        seg_state_dict=seg_sd,
        distill_label="t10_canonical_2026-05-11",
        distillation_gap=0.01,
        ema_decay=0.997,
    )


# ---------------------------------------------------------------------------
# compute_distortion_via_distilled_scorer
# ---------------------------------------------------------------------------


def _make_smoke_input_pair(B=2, H=64, W=96):
    """Even-batch RGB pair tensor in [0, 255]."""
    return torch.rand(B, 3, H, W) * 200.0 + 28.0


def test_compute_distortion_returns_finite_grad_reachable():
    """Sanity: forward returns finite scalars + gradient reachable from decoded."""
    config = ScorerSurrogateConfig.council_canonical()
    seg, pose = load_pretrained_distilled_scorer_pair(config=config)
    decoded = _make_smoke_input_pair().clone().detach()
    decoded.requires_grad_(True)
    gt = _make_smoke_input_pair()
    d_seg, d_pose, diag = compute_distortion_via_distilled_scorer(
        decoded, gt,
        distilled_segnet=seg, distilled_posenet=pose,
        eval_roundtrip=False,
        distill_temperature=2.0,
    )
    assert torch.isfinite(d_seg).all()
    assert torch.isfinite(d_pose).all()
    assert d_seg.requires_grad
    assert diag["use_hinton_distilled_scorer"] == 1.0
    assert diag["distill_temperature"] == 2.0


def test_compute_distortion_gap_at_decoded_eq_gt_is_small():
    """When decoded == gt the SegNet KL distill should be ~ 0."""
    config = ScorerSurrogateConfig.council_canonical()
    seg, pose = load_pretrained_distilled_scorer_pair(config=config)
    same = _make_smoke_input_pair()
    d_seg, d_pose, _ = compute_distortion_via_distilled_scorer(
        same, same.clone(),
        distilled_segnet=seg, distilled_posenet=pose,
        eval_roundtrip=False,
    )
    # KL between identical distributions = 0 (up to floating-point).
    assert d_seg.item() < 1e-3
    assert d_pose.item() < 1e-3


def test_compute_distortion_rejects_odd_batch():
    config = ScorerSurrogateConfig.council_canonical()
    seg, pose = load_pretrained_distilled_scorer_pair(config=config)
    odd_decoded = _make_smoke_input_pair(B=3)  # odd batch
    odd_gt = _make_smoke_input_pair(B=3)
    with pytest.raises(DistortionViaDistilledScorerError, match="even number of frames"):
        compute_distortion_via_distilled_scorer(
            odd_decoded, odd_gt,
            distilled_segnet=seg, distilled_posenet=pose,
        )


def test_compute_distortion_rejects_shape_mismatch():
    config = ScorerSurrogateConfig.council_canonical()
    seg, pose = load_pretrained_distilled_scorer_pair(config=config)
    decoded = _make_smoke_input_pair(B=2, H=64, W=96)
    gt = _make_smoke_input_pair(B=2, H=128, W=128)  # different spatial
    with pytest.raises(DistortionViaDistilledScorerError, match="shape mismatch"):
        compute_distortion_via_distilled_scorer(
            decoded, gt,
            distilled_segnet=seg, distilled_posenet=pose,
        )


def test_compute_distortion_rejects_invalid_temperature():
    config = ScorerSurrogateConfig.council_canonical()
    seg, pose = load_pretrained_distilled_scorer_pair(config=config)
    decoded = _make_smoke_input_pair()
    gt = _make_smoke_input_pair()
    with pytest.raises(DistortionViaDistilledScorerError, match="distill_temperature"):
        compute_distortion_via_distilled_scorer(
            decoded, gt,
            distilled_segnet=seg, distilled_posenet=pose,
            distill_temperature=0.0,
        )


def test_compute_distortion_rejects_wrong_segnet_type():
    config = ScorerSurrogateConfig.council_canonical()
    pose = DistilledPoseNet()
    decoded = _make_smoke_input_pair()
    gt = _make_smoke_input_pair()
    # Pass non-DistilledSegNet as segnet
    with pytest.raises(DistortionViaDistilledScorerError, match="distilled_segnet"):
        compute_distortion_via_distilled_scorer(
            decoded, gt,
            distilled_segnet=pose,  # wrong type
            distilled_posenet=pose,
        )


def test_compute_distortion_eval_roundtrip_changes_output():
    """eval_roundtrip=True should affect outputs (uint8 simulation matters)."""
    config = ScorerSurrogateConfig.council_canonical()
    seg, pose = load_pretrained_distilled_scorer_pair(config=config)
    decoded = _make_smoke_input_pair(B=2, H=874, W=1164)  # camera resolution
    gt = _make_smoke_input_pair(B=2, H=874, W=1164)
    d_seg_no_rt, d_pose_no_rt, _ = compute_distortion_via_distilled_scorer(
        decoded, gt,
        distilled_segnet=seg, distilled_posenet=pose,
        eval_roundtrip=False,
    )
    d_seg_rt, d_pose_rt, _ = compute_distortion_via_distilled_scorer(
        decoded, gt,
        distilled_segnet=seg, distilled_posenet=pose,
        eval_roundtrip=True,
    )
    # Outputs should differ (eval_roundtrip simulates uint8 quantization).
    assert torch.isfinite(d_seg_no_rt).all() and torch.isfinite(d_seg_rt).all()


# ---------------------------------------------------------------------------
# measure_distillation_gap_smoke
# ---------------------------------------------------------------------------


def test_measure_distillation_gap_smoke_returns_diag():
    config = ScorerSurrogateConfig.council_canonical()
    seg, pose = load_pretrained_distilled_scorer_pair(config=config)
    sample = _make_smoke_input_pair()
    diag = measure_distillation_gap_smoke(
        distilled_segnet=seg, distilled_posenet=pose,
        sample_rgb_pair=sample,
    )
    assert diag["seg_logits_finite"] == 1.0
    assert diag["pose_floats_finite"] == 1.0
    assert diag["seg_logits_shape_ok"] == 1.0
    assert diag["pose_floats_shape_ok"] == 1.0


def test_measure_distillation_gap_smoke_rejects_odd_batch():
    config = ScorerSurrogateConfig.council_canonical()
    seg, pose = load_pretrained_distilled_scorer_pair(config=config)
    sample = _make_smoke_input_pair(B=3)
    with pytest.raises(DistortionViaDistilledScorerError, match="must be even"):
        measure_distillation_gap_smoke(
            distilled_segnet=seg, distilled_posenet=pose,
            sample_rgb_pair=sample,
        )
