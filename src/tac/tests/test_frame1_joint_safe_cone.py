# SPDX-License-Identifier: MIT
"""Behavioral tests for the frame1 JOINT SAFE CONE.

Two layers:

1. **Unit (synthetic deterministic scorers)** — exercise the cone math /
   intersection / config validation / fail-closed gradient-reachability /
   per-region aggregation WITHOUT loading the 90MB upstream models. These verify
   BEHAVIOR (the cone radius actually responds to margin/slope/Jacobian), not
   constants — a body replaced by ``return canonical_markers`` would FAIL them.

2. **Real-scorer NO-FAKE proof** — gated on the presence of the upstream
   models + video. The cone's falsifiable claim: perturb frame1 INSIDE the cone
   -> ``d_seg`` stable + small ``d_pose``; perturb OUTSIDE (fragile pixels at 2x)
   -> measurable ``d_seg`` / ``d_pose`` movement. A cone that does not
   discriminate is FAKE (MEMORY.md Slot RR).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tac.optimization.frame1_joint_safe_cone import (
    FRAME1_JOINT_SAFE_CONE_SCHEMA,
    Frame1ConeConfig,
    Frame1JointSafeCone,
    Frame1JointSafeConeError,
    assemble_joint_cone,
    measure_posenet_frame1_jacobian,
    measure_segnet_frame1_boundary_slope,
    measure_segnet_frame1_margin,
    validate_cone_behaviorally,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
UPSTREAM = REPO_ROOT / "upstream"


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_config_defaults_valid():
    cfg = Frame1ConeConfig()
    assert cfg.seg_tau > 0 and cfg.pose_response_tol > 0


def test_config_rejects_bad_seg_tau():
    with pytest.raises(Frame1JointSafeConeError):
        Frame1ConeConfig(seg_tau=0.0)


def test_config_rejects_bad_margin_tol():
    with pytest.raises(Frame1JointSafeConeError):
        Frame1ConeConfig(seg_margin_tol=1.5)
    with pytest.raises(Frame1JointSafeConeError):
        Frame1ConeConfig(seg_margin_tol=0.0)


def test_config_rejects_negative_d_pose():
    with pytest.raises(Frame1JointSafeConeError):
        Frame1ConeConfig(d_pose=-1.0)


def test_config_rejects_bad_fragile_threshold():
    with pytest.raises(Frame1JointSafeConeError):
        Frame1ConeConfig(fragile_radius_threshold=-0.1)


def test_pose_ail_gain_matches_pr106_frontier():
    """At the PR106 frontier d_pose ~ 3.4e-5 the pose AIL marginal gain is
    5/sqrt(10*d_pose) ~ 271 = 100 * 2.71 (the CLAUDE.md 'pose 2.71x SegNet'
    marginal-value-flip operating point)."""

    cfg = Frame1ConeConfig(d_pose=3.4e-5)
    assert cfg.pose_ail_gain == pytest.approx(271.16, rel=1e-3)


def test_pose_ail_gain_grows_as_d_pose_shrinks():
    """The pose marginal value grows unboundedly as d_pose -> 0 (the contest
    sqrt(10*d_pose) term derivative); the cone must bind harder on pose at the
    frontier than at the old 1.x operating point."""

    near = Frame1ConeConfig(d_pose=1e-5).pose_ail_gain
    far = Frame1ConeConfig(d_pose=1e-1).pose_ail_gain
    assert near > far


# ---------------------------------------------------------------------------
# Cone assembly (synthetic deterministic inputs) — verifies BEHAVIOR
# ---------------------------------------------------------------------------


def _synthetic_inputs(h=16, w=24):
    """A deterministic synthetic margin/slope/jacobian/class grid with a known
    structure: a boundary column (small margin), a high-pose-Jacobian row, and a
    flat interior so the cone's response can be checked exactly."""

    rng = np.random.default_rng(7)
    margin = np.full((h, w), 5.0)
    margin[:, w // 2] = 0.05  # a boundary column: tiny distance-to-flip
    slope = np.full((h, w), 1.0)
    jpose = np.full((h, w), 1e-4)
    jpose[h // 2, :] = 0.2  # a high-pose-sensitivity row
    cls = (rng.integers(0, 5, size=(h, w))).astype(np.int64)
    cls[:, w // 2] = 1  # boundary column all one class
    return margin, slope, cls, jpose


def test_assemble_returns_cone_with_schema():
    margin, slope, cls, jpose = _synthetic_inputs()
    cone = assemble_joint_cone(
        seg_margin=margin, seg_boundary_slope=slope, seg_argmax_class=cls,
        pose_jacobian_norm=jpose, config=Frame1ConeConfig(),
    )
    assert isinstance(cone, Frame1JointSafeCone)
    assert cone.schema == FRAME1_JOINT_SAFE_CONE_SCHEMA
    assert cone.joint_cone_radius.shape == margin.shape


def test_boundary_column_has_smaller_radius_than_interior():
    """A small-margin (near-flip) column must get a SMALLER cone radius than the
    flat interior — the seg-safe half-cone responds to distance-to-flip."""

    margin, slope, cls, jpose = _synthetic_inputs()
    cone = assemble_joint_cone(
        seg_margin=margin, seg_boundary_slope=slope, seg_argmax_class=cls,
        pose_jacobian_norm=jpose, config=Frame1ConeConfig(),
    )
    w = margin.shape[1]
    boundary_r = cone.joint_cone_radius[:, w // 2].mean()
    interior_r = cone.joint_cone_radius[:, 0].mean()
    assert boundary_r < interior_r


def test_high_pose_jacobian_row_has_smaller_radius():
    """A high-pose-Jacobian row must get a SMALLER cone radius than pose-null
    rows — the pose-null half-cone responds to pose sensitivity."""

    margin, slope, cls, jpose = _synthetic_inputs()
    cone = assemble_joint_cone(
        seg_margin=margin, seg_boundary_slope=slope, seg_argmax_class=cls,
        pose_jacobian_norm=jpose, config=Frame1ConeConfig(),
    )
    h = margin.shape[0]
    hot_row_r = cone.joint_cone_radius[h // 2, :].mean()
    null_row_r = cone.joint_cone_radius[0, :].mean()
    assert hot_row_r < null_row_r


def test_radius_is_intersection_min_of_half_cones():
    """The joint cone radius is the MIN (intersection) of the seg + pose budgets:
    radius <= seg_margin_budget AND radius <= pose_budget everywhere."""

    margin, slope, cls, jpose = _synthetic_inputs()
    cone = assemble_joint_cone(
        seg_margin=margin, seg_boundary_slope=slope, seg_argmax_class=cls,
        pose_jacobian_norm=jpose, config=Frame1ConeConfig(max_radius=1e9),
    )
    assert np.all(cone.joint_cone_radius <= cone.seg_margin_budget + 1e-9)
    assert np.all(cone.joint_cone_radius <= cone.pose_budget + 1e-9)


def test_joint_sensitivity_is_p18_p19_coupling():
    """The reported joint sensitivity is the canonical P18/P19 coupling
    w = 100*slope + pose_ail_gain*J_pose (high = bind, low = free byte)."""

    margin, slope, cls, jpose = _synthetic_inputs()
    cfg = Frame1ConeConfig(d_pose=3.4e-5)
    cone = assemble_joint_cone(
        seg_margin=margin, seg_boundary_slope=slope, seg_argmax_class=cls,
        pose_jacobian_norm=jpose, config=cfg,
    )
    expected = 100.0 * slope + cfg.pose_ail_gain * jpose
    assert np.allclose(cone.joint_sensitivity, expected, rtol=1e-6)


def test_zero_margin_pixel_is_empty_cone():
    """A pixel exactly at the flip boundary (margin == 0) has zero budget."""

    margin, slope, cls, jpose = _synthetic_inputs()
    margin[3, 3] = 0.0
    cone = assemble_joint_cone(
        seg_margin=margin, seg_boundary_slope=slope, seg_argmax_class=cls,
        pose_jacobian_norm=jpose, config=Frame1ConeConfig(),
    )
    assert cone.joint_cone_radius[3, 3] == 0.0
    assert bool(cone.empty_cone_mask[3, 3])


def test_fragile_mask_marks_sub_threshold_pixels():
    margin, slope, cls, jpose = _synthetic_inputs()
    cone = assemble_joint_cone(
        seg_margin=margin, seg_boundary_slope=slope, seg_argmax_class=cls,
        pose_jacobian_norm=jpose, config=Frame1ConeConfig(fragile_radius_threshold=0.5),
    )
    expected = cone.joint_cone_radius < 0.5
    assert np.array_equal(cone.fragile_cone_mask, expected)


def test_per_region_aggregates_cover_all_classes():
    margin, slope, cls, jpose = _synthetic_inputs()
    cone = assemble_joint_cone(
        seg_margin=margin, seg_boundary_slope=slope, seg_argmax_class=cls,
        pose_jacobian_norm=jpose, config=Frame1ConeConfig(),
    )
    assert set(cone.per_region.keys()) == {int(c) for c in np.unique(cls)}
    total = sum(r["n_pixels"] for r in cone.per_region.values())
    assert total == int(cls.size)


def test_summary_fractions_in_unit_range():
    margin, slope, cls, jpose = _synthetic_inputs()
    cone = assemble_joint_cone(
        seg_margin=margin, seg_boundary_slope=slope, seg_argmax_class=cls,
        pose_jacobian_norm=jpose, config=Frame1ConeConfig(),
    )
    s = cone.summary
    assert 0.0 <= s["usable_budget_fraction"] <= 1.0
    assert 0.0 <= s["empty_cone_fraction"] <= 1.0
    assert s["usable_budget_fraction"] + s["empty_cone_fraction"] == pytest.approx(1.0)


def test_provenance_marks_non_promotable():
    """The cone is [macOS-CPU advisory] / non-promotable per Catalog #341/#323."""

    margin, slope, cls, jpose = _synthetic_inputs()
    cone = assemble_joint_cone(
        seg_margin=margin, seg_boundary_slope=slope, seg_argmax_class=cls,
        pose_jacobian_norm=jpose, config=Frame1ConeConfig(),
    )
    assert cone.provenance["promotable"] is False
    assert cone.provenance["score_claim"] is False
    assert cone.provenance["axis_tag"] == "[macOS-CPU advisory]"


def test_argmax_class_shape_mismatch_raises():
    margin, slope, _cls, jpose = _synthetic_inputs()
    with pytest.raises(Frame1JointSafeConeError):
        assemble_joint_cone(
            seg_margin=margin, seg_boundary_slope=slope,
            seg_argmax_class=np.zeros((2, 2), dtype=np.int64),
            pose_jacobian_norm=jpose, config=Frame1ConeConfig(),
        )


def test_larger_margin_tol_grows_seg_budget():
    """A larger seg_margin_tol allocates more of the distance-to-flip to budget
    -> larger seg_margin_budget (monotone in the tolerance)."""

    margin, slope, cls, jpose = _synthetic_inputs()
    small = assemble_joint_cone(
        seg_margin=margin, seg_boundary_slope=slope, seg_argmax_class=cls,
        pose_jacobian_norm=jpose, config=Frame1ConeConfig(seg_margin_tol=0.2, max_radius=1e9),
    )
    large = assemble_joint_cone(
        seg_margin=margin, seg_boundary_slope=slope, seg_argmax_class=cls,
        pose_jacobian_norm=jpose, config=Frame1ConeConfig(seg_margin_tol=0.8, max_radius=1e9),
    )
    assert np.all(large.seg_margin_budget >= small.seg_margin_budget - 1e-9)
    assert large.seg_margin_budget.mean() > small.seg_margin_budget.mean()


def test_tighter_pose_tol_shrinks_pose_budget():
    margin, slope, cls, jpose = _synthetic_inputs()
    loose = assemble_joint_cone(
        seg_margin=margin, seg_boundary_slope=slope, seg_argmax_class=cls,
        pose_jacobian_norm=jpose, config=Frame1ConeConfig(pose_response_tol=1e-2, max_radius=1e9),
    )
    tight = assemble_joint_cone(
        seg_margin=margin, seg_boundary_slope=slope, seg_argmax_class=cls,
        pose_jacobian_norm=jpose, config=Frame1ConeConfig(pose_response_tol=1e-4, max_radius=1e9),
    )
    assert tight.pose_budget.mean() < loose.pose_budget.mean()


def test_resize_aligns_native_slope_to_segnet_grid():
    """When the pose/slope native grid differs from the SegNet margin grid the
    cone resizes them onto the margin grid (a real intersection, not a crash)."""

    margin = np.full((32, 48), 4.0)
    cls = np.zeros((32, 48), dtype=np.int64)
    slope_native = np.full((16, 24), 1.0)  # half-res native grid
    jpose_native = np.full((16, 24), 1e-4)
    cone = assemble_joint_cone(
        seg_margin=margin, seg_boundary_slope=slope_native, seg_argmax_class=cls,
        pose_jacobian_norm=jpose_native, config=Frame1ConeConfig(),
    )
    assert cone.joint_cone_radius.shape == (32, 48)


# ---------------------------------------------------------------------------
# Fail-closed gradient reachability (the differentiable-YUV6 non-negotiable)
# ---------------------------------------------------------------------------


def test_posenet_jacobian_fails_closed_when_gradient_not_reachable():
    """If the PoseNet YUV6 graph is severed (the upstream @torch.no_grad path),
    the Jacobian is identically zero — the function MUST RAISE rather than emit
    an all-permissive pose-null cone. This is the fail-closed NO-FAKE guard."""

    torch = pytest.importorskip("torch")

    class _SeveringPoseNet:
        """A PoseNet whose preprocess detaches the input (severs the graph),
        exactly mimicking the upstream @torch.no_grad rgb_to_yuv6 failure."""

        def preprocess_input(self, x):
            return x.detach().reshape(x.shape[0], -1, x.shape[3], x.shape[4])[:, :12]

        def __call__(self, x):
            # produce a 'pose' head that does NOT depend on the (detached) input
            return {"pose": torch.zeros(1, 12, requires_grad=True)}

    pair = torch.zeros(1, 2, 8, 8, 3, requires_grad=False)
    with pytest.raises(Frame1JointSafeConeError):
        measure_posenet_frame1_jacobian(_SeveringPoseNet(), pair)


def test_ensure_pair_rejects_wrong_shape():
    torch = pytest.importorskip("torch")
    from tac.optimization.frame1_joint_safe_cone import _ensure_pair_btchwc

    with pytest.raises(Frame1JointSafeConeError):
        _ensure_pair_btchwc(torch.zeros(1, 3, 8, 8, 3))  # 3 frames, not 2
    with pytest.raises(Frame1JointSafeConeError):
        _ensure_pair_btchwc(torch.zeros(1, 2, 8, 8, 1))  # 1 channel, not 3


# ---------------------------------------------------------------------------
# REAL-SCORER NO-FAKE PROOF (gated on model + video presence)
# ---------------------------------------------------------------------------


def _have_real_assets() -> bool:
    return (
        (UPSTREAM / "models" / "segnet.safetensors").is_file()
        and (UPSTREAM / "models" / "posenet.safetensors").is_file()
        and (UPSTREAM / "videos" / "0.mkv").is_file()
    )


@pytest.fixture(scope="module")
def real_scorers_and_pair():
    pytest.importorskip("torch")
    if not _have_real_assets():
        pytest.skip("upstream models or video missing")
    import sys

    import torch

    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    from tac.data import decode_video
    from tac.scorer import load_differentiable_scorers

    pose, seg = load_differentiable_scorers(str(UPSTREAM), device="cpu")
    frames = decode_video(str(UPSTREAM / "videos" / "0.mkv"), target_h=384, target_w=512, max_frames=2)
    gt = np.stack([f.numpy() for f in frames[:2]], axis=0)
    pair = torch.from_numpy(gt[None]).float()
    return pose, seg, pair


def test_real_segnet_margin_is_nonconstant(real_scorers_and_pair):
    """SegNet frame1 margin from a REAL forward varies across pixels (boundaries
    have small margin) — NOT a constant. Slot RR NO-FAKE behavioral check."""

    _pose, seg, pair = real_scorers_and_pair
    margin, argmax = measure_segnet_frame1_margin(seg, pair)
    assert margin.shape == (384, 512)
    assert float(margin.std()) > 1e-6
    assert float(margin.min()) >= 0.0
    assert set(np.unique(argmax)).issubset(set(range(5)))


def test_real_segnet_boundary_slope_is_real_gradient(real_scorers_and_pair):
    """The SegNet margin slope is a REAL backward w.r.t. frame1 pixels (non-zero,
    varying) — proving the seg-safe sensitivity is measured, not stubbed."""

    _pose, seg, pair = real_scorers_and_pair
    slope = measure_segnet_frame1_boundary_slope(seg, pair)
    assert slope.shape[0] > 0 and slope.shape[1] > 0
    assert float(slope.max()) > 0.0
    assert float(slope.std()) > 0.0


def test_real_posenet_frame1_jacobian_is_real_and_sparse(real_scorers_and_pair):
    """The PoseNet frame1 Jacobian (differentiable YUV6 patched by
    load_differentiable_scorers) is non-zero with a meaningful pose-null subset
    — the fail-closed guard passes ONLY because the gradient is truly reachable."""

    pose, _seg, pair = real_scorers_and_pair
    jac = measure_posenet_frame1_jacobian(pose, pair)
    assert jac.shape == (384, 512)
    assert float(jac.max()) > 0.0
    null_frac = float((jac < 0.05 * jac.max()).mean())
    assert 0.0 < null_frac < 1.0


def test_real_cone_discriminates_inside_vs_outside(real_scorers_and_pair):
    """THE NO-FAKE FALSIFIABLE PROOF: perturb frame1 INSIDE the cone -> d_seg
    stable + small d_pose; perturb OUTSIDE (fragile at 2x) -> measurably larger
    d_seg AND d_pose. A cone that doesn't discriminate is FAKE."""

    import sys

    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    from modules import DistortionNet  # type: ignore[import-not-found]

    from tac.optimization.frame1_joint_safe_cone import compute_frame1_joint_safe_cone

    pose, seg, pair = real_scorers_and_pair
    cone = compute_frame1_joint_safe_cone(
        segnet=seg, posenet=pose, pair_btchwc_unit255=pair,
        config=Frame1ConeConfig(d_pose=3.4e-5),
    )
    dn = DistortionNet().eval()
    dn.load_state_dicts(
        str(UPSTREAM / "models" / "posenet.safetensors"),
        str(UPSTREAM / "models" / "segnet.safetensors"),
        "cpu",
    )
    val = validate_cone_behaviorally(distortion_net=dn, gt_pair_btchwc_unit255=pair, cone=cone)
    # Outside the cone moves the score substantially more than inside it.
    assert abs(val["outside_seg_delta"]) > abs(val["inside_seg_delta"])
    assert abs(val["outside_pose_delta"]) > abs(val["inside_pose_delta"])
    assert val["seg_discrimination_ratio"] >= 3.0
    assert val["pose_discrimination_ratio"] >= 3.0
    assert val["cone_discriminates"] is True


def test_real_cone_pose_binds_at_frontier(real_scorers_and_pair):
    """At the PR106 frontier operating point a majority of frame1 pixels are
    pose-bound (the pose budget is the binding constraint) — confirming the
    CLAUDE.md marginal-value flip (pose 2.71x SegNet at d_pose ~ 3.4e-5)."""

    pose, seg, pair = real_scorers_and_pair
    from tac.optimization.frame1_joint_safe_cone import compute_frame1_joint_safe_cone

    cone = compute_frame1_joint_safe_cone(
        segnet=seg, posenet=pose, pair_btchwc_unit255=pair,
        config=Frame1ConeConfig(d_pose=3.4e-5),
    )
    assert cone.summary["pose_binds_fraction"] > 0.5
    assert cone.summary["pose_ail_gain"] == pytest.approx(271.16, rel=1e-3)
