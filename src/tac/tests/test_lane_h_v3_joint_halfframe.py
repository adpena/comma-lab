"""Tests for Lane H-V3 (half-frame revival via JOINT warp-expansion training).

Five tests with magnitude anchors per Round 26 convention. Lane H-V3 design:
project_lane_h_v3_revival_design_20260428.

Lane H-V3 fixes the failure modes of Lane D V1/V2/V3 + Lane V V1/V2:
  * JOINT training from epoch 0 (NOT mid-train retrofit).
  * Train endpoint = inflate distribution (mask_half_sim_prob=1.0 NOT 0.5).
  * 288K dilated-h64 arch (NOT 88K DSConv → bypasses Lane V channel bug).
"""

from tac.profiles import PROFILES


def test_profile_registered():
    """Lane H-V3 must be discoverable by name in PROFILES (the registry).

    Without registration, every CLI / script invocation that passes
    `--profile h_v3_joint_halfframe` fails with KeyError before any GPU
    spend. This catches the "wrote profile but forgot to wire" class of bug.
    """
    assert "h_v3_joint_halfframe" in PROFILES, (
        "PROFILES missing h_v3_joint_halfframe — registration gap. "
        "Add `'h_v3_joint_halfframe': H_V3_JOINT_HALFFRAME` to the dict at "
        "the bottom of src/tac/profiles.py."
    )


def test_curriculum_schedule_ramps_0_to_1():
    """Curriculum must explicitly ramp from 0.0 → 1.0 (not 0.0 → 0.5).

    Lane D-V3 had end_value=0.5 — that's a HALF-half-frame curriculum, with
    a train/inflate distribution mismatch (inflate sees 100% warped masks,
    training endpoint sees 50%). Lane H-V3 fixes by setting end_value=1.0.

    Magnitude anchors:
      * start_value EXACTLY 0.0 (warmup distribution = full-frame baseline)
      * end_value EXACTLY 1.0 (endpoint = inflate distribution)
      * ramp_start_frac < ramp_end_frac (ramp is forward-going)
      * ramp window between 5%-15% of training (aggressive, not slow drift)
    """
    p = PROFILES["h_v3_joint_halfframe"]
    sched = p["mask_half_sim_prob_anneal"]
    assert sched["start_value"] == 0.0, (
        f"start_value={sched['start_value']} must be 0.0 (full-frame warmup)"
    )
    assert sched["end_value"] == 1.0, (
        f"end_value={sched['end_value']} must be 1.0 (inflate-distribution "
        f"endpoint). Lane D-V3 had 0.5 — that's the bug Lane H-V3 fixes."
    )
    assert sched["ramp_start_frac"] < sched["ramp_end_frac"], (
        f"ramp_start_frac={sched['ramp_start_frac']} must be < "
        f"ramp_end_frac={sched['ramp_end_frac']}"
    )
    assert 0.0 < sched["ramp_start_frac"] <= 0.10, (
        f"ramp_start_frac={sched['ramp_start_frac']} must be in (0, 0.10] "
        f"to leave a brief warmup phase"
    )
    assert 0.10 <= sched["ramp_end_frac"] <= 0.30, (
        f"ramp_end_frac={sched['ramp_end_frac']} must be in [0.10, 0.30] "
        f"to keep the curriculum aggressive (most epochs at endpoint)"
    )


def test_use_zoom_flow_true():
    """use_zoom_flow=True is required when mask_half_sim_prob > 0.

    Per train_renderer.py preflight at line ~1933:
        getattr(args, "mask_half_sim_prob", 0.0) > 0
        or getattr(args, "use_zoom_flow", False)

    Half-frame masks at inflate REQUIRE warp_inverse_masks() which only
    fires when use_zoom_flow=True. Without it, the inflate-side masks are
    identity-passed and PoseNet collapses.
    """
    p = PROFILES["h_v3_joint_halfframe"]
    assert p["use_zoom_flow"] is True, (
        f"use_zoom_flow={p['use_zoom_flow']} must be True for half-frame "
        f"inflate (warp_inverse_masks needs RadialZoomWarp)"
    )


def test_static_endpoint_matches_inflate_distribution():
    """Static mask_half_sim_prob must equal 1.0 (the inflate distribution).

    Lane D-V3 BUG-1: train endpoint=0.5 vs inflate=1.0 = 2× distribution
    mismatch (renderer trained on 50/50 mix but tested on 100% warped). This
    is the same bug class as Lane M-V2 BUG-1 (Check 42 — train_inference_parity).
    Lane H-V3 fixes by setting endpoint=1.0.
    """
    p = PROFILES["h_v3_joint_halfframe"]
    assert p["mask_half_sim_prob"] == 1.0, (
        f"mask_half_sim_prob={p['mask_half_sim_prob']} (static endpoint) "
        f"must be 1.0 to match inflate-time distribution. Lane D-V3 had 0.5 — "
        f"that's the train/inference mismatch BUG-1 class. See "
        f"feedback_check_42_train_inference_parity_20260428."
    )


def test_arch_inherits_lane_g_v3():
    """Arch must be 288K dilated-h64 (NOT 88K DSConv).

    Lane V V1/V2 used `use_dsconv=True, base_ch=24, mid_ch=32, motion_hidden=16`
    (88K params) and crashed with channel mismatch (input[1,1,...] vs weight[32,3,...]).
    Lane H-V3 inherits Lane G v3's arch (`use_dsconv=False, base_ch=36,
    mid_ch=60, motion_hidden=32`, ~288K params) which has a known-working
    pipeline at 1.05 contest-CUDA.

    Magnitude anchors:
      * base_ch=36 (Lane G v3 anchor; NOT 24 = Lane V)
      * mid_ch=60 (Lane G v3 anchor; NOT 32 = Lane V)
      * motion_hidden=32 (Lane G v3 anchor; NOT 16 = Lane V)
      * use_dsconv=False (Lane G v3 anchor; NOT True = Lane V)
      * depth=1 (single-scale U-Net like Lane G v3 / dilated-h64 baseline)
    """
    p = PROFILES["h_v3_joint_halfframe"]
    assert p["base_ch"] == 36, (
        f"base_ch={p['base_ch']} must be 36 (Lane G v3 anchor). "
        f"Lane V used 24 = 88K DSConv path which had channel bug."
    )
    assert p["mid_ch"] == 60, (
        f"mid_ch={p['mid_ch']} must be 60 (Lane G v3 anchor). "
        f"Lane V used 32 = 88K DSConv path."
    )
    assert p["motion_hidden"] == 32, (
        f"motion_hidden={p['motion_hidden']} must be 32 (Lane G v3 anchor). "
        f"Lane V used 16 = 88K DSConv path."
    )
    assert p["use_dsconv"] is False, (
        f"use_dsconv={p['use_dsconv']} must be False (Lane G v3 anchor). "
        f"Lane V used True (DSConv) and crashed at conv layer."
    )
    assert p["depth"] == 1, (
        f"depth={p['depth']} must be 1 (single-scale U-Net like Lane G v3)"
    )
