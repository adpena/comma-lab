# SPDX-License-Identifier: MIT
"""Lane V (Quantizr-replica 88K half-frame) — profile pinning tests.

Council 2026-04-27. Lane V is the BIGGEST single swing in the council's
strategy: 12h training, $4-5 cost, predicted band [0.50, 1.10]. Failure
modes are common when stacking many tricks at once (88K params + DSConv +
FiLM + KL distill + half-frame from epoch 0 + QAT). These tests pin
every claim the profile makes:

  1. Profile registered + canonical key match.
  2. Architecture matches Quantizr's 88K param target (within ±15K budget).
  3. Half-frame is JOINT from epoch 0 (mask_half_sim_prob=1.0, NOT 0.5
     like Lane D's failed retrofit).
  4. KL distill weight is the POST-FIX value 0.002 (NOT the pre-fix 1.0
     that was running 5000× over intended on every WILDE/SHIRAZ/DEN/Lane-D
     run before the 2026-04-27 reduction fix in losses.py).
  5. KL distill temperature is 2.0 (Quantizr's published recipe).
  6. FiLM enabled from epoch 0 (pose_dim=6).
  7. DSConv enabled (Quantizr's depthwise-separable trick).
  8. eval_roundtrip=True (NON-NEGOTIABLE per CLAUDE.md).
  9. posetto_noise_std=0.5 surfaced as profile metadata for pose-TTO stage.
 10. 5-phase QAT schedule sums to 3000 epochs (matches user spec total_epochs).
 11. Profile passes preflight (mask_half_sim_prob>0 implies use_zoom_flow=True).
 12. parse_args resolves every Lane V key from the profile (no dead resolvers).
 13. Deterministic seed=1234 pinned (different from Lane D's 42).

Cost paranoia: zero GPU dollars wasted on a misconfigured 12h run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]


# ── 1. Profile registration & canonical key match ──────────────────────


def test_quantizr_replica_profile_registered() -> None:
    """The profile MUST be in PROFILES dict so
    --profile quantizr_replica_88k_halfframe works. If this fails, the
    bootstrap script will SystemExit before launch."""
    from tac.profiles import PROFILES, QUANTIZR_REPLICA_88K_HALFFRAME

    assert "quantizr_replica_88k_halfframe" in PROFILES, (
        "QUANTIZR_REPLICA_88K_HALFFRAME not registered in PROFILES dict. "
        "scripts/remote_lane_v_quantizr_replica_88k_halfframe.sh expects "
        "--profile quantizr_replica_88k_halfframe to resolve."
    )
    assert (
        PROFILES["quantizr_replica_88k_halfframe"]
        is QUANTIZR_REPLICA_88K_HALFFRAME
    )


def test_quantizr_replica_experiment_type() -> None:
    """The profile must declare experiment_type='renderer_training' so
    train_renderer.py recognises it as a renderer profile (not a postfilter
    or distillation profile)."""
    from tac.profiles import QUANTIZR_REPLICA_88K_HALFFRAME

    assert (
        QUANTIZR_REPLICA_88K_HALFFRAME.get("experiment_type")
        == "renderer_training"
    )


# ── 2. Architecture: 88K param target ──────────────────────────────────


def test_quantizr_replica_arch_keys() -> None:
    """User spec: base_ch=24, hidden_ch=32 (mid_ch=32 in canonical naming),
    embed_dim=6, depth=1, pose_dim=6, use_dsconv=True. These are the exact
    knobs that determine the param count budget."""
    from tac.profiles import QUANTIZR_REPLICA_88K_HALFFRAME as p

    assert p["base_ch"] == 24, f"user spec base_ch=24, got {p['base_ch']}"
    # User spec: hidden_ch=32. Canonical name in codebase: mid_ch.
    assert p["mid_ch"] == 32, (
        f"user spec hidden_ch=32 (canonical mid_ch); got {p['mid_ch']}"
    )
    assert p["embed_dim"] == 6, f"user spec embed_dim=6, got {p['embed_dim']}"
    assert p["depth"] == 1, f"user spec depth=1, got {p['depth']}"
    assert p["pose_dim"] == 6, (
        f"user spec pose_dim=6 (FiLM from epoch 0), got {p['pose_dim']}"
    )
    assert p["use_dsconv"] is True, (
        f"user spec use_dsconv=True (Quantizr trick); got {p['use_dsconv']}"
    )


def test_quantizr_replica_param_count_in_88k_band() -> None:
    """The actual built model must land in the 88K Quantizr-class budget
    (80K-100K). If this fails the bootstrap pre-flight will reject the
    arch before any GPU spend, but we want the test gate to fire LOCALLY
    so a council member can catch a profile-knob drift without burning $0.05
    on a Vast.ai pre-flight."""
    from tac.profiles import QUANTIZR_REPLICA_88K_HALFFRAME as p
    from tac.renderer import build_renderer

    model = build_renderer(
        num_classes=5,
        embed_dim=p["embed_dim"],
        base_ch=p["base_ch"],
        mid_ch=p["mid_ch"],
        motion_hidden=p["motion_hidden"],
        depth=p["depth"],
        pose_dim=p["pose_dim"],
        use_dsconv=p["use_dsconv"],
        padding_mode=p["padding_mode"],
        use_dilation=p["use_dilation"],
        use_zoom_flow=p["use_zoom_flow"],
    )
    total = sum(pp.numel() for pp in model.parameters())
    assert 80_000 <= total <= 100_000, (
        f"param count {total} ({total/1000:.1f}K) outside Quantizr-class "
        f"budget [80K, 100K]. Council target ~88K."
    )


# ── 3. Half-frame: JOINT from epoch 0 (the Lane V bet) ─────────────────


def test_quantizr_replica_mask_half_sim_prob_is_one() -> None:
    """Lane V's whole bet: mask_half_sim_prob=1.0 (always-on) so the motion
    module is FORCED to learn the warp-expansion premise from epoch 0,
    never seeing the unwarped distribution. Lane D used 0.5 (mid-train
    retrofit) and FAILED (memory: feedback_half_frame_breaks_posenet,
    score 17.55). If someone bumps this back to 0.5 we collapse to Lane D."""
    from tac.profiles import QUANTIZR_REPLICA_88K_HALFFRAME as p

    assert p["mask_half_sim_prob"] == pytest.approx(1.0), (
        f"Lane V requires mask_half_sim_prob=1.0 (always-on warp expansion "
        f"from epoch 0). Got {p['mask_half_sim_prob']!r} — this is the "
        f"Lane V bet vs Lane D's 0.5 retrofit. Do NOT change without "
        f"council deliberation."
    )


def test_quantizr_replica_use_zoom_flow_required() -> None:
    """use_zoom_flow=True is REQUIRED by tac.preflight when
    mask_half_sim_prob>0 (preflight.py:4426). Without it, the training-side
    half-frame simulation runs but the renderer doesn't accept the flow
    signal — the simulation is dead weight (consumes compute, doesn't
    shift the trained distribution)."""
    from tac.profiles import QUANTIZR_REPLICA_88K_HALFFRAME as p

    assert p["use_zoom_flow"] is True, (
        f"mask_half_sim_prob>0 requires use_zoom_flow=True — preflight "
        f"will fail otherwise. Got {p['use_zoom_flow']!r}."
    )


# ── 4. KL distillation: POST-FIX weight (0.002), Quantizr T=2.0 ─────────


def test_quantizr_replica_kl_distill_weight_is_post_fix_value() -> None:
    """KL distill weight 0.002 reflects the POST-2026-04-27 reduction fix in
    losses.py:705 (kl_distill_segnet_only). Pre-fix, raw KL was being divided
    only by B (not B*H*W), making it ~5000× too large. Every WILDE/SHIRAZ/
    DEN/Lane-D run with weight=1.0 was implicitly running at 5000× intended.
    Post-fix raw KL ≈ 0.025 (after T² scaling); weight=0.002 makes KL
    contribution ~5e-5 ≈ 1% of scorer loss (~0.005). Reverting this to 1.0
    would resurrect the dead-amplification bug under the post-fix code."""
    from tac.profiles import QUANTIZR_REPLICA_88K_HALFFRAME as p

    assert p["kl_distill_weight"] == pytest.approx(0.002), (
        f"Lane V KL distill weight must be 0.002 (POST-FIX value). Got "
        f"{p['kl_distill_weight']!r}. Pre-fix value 1.0 implicitly ran at "
        f"5000× intended; do NOT use it under post-2026-04-27 losses.py."
    )


def test_quantizr_replica_kl_distill_temperature_is_quantizr_recipe() -> None:
    """T=2.0 is Quantizr's published softmax temperature for SegNet
    distillation. Same as DEN/SHIRAZ/Lane D so KL behavior is comparable
    across lanes."""
    from tac.profiles import QUANTIZR_REPLICA_88K_HALFFRAME as p

    assert p["kl_distill_temperature"] == pytest.approx(2.0), (
        f"Lane V KL distill T=2.0 (Quantizr recipe); got "
        f"{p['kl_distill_temperature']!r}"
    )


def test_quantizr_replica_kl_distill_math_sanity() -> None:
    """Post-fix sanity: raw KL ≈ T² × per-pixel-per-class mean ≈ 4 × 6.2e-3
    = 0.0248. With weight=0.002, KL contribution to total loss is ~5e-5.
    Compare to scorer loss ~0.005 → KL is ~1% of scorer (council-targeted).
    This test enforces the math contract documented in the profile header."""
    from tac.profiles import QUANTIZR_REPLICA_88K_HALFFRAME as p

    # Council math: with weight=0.002 and post-fix raw KL ≈ 0.025, the KL
    # contribution to total loss should be in the [1e-5, 1e-4] band.
    expected_raw_kl = 0.025  # T² × per-pixel-per-class mean (post-fix)
    contribution = p["kl_distill_weight"] * expected_raw_kl
    assert 1e-5 <= contribution <= 1e-4, (
        f"KL contribution {contribution} (= weight {p['kl_distill_weight']} "
        f"× raw KL ~{expected_raw_kl}) outside the council-targeted "
        f"[1e-5, 1e-4] band. Either the math drifted or the weight needs "
        f"recomputation."
    )


# ── 5. eval_roundtrip + posetto_noise_std ──────────────────────────────


def test_quantizr_replica_eval_roundtrip_true() -> None:
    """CLAUDE.md NON-NEGOTIABLE: every training path MUST use eval_roundtrip.
    Without it, proxy-auth gap is 2-11x on PoseNet."""
    from tac.profiles import QUANTIZR_REPLICA_88K_HALFFRAME as p

    assert p.get("eval_roundtrip") is True, (
        f"eval_roundtrip MUST be True per CLAUDE.md non-negotiable. "
        f"Got {p.get('eval_roundtrip')!r}."
    )


def test_quantizr_replica_posetto_noise_std() -> None:
    """User spec: posetto_noise_std=0.5. This is consumed by
    experiments/optimize_poses.py at pose-TTO stage (Stage 3 of the
    bootstrap script). Surfaced as profile metadata so the script can
    read it without a separate train_renderer resolver."""
    from tac.profiles import QUANTIZR_REPLICA_88K_HALFFRAME as p

    assert p.get("posetto_noise_std") == pytest.approx(0.5), (
        f"user spec posetto_noise_std=0.5; got {p.get('posetto_noise_std')!r}"
    )


# ── 6. 5-phase schedule sums to 3000 epochs ────────────────────────────


def test_quantizr_replica_5phase_total_epochs() -> None:
    """User spec: total_epochs=3000. The 5-phase split (anchor → finetune →
    joint → QAT → final) MUST sum to 3000. If someone bumps phase epochs the
    cost estimate must be updated in remote_lane_v_*.sh (currently quoted
    at 12h / $3.00 on 4090 + $1-2 for TTO and eval = $4-5 budget)."""
    from tac.profiles import QUANTIZR_REPLICA_88K_HALFFRAME as p

    total = sum(p[f"phase{i}_epochs"] for i in range(1, 6))
    assert total == 3000, (
        f"Lane V total epochs changed from 3000 to {total}. Update the cost "
        f"estimate in remote_lane_v_quantizr_replica_88k_halfframe.sh "
        f"(currently quoted at 12h / $3.00 on 4090) before promoting."
    )


def test_quantizr_replica_lr_main_phase_matches_user_spec() -> None:
    """User spec: lr=5e-4. We use this as phase2_lr (the main scorer-driven
    phase). Phase 1 anchor uses 2x for warmup; phases 4/5 ramp down for QAT."""
    from tac.profiles import QUANTIZR_REPLICA_88K_HALFFRAME as p

    assert p["phase2_lr"] == pytest.approx(5e-4), (
        f"user spec lr=5e-4 → phase2_lr; got {p['phase2_lr']!r}"
    )
    # Also keep top-level lr=5e-4 for compat with profile consumers that
    # read lr without phase awareness (a few research scripts do).
    assert p.get("lr") == pytest.approx(5e-4), (
        f"top-level lr should match user spec 5e-4; got {p.get('lr')!r}"
    )


def test_quantizr_replica_batch_size_matches_user_spec() -> None:
    """User spec: batch_size=8. Applied to all phases (vs Lane D's
    phase1=16, phase2=8, phase3=8 — Lane V has less VRAM headroom from
    use_zoom_flow always-on warp computation)."""
    from tac.profiles import QUANTIZR_REPLICA_88K_HALFFRAME as p

    for phase in (1, 2, 3):
        bs = p[f"phase{phase}_batch_size"]
        assert bs == 8, (
            f"user spec batch_size=8; phase{phase}_batch_size={bs}"
        )


# ── 7. Loss config: hinge SegNet, score weights ────────────────────────


def test_quantizr_replica_loss_config() -> None:
    """User spec: seg_loss_mode='hinge', seg_margin=0.5, seg_weight=100,
    pose_weight=10. Canonical names in codebase: segnet_loss_mode,
    hinge_margin, seg_weight, pose_weight."""
    from tac.profiles import QUANTIZR_REPLICA_88K_HALFFRAME as p

    # User spec seg_loss_mode → canonical segnet_loss_mode
    assert p["segnet_loss_mode"] == "hinge", (
        f"user spec seg_loss_mode='hinge' (canonical segnet_loss_mode); "
        f"got {p['segnet_loss_mode']!r}"
    )
    # User spec seg_margin → canonical hinge_margin
    assert p["hinge_margin"] == pytest.approx(0.5), (
        f"user spec seg_margin=0.5 (canonical hinge_margin); "
        f"got {p['hinge_margin']!r}"
    )
    assert p["seg_weight"] == pytest.approx(100.0)
    assert p["pose_weight"] == pytest.approx(10.0)


# ── 8. Preflight gate ──────────────────────────────────────────────────


def test_quantizr_replica_passes_preflight() -> None:
    """The profile must pass tac.preflight.preflight_profiles — specifically
    the rule that mask_half_sim_prob>0 requires use_zoom_flow=True
    (preflight.py:4415-4434)."""
    from tac.preflight import preflight_profiles

    violations = preflight_profiles(strict=False, verbose=False)
    ours = [v for v in violations if "quantizr_replica_88k_halfframe" in v]
    assert not ours, (
        f"QUANTIZR_REPLICA_88K_HALFFRAME violates preflight rules: {ours}"
    )


# ── 9. parse_args resolution (no dead resolvers) ───────────────────────


def test_parse_args_resolves_lane_v_keys_from_profile(monkeypatch) -> None:
    """train_renderer.parse_args must surface every Lane V key from the
    profile. Same dead-resolver class as the pose_dim bug (memory:
    feedback_pose_dim_dead_resolver) — if any key is silently dropped, the
    training run is invalid before it starts."""
    train_renderer = pytest.importorskip("tac.experiments.train_renderer")

    argv = [
        "train_renderer.py",
        "--profile", "quantizr_replica_88k_halfframe",
        "--tag", "_unit_test_lane_v",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    args = train_renderer.parse_args(argv[1:])

    # Architecture keys
    assert args.base_ch == 24
    assert args.mid_ch == 32
    assert args.motion_hidden == 16
    assert args.embed_dim == 6
    assert args.depth == 1
    assert args.pose_dim == 6
    assert args.use_dsconv is True
    assert args.use_zoom_flow is True
    # Half-frame
    assert args.mask_half_sim_prob == pytest.approx(1.0)
    # KL distill
    assert args.kl_distill_weight == pytest.approx(0.002)
    assert args.kl_distill_temperature == pytest.approx(2.0)
    # Fridrich aux losses
    assert args.use_texture_loss is True
    assert args.texture_loss_weight == pytest.approx(0.5)
    assert args.use_linf_penalty is True
    assert args.use_markov_loss is True
    assert args.use_uncertainty_loss is True
    assert args.uncertainty_loss_floor == pytest.approx(0.1)


# ── 10. Determinism + seed isolation ───────────────────────────────────


def test_quantizr_replica_deterministic_pinned() -> None:
    """Lane V MUST pin seed + deterministic for bit-exact retrain across
    runs (CLAUDE.md canonical pipeline standard)."""
    from tac.profiles import QUANTIZR_REPLICA_88K_HALFFRAME as p

    assert p.get("seed") == 1234, (
        f"user spec seed=1234 (different from Lane D's 42 so the two "
        f"from-scratch rebuilds explore different RNG basins); got "
        f"{p.get('seed')!r}"
    )
    assert p.get("deterministic") is True


def test_quantizr_replica_seed_differs_from_lane_d() -> None:
    """Lane V uses seed=1234; Lane D uses seed=42. Different seeds let the
    two from-scratch rebuilds explore different RNG basins (council:
    reduces correlated failure modes)."""
    from tac.profiles import (
        DILATED_H64_HALF_FRAME,
        QUANTIZR_REPLICA_88K_HALFFRAME,
    )

    assert (
        QUANTIZR_REPLICA_88K_HALFFRAME["seed"]
        != DILATED_H64_HALF_FRAME["seed"]
    ), (
        "Lane V and Lane D should use different seeds to explore "
        "different RNG basins"
    )


# ── 11. Quantizr-trick stack completeness ──────────────────────────────


def test_quantizr_replica_full_trick_stack() -> None:
    """Lane V composes ALL Quantizr tricks. If any of these is missing,
    the Lane V premise (every Quantizr feature stacked) is invalid."""
    from tac.profiles import QUANTIZR_REPLICA_88K_HALFFRAME as p

    # Quantizr signature: depthwise-separable + FiLM + KL distill T=2.0
    assert p["use_dsconv"] is True
    assert p["pose_dim"] == 6  # FiLM
    assert p["kl_distill_weight"] > 0
    assert p["kl_distill_temperature"] == pytest.approx(2.0)
    # 5-stage QAT pipeline
    for phase in range(1, 6):
        assert f"phase{phase}_epochs" in p, (
            f"5-stage QAT requires phase{phase}_epochs"
        )
    # FP4 export (the 5-stage's terminal stage)
    assert p["fp4_codebook"] == "residual", (
        "Lane V uses RESIDUAL FP4 codebook (our advantage over Quantizr's "
        "vanilla 5-stage)"
    )
    assert p["fp4_robust_scale"] is True
    assert p["fp4_stochastic"] is True
