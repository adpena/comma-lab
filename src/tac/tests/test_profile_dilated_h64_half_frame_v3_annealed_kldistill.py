"""Lane D-V3 (annealed half-frame + KL distill weight fix) — profile +
schedule + KL-weight + LR-floor + Phase-0 instrumentation contract tests.

Lane D-V1 (DILATED_H64_HALF_FRAME) plateaued at ep ~700 in fp4_scorer with
mask_half_sim_prob=0.5 (static), kl_distill_weight=1.0 (pre-bug-fix value),
and V1 LRs (P2=3e-4 etc.) that cosine-decayed to ~3.3e-5 in the back-half
of P2 (eta_min=1e-6 starvation).

Lane D-V2 (in flight) tries CHOICE B = higher per-phase LR floor only
(single-variable A/B vs V1).

Lane D-V3 (this profile) STACKS V2's LR fix with two additional levers:
  1. ANNEALED mask_half_sim_prob 0.0 → 0.5 (Lane V-V2 paradigm, adapted for
     Lane D's 0.5 endpoint instead of Lane V's 1.0).
  2. KL DISTILL WEIGHT 1.0 → 0.002 (post-bug-fix value matching Lane V).

These tests pin every claim:

  1. Profile registered + canonical key match.
  2. experiment_type = 'renderer_training'.
  3. Annealing schedule present + correctly shaped (4 keys, all in [0,1])
     + canonical values (0.0 → 0.5 over 30%→70%).
  4. KL distill weight is 0.002 (NOT V1's 1.0).
  5. LR floor matches V2 (single-variable inheritance check).
  6. Inheritance from V1 (every V1 arch + Fridrich knob present + matches V1,
     except seed, mask_half_sim_prob_anneal, kl_distill_weight, phase LRs).
  7. Annealing helper math evaluates correctly at boundaries (warmup,
     ramp midpoint, endpoint).
  8. Training-loop wiring (current_mask_half_sim_prob computed before
     step loop; inner `random.random() < ...` reads the per-epoch value).
  9. Phase-0 instrumentation present (halfframe_branch_fires counter,
     halfframe_warp_diff_sum accumulator, JSONL telemetry keys).
 10. Preflight passes (eval_roundtrip=True, half-frame compat, etc.).
 11. Param count matches V1 dilated-h64 class (288K).
"""
from __future__ import annotations

from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
PROFILE_KEY = "dilated_h64_half_frame_v3_annealed_kldistill"


# ── 1. Profile registration ─────────────────────────────────────────────


def test_v3_profile_registered() -> None:
    """The V3 profile MUST be in PROFILES under its canonical key."""
    from tac.profiles import (
        PROFILES,
        DILATED_H64_HALF_FRAME_V3_ANNEALED_KLDISTILL,
    )

    assert PROFILE_KEY in PROFILES, (
        f"Lane D-V3 profile not registered under {PROFILE_KEY!r}"
    )
    assert (
        PROFILES[PROFILE_KEY]
        is DILATED_H64_HALF_FRAME_V3_ANNEALED_KLDISTILL
    )


def test_v3_profile_experiment_type() -> None:
    """experiment_type must be 'renderer_training' so train_renderer
    treats this as a renderer profile."""
    from tac.profiles import DILATED_H64_HALF_FRAME_V3_ANNEALED_KLDISTILL as p

    assert p.get("experiment_type") == "renderer_training"


# ── 2. Annealing schedule contract ──────────────────────────────────────


def test_v3_annealing_schedule_present_and_shaped() -> None:
    """The schedule dict has exactly the 4 expected keys, all in [0, 1]."""
    from tac.profiles import DILATED_H64_HALF_FRAME_V3_ANNEALED_KLDISTILL as p

    sched = p.get("mask_half_sim_prob_anneal")
    assert sched is not None, (
        "Lane D-V3 profile missing mask_half_sim_prob_anneal — annealing "
        "is one of V3's two new levers and must be declared explicitly"
    )
    assert set(sched.keys()) == {
        "start_value", "ramp_start_frac", "end_value", "ramp_end_frac",
    }
    for key, val in sched.items():
        assert isinstance(val, (int, float)), (
            f"sched[{key!r}]={val!r} must be a float, got {type(val).__name__}"
        )
        assert 0.0 <= float(val) <= 1.0, (
            f"sched[{key!r}]={val} must be in [0, 1]"
        )
    # Ramp must move forward in time.
    assert sched["ramp_start_frac"] <= sched["ramp_end_frac"]


def test_v3_annealing_schedule_canonical_values() -> None:
    """Pin the council-decided V3 schedule (0 → 0.5 over 30%→70%). Lane D
    endpoint is 0.5 (not Lane V's 1.0) because Lane D's RETROFIT premise
    targets the 0.5 mixed distribution."""
    from tac.profiles import DILATED_H64_HALF_FRAME_V3_ANNEALED_KLDISTILL as p

    sched = p["mask_half_sim_prob_anneal"]
    assert sched["start_value"] == 0.0
    assert sched["end_value"] == 0.5
    assert sched["ramp_start_frac"] == 0.30
    assert sched["ramp_end_frac"] == 0.70


def test_v3_static_mask_half_sim_prob_matches_anneal_endpoint() -> None:
    """The static mask_half_sim_prob is the END (post-ramp) value so that
    code paths reading the static value (e.g. preflight checks) see the
    final distribution, not the warmup."""
    from tac.profiles import DILATED_H64_HALF_FRAME_V3_ANNEALED_KLDISTILL as p

    assert p["mask_half_sim_prob"] == p["mask_half_sim_prob_anneal"]["end_value"]
    assert p["mask_half_sim_prob"] == 0.5


# ── 3. KL distill weight (post-bug-fix value) ───────────────────────────


def test_v3_kl_distill_weight_is_post_bug_fix_value() -> None:
    """V3 must use kl_distill_weight=0.002 (post-bug-fix). V1/V2 inherited
    1.0 from before the kl_distill_segnet_only reduction fix (losses.py:705).
    With weight=1.0 the post-fix KL contribution drowns the scorer signal."""
    from tac.profiles import DILATED_H64_HALF_FRAME_V3_ANNEALED_KLDISTILL as p

    assert p["kl_distill_weight"] == 0.002, (
        f"V3 kl_distill_weight should be 0.002 (post-fix); got {p['kl_distill_weight']}"
    )


def test_v3_kl_distill_temperature_unchanged() -> None:
    """KL distill temperature stays at 2.0 (matches Quantizr / Lane V / V1).
    Only the WEIGHT changes — temperature is the right value already."""
    from tac.profiles import DILATED_H64_HALF_FRAME_V3_ANNEALED_KLDISTILL as p

    assert p["kl_distill_temperature"] == 2.0


def test_v3_kl_distill_weight_matches_lane_v() -> None:
    """V3 kl_distill_weight must match Lane V's value — same post-fix
    convention. If they ever drift, an explicit acknowledgement is required."""
    from tac.profiles import (
        DILATED_H64_HALF_FRAME_V3_ANNEALED_KLDISTILL as v3,
        QUANTIZR_REPLICA_88K_HALFFRAME as lane_v,
    )

    assert v3["kl_distill_weight"] == lane_v["kl_distill_weight"], (
        f"V3 kl_distill_weight ({v3['kl_distill_weight']}) must match Lane V "
        f"({lane_v['kl_distill_weight']}) — same post-fix convention"
    )


# ── 4. LR floor (V2 inheritance) ────────────────────────────────────────


def test_v3_phase_lrs_match_v2_higher_floor() -> None:
    """V3 inherits V2's higher per-phase LR floor (CHOICE B)."""
    from tac.profiles import DILATED_H64_HALF_FRAME_V3_ANNEALED_KLDISTILL as p

    # Pin the V2 choice-B values explicitly.
    assert p["phase1_lr"] == 1e-3
    assert p["phase2_lr"] == 5e-4   # V1 had 3e-4 (1.67× raise)
    assert p["phase3_lr"] == 2e-4   # V1 had 1e-4 (2.0× raise)
    assert p["phase4_lr"] == 1e-4   # V1 had 5e-5 (2.0× raise)
    assert p["phase5_lr"] == 2e-5   # V1 had 1e-5 (2.0× raise)


def test_v3_lrs_strictly_above_v1() -> None:
    """V3 LRs must be >= V1 LRs (the V2 fix raises the floor; V3 inherits)."""
    from tac.profiles import (
        DILATED_H64_HALF_FRAME as v1,
        DILATED_H64_HALF_FRAME_V3_ANNEALED_KLDISTILL as v3,
    )
    for i in range(1, 6):
        key = f"phase{i}_lr"
        assert v3[key] >= v1[key], (
            f"V3 {key}={v3[key]} should be >= V1 {key}={v1[key]} "
            f"(V2 raises the floor; V3 inherits)"
        )


# ── 5. Inheritance from V1 (single-variable A/B traceability) ───────────


def test_v3_inherits_v1_arch_knobs() -> None:
    """Every architecture knob in V1 must be unchanged in V3 — V3 changes
    schedule + KL weight + LR floor only, not the architecture."""
    from tac.profiles import (
        DILATED_H64_HALF_FRAME as v1,
        DILATED_H64_HALF_FRAME_V3_ANNEALED_KLDISTILL as v3,
    )
    arch_keys = (
        "base_ch", "mid_ch", "motion_hidden", "embed_dim", "depth",
        "pose_dim", "use_dsconv", "padding_mode", "use_dilation",
        "use_zoom_flow", "variant",
    )
    for key in arch_keys:
        assert v3[key] == v1[key], (
            f"V3 changed arch key {key!r}: V1={v1[key]!r}, V3={v3[key]!r}"
        )


def test_v3_inherits_v1_fridrich_aux_loss_knobs() -> None:
    """Fridrich aux losses unchanged from V1 — direct A/B with V1/V2."""
    from tac.profiles import (
        DILATED_H64_HALF_FRAME as v1,
        DILATED_H64_HALF_FRAME_V3_ANNEALED_KLDISTILL as v3,
    )
    fridrich_keys = (
        "use_texture_loss", "texture_loss_weight",
        "use_linf_penalty", "linf_weight",
        "use_markov_loss", "markov_weight",
        "use_variance_noise", "variance_noise_weight",
        "variance_noise_base_std", "variance_noise_kernel",
        "variance_noise_mode",
        "use_uncertainty_loss", "uncertainty_loss_weight",
        "uncertainty_loss_floor",
    )
    for key in fridrich_keys:
        assert v3[key] == v1[key], (
            f"V3 changed Fridrich key {key!r}: V1={v1[key]!r}, V3={v3[key]!r}"
        )


def test_v3_inherits_v1_phase_epochs() -> None:
    """5-phase epoch counts must match V1 (only the LRs change, schedule
    shape is unchanged for direct A/B)."""
    from tac.profiles import (
        DILATED_H64_HALF_FRAME as v1,
        DILATED_H64_HALF_FRAME_V3_ANNEALED_KLDISTILL as v3,
    )
    for i in range(1, 6):
        ek = f"phase{i}_epochs"
        assert v3[ek] == v1[ek], (
            f"V3 changed {ek}: V1={v1[ek]}, V3={v3[ek]}"
        )
    assert sum(v3[f"phase{i}_epochs"] for i in range(1, 6)) == 1980


def test_v3_seed_differs_from_v1() -> None:
    """V3 must use a DIFFERENT seed from V1 so the runs explore different
    RNG basins (matches Lane V-V1 vs Lane V-V2 convention)."""
    from tac.profiles import (
        DILATED_H64_HALF_FRAME as v1,
        DILATED_H64_HALF_FRAME_V3_ANNEALED_KLDISTILL as v3,
    )
    assert v3["seed"] != v1["seed"], (
        f"V3 seed ({v3['seed']}) must differ from V1 seed ({v1['seed']})"
    )


# ── 6. Annealing helper math (boundary correctness) ─────────────────────


def test_v3_anneal_helper_warmup_phase() -> None:
    """For epoch_frac < 0.30, the helper returns start_value (= 0.0)."""
    from tac.experiments.train_renderer import mask_half_sim_prob_for_epoch
    from tac.profiles import DILATED_H64_HALF_FRAME_V3_ANNEALED_KLDISTILL as p

    sched = p["mask_half_sim_prob_anneal"]
    total = sum(p[f"phase{i}_epochs"] for i in range(1, 6))  # = 1980
    # epoch=0 → frac=0 < 0.30 → start_value
    assert mask_half_sim_prob_for_epoch(
        0, total_epochs=total, static_prob=p["mask_half_sim_prob"],
        schedule=sched,
    ) == 0.0
    # epoch=593 of 1980 → frac~0.2995 < 0.30 → start_value
    assert mask_half_sim_prob_for_epoch(
        593, total_epochs=total, static_prob=p["mask_half_sim_prob"],
        schedule=sched,
    ) == 0.0


def test_v3_anneal_helper_endpoint_phase() -> None:
    """For epoch_frac >= 0.70, the helper returns end_value (= 0.5)."""
    from tac.experiments.train_renderer import mask_half_sim_prob_for_epoch
    from tac.profiles import DILATED_H64_HALF_FRAME_V3_ANNEALED_KLDISTILL as p

    sched = p["mask_half_sim_prob_anneal"]
    total = sum(p[f"phase{i}_epochs"] for i in range(1, 6))  # = 1980
    # epoch=1386 of 1980 → frac=0.70 → end_value (boundary inclusive)
    assert mask_half_sim_prob_for_epoch(
        1386, total_epochs=total, static_prob=0.0, schedule=sched,
    ) == 0.5
    # epoch=1979 of 1980 → frac~0.9995 → end_value
    assert mask_half_sim_prob_for_epoch(
        1979, total_epochs=total, static_prob=0.0, schedule=sched,
    ) == 0.5


def test_v3_anneal_helper_ramp_midpoint() -> None:
    """At the midpoint of the ramp (epoch_frac=0.5), the helper returns
    halfway through the linear ramp 0.0 → 0.5 = 0.25."""
    from tac.experiments.train_renderer import mask_half_sim_prob_for_epoch
    from tac.profiles import DILATED_H64_HALF_FRAME_V3_ANNEALED_KLDISTILL as p

    sched = p["mask_half_sim_prob_anneal"]
    total = sum(p[f"phase{i}_epochs"] for i in range(1, 6))  # = 1980
    val = mask_half_sim_prob_for_epoch(
        990, total_epochs=total, static_prob=0.0, schedule=sched,
    )
    assert abs(val - 0.25) < 1e-6, (
        f"midpoint of ramp should be 0.25 (halfway 0->0.5), got {val}"
    )


def test_v3_anneal_helper_full_curve_monotone() -> None:
    """The schedule must be monotone non-decreasing (0 → 0.5)."""
    from tac.experiments.train_renderer import mask_half_sim_prob_for_epoch
    from tac.profiles import DILATED_H64_HALF_FRAME_V3_ANNEALED_KLDISTILL as p

    sched = p["mask_half_sim_prob_anneal"]
    total = sum(p[f"phase{i}_epochs"] for i in range(1, 6))  # = 1980
    prev = -1.0
    for epoch in range(0, total, 50):
        val = mask_half_sim_prob_for_epoch(
            epoch, total_epochs=total, static_prob=0.0, schedule=sched,
        )
        assert val >= prev - 1e-9, (
            f"schedule must be monotone non-decreasing; "
            f"epoch {epoch} val={val} < prev={prev}"
        )
        prev = val
    # End must reach end_value.
    final = mask_half_sim_prob_for_epoch(
        total - 1, total_epochs=total, static_prob=0.0, schedule=sched,
    )
    assert abs(final - sched["end_value"]) < 1e-6


# ── 7. Training-loop wiring (source-grep) ───────────────────────────────


def test_v3_training_loop_uses_per_epoch_prob_not_static() -> None:
    """Inner step loop's `random.random() < ...` MUST compare against
    `current_mask_half_sim_prob` (per-epoch), NOT `args.mask_half_sim_prob`
    (static). Catches a refactor that removes the annealing wiring."""
    src = (
        REPO / "src" / "tac" / "experiments" / "train_renderer.py"
    ).read_text()
    assert "current_mask_half_sim_prob" in src, (
        "train_renderer.py is missing the per-epoch warp prob assignment"
    )
    assert "random.random() < current_mask_half_sim_prob" in src, (
        "train_renderer.py inner loop is not reading "
        "current_mask_half_sim_prob — annealing has no effect"
    )


def test_v3_training_loop_calls_anneal_helper() -> None:
    """The per-epoch value must be computed via mask_half_sim_prob_for_epoch
    so the helper is the single source of truth."""
    src = (
        REPO / "src" / "tac" / "experiments" / "train_renderer.py"
    ).read_text()
    assert "mask_half_sim_prob_for_epoch(" in src, (
        "train_renderer.py is not calling mask_half_sim_prob_for_epoch — "
        "annealing math would not be applied"
    )


# ── 8. Phase-0 instrumentation contract ─────────────────────────────────


def test_v3_phase_0_instrumentation_counters_present() -> None:
    """train_renderer.py must initialise the half-frame branch counters
    OUTSIDE the inner step loop (so they accumulate across all batches in
    the epoch) — Lane D-V3 mechanism instrumentation."""
    src = (
        REPO / "src" / "tac" / "experiments" / "train_renderer.py"
    ).read_text()
    for token in (
        "halfframe_branch_fires = 0",
        "halfframe_warp_diff_sum = 0.0",
        "halfframe_warp_diff_count = 0",
    ):
        assert token in src, (
            f"train_renderer.py is missing Phase-0 instrumentation counter "
            f"{token!r} (Lane D-V3 mechanism verification)"
        )


def test_v3_phase_0_instrumentation_increments_inside_branch() -> None:
    """The half-frame branch must increment `halfframe_branch_fires` and
    accumulate the warp-diff stat — otherwise the per-epoch log lies."""
    src = (
        REPO / "src" / "tac" / "experiments" / "train_renderer.py"
    ).read_text()
    assert "halfframe_branch_fires += 1" in src, (
        "train_renderer.py is missing the halfframe_branch_fires += 1 "
        "increment inside the half-frame trigger block"
    )
    assert "halfframe_warp_diff_sum +=" in src, (
        "train_renderer.py is missing the halfframe_warp_diff_sum "
        "accumulator inside the half-frame trigger block"
    )


def test_v3_phase_0_instrumentation_logged_per_epoch() -> None:
    """The per-epoch [ep N/M] log line must include the half-frame metrics
    string (hf_fires, hf_warp_diff, hf_target_prob) when the branch fires."""
    src = (
        REPO / "src" / "tac" / "experiments" / "train_renderer.py"
    ).read_text()
    assert "hf_fires=" in src
    assert "hf_warp_diff=" in src
    assert "hf_target_prob=" in src


def test_v3_phase_0_instrumentation_in_telemetry_jsonl() -> None:
    """The JSONL telemetry must include the half-frame mechanism keys so
    post-hoc analysis can verify the branch fired AND produced non-trivial
    mask perturbations."""
    src = (
        REPO / "src" / "tac" / "experiments" / "train_renderer.py"
    ).read_text()
    for key in (
        '"halfframe_target_prob"',
        '"halfframe_branch_fires"',
        '"halfframe_warp_diff_mean"',
    ):
        assert key in src, (
            f"train_renderer.py JSONL telemetry is missing key {key} "
            f"(Lane D-V3 mechanism verification)"
        )


# ── 9. Preflight ────────────────────────────────────────────────────────


def test_v3_profile_passes_preflight() -> None:
    """The V3 profile must satisfy the same preflight invariants as every
    other renderer profile (eval_roundtrip=True, padding_mode valid, etc.)."""
    from tac.preflight import preflight_profiles

    violations = preflight_profiles(strict=False, verbose=False)
    ours = [v for v in violations if PROFILE_KEY in v]
    assert not ours, f"Lane D-V3 profile violates preflight: {ours}"


def test_v3_eval_roundtrip_required() -> None:
    """eval_roundtrip MUST be True (CLAUDE.md non-negotiable). Without it
    proxy-auth gap is 2-11x on PoseNet."""
    from tac.profiles import DILATED_H64_HALF_FRAME_V3_ANNEALED_KLDISTILL as p

    assert p["eval_roundtrip"] is True, (
        "Lane D-V3 must declare eval_roundtrip=True (CLAUDE.md non-negotiable)"
    )


def test_v3_use_zoom_flow_consistent_with_half_sim_prob() -> None:
    """When mask_half_sim_prob > 0, use_zoom_flow MUST be True (the
    half-frame distribution requires the warp_inverse_masks call which
    requires sim_zoom_warp; preflight check enforces this)."""
    from tac.profiles import DILATED_H64_HALF_FRAME_V3_ANNEALED_KLDISTILL as p

    assert p["mask_half_sim_prob"] > 0
    assert p["use_zoom_flow"] is True


# ── 10. Param count parity (inherits V1's dilated-h64 class) ────────────


def test_v3_param_count_matches_v1_class() -> None:
    """V3 inherits V1's arch knobs (base_ch=36, mid_ch=60, motion_hidden=32,
    use_dsconv=False, depth=1, embed_dim=6, pose_dim=6) so param count must
    land in the dilated-h64 class. We test it lands in the broad 200K-300K
    band (V1 is ~287K with use_zoom_flow=True minus 14K for the 4-channel
    motion output vs 6-channel)."""
    from tac.profiles import DILATED_H64_HALF_FRAME_V3_ANNEALED_KLDISTILL as p
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
    assert 200_000 <= total <= 320_000, (
        f"V3 param count {total} outside dilated-h64 class budget [200K, 320K]"
    )
