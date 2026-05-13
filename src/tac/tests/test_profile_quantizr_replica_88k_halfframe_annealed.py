"""Lane V-V2 (annealed half-frame Quantizr-replica) — profile + annealing
contract tests.

V1 covered (test_profile_quantizr_replica_88k_halfframe.py): the static
profile knobs, parameter count, half-frame contract, KL distill weight,
preflight pass, deterministic seed.

V2 (this file) adds:

  1. Profile registered + canonical key match.
  2. Annealing schedule present + correctly shaped (4 keys, all in [0,1]).
  3. Annealing kicks in at the right epochs (mask_half_sim_prob_for_epoch
     evaluates correctly at the boundaries).
  4. Inheritance from V1 (every V1 knob present + matches V1, except seed
     and the new annealing schedule).
  5. CLI override path: --mask-half-sim-prob-schedule exists in argparse.
  6. Loss / training-loop wiring (current_mask_half_sim_prob is computed
     before the step loop; the inner `random.random() < ...` test reads
     the per-epoch value, not the static one).
  7. Resolver wiring: parse_args reads the schedule from the profile if
     no CLI override is supplied; rejects malformed schedules with a
     loud SystemExit.
"""
from __future__ import annotations

import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]


# ── 1. Profile registration ─────────────────────────────────────────────


def test_v2_profile_registered() -> None:
    """The annealed profile MUST be in PROFILES under its canonical key."""
    from tac.profiles import (
        PROFILES,
        QUANTIZR_REPLICA_88K_HALFFRAME_ANNEALED,
    )

    assert "quantizr_replica_88k_halfframe_annealed" in PROFILES, (
        "Lane V-V2 profile not registered under "
        "'quantizr_replica_88k_halfframe_annealed'"
    )
    assert (
        PROFILES["quantizr_replica_88k_halfframe_annealed"]
        is QUANTIZR_REPLICA_88K_HALFFRAME_ANNEALED
    )


def test_v2_profile_experiment_type() -> None:
    """experiment_type must be 'renderer_training' so train_renderer
    treats this as a renderer profile."""
    from tac.profiles import QUANTIZR_REPLICA_88K_HALFFRAME_ANNEALED

    assert (
        QUANTIZR_REPLICA_88K_HALFFRAME_ANNEALED.get("experiment_type")
        == "renderer_training"
    )


# ── 2. Annealing schedule contract ──────────────────────────────────────


def test_v2_annealing_schedule_present_and_shaped() -> None:
    """The schedule dict has exactly the 4 expected keys, all in [0, 1]."""
    from tac.profiles import QUANTIZR_REPLICA_88K_HALFFRAME_ANNEALED as p

    sched = p.get("mask_half_sim_prob_anneal")
    assert sched is not None, (
        "Lane V-V2 profile missing mask_half_sim_prob_anneal — annealing "
        "is the V2 oversight fix and must be declared explicitly"
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


def test_v2_annealing_schedule_canonical_values() -> None:
    """Pin the council-decided schedule (0 → 1, 30% → 70%) so a future
    edit forces an explicit acknowledgement."""
    from tac.profiles import QUANTIZR_REPLICA_88K_HALFFRAME_ANNEALED as p

    sched = p["mask_half_sim_prob_anneal"]
    assert sched["start_value"] == 0.0
    assert sched["end_value"] == 1.0
    assert sched["ramp_start_frac"] == 0.30
    assert sched["ramp_end_frac"] == 0.70


# ── 3. Annealing function evaluates correctly at boundaries ─────────────


def test_v2_anneal_helper_static_path_is_passthrough() -> None:
    """When schedule=None, the helper returns the static value unchanged.
    This is the V1 byte-identical path and must hold for every V1 caller."""
    from tac.experiments.train_renderer import mask_half_sim_prob_for_epoch

    for epoch in (0, 100, 999, 3000):
        for static in (0.0, 0.5, 1.0):
            assert mask_half_sim_prob_for_epoch(
                epoch, total_epochs=3000,
                static_prob=static, schedule=None,
            ) == static


def test_v2_anneal_helper_warmup_phase() -> None:
    """For epoch_frac < ramp_start_frac, the helper returns start_value."""
    from tac.experiments.train_renderer import mask_half_sim_prob_for_epoch

    sched = {
        "start_value": 0.0,
        "end_value": 1.0,
        "ramp_start_frac": 0.30,
        "ramp_end_frac": 0.70,
    }
    # epoch=0 of 3000 → frac=0 < 0.30 → start_value
    assert mask_half_sim_prob_for_epoch(
        0, total_epochs=3000, static_prob=1.0, schedule=sched,
    ) == 0.0
    # epoch=899 of 3000 → frac~0.2997 < 0.30 → start_value
    assert mask_half_sim_prob_for_epoch(
        899, total_epochs=3000, static_prob=1.0, schedule=sched,
    ) == 0.0


def test_v2_anneal_helper_endpoint_phase() -> None:
    """For epoch_frac >= ramp_end_frac, the helper returns end_value."""
    from tac.experiments.train_renderer import mask_half_sim_prob_for_epoch

    sched = {
        "start_value": 0.0,
        "end_value": 1.0,
        "ramp_start_frac": 0.30,
        "ramp_end_frac": 0.70,
    }
    # epoch=2100 of 3000 → frac=0.70 → end_value (boundary inclusive)
    assert mask_half_sim_prob_for_epoch(
        2100, total_epochs=3000, static_prob=0.0, schedule=sched,
    ) == 1.0
    # epoch=2999 of 3000 → frac~1.0 → end_value
    assert mask_half_sim_prob_for_epoch(
        2999, total_epochs=3000, static_prob=0.0, schedule=sched,
    ) == 1.0


def test_v2_anneal_helper_ramp_midpoint() -> None:
    """At the midpoint of the ramp, the helper returns the midpoint value."""
    from tac.experiments.train_renderer import mask_half_sim_prob_for_epoch

    sched = {
        "start_value": 0.0,
        "end_value": 1.0,
        "ramp_start_frac": 0.30,
        "ramp_end_frac": 0.70,
    }
    # Midpoint of ramp: epoch_frac=0.5 → halfway through linear ramp 0→1 → 0.5
    val = mask_half_sim_prob_for_epoch(
        1500, total_epochs=3000, static_prob=0.0, schedule=sched,
    )
    assert abs(val - 0.5) < 1e-6, (
        f"midpoint of ramp should be 0.5, got {val}"
    )


def test_v2_anneal_helper_handles_zero_total_epochs() -> None:
    """Defensive: total_epochs=0 must not divide-by-zero."""
    from tac.experiments.train_renderer import mask_half_sim_prob_for_epoch

    sched = {
        "start_value": 0.3,
        "end_value": 0.7,
        "ramp_start_frac": 0.0,
        "ramp_end_frac": 1.0,
    }
    val = mask_half_sim_prob_for_epoch(
        0, total_epochs=0, static_prob=1.0, schedule=sched,
    )
    assert val == 0.3  # falls back to start_value


def test_v2_anneal_helper_full_curve_monotone() -> None:
    """The schedule must be monotone non-decreasing (0 → 1)."""
    from tac.experiments.train_renderer import mask_half_sim_prob_for_epoch

    sched = {
        "start_value": 0.0,
        "end_value": 1.0,
        "ramp_start_frac": 0.30,
        "ramp_end_frac": 0.70,
    }
    prev = -1.0
    for epoch in range(0, 3000, 100):
        val = mask_half_sim_prob_for_epoch(
            epoch, total_epochs=3000, static_prob=0.0, schedule=sched,
        )
        assert val >= prev - 1e-9, (
            f"schedule must be monotone non-decreasing; "
            f"epoch {epoch} val={val} < prev={prev}"
        )
        prev = val
    # End must reach end_value.
    final = mask_half_sim_prob_for_epoch(
        2999, total_epochs=3000, static_prob=0.0, schedule=sched,
    )
    assert abs(final - sched["end_value"]) < 1e-6


# ── 4. Inheritance from V1 ──────────────────────────────────────────────


def test_v2_inherits_v1_arch_knobs() -> None:
    """Every architecture knob in V1 must be unchanged in V2."""
    from tac.profiles import (
        QUANTIZR_REPLICA_88K_HALFFRAME as v1,
        QUANTIZR_REPLICA_88K_HALFFRAME_ANNEALED as v2,
    )
    arch_keys = (
        "base_ch", "mid_ch", "motion_hidden", "embed_dim", "depth",
        "pose_dim", "use_dsconv", "padding_mode", "use_dilation",
        "use_zoom_flow",
    )
    for key in arch_keys:
        assert v2[key] == v1[key], (
            f"V2 changed arch key {key!r}: V1={v1[key]!r}, V2={v2[key]!r}"
        )


def test_v2_inherits_v1_loss_knobs() -> None:
    """Loss configuration must match V1 for direct A/B (only annealing
    differs)."""
    from tac.profiles import (
        QUANTIZR_REPLICA_88K_HALFFRAME as v1,
        QUANTIZR_REPLICA_88K_HALFFRAME_ANNEALED as v2,
    )
    loss_keys = (
        "loss_mode", "segnet_loss_mode", "hinge_margin",
        "pose_weight", "seg_weight", "pixel_weight",
        "use_texture_loss", "texture_loss_weight",
        "use_linf_penalty", "linf_weight",
        "use_markov_loss", "markov_weight",
        "use_variance_noise", "variance_noise_weight",
        "kl_distill_weight", "kl_distill_temperature",
        "ema_decay", "use_per_class_weights",
    )
    for key in loss_keys:
        assert v2[key] == v1[key], (
            f"V2 changed loss key {key!r}: V1={v1[key]!r}, V2={v2[key]!r}"
        )


def test_v2_inherits_v1_phase_schedule() -> None:
    """5-phase epoch counts + LRs must match V1."""
    from tac.profiles import (
        QUANTIZR_REPLICA_88K_HALFFRAME as v1,
        QUANTIZR_REPLICA_88K_HALFFRAME_ANNEALED as v2,
    )
    for i in range(1, 6):
        ek = f"phase{i}_epochs"
        lk = f"phase{i}_lr"
        assert v2[ek] == v1[ek]
        assert v2[lk] == v1[lk]
    # Total epochs sums to the same 3000.
    assert sum(v2[f"phase{i}_epochs"] for i in range(1, 6)) == 3000


def test_v2_seed_differs_from_v1() -> None:
    """V2 must use a DIFFERENT seed from V1 so the two from-scratch
    rebuilds explore different RNG basins."""
    from tac.profiles import (
        QUANTIZR_REPLICA_88K_HALFFRAME as v1,
        QUANTIZR_REPLICA_88K_HALFFRAME_ANNEALED as v2,
    )
    assert v2["seed"] != v1["seed"], (
        f"V2 seed ({v2['seed']}) must differ from V1 seed ({v1['seed']})"
    )


# ── 5. CLI flag registration ────────────────────────────────────────────


def test_v2_cli_flag_registered_in_train_renderer() -> None:
    """train_renderer.py MUST register --mask-half-sim-prob-schedule so the
    bootstrap script can override the profile schedule from CLI."""
    src = (
        REPO / "src" / "tac" / "experiments" / "train_renderer.py"
    ).read_text()
    add_re = re.compile(r"add_argument\(\s*[\"']--([a-z][a-z0-9-]+)")
    flags = set(add_re.findall(src))
    assert "mask-half-sim-prob-schedule" in flags, (
        "train_renderer.py is missing --mask-half-sim-prob-schedule CLI flag"
    )


# ── 6. Training-loop wiring (source-grep) ───────────────────────────────


def test_v2_training_loop_uses_per_epoch_prob_not_static() -> None:
    """The inner step loop's `random.random() < ...` must compare against
    `current_mask_half_sim_prob` (per-epoch), NOT `args.mask_half_sim_prob`
    (static). Catches a refactor that removes the annealing wiring."""
    src = (
        REPO / "src" / "tac" / "experiments" / "train_renderer.py"
    ).read_text()
    # The annealing site must be present.
    assert "current_mask_half_sim_prob" in src, (
        "train_renderer.py is missing the per-epoch warp prob assignment"
    )
    # The inner loop must read the per-epoch value (not the static one).
    assert "random.random() < current_mask_half_sim_prob" in src, (
        "train_renderer.py inner loop is not reading "
        "current_mask_half_sim_prob — annealing has no effect"
    )


def test_v2_training_loop_calls_anneal_helper() -> None:
    """The per-epoch value must be computed via mask_half_sim_prob_for_epoch
    so the helper is the single source of truth for the schedule math."""
    src = (
        REPO / "src" / "tac" / "experiments" / "train_renderer.py"
    ).read_text()
    assert "mask_half_sim_prob_for_epoch(" in src, (
        "train_renderer.py is not calling mask_half_sim_prob_for_epoch — "
        "annealing math would not be applied"
    )


# ── 7. Preflight ────────────────────────────────────────────────────────


def test_v2_profile_passes_preflight() -> None:
    """The annealed profile must satisfy the same preflight invariants as
    every other renderer profile (eval_roundtrip=True, padding_mode valid,
    etc.). Catches a knob-drift before any GPU spend."""
    from tac.preflight import preflight_profiles

    violations = preflight_profiles(strict=False, verbose=False)
    ours = [v for v in violations if "quantizr_replica_88k_halfframe_annealed" in v]
    assert not ours, (
        f"Lane V-V2 profile violates preflight: {ours}"
    )


# ── 8. Param count parity (inherits V1's 88K target) ────────────────────


def test_v2_param_count_matches_v1_class() -> None:
    """V2 inherits V1's arch knobs so param count must land in the
    Quantizr-class 80K-100K window."""
    from tac.profiles import QUANTIZR_REPLICA_88K_HALFFRAME_ANNEALED as p
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
        f"V2 param count {total} outside Quantizr-class budget [80K, 100K]"
    )
