# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the BIND-ALL integrated A/B launcher
(``experiments/launch_bind_all_taper_ab.py``).

The load-bearing claims (the ones that, if wrong, silently mis-bind the production
run — every one is verified by inspecting the value threaded into the ACTUAL
``TorchVehicleConfig`` the run reads, NOT a constant or a docstring; Slot EEE Class 2):

1. The new P1a flags (``--kd-warm-start-dir`` / ``--kd-warm-epochs`` / ``--kd-warm-lr``
   / ``--(no-)kd-warm-train-latents`` / ``--pose-film-rgb0-pose-trainable``) each set
   their matching ``TorchVehicleConfig`` field — a misnamed field would raise at
   ``_build_torch_vehicle_config`` (the live Config build is exercised, not mocked).
2. The two arms are IDENTICAL in every Config field EXCEPT ``taper_channels``
   (arm_a = vendored ``[20,20,20,15,11,10,10]``, arm_b = solved ``[22,16,15,14,15,14,10]``).
3. The full bind-all production combo BUILDS a valid Config (passes every
   ``__post_init__`` guard: taper + FiLM-v2 + trunk-stopgrad + rgb_0-pose-trainable +
   KD-warm + rate-attack co-exist) — the structural proof that the composed config is legal.
4. ``--print-plan`` is a $0 dry-run (returns 0, NO training, NO scorer load) and prints
   the resolved config for the arm.
5. The OOMPH seg lever overlays the REFINEMENT stages (4-7) only; the coarse stages stay
   vendored CE.
"""

from __future__ import annotations

import dataclasses
import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import pytest

import experiments.launch_bind_all_taper_ab as launcher

_KD_DIR = "experiments/results/torch_vehicle_full_mps_basin_bc20_n600/best"

# The canonical bind-all production flag set (arm-agnostic tail), shared by the tests so a
# drift in the production contract is caught in one place.
_BIND_ALL_TAIL = [
    "--kd-warm-start-dir", _KD_DIR,
    "--out-dir", "/tmp/does_not_run",
    "--pose-film-v2", "--pose-film-trunk-stopgrad", "--pose-film-rgb0-pose-trainable",
    "--train-device", "mps", "--split-by-head", "--pose-grad-every-k", "1", "--rate-attack",
]


def _parse(argv: list[str]):
    return launcher._build_arg_parser().parse_args(argv)


def _cfg_for(arm: str, extra: list[str] | None = None):
    args = _parse(["--arm", arm, *_BIND_ALL_TAIL, *(extra or [])])
    return launcher._build_torch_vehicle_config(args, Path("/tmp/does_not_run"))


# --- 1. the new P1a flags map to the right Config fields ------------------------------


def test_kd_warm_start_dir_flag_sets_config_field() -> None:
    cfg = _cfg_for("arm_b")
    assert cfg.kd_warm_start_dir == Path(_KD_DIR)


def test_kd_warm_epochs_flag_sets_config_field() -> None:
    cfg = _cfg_for("arm_b", ["--kd-warm-epochs", "123"])
    assert cfg.kd_warm_epochs == 123


def test_kd_warm_lr_flag_sets_config_field() -> None:
    cfg = _cfg_for("arm_b", ["--kd-warm-lr", "0.0007"])
    assert cfg.kd_warm_lr == pytest.approx(0.0007)


def test_kd_warm_train_latents_default_true_and_negation_false() -> None:
    assert _cfg_for("arm_b").kd_warm_train_latents is True
    assert _cfg_for("arm_b", ["--no-kd-warm-train-latents"]).kd_warm_train_latents is False


def test_pose_film_rgb0_pose_trainable_flag_sets_config_field() -> None:
    # ON in the bind-all tail.
    assert _cfg_for("arm_b").pose_film_rgb0_pose_trainable is True


def test_pose_film_v2_sets_enabled_and_version2() -> None:
    cfg = _cfg_for("arm_b")
    assert cfg.pose_film_enabled is True
    assert cfg.pose_film_version == 2


def test_pose_film_trunk_stopgrad_flag_sets_config_field() -> None:
    assert _cfg_for("arm_b").pose_film_trunk_stopgrad is True


def test_pose_grad_every_k_one_is_pose_every_epoch() -> None:
    assert _cfg_for("arm_b").pose_grad_every_k == 1


def test_rate_attack_flag_enables_lever4_export() -> None:
    assert _cfg_for("arm_b").lever4_variable_level_export_enabled is True
    # default OFF when the flag is omitted (build a minimal legal config to check).
    args = _parse(["--arm", "arm_b", "--out-dir", "/tmp/does_not_run"])
    cfg = launcher._build_torch_vehicle_config(args, Path("/tmp/does_not_run"))
    assert cfg.lever4_variable_level_export_enabled is False


def test_ema_warmup_default_on_and_negation_off() -> None:
    assert _cfg_for("arm_b").ema_warmup is True
    assert _cfg_for("arm_b", ["--no-ema-warmup"]).ema_warmup is False


# --- 2. arm_a vs arm_b differ ONLY in taper_channels ---------------------------------


def test_arm_a_taper_is_vendored() -> None:
    assert _cfg_for("arm_a").taper_channels == [20, 20, 20, 15, 11, 10, 10]


def test_arm_b_taper_is_solved() -> None:
    assert _cfg_for("arm_b").taper_channels == [22, 16, 15, 14, 15, 14, 10]


def test_arms_differ_only_in_taper_channels() -> None:
    """The two production arms must be config-identical EXCEPT the taper — the whole
    point of the A/B (isolate the channel SCHEDULE). Diff the full Config field dicts."""
    a = dataclasses.asdict(_cfg_for("arm_a"))
    b = dataclasses.asdict(_cfg_for("arm_b"))
    differing = {k for k in a if a[k] != b[k]}
    assert differing == {"taper_channels"}, (
        f"arms must differ ONLY in taper_channels; also differ in: "
        f"{differing - {'taper_channels'}}"
    )


# --- 3. the full bind-all combo is a LEGAL config ------------------------------------


def test_full_bind_all_combo_builds_valid_config_both_arms() -> None:
    """The composed production config (taper + FiLM-v2 + trunk-stopgrad + rgb_0-pose-
    trainable + KD-warm + rate-attack) passes every ``__post_init__`` guard for BOTH
    arms. If any guard refused the combo this raises ValueError."""
    for arm in ("arm_a", "arm_b"):
        cfg = _cfg_for(arm)
        # spot-check the bound ingredients all survived the build.
        assert cfg.kd_warm_start_dir is not None
        assert cfg.pose_film_enabled and cfg.pose_film_version == 2
        assert cfg.pose_film_trunk_stopgrad and cfg.pose_film_rgb0_pose_trainable
        assert cfg.lever4_variable_level_export_enabled
        assert cfg.taper_channels is not None and len(cfg.taper_channels) == 7


# --- 4. --print-plan is a $0 dry-run -------------------------------------------------


def test_print_plan_is_zero_cost_dry_run_returns_zero() -> None:
    """``--print-plan`` returns 0 WITHOUT loading the scorer or training (it never
    reaches the --go gate / RealScorerContext). The resolved config is printed."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = launcher.main([
            "--arm", "arm_b", *_BIND_ALL_TAIL, "--total-epoch-budget", "80", "--print-plan",
        ])
    assert rc == 0
    out = json.loads(buf.getvalue())
    assert out["arm"] == "arm_b"
    assert out["taper_channels"] == [22, 16, 15, 14, 15, 14, 10]
    assert out["kd_warm_start_dir"] == _KD_DIR
    assert out["pose_film_trunk_stopgrad"] is True
    assert out["pose_film_rgb0_pose_trainable"] is True
    assert out["lever4_variable_level_export_enabled"] is True
    assert "[advisory]" in out["authority"].lower() or "advisory" in out["authority"].lower()
    assert "stages" in out and len(out["stages"]) == 8


def test_print_plan_arm_a_taper_is_vendored() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = launcher.main(["--arm", "arm_a", *_BIND_ALL_TAIL,
                            "--total-epoch-budget", "80", "--print-plan"])
    assert rc == 0
    assert json.loads(buf.getvalue())["taper_channels"] == [20, 20, 20, 15, 11, 10, 10]


def test_resolved_config_dict_matches_built_config_for_key_fields() -> None:
    """The --print-plan resolved dict is the SAME source the run reads (the dict and the
    live Config must agree on every load-bearing field — no drift between preview and run)."""
    args = _parse(["--arm", "arm_b", *_BIND_ALL_TAIL])
    d = launcher._resolve_config_dict(args)
    cfg = launcher._build_torch_vehicle_config(args, Path("/tmp/does_not_run"))
    assert d["taper_channels"] == cfg.taper_channels
    assert d["kd_warm_start_dir"] == str(cfg.kd_warm_start_dir)
    assert d["kd_warm_epochs"] == cfg.kd_warm_epochs
    assert d["pose_film_trunk_stopgrad"] == cfg.pose_film_trunk_stopgrad
    assert d["pose_film_rgb0_pose_trainable"] == cfg.pose_film_rgb0_pose_trainable
    assert d["lever4_variable_level_export_enabled"] == cfg.lever4_variable_level_export_enabled
    assert d["pose_grad_every_k"] == cfg.pose_grad_every_k


# --- 5. oomph overlays the refinement stages only ------------------------------------


def test_oomph_overlays_refinement_stages_only() -> None:
    """The oomph sharp soft_cosine seg lever is on stages >= 4 (refinement); the coarse
    stages 0-3 stay vendored CE (seg_surrogate is None)."""
    from tac.torch_vehicle.curriculum import build_curriculum

    specs = build_curriculum(total_epoch_budget=800)
    plan = launcher._plan_stages(specs, oomph_seg_weight_mult=1.5)
    for s in plan:
        if s["stage"] >= launcher._REFINEMENT_FIRST_STAGE:
            assert s["seg_surrogate"] == "soft_cosine", f"stage {s['stage']} missing oomph"
            assert s["seg_temperature"] == 0.3 and s["seg_temperature_end"] == 0.05
            assert s["margin_weight_tau"] == 0.5 and s["margin_weight_renorm"] is True
            # seg_weight is the stage base * 1.5
            assert s["seg_weight"] == pytest.approx(100.0 * 1.5)
        else:
            assert s["seg_surrogate"] is None, f"coarse stage {s['stage']} got a seg lever"


def test_oomph_seg_weight_mult_one_isolates_sharpness() -> None:
    """``--oomph-seg-weight-mult 1.0`` keeps the stage base seg_weight (so an oomph win
    cannot be attributed to the seg_weight bump) while still applying the sharp anneal."""
    from tac.torch_vehicle.curriculum import build_curriculum

    specs = build_curriculum(total_epoch_budget=800)
    plan = launcher._plan_stages(specs, oomph_seg_weight_mult=1.0)
    ref = [s for s in plan if s["stage"] >= launcher._REFINEMENT_FIRST_STAGE]
    assert all(s["seg_surrogate"] == "soft_cosine" for s in ref)
    assert all(s["seg_weight"] == pytest.approx(100.0) for s in ref)


# --- guardrails ----------------------------------------------------------------------


def test_go_required_to_launch_real_run() -> None:
    """Without --go (and not --print-plan) the launcher REFUSES (no scorer load)."""
    with pytest.raises(SystemExit):
        launcher.main(["--arm", "arm_b", "--kd-warm-start-dir", _KD_DIR,
                       "--out-dir", "/tmp/does_not_run"])


def test_split_by_head_requires_mps_train_device() -> None:
    with pytest.raises(SystemExit):
        launcher.main(["--arm", "arm_b", "--kd-warm-start-dir", _KD_DIR,
                       "--out-dir", "/tmp/does_not_run", "--split-by-head",
                       "--train-device", "cpu", "--go"])


# --- 6. REGRESSION GUARD (2026-06-17): the MPS-pose UNBUNDLE + the self-protect warning ----------
# Closes the gap that caused the 7.3x CPU-pose regression: the unbundle flag
# (--pose-grad-on-train-device) was wired but launch commands OMITTED it. These tests verify the
# flag binds the Config field, the FINAL canonical combo (split + unbundle + equimarginal +
# Mahalanobis, NO trunk-stopgrad) builds, and main() WARNS when mps+split runs WITHOUT the unbundle.
from contextlib import redirect_stderr

# canonical FINAL combo (the reconciled optimal — NO trunk-stopgrad): split + unbundle + levers.
_CANONICAL_TAIL = [
    "--kd-warm-start-dir", _KD_DIR, "--out-dir", "/tmp/does_not_run",
    "--pose-film-v2", "--pose-equimarginal", "--pose-dim-weights-auto",
    "--split-by-head", "--pose-grad-on-train-device", "--train-device", "mps",
    "--rate-attack", "--ema-warmup", "--async-eval",
]


def test_unbundle_flag_sets_pose_grad_on_train_device() -> None:
    """--pose-grad-on-train-device binds cfg.pose_grad_on_train_device (the MPS-pose unbundle)."""
    on = launcher._build_torch_vehicle_config(
        _parse(["--arm", "arm_b", *_CANONICAL_TAIL]), Path("/tmp/does_not_run")
    )
    assert on.pose_grad_on_train_device is True
    # default (omit the flag): CPU-pose path = False (the slow path that the warning guards).
    off_args = _parse(["--arm", "arm_b", "--kd-warm-start-dir", _KD_DIR,
                       "--out-dir", "/tmp/does_not_run", "--pose-film-v2",
                       "--split-by-head", "--train-device", "mps"])
    off = launcher._build_torch_vehicle_config(off_args, Path("/tmp/does_not_run"))
    assert off.pose_grad_on_train_device is False


def test_canonical_fullstack_combo_builds_valid_config() -> None:
    """The FINAL reconciled config (split + unbundle + equimarginal + Mahalanobis, trunk-stopgrad
    OFF, pose every epoch) passes every __post_init__ cross-validation + keeps MPS off the score."""
    cfg = launcher._build_torch_vehicle_config(
        _parse(["--arm", "arm_b", *_CANONICAL_TAIL]), Path("/tmp/does_not_run")
    )
    assert cfg.split_by_head and cfg.pose_grad_on_train_device
    assert cfg.pose_film_trunk_stopgrad is False        # the pose FIX (no 0.19 cap)
    assert cfg.pose_equimarginal_enabled                # the working synergy
    assert cfg.pose_grad_every_k == 1                   # pose every epoch
    assert str(cfg.device) == "cpu"                     # AUTHORITY stays CPU — MPS never scores
    assert str(cfg.train_device) == "mps"               # gradient on MPS


def _main_stderr(argv: list[str]) -> str:
    """Run launcher.main(argv) (no --go ⇒ SystemExit REFUSED) and capture stderr (the warning)."""
    buf = io.StringIO()
    with redirect_stderr(buf), pytest.raises(SystemExit):
        launcher.main(argv)
    return buf.getvalue()


def test_unbundle_omission_warns_on_mps_split() -> None:
    """SELF-PROTECT: mps + split WITHOUT --pose-grad-on-train-device prints the CPU-PoseNet warning
    (the regression that silently took the 7.3x slow path)."""
    err = _main_stderr(["--arm", "arm_b", "--kd-warm-start-dir", _KD_DIR,
                        "--out-dir", "/tmp/does_not_run", "--pose-film-v2",
                        "--split-by-head", "--train-device", "mps"])
    assert "CPU-PoseNet path" in err and "7.3x" in err


def test_unbundle_present_suppresses_warn() -> None:
    """No warning when the unbundle flag IS set (the canonical fast path)."""
    err = _main_stderr(["--arm", "arm_b", *_CANONICAL_TAIL])
    assert "CPU-PoseNet path" not in err


# --- ACCELERATOR #1: margin-hinge THROUGHOUT (validated 2026-06-17) -------------------


def test_margin_hinge_throughout_replaces_surrogate_on_every_stage() -> None:
    """--seg-margin-hinge-throughout makes margin_hinge the seg surrogate for EVERY stage
    (coarse + refinement) with the validated PLAIN config (margin_target 1.0, road_lane 1.0
    OFF, NO Lever-5 margin-weight) — the contest fix for the soft_cosine/CE gradient-vanish."""
    from tac.torch_vehicle.curriculum import build_curriculum

    specs = build_curriculum(total_epoch_budget=800)
    plan = launcher._plan_stages(
        specs, oomph_seg_weight_mult=1.0, seg_margin_hinge_throughout=True
    )
    assert plan, "empty plan"
    for s in plan:
        assert s["seg_surrogate"] == "margin_hinge", f"stage {s['stage']} not margin_hinge"
        assert s["seg_margin_hinge_target"] == pytest.approx(1.0)
        assert s["road_lane_emphasis"] == pytest.approx(1.0)   # OFF — the 2.0 arm HURT
        assert s["margin_weight_tau"] is None                  # margin_hinge self-targets
        assert s["margin_weight_renorm"] is False


def test_margin_hinge_throughout_keeps_oomph_weight_crank_refinement_only() -> None:
    """The d_seg-FOCUS crank (oomph seg_weight x mult) is orthogonal to the surrogate: with
    mult=2.0 ONLY the refinement stages (>= 4) get the bumped seg_weight; coarse stays base."""
    from tac.torch_vehicle.curriculum import build_curriculum

    specs = build_curriculum(total_epoch_budget=800)
    base = {s.name: float(s.seg_weight) for s in specs}
    plan = launcher._plan_stages(
        specs, oomph_seg_weight_mult=2.0, seg_margin_hinge_throughout=True
    )
    for s in plan:
        expect = base[s["name"]] * (2.0 if s["stage"] >= launcher._REFINEMENT_FIRST_STAGE else 1.0)
        assert s["seg_weight"] == pytest.approx(expect), f"stage {s['stage']} weight crank wrong"
        assert s["seg_surrogate"] == "margin_hinge"


def test_margin_hinge_throughout_default_off_is_byte_identical_to_oomph() -> None:
    """The flag DEFAULTS off → the plan is the legacy soft_cosine-oomph overlay (coarse CE,
    refinement soft_cosine) — the determinism-preserving default the operator relies on."""
    from tac.torch_vehicle.curriculum import build_curriculum

    specs = build_curriculum(total_epoch_budget=800)
    default = launcher._plan_stages(specs, oomph_seg_weight_mult=1.0)
    explicit_off = launcher._plan_stages(
        specs, oomph_seg_weight_mult=1.0, seg_margin_hinge_throughout=False
    )
    assert default == explicit_off
    # the default is NOT margin_hinge anywhere (legacy path preserved)
    assert all(s["seg_surrogate"] in (None, "soft_cosine") for s in default)


def test_margin_hinge_throughout_flag_recorded_in_resolved_config() -> None:
    """The flag is threaded into the resolved-config dict (observability / provenance)."""
    on = launcher._resolve_config_dict(
        _parse(["--arm", "arm_b", "--out-dir", "/tmp/x", "--seg-margin-hinge-throughout"])
    )
    off = launcher._resolve_config_dict(_parse(["--arm", "arm_b", "--out-dir", "/tmp/x"]))
    assert on["seg_margin_hinge_throughout"] is True
    assert off["seg_margin_hinge_throughout"] is False
