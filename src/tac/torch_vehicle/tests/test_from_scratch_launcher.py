# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the ``--from-scratch`` DECISIVE-A/B launch mode on
``experiments/launch_l2_combined_attacks.py``.

The from-scratch arm is the clean apples-to-apples A/B against the levers-OFF
basin control (``experiments/results/torch_vehicle_full_mps_basin_bc20_n600``):
a FRESH decoder at epoch 0 (no forkpoint remap) running the FULL basin curriculum
with levers 2+3+5 ON from the first step. The three load-bearing claims (each, if
wrong, INVALIDATES the matched-epoch A/B):

1. **EPOCH-0 IDENTITY (the shared-init claim).** The FiLM wrapper is identity-init
   (``fc2`` zero-init → gamma=1/beta=0), so the from-scratch arm's epoch-0 render is
   BIT-EQUAL to the vendored no-FiLM render the basin starts from. NO-FAKE: the test
   would FAIL if FiLM were not identity (perturb ``fc2`` → the render diverges), so it
   verifies BEHAVIOR not a constant.

2. **CURRICULUM-SCHEDULE MATCH (the matched-epoch validity).** The from-scratch
   curriculum's TRAINING-schedule fields (name/epochs/eval_every/batch_size/ema_decay/
   lrs/clips/weights/cat/qat/init_latents_random) are IDENTICAL to the basin's
   ``build_curriculum(...)``; the levers are the ONLY allowed difference. The launcher
   FAILS CLOSED if the schedule diverges.

3. **LEVERS GENUINELY ACTIVE FROM EPOCH 0 (NO-FAKE behavior).** With the seg surrogate
   + margin + pose-FiLM enabled, every stage's StageSpec carries the lever fields (so
   the driver routes the seg term through the surrogate from epoch 0 of stage 1), the
   pose-FiLM cfg is on, and a synthetic-scorer 3-epoch end-to-end run trains + exports
   a parseable archive with the pose section (proving the composed lever path runs).
"""

from __future__ import annotations

# Import the launcher module under test (experiments/ is on sys.path via conftest /
# the repo layout; import by file path to be robust).
import importlib.util
import json
from pathlib import Path

import torch

from tac.torch_vehicle.curriculum import (
    SEG_ANNEAL_GRADIENT_FLOOR_T,
    build_curriculum,
    seg_anneal_temperature_is_gradient_alive,
)

_LAUNCHER_PATH = (
    Path(__file__).resolve().parents[4]
    / "experiments"
    / "launch_l2_combined_attacks.py"
)
_spec = importlib.util.spec_from_file_location("_l2_launcher", _LAUNCHER_PATH)
launcher = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(launcher)


def _parse(argv):
    return launcher._build_arg_parser().parse_args(argv)


# ---------------------------------------------------------------------------
# LENS 1 — EPOCH-0 IDENTITY (the shared-init claim; NO-FAKE)
# ---------------------------------------------------------------------------
def test_from_scratch_epoch0_render_is_bit_equal_to_vendored_no_film():
    """The FiLM-identity-init render == the vendored no-FiLM render at init."""
    from tac.torch_vehicle.pose_film import PoseFiLMHNeRVWrapper
    from tac.torch_vehicle.vendored_imports import import_vendored

    torch.manual_seed(11)
    vendored = import_vendored("model").HNeRVDecoder(
        latent_dim=28, base_channels=20, eval_size=(384, 512)
    )
    wrap = PoseFiLMHNeRVWrapper(vendored, n_pairs=4, film_hidden=8)
    wrap.set_stored_pose(torch.randn(4, 6) * 3.0)  # non-trivial stored pose
    vendored.eval()
    wrap.eval()

    z = torch.randn(2, 28)
    idx = torch.tensor([0, 2])
    with torch.no_grad():
        out_base = vendored(z)
        out_film = wrap(z, idx)
    assert torch.equal(out_base, out_film), (
        "FiLM identity-init must reproduce the vendored render bit-for-bit at epoch 0"
    )


def test_epoch0_identity_is_not_vacuous_perturbed_film_diverges():
    """NO-FAKE guard: if FiLM were NOT identity, the render diverges — so the
    identity test above verifies BEHAVIOR, not a trivially-true constant."""
    from tac.torch_vehicle.pose_film import PoseFiLMHNeRVWrapper
    from tac.torch_vehicle.vendored_imports import import_vendored

    torch.manual_seed(11)
    vendored = import_vendored("model").HNeRVDecoder(
        latent_dim=28, base_channels=20, eval_size=(384, 512)
    )
    wrap = PoseFiLMHNeRVWrapper(vendored, n_pairs=4, film_hidden=8)
    wrap.set_stored_pose(torch.randn(4, 6) * 3.0)
    vendored.eval()
    wrap.eval()
    z = torch.randn(2, 28)
    idx = torch.tensor([0, 2])
    with torch.no_grad():
        out_base = vendored(z)
        # Break identity-init: perturb fc2 so gamma != 1 / beta != 0.
        wrap.pose_film.fc2.weight.add_(torch.randn_like(wrap.pose_film.fc2.weight))
        out_perturbed = wrap(z, idx)
    assert (out_base - out_perturbed).abs().max().item() > 1.0, (
        "a NON-identity FiLM MUST diverge from the vendored render (proves the "
        "identity test is not vacuous)"
    )


def test_from_scratch_initial_latents_bit_match_basin_rng_neutral_film():
    """The from-scratch (FiLM-ON) initial latents bit-match the no-FiLM basin's —
    proves the RNG-neutral FiLM construction (snapshot/restore around the FiLM build
    in driver._new_decoder) keeps the stage-1 latent draw at the SAME global-RNG
    stream position the basin draws from. WITHOUT the fix the FiLM init advances the
    stream and the latents diverge (the no-fake guard below proves the bit-match is
    real)."""
    from tac.torch_vehicle.pose_film import PoseFiLMHNeRVWrapper
    from tac.torch_vehicle.vendored_imports import import_vendored

    def _init_latents(pose_film: bool):
        torch.manual_seed(0)
        v = import_vendored("model").HNeRVDecoder(
            latent_dim=28, base_channels=20, eval_size=(384, 512)
        )
        if pose_film:
            # MIRROR driver._new_decoder's RNG-neutral FiLM build (snapshot/restore).
            st = torch.get_rng_state()
            PoseFiLMHNeRVWrapper(v, n_pairs=8, film_hidden=8)
            torch.set_rng_state(st)
        return (torch.randn(8, 28) * 0.1).clone()

    lat_basin = _init_latents(False)
    lat_from_scratch = _init_latents(True)
    assert torch.equal(lat_basin, lat_from_scratch), (
        "from-scratch initial latents must bit-match the basin (RNG-neutral FiLM build)"
    )


def test_latent_bit_match_is_not_vacuous_without_rng_restore_they_diverge():
    """NO-FAKE guard: WITHOUT the snapshot/restore, the FiLM init advances the global
    RNG stream and the latents DIVERGE — so the bit-match test above verifies the fix,
    not a trivially-true property."""
    from tac.torch_vehicle.pose_film import PoseFiLMHNeRVWrapper
    from tac.torch_vehicle.vendored_imports import import_vendored

    torch.manual_seed(0)
    import_vendored("model").HNeRVDecoder(
        latent_dim=28, base_channels=20, eval_size=(384, 512)
    )
    lat_basin = (torch.randn(8, 28) * 0.1).clone()

    torch.manual_seed(0)
    v_fs = import_vendored("model").HNeRVDecoder(
        latent_dim=28, base_channels=20, eval_size=(384, 512)
    )
    PoseFiLMHNeRVWrapper(v_fs, n_pairs=8, film_hidden=8)  # NO restore -> stream shifts
    lat_no_restore = (torch.randn(8, 28) * 0.1).clone()
    assert not torch.equal(lat_basin, lat_no_restore), (
        "without RNG restore the FiLM init MUST shift the latent draw (proves the "
        "bit-match test is not vacuous)"
    )


# ---------------------------------------------------------------------------
# LENS 2 — CURRICULUM-SCHEDULE MATCH (matched-epoch A/B validity)
# ---------------------------------------------------------------------------
_BASIN_KW = {"total_epoch_budget": None, "ema_decay": 0.999, "eval_every": 10}


def test_from_scratch_curriculum_schedule_is_identical_to_basin():
    """Every schedule field matches the basin; only lever fields differ."""
    args = _parse([
        "--from-scratch", "--levers", "seg_pose",
        "--seg-surrogate", "soft_cosine", "--seg-temperature", "1.0",
        "--seg-temperature-end", "0.10", "--margin-weight-tau", "2.0",
        "--total-epoch-budget", "29650", "--eval-every", "25",
        "--ema-decay", "0.999", "--n-pairs", "600", "--base-channels", "20",
    ])
    overrides = launcher._resolve_lever_overrides(args)
    arm = launcher._build_from_scratch_curriculum(
        total_epoch_budget=29650, ema_decay=0.999, eval_every=25,
        lever_overrides=overrides,
    )
    basin = launcher._basin_reference_curriculum(
        total_epoch_budget=29650, ema_decay=0.999, eval_every=25,
    )
    match = launcher._curriculum_match_table(arm, basin)
    assert match["schedule_identical_to_basin"] is True, json.dumps(
        [r for r in match["stages"] if r["schedule_mismatch_vs_basin"]], indent=2
    )
    assert match["n_stages"] == 8


def test_match_table_detects_a_schedule_divergence_no_fake():
    """NO-FAKE guard: if the schedules ACTUALLY differ, the match table catches it
    (so the match==True result above is a real comparison, not a constant)."""
    from dataclasses import replace

    arm = build_curriculum(**_BASIN_KW)
    basin = build_curriculum(**_BASIN_KW)
    # Corrupt one SCHEDULE field on the arm (epochs) — must be detected.
    arm = [replace(arm[0], epochs=arm[0].epochs + 1), *arm[1:]]
    match = launcher._curriculum_match_table(arm, basin)
    assert match["schedule_identical_to_basin"] is False
    assert any(
        "epochs" in r["schedule_mismatch_vs_basin"] for r in match["stages"]
    )


def test_seg_loss_fn_closure_identity_is_not_falsely_flagged_as_divergence():
    """``seg_loss_fn`` is a fresh per-call lambda closure (object-identity always
    differs), but the arm does NOT override it — the match table compares it by
    module+qualname so it is NOT flagged as a schedule divergence."""
    arm = build_curriculum(**_BASIN_KW)
    basin = build_curriculum(**_BASIN_KW)
    # Object identity DIFFERS (fresh closures) — the naive `!=` trap.
    assert arm[0].seg_loss_fn is not basin[0].seg_loss_fn
    # But the match table (qualname+module) does NOT flag it.
    match = launcher._curriculum_match_table(arm, basin)
    assert match["schedule_identical_to_basin"] is True
    for r in match["stages"]:
        assert "seg_loss_fn" not in r["schedule_mismatch_vs_basin"]


def test_match_table_detects_a_real_seg_loss_fn_swap_no_fake():
    """NO-FAKE guard: if the arm's ``seg_loss_fn`` were ACTUALLY swapped for a
    different function (different qualname), the match table catches it — so the
    qualname check is a real comparison, not a constant True."""
    from dataclasses import replace

    arm = build_curriculum(**_BASIN_KW)
    basin = build_curriculum(**_BASIN_KW)

    def _alien_seg_loss(seg_logits, targets_hard):  # different qualname
        return seg_logits.sum() * 0.0

    arm = [replace(arm[0], seg_loss_fn=_alien_seg_loss), *arm[1:]]
    match = launcher._curriculum_match_table(arm, basin)
    assert match["schedule_identical_to_basin"] is False
    assert any(
        "seg_loss_fn" in r["schedule_mismatch_vs_basin"] for r in match["stages"]
    )


def test_lever_overrides_do_not_touch_schedule_fields():
    """The lever overrides replace ONLY lever fields — the schedule is preserved."""
    args = _parse([
        "--from-scratch", "--levers", "seg_pose",
        "--seg-surrogate", "soft_cosine", "--seg-temperature-end", "0.10",
        "--margin-weight-tau", "2.0",
    ])
    overrides = launcher._resolve_lever_overrides(args)
    schedule_fields = {
        "name", "epochs", "eval_every", "batch_size", "ema_decay", "use_muon",
        "adamw_lr", "muon_lr", "muon_weight_decay", "latent_lr_mult", "grad_clip",
        "grad_clip_muon", "lr_floor_ratio", "seg_weight", "pose_weight",
        "cat_lambda", "cat_sigma", "use_qat", "init_latents_random",
    }
    assert schedule_fields.isdisjoint(overrides.keys()), (
        f"lever overrides must NOT include schedule fields: "
        f"{schedule_fields & set(overrides.keys())}"
    )


def test_seg_pose_mode_keeps_levers_1_and_4_off():
    """The distortion arm is levers 2+3+5 ONLY — rate (1) + score-aware-QAT (4) OFF."""
    args = _parse([
        "--from-scratch", "--levers", "seg_pose",
        "--seg-surrogate", "soft_cosine", "--seg-temperature-end", "0.10",
        "--margin-weight-tau", "2.0", "--no-score-aware-qat",
    ])
    overrides = launcher._resolve_lever_overrides(args)
    assert overrides["rate_lambda_w"] == 0.0  # Lever 1 OFF
    assert overrides["rate_lambda_lat"] == 0.0  # Lever 1 OFF
    assert overrides["score_aware_qat"] is False  # Lever 4 OFF
    # Levers 2 + 5 ON.
    assert overrides["seg_surrogate"] == "soft_cosine"
    assert overrides["seg_temperature_end"] == 0.10
    assert overrides["margin_weight_tau"] == 2.0


# ---------------------------------------------------------------------------
# LENS 3 — LEVERS GENUINELY ACTIVE FROM EPOCH 0 (NO-FAKE behavior)
# ---------------------------------------------------------------------------
def test_levers_active_on_every_stage_from_stage0():
    """The seg surrogate + anneal + margin are set on EVERY stage's StageSpec — so
    the driver routes the seg term through the surrogate from epoch 0 of stage 1."""
    args = _parse([
        "--from-scratch", "--levers", "seg_pose",
        "--seg-surrogate", "soft_cosine", "--seg-temperature", "1.0",
        "--seg-temperature-end", "0.10", "--margin-weight-tau", "2.0",
    ])
    overrides = launcher._resolve_lever_overrides(args)
    arm = launcher._build_from_scratch_curriculum(
        total_epoch_budget=None, ema_decay=0.999, eval_every=25,
        lever_overrides=overrides,
    )
    # Stage 0 (stage1_v328_ce) is the from-epoch-0 stage — levers MUST be active there.
    assert arm[0].name == "stage1_v328_ce"
    for i, spec in enumerate(arm):
        assert spec.seg_surrogate == "soft_cosine", f"stage {i} missing seg surrogate"
        assert spec.seg_temperature_end == 0.10, f"stage {i} missing anneal endpoint"
        assert spec.margin_weight_tau == 2.0, f"stage {i} missing margin weight"


def test_basin_reference_has_levers_off():
    """The basin reference curriculum is levers-OFF (the A/B reference)."""
    basin = launcher._basin_reference_curriculum(
        total_epoch_budget=None, ema_decay=0.999, eval_every=25,
    )
    for spec in basin:
        assert spec.seg_surrogate is None
        assert spec.seg_temperature_end is None
        assert spec.margin_weight_tau is None
        assert spec.rate_lambda_w == 0.0
        assert spec.score_aware_qat is False


def test_from_scratch_self_test_runs_levers_end_to_end():
    """A synthetic-scorer ~3-epoch end-to-end run with the seg+margin+pose-FiLM
    levers trains + exports a parseable archive WITH the pose section — proving the
    composed lever path actually runs from a fresh init (NO-FAKE: a real archive +
    real pose round-trip, not a metadata stub)."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / "from_scratch_selftest"
        args = _parse([
            "--self-test", "--levers", "seg_pose",
            "--seg-surrogate", "soft_cosine", "--seg-temperature", "1.0",
            "--seg-temperature-end", "0.10", "--margin-weight-tau", "2.0",
            "--base-channels", "20", "--latent-dim", "28",
            "--out-dir", str(out_dir),
        ])
        rc = launcher._run_self_test(args, out_dir)
        assert rc == 0, "self-test (seg + margin + pose-FiLM) must PASS end-to-end"
        assert (out_dir / "best" / "best_archive.bin").exists()


# ---------------------------------------------------------------------------
# Gradient-alive observability (the anneal endpoint vs the MEASURED floor)
# ---------------------------------------------------------------------------
def test_seg_anneal_endpoint_010_is_below_the_measured_gradient_floor():
    """HONEST: per the R12-corrected floor (0.3), end=0.10 is NOT gradient-alive.
    This is a load-bearing finding — a 0.10 endpoint drives the seg surrogate's
    gradient into the cold-dead zone for the cold portion of each stage. The launcher
    SURFACES this (it does not silently pass)."""
    assert SEG_ANNEAL_GRADIENT_FLOOR_T == 0.3
    assert seg_anneal_temperature_is_gradient_alive(0.10) is False
    assert seg_anneal_temperature_is_gradient_alive(0.05) is False
    # Only T >= 0.3 is gradient-alive per the measured real-scorer sweep.
    assert seg_anneal_temperature_is_gradient_alive(0.30) is True


def test_default_anneal_endpoint_is_the_r12_gradient_alive_floor_030():
    """R12-CORRECTED DEFAULT (load-bearing): with NO explicit --seg-temperature-end,
    BOTH seg_pose and all modes default the anneal endpoint to 0.30 (the measured
    gradient-alive floor) — NOT the pre-R12 0.05/0.10 dead-zone value. This guards the
    correction so the delivered launch command keeps the seg lever gradient-ALIVE
    through the full anneal; a silent regression to a sub-floor default would FAIL."""
    for mode in ("seg_pose", "all"):
        args = _parse([
            "--from-scratch", "--levers", mode,
            "--seg-surrogate", "soft_cosine", "--seg-temperature", "1.0",
            "--margin-weight-tau", "2.0",
        ])
        overrides = launcher._resolve_lever_overrides(args)
        end_t = overrides["seg_temperature_end"]
        assert end_t == 0.30, f"{mode}: default anneal endpoint must be 0.30, got {end_t}"
        assert seg_anneal_temperature_is_gradient_alive(end_t) is True, (
            f"{mode}: the default endpoint {end_t} must be gradient-ALIVE (>= floor 0.3)"
        )


def test_explicit_endpoint_still_overrides_the_030_default():
    """An explicit --seg-temperature-end OVERRIDES the 0.30 default (so the operator
    can still probe a sub-floor endpoint deliberately — the launcher surfaces the
    gradient-alive verdict but does not refuse it)."""
    args = _parse([
        "--from-scratch", "--levers", "seg_pose",
        "--seg-surrogate", "soft_cosine", "--seg-temperature-end", "0.15",
        "--margin-weight-tau", "2.0",
    ])
    overrides = launcher._resolve_lever_overrides(args)
    assert overrides["seg_temperature_end"] == 0.15
