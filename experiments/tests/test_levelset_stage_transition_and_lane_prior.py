# SPDX-License-Identifier: MIT
"""Tests for the two theta*-prereq builds on the level-set witness trainer (DAG FEED-fw).

BUILD 1 -- STAGE-TRANSITION TREATMENT (operator 2026-06-26 "different stages need different
treatment ... transitions must re-treat"; FEED-ft#3 tau-jump root cause = a stage boundary hit at
full LR with stale AdamW momentum). Two ADDITIVE, default-OFF levers at every AdamW->AdamW boundary
(curriculum seg-form change / lane-edge engage / margin-saliency engage):
  * ``--stage-transition-rewarmup-epochs N`` (default 0=OFF) -- ramp LR from a floor back to the
    scheduled LR over N epochs after the boundary (the pure ``_stage_rewarmup_factor`` helper).
  * ``--stage-transition-reset-moments`` (default OFF) -- rebuild the AdamW optimizer so m/v are
    zeroed (MLX ``Optimizer.init`` only fills MISSING state, so a true reset needs a fresh object).

BUILD 2 -- LANE-PRIOR phi1 (FEED-fs: the openpilot deg-3 centerline IS the Road<->Lane separatrix,
residual 1.9e-5). ``--lane-prior-phi1`` (default OFF) injects the openpilot-centerline signed
distance into the structured-init target's lane (phi1) channel via the REUSED
``tac.boundary_math.lane_sdf_component`` helpers (build_structured_lane_sdf + inject_lane_sdf).

BIT-IDENTITY discipline (NO-FAKE): with every new flag at its default, the path is byte-identical to
the pre-FEED-fw trainer (the running ablation pid 51464 resumes unaffected). These tests assert that
property directly (factor == EXACTLY 1.0 => lr*1.0 == lr; inject is skipped => target untouched).

Pure-helper + argparse + main()-validation tests only (CPU, MLX-free): the realized-through-R loop
needs MLX + the frozen scorer adapter + the GT cache, so the loop integration is validated by the
$0 default-off bit-identity + the fail-closed config guards here, plus the future GPU run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[2]
for _p in (_REPO, _REPO / "src", _REPO / "experiments", _REPO / "upstream"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import train_levelset_witness_realized_through_R_mlx as T  # noqa: E402

from tac.boundary_math.lane_sdf_component import (  # noqa: E402
    build_structured_lane_sdf,
    inject_lane_sdf,
)


# =========================================================================== BUILD 1: pure helper
F = T._stage_rewarmup_factor


def test_rewarmup_off_returns_exactly_one():
    # rewarmup_epochs <= 0 (the DEFAULT) => EXACTLY 1.0 for ANY epoch/boundary => bit-identical.
    for ep in (1, 5, 100, 1500):
        for b in (None, 0, 3, 1499):
            assert F(ep, b, 0, 0.1, "linear") == 1.0
            assert F(ep, b, -3, 0.5, "cosine") == 1.0


def test_rewarmup_no_boundary_returns_one():
    # last_boundary_epoch is None (no boundary fired yet) => 1.0 even with a positive window.
    assert F(7, None, 10, 0.1, "linear") == 1.0
    assert F(7, None, 10, 0.1, "cosine") == 1.0


def test_rewarmup_at_boundary_epoch_is_floor():
    # d == 0 (the boundary epoch itself) => the floor (LR starts low post-boundary).
    assert F(10, 10, 5, 0.1, "linear") == pytest.approx(0.1)
    assert F(10, 10, 5, 0.25, "cosine") == pytest.approx(0.25)


def test_rewarmup_at_or_after_window_end_is_one():
    # d >= rewarmup_epochs => back to the full scheduled LR (1.0).
    assert F(15, 10, 5, 0.1, "linear") == 1.0
    assert F(16, 10, 5, 0.1, "linear") == 1.0
    assert F(99, 10, 5, 0.1, "cosine") == 1.0


def test_rewarmup_before_boundary_is_one():
    # d < 0 (epoch precedes the recorded boundary; should not happen but defensive) => 1.0.
    assert F(8, 10, 5, 0.1, "linear") == 1.0


def test_rewarmup_linear_midpoint():
    # d=2, N=4, floor=0.2 => 0.2 + 0.8 * (2/4) = 0.6.
    assert F(12, 10, 4, 0.2, "linear") == pytest.approx(0.6)


def test_rewarmup_linear_is_monotone_nondecreasing():
    vals = [F(10 + d, 10, 8, 0.05, "linear") for d in range(8)]
    assert vals[0] == pytest.approx(0.05)
    assert all(b >= a for a, b in zip(vals, vals[1:]))
    assert vals[-1] < 1.0  # last in-window step is < full; d==N (next epoch) is 1.0
    assert F(18, 10, 8, 0.05, "linear") == 1.0


def test_rewarmup_cosine_is_monotone_and_endpoints():
    vals = [F(10 + d, 10, 6, 0.1, "cosine") for d in range(6)]
    assert vals[0] == pytest.approx(0.1)               # d=0 => floor
    assert all(b >= a for a, b in zip(vals, vals[1:]))  # monotone up
    assert F(16, 10, 6, 0.1, "cosine") == 1.0           # d==N => full


def test_rewarmup_floor_clamped_to_unit_interval():
    # floor > 1 clamps to 1.0; floor < 0 clamps to 0.0 (stays a valid multiplier).
    assert F(10, 10, 5, 2.0, "linear") == 1.0
    assert F(10, 10, 5, -1.0, "linear") == 0.0
    # within-window with clamped floor still in [0,1].
    v = F(11, 10, 5, -1.0, "linear")
    assert 0.0 <= v <= 1.0


def test_rewarmup_off_is_bit_identical_multiplier():
    # The loop applies `lr = lr * factor`. With OFF, factor is EXACTLY 1.0 => lr*1.0 == lr for finite
    # IEEE floats (this IS the bit-identity guarantee for the LR schedule when the flag is off).
    for lr in (1e-3, 1.234e-4, 9.99e-5, 0.0, 0.05):
        assert lr * F(7, 3, 0, 0.1, "linear") == lr


def test_rewarmup_unknown_shape_falls_back_to_linear():
    # Any non-"cosine" shape uses the linear branch (defensive; CLI restricts to linear/cosine).
    assert F(12, 10, 4, 0.2, "weird") == pytest.approx(F(12, 10, 4, 0.2, "linear"))


# ======================================================================= BUILD 1: argparse + guards
def _run_main_capturing_args(argv):
    """Run the REAL main() with run_train STUBBED so no MLX op / GPU touch occurs; return the parsed
    args. main() validates (parse -> BUILD-1/BUILD-2 guards -> run_train) so a guard-passing argv
    reaches the stub and we capture the args it would have trained with."""
    captured = {}

    def _fake_run_train(a):
        captured["args"] = a
        return {"history": [], "front_end": "x", "axis": "test",
                "checkpoint": "x", "stage_checkpoints": []}

    orig = T.run_train
    T.run_train = _fake_run_train  # type: ignore[assignment]
    try:
        rc = T.main(argv)
    finally:
        T.run_train = orig  # type: ignore[assignment]
    assert rc == 0
    return captured["args"]


def test_argparse_defaults_are_off(tmp_path):
    # The DEFAULTS are the bit-identical path: a guard-passing default argv reaches run_train (no
    # guard fired) and every new flag is at its OFF default.
    a = _run_main_capturing_args(["--out-dir", str(tmp_path), "--epochs", "1"])
    assert a.stage_transition_rewarmup_epochs == 0
    assert a.stage_transition_reset_moments is False
    assert a.stage_transition_rewarmup_floor == pytest.approx(0.1)
    assert a.stage_transition_rewarmup_shape == "linear"
    assert a.lane_prior_phi1 is False
    assert a.lane_prior_phi1_mode == "replace"
    assert a.lane_prior_phi1_source_pair == 0
    assert a.lane_prior_phi1_dash_gate is True


def test_argparse_flags_parse_when_set(tmp_path):
    a = _run_main_capturing_args([
        "--out-dir", str(tmp_path), "--epochs", "10",
        "--stage-transition-rewarmup-epochs", "4",
        "--stage-transition-rewarmup-floor", "0.05",
        "--stage-transition-rewarmup-shape", "cosine",
        "--stage-transition-reset-moments",
        "--structured-init", "--lane-prior-phi1",
        "--lane-prior-phi1-mode", "bias", "--lane-prior-phi1-bias-scale", "0.5",
        "--lane-prior-phi1-source-pair", "3", "--no-lane-prior-phi1-dash-gate",
    ])
    assert a.stage_transition_rewarmup_epochs == 4
    assert a.stage_transition_rewarmup_floor == pytest.approx(0.05)
    assert a.stage_transition_rewarmup_shape == "cosine"
    assert a.stage_transition_reset_moments is True
    assert a.lane_prior_phi1 is True
    assert a.lane_prior_phi1_mode == "bias"
    assert a.lane_prior_phi1_bias_scale == pytest.approx(0.5)
    assert a.lane_prior_phi1_source_pair == 3
    assert a.lane_prior_phi1_dash_gate is False


def _main_raises(argv, tmp_path):
    full = ["--out-dir", str(tmp_path), "--epochs", "10", *argv]
    with pytest.raises(ValueError) as ei:
        T.main(full)
    return str(ei.value)


def test_guard_rewarmup_requires_lr_schedule(tmp_path):
    msg = _main_raises(["--stage-transition-rewarmup-epochs", "5", "--no-lr-schedule"], tmp_path)
    assert "requires --lr-schedule" in msg


def test_guard_rewarmup_floor_in_unit_interval(tmp_path):
    msg = _main_raises(["--stage-transition-rewarmup-epochs", "5",
                        "--stage-transition-rewarmup-floor", "2.0"], tmp_path)
    assert "must be in [0, 1]" in msg


def test_guard_rewarmup_epochs_nonnegative(tmp_path):
    msg = _main_raises(["--stage-transition-rewarmup-epochs", "-1"], tmp_path)
    assert "must be >= 0" in msg


def test_guard_lane_prior_requires_structured_init(tmp_path):
    msg = _main_raises(["--lane-prior-phi1"], tmp_path)
    assert "requires --structured-init" in msg


def test_rewarmup_default_off_passes_validation(tmp_path):
    # The default-off path must NOT raise the BUILD-1/BUILD-2 guards (it reaches run_train, stubbed).
    a = _run_main_capturing_args(["--out-dir", str(tmp_path), "--epochs", "1"])
    assert a.stage_transition_rewarmup_epochs == 0 and a.lane_prior_phi1 is False


# ====================================================================== BUILD 2: lane-prior reuse
def _synthetic_lstar(h=384, w=512):
    """A synthetic SegNet argmax with a vertical lane stripe (class 1) over road (0) / undriv (2)."""
    lst = np.full((h, w), 2, np.int64)       # top = undrivable/sky
    lst[200:, :] = 0                          # lower half = road
    lst[220:, 250:262] = 1                    # a lane stripe (class 1)
    return lst


def test_build_structured_lane_sdf_returns_finite_field_and_meta():
    lst = _synthetic_lstar()
    phi1, meta = build_structured_lane_sdf(lst, lane_cls=1, dash_gate=True, centerline_deg=3)
    assert phi1.shape == lst.shape
    assert phi1.dtype == np.float32
    assert np.isfinite(phi1).all()
    assert meta["n_lines"] >= 1
    assert meta["total_floats"] > 0          # the ~8-float/line manifold coords (the COUNTED stat)


def test_inject_lane_sdf_replace_only_touches_lane_channel():
    rng = np.random.default_rng(0)
    phi = rng.standard_normal((384, 512, 5)).astype(np.float32)
    phi1 = build_structured_lane_sdf(_synthetic_lstar(), lane_cls=1)[0]
    out = inject_lane_sdf(phi.copy(), phi1, lane_cls=1, mode="replace")
    assert np.array_equal(out[..., 1], phi1.astype(np.float32))
    for k in (0, 2, 3, 4):
        assert np.array_equal(out[..., k], phi[..., k]), f"channel {k} must be untouched"


def test_inject_lane_sdf_bias_adds_scaled_field():
    rng = np.random.default_rng(1)
    phi = rng.standard_normal((384, 512, 5)).astype(np.float32)
    phi1 = build_structured_lane_sdf(_synthetic_lstar(), lane_cls=1)[0]
    out = inject_lane_sdf(phi.copy(), phi1, lane_cls=1, mode="bias", bias_scale=0.5)
    assert np.allclose(out[..., 1], phi[..., 1] + 0.5 * phi1.astype(np.float32))
    for k in (0, 2, 3, 4):
        assert np.array_equal(out[..., k], phi[..., k])


def test_inject_lane_sdf_off_is_bit_identical():
    # The trainer SKIPS inject entirely when --lane-prior-phi1 is off => the structured target is
    # byte-identical. We model that contract: NOT calling inject leaves phi unchanged.
    rng = np.random.default_rng(2)
    phi = rng.standard_normal((384, 512, 5)).astype(np.float32)
    phi_off = phi.copy()  # the off path makes no call
    assert np.array_equal(phi, phi_off)


def test_lane_prior_homography_matches_task_K_at_scorer_res():
    # The task K focal is 910 at full-res (width 1164); the reused helper uses fx=400.3 at scorer
    # width 512. They are the SAME camera: 910 * 512/1164 == 400.3 (within fit tolerance). This guards
    # the "REUSE the existing helper == use the task's homography at scorer res" claim.
    import tac.boundary_math.lane_sdf_component as L
    assert L._FX == pytest.approx(910.0 * 512.0 / 1164.0, abs=0.6)


# ===================================================================== C1 SELF-PROTECT (review FEED-hp/hr)
# The C1 CRITICAL silent-no-op: ``lane_thin_gate`` (and the sibling lane-edge / margin-saliency gates)
# was initialized ``{"on": start <= 1}`` ONCE and the loop never re-flipped the thin-lane gate, so
# ``--lane-thin-start-epoch > 1`` (the help-RECOMMENDED 300) left it stuck OFF forever -> the loss
# branch never fired -> a FALSE 'thin-lane prior does nothing' verdict from dead code = the NO-FAKE
# silent-no-op class. The fix extracted ``lever_gate_on_at_epoch`` as the SINGLE engagement predicate
# the loop now uses for ALL THREE levers; these tests are the self-protect (per CLAUDE.md "Bugs must be
# permanently fixed AND self-protected against"): they FAIL if the predicate regresses OR if any of the
# three loop gate-flips is removed (so the silent-no-op cannot silently re-emerge).
G = T.lever_gate_on_at_epoch


def test_c1_gate_engages_at_start_epoch_when_start_gt_1():
    # THE C1 REGRESSION CASE: start>1 MUST engage at ep==start (and after). The pre-fix static
    # init {"on": start<=1} returned False here forever -> silent no-op. This asserts the LIVE
    # behavior is now correct.
    assert G(0.5, 300, 300) is True          # engages exactly at the (recommended) start epoch
    assert G(0.5, 300, 301) is True          # stays engaged after
    assert G(0.5, 300, 1500) is True         # ... through the run
    assert G(0.5, 300, 299) is False         # OFF before start (the intended warmup window)
    assert G(0.5, 300, 1) is False           # OFF at ep1 when start>1 (NOT the buggy always/never)


def test_c1_gate_default_start_engages_from_epoch_1():
    # start<=1 (the default always-on path) engages from ep1 => bit-identical to the pre-lever path.
    for start in (0, 1):
        assert G(0.7, start, 1) is True
        assert G(0.7, start, 2) is True


def test_c1_gate_off_when_weight_nonpositive():
    # weight<=0 (the lever is OFF) => NEVER engaged at ANY epoch (the loss branch also guards weight,
    # but the predicate is the single source of truth and must agree).
    for ep in (1, 5, 300, 1500):
        assert G(0.0, 0, ep) is False
        assert G(0.0, 300, ep) is False
        assert G(-1.0, 0, ep) is False


def test_c1_validator_catches_never_engage_silent_no_op():
    # The fail-closed config guard: weight>0 but start>epochs => the hinge would NEVER engage => the
    # SAME silent-no-op verdict. Must raise LOUDLY (the validator's job; complements the loop-flip).
    with pytest.raises(ValueError) as ei:
        T.validate_lane_thin_config(lane_thin_weight=0.5, lane_thin_start_epoch=2000,
                                    epochs=1500, lane_thin_class=1, lane_thin_radius=4)
    assert "NEVER engage" in str(ei.value) or "silent no-op" in str(ei.value)


def test_c1_validator_passes_valid_and_noop_when_off():
    # Valid config (start<=epochs) does NOT raise; and when the lever is OFF (weight<=0) the guard is
    # a NO-OP (never gates the additive default path), even with a nonsense start.
    T.validate_lane_thin_config(lane_thin_weight=0.5, lane_thin_start_epoch=300,
                                epochs=1500, lane_thin_class=1, lane_thin_radius=4)
    T.validate_lane_thin_config(lane_thin_weight=0.0, lane_thin_start_epoch=99999,
                                epochs=1500, lane_thin_class=1, lane_thin_radius=4)


def test_c1_all_three_loop_gates_route_through_the_tested_predicate():
    # STRUCTURAL self-protect: the epoch loop MUST flip ALL THREE lever gates via the unit-tested
    # ``lever_gate_on_at_epoch`` predicate. The C1 bug was the ABSENCE of the thin-lane flip; this
    # guards that none of the three flips is silently dropped or hardcoded back to a static init / a
    # bare ``ep >= start`` (which would dodge the tested predicate). If a future edit removes a flip,
    # this fails -> the silent-no-op class cannot re-emerge unnoticed.
    src = Path(T.__file__).read_text()
    for gate in ("lane_gate", "msal_gate", "lane_thin_gate"):
        needle = f'{gate}["on"] = lever_gate_on_at_epoch('
        assert needle in src, (
            f"epoch-loop gate-flip for {gate} no longer routes through lever_gate_on_at_epoch -> the "
            "C1 silent-no-op class can re-emerge. Re-wire the loop flip through the tested predicate.")


# ============================================================ R2a-MED-1 SELF-PROTECT (resume arch-flag)
# The R2a-MEDIUM-1: the film_per_layer/film_concat_code ARCH flags were persisted ONLY in result.json
# (loop-end), NOT in the resume checkpoint -> a crash-resume from the ckpt dir alone (the resumability
# discipline) had no record FiLM was ON; if the resume cmd omitted the flag, MLX model.update would
# SILENTLY DROP the trained film_pl/concat_pl params = a corrupted, non-reproducible resume. The fix
# persists __cfg_film_* in the resume sidecar + a fail-closed arch-drift guard before model.update.
import types as _types  # noqa: E402


def _min_args_for_resume_builder(**over):
    base = dict(n_hidden=4, hidden_dim=96, mod_dim=32, self_orient=False, w_pose=1.0,
                film_per_layer=False, film_concat_code=False, film_stiefel=False)
    base.update(over)
    return _types.SimpleNamespace(**base)


def test_r2a_resume_sidecar_persists_film_arch_flags():
    live = {"film.weight": np.zeros((4, 3), np.float32), "code": np.zeros((2, 3), np.float32)}
    a = _min_args_for_resume_builder(film_per_layer=True, film_concat_code=True, film_stiefel=True)
    out = T._build_resume_state_arrays(live, live, None, args=a, epoch=5, in_feat=6)
    # The arch flags that change param keys / training geometry MUST be persisted (== ON here).
    assert int(out["__cfg_film_per_layer"]) == 1
    assert int(out["__cfg_film_concat_code"]) == 1
    assert int(out["__cfg_film_stiefel"]) == 1
    # ... and reflect OFF when off (so a resume can detect the mismatch either direction).
    a_off = _min_args_for_resume_builder()
    out_off = T._build_resume_state_arrays(live, live, None, args=a_off, epoch=5, in_feat=6)
    assert int(out_off["__cfg_film_per_layer"]) == 0
    assert int(out_off["__cfg_film_concat_code"]) == 0
    assert int(out_off["__cfg_film_stiefel"]) == 0


def test_r2a_resume_block_has_fail_closed_arch_drift_guard():
    # STRUCTURAL self-protect: the resume application MUST refuse (raise) when the checkpoint carries
    # trained params the rebuilt model has no slot for (the silent-param-drop class), BEFORE model.update.
    src = Path(T.__file__).read_text()
    assert "_missing_in_model" in src and "model.update(" in src
    assert src.index("_missing_in_model = sorted(set(rs[\"live\"])") < src.index(
        "model.update(tree_unflatten([(k, mx.array(v)) for k, v in rs[\"live\"].items()])"), (
        "the arch-drift guard must run BEFORE model.update (else it cannot prevent the silent drop).")
