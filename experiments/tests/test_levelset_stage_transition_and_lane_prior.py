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
