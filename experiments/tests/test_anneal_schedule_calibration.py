# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the ANNEAL-SCHEDULE CALIBRATION probe (#119) + the new
``curriculum.seg_temperature_margin_adaptive`` schedule fn.

These tests assert BEHAVIOR, not constants (Slot EEE Class 2 discipline):
  * the margin-adaptive schedule fn actually computes ``clamp(median, floor, max)`` —
    including the resonance ``T* = Δ`` mid-band passthrough, the dead-zone floor, and the
    untargeted cap;
  * the FOUR arms produce DISTINCT per-step T trajectories (static / cosine / fast-cool /
    margin-adaptive are not the same schedule under different names);
  * the margin-adaptive arm's T actually TRACKS the measured live flip-margin median
    (not a constant) — fed a controlled descending median sequence, T descends with it
    (clamped), and τ is SLAVED to T;
  * d_seg is measured on the REAL SegNet argmax-flip set (the contest metric), not a
    constant — a decoder change that flips pixels changes the measured d_seg;
  * default-preservation: ``seg_temperature_for_epoch`` is byte-identical for the
    static/cosine/fast-cool arms (the library extension added a NEW fn, it did not touch
    the existing one).

The heavy real-scorer arms (the full 4-arm window) are run by the probe itself
synchronously; these tests cover the schedule MATH + the measurement-is-real invariants
at $0 with tiny synthetic tensors + (one) real-SegNet-on-a-tiny-frame check guarded by
artifact availability.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

import experiments.probe_anneal_schedule_calibration as P
from tac.torch_vehicle.curriculum import (
    SEG_ANNEAL_MARGIN_ADAPTIVE_T_FLOOR,
    SEG_ANNEAL_MARGIN_ADAPTIVE_T_MAX,
    seg_temperature_for_epoch,
    seg_temperature_margin_adaptive,
    seg_temperature_weighted_mean_resonant,
)
from tac.torch_vehicle.vendored_imports import import_vendored

_CE = import_vendored("stages.stage1_v328_ce").ce_seg_loss


def _spec(**kw):
    """A minimal soft_cosine seg-only spec for the schedule-fn tests."""
    base = dict(
        margin_weight_tau=0.3,
        seg_weight=100.0,
        adamw_lr=1e-3,
        grad_clip=1.0,
        n_steps=8,
        seg_temperature=1.0,
        seg_temperature_end=None,
        seg_temperature_hold_frac=None,
        ce_seg_loss=_CE,
    )
    base.update(kw)
    return P._make_spec(**base)


# ---------------------------------------------------------------------------
# (1) margin-adaptive schedule fn: the clamp(median, floor, max) BEHAVIOR
# ---------------------------------------------------------------------------
def test_margin_adaptive_midband_is_passthrough_resonance():
    """In the gradient-alive band [0.3, 0.6] the temperature EQUALS the median margin
    (the T* = Δ resonance) — a measured median of 0.45 yields T=0.45, NOT a constant."""
    assert seg_temperature_margin_adaptive(0.45) == pytest.approx(0.45)
    assert seg_temperature_margin_adaptive(0.31) == pytest.approx(0.31)
    assert seg_temperature_margin_adaptive(0.59) == pytest.approx(0.59)


def test_margin_adaptive_floors_at_dead_zone():
    """Below the gradient-alive knee the temperature is FLOORED at 0.3 (never enters the
    R12 dead zone where e^{-Δ/T} -> 0)."""
    assert seg_temperature_margin_adaptive(0.1) == pytest.approx(SEG_ANNEAL_MARGIN_ADAPTIVE_T_FLOOR)
    assert seg_temperature_margin_adaptive(0.0) == pytest.approx(SEG_ANNEAL_MARGIN_ADAPTIVE_T_FLOOR)
    assert seg_temperature_margin_adaptive(-5.0) == pytest.approx(SEG_ANNEAL_MARGIN_ADAPTIVE_T_FLOOR)


def test_margin_adaptive_caps_at_untargeted_max():
    """Above the cap the temperature is CAPPED at 0.6 (never warms into the untargeted
    regime that lost to CE at T=1.0)."""
    assert seg_temperature_margin_adaptive(0.9) == pytest.approx(SEG_ANNEAL_MARGIN_ADAPTIVE_T_MAX)
    assert seg_temperature_margin_adaptive(10.0) == pytest.approx(SEG_ANNEAL_MARGIN_ADAPTIVE_T_MAX)


def test_margin_adaptive_nan_returns_floor():
    """A non-finite median (no flips this step) returns the safe gradient-alive floor."""
    assert seg_temperature_margin_adaptive(float("nan")) == pytest.approx(SEG_ANNEAL_MARGIN_ADAPTIVE_T_FLOOR)
    assert seg_temperature_margin_adaptive(float("inf")) == pytest.approx(SEG_ANNEAL_MARGIN_ADAPTIVE_T_FLOOR)


def test_margin_adaptive_custom_window_respected():
    """The floor/max window is honored (not hardcoded): a tighter window clamps tighter."""
    assert seg_temperature_margin_adaptive(0.45, t_floor=0.5, t_max=0.55) == pytest.approx(0.5)
    assert seg_temperature_margin_adaptive(0.7, t_floor=0.5, t_max=0.55) == pytest.approx(0.55)


def test_margin_adaptive_rejects_inverted_window():
    """t_max < t_floor is a programming error -> ValueError (fail-closed)."""
    with pytest.raises(ValueError):
        seg_temperature_margin_adaptive(0.4, t_floor=0.6, t_max=0.3)


# ---------------------------------------------------------------------------
# (2) the FOUR arms are DISTINCT per-step T trajectories
# ---------------------------------------------------------------------------
def test_static_arm_holds_constant_temperature():
    s = _spec(seg_temperature=0.3, seg_temperature_end=None)
    temps = [seg_temperature_for_epoch(s, i) for i in range(8)]
    assert temps == [pytest.approx(0.3)] * 8


def test_cosine_arm_descends_high_to_cold():
    s = _spec(seg_temperature=1.0, seg_temperature_end=0.05)
    temps = [seg_temperature_for_epoch(s, i) for i in range(8)]
    assert temps[0] == pytest.approx(1.0)
    assert temps[-1] == pytest.approx(0.05)
    # monotone non-increasing
    assert all(temps[i] >= temps[i + 1] - 1e-9 for i in range(len(temps) - 1))
    # spends the EARLY window WARM (the miscalibration): step 2 still >= 0.5
    assert temps[2] >= 0.5
    # and reaches the DEAD zone late (the other miscalibration): last step < 0.3
    assert temps[-1] < SEG_ANNEAL_MARGIN_ADAPTIVE_T_FLOOR


def test_fastcool_arm_warms_then_holds_at_floor():
    s = _spec(seg_temperature=1.0, seg_temperature_end=0.3, seg_temperature_hold_frac=0.3)
    temps = [seg_temperature_for_epoch(s, i) for i in range(8)]
    assert temps[0] == pytest.approx(1.0)  # brief warm phase for from-0 stability
    # then HOLDS at 0.3 for the rest (no descent into the dead zone)
    assert all(t == pytest.approx(0.3) for t in temps[2:])
    assert min(temps) == pytest.approx(0.3)


def test_four_arm_trajectories_are_pairwise_distinct():
    """No two of the four schedules produce the same per-step T sequence (they are
    genuinely different schedules, not enum-padding)."""
    static = tuple(round(seg_temperature_for_epoch(_spec(seg_temperature=0.3), i), 4) for i in range(8))
    cosine = tuple(round(seg_temperature_for_epoch(_spec(seg_temperature=1.0, seg_temperature_end=0.05), i), 4) for i in range(8))
    fastcool = tuple(round(seg_temperature_for_epoch(_spec(seg_temperature=1.0, seg_temperature_end=0.3, seg_temperature_hold_frac=0.3), i), 4) for i in range(8))
    # margin-adaptive trajectory depends on data; simulate a descending median sequence
    medians = [0.7, 0.55, 0.45, 0.4, 0.35, 0.32, 0.3, 0.28]
    adaptive = tuple(round(seg_temperature_margin_adaptive(m), 4) for m in medians)
    seqs = {"static": static, "cosine": cosine, "fastcool": fastcool, "adaptive": adaptive}
    labels = list(seqs)
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            assert seqs[labels[i]] != seqs[labels[j]], f"{labels[i]} == {labels[j]} (not distinct)"


def test_arm_schedule_table_has_five_distinct_arms():
    arms = [a[0] for a in P._ARM_SCHEDULES]
    assert len(arms) == 5
    assert len(set(arms)) == 5
    sched_ids = {a[1] for a in P._ARM_SCHEDULES}
    assert sched_ids == {"static", "fast_cool_hold", "cosine", "margin_adaptive", "weighted_mean"}


# ---------------------------------------------------------------------------
# (1b) the CORRECTED weighted-mean-resonant schedule fn (the #119 crux fix)
# ---------------------------------------------------------------------------
def _ref_weighted_mean_fixed_point(margins, t0, n_iter=40):
    """Independent pure-python reference for T* = Σ Δ e^{-Δ/T*} / Σ e^{-Δ/T*} (many iters,
    NOT the library's vectorized 5-iter torch path) — the apples-to-apples NO-FAKE oracle."""
    T = float(t0)
    for _ in range(n_iter):
        ws = [math.exp(-m / T) for m in margins]
        denom = sum(ws)
        if denom <= 0:
            break
        T = sum(m * w for m, w in zip(margins, ws)) / denom
    return T


def test_weighted_mean_matches_independent_fixed_point_reference():
    """The library fn computes the SAME weighted-mean fixed point as an independent
    pure-python iterator (BEHAVIOR, not a constant). Window wide so no clamp interferes."""
    margins = [0.3, 0.31, 0.35, 0.32, 0.33, 1.8, 2.4]  # right-skewed: boundary mass + deep tail
    ref = _ref_weighted_mean_fixed_point(margins, t0=0.5)
    got = seg_temperature_weighted_mean_resonant(
        torch.tensor(margins), t_floor=0.0, t_max=10.0
    )
    assert got == pytest.approx(ref, abs=1e-3)


def test_weighted_mean_sits_below_median_on_right_skewed_distribution():
    """THE CRUX: on a right-skewed flip-margin distribution (boundary mass + deep tail) the
    reachability weight e^{-Δ/T} discounts the deep flips, so T* < MEDIAN — exactly why the
    median-adaptive arm parked too warm and lost slice-0 to the committed-low fast-cool."""
    margins = [0.30, 0.31, 0.33, 0.35, 0.40, 2.0, 2.5, 3.0]  # median = (0.35+0.40)/2 = 0.375
    med = float(torch.tensor(margins).median().item())
    wmean = seg_temperature_weighted_mean_resonant(
        torch.tensor(margins), t_floor=0.0, t_max=10.0  # unclamped to see the true statistic
    )
    assert wmean < med, f"weighted-mean {wmean:.4f} must sit below median {med:.4f} (the crux)"


def test_weighted_mean_equals_value_when_all_margins_equal():
    """Degenerate: every flip at the SAME margin Δ -> weighted mean = Δ (and = the median);
    the two statistics only diverge under spread."""
    got = seg_temperature_weighted_mean_resonant(
        torch.full((50,), 0.42), t_floor=0.0, t_max=10.0
    )
    assert got == pytest.approx(0.42, abs=1e-4)


def test_weighted_mean_clamps_to_alive_band():
    """The assigned T is clamped to [floor, max] AFTER the fixed point (a tiny-margin
    population whose T* < 0.3 is floored to the gradient-alive knee; a deep population whose
    T* > 0.6 is capped)."""
    tiny = seg_temperature_weighted_mean_resonant(torch.full((20,), 0.05))
    assert tiny == pytest.approx(SEG_ANNEAL_MARGIN_ADAPTIVE_T_FLOOR)
    deep = seg_temperature_weighted_mean_resonant(torch.full((20,), 5.0))
    assert deep == pytest.approx(SEG_ANNEAL_MARGIN_ADAPTIVE_T_MAX)


def test_weighted_mean_empty_flip_set_returns_floor():
    """No flips this step -> the safe gradient-alive floor (the caller may hold prior T)."""
    assert seg_temperature_weighted_mean_resonant(torch.empty(0)) == pytest.approx(
        SEG_ANNEAL_MARGIN_ADAPTIVE_T_FLOOR
    )
    assert seg_temperature_weighted_mean_resonant([]) == pytest.approx(
        SEG_ANNEAL_MARGIN_ADAPTIVE_T_FLOOR
    )


def test_weighted_mean_drops_nonfinite_entries():
    """Non-finite margins (numerical degeneracies) are dropped, not propagated to NaN."""
    got = seg_temperature_weighted_mean_resonant(
        torch.tensor([0.4, float("nan"), 0.4, float("inf")]), t_floor=0.0, t_max=10.0
    )
    assert got == pytest.approx(0.4, abs=1e-3)


def test_weighted_mean_rejects_inverted_window():
    with pytest.raises(ValueError):
        seg_temperature_weighted_mean_resonant(torch.tensor([0.4]), t_floor=0.6, t_max=0.3)


def test_weighted_mean_is_self_consistent_fixed_point():
    """THE NEXUS: the returned T* satisfies T* == Σ Δ e^{-Δ/T*} / Σ e^{-Δ/T*} (it equals its
    OWN reachability-weighted mean) — verified by plugging the result back in. Window wide so
    the clamp does not mask the fixed-point property."""
    margins = torch.tensor([0.3, 0.34, 0.5, 0.9, 1.6, 2.7])
    T = seg_temperature_weighted_mean_resonant(margins, t_floor=0.0, t_max=10.0)
    w = torch.exp(-margins / T)
    recomputed = float((margins * w).sum() / w.sum())
    assert T == pytest.approx(recomputed, abs=2e-3), "T* must equal its own weighted mean"


# ---------------------------------------------------------------------------
# (3) margin-adaptive arm actually TRACKS the measured median (+ τ slaving)
# ---------------------------------------------------------------------------
def test_margin_adaptive_tracks_descending_median_into_floor():
    """Fed a DESCENDING median-margin sequence (the §6 collapse), the margin-adaptive T
    DESCENDS with it until it hits the floor, then HOLDS — proving T tracks the live
    median, not a fixed schedule."""
    medians = [0.8, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2]
    temps = [seg_temperature_margin_adaptive(m) for m in medians]
    # in-band region descends 1:1 with the median
    assert temps[1] == pytest.approx(0.6)  # 0.6 capped
    assert temps[2] == pytest.approx(0.5)
    assert temps[3] == pytest.approx(0.45)
    # below floor, holds at the floor (does not chase into the dead zone)
    assert temps[-1] == pytest.approx(SEG_ANNEAL_MARGIN_ADAPTIVE_T_FLOOR)
    assert temps[-2] == pytest.approx(SEG_ANNEAL_MARGIN_ADAPTIVE_T_FLOOR)
    # overall: T is non-increasing while the median is non-increasing
    assert all(temps[i] >= temps[i + 1] - 1e-9 for i in range(len(temps) - 1))


def test_tau_is_slaved_to_temperature_in_adaptive_arm(real_scorer_window):
    """In the live margin-adaptive run, the recorded τ(t) EQUALS T(t) at every step
    (the §7 co-tuning) — NOT a static τ."""
    res = real_scorer_window["margin_adaptive"]
    assert len(res.taus_per_step) == len(res.temps_per_step) > 0
    for tau, temp in zip(res.taus_per_step, res.temps_per_step, strict=True):
        assert tau == pytest.approx(temp), "margin-adaptive τ must be slaved to T"


def test_static_arm_tau_is_held_not_slaved(real_scorer_window):
    """A non-adaptive arm holds τ at the STATIC margin_weight_tau (not data-tracking)."""
    res = real_scorer_window["static_T0.3"]
    assert len(set(round(t, 6) for t in res.taus_per_step)) == 1  # constant τ
    assert res.taus_per_step[0] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# (4) d_seg is measured on the REAL SegNet argmax-flip set, not a constant
# ---------------------------------------------------------------------------
def test_flip_margin_median_is_real_topgap_on_flips():
    """``_flip_margin_median`` returns the median of (top1 − gt_logit) over the EXACT
    flip set — verified against a hand-computed reference on controlled logits."""
    # 1 pixel, gt class 0, pred argmax class 1 with margin 2.0 -> flip, Δ=2.0
    seg = torch.zeros(1, 5, 1, 2)
    seg[0, 0, 0, 0] = 5.0  # pixel 0: argmax=gt(0), NOT a flip
    seg[0, 1, 0, 1] = 3.0  # pixel 1: argmax=1
    seg[0, 0, 0, 1] = 1.0  # gt logit at pixel 1 = 1.0 -> Δ = 3.0 - 1.0 = 2.0 (a flip)
    gt = torch.zeros(1, 1, 2, dtype=torch.long)  # gt class 0 everywhere
    med = P._flip_margin_median(seg, gt)
    assert med == pytest.approx(2.0)  # only pixel 1 flips; its margin is 2.0


def test_flip_margin_median_nan_when_no_flips():
    seg = torch.zeros(1, 5, 2, 2)
    seg[0, 0] = 10.0  # argmax = class 0 everywhere
    gt = torch.zeros(1, 2, 2, dtype=torch.long)  # gt = class 0 -> no flips
    assert math.isnan(P._flip_margin_median(seg, gt))


def test_measure_dseg_changes_when_decoder_changes(real_scorer_window, _real_scorer_fixture):
    """d_seg is a REAL measurement: two DIFFERENT decoder states on the same slice give
    DIFFERENT measured d_seg (a constant would not). Uses the converged forkpoint vs a
    perturbed copy."""
    scorer, wrapper_factory, latents, stored_pose, base_state = _real_scorer_fixture
    idx = torch.arange(0, 4)
    import copy as _copy

    w0 = wrapper_factory()
    w0.load_state_dict(_copy.deepcopy(base_state))
    with torch.no_grad():
        w0.set_stored_pose(stored_pose)
    d0 = P._measure_dseg(w0, latents, idx, scorer).d_seg

    w1 = wrapper_factory()
    pert = _copy.deepcopy(base_state)
    with torch.no_grad():
        # perturb the rgb_1 head (the SegNet frame) so f1 changes -> d_seg must change
        for k in list(pert.keys()):
            if "rgb_1" in k:
                pert[k] = pert[k] + 5.0
    w1.load_state_dict(pert)
    with torch.no_grad():
        w1.set_stored_pose(stored_pose)
    d1 = P._measure_dseg(w1, latents, idx, scorer).d_seg

    assert d0 != d1, "d_seg must respond to a decoder change (not a constant)"
    assert 0.0 <= d0 <= 1.0 and 0.0 <= d1 <= 1.0  # it is a rate


def test_dseg_invariant_to_film_on_clean_rgb1(_real_scorer_fixture):
    """The v2 decoder's f1 (SegNet frame) is FiLM-clean -> d_seg is INVARIANT to the
    stored pose / FiLM (the pose-decoupling the calibration relies on)."""
    scorer, wrapper_factory, latents, _stored_pose, base_state = _real_scorer_fixture
    idx = torch.arange(0, 4)
    import copy as _copy

    w = wrapper_factory()
    w.load_state_dict(_copy.deepcopy(base_state))
    with torch.no_grad():
        w.set_stored_pose(torch.zeros_like(w.stored_pose))
    d_zero = P._measure_dseg(w, latents, idx, scorer).d_seg
    with torch.no_grad():
        torch.manual_seed(1)
        w.set_stored_pose(torch.randn_like(w.stored_pose) * 3.0)
    d_rand = P._measure_dseg(w, latents, idx, scorer).d_seg
    assert d_zero == pytest.approx(d_rand), "d_seg must be FiLM-invariant on clean rgb_1"


# ---------------------------------------------------------------------------
# (5) default-preservation: the library extension did not change the old fn
# ---------------------------------------------------------------------------
def test_seg_temperature_for_epoch_static_unchanged():
    """When seg_temperature_end is None the per-epoch fn is STATIC (byte-identical to the
    pre-extension behavior) — the margin-adaptive addition is a NEW fn, not a mutation."""
    s = _spec(seg_temperature=0.7, seg_temperature_end=None)
    assert all(seg_temperature_for_epoch(s, e) == pytest.approx(0.7) for e in range(8))


def test_seg_temperature_for_epoch_cosine_unchanged():
    """The cosine path is unchanged (clamped to [end, start], cosine progress)."""
    s = _spec(seg_temperature=1.0, seg_temperature_end=0.1)
    t0 = seg_temperature_for_epoch(s, 0)
    tend = seg_temperature_for_epoch(s, 7)
    assert t0 == pytest.approx(1.0)
    assert tend == pytest.approx(0.1)
    # never overshoots the clamp window
    for e in range(8):
        t = seg_temperature_for_epoch(s, e)
        assert 0.1 - 1e-9 <= t <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# Fixtures: the real-scorer window (run ONCE, tiny, synchronously) + the scorer factory
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def _real_scorer_fixture():
    """Build the real frozen SegNet + the v2 wrapper factory from the forkpoint. Skips if
    the forkpoint / video artifacts are not present (CI without the basin)."""
    if not P._FORKPOINT.exists() or not P._VIDEO.exists():
        pytest.skip("forkpoint or video artifact not present (real-scorer tests skipped)")
    import copy as _copy

    ckpt = torch.load(P._FORKPOINT, map_location="cpu", weights_only=False)
    latents = ckpt["ema_latents"].clone().detach()
    n_pairs = latents.shape[0]
    from tac.torch_vehicle.driver import import_vendored_bundle
    from tac.torch_vehicle.pose_film_v2 import PoseFiLMHNeRVWrapperV2, _POSE_DIM

    vb = import_vendored_bundle()

    def wrapper_factory():
        dec = vb.HNeRVDecoder(latent_dim=28, base_channels=20, eval_size=(384, 512)).to("cpu")
        return PoseFiLMHNeRVWrapperV2(dec, n_pairs=n_pairs, pose_dim=_POSE_DIM).to("cpu")

    scorer = P.RealScorerContext(
        video_path=P._VIDEO, device="cpu", max_pairs=8, targets_cache=P._TARGETS_CACHE
    )
    shared = wrapper_factory()
    shared.decoder.load_state_dict(ckpt["ema_decoder"])
    stored_pose = torch.zeros(n_pairs, _POSE_DIM)
    pt = getattr(scorer, "pose_targets", None)
    if pt is not None:
        n = min(int(pt.shape[0]), n_pairs)
        stored_pose[:n] = pt[:n].detach().to(stored_pose.dtype).cpu()
    base_state = _copy.deepcopy(shared.state_dict())
    return scorer, wrapper_factory, latents, stored_pose, base_state


@pytest.fixture(scope="module")
def real_scorer_window(_real_scorer_fixture):
    """Run a TINY (4 pairs, 3 steps) static + margin-adaptive window on the real scorer —
    just enough to assert the τ-slaving + tracking invariants without a long run."""
    scorer, wrapper_factory, latents, stored_pose, base_state = _real_scorer_fixture
    idx = torch.arange(0, 4)
    out = {}
    for arm_label, sched_id, kwargs in P._ARM_SCHEDULES:
        if arm_label not in ("static_T0.3", "margin_adaptive"):
            continue
        out[arm_label] = P._run_arm(
            arm_label=arm_label,
            schedule_id=sched_id,
            schedule_kwargs=kwargs,
            base_wrapper_state=base_state,
            base_latents=latents,
            base_stored_pose=stored_pose,
            scorer=scorer,
            wrapper_factory=wrapper_factory,
            idx=idx,
            n_steps=3,
            seg_weight=100.0,
            adamw_lr=1e-3,
            grad_clip=1.0,
            margin_weight_tau=0.3,
            ce_seg_loss=_CE,
            t_floor=SEG_ANNEAL_MARGIN_ADAPTIVE_T_FLOOR,
            t_max=SEG_ANNEAL_MARGIN_ADAPTIVE_T_MAX,
        )
    return out
