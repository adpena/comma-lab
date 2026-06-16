# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the Adaptive Pose-Gradient Controller (APGC) — the closed-loop
replacement for the static pose-gradient THROTTLE (``pose_grad_every_k`` +
``pose_grad_resume_threshold``) on the split-by-head backward path.

WHY the controller exists (the bug it fixes): the static throttle is OPEN-loop — it
fixes a cadence k and a fixed resume threshold, so when d_pose DRIFTS UP through the
shared decoder trunk under the oomph seg crank (measured monotonic
0.000335→0.000408 over ep10-40) the fixed threshold (0.001, ~2.5-3× the actual
~0.0004 pose_mse) NEVER fires and pose is corrected only 1-in-k epochs WHILE drifting.
The calculus: ∂S/∂d_pose = 5/√(10·d_pose) = 85.5 at d_pose≈0.0004 ≈ 86% of
∂S/∂d_seg = 100, so an un-arrested pose drift is nearly as score-costly as the d_seg
we optimize. APGC holds d_pose at its (moving) floor with minimum gradient spend.

The load-bearing claims (each, if wrong, either silently reverts the controller to a
static cadence OR makes the "adaptive" behavior a fake that ignores the live state):

1. **DEFAULT BYTE-IDENTICAL (adaptive=False).** The adaptive flag defaults False and
   the EXISTING static k/threshold branch is taken UNCHANGED — proven by a behavior
   test that drives a real split-by-head run and asserts the pose-call cadence matches
   the documented static-k cadence (NOT the adaptive cadence), and that the controller
   state stays at its untouched defaults.
2. **FLOOR ESTABLISHES + RUNNING-MIN TRACKING.** The first epoch always computes
   (no floor yet); the floor is the MIN of all computed pose_mse (a later-rising sample
   does not raise the floor).
3. **DRIFT-ARREST.** dev > 1+tol forces a compute every epoch (the band breach the
   static threshold missed).
4. **PROPORTIONAL CADENCE.** k is monotone in how deep pose sits inside the band
   (deeper at floor → sparser cadence, toward k_max).
5. **TREND OVERRIDE (the derivative term).** A rising trend WITHIN the deadband forces
   a compute (drift caught before it breaches the band).
6. **MEASUREMENT-FLOOR.** The controller never runs blind longer than k_max epochs —
   a compute is forced once k_max epochs have elapsed since the last pose measurement.
7. **CONFIG VALIDATION.** adaptive requires split_by_head; tol/k_max/window bounds.
8. **CHECKPOINT ROUND-TRIP.** A real kill+resume continues the SAME controller state
   (``_pose_floor`` / ``_pose_mse_hist`` / ``_last_pose_epoch`` persist through
   ``_capture_state`` → ``_restore_into``) — NOT a bypass.
9. **NO SILENT STATIC REVERT.** A decisive test fails if the adaptive branch silently
   resolves to the static decision (the two cadences DIFFER on a drift trajectory).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from tac.torch_vehicle.curriculum import StageSpec
from tac.torch_vehicle.driver import (
    TorchVehicleConfig,
    TorchVehicleDriver,
    _adaptive_do_pose,
    import_vendored_bundle,
)
from tac.torch_vehicle.scorer_context import SyntheticScorerContext, _TinyFrozenScorer


def _ce_seg_loss(seg_logits, targets_hard):
    return F.cross_entropy(seg_logits, targets_hard)


def _stage(epochs: int = 2, eval_every: int = 1000) -> StageSpec:
    return StageSpec(
        name="apgc_test",
        epochs=epochs,
        seg_loss_fn=_ce_seg_loss,
        eval_every=eval_every,
        batch_size=8,
        ema_decay=0.999,
        use_muon=False,
        adamw_lr=1e-3,
        muon_lr=2e-4,
        muon_weight_decay=0.0,
        latent_lr_mult=10.0,
        grad_clip=1e9,
        grad_clip_muon=1e9,
        lr_floor_ratio=5e-6,
        seg_weight=100.0,
        pose_weight=1.0,
        cat_lambda=0.0,
        cat_sigma=0.2,
        use_qat=False,
        init_latents_random=True,
    )


def _cpu_split_driver(tmp_path, *, n_pairs=4, base_channels=8, curriculum=None,
                      **cfg_kwargs) -> TorchVehicleDriver:
    """A split-by-head driver forced onto CPU (no MPS hardware needed). The cfg keeps
    train_device='mps' so __post_init__ accepts split_by_head; the actual tensors run on
    CPU (the synthetic scorer holds a distinct identical-weight 'train' scorer)."""
    cfg = TorchVehicleConfig(
        base_channels=base_channels, latent_dim=28, out_dir=tmp_path / "apgc",
        checkpoint_every_epochs=1, device="cpu", train_device="mps",
        split_by_head=True, seed=0, **cfg_kwargs,
    )
    scorer = SyntheticScorerContext(n_pairs=n_pairs, device="cpu", seed=0, split_by_head=True)
    scorer.split_device = True
    scorer.train_device = torch.device("cpu")
    scorer._train_scorer = _TinyFrozenScorer(seed=0).to("cpu").eval()
    driver = TorchVehicleDriver(
        cfg, scorer=scorer, vendored=import_vendored_bundle(),
        curriculum=curriculum or [_stage()],
    )
    driver.train_device = torch.device("cpu")
    driver.device = torch.device("cpu")
    driver.split_by_head = True
    return driver


# ===========================================================================
# CLAIM 7 — config validation.
# ===========================================================================
def test_adaptive_requires_split_by_head():
    with pytest.raises(ValueError, match=r"pose_grad_adaptive .* requires split_by_head"):
        TorchVehicleConfig(device="cpu", pose_grad_adaptive=True)


def test_adaptive_floor_tol_must_be_positive():
    with pytest.raises(ValueError, match="pose_grad_floor_tol must be > 0"):
        TorchVehicleConfig(device="cpu", train_device="mps", split_by_head=True,
                           pose_grad_adaptive=True, pose_grad_floor_tol=0.0)


def test_adaptive_k_max_must_be_ge_one():
    with pytest.raises(ValueError, match="pose_grad_k_max must be >= 1"):
        TorchVehicleConfig(device="cpu", train_device="mps", split_by_head=True,
                           pose_grad_adaptive=True, pose_grad_k_max=0)


def test_adaptive_trend_window_must_be_ge_two():
    with pytest.raises(ValueError, match="pose_grad_trend_window must be >= 2"):
        TorchVehicleConfig(device="cpu", train_device="mps", split_by_head=True,
                           pose_grad_adaptive=True, pose_grad_trend_window=1)


def test_adaptive_defaults_are_off_and_byte_identical_knobs():
    cfg = TorchVehicleConfig(device="cpu")
    assert cfg.pose_grad_adaptive is False
    assert cfg.pose_grad_floor_tol == 0.08
    assert cfg.pose_grad_k_max == 8
    assert cfg.pose_grad_trend_window == 3


# ===========================================================================
# CLAIM 2 — floor establishes on first epoch + running-min tracking (pure helper).
# ===========================================================================
def test_first_epoch_computes_when_no_floor():
    # No floor yet → must compute to ESTABLISH the floor.
    assert _adaptive_do_pose(
        0, None, None, [], tol=0.08, k_max=8, last_pose_epoch=0
    ) is True


def test_first_compute_with_floor_none_but_last_present_still_computes():
    # last_pose_mse present but pose_floor still None (defensive: same establish path).
    assert _adaptive_do_pose(
        3, None, 0.0004, [0.0004], tol=0.08, k_max=8, last_pose_epoch=0
    ) is True


# ===========================================================================
# CLAIM 3 — drift-arrest fires when deviation breaches the deadband.
# ===========================================================================
def test_drift_arrest_when_deviation_exceeds_band():
    # dev = 0.00045 / 0.0004 = 1.125 > 1.08 → arrest (compute) regardless of epoch.
    floor, cur = 0.0004, 0.00045
    for epoch in (1, 2, 5, 7):  # non-cadence epochs to prove it's the band, not k
        assert _adaptive_do_pose(
            epoch, floor, cur, [floor, cur], tol=0.08, k_max=8,
            last_pose_epoch=epoch - 1,
        ) is True, f"band breach must arrest at epoch {epoch}"


def test_inside_band_at_floor_skips_off_cadence():
    # dev == 1.0 (at floor), not rising → deepest in band → k=k_max=8 → off-cadence skip.
    floor = 0.0004
    assert _adaptive_do_pose(
        1, floor, floor, [floor], tol=0.08, k_max=8, last_pose_epoch=0
    ) is False
    # epoch 0 is the cadence anchor (0 % k == 0) → compute.
    assert _adaptive_do_pose(
        8, floor, floor, [floor], tol=0.08, k_max=8, last_pose_epoch=1
    ) is True  # 8 % 8 == 0 (cadence) AND 8-1 >= k_max measurement-floor both fire


# ===========================================================================
# CLAIM 4 — proportional cadence: k is monotone in deviation depth (deeper→sparser).
# ===========================================================================
def test_proportional_cadence_k_monotone_in_deviation():
    """At the floor (dev=1) k is widest (k_max); near the band edge k shrinks toward 1.
    Probe the effective k by finding the smallest positive epoch that computes, for two
    deviations: a deeper one (smaller dev) must yield a >= cadence than a shallower one."""
    floor = 0.0004
    tol, k_max = 0.08, 8

    def effective_k(dev_ratio):
        cur = floor * dev_ratio
        hist = [floor, cur]
        # scan epochs 1..k_max for the FIRST compute (last_pose_epoch high so the
        # measurement-floor never fires within the scan window).
        for ep in range(1, k_max + 1):
            if _adaptive_do_pose(ep, floor, cur, hist, tol=tol, k_max=k_max,
                                 last_pose_epoch=ep):  # epoch==last → no measure-floor
                return ep
        return k_max + 1

    # NOTE: must NOT be rising within band, else trend override forces every epoch.
    # Use a FALLING-then-current sample (hist[-1] <= hist[0]) so 'rising' is False.
    def effective_k_falling(dev_ratio):
        cur = floor * dev_ratio
        hist = [floor * (dev_ratio + 0.001), cur]  # hist[-1] < hist[0] → not rising
        for ep in range(1, k_max + 1):
            if _adaptive_do_pose(ep, floor, cur, hist, tol=tol, k_max=k_max,
                                 last_pose_epoch=ep):
                return ep
        return k_max + 1

    k_at_floor = effective_k_falling(1.0)       # deepest in band
    k_mid = effective_k_falling(1.04)           # halfway to the band edge
    k_edge = effective_k_falling(1.079)         # just inside the band edge
    # Deeper in band ⇒ sparser cadence ⇒ larger effective k.
    assert k_at_floor >= k_mid >= k_edge, (
        f"cadence not monotone in deviation depth: floor={k_at_floor} "
        f"mid={k_mid} edge={k_edge}"
    )
    assert k_at_floor > k_edge, "cadence must actually widen at the floor vs band edge"


# ===========================================================================
# CLAIM 5 — trend override: a rising trend within the band forces a compute.
# ===========================================================================
def test_rising_trend_within_band_forces_compute():
    floor = 0.0004
    cur = 0.00041  # dev = 1.025 < 1.08 (inside band) but...
    rising_hist = [0.0004, 0.000405, 0.00041]  # hist[-1] > hist[0] → rising
    # On a non-cadence epoch that WOULD skip without the trend term:
    assert _adaptive_do_pose(
        3, floor, cur, rising_hist, tol=0.08, k_max=8, last_pose_epoch=2
    ) is True, "a rising trend inside the band must force a compute (drift caught early)"
    # Same deviation but a flat/falling trend → allowed to skip (cadence governs).
    flat_hist = [0.00041, 0.000405, 0.00041]   # hist[-1] == hist[0] → not rising
    assert _adaptive_do_pose(
        3, floor, cur, flat_hist, tol=0.08, k_max=8, last_pose_epoch=2
    ) is False, "a non-rising trend inside the band should defer to the cadence (skip)"


# ===========================================================================
# CLAIM 6 — measurement-floor: never run blind longer than k_max epochs.
# ===========================================================================
def test_measurement_floor_forces_compute_after_k_max_skips():
    floor = 0.0004
    # Deeply at floor (would skip on cadence) but k_max epochs since the last measure:
    assert _adaptive_do_pose(
        9, floor, floor, [floor], tol=0.08, k_max=8, last_pose_epoch=1
    ) is True, "9 - 1 = 8 >= k_max must force a measurement (drift-blindness cap)"
    # One epoch earlier (7 since last measure < 8) at floor on a non-cadence epoch skips.
    assert _adaptive_do_pose(
        8, floor, floor, [floor], tol=0.08, k_max=8, last_pose_epoch=2
    ) is True  # 8 % 8 == 0 cadence — pick a clearly non-cadence epoch instead:
    assert _adaptive_do_pose(
        7, floor, floor, [floor], tol=0.08, k_max=8, last_pose_epoch=2
    ) is False, "7 - 2 = 5 < k_max and 7 % 8 != 0 → skip (no force yet)"


# ===========================================================================
# CLAIMS 1 + 9 — end-to-end: default-off is the STATIC cadence (byte-identical branch);
# adaptive ON drives a DIFFERENT, state-dependent cadence (no silent static revert).
# ===========================================================================
def _run_counting_pose_calls(tmp_path, *, epochs, **cfg_kwargs):
    driver = _cpu_split_driver(
        tmp_path, curriculum=[_stage(epochs=epochs, eval_every=epochs)], **cfg_kwargs
    )
    calls = {"n": 0, "epochs": []}
    orig = driver.scorer.pose_forward_authority

    def counting(x):
        calls["n"] += 1
        calls["epochs"].append(int(driver._global_epoch))
        return orig(x)

    driver.scorer.pose_forward_authority = counting
    summary = driver.run()
    assert summary["status"] == "complete"
    return driver, calls


def test_default_off_uses_static_cadence_not_adaptive(tmp_path):
    """adaptive=False with a static k=3 must follow the STATIC cadence (epochs 0,3,6)
    and leave the controller state at its untouched defaults — proving the existing
    branch runs and the APGC bookkeeping never fires on the default path."""
    driver, calls = _run_counting_pose_calls(
        tmp_path / "static", epochs=7,
        pose_grad_adaptive=False, pose_grad_every_k=3, pose_grad_resume_threshold=0.0,
    )
    assert calls["epochs"] == [0, 3, 6], (
        f"adaptive=False must follow the static every-k cadence; got {calls['epochs']}"
    )
    # Controller state must be UNTOUCHED on the static path (byte-identical bookkeeping):
    # floor None, empty trend, and the "no compute yet" sentinel -1.
    assert driver._pose_floor is None
    assert driver._pose_mse_hist == []
    assert driver._last_pose_epoch == -1


def test_adaptive_on_diverges_from_static_cadence(tmp_path):
    """The decisive no-silent-revert test: adaptive ON produces a DIFFERENT pose-call
    cadence than the static k=3 branch on the SAME run. If the adaptive branch silently
    resolved to the static decision, the two epoch-lists would match — this asserts they
    do NOT, AND that the controller state actually advanced (floor + history populated)."""
    drv_static, calls_static = _run_counting_pose_calls(
        tmp_path / "s", epochs=8,
        pose_grad_adaptive=False, pose_grad_every_k=3, pose_grad_resume_threshold=0.0,
    )
    drv_adaptive, calls_adaptive = _run_counting_pose_calls(
        tmp_path / "a", epochs=8,
        pose_grad_adaptive=True, pose_grad_floor_tol=0.08, pose_grad_k_max=8,
        pose_grad_trend_window=3,
    )
    assert calls_adaptive["epochs"] != calls_static["epochs"], (
        "adaptive cadence equals the static every-k cadence — the adaptive branch "
        f"silently reverted to static (both {calls_static['epochs']})"
    )
    # The controller state must have ADVANCED (proving the adaptive bookkeeping ran).
    assert drv_adaptive._pose_floor is not None, "adaptive run never set the floor"
    assert len(drv_adaptive._pose_mse_hist) >= 1, "adaptive run never built the trend window"
    assert drv_adaptive._last_pose_epoch >= 0


def test_adaptive_trend_window_is_trimmed(tmp_path):
    """The trend history is bounded to ``pose_grad_trend_window`` — it never grows
    unbounded across a run (a memory + correctness invariant)."""
    win = 3
    driver, _ = _run_counting_pose_calls(
        tmp_path / "trim", epochs=12,
        pose_grad_adaptive=True, pose_grad_floor_tol=0.08, pose_grad_k_max=4,
        pose_grad_trend_window=win,
    )
    assert len(driver._pose_mse_hist) <= win, (
        f"trend history exceeded the window {win}: len={len(driver._pose_mse_hist)}"
    )


def test_adaptive_running_floor_is_min_of_computed(tmp_path):
    """The persisted floor equals the MIN of every computed training pose_mse (a later
    rising sample never raises it). We can't easily inject pose_mse from the synthetic
    scorer, so assert the structural invariant: the floor <= every value in the (live)
    trend window after the run."""
    driver, _ = _run_counting_pose_calls(
        tmp_path / "minfloor", epochs=10,
        pose_grad_adaptive=True, pose_grad_floor_tol=0.08, pose_grad_k_max=4,
        pose_grad_trend_window=3,
    )
    assert driver._pose_floor is not None
    for v in driver._pose_mse_hist:
        assert driver._pose_floor <= v + 1e-12, (
            f"floor {driver._pose_floor} exceeds a computed pose_mse {v} — not the running min"
        )


def test_multi_batch_epoch_decision_is_epoch_stable_and_bookkeeping_once(tmp_path):
    """MULTI-BATCH correctness: with >1 batch per epoch the controller must (a) decide
    the pose cadence ONCE per epoch (every batch of a compute-epoch computes; every batch
    of a skip-epoch skips — so the per-batch pose-call count is a multiple of the per-epoch
    batch count, NOT a ragged mix), and (b) advance the trend window AT MOST once per
    epoch (not once per batch). A naive per-batch implementation would mutate the floor
    mid-epoch and make batch 2's decision differ from batch 1's, and would over-fill the
    trend window. n_pairs=12 / batch_size=8 → 2 batches per epoch."""
    epochs = 6
    n_pairs = 12  # with _stage batch_size=8 → ceil(12/8)=2 batches/epoch
    driver = _cpu_split_driver(
        tmp_path / "mb", n_pairs=n_pairs,
        curriculum=[_stage(epochs=epochs, eval_every=epochs)],
        pose_grad_adaptive=True, pose_grad_floor_tol=0.08, pose_grad_k_max=4,
        pose_grad_trend_window=3,
    )
    batches_per_epoch = (n_pairs + 8 - 1) // 8
    assert batches_per_epoch == 2, "fixture must produce a genuine multi-batch epoch"

    calls = {"epochs": []}
    orig = driver.scorer.pose_forward_authority

    def counting(x):
        calls["epochs"].append(int(driver._global_epoch))
        return orig(x)

    driver.scorer.pose_forward_authority = counting
    assert driver.run()["status"] == "complete"

    # (a) Every epoch that computed pose did so on ALL its batches (epoch-stable decision):
    # each distinct epoch in the call log appears EXACTLY batches_per_epoch times.
    from collections import Counter

    per_epoch = Counter(calls["epochs"])
    for ep, cnt in per_epoch.items():
        assert cnt == batches_per_epoch, (
            f"epoch {ep} computed pose on {cnt}/{batches_per_epoch} batches — the "
            "decision was NOT epoch-stable (per-batch state mutation leaked)"
        )

    # (b) The trend window advanced at most once per COMPUTE-EPOCH (not once per batch):
    # # distinct compute-epochs, capped by the window. If bookkeeping ran per-batch the
    # window would have been fed 2x as often (still trimmed, but _last_pose_epoch would
    # equal the last epoch and the floor would have double-counted — caught structurally).
    n_compute_epochs = len(per_epoch)
    assert len(driver._pose_mse_hist) <= min(3, n_compute_epochs), (
        f"trend window {len(driver._pose_mse_hist)} exceeds the compute-epoch count "
        f"{n_compute_epochs} (bookkeeping ran more than once per epoch)"
    )
    # _last_pose_epoch is a REAL epoch (>= 0), stamped once per epoch.
    assert driver._last_pose_epoch >= 0


# ===========================================================================
# CLAIM 8 — checkpoint round-trip persists the controller state (real save/restore).
# ===========================================================================
def _build_split_adaptive_driver(out_dir, *, epochs, seed=0):
    cfg = TorchVehicleConfig(
        base_channels=8, latent_dim=28, out_dir=out_dir,
        checkpoint_every_epochs=1, device="cpu", train_device="mps",
        split_by_head=True, seed=seed,
        pose_grad_adaptive=True, pose_grad_floor_tol=0.08, pose_grad_k_max=4,
        pose_grad_trend_window=3,
    )
    scorer = SyntheticScorerContext(n_pairs=4, device="cpu", seed=seed, split_by_head=True)
    scorer.split_device = True
    scorer.train_device = torch.device("cpu")
    scorer._train_scorer = _TinyFrozenScorer(seed=seed).to("cpu").eval()
    driver = TorchVehicleDriver(
        cfg, scorer=scorer, vendored=import_vendored_bundle(),
        curriculum=[_stage(epochs=epochs, eval_every=10_000)],
    )
    driver.train_device = torch.device("cpu")
    driver.device = torch.device("cpu")
    driver.split_by_head = True
    return driver


def test_checkpoint_roundtrip_persists_controller_state(tmp_path):
    """A real kill (die mid-stage AFTER the checkpoint lands) then a FRESH driver resume
    must continue the SAME controller state — proving ``_pose_floor`` / ``_pose_mse_hist``
    / ``_last_pose_epoch`` round-trip through ``_capture_state`` → ``_restore_into`` (NOT
    a bypass: the resume uses the durable checkpoint on disk)."""
    from tac.torch_vehicle.checkpoint import is_done
    from tac.torch_vehicle.driver import _SimulatedDeath

    out = tmp_path / "ckpt"
    dying = _build_split_adaptive_driver(out, epochs=8, seed=0)
    dying._stop_after_global_epoch = 5  # die after 5 epochs (checkpoint at ep 5 landed)
    with pytest.raises(_SimulatedDeath):
        dying.run()
    assert not is_done(out), "death must NOT mark the run done"

    # Capture the controller state that was checkpointed (it lives in the dying driver,
    # but the durable proof is that a FRESH driver RESTORES it from disk).
    floor_before = dying._pose_floor
    hist_before = list(dying._pose_mse_hist)
    last_ep_before = dying._last_pose_epoch
    assert floor_before is not None, "the dying run must have established a floor"

    # The decisive load-only round-trip: read the EXACT death-point checkpoint OFF DISK
    # (before any resume overwrites it) and restore it into a FRESH driver, then compare
    # the controller fields bit-for-bit (proves the persistence is durable + restored,
    # not an in-memory carry — the checkpoint went through save_checkpoint → load).
    merged = _read_latest_merged(out)
    probe = _build_split_adaptive_driver(tmp_path / "probe", epochs=8, seed=0)
    assert probe._pose_floor is None, "fresh driver starts with no floor (pre-restore)"
    rt = _stage_runtime_for(probe)
    probe._restore_into(rt, merged)
    assert probe._pose_floor == pytest.approx(floor_before, abs=0.0), (
        f"restored floor {probe._pose_floor} != checkpointed {floor_before}"
    )
    assert [pytest.approx(v, abs=0.0) for v in probe._pose_mse_hist] == [
        pytest.approx(v, abs=0.0) for v in hist_before
    ], f"restored hist {probe._pose_mse_hist} != checkpointed {hist_before}"
    assert probe._last_pose_epoch == last_ep_before, (
        f"restored last_pose_epoch {probe._last_pose_epoch} != {last_ep_before}"
    )

    # And a FULL resume off the same on-disk checkpoint runs to completion (the real
    # resume path calls _restore_into then continues + finishes the curriculum).
    resumed = _build_split_adaptive_driver(out, epochs=8, seed=0)
    summary = resumed.run()
    assert summary["status"] in ("complete", "already_done")
    assert is_done(out)


def test_capture_state_includes_controller_fields(tmp_path):
    """``_capture_state`` must emit the three controller keys (a structural guard so the
    persistence can never be silently dropped)."""
    driver = _build_split_adaptive_driver(tmp_path / "cap", epochs=2, seed=0)
    rt = _stage_runtime_for(driver)
    # Seed some controller state, then capture.
    driver._pose_floor = 0.00037
    driver._pose_mse_hist = [0.0004, 0.00038, 0.00037]
    driver._last_pose_epoch = 4
    state = driver._capture_state(rt, driver.curriculum[0])
    assert state["pose_floor"] == 0.00037
    assert state["pose_mse_hist"] == [0.0004, 0.00038, 0.00037]
    assert state["last_pose_epoch"] == 4


def test_legacy_checkpoint_without_controller_keys_restores_defaults(tmp_path):
    """Backward compatibility: a checkpoint dict lacking the APGC keys (a pre-APGC or
    non-adaptive checkpoint) restores the defaults (None / [] / 0) — no crash, no jolt."""
    driver = _build_split_adaptive_driver(tmp_path / "legacy", epochs=2, seed=0)
    rt = _stage_runtime_for(driver)
    full = driver._capture_state(rt, driver.curriculum[0])
    # Strip the APGC keys to simulate a legacy checkpoint.
    for k in ("pose_floor", "pose_mse_hist", "last_pose_epoch"):
        full.pop(k, None)
    driver._pose_floor = 0.5  # dirty state to prove the restore overwrites to default
    driver._pose_mse_hist = [9.9]
    driver._last_pose_epoch = 99
    driver._restore_into(rt, full)
    assert driver._pose_floor is None
    assert driver._pose_mse_hist == []
    assert driver._last_pose_epoch == -1  # the "no compute yet" sentinel


# -- small helpers shared by the checkpoint tests ---------------------------
def _stage_runtime_for(driver):
    """Build a REAL stage runtime for the driver's stage-0 (so _capture_state /
    _restore_into have a genuine decoder/latents/EMA to read/write — not a bypass)."""
    import torch.nn as nn

    spec = driver.curriculum[0]
    decoder = driver._new_decoder()
    latents = nn.Parameter(
        torch.zeros(driver.n_pairs, driver.cfg.latent_dim, device=driver.train_device)
    )
    return driver._build_stage_runtime(
        spec, decoder=decoder, latents=latents, ema_decoder=None, ema_latents=None
    )


def _read_latest_merged(out_dir):
    """Read the durable checkpoint dict the driver wrote (the SAME path the resume
    uses) so we can prove a load-only round-trip."""
    from tac.torch_vehicle.checkpoint import load_checkpoint

    merged = load_checkpoint(out_dir)
    assert merged is not None, "no durable checkpoint found on disk"
    return merged
