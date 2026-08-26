# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the E#5 stage-transition pose-kick fix (``stage_lr_warmup_frac``).

The MEASURED problem (from the live GREEN run): at each stage→stage boundary PR95
resets the optimizer and the per-stage cosine LR restarts to PEAK (1e-3), which SLAMS
the shared trunk → d_pose spiked 0.00021→0.00142 (6.8×) before recovering, and d_seg
0.00224→0.00287, at the stage-1→2 boundary. 6 more transitions remain.

The fix is the standard warmup-after-restart: the first ``frac·stage_epochs`` epochs of
EACH stage LINEARLY ramp the LR from a small floor up to the cosine value, so the trunk
is eased in instead of slammed.

Test discipline:
* ``stage_lr_warmup_frac`` defaults to 0.0 (OFF) and the lambda + the whole stage runtime
  is BYTE-IDENTICAL to the legacy faithful path (the live basin resuming onto this code
  is unaffected).
* the warmup lambda has the correct SHAPE (epoch 0 = start_ratio; ramps to the cosine
  value at the warmup-end epoch with NO jump; == cosine after).
* HEADLINE: the BOUNDARY trunk perturbation (the L2 of the decoder weight change over the
  FIRST epoch of a fresh stage — the direct CAUSE of the d_pose kick) is MEASURABLY
  SMALLER with warmup>0 than with warmup=0. This test FAILS if the warmup is a no-op.
* the LambdaLR ``last_epoch`` is checkpointed, so a resume mid-warmup continues the same
  ramp position (the warmup is not re-started on resume).

Authority: synthetic scorer (RESEARCH-ONLY, no score claim). The boundary weight-delta
is the FAITHFUL proxy for the d_pose kick: the kick is caused by the LR-driven trunk
weight perturbation; a real-scorer d_pose run is deliberately NOT used here (it would
consume the CPU authority lane the live full 600-pair eval needs — the contention
discipline). ``[contest-CPU advisory]`` NON-PROMOTABLE.
"""
from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from tac.torch_vehicle.curriculum import StageSpec
from tac.torch_vehicle.driver import (
    TorchVehicleConfig,
    TorchVehicleDriver,
    _warmup_wrap,
    import_vendored_bundle,
)
from tac.torch_vehicle.scorer_context import SyntheticScorerContext


def _ce(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(s, t)


def _spec(name: str, epochs: int, adamw_lr: float, **kw) -> StageSpec:
    base = {
        "name": name, "epochs": epochs, "seg_loss_fn": _ce, "eval_every": epochs,
        "batch_size": 4, "ema_decay": 0.999, "use_muon": False, "adamw_lr": adamw_lr,
        "muon_lr": 2e-4, "muon_weight_decay": 0.0, "latent_lr_mult": 10.0,
        "grad_clip": 1e9, "grad_clip_muon": 1e9, "lr_floor_ratio": 5e-6,
        "seg_weight": 100.0, "pose_weight": 1.0, "cat_lambda": 0.0, "cat_sigma": 0.2,
        "use_qat": False, "init_latents_random": True,
    }
    base.update(kw)
    return StageSpec(**base)


def _driver(tmp_path: Path, *, warmup_frac: float, start_ratio: float = 0.1,
            sub: str = "d") -> TorchVehicleDriver:
    cfg = TorchVehicleConfig(
        base_channels=8, latent_dim=28, out_dir=tmp_path / sub, device="cpu", seed=0,
        eval_every=999, checkpoint_every_epochs=999,
        stage_lr_warmup_frac=warmup_frac, stage_lr_warmup_start_ratio=start_ratio,
    )
    scorer = SyntheticScorerContext(n_pairs=8, device="cpu", seed=0)
    return TorchVehicleDriver(
        cfg, scorer=scorer, vendored=import_vendored_bundle(),
        curriculum=[_spec("s", 12, 1e-3)],
    )


# --------------------------------------------------------------------------- #
# 1. defaults + config validation
# --------------------------------------------------------------------------- #
def test_stage_lr_warmup_frac_default_is_zero():
    assert TorchVehicleConfig(device="cpu").stage_lr_warmup_frac == 0.0
    assert TorchVehicleConfig(device="cpu").stage_lr_warmup_start_ratio == 0.1


def test_stage_lr_warmup_frac_out_of_range_rejected():
    with pytest.raises(ValueError, match="stage_lr_warmup_frac"):
        TorchVehicleConfig(device="cpu", stage_lr_warmup_frac=0.6)
    with pytest.raises(ValueError, match="stage_lr_warmup_frac"):
        TorchVehicleConfig(device="cpu", stage_lr_warmup_frac=-0.1)


def test_stage_lr_warmup_start_ratio_out_of_range_rejected():
    with pytest.raises(ValueError, match="stage_lr_warmup_start_ratio"):
        TorchVehicleConfig(device="cpu", stage_lr_warmup_start_ratio=0.0)
    with pytest.raises(ValueError, match="stage_lr_warmup_start_ratio"):
        TorchVehicleConfig(device="cpu", stage_lr_warmup_start_ratio=1.5)


# --------------------------------------------------------------------------- #
# 2. _warmup_wrap unit behavior (the lambda shape)
# --------------------------------------------------------------------------- #
def _cosine_lambda(stage_epochs: int, eta: float = 0.005):
    def f(epoch: int) -> float:
        return max(0.5 * (1 + math.cos(math.pi * epoch / stage_epochs)), eta)
    return f


def test_warmup_off_returns_identical_lambda_object():
    """frac<=0 returns the SAME lambda object → the LambdaLR is byte-identical."""
    base = _cosine_lambda(20)
    assert _warmup_wrap(base, warmup_frac=0.0, stage_epochs=20, start_ratio=0.1) is base
    assert _warmup_wrap(base, warmup_frac=-1.0, stage_epochs=20, start_ratio=0.1) is base


def test_warmup_shape_epoch0_starts_at_start_ratio():
    base = _cosine_lambda(20)
    w = _warmup_wrap(base, warmup_frac=0.25, stage_epochs=20, start_ratio=0.1)
    assert abs(w(0) - 0.1) < 1e-12  # epoch 0 = start_ratio (NOT peak 1.0)
    assert abs(base(0) - 1.0) < 1e-12  # the legacy path WOULD start at peak


def test_warmup_shape_continuous_then_equals_cosine():
    base = _cosine_lambda(20)
    w = _warmup_wrap(base, warmup_frac=0.25, stage_epochs=20, start_ratio=0.1)
    wend = math.ceil(0.25 * 20)  # = 5
    # C0-continuous: warmup ends exactly where the cosine is at the warmup-end epoch.
    assert abs(w(wend) - base(wend)) < 1e-12
    # after the window the wrapped lambda IS the cosine.
    for e in range(wend, 20):
        assert abs(w(e) - base(e)) < 1e-12
    # monotone ramp UP across the window (eased in).
    for e in range(wend):
        assert w(e) <= w(e + 1) + 1e-12


def test_warmup_window_at_least_one_epoch_for_tiny_stage():
    base = _cosine_lambda(1)
    w = _warmup_wrap(base, warmup_frac=0.5, stage_epochs=1, start_ratio=0.1)
    # a 1-epoch stage clamps the window to >= 1 but leaves the function well-defined.
    assert math.isfinite(w(0))


# --------------------------------------------------------------------------- #
# 3. byte-identity of the built stage runtime on the default (OFF) path
# --------------------------------------------------------------------------- #
def test_default_stage_runtime_lr_schedule_is_byte_identical(tmp_path):
    """OFF (the default) ⇒ the AdamW LR sequence across the stage is the legacy cosine
    (epoch 0 at peak), bit-for-bit — the live basin resuming onto this code is unchanged."""
    d = _driver(tmp_path, warmup_frac=0.0, sub="off")
    spec = _spec("s", 12, 1e-3)
    dec = d._new_decoder()
    lat = torch.nn.Parameter(torch.zeros(8, 28))
    rt = d._build_stage_runtime(spec, decoder=dec, latents=lat, ema_decoder=None,
                                ema_latents=None)
    seq = []
    for _ in range(spec.epochs):
        seq.append(rt.adamw_opt.param_groups[0]["lr"])
        rt.adamw_sched.step()
    # legacy cosine reference (no warmup): lr = peak · max(0.5(1+cos(πe/E)), eta).
    eta = max(spec.lr_floor_ratio / spec.adamw_lr, 1e-3)
    ref = [1e-3 * max(0.5 * (1 + math.cos(math.pi * e / spec.epochs)), eta)
           for e in range(spec.epochs)]
    for got, want in zip(seq, ref, strict=True):
        assert abs(got - want) < 1e-12
    # epoch 0 is at PEAK on the default path (this is exactly the kick the fix targets).
    assert abs(seq[0] - 1e-3) < 1e-12


def test_warmup_on_lowers_boundary_epoch_lr(tmp_path):
    """ON ⇒ the FIRST-epoch (boundary) AdamW LR is start_ratio×peak, NOT peak."""
    d = _driver(tmp_path, warmup_frac=0.3, start_ratio=0.1, sub="on")
    spec = _spec("s", 12, 1e-3)
    dec = d._new_decoder()
    lat = torch.nn.Parameter(torch.zeros(8, 28))
    rt = d._build_stage_runtime(spec, decoder=dec, latents=lat, ema_decoder=None,
                                ema_latents=None)
    boundary_lr = rt.adamw_opt.param_groups[0]["lr"]
    assert abs(boundary_lr - 0.1 * 1e-3) < 1e-9, (
        f"boundary LR should be start_ratio×peak = 1e-4; got {boundary_lr:.2e}"
    )


# --------------------------------------------------------------------------- #
# 4. HEADLINE — the boundary trunk-slam (the direct cause of the d_pose kick) is
#    measurably SMALLER with warmup. FAILS if the warmup is a no-op.
# --------------------------------------------------------------------------- #
def _boundary_trunk_weight_delta(tmp_path, *, warmup_frac: float, sub: str) -> float:
    """L2 norm of the decoder weight change over the FIRST epoch of a fresh stage at a
    BIG peak LR — the direct trunk perturbation that slams the shared frame and kicks
    d_pose at the stage boundary. Same seed, same data, same grads → the ONLY difference
    between the two arms is the boundary-epoch LR (peak vs start_ratio×peak)."""
    torch.manual_seed(0)
    d = _driver(tmp_path, warmup_frac=warmup_frac, sub=sub)
    # a fresh stage with a BIG peak LR (the post-restart slam the real curriculum does).
    spec = _spec("s2", 12, 5e-3)
    dec = d._new_decoder()
    lat = torch.nn.Parameter(torch.randn(8, 28) * 0.1)
    rt = d._build_stage_runtime(spec, decoder=dec, latents=lat, ema_decoder=None,
                                ema_latents=None)
    w0 = deepcopy(dec.state_dict())
    d._train_one_epoch(rt, spec, epoch_in_stage=0)
    return sum((dec.state_dict()[k] - w0[k]).pow(2).sum().item() for k in w0) ** 0.5


def test_headline_warmup_reduces_boundary_trunk_slam(tmp_path):
    off = _boundary_trunk_weight_delta(tmp_path, warmup_frac=0.0, sub="kick_off")
    on = _boundary_trunk_weight_delta(tmp_path, warmup_frac=0.3, sub="kick_on")
    assert off > 0.0  # there IS a slam to reduce (the test is non-vacuous)
    assert on < off, (
        f"warmup must REDUCE the boundary trunk slam (the d_pose kick cause): "
        f"on {on:.6f} !< off {off:.6f} — if equal the warmup is a NO-OP"
    )
    # the reduction should track the start_ratio (the boundary LR is 10% of peak), so the
    # first-step weight delta is ~10× smaller (AdamW first-step is ~lr-scaled). Generous
    # bound: at least a 2× reduction (proves it is not a marginal / numerical wobble).
    assert on < 0.5 * off, (
        f"warmup should materially cut the slam (>=2×): on {on:.6f} vs off {off:.6f}"
    )


# --------------------------------------------------------------------------- #
# 5. resume preserves the warmup ramp position (last_epoch is checkpointed)
# --------------------------------------------------------------------------- #
def test_warmup_lr_schedule_state_roundtrips(tmp_path):
    """The LambdaLR state_dict (last_epoch) carries the warmup position, so a resume
    mid-warmup CONTINUES the SAME ramp (not a re-start). The faithful resume contract is
    that the scheduler's REMAINING trajectory (the LRs from the next step onward) matches
    a never-interrupted run — exactly how the driver's checkpoint/restore works."""
    d = _driver(tmp_path, warmup_frac=0.3, sub="resume")
    spec = _spec("s", 12, 1e-3)

    def lr_trajectory(n_pre_steps: int, *, do_resume: bool) -> list[float]:
        rt = d._build_stage_runtime(spec, decoder=d._new_decoder(),
                                    latents=torch.nn.Parameter(torch.zeros(8, 28)),
                                    ema_decoder=None, ema_latents=None)
        for _ in range(n_pre_steps):
            rt.adamw_sched.step()
        if do_resume:
            saved = rt.adamw_sched.state_dict()
            rt = d._build_stage_runtime(spec, decoder=d._new_decoder(),
                                        latents=torch.nn.Parameter(torch.zeros(8, 28)),
                                        ema_decoder=None, ema_latents=None)
            rt.adamw_sched.load_state_dict(saved)
        # The driver's resume contract (faithful to the epoch loop: train-epoch THEN
        # sched.step()): the trajectory PRODUCED BY the remaining .step()s must match an
        # uninterrupted run. (PyTorch LambdaLR.load_state_dict restores last_epoch but
        # does not re-push to param_groups until the next step — exactly what the driver
        # relies on, since it steps every epoch before the next telemetry read.)
        out = []
        for _ in range(spec.epochs - n_pre_steps):
            rt.adamw_sched.step()
            out.append(rt.adamw_opt.param_groups[0]["lr"])
        return out

    # ceil(0.3*12)=4 → step to epoch 2 (mid-warmup), checkpoint, restore, finish.
    uninterrupted = lr_trajectory(2, do_resume=False)
    resumed = lr_trajectory(2, do_resume=True)
    assert len(resumed) == spec.epochs - 2
    for got, want in zip(resumed, uninterrupted, strict=True):
        assert abs(got - want) < 1e-12, "resume must continue the SAME warmup ramp"


# --------------------------------------------------------------------------- #
# 5b. resume guard — a CHANGED warmup shape fails closed MID-STAGE (sister of the
#     muon_lr_floor_fix guard: a mid-stage LambdaLR step-count is tuned to the old
#     shape; resuming under a new shape would mis-schedule the stage LR).
# --------------------------------------------------------------------------- #
def _warmup_driver_with_death(tmp_path, *, warmup_frac, die_at, sub):
    from tac.torch_vehicle.driver import _SimulatedDeath

    cfg = TorchVehicleConfig(
        base_channels=8, latent_dim=28, out_dir=tmp_path / sub, device="cpu", seed=0,
        eval_every=999, checkpoint_every_epochs=1, stage_lr_warmup_frac=warmup_frac,
    )
    scorer = SyntheticScorerContext(n_pairs=6, device="cpu", seed=0)
    d = TorchVehicleDriver(cfg, scorer=scorer, vendored=import_vendored_bundle(),
                           curriculum=[_spec("s", 12, 1e-3)])
    d._stop_after_global_epoch = die_at
    with pytest.raises(_SimulatedDeath):
        d.run()
    return tmp_path / sub


def test_resume_rejects_changed_warmup_frac_mid_stage(tmp_path):
    """Die mid-stage (epoch 4 of 12) under warmup=0.3, then attempt resume under
    warmup=0.0 → fail closed (the LambdaLR step-count is tuned to the 0.3 shape)."""
    out = _warmup_driver_with_death(tmp_path, warmup_frac=0.3, die_at=4, sub="r")
    cfg = TorchVehicleConfig(
        base_channels=8, latent_dim=28, out_dir=out, device="cpu", seed=0,
        eval_every=999, checkpoint_every_epochs=1, stage_lr_warmup_frac=0.0,  # CHANGED
    )
    scorer = SyntheticScorerContext(n_pairs=6, device="cpu", seed=0)
    d2 = TorchVehicleDriver(cfg, scorer=scorer, vendored=import_vendored_bundle(),
                            curriculum=[_spec("s", 12, 1e-3)])
    with pytest.raises(ValueError, match="changed warmup shape"):
        d2.run()


def test_resume_allows_same_warmup_frac_mid_stage(tmp_path):
    """Die mid-stage under warmup=0.3, resume under the SAME warmup=0.3 → resumes
    cleanly to completion (the guard does NOT block a faithful resume)."""
    out = _warmup_driver_with_death(tmp_path, warmup_frac=0.3, die_at=4, sub="r2")
    cfg = TorchVehicleConfig(
        base_channels=8, latent_dim=28, out_dir=out, device="cpu", seed=0,
        eval_every=999, checkpoint_every_epochs=1, stage_lr_warmup_frac=0.3,  # SAME
    )
    scorer = SyntheticScorerContext(n_pairs=6, device="cpu", seed=0)
    d2 = TorchVehicleDriver(cfg, scorer=scorer, vendored=import_vendored_bundle(),
                            curriculum=[_spec("s", 12, 1e-3)])
    summary = d2.run()  # no raise
    assert summary is not None


# --------------------------------------------------------------------------- #
# 6. end-to-end run with warmup completes + emits telemetry (no crash; the flag is wired)
# --------------------------------------------------------------------------- #
def test_two_stage_run_with_warmup_completes(tmp_path):
    from tac.torch_vehicle.telemetry import read_trajectory

    cfg = TorchVehicleConfig(
        base_channels=8, latent_dim=28, out_dir=tmp_path / "e2e", device="cpu", seed=0,
        eval_every=999, checkpoint_every_epochs=999, stage_lr_warmup_frac=0.3,
    )
    scorer = SyntheticScorerContext(n_pairs=8, device="cpu", seed=0)
    cur = [_spec("s1", 6, 1e-4), _spec("s2", 6, 1e-3)]
    TorchVehicleDriver(cfg, scorer=scorer, vendored=import_vendored_bundle(),
                       curriculum=cur).run()
    rows = read_trajectory(tmp_path / "e2e")
    # both stages ran; the stage-2 boundary epoch (epoch_in_stage=1) has a low AdamW LR.
    s2_first = next(r for r in rows if r["stage_index"] == 1 and r["epoch_in_stage"] == 1)
    assert s2_first["adamw_lr"] < 1e-3, (
        f"stage-2 boundary LR should be warmed-up (< peak 1e-3); got {s2_first['adamw_lr']:.2e}"
    )
