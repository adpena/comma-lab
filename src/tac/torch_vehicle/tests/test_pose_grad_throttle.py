# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the pose-gradient THROTTLE speed lever (``pose_grad_every_k`` +
``pose_grad_resume_threshold``) on the split-by-head backward path.

The lever exists because the split-by-head PoseNet path costs ~51% of the epoch (CPU
FastViT fwd+bwd, measured 10.2s of a 20s epoch @96 pairs) while d_pose converges to
~0.0015 early — computing a near-noise √(10·d_pose) gradient every epoch is the waste.

The load-bearing claims (each, if wrong, either silently corrupts the basin OR makes
the "speedup" a fake that still pays the pose cost):

1. **DEFAULT BYTE-IDENTICAL (k=1).** ``compute_pose=True`` (the default; k=1 always
   resolves to it) produces the SAME combined decoder gradient as before — seg + pose.
   Proved by equality to a single-graph ``(w_seg*seg + w_pose*pose).backward()``.
2. **THROTTLE ACTUALLY DROPS THE POSE GRADIENT (not a no-op fake).** ``compute_pose=
   False`` produces a decoder gradient EQUAL to the SegNet-only backward and DIFFERENT
   from the seg+pose gradient. If the throttle silently still added pose (or changed
   nothing), claims 2a/2b would FAIL — this is a behavior test, not a constant test.
3. **THROTTLE ACTUALLY SKIPS THE EXPENSIVE FORWARD.** ``compute_pose=False`` does NOT
   call ``pose_forward_authority`` at all (a call-counter proves the 51% CPU cost is
   genuinely not paid), and an end-to-end k=3 run calls the pose head a STRICT SUBSET
   of the epochs a k=1 run does.
4. **SELF-PROTECT / WARM via the resume threshold.** With a high threshold the pose
   head fires EVERY epoch (pose is "above threshold" → never starved); with threshold
   0 and a large k it fires only on the cadence epoch — proving the threshold is the
   warm/guard mechanism, not an ignored field.
5. **CONFIG VALIDATION.** k<1 and threshold<0 are refused at construction.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from tac.torch_vehicle.curriculum import StageSpec
from tac.torch_vehicle.driver import (
    TorchVehicleConfig,
    TorchVehicleDriver,
    _seg_loss_for_spec,
    import_vendored_bundle,
)
from tac.torch_vehicle.scorer_context import SyntheticScorerContext, _TinyFrozenScorer


def _ce_seg_loss(seg_logits, targets_hard):
    return F.cross_entropy(seg_logits, targets_hard)


def _stage(epochs: int = 2, eval_every: int = 100) -> StageSpec:
    return StageSpec(
        name="pose_throttle_test",
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
        base_channels=base_channels, latent_dim=28, out_dir=tmp_path / "throttle",
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


# ---------------------------------------------------------------------------
# CLAIM 5 — config validation.
# ---------------------------------------------------------------------------
def test_pose_grad_every_k_below_one_refused():
    with pytest.raises(ValueError, match="pose_grad_every_k must be >= 1"):
        TorchVehicleConfig(device="cpu", train_device="mps", split_by_head=True,
                           pose_grad_every_k=0)


def test_pose_grad_resume_threshold_negative_refused():
    with pytest.raises(ValueError, match="pose_grad_resume_threshold must be >= 0"):
        TorchVehicleConfig(device="cpu", train_device="mps", split_by_head=True,
                           pose_grad_resume_threshold=-0.1)


def test_pose_grad_throttle_defaults_are_byte_identical_knobs():
    cfg = TorchVehicleConfig(device="cpu")
    assert cfg.pose_grad_every_k == 1
    assert cfg.pose_grad_resume_threshold == 0.0


# ---------------------------------------------------------------------------
# CLAIMS 1 + 2 — compute_pose True == seg+pose; False == seg-only (the math proof).
# ---------------------------------------------------------------------------
def _theta_and_idx(B=3):
    torch.manual_seed(0)
    theta = torch.nn.Parameter(torch.rand(B, 2, 384, 512, 3) * 200.0 + 30.0)
    idx = torch.arange(B)
    return theta, idx


def _grad_from_split(driver, theta, idx, spec, *, compute_pose):
    theta.grad = None
    decoded = theta * 1.0  # non-leaf so theta.grad captures decoded_bhwc.backward
    driver._split_by_head_backward(decoded, idx, spec, compute_pose=compute_pose)
    return theta.grad.detach().clone()


def test_compute_pose_true_includes_pose_false_is_seg_only(tmp_path):
    driver = _cpu_split_driver(tmp_path)
    spec = _stage()
    theta, idx = _theta_and_idx()

    grad_true = _grad_from_split(driver, theta, idx, spec, compute_pose=True)
    grad_false = _grad_from_split(driver, theta, idx, spec, compute_pose=False)

    # The throttle must CHANGE the gradient (it drops the pose term) — not a no-op fake.
    assert not torch.equal(grad_true, grad_false), (
        "compute_pose=False produced the SAME gradient as True — the throttle did not "
        "drop the pose contribution (no-op fake)"
    )

    # And compute_pose=False must equal the SegNet-ONLY backward exactly (the pose term
    # is dropped, nothing else changes).
    theta.grad = None
    decoded = theta * 1.0
    frames_seg = decoded.detach().requires_grad_(True)
    seg_out = driver.scorer.seg_forward_train(frames_seg)
    seg_l = _seg_loss_for_spec(spec, seg_out, driver.scorer.seg_targets_hard[idx])
    (spec.seg_weight * seg_l).backward()
    decoded.backward(gradient=frames_seg.grad)
    grad_seg_only = theta.grad.detach().clone()

    assert torch.equal(grad_false, grad_seg_only), (
        "compute_pose=False gradient != the pure SegNet-only gradient — the throttle "
        "changed more than just dropping the pose term"
    )


def test_compute_pose_true_equals_single_graph_seg_plus_pose(tmp_path):
    """compute_pose=True (the byte-identical default) == the single-graph
    (w_seg*seg + w_pose*pose).backward() gradient — the basin-resume safety proof."""
    driver = _cpu_split_driver(tmp_path)
    spec = _stage()
    theta, idx = _theta_and_idx()

    grad_true = _grad_from_split(driver, theta, idx, spec, compute_pose=True)

    # Single-graph reference using the SAME (identical-weight) scorers.
    theta.grad = None
    decoded = theta * 1.0
    seg_out = driver.scorer.seg_forward_train(decoded)
    seg_l = _seg_loss_for_spec(spec, seg_out, driver.scorer.seg_targets_hard[idx])
    pose_pred = driver.scorer.pose_forward_authority(decoded)
    pose_mse = F.mse_loss(pose_pred, driver.scorer.pose_targets[idx])
    pose_l = torch.sqrt(10.0 * pose_mse + 1e-12)
    (spec.seg_weight * seg_l + spec.pose_weight * pose_l).backward()
    grad_single = theta.grad.detach().clone()

    assert torch.allclose(grad_true, grad_single, atol=1e-5), (
        "compute_pose=True combined gradient differs from the single-graph seg+pose "
        f"gradient (max abs diff {(grad_true - grad_single).abs().max().item():.3e})"
    )


# ---------------------------------------------------------------------------
# CLAIM 3 — the throttle actually SKIPS the expensive PoseNet forward.
# ---------------------------------------------------------------------------
def test_compute_pose_false_does_not_call_posenet(tmp_path):
    driver = _cpu_split_driver(tmp_path)
    spec = _stage()
    theta, idx = _theta_and_idx()

    calls = {"n": 0}
    orig = driver.scorer.pose_forward_authority

    def counting(x):
        calls["n"] += 1
        return orig(x)

    driver.scorer.pose_forward_authority = counting

    _grad_from_split(driver, theta, idx, spec, compute_pose=True)
    assert calls["n"] == 1, "compute_pose=True should call the PoseNet head exactly once"
    _grad_from_split(driver, theta, idx, spec, compute_pose=False)
    assert calls["n"] == 1, (
        "compute_pose=False called the PoseNet head — the 51% CPU cost was still paid "
        "(the throttle is a fake speedup)"
    )


def test_last_pose_mse_updates_on_true_carries_on_false(tmp_path):
    driver = _cpu_split_driver(tmp_path)
    spec = _stage()
    theta, idx = _theta_and_idx()

    assert driver._last_pose_mse is None
    _grad_from_split(driver, theta, idx, spec, compute_pose=True)
    assert driver._last_pose_mse is not None and driver._last_pose_mse >= 0.0
    pinned = driver._last_pose_mse
    _grad_from_split(driver, theta, idx, spec, compute_pose=False)
    assert driver._last_pose_mse == pinned, (
        "_last_pose_mse changed on a skipped epoch — it must only update when pose is "
        "actually computed"
    )


# ---------------------------------------------------------------------------
# CLAIMS 3 + 4 — END-TO-END cadence: a real curriculum run calls the pose head the
# right NUMBER of times under k / threshold.
# ---------------------------------------------------------------------------
def _run_counting_pose_calls(tmp_path, *, every_k, resume_threshold, epochs=7):
    driver = _cpu_split_driver(
        tmp_path, curriculum=[_stage(epochs=epochs, eval_every=epochs)],
        pose_grad_every_k=every_k, pose_grad_resume_threshold=resume_threshold,
    )
    calls = {"n": 0}
    orig = driver.scorer.pose_forward_authority

    def counting(x):
        calls["n"] += 1
        return orig(x)

    driver.scorer.pose_forward_authority = counting
    summary = driver.run()
    assert summary["status"] == "complete"
    return calls["n"]


def test_k1_calls_pose_every_epoch(tmp_path):
    """k=1 (default) computes pose EVERY epoch — the byte-identical baseline count."""
    n = _run_counting_pose_calls(tmp_path / "k1", every_k=1, resume_threshold=0.0, epochs=7)
    assert n == 7, f"k=1 should call pose once per epoch (7); got {n}"


def test_throttle_k3_reduces_pose_calls_to_strict_subset(tmp_path):
    """k=3 with threshold 0 computes pose only on cadence epochs (0,3,6) → strictly
    fewer than the 7 a k=1 run pays. This is the NO-FAKE speedup proof."""
    n = _run_counting_pose_calls(tmp_path / "k3", every_k=3, resume_threshold=0.0, epochs=7)
    # global epochs 0..6: do_pose on 0,3,6 (cadence) → 3 calls. Strictly < 7.
    assert n < 7, f"k=3 must call pose fewer than every-epoch (7); got {n}"
    assert n == 3, f"k=3 over 7 epochs should compute pose on epochs 0,3,6 (3); got {n}"


def test_resume_threshold_forces_pose_while_above(tmp_path):
    """The resume threshold is the warm/self-protect guard: pose is force-computed while
    the running pose_mse is ABOVE it (still converging / drifted). With a tiny positive
    threshold (below the synthetic run's pose_mse) pose fires EVERY epoch even under a
    huge k; with threshold 0 (disabled) the cadence governs and pose fires only on
    epoch 0. (Higher pose_mse than the threshold ⇒ 'not solved yet' ⇒ keep computing.)"""
    big_k = 1000
    n_guarded = _run_counting_pose_calls(
        tmp_path / "guard", every_k=big_k, resume_threshold=1e-6, epochs=6)
    assert n_guarded == 6, (
        f"a threshold below the running pose_mse must force pose every epoch (6); "
        f"got {n_guarded}"
    )
    n_cadence = _run_counting_pose_calls(
        tmp_path / "cad", every_k=big_k, resume_threshold=0.0, epochs=6)
    # epoch 0: _last_pose_mse is None → compute. epochs 1..5: %1000!=0, thr disabled → skip.
    assert n_cadence == 1, (
        f"with threshold 0 + huge k, pose fires only on epoch 0 (1); got {n_cadence}"
    )
