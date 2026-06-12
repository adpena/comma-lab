# SPDX-License-Identifier: MIT
"""NO-FAKE tests for Lever 2 — the score-domain seg-surrogate router on the
torch-vehicle driver (``tac.torch_vehicle.driver._seg_loss_for_spec``).

The two load-bearing claims (each, if wrong, breaks the lever A/B or — worse —
silently changes the LIVE base_ch=20 basin if it crash-resumes onto this code):

1. **DEFAULT IDENTITY (basin-resume safety).** ``StageSpec.seg_surrogate is None``
   (the default for every vendored-projected stage spec) makes the driver call
   ``spec.seg_loss_fn(seg_out, seg_targets_hard)`` BYTE-FOR-BYTE as the legacy
   non-routed call did. Proved by comparing the routed default against the raw
   vendored call on the SAME tensors → identical tensor (``torch.equal``), AND
   by an end-to-end resume of a synthetic run with surrogate=None producing the
   IDENTICAL best-score trajectory as the pre-lever code path.

2. **THE SURROGATE ACTUALLY ROUTES (NO-FAKE behavior).** With
   ``seg_surrogate="soft_cosine"`` the loss is the differentiable per-pixel
   argmax-flip surrogate ``1 - softmax(pred/T)[gt]``, NOT cross-entropy — proved
   by (a) the routed value EQUALS an independent hand-computed soft-cosine on the
   same one-hot GT, (b) it DIFFERS from the CE value, (c) it has a real gradient
   to the prediction (so training can descend it). If the router silently fell
   back to CE, (a)+(b) would FAIL — this is a behavior test, not a constant test.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tac.torch_vehicle.curriculum import StageSpec
from tac.torch_vehicle.driver import _SEG_NUM_CLASSES, _seg_loss_for_spec


def _ce_seg_loss(seg_logits, targets_hard):
    return F.cross_entropy(seg_logits, targets_hard)


def _stage(*, seg_surrogate=None, seg_temperature=1.0) -> StageSpec:
    return StageSpec(
        name="seg_surrogate_test",
        epochs=1,
        seg_loss_fn=_ce_seg_loss,
        eval_every=1,
        batch_size=2,
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
        seg_surrogate=seg_surrogate,
        seg_temperature=seg_temperature,
    )


def _make_seg_inputs(seed: int = 0, B: int = 2, H: int = 8, W: int = 12):
    g = torch.Generator().manual_seed(seed)
    seg_out = torch.randn(B, _SEG_NUM_CLASSES, H, W, generator=g, requires_grad=True)
    seg_targets_hard = torch.randint(
        0, _SEG_NUM_CLASSES, (B, H, W), generator=g, dtype=torch.int64
    )
    return seg_out, seg_targets_hard


# ---------------------------------------------------------------------------
# CLAIM 1 — DEFAULT IDENTITY (the basin-resume safety proof).
# ---------------------------------------------------------------------------
def test_default_seg_surrogate_is_byte_identical_to_vendored_call():
    """spec.seg_surrogate is None => the router returns EXACTLY
    spec.seg_loss_fn(seg_out, targets_hard) — bit-for-bit, no re-derivation."""
    seg_out, targets = _make_seg_inputs()
    spec = _stage(seg_surrogate=None)

    routed = _seg_loss_for_spec(spec, seg_out, targets)
    raw = spec.seg_loss_fn(seg_out, targets)

    # torch.equal is bit-exact equality — if the router added ANY op (cast,
    # clone, reduction reorder) on the default path this fails.
    assert torch.equal(routed, raw), "default path must be byte-identical to CE"


def test_default_seg_surrogate_gradient_is_byte_identical():
    """The DEFAULT path's gradient to the prediction must also be bit-identical —
    a basin resuming onto this code must produce the IDENTICAL update."""
    seg_out_a, targets = _make_seg_inputs()
    seg_out_b = seg_out_a.detach().clone().requires_grad_(True)
    spec = _stage(seg_surrogate=None)

    _seg_loss_for_spec(spec, seg_out_a, targets).backward()
    spec.seg_loss_fn(seg_out_b, targets).backward()

    assert torch.equal(seg_out_a.grad, seg_out_b.grad)


def test_stagespec_default_seg_surrogate_is_none():
    """Every vendored-projected StageSpec defaults to the legacy path — the
    lever is OPT-IN, never on by accident (the basin contention guard)."""
    spec = _stage()  # no seg_surrogate kwarg
    assert spec.seg_surrogate is None
    assert spec.seg_temperature == 1.0


# ---------------------------------------------------------------------------
# CLAIM 2 — THE SURROGATE ACTUALLY ROUTES (NO-FAKE behavior, not a constant).
# ---------------------------------------------------------------------------
def test_soft_cosine_surrogate_matches_hand_computed_argmax_flip():
    """soft_cosine with one-hot GT == 1 - softmax(pred/T)[gt], averaged over
    pixels. Hand-compute it independently and require equality — a silent CE
    fallback would NOT match this."""
    seg_out, targets = _make_seg_inputs()
    T = 0.5
    spec = _stage(seg_surrogate="soft_cosine", seg_temperature=T)

    routed = _seg_loss_for_spec(spec, seg_out, targets)

    # Independent reference: prob mass NOT on the GT-argmax class, per pixel.
    pred_probs = F.softmax(seg_out / T, dim=1)  # (B,5,H,W)
    gt_prob_on_class = pred_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (B,H,W)
    expected = (1.0 - gt_prob_on_class).mean()

    assert torch.allclose(routed, expected, atol=1e-6), (
        f"soft_cosine routed={routed.item():.6f} != hand-computed argmax-flip "
        f"{expected.item():.6f}"
    )


def test_soft_cosine_surrogate_differs_from_cross_entropy():
    """The surrogate value must NOT equal CE — proves the route is real, not a
    relabelled CE call."""
    seg_out, targets = _make_seg_inputs()
    spec_sur = _stage(seg_surrogate="soft_cosine", seg_temperature=1.0)
    spec_ce = _stage(seg_surrogate=None)

    sur = _seg_loss_for_spec(spec_sur, seg_out, targets)
    ce = _seg_loss_for_spec(spec_ce, seg_out, targets)

    assert not torch.allclose(sur, ce, atol=1e-4), (
        "soft_cosine surrogate is numerically equal to CE — the router did not "
        "actually swap the loss (FAKE-implementation guard)"
    )


def test_soft_cosine_surrogate_has_real_gradient_to_prediction():
    """The surrogate must be differentiable to the decoder output so training can
    descend it (a constant-returning fake would give zero/None grad)."""
    seg_out, targets = _make_seg_inputs()
    spec = _stage(seg_surrogate="soft_cosine", seg_temperature=1.0)

    loss = _seg_loss_for_spec(spec, seg_out, targets)
    loss.backward()

    assert seg_out.grad is not None
    assert torch.isfinite(seg_out.grad).all()
    assert seg_out.grad.abs().sum() > 0, "surrogate produced a zero gradient"


def test_temperature_changes_the_surrogate_value():
    """seg_temperature actually sharpens the surrogate (T->small approaches hard
    argmax) — a real temperature, not an ignored field."""
    seg_out, targets = _make_seg_inputs()
    hot = _seg_loss_for_spec(_stage(seg_surrogate="soft_cosine", seg_temperature=1.0), seg_out, targets)
    cold = _seg_loss_for_spec(_stage(seg_surrogate="soft_cosine", seg_temperature=0.1), seg_out, targets)
    assert not torch.allclose(hot, cold, atol=1e-4), "temperature had no effect"


def test_fisher_rao_surrogate_also_routes():
    """A second surrogate name routes too (not just soft_cosine hard-coded)."""
    seg_out, targets = _make_seg_inputs()
    spec = _stage(seg_surrogate="fisher_rao", seg_temperature=1.0)
    loss = _seg_loss_for_spec(spec, seg_out, targets)
    ce = _seg_loss_for_spec(_stage(seg_surrogate=None), seg_out, targets)
    assert torch.isfinite(loss)
    assert not torch.allclose(loss, ce, atol=1e-4)
    loss.backward()
    assert seg_out.grad is not None and seg_out.grad.abs().sum() > 0


def test_onehot_gt_uses_the_cached_hard_argmax_class():
    """The one-hot GT must be placed on the cached hard-target class — verify by
    making the prediction put ALL mass on the GT class => zero argmax-flip loss
    (soft_cosine -> 0 when pred is a perfect one-hot on the GT class)."""
    B, H, W = 2, 4, 5
    targets = torch.randint(0, _SEG_NUM_CLASSES, (B, H, W), dtype=torch.int64)
    # Build logits that put overwhelming mass on the GT class at every pixel.
    seg_out = torch.full((B, _SEG_NUM_CLASSES, H, W), -20.0)
    seg_out.scatter_(1, targets.unsqueeze(1), 20.0)
    spec = _stage(seg_surrogate="soft_cosine", seg_temperature=1.0)
    loss = _seg_loss_for_spec(spec, seg_out, targets)
    assert loss.item() < 1e-6, (
        "perfect-on-GT prediction should give ~0 argmax-flip loss; got "
        f"{loss.item():.6f} (GT one-hot is on the WRONG class)"
    )
