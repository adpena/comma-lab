# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the WS-A meso fixes (default-OFF opt-in flags → the live A/B stays
apples-to-apples; the scaled run opts in). Two code fixes:

1. **margin-weight RENORM** (`StageSpec.margin_weight_renorm`, M7): off ⇒ `(per_pixel·w).mean()`
   (byte-identical); on ⇒ `(per_pixel·w).sum()/Σw` (a true weighted mean — redistributes to
   the boundary without the total-magnitude decay as margins grow).
2. **Muon LR-floor fix** (`TorchVehicleConfig.muon_lr_floor_fix`, M3): off ⇒ Muon shares the
   AdamW lr_lambda (floor mis-keyed to adamw_lr → ~0.5× at stage 8); on ⇒ Muon's own cosine
   floor keyed to muon_lr (the intended absolute floor).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from tac.torch_vehicle.curriculum import StageSpec
from tac.torch_vehicle.driver import (
    _SEG_NUM_CLASSES,
    _seg_loss_for_spec,
    _segnet_logit_margin_map,
)


def _ce(s, t):
    return F.cross_entropy(s, t)


def _spec(**kw) -> StageSpec:
    base = dict(
        name="m", epochs=10, seg_loss_fn=_ce, eval_every=10, batch_size=4, ema_decay=0.999,
        use_muon=False, adamw_lr=1e-3, muon_lr=2e-4, muon_weight_decay=0.0, latent_lr_mult=10.0,
        grad_clip=1e9, grad_clip_muon=1e9, lr_floor_ratio=5e-6, seg_weight=100.0, pose_weight=1.0,
        cat_lambda=0.0, cat_sigma=0.2, use_qat=False, init_latents_random=True,
    )
    base.update(kw)
    return StageSpec(**base)


def _inputs(seed=0, B=2, H=8, W=12):
    g = torch.Generator().manual_seed(seed)
    seg_out = torch.randn(B, _SEG_NUM_CLASSES, H, W, generator=g, requires_grad=True)
    tgt = torch.randint(0, _SEG_NUM_CLASSES, (B, H, W), generator=g, dtype=torch.int64)
    return seg_out, tgt


# --- margin-renorm -----------------------------------------------------------
def test_margin_renorm_default_is_false():
    assert _spec().margin_weight_renorm is False


def test_margin_renorm_off_is_unnormalized_mean():
    """off ⇒ exactly (per_pixel·w).mean() — byte-identical to the legacy Lever-5 path."""
    seg_out, tgt = _inputs()
    spec = _spec(seg_surrogate="soft_cosine", seg_temperature=1.0, margin_weight_tau=0.3,
                 margin_weight_renorm=False)
    routed = _seg_loss_for_spec(spec, seg_out, tgt)
    # independent reference: replicate per_pixel·w then .mean()
    from tac.losses.core import segnet_surrogate_per_pixel
    gt = (F.one_hot(tgt, _SEG_NUM_CLASSES).permute(0, 3, 1, 2).to(seg_out.dtype) * 30.0)
    pp = segnet_surrogate_per_pixel(seg_out, gt, surrogate="soft_cosine", temperature=1.0,
                                    gt_already_probs=False, num_classes=_SEG_NUM_CLASSES)
    w = torch.exp(-_segnet_logit_margin_map(seg_out) / 0.3)
    assert torch.allclose(routed, (pp * w).mean(), atol=1e-6)


def test_margin_renorm_on_is_weighted_mean_and_differs():
    """on ⇒ (per_pixel·w).sum()/Σw, and it DIFFERS from the off .mean() (real behavior change)."""
    seg_out, tgt = _inputs()
    off = _seg_loss_for_spec(_spec(seg_surrogate="soft_cosine", seg_temperature=1.0,
                                   margin_weight_tau=0.3, margin_weight_renorm=False), seg_out, tgt)
    on = _seg_loss_for_spec(_spec(seg_surrogate="soft_cosine", seg_temperature=1.0,
                                  margin_weight_tau=0.3, margin_weight_renorm=True), seg_out, tgt)
    from tac.losses.core import segnet_surrogate_per_pixel
    gt = (F.one_hot(tgt, _SEG_NUM_CLASSES).permute(0, 3, 1, 2).to(seg_out.dtype) * 30.0)
    pp = segnet_surrogate_per_pixel(seg_out, gt, surrogate="soft_cosine", temperature=1.0,
                                    gt_already_probs=False, num_classes=_SEG_NUM_CLASSES)
    w = torch.exp(-_segnet_logit_margin_map(seg_out) / 0.3)
    assert torch.allclose(on, (pp * w).sum() / w.sum().clamp_min(1e-6), atol=1e-6)
    assert not torch.allclose(on, off, atol=1e-5), "renorm must change the loss value"
    loss = _seg_loss_for_spec(_spec(seg_surrogate="soft_cosine", margin_weight_tau=0.3,
                                    margin_weight_renorm=True), seg_out, tgt)
    loss.backward()
    assert seg_out.grad is not None and seg_out.grad.abs().sum() > 0


# --- Muon LR-floor fix -------------------------------------------------------
def _muon_spec():
    return _spec(use_muon=True, adamw_lr=1e-5, muon_lr=2e-4, lr_floor_ratio=5e-6,
                 grad_clip=1.0, grad_clip_muon=1.0)


def _build_driver(tmp_path, muon_fix: bool):
    from tac.torch_vehicle.driver import (
        TorchVehicleConfig, TorchVehicleDriver, import_vendored_bundle,
    )
    from tac.torch_vehicle.scorer_context import SyntheticScorerContext
    cfg = TorchVehicleConfig(base_channels=8, latent_dim=28, out_dir=tmp_path / f"m{int(muon_fix)}",
                             device="cpu", seed=0, muon_lr_floor_fix=muon_fix)
    scorer = SyntheticScorerContext(n_pairs=4, device="cpu", seed=0)
    return TorchVehicleDriver(cfg, scorer=scorer, vendored=import_vendored_bundle(),
                              curriculum=[_muon_spec()])


def _final_muon_lr(tmp_path, muon_fix: bool) -> float:
    d = _build_driver(tmp_path, muon_fix)
    spec = _muon_spec()
    decoder = d._new_decoder()
    latents = torch.nn.Parameter(torch.zeros(4, 28))
    rt = d._build_stage_runtime(spec, decoder=decoder, latents=latents,
                                ema_decoder=None, ema_latents=None)
    assert rt.muon_opt is not None and rt.muon_sched is not None
    for _ in range(spec.epochs):  # step to the last epoch (cosine floor)
        rt.muon_sched.step()
    return rt.muon_opt.param_groups[0]["lr"]


def test_muon_lr_floor_fix_default_false():
    from tac.torch_vehicle.driver import TorchVehicleConfig
    assert TorchVehicleConfig(device="cpu").muon_lr_floor_fix is False


def test_muon_lr_floor_off_shares_adamw_floor(tmp_path):
    """off ⇒ Muon floors at lr_floor_ratio/adamw_lr = 0.5 → final LR ≈ 0.5×2e-4 = 1e-4."""
    lr = _final_muon_lr(tmp_path, muon_fix=False)
    assert abs(lr - 0.5 * 2e-4) < 1e-6, f"off muon floor should be ~1e-4; got {lr:.2e}"


def test_muon_lr_floor_on_keys_to_muon_lr(tmp_path):
    """on ⇒ Muon floors at lr_floor_ratio/muon_lr = 0.025 → final LR ≈ 0.025×2e-4 = 5e-6,
    ~20× LOWER than off (Muon actually anneals)."""
    off = _final_muon_lr(tmp_path, muon_fix=False)
    on = _final_muon_lr(tmp_path, muon_fix=True)
    assert on < off, f"fix must lower the Muon floor (on {on:.2e} !< off {off:.2e})"
    assert abs(on - 0.025 * 2e-4) < 1e-7, f"on muon floor should be ~5e-6; got {on:.2e}"
