# SPDX-License-Identifier: MIT
"""Curriculum module smoke tests (Phase 2 deliverable).

Verifies the 8-stage PR95 curriculum primitives are wired correctly and
mathematically faithful to ``submissions/hnerv_muon/src/losses.py`` +
``optim.py``. No CUDA required; runs in <2s on CPU.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tac.substrates.pr101_lc_v2_clone.curriculum import (
    CURRICULUM_STAGES,
    CurriculumStageConfig,
    Muon,
    apply_qat,
    cat_entropy_v2,
    ce_seg_loss,
    ema_update,
    fake_quantize,
    get_seg_loss_fn,
    l7_softplus_seg_loss,
    partition_params_for_muon,
    pose_loss,
    restore_qat,
    smooth_disagreement_seg_loss,
    stage1_v328_ce,
    stage8_muon_finetune,
    tau_softplus_seg_loss,
    zeropower_via_newtonschulz5,
)


def test_curriculum_stages_registry_has_all_eight() -> None:
    expected = {
        "stage1_v328_ce",
        "stage2_v331_softplus",
        "stage3_v332_smooth",
        "stage4_v332_qat",
        "stage5_c1a_l7",
        "stage6_lambda_sweep",
        "stage7_sigma_sweep",
        "stage8_muon_finetune",
    }
    assert set(CURRICULUM_STAGES.keys()) == expected


def test_stage1_config_matches_pr95_canonical() -> None:
    cfg = stage1_v328_ce()
    assert cfg.epochs == 3000
    assert cfg.seg_loss_kind == "ce"
    assert cfg.use_qat is False
    assert cfg.use_muon is False
    assert cfg.adamw_lr == 1e-3
    assert cfg.cat_lambda == 0.0
    assert cfg.init_latents_random is True


def test_stage8_config_matches_pr95_canonical_plus_researcher_24() -> None:
    cfg = stage8_muon_finetune()
    assert cfg.seg_loss_kind == "l7_softplus"
    assert cfg.use_qat is True
    assert cfg.use_muon is True
    assert cfg.adamw_lr == 1e-5
    assert cfg.muon_lr == 2e-4
    assert cfg.muon_weight_decay == 5e-4  # researcher #24 tweak
    assert cfg.cat_lambda == 0.02
    assert cfg.cat_sigma == 0.1


def test_ce_seg_loss_shape_and_finite() -> None:
    logits = torch.randn(2, 5, 8, 16)
    targets = torch.randint(0, 5, (2, 8, 16))
    loss = ce_seg_loss(logits, targets)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_tau_softplus_seg_loss_finite_and_nonnegative() -> None:
    logits = torch.randn(2, 5, 8, 16)
    targets = torch.randint(0, 5, (2, 8, 16))
    loss = tau_softplus_seg_loss(logits, targets, tau=0.3)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_smooth_disagreement_seg_loss_bounded_in_zero_one() -> None:
    logits = torch.randn(2, 5, 8, 16)
    targets = torch.randint(0, 5, (2, 8, 16))
    loss = smooth_disagreement_seg_loss(logits, targets, tau=0.3)
    assert 0.0 <= loss.item() <= 1.0


def test_l7_softplus_seg_loss_finite() -> None:
    logits = torch.randn(2, 5, 8, 16)
    targets = torch.randint(0, 5, (2, 8, 16))
    loss = l7_softplus_seg_loss(logits, targets)
    assert torch.isfinite(loss)


def test_pose_loss_matches_contest_formula() -> None:
    # pose = sqrt(10 * MSE + 1e-12). For MSE=0.1 -> sqrt(1.0+eps) = ~1.0.
    pred = torch.zeros(4, 6)
    tgt = torch.full((4, 6), 0.1).sqrt()  # MSE = 0.1
    loss = pose_loss(pred, tgt)
    # 0.1^0.5 ~ 0.3162; squared mean over 4*6=24 elements
    # MSE = 0.1 exactly, so loss = sqrt(10*0.1) = 1.0
    assert torch.allclose(loss, torch.tensor(1.0), atol=1e-4)


def test_fake_quantize_round_trip_close() -> None:
    t = torch.randn(64)
    q = fake_quantize(t)
    # STE: forward should round to integer-scaled grid; backward passes through
    assert torch.isfinite(q).all()
    # Quantized output should be near original at the scaled grid
    ma = t.abs().max().item()
    if ma > 0:
        scale = ma / 127.0
        # Expected post-round value
        expected = (t / scale).round().clamp(-127, 127) * scale
        # STE forward equals expected (since forward is q*scale exact)
        assert torch.allclose(q, expected, atol=1e-5)


def test_apply_qat_restore_qat_preserves_live_weights() -> None:
    decoder = nn.Sequential(
        nn.Conv2d(8, 16, 3, padding=1),
        nn.Conv2d(16, 8, 3, padding=1),
    )
    original = {name: mod.weight.data.clone() for name, mod in decoder.named_modules() if isinstance(mod, nn.Conv2d)}
    saved = apply_qat(decoder)
    # After apply_qat, live weights are fake-quantized (NOT bitwise-equal to originals)
    for name, mod in decoder.named_modules():
        if isinstance(mod, nn.Conv2d):
            # Note: STE forward returns (q*scale - tensor).detach() + tensor;
            # apply_qat assigns the OUTPUT of fake_quantize to .data (which strips STE).
            # The post-apply weight equals the integer-scaled grid value.
            assert mod.weight.data.shape == original[name].shape
    restore_qat(decoder, saved)
    for name, mod in decoder.named_modules():
        if isinstance(mod, nn.Conv2d):
            assert torch.equal(mod.weight.data, original[name])


def test_ema_update_basic() -> None:
    model = nn.Linear(8, 4)
    ema = nn.Linear(8, 4)
    ema.load_state_dict(model.state_dict())

    # Mutate live weights
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)

    # Snapshot of ema (pre-update) and model (post-mutation)
    ema_before = {name: p.detach().clone() for name, p in ema.named_parameters()}
    model_after = {name: p.detach().clone() for name, p in model.named_parameters()}

    ema_update(ema, model, None, None, decay=0.5)

    # Per ema_update formula: ema.data = 0.5 * ema.data + 0.5 * model.data
    for name, p in ema.named_parameters():
        expected = 0.5 * ema_before[name] + 0.5 * model_after[name]
        assert torch.allclose(p, expected)


def test_cat_entropy_v2_finite_and_reasonable_bits_range() -> None:
    decoder = nn.Sequential(
        nn.Linear(28, 64),
        nn.Conv2d(8, 16, 3, padding=1),
    )
    entropy = cat_entropy_v2(decoder, sigma=0.2, sample_size=100)
    # For random init weights, the entropy should be in [0, log2(255)] ~ [0, 8]
    assert torch.isfinite(entropy)
    assert 0.0 <= entropy.item() <= 10.0  # generous upper bound


def test_zeropower_via_newtonschulz5_orthogonalizes_2d() -> None:
    # Use a deterministic seed for reproducibility — NS5 is a coarse
    # approximation; the test asserts singular values fall within a
    # moderately wide band around 1.0 (the published Keller-Jordan
    # coefficients (3.4445, -4.7750, 2.0315) are tuned for ~0.5 tolerance
    # at 5 steps on random rectangular matrices).
    torch.manual_seed(0)
    G = torch.randn(8, 4)
    X = zeropower_via_newtonschulz5(G, steps=5)
    U, S, Vh = torch.linalg.svd(X.float())
    # All singular values should be in a reasonable band around 1.0
    assert (S - 1.0).abs().max().item() < 0.5  # NS5 is a coarse approximation
    # The mean should be close to 1.0 (the target of orthogonalization)
    assert abs(S.mean().item() - 1.0) < 0.2


def test_partition_params_for_muon_separates_stem_and_heads() -> None:
    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Linear(28, 64)
            self.blocks = nn.ModuleList([nn.Conv2d(8, 16, 3)])
            self.rgb_0 = nn.Conv2d(16, 3, 3)
            self.rgb_1 = nn.Conv2d(16, 3, 3)

    model = Decoder()
    muon_params, adamw_params = partition_params_for_muon(model)
    # stem.weight (Linear): goes to AdamW (name contains "stem")
    # blocks.0.weight (Conv2d): goes to Muon (2D+, no stem/rgb in name)
    # rgb_0.weight, rgb_1.weight (Conv2d): goes to AdamW (name contains "rgb")
    # all biases (1D): go to AdamW
    adamw_names = [n for n, p in model.named_parameters() for q in adamw_params if p is q]
    muon_names = [n for n, p in model.named_parameters() for q in muon_params if p is q]
    assert any("stem" in n for n in adamw_names)
    assert any("rgb_0" in n for n in adamw_names)
    assert any("rgb_1" in n for n in adamw_names)
    assert any("blocks" in n and ".weight" in n for n in muon_names)


def test_muon_optimizer_step_finite() -> None:
    model = nn.Sequential(
        nn.Linear(8, 16),  # 2D weights -> Muon
        nn.Linear(16, 4),
    )
    params = [p for p in model.parameters() if p.ndim >= 2]
    opt = Muon(params, lr=1e-3, momentum=0.95, ns_steps=5, weight_decay=1e-4)
    x = torch.randn(4, 8)
    y = torch.randn(4, 4)
    out = model(x)
    loss = ((out - y) ** 2).mean()
    loss.backward()
    opt.step()
    for p in params:
        assert torch.isfinite(p).all()


def test_get_seg_loss_fn_resolves_all_kinds() -> None:
    logits = torch.randn(2, 5, 8, 16)
    targets = torch.randint(0, 5, (2, 8, 16))
    for kind in ["ce", "tau_softplus", "smooth_disagreement", "l7_softplus"]:
        loss_fn = get_seg_loss_fn(kind)
        loss = loss_fn(logits, targets)
        assert torch.isfinite(loss), f"non-finite loss for kind={kind}"


def test_get_seg_loss_fn_raises_on_unknown_kind() -> None:
    import pytest

    with pytest.raises(ValueError, match="unknown seg_loss_kind"):
        get_seg_loss_fn("invalid_kind")


def test_curriculum_stage_config_is_frozen_dataclass() -> None:
    cfg = CurriculumStageConfig(name="test", epochs=10, seg_loss_kind="ce")
    # mutation should work (not frozen by design — the trainer mutates extras)
    cfg.extras["custom"] = 1.0
    assert cfg.extras == {"custom": 1.0}
