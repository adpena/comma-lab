"""Round 7 Defect #2 regression — KL-distill weight is operator-controllable.

Council Round 7 §6.2 caught that ``experiments/train_segmap.py`` exposed
``--kl-distill-weight 0.002`` and routed it through a conditional plumbing
block at line 191 (``if "kl_distill_weight" in fields:``) — but
``tac.training.TrainConfig`` had NO ``kl_distill_weight`` field, so the
condition was a no-op and the value was silently dropped. The trainer
then hard-coded ``0.002 * kl_loss`` in ``src/tac/segmap_renderer.py:667``.

Today no live miscalibration (8 of 8 SegMap-class lane scripts pass
``--kl-distill-weight 0.002`` matching the hardcode), but a future operator
running a KL sensitivity sweep with ``--kl-distill-weight 0.005`` would be
silently overridden. Same silent-default pattern flagged in
``feedback_silent_default_bug_class_findings_20260429.md`` (the 246-flag
audit).

The fix:
  1. Add ``kl_distill_weight: float = Field(0.002, ge=0.0)`` to TrainConfig.
  2. Replace the conditional plumbing at train_segmap.py:191 with an
     unconditional ``cfg_kwargs["kl_distill_weight"] = args.kl_distill_weight``.
  3. Replace ``0.002 * kl_loss`` at segmap_renderer.py:667 with
     ``self.config.kl_distill_weight * kl_loss``.

This test asserts that when the operator passes a non-default weight, the
trainer's loss assembly honours it — empirically.

Memory:
  - .omx/research/council_round7_adversarial_20260429.md (Defect #2)
  - feedback_silent_default_bug_class_findings_20260429.md (the bug class)
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from tac.preflight import PreflightError
from tac.segmap_renderer import SEGMAP_INPUT_SIZE, SegMap, SegMapTrainer
from tac.training import TrainConfig


# ── Mock scorers shaped like upstream PoseNet/SegNet (mirrored from
#    test_segmap_renderer.py to keep this file self-contained). ──


class _MockPoseNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(12, 8, kernel_size=3, padding=1)
        self.fc = nn.Linear(8, 12)

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        x = F.interpolate(x, size=(96, 128), mode="bilinear", align_corners=False)
        x = x.reshape(b, t * c, 96, 128)
        if x.shape[1] < 12:
            pad = torch.zeros(
                b, 12 - x.shape[1], 96, 128, device=x.device, dtype=x.dtype
            )
            x = torch.cat([x, pad], dim=1)
        return x[:, :12]

    def forward(self, x: torch.Tensor) -> dict:
        h = self.conv(x).mean(dim=(2, 3))
        return {"pose": self.fc(h)}


class _MockSegNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 5, kernel_size=3, padding=1)

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        last = x[:, -1, ...]
        return F.interpolate(last, size=(48, 64), mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def _make_tiny_segmap() -> SegMap:
    return SegMap(hidden=8, block_hidden=8, num_blocks=2, max_frame_index=16)


def _make_kl_config(kl_weight: float) -> TrainConfig:
    """KL-distill TrainConfig with the supplied weight."""
    return TrainConfig(
        hidden=8,
        epochs=100,
        warmup_epochs=10,
        tag="test-kl-weight",
        lr=1e-3,
        eval_roundtrip=True,
        loss_mode="kl_distill",
        kl_distill_scope="segnet_aux",
        # kl_distill validator requires temperature_start >= 2.0
        temperature_start=2.0,
        temperature_end=1.0,
        kl_distill_weight=kl_weight,
        kl_distill_temperature=2.0,
    )


def test_train_config_exposes_kl_distill_weight_field() -> None:
    """TrainConfig must declare kl_distill_weight as a configurable field
    with default 0.002 (Lane G v3 canonical) and non-negative bound."""
    fields = TrainConfig.model_fields
    assert "kl_distill_weight" in fields, (
        "Round 7 Defect #2 fix: TrainConfig must declare kl_distill_weight."
    )
    f = fields["kl_distill_weight"]
    assert f.default == 0.002, (
        f"kl_distill_weight default must be 0.002 (Lane G v3 canonical), "
        f"got {f.default}."
    )
    cfg = TrainConfig(
        hidden=8, epochs=100, warmup_epochs=10, tag="t",
        eval_roundtrip=True, loss_mode="standard",
    )
    assert cfg.kl_distill_weight == 0.002


def test_train_config_exposes_explicit_kl_temperature_and_scope_fields() -> None:
    """KL auxiliary configs need explicit units/scope, not an overloaded
    primary-loss temperature schedule."""
    fields = TrainConfig.model_fields
    assert "kl_distill_temperature" in fields
    assert fields["kl_distill_temperature"].default == 2.0
    assert "kl_distill_scope" in fields
    assert fields["kl_distill_scope"].default == "none"


def test_trainer_checkpoint_metadata_records_kl_scope_and_weights() -> None:
    """Generic Trainer checkpoint sidecars must preserve enough KL metadata
    to tell primary forensic KL from scoped SegNet-aux KL after harvest."""
    src = (Path(__file__).resolve().parents[1] / "training.py").read_text()
    for key in (
        '"kl_distill_scope"',
        '"kl_distill_weight"',
        '"kl_distill_temperature"',
        '"allow_banned_primary_kl_distill"',
        '"promotion_eligible"',
        '"distillation_policy"',
    ):
        assert key in src


def test_train_config_distillation_policy_provenance_records_schema() -> None:
    cfg = _make_kl_config(kl_weight=0.005)

    provenance = cfg.distillation_policy_provenance()
    assert provenance["format"] == "distillation_policy_v1"
    assert provenance["family"] == "segnet_aux_kl"
    assert provenance["scope"] == "segnet_aux"
    assert provenance["weight"] == 0.005
    assert provenance["temperature"] == 2.0
    assert provenance["promotion_capable"] is True


def test_train_config_zero_weight_kl_serializes_as_inactive_policy() -> None:
    cfg = _make_kl_config(kl_weight=0.0)

    provenance = cfg.distillation_policy_provenance()
    assert provenance["format"] == "distillation_policy_v1"
    assert provenance["family"] == "none"
    assert provenance["scope"] == "none"
    assert provenance["weight"] == 0.0
    assert provenance["promotion_blockers"] == []


def test_train_config_accepts_non_default_kl_weight() -> None:
    """The operator MUST be able to override kl_distill_weight (this was the
    silent-drop bug — the field didn't exist so override was a no-op)."""
    cfg = _make_kl_config(kl_weight=0.005)
    assert cfg.kl_distill_weight == 0.005


def test_train_config_rejects_negative_kl_weight() -> None:
    """Defense in depth: weight must be >= 0."""
    import pytest
    with pytest.raises(Exception):
        TrainConfig(
            hidden=8, epochs=100, warmup_epochs=10, tag="t",
            eval_roundtrip=True, loss_mode="standard",
            kl_distill_weight=-0.1,
        )


def test_active_kl_temperature_fails_before_trainer_construction() -> None:
    """The policy validator must catch the active KL temperature, not just
    the primary-loss annealing schedule."""
    import pytest

    with pytest.raises(Exception, match="temperature >= 2.0"):
        TrainConfig(
            hidden=8,
            epochs=100,
            warmup_epochs=10,
            tag="kl-temp-too-low",
            eval_roundtrip=True,
            loss_mode="kl_distill",
            kl_distill_scope="segnet_aux",
            kl_distill_weight=0.002,
            kl_distill_temperature=1.0,
            temperature_start=2.0,
            temperature_end=1.0,
        )


def test_segmap_trainer_rejects_primary_kl_scope() -> None:
    """SegMapTrainer only implements the SegNet-auxiliary KL path.

    A forensic ``primary_scorer`` config is valid at the TrainConfig layer
    only so legacy primary KL can be audited. It must not be silently routed
    through SegMapTrainer, where ``loss_mode='kl_distill'`` means
    ``standard scorer loss + SegNet-only KL auxiliary``.
    """
    import pytest

    cfg = TrainConfig(
        hidden=8,
        epochs=100,
        warmup_epochs=10,
        tag="primary-kl-forensic",
        eval_roundtrip=True,
        loss_mode="kl_distill",
        kl_distill_scope="primary_scorer",
        allow_banned_primary_kl_distill=True,
        promotion_eligible=False,
        forensic_reason="segmap must reject primary scorer KL routing",
        temperature_start=2.0,
        temperature_end=1.0,
    )
    with pytest.raises(PreflightError, match="only kl_distill_scope='segnet_aux'"):
        SegMapTrainer(_make_tiny_segmap(), cfg, _MockPoseNet(), _MockSegNet(), device="cpu")


def test_train_config_rejects_unfenced_segnet_kl() -> None:
    """Legacy loss_mode='segnet_kl' must not silently stay promotion-capable.

    Grand Council KL hardening found this was the remaining KL-like path not
    covered by the explicit scope/promotion contract.
    """
    import pytest

    with pytest.raises(Exception, match="kl_distill_scope='segnet_aux'"):
        TrainConfig(
            hidden=8,
            epochs=100,
            warmup_epochs=10,
            tag="segnet-kl-unfenced",
            eval_roundtrip=True,
            loss_mode="segnet_kl",
        )


def test_train_config_accepts_forensic_segnet_kl_only_when_non_promotable() -> None:
    cfg = TrainConfig(
        hidden=8,
        epochs=100,
        warmup_epochs=10,
        tag="segnet-kl-forensic",
        eval_roundtrip=True,
        loss_mode="segnet_kl",
        kl_distill_scope="segnet_aux",
        promotion_eligible=False,
        forensic_reason="legacy SegNet-KL unit-test forensic path",
    )

    assert cfg.loss_mode == "segnet_kl"
    assert cfg.kl_distill_scope == "segnet_aux"
    assert cfg.promotion_eligible is False


def test_segnet_kl_profiles_are_forensic_only() -> None:
    from tac.profiles import SEGNET_KL_FULL, SEGNET_KL_SMOKE

    for profile in (SEGNET_KL_SMOKE, SEGNET_KL_FULL):
        assert profile["loss_mode"] == "segnet_kl"
        assert profile["kl_distill_scope"] == "segnet_aux"
        assert profile["promotion_eligible"] is False


def test_train_segmap_film_canvas_kl_variant_sets_explicit_scope() -> None:
    """FilmCanvas reuses SegMapTrainer, so it must satisfy the same explicit
    KL scope contract as experiments/train_segmap.py."""
    script = Path(__file__).resolve().parents[3] / "experiments" / "train_segmap_film_canvas.py"
    spec = importlib.util.spec_from_file_location("train_segmap_film_canvas_under_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    args = argparse.Namespace(
        variant="kl_distill",
        roundtrip_noise_std=0.5,
        epochs=120,
        batch_size=2,
        lr=1e-3,
        weight_decay=1e-4,
        ema_decay=0.997,
        tag="film_canvas_kl_test",
        output_dir="experiments/results/film_canvas_kl_test",
        bf16=False,
        scorer_chunk=0,
        kl_distill_weight=0.002,
        kl_distill_temperature=2.0,
    )

    cfg = module._build_trainer_config(args, torch.device("cpu"))
    assert cfg.loss_mode == "kl_distill"
    assert cfg.kl_distill_scope == "segnet_aux"
    assert cfg.kl_distill_weight == 0.002


def _train_and_get_loss(kl_weight: float, seed: int = 1234) -> float:
    """Run a single train_epoch in kl_distill mode at the given weight and
    return the final loss. Setup is fully seeded so the only difference
    across calls is kl_weight (and therefore the loss is monotonic in
    kl_weight if the wiring is correct)."""
    torch.manual_seed(seed)
    cfg = _make_kl_config(kl_weight=kl_weight)
    h = SEGMAP_INPUT_SIZE[1] // 16
    w = SEGMAP_INPUT_SIZE[0] // 16

    torch.manual_seed(seed)
    model = _make_tiny_segmap()
    torch.manual_seed(seed)
    posenet, segnet = _MockPoseNet(), _MockSegNet()
    trainer = SegMapTrainer(model, cfg, posenet, segnet, device="cpu")

    b = 1
    t = 2
    masks = F.softmax(torch.randn(b, t, 5, h, w), dim=2)
    gt = torch.rand(b, t, h, w, 3) * 255.0

    torch.manual_seed(seed)
    stats = trainer.train_epoch(masks, gt, ema=None)
    return float(stats["loss"])


def test_kl_weight_is_threaded_into_loss_not_hardcoded() -> None:
    """Empirical proof that the operator-supplied kl_distill_weight changes
    the loss. With the hard-coded 0.002 (pre-fix), changing the config
    weight would have NO effect — the loss would be identical for any
    weight. With the fix, the loss differs.

    We use kl_weight=0.0 vs kl_weight=10.0 — extreme values where the
    KL-loss contribution should be either zero or large. The empirical
    delta MUST be non-trivial (>1e-4) to prove the wiring.

    [empirical:src/tac/tests/test_kl_distill_weight_plumbed.py]
    """
    loss_zero = _train_and_get_loss(kl_weight=0.0, seed=1234)
    loss_large = _train_and_get_loss(kl_weight=10.0, seed=1234)
    assert math.isfinite(loss_zero), f"loss_zero not finite: {loss_zero}"
    assert math.isfinite(loss_large), f"loss_large not finite: {loss_large}"
    delta = abs(loss_large - loss_zero)
    assert delta > 1e-4, (
        f"kl_distill_weight is not threaded into the loss: "
        f"loss(weight=0.0)={loss_zero:.6f} vs loss(weight=10.0)={loss_large:.6f}, "
        f"delta={delta:.2e}. Pre-fix segmap_renderer.py:667 hardcoded "
        f"`0.002 * kl_loss` so this delta would be zero."
    )


def test_kl_weight_zero_isolates_to_standard_loss() -> None:
    """When kl_distill_weight=0.0, the KL term vanishes from the loss
    arithmetic (loss = standard_loss + 0 * kl_loss). This is a stronger
    statement than the previous test — proves the *direction* of the
    threading, not just non-zero presence."""
    # The bare-standard loss (loss_mode='standard' so no KL term at all).
    torch.manual_seed(1234)
    cfg_std = TrainConfig(
        hidden=8, epochs=100, warmup_epochs=10, tag="t",
        eval_roundtrip=True, loss_mode="standard",
    )
    h = SEGMAP_INPUT_SIZE[1] // 16
    w = SEGMAP_INPUT_SIZE[0] // 16
    torch.manual_seed(1234)
    model_std = _make_tiny_segmap()
    torch.manual_seed(1234)
    posenet_std, segnet_std = _MockPoseNet(), _MockSegNet()
    trainer_std = SegMapTrainer(model_std, cfg_std, posenet_std, segnet_std, device="cpu")
    b = 1
    t = 2
    torch.manual_seed(99)
    masks = F.softmax(torch.randn(b, t, 5, h, w), dim=2)
    gt = torch.rand(b, t, h, w, 3) * 255.0
    torch.manual_seed(123)
    stats_std = trainer_std.train_epoch(masks, gt, ema=None)
    # The KL-distill loss with weight=0.0 should equal the standard-mode
    # loss to within numerical precision. (If the hard-coded 0.002 were
    # still in the file, the loss would include the 0.002*kl_loss term
    # and DIFFER from standard.)
    torch.manual_seed(1234)
    cfg_kl0 = _make_kl_config(kl_weight=0.0)
    torch.manual_seed(1234)
    model_kl0 = _make_tiny_segmap()
    torch.manual_seed(1234)
    posenet_kl0, segnet_kl0 = _MockPoseNet(), _MockSegNet()
    trainer_kl0 = SegMapTrainer(model_kl0, cfg_kl0, posenet_kl0, segnet_kl0, device="cpu")
    torch.manual_seed(123)
    stats_kl0 = trainer_kl0.train_epoch(masks, gt, ema=None)

    delta = abs(stats_std["loss"] - stats_kl0["loss"])
    assert delta < 1e-3, (
        f"loss(standard) vs loss(kl_distill, weight=0.0) should match "
        f"within 1e-3, got delta={delta:.6f}. Indicates the KL term is "
        f"still being added regardless of the operator-supplied weight."
    )
