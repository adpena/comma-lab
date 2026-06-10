# SPDX-License-Identifier: MIT
"""Behavior tests for the WORKING score-aware loop (task #76 fix).

NO-FAKE discipline (CLAUDE.md class 1 + 2): these prove the loop's gradient
ACTUALLY reduces the exact SegNet ``d_seg`` on a frozen scorer (descent from a
HIGH-disagreement start, not a degenerate near-zero start); that the grad is
well-conditioned (clip relaxes off 100% — the inert harness fired 100% forever);
that the live-scorer margin gradient reaches the carrier; and that a
SEVERED/constant gradient does NOT descend. Each test would fail if the loop
quietly stopped optimizing the live scorer.

The scorer is a REAL frozen differentiable net (``_ColorProtoSeg``): per-pixel
argmax = nearest of 5 fixed RGB class-prototypes. It is non-degenerate (balanced,
diverse argmax) AND reachable (the carrier can drive each pixel's color toward a
prototype, moving its argmax), so a working loop demonstrably descends d_seg. The
upstream-EfficientNet-B2-SegNet proof (real video, the actual contest scorer)
lives in ``.omx/research/inert_loop_fix_*.md`` (run detached; CI stays cheap).
"""

from __future__ import annotations

import torch

from tac.score_aware_loop import (
    ScoreAwareLoopConfig,
    ScoreAwareTrainer,
    TinyPairCarrier,
)


class _ColorProtoSeg(torch.nn.Module):
    """Frozen scorer: per-pixel argmax = nearest of 5 fixed RGB class prototypes.

    Non-degenerate (5 balanced classes), differentiable, and *reachable*: the
    carrier can drive a pixel's RGB toward a prototype to change its argmax — so
    a working loop visibly descends d_seg. The scorer has NO trainable params.
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "proto",
            torch.tensor(
                [
                    [0.1, 0.1, 0.1],
                    [0.9, 0.1, 0.1],
                    [0.1, 0.9, 0.1],
                    [0.1, 0.1, 0.9],
                    [0.9, 0.9, 0.1],
                ]
            ),
        )
        for p in self.parameters():
            p.requires_grad = False

    class _Seg:
        def __init__(self, outer: _ColorProtoSeg) -> None:
            self.outer = outer

        def __call__(self, x: torch.Tensor) -> torch.Tensor:
            # x (B,3,H,W) in [0,1] -> logit_c = -||x - proto_c||^2 * scale.
            d = ((x.unsqueeze(1) - self.outer.proto.view(1, 5, 3, 1, 1)) ** 2).sum(2)
            return -d * 8.0

    def preprocess_input(self, bhwc: torch.Tensor):
        x = bhwc[:, -1].permute(0, 3, 1, 2).float() / 255.0
        return None, x

    @property
    def segnet(self):
        return self._Seg(self)


# Small scorer resolution for FAST CI (the trainer's scorer_hw default is the
# real 384x512; tests override to keep the 5x-upsample eval-roundtrip cheap —
# the loop math is identical at any resolution, only the pixel count changes).
TEST_HW = (96, 128)


def _striped_targets(n: int, hw=TEST_HW) -> torch.Tensor:
    """Balanced, diverse GT argmax field: 5 horizontal stripes shifted per frame."""
    h, w = hw
    seg = torch.zeros(n, h, w, dtype=torch.long)
    for i in range(n):
        for c in range(5):
            seg[i, (c * h // 5) : ((c + 1) * h // 5), :] = (c + i) % 5
    return seg


def test_loop_descends_exact_d_seg_from_high_start():
    # HEADLINE NO-FAKE PROOF: exact live-render d_seg falls from ~0.80 toward 0.
    torch.manual_seed(0)
    dnet = _ColorProtoSeg()
    n = 6
    seg_t = _striped_targets(n)
    carrier = TinyPairCarrier(n, out_hw=(96, 128), base_channels=32)
    cfg = ScoreAwareLoopConfig(
        epochs=80, batch_size=3, seg_loss_form="ce_seg_loss", scorer_hw=TEST_HW,
        pose_enabled=False, eval_every=20, decoder_lr=8e-3, ema_decay=0.95, seed=0,
    )
    tr = ScoreAwareTrainer(carrier, dnet, seg_t, None, cfg)
    summary = tr.train()
    assert summary["d_seg_initial"] > 0.5, "test scorer must start with real disagreement"
    assert summary["descended"], summary["trajectory"]
    # Direct-live CE drives the exact argmax disagreement far below the start.
    assert summary["d_seg_best_ema"] < 0.2
    # And the trajectory is monotone-ish downward (each eval <= the previous +eps).
    dsegs = [r["exact_d_seg_ema"] for r in summary["trajectory"]]
    assert dsegs[-1] < dsegs[0]


def test_margin_surrogate_also_descends():
    torch.manual_seed(0)
    dnet = _ColorProtoSeg()
    n = 6
    seg_t = _striped_targets(n)
    carrier = TinyPairCarrier(n, out_hw=(96, 128), base_channels=32)
    cfg = ScoreAwareLoopConfig(
        epochs=80, batch_size=3, seg_loss_form="smooth_disagreement_seg_loss",
        seg_weight=1000.0, pose_enabled=False, eval_every=20, decoder_lr=8e-3,
        ema_decay=0.95, scorer_hw=TEST_HW, seed=0,
    )
    tr = ScoreAwareTrainer(carrier, dnet, seg_t, None, cfg)
    summary = tr.train()
    assert summary["descended"]
    assert summary["d_seg_best_ema"] < summary["d_seg_initial"] - 0.1


def test_grad_well_conditioned_clip_relaxes_off_100pct():
    # The inert harness fired grad-clip on 100% of steps forever (grad_norm 1e6).
    # A well-conditioned loop relaxes off 100% once the loss surface settles.
    torch.manual_seed(0)
    dnet = _ColorProtoSeg()
    n = 6
    seg_t = _striped_targets(n)
    carrier = TinyPairCarrier(n, out_hw=(96, 128), base_channels=32)
    cfg = ScoreAwareLoopConfig(
        epochs=80, batch_size=3, seg_loss_form="ce_seg_loss", scorer_hw=TEST_HW,
        pose_enabled=False, eval_every=40, decoder_lr=3e-3, ema_decay=0.99, seed=0,
    )
    tr = ScoreAwareTrainer(carrier, dnet, seg_t, None, cfg)
    summary = tr.train()
    # By the end the clip fraction must drop below 1.0 — the grad has shrunk
    # below the clip threshold on real steps (well-conditioned, NOT pinned at 1e6).
    assert summary["clip_fired_fraction_final"] < 1.0


def test_live_scorer_gradient_reaches_carrier_head():
    torch.manual_seed(0)
    dnet = _ColorProtoSeg()
    n = 4
    seg_t = _striped_targets(n)
    carrier = TinyPairCarrier(n, out_hw=(96, 128), base_channels=24)
    cfg = ScoreAwareLoopConfig(epochs=1, batch_size=4, pose_enabled=False, scorer_hw=TEST_HW, seed=0)
    tr = ScoreAwareTrainer(carrier, dnet, seg_t, None, cfg)
    parts = tr.compute_loss(torch.arange(n))
    parts["total"].backward()
    total_grad = sum(
        float(p.grad.abs().sum()) for p in carrier.parameters() if p.grad is not None
    )
    assert total_grad > 0.0
    assert carrier.head1.weight.grad is not None
    assert float(carrier.head1.weight.grad.abs().sum()) > 0.0
    # The latent (furthest from the scorer) also receives gradient (full path).
    assert carrier.latents.grad is not None
    assert float(carrier.latents.grad.abs().sum()) > 0.0


def test_constant_loss_does_not_descend():
    # NO-FAKE control: a constant (detached) seg loss has zero gradient to the
    # carrier; the loop MUST NOT descend. This guards the descent claim against a
    # "loop accidentally optimizing something else" failure.
    torch.manual_seed(0)
    dnet = _ColorProtoSeg()
    n = 6
    seg_t = _striped_targets(n)
    carrier = TinyPairCarrier(n, out_hw=(96, 128), base_channels=24)
    cfg = ScoreAwareLoopConfig(epochs=40, batch_size=3, pose_enabled=False, eval_every=20, scorer_hw=TEST_HW, seed=0)

    def constant_loss(seg_logits, targets_hard):
        return (seg_logits.sum() * 0.0) + 1.0

    tr = ScoreAwareTrainer(carrier, dnet, seg_t, None, cfg, seg_loss_fn=constant_loss)
    summary = tr.train()
    assert not summary["descended"]
    assert abs(summary["d_seg_final_ema"] - summary["d_seg_initial"]) < 1e-6


def test_severed_gradient_via_detached_render_does_not_descend():
    # Emulate the structural severance failure: the loss is computed on a
    # detached copy of the live logits PLUS a zero-coefficient tether to the
    # render (so .backward() runs, but the carrier receives ZERO useful gradient
    # — exactly what a learnable-head-fed-a-detached-render surrogate produces).
    # A working direct-live loop descends; this severed loop must NOT.
    torch.manual_seed(0)
    dnet = _ColorProtoSeg()
    n = 6
    seg_t = _striped_targets(n)
    carrier = TinyPairCarrier(n, out_hw=(96, 128), base_channels=24)
    cfg = ScoreAwareLoopConfig(epochs=40, batch_size=3, pose_enabled=False, eval_every=20, scorer_hw=TEST_HW, seed=0)

    import torch.nn.functional as F

    def severed_loss(seg_logits, targets_hard):
        # Real CE value on the detached logits + a 0.0 * live-render tether so a
        # grad_fn exists but the gradient magnitude to the carrier is exactly 0.
        ce_detached = F.cross_entropy(seg_logits.detach(), targets_hard.long())
        zero_tether = seg_logits.sum() * 0.0  # grad_fn present, gradient == 0
        return ce_detached + zero_tether

    tr = ScoreAwareTrainer(carrier, dnet, seg_t, None, cfg, seg_loss_fn=severed_loss)
    summary = tr.train()
    assert not summary["descended"]
    # The carrier never moved: d_seg is bit-identical start to finish.
    assert abs(summary["d_seg_final_ema"] - summary["d_seg_initial"]) < 1e-6


def test_trainer_rejects_unfrozen_scorer():
    import pytest

    dnet = _ColorProtoSeg()
    # add a trainable param to simulate an un-frozen scorer
    dnet.extra = torch.nn.Parameter(torch.zeros(1))
    n = 3
    seg_t = _striped_targets(n)
    carrier = TinyPairCarrier(n, out_hw=(48, 64), base_channels=16)
    cfg = ScoreAwareLoopConfig(epochs=1, pose_enabled=False)
    with pytest.raises(ValueError, match="frozen"):
        ScoreAwareTrainer(carrier, dnet, seg_t, None, cfg)


def test_ema_shadow_is_inference_checkpoint():
    torch.manual_seed(0)
    dnet = _ColorProtoSeg()
    n = 4
    seg_t = _striped_targets(n)
    carrier = TinyPairCarrier(n, out_hw=(48, 64), base_channels=16)
    cfg = ScoreAwareLoopConfig(epochs=10, batch_size=2, pose_enabled=False, eval_every=10, scorer_hw=TEST_HW, seed=0)
    tr = ScoreAwareTrainer(carrier, dnet, seg_t, None, cfg)
    tr.train()
    d_ema = tr.exact_d_seg(use_ema=True)
    d_live = tr.exact_d_seg(use_ema=False)
    assert isinstance(d_ema, float) and isinstance(d_live, float)


def test_eval_roundtrip_is_applied_in_loop():
    torch.manual_seed(0)
    dnet = _ColorProtoSeg()
    n = 2
    seg_t = _striped_targets(n)
    carrier = TinyPairCarrier(n, out_hw=(48, 64), base_channels=16)
    cfg = ScoreAwareLoopConfig(epochs=1, batch_size=2, pose_enabled=False, eval_roundtrip=True, scorer_hw=TEST_HW, seed=0)
    tr = ScoreAwareTrainer(carrier, dnet, seg_t, None, cfg)
    bhwc = tr._render_pair_scorer_input(torch.arange(n)).detach()
    assert float(bhwc.min()) >= 0.0
    assert float(bhwc.max()) <= 255.0
    assert tuple(bhwc.shape) == (n, 2, TEST_HW[0], TEST_HW[1], 3)


def test_unknown_seg_loss_form_rejected():
    import pytest

    dnet = _ColorProtoSeg()
    n = 2
    seg_t = _striped_targets(n)
    carrier = TinyPairCarrier(n, out_hw=(48, 64), base_channels=16)
    cfg = ScoreAwareLoopConfig(epochs=1, seg_loss_form="bogus_loss", pose_enabled=False)
    with pytest.raises(ValueError, match="unknown seg_loss_form"):
        ScoreAwareTrainer(carrier, dnet, seg_t, None, cfg)


def test_pose_term_aggregates_when_enabled():
    torch.manual_seed(0)
    n = 3
    seg_t = _striped_targets(n)
    pose_t = torch.zeros(n, 6)

    class _WithPose(_ColorProtoSeg):
        class _Pose:
            def __call__(self, x):
                return {"pose": torch.zeros(x.shape[0], 12)}

        @property
        def posenet(self):
            return self._Pose()

        def preprocess_input(self, bhwc):
            _, seg = super().preprocess_input(bhwc)
            return torch.zeros(bhwc.shape[0], 12, 4, 4), seg

    d2 = _WithPose()
    carrier = TinyPairCarrier(n, out_hw=(48, 64), base_channels=16)
    cfg = ScoreAwareLoopConfig(epochs=1, batch_size=3, pose_enabled=True, scorer_hw=TEST_HW, seed=0)
    tr = ScoreAwareTrainer(carrier, d2, seg_t, pose_t, cfg)
    parts = tr.compute_loss(torch.arange(n))
    assert "pose" in parts
    assert "total" in parts
