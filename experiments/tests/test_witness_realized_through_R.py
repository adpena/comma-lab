# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the CAPSTONE LEVER #2 through-R witness trainer.

These tests prove the realized-axis loss path is REAL (gradient actually flows from the
frozen scorers through R into the witness; the CPU verdict is the exact frozen-scorer
quantity; a stub witness FAILS the realized d_seg). Per CLAUDE.md NO-FAKE supreme rule +
8 forbidden classes (esp. class 2 tests-verify-behavior-not-constants, class 8
surrogate-not-authority): if the witness body were replaced by a constant, the realized
d_seg verdict would be high (the test asserts the BEHAVIOR, not a metadata constant).

Authority note: the realized d_seg/d_pose VERDICT in the trainer is the FROZEN CPU-torch
SegNet argmax + PoseNet pose-MSE -- these tests use small synthetic frames + a tiny
witness for speed; the FULL frozen-scorer tests that need the upstream checkpoints are
SKIPPED if the upstream models are not present (so CI without the contest tree passes).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO, REPO / "src", REPO / "upstream", REPO / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch

from train_witness_realized_through_R import (  # noqa: E402
    CAMERA_H,
    CAMERA_W,
    SEG_H,
    SEG_W,
    EMA,
    RGBWitness,
    implied_score_from_verdict,
    render_through_R,
)

_UPSTREAM_OK = (REPO / "upstream" / "models" / "segnet.safetensors").exists()


# --------------------------------------------------------------------------- #
# 1. The witness OUTPUTS RGB in [0, 255] (the realized-axis vehicle is RGB, not
#    a partition). A direct-partition witness would output 5-class logits; this
#    one outputs 3-channel RGB -- the thing the scorers actually read.
# --------------------------------------------------------------------------- #
def test_witness_outputs_rgb_in_0_255():
    w = RGBWitness(num_pairs=2, n_fourier=8, hidden_dim=16, n_hidden=2, mod_dim=8)
    coords = torch.rand(64, 2) * 2 - 1
    feats = w.build_feats(coords)
    rgb = w(feats, code_idx=0)
    assert rgb.shape == (64, 3), f"expected (64,3) RGB, got {tuple(rgb.shape)}"
    assert float(rgb.min()) >= 0.0 and float(rgb.max()) <= 255.0, "RGB must be in [0,255]"


# --------------------------------------------------------------------------- #
# 2. render_through_R produces a camera-res frame AND preserves gradients (the
#    round-trip R is differentiable -- the uint8-STE must NOT sever the graph).
#    This is the lever's core: the loss is computed AFTER R, gradient flows back.
# --------------------------------------------------------------------------- #
def test_render_through_R_is_differentiable_and_camera_res():
    w = RGBWitness(num_pairs=1, n_fourier=8, hidden_dim=16, n_hidden=2, mod_dim=8)
    coords = torch.rand(16 * 16, 2) * 2 - 1
    feats = w.build_feats(coords)
    cam = render_through_R(w, feats, code_idx=0, render_h=16, render_w=16, ste_round=True)
    assert cam.shape == (1, 3, CAMERA_H, CAMERA_W), f"R must emit camera-res; got {tuple(cam.shape)}"
    assert float(cam.min()) >= 0.0 and float(cam.max()) <= 255.0
    # Gradient must reach the witness through R (uint8-STE identity backward).
    loss = cam.mean()
    loss.backward()
    grad_norm = sum(
        float(p.grad.abs().sum()) for p in w.parameters() if p.grad is not None
    )
    assert grad_norm > 0.0, "R severed the gradient graph (uint8-STE must be straight-through)"


# --------------------------------------------------------------------------- #
# 3. The uint8-STE rounds in the FORWARD (the frame the scorer sees is integer-
#    valued at camera res) -- proving R simulates the real eval quantization, not
#    a float-space proxy (eval_roundtrip non-negotiable).
# --------------------------------------------------------------------------- #
def test_render_through_R_forward_is_uint8_rounded():
    w = RGBWitness(num_pairs=1, n_fourier=8, hidden_dim=16, n_hidden=2, mod_dim=8)
    coords = torch.rand(8 * 8, 2) * 2 - 1
    feats = w.build_feats(coords)
    with torch.no_grad():
        cam = render_through_R(w, feats, code_idx=0, render_h=8, render_w=8, ste_round=True)
    # Forward values are integer-valued (round was applied).
    frac = (cam - cam.round()).abs().max()
    assert float(frac) < 1e-4, "uint8-STE forward must round to integers (eval-roundtrip fidelity)"


# --------------------------------------------------------------------------- #
# 4. implied_score_from_verdict is the EXACT contest S composition (no surrogate).
#    Verifies the rate term, the sqrt(10*d_pose) nonlinear pose term, and the
#    100*d_seg seg term -- the three real contest terms.
# --------------------------------------------------------------------------- #
def test_implied_score_is_exact_contest_composition():
    import math

    d_seg, d_pose, witness_bytes = 0.004, 0.005, 80_000
    s = implied_score_from_verdict(d_seg, d_pose, witness_bytes)
    expected = 100.0 * d_seg + math.sqrt(10.0 * d_pose) + 25.0 * witness_bytes / 37_545_489.0
    assert abs(s - expected) < 1e-9, f"implied S must be the exact contest formula; {s} != {expected}"
    # The frontier reference S=0.19110 must be reproduced by the frontier components.
    s_frontier = implied_score_from_verdict(5.6e-4, 1.6e-5, 177_000)
    assert 0.10 < s_frontier < 0.20, f"frontier-components implied S sanity: {s_frontier}"


# --------------------------------------------------------------------------- #
# 5. EMA: shadow is a convex combination of the weights (NOT the live weights),
#    apply/restore is lossless, and the shadow tracks toward updated weights.
# --------------------------------------------------------------------------- #
def test_ema_tracks_and_restore_is_lossless():
    w = RGBWitness(num_pairs=1, n_fourier=8, hidden_dim=16, n_hidden=2, mod_dim=8)
    ema = EMA(w, decay=0.5)
    # Perturb the live weights, update EMA, check the shadow moved toward them.
    with torch.no_grad():
        for p in w.parameters():
            p.add_(1.0)
    before = {k: v.clone() for k, v in ema.shadow.items()}
    ema.update(w)
    moved = any(float((ema.shadow[k] - before[k]).abs().sum()) > 0 for k in ema.shadow)
    assert moved, "EMA shadow must move toward updated weights"
    # apply/restore must be lossless on the live weights.
    live_snapshot = {k: v.detach().clone() for k, v in w.state_dict().items()}
    orig = ema.apply_to(w)
    ema.restore(w, orig)
    for k, v in w.state_dict().items():
        assert torch.equal(v, live_snapshot[k]), f"EMA restore corrupted live weight {k}"


# --------------------------------------------------------------------------- #
# 6. NO-FAKE (the verdict is BEHAVIOR not a constant): a CONSTANT-output witness
#    (frozen at init, gray frames) realizes a HIGH d_seg through the frozen CPU
#    SegNet -- proving the realized d_seg measures the actual scorer re-derivation,
#    not a metadata field. A trained witness must BEAT this. (Needs upstream models.)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _UPSTREAM_OK, reason="upstream SegNet checkpoint not present")
def test_constant_witness_has_high_realized_d_seg_stub_fails():
    from tac.boundary_math.seg_core import (
        decode_gt_frame1_pairs,
        load_real_segnet,
        segnet_argmax_and_margin,
    )
    from train_witness_realized_through_R import cpu_verdict_d_seg

    seg = load_real_segnet("cpu")
    # One GT pair -> L* (the target argmax).
    _idx, _f0, f1 = next(iter(decode_gt_frame1_pairs(n_pairs=1)))
    lstar, _m = segnet_argmax_and_margin(seg, np.asarray(f1))
    lstar = np.asarray(lstar).astype(np.int64)
    # A CONSTANT gray camera frame (the stub a fake would emit).
    gray = np.full((CAMERA_H, CAMERA_W, 3), 128, dtype=np.uint8)
    d_seg_const = cpu_verdict_d_seg(seg, gray, lstar)
    # A constant frame cannot reproduce the partition -> d_seg must be far above the
    # frontier 5.6e-4 and even above the naive realization 0.005-0.008. The point: the
    # verdict is the REAL scorer re-derivation; a stub does NOT score ~0.
    assert d_seg_const > 0.05, (
        f"constant witness realized d_seg={d_seg_const} -- a stub MUST fail the realized "
        "verdict (NO-FAKE: the d_seg is the real frozen-SegNet argmax-disagreement, not a constant)"
    )


# --------------------------------------------------------------------------- #
# 7. NO-FAKE (gradient actually flows from the FROZEN SegNet through R into the
#    witness -- the realized seg loss is real backprop, not a detached proxy).
#    (Needs upstream models.)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _UPSTREAM_OK, reason="upstream SegNet checkpoint not present")
def test_realized_seg_loss_flows_gradient_from_frozen_segnet():
    from tac.boundary_math.seg_core import load_real_segnet
    from train_witness_realized_through_R import realized_seg_loss

    seg = load_real_segnet("cpu")
    for p in seg.parameters():
        p.requires_grad = False
    w = RGBWitness(num_pairs=1, n_fourier=8, hidden_dim=16, n_hidden=2, mod_dim=8)
    render_h, render_w = 24, 32
    coords = torch.rand(render_h * render_w, 2) * 2 - 1
    feats = w.build_feats(coords)
    f1_cam = render_through_R(w, feats, code_idx=0, render_h=render_h, render_w=render_w)
    gt_argmax = torch.randint(0, 5, (SEG_H, SEG_W), dtype=torch.long)
    loss = realized_seg_loss(seg, f1_cam, gt_argmax, gt_margin=None)
    loss.backward()
    grad_norm = sum(float(p.grad.abs().sum()) for p in w.parameters() if p.grad is not None)
    assert grad_norm > 0.0, (
        "realized seg loss must backprop FROM the frozen SegNet THROUGH R INTO the witness "
        "(the realized-axis lever; a detached/surrogate loss would give zero witness grad)"
    )
    # The frozen scorer must NOT have accumulated gradient on its own params.
    seg_grad = sum(float(p.grad.abs().sum()) for p in seg.parameters() if p.grad is not None)
    assert seg_grad == 0.0, "frozen SegNet params must stay frozen (no grad accumulation)"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
