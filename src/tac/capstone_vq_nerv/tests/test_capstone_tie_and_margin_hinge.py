# SPDX-License-Identifier: MIT
"""Tests for the L1 weight-tie + cross-hardware margin hinge + CLI passthrough.

The capstone-build deliverable (``.omx/research/optimal_capstone_vehicle_spec_
20260611.md`` section 6 steps 1-3). Three NO-FAKE properties:

1. **Weight-tie** (``tie_depth``): the leading ``base_ch->base_ch`` blocks ACTUALLY
   share ONE conv (fewer STORED/exported params) AND the numpy inflate reproduces
   the tied MLX render op-for-op (the grid-PE fake-parity lesson: a no-op/zero tie
   that did not really share would FAIL the param-reduction test; a numpy path that
   did not dispatch the tied stages would FAIL the parity test). ``tie_depth<=1`` is
   byte-identical to the untied decoder.

2. **Margin hinge**: a REAL loss term penalizing small-margin-correct pixels with a
   REAL gradient that pushes the margin up; zero on clear-of-floor pixels; a no-op
   at ``hinge_weight=0`` (so default-off is provably inert).

3. **CLI passthrough**: the campaign argparse exposes ``--tie-depth`` /
   ``--hinerv-grid-pe`` / ``--grid-pe-num-freqs`` (+ the hinge flags) and threads
   them into the bundle/train configs + the inflate config sidecar.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tac.capstone_vq_nerv.cross_hw_margin_hinge import (
    CrossHwMarginHingeSegLoss,
    margin_floor_hinge,
)
from tac.score_aware_loop.live_segnet_loss import ce_seg_loss

try:
    import mlx.core as mx
    from mlx.utils import tree_flatten

    _HAVE_MLX = True
except Exception:  # pragma: no cover
    mx = None  # type: ignore[assignment]
    tree_flatten = None  # type: ignore[assignment]
    _HAVE_MLX = False

skip_no_mlx = pytest.mark.skipif(not _HAVE_MLX, reason="mlx not available")


def _exported_param_count(weights: dict[str, np.ndarray]) -> int:
    return sum(int(np.prod(v.shape)) for v in weights.values())


# ---------------------------------------------------------------------------
# (1) L1 weight-tie
# ---------------------------------------------------------------------------


@skip_no_mlx
def test_tieable_count_matches_canonical_taper():
    from tac.capstone_vq_nerv.vq_nerv_bundle import tieable_leading_block_count

    # base_ch=24 taper [24,24,24,18,13,12,12]: only blocks 0,1 are base_ch->base_ch.
    assert tieable_leading_block_count([24, 24, 24, 18, 13, 12, 12]) == 2
    assert tieable_leading_block_count([20, 20, 20, 15, 11, 10, 10]) == 2


@skip_no_mlx
def test_tie_depth_le_1_is_byte_identical_to_untied():
    """tie_depth<=1 must render BYTE-IDENTICAL to the untied decoder (no-op default)."""
    from tac.capstone_vq_nerv.vq_nerv_bundle import (
        CapstoneVqNervBundle,
        CapstoneVqNervConfig,
    )

    b0 = CapstoneVqNervBundle(
        CapstoneVqNervConfig(num_pairs=4, base_channels=24, codebook_size=16, seed=0, tie_depth=0)
    )
    b1 = CapstoneVqNervBundle(
        CapstoneVqNervConfig(num_pairs=4, base_channels=24, codebook_size=16, seed=0, tie_depth=1)
    )
    idx = mx.arange(4)
    r0 = np.asarray(b0(idx, pose=None))
    r1 = np.asarray(b1(idx, pose=None))
    assert float(np.max(np.abs(r0 - r1))) == 0.0
    # tie_depth=1 must NOT build the shared conv (no extra stored params).
    assert not hasattr(b1, "tied_conv")


@skip_no_mlx
def test_tie_actually_shares_weights_fewer_stored_params():
    """NO-FAKE: the tie ACTUALLY removes conv tensors from the EXPORTED basis.

    A fake tie (still storing the per-block convs, or a no-op) would NOT reduce the
    stored param count. We assert the exported render-basis param count DROPS and
    that the dropped tensors are exactly the leading per-block convs, replaced by
    the ONE shared tied_conv + the per-stage FiLMs.
    """
    from tac.capstone_vq_nerv.numpy_reference import full_render_weights_from_bundle
    from tac.capstone_vq_nerv.vq_nerv_bundle import (
        CapstoneVqNervBundle,
        CapstoneVqNervConfig,
    )

    b0 = CapstoneVqNervBundle(
        CapstoneVqNervConfig(num_pairs=4, base_channels=24, codebook_size=16, seed=0, tie_depth=0)
    )
    b2 = CapstoneVqNervBundle(
        CapstoneVqNervConfig(num_pairs=4, base_channels=24, codebook_size=16, seed=0, tie_depth=2)
    )
    w0 = full_render_weights_from_bundle(b0)
    w2 = full_render_weights_from_bundle(b2)
    n0 = _exported_param_count(w0)
    n2 = _exported_param_count(w2)
    # The shared conv removes one of the two ~20.8K-param leading convs.
    assert n2 < n0, f"tie must reduce stored params: {n0} -> {n2}"
    assert (n0 - n2) > 15_000, f"expected >15K param drop, got {n0 - n2}"
    # The leading per-block convs are GONE; the shared conv + a stage FiLM are present.
    assert "blocks.0.conv.weight" in w0
    assert "blocks.0.conv.weight" not in w2 and "blocks.1.conv.weight" not in w2
    assert "tied_conv.weight" in w2 and "tied_conv.bias" in w2
    assert "tied_stage_films.0.gamma_delta" in w2
    # The trainable tree also drops the dead convs (honest optimizer work).
    tp2 = dict(tree_flatten(b2.trainable_parameters()))
    assert not any(k.startswith("decoder.blocks.0.conv") for k in tp2)
    assert not any(k.startswith("decoder.blocks.1.conv") for k in tp2)
    assert any("tied_conv" in k for k in tp2)


@skip_no_mlx
@pytest.mark.parametrize("base_ch", [20, 24])
def test_tied_numpy_inflate_reproduces_mlx_render(base_ch):
    """NO-FAKE (grid-PE fake-parity lesson): the numpy inflate of the TIED decoder
    reproduces the tied MLX render op-for-op (sub-0.05 on [0,255]). A numpy path
    that did NOT dispatch the tied stages to the shared conv would render a wrong
    frame (~255 off)."""
    from tac.capstone_vq_nerv.numpy_reference import (
        decode_config_from_bundle,
        full_render_weights_from_bundle,
        numpy_decode_pair,
    )
    from tac.capstone_vq_nerv.vq_nerv_bundle import (
        CapstoneVqNervBundle,
        CapstoneVqNervConfig,
    )

    b = CapstoneVqNervBundle(
        CapstoneVqNervConfig(
            num_pairs=4, base_channels=base_ch, codebook_size=16, seed=base_ch, tie_depth=2
        )
    )
    rng = np.random.default_rng(base_ch)
    b.set_pose_stats(
        rng.standard_normal(6).astype(np.float32),
        (np.abs(rng.standard_normal(6)) + 0.5).astype(np.float32),
    )
    # Nudge the per-stage FiLM + tied conv off init so the tie path is non-trivial.
    b.tied_stage_films[0].gamma_delta = mx.array(
        (0.1 * rng.standard_normal(b.stem_channels)).astype(np.float32)
    )
    b.tied_stage_films[0].beta = mx.array(
        (0.2 * rng.standard_normal(b.stem_channels)).astype(np.float32)
    )
    idx = mx.arange(4)
    pose = mx.array(rng.standard_normal((4, 6)).astype(np.float32))
    r_mlx = np.asarray(b(idx, pose=pose))
    z_q = np.asarray(b._quantize(idx))
    weights = full_render_weights_from_bundle(b)
    cfg = decode_config_from_bundle(b)
    assert cfg.tie_depth == 2
    r_np = numpy_decode_pair(z_q, np.asarray(pose), weights, cfg)
    drift = float(np.max(np.abs(r_mlx - r_np)))
    assert drift < 0.05, f"tied numpy<->MLX render drift too large: {drift}"


@skip_no_mlx
def test_tie_depth_rejects_above_max():
    from tac.capstone_vq_nerv.vq_nerv_bundle import (
        CapstoneVqNervBundle,
        CapstoneVqNervConfig,
    )

    with pytest.raises(ValueError, match="exceeds"):
        CapstoneVqNervBundle(
            CapstoneVqNervConfig(num_pairs=2, base_channels=24, tie_depth=3)
        )


@skip_no_mlx
def test_tied_stage_film_is_identity_at_init():
    """At init the per-stage FiLM is identity (gamma=1, beta=0), so the UNTRAINED
    tied render equals what the shared conv alone produces (no spurious modulation)."""
    from tac.capstone_vq_nerv.vq_nerv_bundle import _TiedStageFiLM

    f = _TiedStageFiLM(channels=8)
    x = mx.random.normal((2, 3, 3, 8))
    y = f(x)
    assert float(mx.max(mx.abs(y - x))) == 0.0


@skip_no_mlx
def test_tied_conv_is_trained_and_in_ema_shadow():
    """The shared tied conv is a REAL trainable param: a training step moves it and
    the EMA shadow tracks it (so the exported shadow carries the trained tie)."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from test_capstone_vq_nerv import _build_frozen_dnet

    from tac.capstone_vq_nerv.capstone_trainer import (
        CapstoneTrainConfig,
        CapstoneTrainer,
    )
    from tac.capstone_vq_nerv.vq_nerv_bundle import (
        CapstoneVqNervBundle,
        CapstoneVqNervConfig,
    )
    from tac.mlx_pr95_port.score_bridge import TorchScorerBridge

    n, h, w = 4, 48, 64
    dnet = _build_frozen_dnet(with_pose=True)
    seg = torch.zeros(n, h, w, dtype=torch.long)
    for i in range(n):
        for c, (r0, r1) in enumerate([(0, 10), (10, 20), (20, 30), (30, 40), (40, h)]):
            seg[i, r0:r1, :] = (c + i) % 5
    pose = torch.from_numpy(np.random.default_rng(0).standard_normal((n, 6)).astype(np.float32))
    bridge = TorchScorerBridge(dnet, seg, pose, scorer_hw=(h, w), eval_roundtrip=False)
    b = CapstoneVqNervBundle(
        CapstoneVqNervConfig(
            num_pairs=n, base_channels=24, codebook_size=16, seed=0,
            tie_depth=2, carrier="stored_latent",
        )
    )
    cfg = CapstoneTrainConfig(
        epochs=2, batch_size=n, eval_every=1, seed=0,
        muon_lr=3e-2, adamw_lr=2e-2, ema_decay=0.9, grad_clip=50, grad_clip_muon=50,
    )
    tr = CapstoneTrainer(b, bridge, pose.numpy(), cfg)
    before = np.asarray(b.tied_conv.weight).copy()
    tr.train()
    after = np.asarray(b.tied_conv.weight)
    assert float(np.max(np.abs(after - before))) > 0.0, "tied conv must train"
    assert any("tied_conv" in k for k in tr._ema.shadow), "EMA must cover tied conv"


# ---------------------------------------------------------------------------
# (2) Cross-hardware margin hinge
# ---------------------------------------------------------------------------


def test_margin_hinge_penalizes_small_margin_and_pushes_it_up():
    """NO-FAKE: the hinge is a REAL loss with a REAL gradient (not a constant).

    A small-positive-margin pixel (margin 0.05 < floor 0.1) gets a positive hinge,
    and the gradient of the hinge w.r.t. the target logit is NEGATIVE (raising the
    target logit -> larger margin -> lower hinge). A constant/no-op would have a
    zero gradient and would FAIL this test.
    """
    B, C, H, W = 2, 5, 6, 6
    targets = torch.zeros(B, H, W, dtype=torch.long)
    logits = torch.zeros(B, C, H, W, requires_grad=True)
    with torch.no_grad():
        logits[:, 0] = 0.05  # target margin 0.05 (below the 0.1 floor)
    hinge = margin_floor_hinge(logits, targets, margin_floor=0.1)
    assert float(hinge.detach()) > 0.0
    hinge.backward()
    assert float(logits.grad[:, 0].mean()) < 0.0  # push target logit UP


def test_margin_hinge_zero_on_clear_of_floor_pixels():
    """A comfortably-above-floor margin contributes EXACTLY zero hinge."""
    B, C, H, W = 2, 5, 6, 6
    targets = torch.zeros(B, H, W, dtype=torch.long)
    logits = torch.zeros(B, C, H, W)
    logits[:, 0] = 5.0  # margin 5.0 >> floor
    assert float(margin_floor_hinge(logits, targets, margin_floor=0.1)) == 0.0


def test_margin_hinge_floor_must_be_positive():
    logits = torch.zeros(1, 5, 4, 4)
    targets = torch.zeros(1, 4, 4, dtype=torch.long)
    with pytest.raises(ValueError):
        margin_floor_hinge(logits, targets, margin_floor=0.0)
    with pytest.raises(ValueError):
        CrossHwMarginHingeSegLoss(ce_seg_loss, margin_floor=-1.0, hinge_weight=1.0)


def test_hinge_wrapper_weight_zero_is_byte_identical_to_base():
    """hinge_weight=0 wrapper == bare base loss (default-off is provably inert)."""
    w0 = CrossHwMarginHingeSegLoss(ce_seg_loss, margin_floor=0.1, hinge_weight=0.0)
    lg = torch.randn(2, 5, 6, 6)
    tg = torch.randint(0, 5, (2, 6, 6))
    assert float(w0(lg, tg) - ce_seg_loss(lg, tg)) == 0.0


def test_hinge_wrapper_weight_positive_raises_loss_above_base():
    """With a high floor, the wrapper's loss is strictly above the bare base loss."""
    wp = CrossHwMarginHingeSegLoss(ce_seg_loss, margin_floor=3.0, hinge_weight=1.0)
    lg = torch.randn(2, 5, 6, 6)
    tg = torch.randint(0, 5, (2, 6, 6))
    assert float(wp(lg, tg)) > float(ce_seg_loss(lg, tg))


def test_hinge_wrapper_rebases_on_stage_switch():
    """set_base_loss_fn re-points the wrapped loss (curriculum stage transition)."""
    from tac.score_aware_loop.live_segnet_loss import tau_softplus_seg_loss

    w = CrossHwMarginHingeSegLoss(ce_seg_loss, margin_floor=0.1, hinge_weight=0.0)
    lg = torch.randn(2, 5, 6, 6)
    tg = torch.randint(0, 5, (2, 6, 6))
    assert float(w(lg, tg)) == pytest.approx(float(ce_seg_loss(lg, tg)))
    w.set_base_loss_fn(tau_softplus_seg_loss)
    assert float(w(lg, tg)) == pytest.approx(float(tau_softplus_seg_loss(lg, tg)))


@skip_no_mlx
def test_trainer_installs_hinge_on_torch_cpu_path():
    """torch-CPU + hinge installs the seg_loss_fn wrapper (the gradient backend).

    NOTE: the mlx_gpu path is NO LONGER fail-closed (the MLX-native hinge wire-in):
    that branch builds the real-upstream MLX adapter, so its hinge-wiring assertion
    lives in the real-net GPU bridge suite
    (``test_mlx_gpu_score_bridge.py::test_trainer_mlx_gpu_backend_wires_in_hinge``)
    — the cheap color-proto stub net here cannot convert to the MLX adapter.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from test_capstone_vq_nerv import _build_frozen_dnet

    from tac.capstone_vq_nerv.capstone_trainer import (
        CapstoneTrainConfig,
        CapstoneTrainer,
    )
    from tac.capstone_vq_nerv.vq_nerv_bundle import (
        CapstoneVqNervBundle,
        CapstoneVqNervConfig,
    )
    from tac.mlx_pr95_port.score_bridge import TorchScorerBridge

    n, h, w = 2, 48, 64
    dnet = _build_frozen_dnet(with_pose=True)
    seg = torch.zeros(n, h, w, dtype=torch.long)
    pose = torch.zeros(n, 6)
    bridge = TorchScorerBridge(dnet, seg, pose, scorer_hw=(h, w), eval_roundtrip=False)
    b = CapstoneVqNervBundle(CapstoneVqNervConfig(num_pairs=n, base_channels=24, seed=0))
    # torch-CPU + hinge: installs the seg_loss_fn wrapper.
    tr = CapstoneTrainer(
        b, bridge, pose.numpy(),
        CapstoneTrainConfig(epochs=1, batch_size=n, margin_hinge_weight=0.5),
    )
    assert isinstance(bridge.seg_loss_fn, CrossHwMarginHingeSegLoss)
    assert tr._margin_hinge is not None


# ---------------------------------------------------------------------------
# (3) CLI passthrough
# ---------------------------------------------------------------------------


def test_campaign_cli_exposes_new_flags():
    """The campaign argparse exposes --tie-depth / --hinerv-grid-pe /
    --grid-pe-num-freqs (+ margin hinge flags). Checked against the live --help."""
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[4]
    cli_path = repo / "experiments" / "run_capstone_campaign.py"
    out = subprocess.run(
        [sys.executable, str(cli_path), "--help"],
        capture_output=True,
        text=True,
        cwd=str(repo),
    )
    help_text = out.stdout + out.stderr
    for flag in (
        "--tie-depth",
        "--hinerv-grid-pe",
        "--grid-pe-num-freqs",
        "--margin-hinge-weight",
        "--margin-hinge-floor",
    ):
        assert flag in help_text, f"{flag} missing from campaign CLI help"


def test_decode_config_carries_tie_depth_to_sidecar():
    """The inflate config sidecar carries tie_depth (so the contest inflate
    dispatches the tied stages). decode_config_from_bundle -> asdict round-trips it."""
    import dataclasses

    if not _HAVE_MLX:
        pytest.skip("mlx not available")
    from tac.capstone_vq_nerv.numpy_reference import decode_config_from_bundle
    from tac.capstone_vq_nerv.vq_nerv_bundle import (
        CapstoneVqNervBundle,
        CapstoneVqNervConfig,
    )

    b = CapstoneVqNervBundle(
        CapstoneVqNervConfig(
            num_pairs=2, base_channels=24, seed=0, tie_depth=2,
            hinerv_grid_pe=True, grid_pe_num_freqs=4,
        )
    )
    cfg = decode_config_from_bundle(b)
    sidecar = dataclasses.asdict(cfg)
    assert sidecar["tie_depth"] == 2
    assert sidecar["hinerv_grid_pe"] is True
    assert sidecar["grid_pe_num_freqs"] == 4
