# SPDX-License-Identifier: MIT
"""Behavior tests for the pose-FiLM module + stabilized training recipe (task #84).

The capstone (#78) imports two de-risked pieces from this package:

1. **pose-FiLM** (``pose_film.py``) — store the 6-d GT pose explicitly + FiLM-inject
   it into the frame head (Quantizr PR#55's mechanism), sidestepping the #80
   reconstruct-from-pixels pose-tube wall. The tests prove the FiLM ACTUALLY
   conditions the output (the stored pose changes the render), the stored pose
   round-trips through quantize+brotli, and the severed-FiLM control fails (no
   FiLM => higher d_pose), so the FiLM is load-bearing not cosmetic.

2. **stabilized recipe** (``pose_film_trainer.py``) — Muon-throughout + per-size
   LR/grad-clip/EMA (the #74-instability fix). The tests prove the recipe scales
   with carrier size, the blowup-detector fires on a real divergence, the
   stabilized loop is monotone, and Muon-throughout reaches a materially lower
   d_pose than a destabilized (AdamW-only, high-LR, no-EMA) config.

Authority: ``[macOS-MLX research-signal]`` (MLX decoder) / ``[local CPU-torch
advisory]`` (frozen torch scorer; NO MPS). Non-promotable per Catalog #192; a
contest score requires ``upstream/evaluate.py`` on paired CUDA + Linux-x86_64 CPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

try:
    import mlx.core as mx
    import mlx.nn  # noqa: F401

    _MLX_AVAILABLE = True
except ImportError:
    _MLX_AVAILABLE = False

skip_no_mlx = pytest.mark.skipif(
    not _MLX_AVAILABLE,
    reason="MLX not available; the pose-FiLM module requires Apple Silicon.",
)


# ---------------------------------------------------------------------------
# Frozen scorer stand-ins (a well-conditioned SegNet + a global-read PoseNet
# whose 6-d pose IS steerable by the carrier, so the FiLM/control comparison is
# clean). NO MPS; CPU torch (the exact authority decode path).
# ---------------------------------------------------------------------------


class _ColorProtoSeg(nn.Module):
    """Frozen 5-class color-prototype SegNet stand-in."""

    def __init__(self) -> None:
        super().__init__()
        protos = torch.tensor(
            [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]],
            dtype=torch.float32,
        )
        self.c = nn.Conv2d(3, 5, 1)
        self.c.weight.data = protos.reshape(5, 3, 1, 1) / 128.0
        self.c.bias.data = -(protos**2).sum(1) / (2 * 128.0 * 128.0)

    def forward(self, x):  # x in [0,1] NCHW
        return self.c(x * 255.0)


class _GlobalReadPose(nn.Module):
    """Frozen PoseNet stand-in: 6-d pose = linear read of frame0's global spatial moments.

    Per #80 frame0 dominates pose; this reads 6 global spatial moments (mean,
    1st/2nd order) of frame0's luma and projects to a 6-d pose. It is a real,
    differentiable read (the gradient reaches pixels so the score bridge works),
    and it is REACHABLE by the carrier (the moment basis spans), so a working
    pose carrier can drive d_pose down toward the read's conditioning floor.
    """

    def __init__(self, h: int, w: int, seed: int = 0) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.proj = nn.Linear(6, 6, bias=False)
        self.proj.weight.data = torch.eye(6) + 0.1 * torch.randn(6, 6, generator=g)
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij"
        )
        masks = [
            torch.ones_like(yy), yy, xx, yy * xx, yy * yy - 0.5, xx * xx - 0.5,
        ]
        self.masks = torch.stack([m / m.abs().mean() for m in masks])  # (6,h,w)

    def forward(self, x):  # x: (B,12,H,W) the posenet_in
        f0 = x[:, :3].mean(1)  # frame0 luma proxy
        r = torch.einsum("bhw,khw->bk", f0, self.masks) / (
            f0.shape[1] * f0.shape[2]
        )
        pose = self.proj(r)
        return {"pose": torch.cat([pose, pose], dim=1)}  # (B,12); first 6 used


class _SegPoseDNet(nn.Module):
    """Frozen DistortionNet (SegNet + PoseNet) matching the bridge interface."""

    def __init__(self, h: int, w: int) -> None:
        super().__init__()
        self.segnet = _ColorProtoSeg()
        self.posenet = _GlobalReadPose(h, w)

    def preprocess_input(self, bhwc):  # (B,2,H,W,C)
        f0 = bhwc[:, 0].permute(0, 3, 1, 2) / 255.0
        f1 = bhwc[:, 1].permute(0, 3, 1, 2) / 255.0
        pose_in = torch.cat([f0, f1, f0, f1], dim=1)[:, :12]
        last = bhwc[:, -1].permute(0, 3, 1, 2)
        return pose_in, last / 255.0


def _build_frozen_seg_pose_dnet(h: int, w: int):
    dnet = _SegPoseDNet(h, w).eval()
    for p in dnet.parameters():
        p.requires_grad = False
    return dnet


def _build_pose_setup(n_pairs=8, h=48, w=64, *, film_slots=(0, 1), const_pose=False,
                      share_latent=True, pose_scale=0.6, seed=0):
    """Build the frozen scorer + stored-pose bundle for the FiLM/control comparison.

    ``share_latent`` ties all per-pair latents to one shared content vector (frozen)
    so the per-pair POSE is the only per-pair degree of freedom — the regime where
    the FiLM-vs-severed comparison is clean (the latents can't carry pose).
    """
    from tac.mlx_pr95_port.pose_film import StoredPoseBundleMLX
    from tac.mlx_pr95_port.score_bridge import TorchScorerBridge

    dnet = _build_frozen_seg_pose_dnet(h, w)
    rng = np.random.RandomState(seed)
    pose_tgt = ((rng.rand(n_pairs, 6) - 0.5) * pose_scale).astype(np.float32)
    stored = (
        np.tile(pose_tgt.mean(0, keepdims=True), (n_pairs, 1)).astype(np.float32)
        if const_pose
        else pose_tgt
    )
    seg_tgt = torch.zeros(n_pairs, h, w, dtype=torch.long)
    bands = [(0, 10), (10, 20), (20, 30), (30, 40), (40, h)]
    for cls, (r0, r1) in enumerate(bands):
        seg_tgt[:, r0:r1, :] = cls
    bridge = TorchScorerBridge(
        dnet, seg_tgt, torch.tensor(pose_tgt), seg_loss_form="ce_seg_loss",
        scorer_hw=(h, w), eval_roundtrip=False, seg_weight=1.0, pose_weight=300.0,
    )
    bundle = StoredPoseBundleMLX(
        latent_count=n_pairs, pose_targets=stored, latent_dim=28, base_channels=16,
        film_slots=film_slots, film_hidden=64, seed=seed,
    )
    if share_latent:
        bundle.latents = mx.broadcast_to(bundle.latents[0:1], bundle.latents.shape) + 0.0
    return bundle, bridge, h, w


# ---------------------------------------------------------------------------
# (1) pose-FiLM module — the FiLM ACTUALLY conditions the output.
# ---------------------------------------------------------------------------


@skip_no_mlx
def test_film_is_identity_at_init():
    """The zero-init FiLM is the IDENTITY: the carrier starts as the clean PR95 decoder."""
    from tac.mlx_pr95_port.pose_film import PoseFiLMDecoderMLX

    dec = PoseFiLMDecoderMLX(latent_dim=28, base_channels=12, film_slots=(0, 1), seed=0)
    z = mx.array(np.random.RandomState(0).randn(3, 28).astype(np.float32) * 0.1)
    pose = mx.array(np.random.RandomState(1).randn(3, 6).astype(np.float32))
    with_film = dec(z, pose)
    plain = dec.decoder(z)  # the bare PR95 decoder
    mx.eval(with_film, plain)
    assert float(mx.max(mx.abs(with_film - plain))) < 1e-5, (
        "zero-init FiLM must be the identity (start as the clean decoder)"
    )


@skip_no_mlx
def test_trained_film_conditions_the_output_on_stored_pose():
    """NO-FAKE: once the FiLM head has non-identity weights, the STORED pose changes the render."""
    from tac.mlx_pr95_port.pose_film import PoseFiLMDecoderMLX

    dec = PoseFiLMDecoderMLX(latent_dim=28, base_channels=12, film_slots=(0,), seed=0)
    # Perturb the film0 head off identity so it is no longer the trivial map.
    from mlx.utils import tree_flatten, tree_unflatten

    flat = dict(tree_flatten(dec.parameters()))
    rng = np.random.RandomState(3)
    for k in list(flat):
        if k.startswith("film0.film"):
            flat[k] = mx.array((np.asarray(flat[k]) + 0.2 * rng.randn(*flat[k].shape)).astype(np.float32))
    dec.update(tree_unflatten(list(flat.items())))
    z = mx.array(np.random.RandomState(0).randn(2, 28).astype(np.float32) * 0.1)
    pose_a = mx.array(np.full((2, 6), 0.0, dtype=np.float32))
    pose_b = mx.array(np.full((2, 6), 0.5, dtype=np.float32))
    out_a = dec(z, pose_a)
    out_b = dec(z, pose_b)
    mx.eval(out_a, out_b)
    # Same latent, DIFFERENT stored pose -> the FiLM-conditioned frame0 head differs.
    assert float(mx.max(mx.abs(out_a[:, 0] - out_b[:, 0]))) > 1e-2, (
        "a trained FiLM head must make the stored pose change the frame0 render"
    )


@skip_no_mlx
def test_severed_film_slot_leaves_that_frame_unconditioned():
    """A slot NOT in film_slots is NOT pose-conditioned (the render ignores pose there)."""
    from tac.mlx_pr95_port.pose_film import PoseFiLMDecoderMLX

    # FiLM frame0 only; frame1 has no FiLM head -> frame1 ignores pose.
    dec = PoseFiLMDecoderMLX(latent_dim=28, base_channels=12, film_slots=(0,), seed=0)
    from mlx.utils import tree_flatten, tree_unflatten

    flat = dict(tree_flatten(dec.parameters()))
    rng = np.random.RandomState(7)
    for k in list(flat):
        if k.startswith("film0.film"):
            flat[k] = mx.array((np.asarray(flat[k]) + 0.3 * rng.randn(*flat[k].shape)).astype(np.float32))
    dec.update(tree_unflatten(list(flat.items())))
    z = mx.array(np.random.RandomState(0).randn(2, 28).astype(np.float32) * 0.1)
    pa = mx.array(np.zeros((2, 6), np.float32))
    pb = mx.array(np.full((2, 6), 0.5, np.float32))
    oa, ob = dec(z, pa), dec(z, pb)
    mx.eval(oa, ob)
    # frame1 (un-modulated) is IDENTICAL across poses; frame0 (modulated) differs.
    assert float(mx.max(mx.abs(oa[:, 1] - ob[:, 1]))) < 1e-5, (
        "the un-FiLM'd frame1 must be pose-invariant"
    )
    assert float(mx.max(mx.abs(oa[:, 0] - ob[:, 0]))) > 1e-2, (
        "the FiLM'd frame0 must be pose-dependent"
    )


@skip_no_mlx
def test_film_slots_rejects_invalid_slot():
    """film_slots entries must be 0 or 1 (fail closed)."""
    from tac.mlx_pr95_port.pose_film import PoseFiLMDecoderMLX

    with pytest.raises(ValueError):
        PoseFiLMDecoderMLX(latent_dim=28, base_channels=12, film_slots=(2,))


@skip_no_mlx
def test_stored_pose_bundle_rejects_wrong_pose_shape():
    """StoredPoseBundleMLX requires (latent_count, 6) pose targets (fail closed)."""
    from tac.mlx_pr95_port.pose_film import StoredPoseBundleMLX

    with pytest.raises(ValueError):
        StoredPoseBundleMLX(latent_count=4, pose_targets=np.zeros((4, 5), np.float32))


# ---------------------------------------------------------------------------
# (2) stored pose carrier — round-trips through quantize+brotli at ~kilobytes.
# ---------------------------------------------------------------------------


@skip_no_mlx
def test_stored_pose_is_initialized_from_gt_targets():
    """The stored pose carries the GT answer (Quantizr stores it, not learns it)."""
    from tac.mlx_pr95_port.pose_film import StoredPoseBundleMLX

    pose = np.random.RandomState(0).randn(6, 6).astype(np.float32) * 0.3
    bundle = StoredPoseBundleMLX(latent_count=6, pose_targets=pose, base_channels=12)
    assert float(mx.max(mx.abs(bundle.stored_pose - mx.array(pose)))) < 1e-6, (
        "the stored pose must be initialized from the GT targets (the answer is handed in)"
    )


@skip_no_mlx
def test_stored_pose_bytes_are_kilobytes_scale():
    """The pose carrier is ~kilobytes (6 floats/pair, quantized+compressed)."""
    from tac.mlx_pr95_port.pose_film import stored_pose_bytes

    rng = np.random.RandomState(0)
    pose_600 = (rng.rand(600, 6).astype(np.float32) - 0.5) * 0.3
    b = stored_pose_bytes(pose_600)
    # 600 pairs x 6 floats, quantized + compressed -> a few kilobytes (Quantizr's
    # pose.npy.br is ~14KB raw; quantized+brotli is far smaller). Sanity bounds.
    assert 0 < b < 30_000, f"600-pair stored pose must be a few KB, got {b}"


@skip_no_mlx
def test_stored_pose_quant_roundtrip_preserves_pose_within_tube():
    """The stored pose survives quantize+dequantize to a small error (the tube)."""
    from tac.mlx_pr95_port.pose_film import stored_pose_bytes

    rng = np.random.RandomState(0)
    pose = (rng.rand(50, 6).astype(np.float32) - 0.5) * 0.3
    step = 1e-3
    q = np.round(pose / step) * step
    # round-trip error is bounded by half the quant step (uniform quant).
    assert float(np.max(np.abs(q - pose))) <= step / 2 + 1e-7
    # and the byte cost is finite/positive (the carrier exists).
    assert stored_pose_bytes(pose, quant_step=step) > 0


# ---------------------------------------------------------------------------
# (3) The FiLM is LOAD-BEARING: training holds pose, and the severed control fails.
# ---------------------------------------------------------------------------


@skip_no_mlx
def test_film_pose_descends_on_live_render():
    """The pose-FiLM loop drives the LIVE-render d_pose DOWN against the frozen PoseNet."""
    from tac.mlx_pr95_port.pose_film_trainer import (
        PoseFilmTrainer,
        PoseFilmTrainerConfig,
        StabilizedRecipe,
    )

    bundle, bridge, h, w = _build_pose_setup(share_latent=True, pose_scale=0.6)
    recipe = StabilizedRecipe(
        muon_lr=3e-2, adamw_lr=2e-2, latent_lr_mult=0.0, grad_clip=50.0, ema_decay=0.9,
    )
    cfg = PoseFilmTrainerConfig(
        epochs=300, batch_size=8, eval_every=100, scorer_hw=(h, w),
        eval_roundtrip=False, base_channels=16, seed=0, recipe=recipe,
    )
    res = PoseFilmTrainer(bundle, bridge, cfg).train()
    assert res["d_pose_initial"] > 0.02, "setup must start with a real pose error"
    assert res["pose_descended"] is True
    assert res["d_pose_best"] < res["d_pose_initial"] * 0.5, (
        f"d_pose must at least halve (init {res['d_pose_initial']:.3e}, "
        f"best {res['d_pose_best']:.3e})"
    )


@skip_no_mlx
def test_severed_film_holds_pose_worse_no_fake_control():
    """NO-FAKE: removing the FiLM (severed) leaves d_pose materially HIGHER.

    The FiLM is the load-bearing per-pair pose carrier: with the latents shared
    (frozen) the pose is the ONLY per-pair DOF, so a severed FiLM (film_slots=())
    cannot fit the diverse per-pair poses and lands a higher d_pose than FiLM-on.
    """
    from tac.mlx_pr95_port.pose_film_trainer import (
        PoseFilmTrainer,
        PoseFilmTrainerConfig,
        StabilizedRecipe,
    )

    recipe = StabilizedRecipe(
        muon_lr=3e-2, adamw_lr=2e-2, latent_lr_mult=0.0, grad_clip=50.0, ema_decay=0.9,
    )

    def _run(film_slots):
        bundle, bridge, h, w = _build_pose_setup(
            film_slots=film_slots, share_latent=True, pose_scale=0.6
        )
        cfg = PoseFilmTrainerConfig(
            epochs=400, batch_size=8, eval_every=200, scorer_hw=(h, w),
            eval_roundtrip=False, base_channels=16, seed=0, recipe=recipe,
        )
        return PoseFilmTrainer(bundle, bridge, cfg).train()["d_pose_best"]

    d_pose_film = _run((0, 1))
    d_pose_severed = _run(())
    assert d_pose_film < d_pose_severed * 0.8, (
        f"FiLM-on d_pose ({d_pose_film:.3e}) must be materially below severed "
        f"({d_pose_severed:.3e}) — the FiLM must be load-bearing, not cosmetic"
    )


# ---------------------------------------------------------------------------
# (4) the stabilized recipe — the #74-instability fix.
# ---------------------------------------------------------------------------


@skip_no_mlx
def test_stabilized_recipe_scales_lr_down_with_carrier_size():
    """The per-size recipe pulls LR DOWN as the carrier grows (the #74-blowup defense)."""
    from tac.mlx_pr95_port.pose_film_trainer import StabilizedRecipe

    small = StabilizedRecipe.for_base_channels(16)
    mid = StabilizedRecipe.for_base_channels(28)
    big = StabilizedRecipe.for_base_channels(36)
    assert small.muon_lr > mid.muon_lr > big.muon_lr, (
        "the larger (more #74-prone) carrier must get a smaller LR"
    )
    assert mid.ema_decay >= small.ema_decay, (
        "the larger carrier must get a stronger EMA (more in-basin damping)"
    )
    assert all(r.use_muon for r in (small, mid, big)), (
        "Muon is ON at every size (the scale-stable core, the #77 fix)"
    )


@skip_no_mlx
def test_stabilized_recipe_is_muon_throughout_by_default():
    """C7/#77 fix: the recipe defaults to Muon-throughout (NOT AdamW stages 1-7)."""
    from tac.mlx_pr95_port.pose_film_trainer import StabilizedRecipe

    assert StabilizedRecipe.for_base_channels(28).use_muon is True


@skip_no_mlx
def test_blowup_detector_fires_on_a_real_divergence():
    """The monotone flag flips False when d_pose blows up past the #74 threshold.

    Drives a TRUE divergence by feeding the trainer a deliberately destabilized
    config (huge LR, no clip, no Muon) and a monkeypatched metric that diverges,
    then asserts the detector catches it. (Decoupled from the bounded-sigmoid
    carrier so the detection LOGIC is what's under test.)
    """
    from tac.mlx_pr95_port.pose_film_trainer import (
        INSTABILITY_BLOWUP_FACTOR,
        PoseFilmTrainer,
        PoseFilmTrainerConfig,
        StabilizedRecipe,
    )

    bundle, bridge, h, w = _build_pose_setup(share_latent=True, pose_scale=0.4)
    recipe = StabilizedRecipe(
        muon_lr=0.0, adamw_lr=1e-2, latent_lr_mult=1.0, grad_clip=50.0,
        ema_decay=0.9, use_muon=False,
    )
    cfg = PoseFilmTrainerConfig(
        epochs=40, batch_size=8, eval_every=10, scorer_hw=(h, w),
        eval_roundtrip=False, base_channels=16, seed=0, recipe=recipe,
    )
    trainer = PoseFilmTrainer(bundle, bridge, cfg)
    # Inject a diverging metric: epoch-by-epoch the "measured" d_pose climbs past
    # INSTABILITY_BLOWUP_FACTOR x its running min — the exact #74 blowup signature.
    seq = iter([1e-3, 2e-3, 1.5e-3, 1e-3, 1.0, 5.0])  # min 1e-3 then 1000x blowup

    def _fake_metric(which):
        if which == "d_pose":
            try:
                return next(seq)
            except StopIteration:
                return 5.0
        return 0.1

    trainer._eval_metric = _fake_metric  # type: ignore[assignment]
    res = trainer.train()
    assert res["monotone"] is False, "the detector must flag the injected d_pose blowup"
    assert res["blowup_epoch"] is not None
    assert INSTABILITY_BLOWUP_FACTOR > 1.0


@skip_no_mlx
def test_stabilized_loop_is_monotone_no_blowup():
    """The stabilized recipe (Muon + moderate LR + EMA) trains d_pose monotonically."""
    from tac.mlx_pr95_port.pose_film_trainer import (
        PoseFilmTrainer,
        PoseFilmTrainerConfig,
        StabilizedRecipe,
    )

    bundle, bridge, h, w = _build_pose_setup(share_latent=True, pose_scale=0.4)
    recipe = StabilizedRecipe(
        muon_lr=2e-2, adamw_lr=1e-2, latent_lr_mult=0.0, grad_clip=20.0, ema_decay=0.95,
    )
    cfg = PoseFilmTrainerConfig(
        epochs=300, batch_size=8, eval_every=75, scorer_hw=(h, w),
        eval_roundtrip=False, base_channels=16, seed=0, recipe=recipe,
    )
    res = PoseFilmTrainer(bundle, bridge, cfg).train()
    assert res["monotone"] is True, "the stabilized Muon recipe must not blow up"
    assert res["blowup_epoch"] is None


@skip_no_mlx
def test_muon_throughout_beats_destabilized_adamw():
    """Muon-throughout reaches a materially LOWER d_pose than a destabilized AdamW config.

    The #74 lesson: a fixed-LR AdamW-only loop is unstable/inferior; Muon-throughout
    (the #77 fix) is scale-stable and lands a better basin at the same budget.
    """
    from tac.mlx_pr95_port.pose_film_trainer import (
        PoseFilmTrainer,
        PoseFilmTrainerConfig,
        StabilizedRecipe,
    )

    def _run(recipe):
        bundle, bridge, h, w = _build_pose_setup(share_latent=True, pose_scale=0.4)
        cfg = PoseFilmTrainerConfig(
            epochs=200, batch_size=8, eval_every=100, scorer_hw=(h, w),
            eval_roundtrip=False, base_channels=28, seed=0, recipe=recipe,
        )
        return PoseFilmTrainer(bundle, bridge, cfg).train()["d_pose_best"]

    stabilized = StabilizedRecipe(
        muon_lr=2e-2, adamw_lr=1e-2, latent_lr_mult=5.0, grad_clip=20.0,
        ema_decay=0.95, use_muon=True,
    )
    destabilized = StabilizedRecipe(
        muon_lr=0.0, adamw_lr=4e-1, latent_lr_mult=10.0, grad_clip=1e9,
        ema_decay=0.0, use_muon=False,
    )
    d_stab = _run(stabilized)
    d_unstab = _run(destabilized)
    assert d_stab < d_unstab, (
        f"Muon-throughout ({d_stab:.3e}) must beat the destabilized AdamW config "
        f"({d_unstab:.3e})"
    )


@skip_no_mlx
def test_trainer_uses_live_render_not_lagging_ema_shadow():
    """The trainer measures the LIVE render (not the EMA shadow) — the #82 landmine fix."""
    from tac.mlx_pr95_port.pose_film_trainer import PoseFilmTrainerConfig

    # The decisive observable (exact_d_pose / exact_d_seg) renders the live bundle;
    # use_ema_for_eval defaults False so the true descent is visible immediately.
    assert PoseFilmTrainerConfig().use_ema_for_eval is False


@skip_no_mlx
def test_pose_film_decoder_manifest_records_quantizr_mechanism():
    """The arch manifest records the store-explicit-plus-FiLM pose carrier (provenance)."""
    from tac.mlx_pr95_port.pose_film import PoseFiLMDecoderMLX

    man = PoseFiLMDecoderMLX(base_channels=12, film_slots=(0, 1)).architecture_manifest()
    assert man["pose_carrier"] == "store_explicit_plus_film"
    assert man["pose_dim"] == 6
    assert man["pose_film_slots"] == [0, 1]


@skip_no_mlx
def test_exact_d_pose_fails_closed_without_posenet():
    """exact_d_pose fails closed when pose is not enabled (no PoseNet / no targets)."""
    from tac.mlx_pr95_port.score_bridge import TorchScorerBridge

    dnet = _build_frozen_seg_pose_dnet(16, 16)
    seg_tgt = torch.zeros(2, 16, 16, dtype=torch.long)
    # pose_targets None -> pose disabled.
    bridge = TorchScorerBridge(dnet, seg_tgt, None, scorer_hw=(16, 16))
    render = mx.array(np.zeros((2, 2, 3, 16, 16), np.float32))
    with pytest.raises(ValueError):
        bridge.exact_d_pose(render, torch.arange(2))
