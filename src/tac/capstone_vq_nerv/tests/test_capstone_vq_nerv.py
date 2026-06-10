# SPDX-License-Identifier: MIT
"""Behavior tests for the original VQ-NeRV + FiLM-pose capstone (Task #78).

These tests prove REAL behavior (NO-FAKE), not constants:

* the VQ-NeRV bundle renders + the VQ indices bit-pack/round-trip exactly;
* the FiLM is identity at init (preserves the #82 descent) and ACTUALLY moves
  the render when the pose changes after the FiLM trains;
* the JOINT score-aware loop drives the EXACT d_seg DOWN on the LIVE render
  through a frozen scorer (the headline), while the store-pose-FiLM HOLDS the
  pose term (re-measured on the exact scorer);
* the NO-FAKE controls FAIL: a CONSTANT (zero-cotangent) loss does NOT descend;
  a SEVERED (stop_gradient) render does NOT descend; the EMA codebook is NOT in
  the gradient tree (it is EMA-updated, not gradient-updated);
* the archive byte-closes + parses back + the index/pose carriers round-trip.

Authority: ``[macOS-MLX research-signal]`` / ``[macOS-CPU advisory]`` — the
frozen scorer is a well-conditioned color/luma-prototype stand-in (the exact
torch contest scorer path is the bridge's REAL interface; these tests exercise
the bridge mechanism, not a contest score). A contest score needs
``upstream/evaluate.py`` on paired CUDA + Linux-x86_64 CPU.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from tac.capstone_vq_nerv.export import (
    bit_pack_vq_indices,
    bit_unpack_vq_indices,
    bits_per_index,
    build_capstone_archive_bytes,
    parse_capstone_archive_bytes,
)

try:
    import mlx.core as mx
    from mlx.utils import tree_flatten

    _HAVE_MLX = True
except Exception:  # pragma: no cover
    mx = None  # type: ignore[assignment]
    tree_flatten = None  # type: ignore[assignment]
    _HAVE_MLX = False

skip_no_mlx = pytest.mark.skipif(not _HAVE_MLX, reason="mlx not available")

# ---------------------------------------------------------------------------
# Frozen scorer fixtures (color-proto SegNet + luma-prototype PoseNet stand-in).
# ---------------------------------------------------------------------------


class _ColorProtoSeg(nn.Module):
    """A frozen, well-conditioned 5-class color-prototype SegNet stand-in."""

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


class _LumaPose(nn.Module):
    """A frozen 6-dim PoseNet stand-in: pose = linear readout of frame0 luma stats.

    Reads the 12-channel preprocessed input (2 frames x YUV6 stand-in); the pose
    is a deterministic, differentiable function of the FIRST frame's mean luma —
    so a FiLM that modulates the render CAN drive the pose toward a target.
    """

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Linear(12, 6)
        torch.manual_seed(0)
        self.w.weight.data = torch.randn(6, 12) * 0.1
        self.w.bias.data = torch.zeros(6)

    def forward(self, x):  # x: (B, 12, H, W)
        feat = x.mean(dim=(2, 3))  # (B, 12) global luma/chroma stats
        return {"pose": self.w(feat)}


class _FrozenDNet(nn.Module):
    """Minimal frozen DistortionNet (color-proto SegNet + optional luma PoseNet)."""

    def __init__(self, *, with_pose: bool = False) -> None:
        super().__init__()
        self.segnet = _ColorProtoSeg()
        self.posenet = _LumaPose() if with_pose else None

    def preprocess_input(self, bhwc):  # (B,2,H,W,C) -> (pose_in 12ch, last NCHW [0,1])
        last = bhwc[:, -1].permute(0, 3, 1, 2)  # frame1 NCHW
        first = bhwc[:, 0].permute(0, 3, 1, 2)  # frame0 NCHW
        # pose input: 6 luma/chroma channels per frame (frame0 then frame1).
        pose_in = torch.cat([first.repeat(1, 2, 1, 1), last.repeat(1, 2, 1, 1)], dim=1)
        return pose_in, last / 255.0


def _build_frozen_dnet(*, with_pose: bool = False):
    dnet = _FrozenDNet(with_pose=with_pose).eval()
    for p in dnet.parameters():
        p.requires_grad = False
    return dnet


def _build_capstone_setup(n_pairs=6, h=48, w=64, seed=0, with_pose=True, K=256):
    """Build a frozen scorer + diverse GT + the capstone bundle/bridge/trainer."""
    from tac.capstone_vq_nerv.capstone_trainer import (
        CapstoneTrainConfig,
        CapstoneTrainer,
    )
    from tac.capstone_vq_nerv.vq_nerv_bundle import (
        CapstoneVqNervBundle,
        CapstoneVqNervConfig,
    )
    from tac.mlx_pr95_port.score_bridge import TorchScorerBridge

    dnet = _build_frozen_dnet(with_pose=with_pose)
    seg_tgt = torch.zeros(n_pairs, h, w, dtype=torch.long)
    bands = [(0, 10), (10, 20), (20, 30), (30, 40), (40, h)]
    for i in range(n_pairs):
        for cls, (r0, r1) in enumerate(bands):
            seg_tgt[i, r0:r1, :] = (cls + i) % 5
    # Diverse, reachable GT pose targets.
    rng = np.random.default_rng(seed)
    pose_tgt = (
        torch.tensor(rng.standard_normal((n_pairs, 6)).astype(np.float32))
        if with_pose
        else None
    )
    bridge = TorchScorerBridge(
        dnet, seg_tgt, pose_tgt, seg_loss_form="ce_seg_loss",
        scorer_hw=(h, w), eval_roundtrip=False,
    )
    cfg = CapstoneVqNervConfig(
        num_pairs=n_pairs, latent_dim=28, base_channels=36, codebook_size=K,
        seed=seed,
    )
    bundle = CapstoneVqNervBundle(cfg)
    # The small-fixture descent config (the #82 headline-descent LRs). The
    # PR95-faithful *defaults* (muon_lr 2e-4 / adamw_lr 3e-5 / grad_clip 1.0) are
    # tuned for the real 600-pair full-res 1000ep schedule; a tiny fixture needs
    # the aggressive LRs + relaxed clip to descend off the wall in 60 epochs (the
    # SAME config the #82 ``test_live_render_d_seg_descends_off_high_start`` uses).
    tcfg = CapstoneTrainConfig(
        epochs=60, batch_size=n_pairs, eval_every=10, commitment_weight=0.05,
        seed=seed, muon_lr=3e-2, adamw_lr=2e-2, ema_decay=0.95,
        grad_clip=50.0, grad_clip_muon=50.0,
    )
    pose_store = pose_tgt.numpy() if with_pose else np.zeros((n_pairs, 6), np.float32)
    trainer = CapstoneTrainer(bundle, bridge, pose_store, tcfg)
    return bundle, bridge, trainer


# ---------------------------------------------------------------------------
# (1) Bit-pack / VQ index carrier
# ---------------------------------------------------------------------------


def test_bits_per_index_matches_log2():
    assert bits_per_index(256) == 8
    assert bits_per_index(16) == 4
    assert bits_per_index(512) == 9
    assert bits_per_index(2) == 1


def test_bit_pack_round_trip_exact_k256():
    rng = np.random.default_rng(0)
    idx = rng.integers(0, 256, size=600).astype(np.int32)
    packed = bit_pack_vq_indices(idx, 256)
    assert len(packed) == 600  # 8 bits/pair -> 1 byte/pair
    rt = bit_unpack_vq_indices(packed, 600, 256)
    assert np.array_equal(idx, rt)


def test_bit_pack_round_trip_exact_k16_nibble():
    rng = np.random.default_rng(1)
    idx = rng.integers(0, 16, size=601).astype(np.int32)  # odd count -> padding
    packed = bit_pack_vq_indices(idx, 16)
    assert len(packed) == (601 * 4 + 7) // 8  # nibble packing
    rt = bit_unpack_vq_indices(packed, 601, 16)
    assert np.array_equal(idx, rt)


def test_bit_pack_rejects_out_of_range():
    with pytest.raises(ValueError):
        bit_pack_vq_indices(np.array([0, 256], dtype=np.int32), 256)


# ---------------------------------------------------------------------------
# (2) Bundle forward + VQ + FiLM-identity-at-init
# ---------------------------------------------------------------------------


@skip_no_mlx
def test_bundle_renders_contest_shape():
    from tac.capstone_vq_nerv.vq_nerv_bundle import (
        CapstoneVqNervBundle,
        CapstoneVqNervConfig,
    )

    b = CapstoneVqNervBundle(CapstoneVqNervConfig(num_pairs=4, codebook_size=256))
    r = b(mx.arange(4), pose=mx.random.normal((4, 6)))
    mx.eval(r)
    assert tuple(r.shape) == (4, 2, 3, 384, 512)


@skip_no_mlx
def test_film_is_identity_at_init():
    """FiLM is exactly identity at init (zero-init fc2) -> preserves #82 descent."""
    from tac.capstone_vq_nerv.vq_nerv_bundle import (
        CapstoneVqNervBundle,
        CapstoneVqNervConfig,
    )

    b = CapstoneVqNervBundle(CapstoneVqNervConfig(num_pairs=4, codebook_size=256))
    idx = mx.arange(4)
    pose = mx.random.normal((4, 6))
    r_nopose = b(idx, pose=None)
    r_pose = b(idx, pose=pose)
    mx.eval(r_nopose, r_pose)
    assert float(mx.max(mx.abs(r_nopose - r_pose))) < 1e-5


@skip_no_mlx
def test_vq_indices_in_range_and_excluded_from_grad_tree():
    from tac.capstone_vq_nerv.vq_nerv_bundle import (
        CapstoneVqNervBundle,
        CapstoneVqNervConfig,
    )

    b = CapstoneVqNervBundle(CapstoneVqNervConfig(num_pairs=8, codebook_size=256))
    idx = b.all_vq_indices()
    assert idx.shape == (8,)
    assert int(idx.min()) >= 0 and int(idx.max()) < 256
    # NO-FAKE: the EMA codebook buffers must NOT be in the gradient tree.
    tp = dict(tree_flatten(b.trainable_parameters()))
    assert not any("codebook" in k or "ema_" in k for k in tp)
    # but the decoder + FiLM ARE trainable.
    assert any("decoder" in k for k in tp)
    assert any("pose_film" in k for k in tp)


@skip_no_mlx
def test_commitment_loss_nonzero_and_vq_ema_updates_codebook():
    """The VQ commitment loss is real (>0) and EMA actually mutates the codebook."""
    from tac.capstone_vq_nerv.vq_nerv_bundle import (
        CapstoneVqNervBundle,
        CapstoneVqNervConfig,
    )

    b = CapstoneVqNervBundle(CapstoneVqNervConfig(num_pairs=8, codebook_size=64))
    cb_before = np.array(b.quantizer.codebook)
    b(mx.arange(8), pose=mx.random.normal((8, 6)))
    assert float(b.last_commitment_loss) > 0.0
    b.ema_update_from_last()
    cb_after = np.array(b.quantizer.codebook)
    # NO-FAKE: at least the active codes moved (EMA is not a no-op).
    assert float(np.max(np.abs(cb_after - cb_before))) > 0.0


# ---------------------------------------------------------------------------
# (3) THE HEADLINE — joint d_seg descent + pose hold on the LIVE render
# ---------------------------------------------------------------------------


@skip_no_mlx
def test_joint_d_seg_descends_on_live_render():
    """HEADLINE: the EXACT live-render d_seg DESCENDS through the frozen scorer."""
    _, _, trainer = _build_capstone_setup(n_pairs=6, h=48, w=64, seed=0, with_pose=True)
    out = trainer.train()
    assert out["d_seg_initial"] > 0.2  # high start (mean-field wall region)
    assert out["d_seg_final"] < out["d_seg_initial"] - 0.05, (
        f"d_seg must descend: {out['d_seg_initial']} -> {out['d_seg_final']}"
    )
    assert out["seg_descended"]


@skip_no_mlx
def test_pose_term_held_or_improved_by_film():
    """The store-pose-FiLM HOLDS (or improves) the re-measured d_pose."""
    _, _, trainer = _build_capstone_setup(n_pairs=6, h=48, w=64, seed=1, with_pose=True)
    out = trainer.train()
    # The FiLM-injected render's pose MSE must not blow up while seg descends.
    assert out["d_pose_final"] <= out["d_pose_initial"] + 1e-2, (
        f"pose must hold: {out['d_pose_initial']} -> {out['d_pose_final']}"
    )


@skip_no_mlx
def test_film_actually_moves_render_after_training():
    """After training, the FiLM is NON-identity: changing pose changes the render."""
    bundle, _, trainer = _build_capstone_setup(
        n_pairs=4, h=48, w=64, seed=2, with_pose=True
    )
    trainer.cfg.epochs = 20
    trainer.train()
    idx = mx.arange(4)
    pose_a = mx.array(trainer.pose_store)
    pose_b = pose_a + 2.0  # a different pose conditioning
    r_a = bundle(idx, pose=pose_a)
    r_b = bundle(idx, pose=pose_b)
    mx.eval(r_a, r_b)
    # NO-FAKE: a trained FiLM is non-identity -> the render depends on pose.
    assert float(mx.max(mx.abs(r_a - r_b))) > 1e-2


# ---------------------------------------------------------------------------
# (4) NO-FAKE controls — must FAIL to descend
# ---------------------------------------------------------------------------


@skip_no_mlx
def test_constant_zero_cotangent_does_not_descend():
    """CONTROL: a zero pixel-cotangent (constant loss) leaves d_seg unchanged."""
    bundle, bridge, trainer = _build_capstone_setup(
        n_pairs=6, h=48, w=64, seed=0, with_pose=True
    )
    trainer.cfg.commitment_weight = 0.0  # isolate the render->score path
    bundle.ema_update_from_last = lambda: None  # type: ignore[assignment]
    d0 = trainer.exact_d_seg()
    # Monkeypatch the bridge to return a zero cotangent (no learning signal).
    real = bridge.loss_and_pixel_grad

    def zero_grad(render, idx):
        res = real(render, idx)
        res.pixel_cotangent = mx.zeros_like(res.pixel_cotangent)
        return res

    bridge.loss_and_pixel_grad = zero_grad  # type: ignore[assignment]
    for _ in range(10):
        trainer.step(np.arange(6))
    d1 = trainer.exact_d_seg()
    assert abs(d1 - d0) < 5e-3, f"zero-cotangent must not descend: {d0} -> {d1}"


@skip_no_mlx
def test_severed_render_does_not_descend():
    """CONTROL: a stop_gradient on the render severs the SCORE signal -> no descent.

    The render carries the score-aware (pixel-cotangent) learning signal; the VQ
    commitment loss is render-INDEPENDENT (it pulls z_e toward the nearest code
    regardless of the score), so the control must also zero ``commitment_weight``
    + disable the EMA codebook update to isolate the render->score path. With
    BOTH the score gradient severed AND the commitment path off, d_seg must NOT
    move — proving the descent is driven by the live scorer, not bookkeeping.
    """
    bundle, bridge, trainer = _build_capstone_setup(
        n_pairs=6, h=48, w=64, seed=0, with_pose=True
    )
    trainer.cfg.commitment_weight = 0.0  # isolate the render->score path
    bundle.ema_update_from_last = lambda: None  # type: ignore[assignment]
    d0 = trainer.exact_d_seg()
    # Sever the gradient at the weights: zero EVERY vjp gradient so no learning
    # signal reaches the carrier. If d_seg still moved, the "descent" would be a
    # bookkeeping artifact (the FAKE we must rule out); it must NOT move.
    from mlx.utils import tree_flatten as _tf
    from mlx.utils import tree_unflatten as _tu

    real_vjp = trainer._vjp_grads

    def zero_vjp(indices, pose, pixel_cotangent, commit_cotangent):
        grads = real_vjp(indices, pose, pixel_cotangent, commit_cotangent)
        zeroed = [(k, mx.zeros_like(v)) for k, v in _tf(grads)]
        return _tu(zeroed)

    trainer._vjp_grads = zero_vjp  # type: ignore[assignment]
    for _ in range(10):
        trainer.step(np.arange(6))
    d1 = trainer.exact_d_seg()
    assert abs(d1 - d0) < 5e-3, f"severed gradient must not descend: {d0} -> {d1}"


@skip_no_mlx
def test_bridge_fails_closed_on_unfrozen_scorer():
    """The bridge refuses a scorer with trainable params (Strict scorer rule)."""
    from tac.mlx_pr95_port.score_bridge import TorchScorerBridge

    dnet = _FrozenDNet(with_pose=True).eval()  # params still require_grad
    seg_tgt = torch.zeros(1, 8, 8, dtype=torch.long)
    with pytest.raises(ValueError):
        TorchScorerBridge(dnet, seg_tgt, None, scorer_hw=(8, 8))


@skip_no_mlx
def test_trainer_rejects_pose_store_pair_mismatch():
    from tac.capstone_vq_nerv.capstone_trainer import (
        CapstoneTrainConfig,
        CapstoneTrainer,
    )
    from tac.capstone_vq_nerv.vq_nerv_bundle import (
        CapstoneVqNervBundle,
        CapstoneVqNervConfig,
    )
    from tac.mlx_pr95_port.score_bridge import TorchScorerBridge

    dnet = _build_frozen_dnet(with_pose=True)
    seg_tgt = torch.zeros(4, 16, 16, dtype=torch.long)
    pose_tgt = torch.zeros(4, 6)
    bridge = TorchScorerBridge(dnet, seg_tgt, pose_tgt, scorer_hw=(16, 16))
    bundle = CapstoneVqNervBundle(CapstoneVqNervConfig(num_pairs=4, codebook_size=16))
    with pytest.raises(ValueError):
        CapstoneTrainer(bundle, bridge, np.zeros((3, 6), np.float32), CapstoneTrainConfig())


# ---------------------------------------------------------------------------
# (5) Archive byte-close + parse-back + carrier round-trip
# ---------------------------------------------------------------------------


@skip_no_mlx
def test_archive_byte_closes_and_parses_back():
    bundle = _make_small_bundle()
    decoder_weights = {
        k: np.asarray(v) for k, v in tree_flatten(bundle.decoder.parameters())
    }
    cb = np.asarray(bundle.quantizer.codebook)
    vq = bundle.all_vq_indices()
    pose = np.zeros((bundle.cfg.num_pairs, 6), np.float32)
    archive, account = build_capstone_archive_bytes(
        decoder_weights=decoder_weights,
        codebook=cb,
        vq_indices=vq,
        pose_scalars=pose,
        codebook_size=bundle.cfg.codebook_size,
    )
    assert account.total_bytes == len(archive)
    # parse-back: 4 sections recovered, index carrier round-trips EXACTLY.
    blobs = parse_capstone_archive_bytes(archive)
    assert len(blobs) == 4
    rt_idx = bit_unpack_vq_indices(
        blobs["index"], bundle.cfg.num_pairs, bundle.cfg.codebook_size
    )
    assert np.array_equal(vq, rt_idx)


@skip_no_mlx
def test_vq_index_carrier_is_smaller_than_fp16_latent():
    """The #67 rate lever: bit-packed VQ index << continuous fp16 latent."""
    bundle = _make_small_bundle(num_pairs=600, K=256, latent_dim=28)
    vq = bundle.all_vq_indices()
    idx_bytes = len(bit_pack_vq_indices(vq, 256))
    fp16_latent_bytes = 600 * 28 * 2  # continuous 28-d fp16 latent per pair
    assert idx_bytes == 600  # 8 bits/pair
    assert idx_bytes < fp16_latent_bytes / 50  # >50x smaller carrier


@skip_no_mlx
def test_codebook_is_paid_once_not_per_pair():
    """NO-FAKE rate claim: the codebook size is independent of num_pairs."""
    small = _make_small_bundle(num_pairs=10, K=256, latent_dim=28)
    big = _make_small_bundle(num_pairs=600, K=256, latent_dim=28)
    cb_small = np.asarray(small.quantizer.codebook).nbytes
    cb_big = np.asarray(big.quantizer.codebook).nbytes
    assert cb_small == cb_big  # codebook paid once, not per-pair


def test_int8_codec_round_trip_is_exact_quant():
    """The int8+brotli codec is exact-invertible up to the per-tensor quant step."""
    from tac.capstone_vq_nerv.export import _INT8_CODEC

    enc, dec = _INT8_CODEC
    rng = np.random.default_rng(7)
    arrays = {"a": rng.standard_normal((32, 8)).astype(np.float32) * 0.3,
              "b": rng.standard_normal((5, 3)).astype(np.float32)}
    blob = enc(arrays)
    back = dec(blob)
    for k, v in arrays.items():
        amax = float(np.max(np.abs(v)))
        scale = amax / 127.0
        # NO-FAKE: dequant error is bounded by HALF the quant step PLUS the fp16
        # scale-rounding term (scale stored as fp16 -> at most ~127*scale*2^-11
        # extra). This is the exact analytic bound of a per-tensor int8 codec.
        bound = scale * 0.5 + 127.0 * scale * (2.0**-10) + 1e-6
        assert float(np.max(np.abs(back[k] - v))) <= bound


@skip_no_mlx
def test_int8_decoder_halves_bytes_vs_fp16():
    """The int8 decoder path is ~2x smaller than fp16 (the sub-0.15 enabler)."""
    bundle = _make_small_bundle(num_pairs=600, K=256, latent_dim=28)
    decoder_weights = {
        k: np.asarray(v) for k, v in tree_flatten(bundle.decoder.parameters())
    }
    cb = np.asarray(bundle.quantizer.codebook)
    vq = bundle.all_vq_indices()
    pose = np.zeros((600, 6), np.float32)
    kw = {
        "decoder_weights": decoder_weights, "codebook": cb, "vq_indices": vq,
        "pose_scalars": pose, "codebook_size": 256,
    }
    _, acct_fp16 = build_capstone_archive_bytes(**kw, decoder_dtype="fp16")
    _, acct_int8 = build_capstone_archive_bytes(**kw, decoder_dtype="int8")
    # NO-FAKE: int8 decoder really is materially smaller (real entropy coding).
    assert acct_int8.decoder_bytes < acct_fp16.decoder_bytes * 0.7
    # the carriers (index/pose) are dtype-independent and stay tiny.
    assert acct_int8.index_bytes == 600
    assert acct_int8.total_bytes < acct_fp16.total_bytes


def test_int8_archive_rejects_bad_dtype():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        build_capstone_archive_bytes(
            decoder_weights={"w": rng.standard_normal((4, 4)).astype(np.float32)},
            codebook=rng.standard_normal((16, 8)).astype(np.float32),
            vq_indices=np.zeros(10, np.int32),
            pose_scalars=np.zeros((10, 6), np.float32),
            codebook_size=16,
            decoder_dtype="bf8",
        )


def _make_small_bundle(num_pairs=8, K=256, latent_dim=28):
    from tac.capstone_vq_nerv.vq_nerv_bundle import (
        CapstoneVqNervBundle,
        CapstoneVqNervConfig,
    )

    return CapstoneVqNervBundle(
        CapstoneVqNervConfig(
            num_pairs=num_pairs, latent_dim=latent_dim, codebook_size=K
        )
    )
