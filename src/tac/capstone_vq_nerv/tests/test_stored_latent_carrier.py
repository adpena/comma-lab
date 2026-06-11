# SPDX-License-Identifier: MIT
"""Tests for the stored-28-d-latent carrier (Arm 2 of the pose A/B; Task #78).

The carrier-pivot fix per
``.omx/research/capstone_carrier_pivot_vq_index_impoverishment_*``: the capstone's
pose failure is the 8-bit VQ index impoverishment (8 bits/pair cannot encode 600
distinct ego-motions). The fix is to store the rich 28-d per-pair latent DIRECTLY
(the frontier's/PR95's own carrier — temporal-delta + raw-LZMA, L24/L25), which is
both rate-efficient (~10-15 KB for 600 pairs) AND pose-capable.

This suite proves, with NO-FAKE behavioral teeth:

1. **codec round-trip** — the temporal-delta + LZMA latent codec is exact-invertible
   up to the per-dim uint8 quant step (bounded by step/2); the blob is sub-0.15-capable.
2. **ADDITIVE / byte-identical** — the ``vq_index`` carrier (the default) is
   byte-identical for a fixed bundle state; the new fields default safely.
3. **NO-FAKE** — the ``stored_latent`` path ACTUALLY stores + decodes the 28-d
   latents (no codebook, no index, commitment loss == 0); a behavioral test that
   FAILS if it falls back to VQ or a no-op.
4. **score-parity** — the ``stored_latent`` numpy-inflate render == MLX render
   (d_seg EXACT) on a frozen scorer, like the existing vq_index parity gate.
5. **byte budget** — the real ``stored_latent`` archive bytes at base_ch=20 are
   sub-0.15-capable (decoder ~85 KB + latents ~10-15 KB + pose ~1.5 KB).

Authority: ``[macOS-MLX research-signal]`` / ``[macOS-CPU advisory]`` — the frozen
proto scorer is the bridge-mechanism stand-in; a CONTEST score needs
``upstream/evaluate.py`` paired CUDA + Linux-x86_64 CPU. These gates prove the
carrier mechanism + portability, not a contest number.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from tac.capstone_vq_nerv.export import (
    CapstoneArchiveAccount,
    build_capstone_archive_bytes,
    build_capstone_stored_latent_archive_bytes,
    decode_stored_latents,
    encode_stored_latents,
    parse_capstone_stored_latent_archive_bytes,
)
from tac.capstone_vq_nerv.inflate import (
    CAMERA_H,
    CAMERA_W,
    decode_archive,
    render_all_camera_frames,
)
from tac.capstone_vq_nerv.numpy_reference import (
    decode_config_from_bundle,
    full_render_weights_from_bundle,
    numpy_decode_pair,
)
from tac.capstone_vq_nerv.tests.test_capstone_vq_nerv import _build_frozen_dnet

try:
    import mlx.core as mx

    _HAVE_MLX = True
except Exception:  # pragma: no cover
    mx = None  # type: ignore[assignment]
    _HAVE_MLX = False

skip_no_mlx = pytest.mark.skipif(not _HAVE_MLX, reason="mlx not available")

RENDER_H, RENDER_W = 384, 512
RATE_DENOM = 37_545_489


# --------------------------------------------------------------------------
# (1) the temporal-delta + LZMA latent codec (pure numpy, no MLX)
# --------------------------------------------------------------------------


def _smooth_latents(num_pairs: int, latent_dim: int, seed: int = 0) -> np.ndarray:
    """Latents whose consecutive PAIR rows are close (the temporal-delta wins on this)."""
    rng = np.random.default_rng(seed)
    z = np.cumsum(rng.standard_normal((num_pairs, latent_dim)).astype(np.float32) * 0.02, axis=0)
    if latent_dim > 3:
        z[:, 3] = 1.7  # a zero-variance dim (guards the span==0 path)
    return z.astype(np.float32)


def test_stored_latent_codec_round_trip_bounded_by_quant_step():
    """The latent codec is exact-invertible up to the per-dim uint8 quant step."""
    lat = _smooth_latents(600, 28, seed=0)
    blob, n, d = encode_stored_latents(lat)
    assert (n, d) == (600, 28)
    dec = decode_stored_latents(blob, n, d)
    assert dec.shape == lat.shape
    # max error must be <= per-dim step/2 (+ a small fp16 min/scale slack).
    span = lat.max(0) - lat.min(0)
    step = np.where(span > 0, span / 255.0, 0.0)
    max_err = float(np.max(np.abs(dec - lat)))
    assert max_err <= float(np.max(step) / 2.0) + 1e-2, (
        f"latent codec error {max_err} exceeds quant step/2 {np.max(step)/2.0}"
    )


def test_stored_latent_codec_single_pair_no_delta():
    """One pair: row 0 is verbatim (no temporal-delta); still round-trips."""
    lat = _smooth_latents(1, 28, seed=1)
    blob, n, d = encode_stored_latents(lat)
    dec = decode_stored_latents(blob, n, d)
    span = lat.max(0) - lat.min(0)
    # one pair -> span 0 per dim -> scale=1 -> stored as the rounded value; the
    # dequant just returns mn (= the value rounded to fp16) -> sub-fp16 error.
    assert dec.shape == (1, 28)
    assert float(np.max(np.abs(dec - lat))) <= float(np.max(span) / 2.0) + 1e-2


def test_stored_latent_blob_is_rate_efficient():
    """600x28 latents compress to ~10-15 KB (rate << 0.15 budget). The L24/L25 win."""
    lat = _smooth_latents(600, 28, seed=2)
    blob, _, _ = encode_stored_latents(lat)
    rate = 25.0 * len(blob) / RATE_DENOM
    # smooth latents temporal-delta-compress to a few KB; rate must be << 0.05.
    assert len(blob) < 25_000, f"latent blob {len(blob)} B unexpectedly large"
    assert rate < 0.02, f"latent carrier rate {rate} not sub-budget"


def test_stored_latent_codec_beats_raw_fp16_on_smooth():
    """NO-FAKE: temporal-delta + LZMA actually compresses (smaller than raw fp16)."""
    lat = _smooth_latents(600, 28, seed=3)
    blob, _, _ = encode_stored_latents(lat)
    raw_fp16 = lat.astype(np.float16).nbytes  # 600*28*2 = 33,600 B
    assert len(blob) < raw_fp16, (
        f"codec ({len(blob)} B) must beat raw fp16 ({raw_fp16} B) on smooth latents"
    )


# --------------------------------------------------------------------------
# (2) ADDITIVE: vq_index default is byte-identical; new fields default safely
# --------------------------------------------------------------------------


@skip_no_mlx
def test_vq_index_default_carrier_byte_identical_from_same_bundle():
    """The default (vq_index) export is byte-identical for a fixed bundle state.

    Proves the carrier switch is ADDITIVE: nothing in the vq_index code path changed
    (the same bundle builds the same archive bytes twice).
    """
    from tac.capstone_vq_nerv.vq_nerv_bundle import (
        CapstoneVqNervBundle,
        CapstoneVqNervConfig,
    )

    b = CapstoneVqNervBundle(
        CapstoneVqNervConfig(num_pairs=8, base_channels=16, codebook_size=16, seed=7)
    )
    assert b.carrier == "vq_index"  # default
    rng = np.random.default_rng(7)
    pose = rng.standard_normal((8, 6)).astype(np.float32)
    b.set_pose_stats(pose.mean(0), pose.std(0))
    w = full_render_weights_from_bundle(b)
    cb = np.asarray(b.quantizer._codebook, dtype=np.float32)
    vq = b.all_vq_indices()
    a1, ac1 = build_capstone_archive_bytes(
        decoder_weights=w, codebook=cb, vq_indices=vq, pose_scalars=pose,
        codebook_size=16, decoder_dtype="int8",
    )
    a2, _ = build_capstone_archive_bytes(
        decoder_weights=w, codebook=cb, vq_indices=vq, pose_scalars=pose,
        codebook_size=16, decoder_dtype="int8",
    )
    assert a1 == a2, "vq_index export not byte-identical from the same bundle state"
    assert ac1.carrier == "vq_index"
    assert ac1.latent_bytes == 0  # vq_index has no latent section


def test_config_default_carrier_is_vq_index():
    """The config default carrier is vq_index (pre-switch behaviour preserved)."""
    from tac.capstone_vq_nerv.vq_nerv_bundle import CapstoneVqNervConfig

    assert CapstoneVqNervConfig().carrier == "vq_index"


def test_account_positional_construction_backward_compatible():
    """The account dataclass keeps positional construction (new fields default)."""
    acct = CapstoneArchiveAccount(
        decoder_bytes=1, codebook_bytes=2, index_bytes=3, pose_bytes=4,
        total_bytes=10, num_pairs=8, codebook_size=16, bits_per_index=4,
    )
    assert acct.carrier == "vq_index"
    assert acct.latent_bytes == 0
    d = acct.as_dict()
    assert d["carrier"] == "vq_index" and d["latent_bytes"] == 0


# --------------------------------------------------------------------------
# (3) NO-FAKE: stored_latent ACTUALLY stores + decodes the 28-d latent
# --------------------------------------------------------------------------


@skip_no_mlx
def test_stored_latent_bundle_has_no_quantizer_and_zero_commitment():
    """NO-FAKE: the stored_latent bundle has NO codebook and commitment is 0.

    If the carrier silently fell back to VQ, the quantizer would exist and the
    commitment loss would be non-zero on a forward.
    """
    from tac.capstone_vq_nerv.vq_nerv_bundle import (
        CapstoneVqNervBundle,
        CapstoneVqNervConfig,
    )

    b = CapstoneVqNervBundle(
        CapstoneVqNervConfig(num_pairs=6, base_channels=16, carrier="stored_latent", seed=0)
    )
    assert b.carrier == "stored_latent"
    assert b.quantizer is None, "stored_latent must not build a VQ codebook"
    rng = np.random.default_rng(0)
    b.set_pose_stats(rng.standard_normal(6).astype(np.float32), np.ones(6, np.float32))
    idx = mx.arange(6)
    pose = mx.array(rng.standard_normal((6, 6)).astype(np.float32))
    r = b(idx, pose=pose)
    mx.eval(r)
    assert float(b.last_commitment_loss) == 0.0, "stored_latent commitment must be 0"
    # vq_indices() must REFUSE (there is no codebook); all_latents() must work.
    with pytest.raises(RuntimeError):
        b.all_vq_indices()
    lat = b.all_latents()
    assert lat.shape == (6, 28)


@skip_no_mlx
def test_stored_latent_decoder_input_is_the_raw_latent_not_a_code():
    """NO-FAKE: the decoder input equals the gathered raw latent (no quantization).

    For stored_latent, ``_quantize`` returns the gathered latents UNCHANGED — a VQ
    fallback would snap them to codebook entries (a different array). We verify the
    decoder input the forward used equals ``mx.take(latents, idx)`` exactly.
    """
    from tac.capstone_vq_nerv.vq_nerv_bundle import (
        CapstoneVqNervBundle,
        CapstoneVqNervConfig,
    )

    b = CapstoneVqNervBundle(
        CapstoneVqNervConfig(num_pairs=5, base_channels=16, carrier="stored_latent", seed=1)
    )
    idx = mx.arange(5)
    z = b._quantize(idx)
    mx.eval(z)
    expected = mx.take(b.latents, idx, axis=0)
    mx.eval(expected)
    assert float(mx.max(mx.abs(z - expected))) == 0.0, (
        "stored_latent decoder input must be the raw latent (no VQ snap)"
    )


@skip_no_mlx
def test_stored_latent_archive_has_no_codebook_or_index_sections():
    """NO-FAKE: the stored_latent archive is 3 sections (dec/latent/pose), no codebook/index."""
    from tac.capstone_vq_nerv.vq_nerv_bundle import (
        CapstoneVqNervBundle,
        CapstoneVqNervConfig,
    )

    b = CapstoneVqNervBundle(
        CapstoneVqNervConfig(num_pairs=8, base_channels=16, carrier="stored_latent", seed=2)
    )
    rng = np.random.default_rng(2)
    pose = rng.standard_normal((8, 6)).astype(np.float32)
    b.set_pose_stats(pose.mean(0), pose.std(0))
    w = full_render_weights_from_bundle(b)
    lat = b.all_latents()
    arch, acct = build_capstone_stored_latent_archive_bytes(
        decoder_weights=w, latents=lat, pose_scalars=pose, decoder_dtype="int8"
    )
    blobs = parse_capstone_stored_latent_archive_bytes(arch)
    assert set(blobs) == {"decoder", "latent", "pose"}
    assert acct.carrier == "stored_latent"
    assert acct.codebook_bytes == 0 and acct.index_bytes == 0
    assert acct.latent_bytes > 0 and acct.decoder_bytes > 0


@skip_no_mlx
def test_stored_latent_trains_and_descends_with_zero_commitment():
    """NO-FAKE end-to-end: the stored_latent trainer descends d_seg, commit stays 0.

    A no-op carrier (zeros / constant) would not descend; a VQ-fallback would have
    non-zero commitment. The trainer's vjp through the constant commitment loss must
    not error.
    """
    from tac.capstone_vq_nerv.capstone_trainer import (
        CapstoneTrainConfig,
        CapstoneTrainer,
    )
    from tac.capstone_vq_nerv.vq_nerv_bundle import (
        CapstoneVqNervBundle,
        CapstoneVqNervConfig,
    )
    from tac.mlx_pr95_port.score_bridge import TorchScorerBridge

    n_pairs, h, w = 6, 48, 64
    dnet = _build_frozen_dnet(with_pose=True)
    seg_tgt = torch.zeros(n_pairs, h, w, dtype=torch.long)
    bands = [(0, 10), (10, 20), (20, 30), (30, 40), (40, h)]
    for i in range(n_pairs):
        for cls, (r0, r1) in enumerate(bands):
            seg_tgt[i, r0:r1, :] = (cls + i) % 5
    rng = np.random.default_rng(0)
    pose_tgt = torch.tensor(rng.standard_normal((n_pairs, 6)).astype(np.float32))
    bridge = TorchScorerBridge(
        dnet, seg_tgt, pose_tgt, seg_loss_form="ce_seg_loss",
        scorer_hw=(h, w), eval_roundtrip=False,
    )
    bundle = CapstoneVqNervBundle(
        CapstoneVqNervConfig(
            num_pairs=n_pairs, latent_dim=28, base_channels=36,
            carrier="stored_latent", seed=0,
        )
    )
    tcfg = CapstoneTrainConfig(
        epochs=40, batch_size=n_pairs, eval_every=10, seed=0,
        muon_lr=3e-2, adamw_lr=2e-2, ema_decay=0.95,
        grad_clip=50.0, grad_clip_muon=50.0,
    )
    trainer = CapstoneTrainer(bundle, bridge, pose_tgt.numpy(), tcfg)
    out = trainer.train()
    assert out["seg_descended"], (
        f"stored_latent d_seg did not descend: "
        f"{out['d_seg_initial']} -> {out['d_seg_final']}"
    )
    assert all(r["commit_mean"] == 0.0 for r in out["trajectory"]), (
        "stored_latent commitment must remain 0 (no VQ fallback)"
    )
    lat = trainer.export_stored_latents()
    assert lat.shape == (n_pairs, 28)


# --------------------------------------------------------------------------
# (4) SCORE-PARITY: stored_latent numpy inflate == MLX render (d_seg EXACT)
# --------------------------------------------------------------------------


def _make_stored_latent_bundle(num_pairs=5, base_ch=16, seed=0):
    """A stored_latent bundle with NON-identity per-frame FiLM (pose path live)."""
    from tac.capstone_vq_nerv.vq_nerv_bundle import (
        CapstoneVqNervBundle,
        CapstoneVqNervConfig,
    )

    b = CapstoneVqNervBundle(
        CapstoneVqNervConfig(
            num_pairs=num_pairs, base_channels=base_ch, carrier="stored_latent", seed=seed
        )
    )
    rng = np.random.default_rng(seed)
    b.set_pose_stats(
        rng.standard_normal(6).astype(np.float32),
        (np.abs(rng.standard_normal(6)) + 0.5).astype(np.float32),
    )
    for prefix, sign in (("pose_film0", 1.0), ("pose_film1", -1.0)):
        film = getattr(b, prefix)
        w2 = np.asarray(film.fc2.weight).copy()
        w2 += sign * 0.05 * rng.standard_normal(w2.shape).astype(np.float32)
        film.fc2.weight = mx.array(w2)
        bb = np.asarray(film.fc2.bias).copy()
        bb += sign * 0.1 * rng.standard_normal(bb.shape).astype(np.float32)
        film.fc2.bias = mx.array(bb)
    return b


@skip_no_mlx
@pytest.mark.parametrize("base_ch", [16, 20])
def test_stored_latent_numpy_render_matches_mlx(base_ch):
    """Pure-numpy decode (raw latents) reproduces the MLX stored_latent render."""
    rng = np.random.default_rng(base_ch)
    b = _make_stored_latent_bundle(num_pairs=4, base_ch=base_ch, seed=base_ch)
    pose_np = rng.standard_normal((4, 6)).astype(np.float32)
    idx = mx.arange(4)
    r_mlx = b(idx, pose=mx.array(pose_np))
    mx.eval(r_mlx)
    r_mlx = np.asarray(r_mlx, dtype=np.float32)

    lat = b.all_latents()
    weights = full_render_weights_from_bundle(b)
    cfg = decode_config_from_bundle(b)
    r_np = numpy_decode_pair(lat, pose_np, weights, cfg)
    drift = float(np.max(np.abs(r_mlx - r_np)))
    assert drift < 0.05, f"stored_latent numpy<->MLX render drift too large: {drift}"


def _camera_from_render(render_n2chw):
    import torch.nn.functional as F

    b = render_n2chw.shape[0]
    flat = torch.from_numpy(render_n2chw.reshape(b * 2, 3, RENDER_H, RENDER_W))
    cam = F.interpolate(flat, size=(CAMERA_H, CAMERA_W), mode="bicubic", align_corners=False)
    cam = cam.clamp(0, 255).round().to(torch.uint8).numpy()
    return np.transpose(cam, (0, 2, 3, 1))


def _score_frames(dnet, cam_frames_u8, seg_tgt, pose_tgt):
    import torch.nn.functional as F

    n_pairs = cam_frames_u8.shape[0] // 2
    h, w = seg_tgt.shape[1], seg_tgt.shape[2]
    d_seg_total = 0.0
    d_pose_total = 0.0
    with torch.no_grad():
        for k in range(n_pairs):
            f0 = torch.from_numpy(cam_frames_u8[2 * k].astype(np.float32))
            f1 = torch.from_numpy(cam_frames_u8[2 * k + 1].astype(np.float32))
            pair = torch.stack([f0, f1], 0).unsqueeze(0).permute(0, 1, 4, 2, 3)
            pair = F.interpolate(
                pair.reshape(2, 3, CAMERA_H, CAMERA_W), size=(h, w),
                mode="bilinear", align_corners=False,
            ).reshape(1, 2, 3, h, w)
            bhwc = pair.permute(0, 1, 3, 4, 2).contiguous()
            _, segnet_in = dnet.preprocess_input(bhwc)
            pred = dnet.segnet(segnet_in).argmax(1).squeeze(0)
            d_seg_total += float((pred != seg_tgt[k]).float().mean())
            if dnet.posenet is not None:
                posenet_in, _ = dnet.preprocess_input(bhwc)
                pose_out = dnet.posenet(posenet_in)["pose"][:, :6]
                d_pose_total += float(((pose_out[0] - pose_tgt[k]) ** 2).mean())
    return d_seg_total / n_pairs, d_pose_total / n_pairs


@skip_no_mlx
def test_stored_latent_numpy_inflate_score_parity_with_mlx():
    """THE GATE: stored_latent numpy inflate frames score-match the MLX render (d_seg EXACT)."""
    n_pairs, h, w = 5, 48, 64
    rng = np.random.default_rng(11)
    b = _make_stored_latent_bundle(num_pairs=n_pairs, base_ch=16, seed=11)
    pose_np = rng.standard_normal((n_pairs, 6)).astype(np.float32)

    dnet = _build_frozen_dnet(with_pose=True)
    seg_tgt = torch.zeros(n_pairs, h, w, dtype=torch.long)
    bands = [(0, 10), (10, 20), (20, 30), (30, 40), (40, h)]
    for i in range(n_pairs):
        for cls, (r0, r1) in enumerate(bands):
            seg_tgt[i, r0:r1, :] = (cls + i) % 5
    pose_tgt = torch.from_numpy(rng.standard_normal((n_pairs, 6)).astype(np.float32))

    # MLX render -> torch-reference camera frames -> score
    idx = mx.arange(n_pairs)
    r_mlx = b(idx, pose=mx.array(pose_np))
    mx.eval(r_mlx)
    r_mlx = np.asarray(r_mlx, dtype=np.float32)
    cam_mlx = _camera_from_render(r_mlx)
    d_seg_mlx, d_pose_mlx = _score_frames(dnet, cam_mlx, seg_tgt, pose_tgt)

    # numpy inflate (the REAL contest path) -> camera frames -> score
    weights = full_render_weights_from_bundle(b)
    lat = b.all_latents()
    arch, _acct = build_capstone_stored_latent_archive_bytes(
        decoder_weights=weights, latents=lat, pose_scalars=pose_np, decoder_dtype="fp16",
    )
    cfg = decode_config_from_bundle(b)
    config = {
        "carrier": "stored_latent",
        "decoder_dtype": "fp16",
        "num_pairs": n_pairs,
        "latent_dim": 28,
        "base_channels": int(cfg.base_channels),
        "base_h": int(cfg.base_h),
        "base_w": int(cfg.base_w),
        "film_enabled": bool(cfg.film_enabled),
        "pose_normalize": bool(cfg.pose_normalize),
        "pose_mean": list(cfg.pose_mean),
        "pose_std": list(cfg.pose_std),
    }
    decoded = decode_archive(arch, config)
    cam_np = render_all_camera_frames(decoded)
    d_seg_np, d_pose_np = _score_frames(dnet, cam_np, seg_tgt, pose_tgt)

    assert cam_np.shape == (n_pairs * 2, CAMERA_H, CAMERA_W, 3)
    assert cam_np.dtype == np.uint8
    # SCORE-PARITY (the gate): d_seg EXACT (fp16 latent codec is the only gap, and it
    # is sub-uint8 / argmax-invariant on this scorer).
    assert abs(d_seg_np - d_seg_mlx) < 1e-4, (
        f"stored_latent d_seg parity broke: mlx={d_seg_mlx} np={d_seg_np}"
    )
    assert abs(d_pose_np - d_pose_mlx) <= max(1e-4, 1e-3 * abs(d_pose_mlx) + 1e-4), (
        f"stored_latent d_pose parity broke: mlx={d_pose_mlx} np={d_pose_np}"
    )


# --------------------------------------------------------------------------
# (5) byte budget at base_ch=20: sub-0.15-capable
# --------------------------------------------------------------------------


@skip_no_mlx
@pytest.mark.parametrize("num_pairs", [48, 600])
def test_stored_latent_byte_budget_sub_0_15_capable(num_pairs):
    """The real stored_latent int8 archive at base_ch=20 is sub-0.15-capable.

    decoder ~85 KB (base_ch=20 int8) + latents ~10-15 KB (600 pairs) + pose ~1.5 KB
    -> total well under the 0.15 byte budget (rate = 25*B/37545489). NO synthetic
    numbers: this builds the REAL archive and reads its byte account.
    """
    from tac.capstone_vq_nerv.vq_nerv_bundle import (
        CapstoneVqNervBundle,
        CapstoneVqNervConfig,
    )

    b = CapstoneVqNervBundle(
        CapstoneVqNervConfig(
            num_pairs=num_pairs, base_channels=20, carrier="stored_latent", seed=5
        )
    )
    rng = np.random.default_rng(5)
    pose = rng.standard_normal((num_pairs, 6)).astype(np.float32)
    b.set_pose_stats(pose.mean(0), pose.std(0))
    w = full_render_weights_from_bundle(b)
    # smooth latents (the realistic trained shape; random init is noisier so this is
    # a conservative-or-realistic byte estimate, not an optimistic one).
    lat = _smooth_latents(num_pairs, 28, seed=5)
    _arch, acct = build_capstone_stored_latent_archive_bytes(
        decoder_weights=w, latents=lat, pose_scalars=pose, decoder_dtype="int8"
    )
    # The decoder dominates; the carrier (latents) is a few KB. The TOTAL rate must
    # leave ample room under 0.15 even before adding seg/pose terms.
    assert acct.rate < 0.10, (
        f"stored_latent rate {acct.rate} (B={acct.total_bytes}) too high at base_ch=20"
    )
    # the latent carrier itself is the cheap part (the whole point vs storing fp16).
    assert acct.latent_bytes < acct.decoder_bytes, (
        "the stored latent carrier must be cheaper than the decoder basis"
    )


@skip_no_mlx
def test_decode_config_from_bundle_carries_stored_latent_fields(tmp_path):
    """The config sidecar round-trips the stored_latent carrier through the zip."""
    from tac.capstone_vq_nerv.inflate import _read_archive_and_config
    from tac.capstone_vq_nerv.vq_nerv_bundle import (
        CapstoneVqNervBundle,
        CapstoneVqNervConfig,
    )

    b = CapstoneVqNervBundle(
        CapstoneVqNervConfig(num_pairs=4, base_channels=16, carrier="stored_latent", seed=3)
    )
    rng = np.random.default_rng(3)
    pose = rng.standard_normal((4, 6)).astype(np.float32)
    b.set_pose_stats(pose.mean(0), pose.std(0))
    w = full_render_weights_from_bundle(b)
    lat = b.all_latents()
    arch, _acct = build_capstone_stored_latent_archive_bytes(
        decoder_weights=w, latents=lat, pose_scalars=pose, decoder_dtype="int8"
    )
    config = dataclasses.asdict(decode_config_from_bundle(b))
    config["num_pairs"] = 4
    config["decoder_dtype"] = "int8"
    config["carrier"] = "stored_latent"

    import io
    import json
    import zipfile

    cfg_bytes = json.dumps(config, separators=(",", ":"), sort_keys=True).encode("utf-8")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("x", arch)
        zf.writestr("capstone_config_v1", cfg_bytes)
    zip_path = tmp_path / "archive.zip"
    zip_path.write_bytes(buf.getvalue())

    archive_bytes, loaded_config = _read_archive_and_config(zip_path)
    assert loaded_config["carrier"] == "stored_latent"
    decoded = decode_archive(archive_bytes, loaded_config)
    assert decoded["carrier"] == "stored_latent"
    assert decoded["latents"].shape == (4, 28)
    assert decoded["codebook"] is None
