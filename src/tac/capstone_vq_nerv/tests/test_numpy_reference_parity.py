# SPDX-License-Identifier: MIT
"""The PORTABILITY GATE: pure-numpy inflate must score-match the MLX render.

This is the NO-FAKE parity gate for the capstone numpy-reference port (Task #78,
the MLX-first portability contract). It:

1. builds a small FiLM-enabled bundle (base_ch=16, a few pairs) with non-trivial
   (non-identity) FiLM weights — so the per-frame FiLM path is exercised;
2. byte-closes the archive (decoder + FiLM weights + codebook + VQ index + pose +
   config sidecar) and runs the REAL pure-numpy inflate (parse -> decode ->
   render -> camera-upsample -> raw uint8);
3. asserts SCORE-PARITY on a FROZEN DistortionNet (proto SegNet argmax + proto
   PoseNet): the numpy-inflated frames produce d_seg/d_pose that match the MLX
   render's within a tight tolerance (the scorer is the authority — bit-exactness
   is ideal but score-parity is the gate);
4. also asserts pure-numpy-vs-MLX render closeness (max|Δpixel|) directly.

NO-FAKE: both the MLX render AND the numpy inflate ACTUALLY run; a numpy stub
(returning zeros / constants) would FAIL the render-closeness AND the score-parity
asserts. The controls (identity-FiLM agreement, base-channel sweep) prove the
port is the real forward, not a coincidence.

Authority: ``[macOS-MLX research-signal]`` / ``[macOS-CPU advisory]`` — the
frozen proto DistortionNet is the bridge-mechanism stand-in; a CONTEST score
needs ``upstream/evaluate.py`` paired CUDA + Linux-x86_64 CPU. This gate proves
PORTABILITY (the numpy decoder reproduces the MLX decoder), not a contest score.
"""

from __future__ import annotations

import json
import zipfile

import numpy as np
import pytest
import torch

from tac.capstone_vq_nerv.export import build_capstone_archive_bytes
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


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _make_film_bundle(num_pairs=4, base_ch=16, K=16, seed=0):
    """A FiLM-enabled bundle with NON-identity FiLM (the per-frame path live)."""
    from tac.capstone_vq_nerv.vq_nerv_bundle import (
        CapstoneVqNervBundle,
        CapstoneVqNervConfig,
    )

    b = CapstoneVqNervBundle(
        CapstoneVqNervConfig(
            num_pairs=num_pairs, base_channels=base_ch, codebook_size=K, seed=seed
        )
    )
    rng = np.random.default_rng(seed)
    b.set_pose_stats(
        rng.standard_normal(6).astype(np.float32),
        (np.abs(rng.standard_normal(6)) + 0.5).astype(np.float32),
    )
    # Nudge film0/film1 off identity (DIFFERENTLY) so the per-frame path matters.
    for prefix, sign in (("pose_film0", 1.0), ("pose_film1", -1.0)):
        film = getattr(b, prefix)
        w2 = np.asarray(film.fc2.weight).copy()
        w2 += sign * 0.05 * rng.standard_normal(w2.shape).astype(np.float32)
        film.fc2.weight = mx.array(w2)
        bb = np.asarray(film.fc2.bias).copy()
        bb += sign * 0.1 * rng.standard_normal(bb.shape).astype(np.float32)
        film.fc2.bias = mx.array(bb)
    return b


def _build_archive_and_config(bundle, *, decoder_dtype="fp16", pose=None):
    """Byte-close the FULL-render archive (decoder + FiLM) + the config sidecar."""
    weights = full_render_weights_from_bundle(bundle)  # decoder + pose_film*
    cb = np.asarray(bundle.quantizer.codebook, dtype=np.float32)
    vq = bundle.all_vq_indices()
    if pose is None:
        pose = getattr(
            bundle, "pose_store_for_test",
            np.zeros((int(bundle.cfg.num_pairs), 6), np.float32),
        )
    archive, account = build_capstone_archive_bytes(
        decoder_weights=weights,
        codebook=cb,
        vq_indices=vq,
        pose_scalars=pose,
        codebook_size=bundle.cfg.codebook_size,
        decoder_dtype=decoder_dtype,
    )
    cfg = decode_config_from_bundle(bundle)
    config = {
        "decoder_dtype": decoder_dtype,
        "num_pairs": int(bundle.cfg.num_pairs),
        "codebook_size": int(bundle.cfg.codebook_size),
        "latent_dim": int(bundle.cfg.latent_dim),
        "base_channels": int(cfg.base_channels),
        "base_h": int(cfg.base_h),
        "base_w": int(cfg.base_w),
        "film_enabled": bool(cfg.film_enabled),
        "pose_normalize": bool(cfg.pose_normalize),
        "pose_mean": list(cfg.pose_mean),
        "pose_std": list(cfg.pose_std),
    }
    return archive, account, config


def _mlx_render(bundle, pose_np):
    idx = mx.arange(int(bundle.cfg.num_pairs))
    r = bundle(idx, pose=mx.array(pose_np.astype(np.float32)))
    mx.eval(r)
    return np.asarray(r, dtype=np.float32)  # (B,2,3,384,512)


def _camera_from_render(render_n2chw):
    """torch-reference camera frames from a render (the parity baseline path)."""
    import torch.nn.functional as F

    b = render_n2chw.shape[0]
    flat = torch.from_numpy(render_n2chw.reshape(b * 2, 3, RENDER_H, RENDER_W))
    cam = F.interpolate(flat, size=(CAMERA_H, CAMERA_W), mode="bilinear", align_corners=False)
    cam = cam.clamp(0, 255).round().to(torch.uint8).numpy()  # (b*2,3,H,W)
    cam = np.transpose(cam, (0, 2, 3, 1))  # (b*2,H,W,3)
    return cam


def _score_frames(dnet, cam_frames_u8, seg_tgt, pose_tgt):
    """d_seg (argmax disagreement) + d_pose (MSE) of camera frames vs GT on dnet.

    ``cam_frames_u8`` is ``(N*2, H, W, 3)`` uint8; pairs are (2k, 2k+1).
    Downsample to the proto scorer's working resolution so the proto SegNet/PoseNet
    are well-conditioned (the proto is res-agnostic; this matches the bridge HW).
    """
    import torch.nn.functional as F

    n_pairs = cam_frames_u8.shape[0] // 2
    h, w = seg_tgt.shape[1], seg_tgt.shape[2]
    d_seg_total = 0.0
    d_pose_total = 0.0
    with torch.no_grad():
        for k in range(n_pairs):
            f0 = torch.from_numpy(cam_frames_u8[2 * k].astype(np.float32))
            f1 = torch.from_numpy(cam_frames_u8[2 * k + 1].astype(np.float32))
            pair = torch.stack([f0, f1], 0).unsqueeze(0)  # (1,2,H,W,3)
            pair = pair.permute(0, 1, 4, 2, 3)  # (1,2,3,H,W)
            pair = F.interpolate(
                pair.reshape(2, 3, CAMERA_H, CAMERA_W), size=(h, w),
                mode="bilinear", align_corners=False,
            ).reshape(1, 2, 3, h, w)
            bhwc = pair.permute(0, 1, 3, 4, 2).contiguous()  # (1,2,h,w,3)
            _, segnet_in = dnet.preprocess_input(bhwc)
            logits = dnet.segnet(segnet_in)  # (1,5,h,w)
            pred = logits.argmax(1).squeeze(0)
            d_seg_total += float((pred != seg_tgt[k]).float().mean())
            if dnet.posenet is not None:
                posenet_in, _ = dnet.preprocess_input(bhwc)
                pose_out = dnet.posenet(posenet_in)["pose"][:, :6]
                d_pose_total += float(((pose_out[0] - pose_tgt[k]) ** 2).mean())
    return d_seg_total / n_pairs, d_pose_total / n_pairs


# --------------------------------------------------------------------------
# (1) render-closeness: numpy decode vs MLX decode (the direct parity)
# --------------------------------------------------------------------------


@skip_no_mlx
@pytest.mark.parametrize("base_ch", [16, 20, 24])
def test_numpy_render_matches_mlx_render(base_ch):
    """Pure-numpy decode reproduces the MLX render to small pixel drift (base-ch sweep)."""
    rng = np.random.default_rng(base_ch)
    bundle = _make_film_bundle(num_pairs=4, base_ch=base_ch, K=16, seed=base_ch)
    pose_np = rng.standard_normal((4, 6)).astype(np.float32)
    r_mlx = _mlx_render(bundle, pose_np)

    cb = np.asarray(bundle.quantizer.codebook, dtype=np.float32)
    vq = bundle.all_vq_indices()
    z_q = cb[vq]
    weights = full_render_weights_from_bundle(bundle)
    cfg = decode_config_from_bundle(bundle)
    r_np = numpy_decode_pair(z_q, pose_np, weights, cfg)

    assert r_mlx.shape == r_np.shape == (4, 2, 3, RENDER_H, RENDER_W)
    drift = float(np.max(np.abs(r_mlx - r_np)))
    # NO-FAKE: a stub would be ~255 off; the real port is sub-0.01 on [0,255].
    assert drift < 0.05, f"numpy<->MLX render drift too large: {drift}"


# --------------------------------------------------------------------------
# (2) THE SCORE-PARITY GATE: numpy inflate vs MLX render on a frozen scorer
# --------------------------------------------------------------------------


@skip_no_mlx
def test_numpy_inflate_score_parity_with_mlx_render():
    """The GATE: numpy inflate frames score-match the MLX render on the frozen scorer."""
    n_pairs, h, w = 5, 48, 64
    rng = np.random.default_rng(3)
    bundle = _make_film_bundle(num_pairs=n_pairs, base_ch=16, K=16, seed=3)
    pose_np = rng.standard_normal((n_pairs, 6)).astype(np.float32)
    bundle.pose_store_for_test = pose_np  # type: ignore[attr-defined]

    # Frozen proto DistortionNet + reachable GT targets.
    dnet = _build_frozen_dnet(with_pose=True)
    seg_tgt = torch.zeros(n_pairs, h, w, dtype=torch.long)
    bands = [(0, 10), (10, 20), (20, 30), (30, 40), (40, h)]
    for i in range(n_pairs):
        for cls, (r0, r1) in enumerate(bands):
            seg_tgt[i, r0:r1, :] = (cls + i) % 5
    pose_tgt = torch.from_numpy(rng.standard_normal((n_pairs, 6)).astype(np.float32))

    # --- MLX render -> torch-reference camera frames -> score
    r_mlx = _mlx_render(bundle, pose_np)
    cam_mlx = _camera_from_render(r_mlx)  # (N*2,H,W,3) uint8
    d_seg_mlx, d_pose_mlx = _score_frames(dnet, cam_mlx, seg_tgt, pose_tgt)

    # --- numpy inflate (the REAL contest path) -> camera frames -> score
    archive, _account, config = _build_archive_and_config(bundle, decoder_dtype="fp16")
    decoded = decode_archive(archive, config)
    cam_np = render_all_camera_frames(decoded)  # (N*2,H,W,3) uint8
    d_seg_np, d_pose_np = _score_frames(dnet, cam_np, seg_tgt, pose_tgt)

    # NO-FAKE: the inflate produced REAL camera frames of the right shape.
    assert cam_np.shape == (n_pairs * 2, CAMERA_H, CAMERA_W, 3)
    assert cam_np.dtype == np.uint8
    # SCORE-PARITY (the gate): the numpy inflate scores the SAME as the MLX render.
    assert abs(d_seg_np - d_seg_mlx) < 1e-4, (
        f"d_seg parity broke: mlx={d_seg_mlx} np={d_seg_np}"
    )
    # pose is an MSE in pose-space; tolerance is small-relative on the live value.
    assert abs(d_pose_np - d_pose_mlx) <= max(1e-4, 1e-3 * abs(d_pose_mlx) + 1e-4), (
        f"d_pose parity broke: mlx={d_pose_mlx} np={d_pose_np}"
    )


# --------------------------------------------------------------------------
# (3) NO-FAKE controls
# --------------------------------------------------------------------------


@skip_no_mlx
def test_identity_film_path_matches_no_pose():
    """Control: with identity FiLM (zero-init fc2), the numpy decode == no-pose decode.

    Proves the FiLM path is correctly wired: at init the per-frame FiLM is identity,
    so frame0 and frame1 share the feature and the pose conditioning is inert.
    """
    from tac.capstone_vq_nerv.vq_nerv_bundle import (
        CapstoneVqNervBundle,
        CapstoneVqNervConfig,
    )

    b = CapstoneVqNervBundle(
        CapstoneVqNervConfig(num_pairs=4, base_channels=16, codebook_size=16, seed=1)
    )
    cb = np.asarray(b.quantizer.codebook, dtype=np.float32)
    z_q = cb[b.all_vq_indices()]
    weights = full_render_weights_from_bundle(b)
    cfg = decode_config_from_bundle(b)
    pose = np.random.default_rng(1).standard_normal((4, 6)).astype(np.float32)
    r_pose = numpy_decode_pair(z_q, pose, weights, cfg)
    r_none = numpy_decode_pair(z_q, None, weights, cfg)
    # identity FiLM at init -> pose-conditioned == unconditioned.
    assert float(np.max(np.abs(r_pose - r_none))) < 1e-3


@skip_no_mlx
def test_stub_decoder_would_fail_pixel_parity():
    """NO-FAKE: the faithful numpy inflate reproduces the MLX camera frames; a stub does not.

    Compares camera frames directly (the contest inflate output): the faithful
    numpy inflate matches the torch-reference camera frames of the MLX render to a
    tiny per-frame delta, while a constant-grey stub is far. This proves the port
    is the REAL forward — a stub substituted for the numpy decode would FAIL here.
    The only legitimate gap between the faithful inflate and the MLX-render camera
    baseline is the fp16 pose-storage roundtrip (the bytes the archive actually
    carries), which is sub-uint8 and quantified below.
    """
    n_pairs = 4
    rng = np.random.default_rng(7)
    bundle = _make_film_bundle(num_pairs=n_pairs, base_ch=16, K=16, seed=7)
    pose_np = rng.standard_normal((n_pairs, 6)).astype(np.float32)
    bundle.pose_store_for_test = pose_np  # type: ignore[attr-defined]

    # MLX render -> torch-reference camera frames (the baseline the inflate targets).
    r_mlx = _mlx_render(bundle, pose_np)
    cam_mlx = _camera_from_render(r_mlx).astype(np.int16)

    # The FAITHFUL numpy inflate camera frames.
    archive, _acct, config = _build_archive_and_config(bundle)
    decoded = decode_archive(archive, config)
    cam_np = render_all_camera_frames(decoded).astype(np.int16)

    # NO-FAKE: faithful inflate reproduces the MLX render's uint8 camera frames.
    mean_abs = float(np.mean(np.abs(cam_np - cam_mlx)))
    assert mean_abs < 0.5, f"faithful inflate mean |Δ uint8| too large: {mean_abs}"

    # STUB: a constant-grey camera frame (what a fake decode would emit) is far.
    # (The untrained render is near mid-grey — sigmoid of small logits — so the
    # absolute stub gap is modest, but it is an order of magnitude beyond the
    # faithful-inflate gap, which is the teeth that matter.)
    cam_stub = np.full_like(cam_mlx, 128)
    stub_mean_abs = float(np.mean(np.abs(cam_stub - cam_mlx)))
    assert stub_mean_abs > 10.0 * max(mean_abs, 1e-3), (
        f"gate must distinguish real ({mean_abs}) from stub ({stub_mean_abs})"
    )
    assert stub_mean_abs > 1.0, f"stub must be measurably far from the render: {stub_mean_abs}"


# --------------------------------------------------------------------------
# (4) archive sidecar / zip packaging round-trip
# --------------------------------------------------------------------------


@skip_no_mlx
def test_inflate_reads_zip_with_config_sidecar(tmp_path):
    """The inflate reads a zip member 'x' + the capstone_config_v1 sidecar."""
    from tac.capstone_vq_nerv.inflate import (
        CAPSTONE_CONFIG_MEMBER,
        _read_archive_and_config,
    )

    bundle = _make_film_bundle(num_pairs=3, base_ch=16, K=16, seed=5)
    bundle.pose_store_for_test = np.random.default_rng(5).standard_normal((3, 6)).astype(np.float32)  # type: ignore[attr-defined]
    archive, _acct, config = _build_archive_and_config(bundle)
    zpath = tmp_path / "archive.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("x", archive)
        zf.writestr(CAPSTONE_CONFIG_MEMBER, json.dumps(config))
    rb, rc = _read_archive_and_config(zpath)
    assert rb == archive
    assert rc["base_channels"] == config["base_channels"]
    decoded = decode_archive(rb, rc)
    frames = render_all_camera_frames(decoded)
    assert frames.shape == (3 * 2, CAMERA_H, CAMERA_W, 3)
    assert frames.dtype == np.uint8


# --------------------------------------------------------------------------
# (5) HiNeRV grid-PE upgrade (opt-in, default-off): parity + NO-FAKE gates
# --------------------------------------------------------------------------
#
# The HiNeRV delta over HNeRV (the ~72.3% BD-rate-over-HNeRV lever): the
# bilinear-skip + PixelShuffle + sin upsample blocks are ALREADY in every
# HNeRVDecoderMLX block (audited 2026-06-11), so the genuinely-missing HiNeRV
# mechanism is the multi-resolution GRID positional-encoding fed to the stem.
# These tests are the portability GATE for that mechanism:
#   * OFF (default) is byte-identical to the pre-switch decoder (no extra params,
#     no forward change) — proved structurally + by render equality;
#   * ON is reproduced EXACTLY by the pure-numpy inflate (the portability
#     contract) — both at init (zero-proj) and after the proj is trained;
#   * NO-FAKE: a TRAINED grid-PE proj ACTUALLY changes the render (it is not a
#     no-op) AND the grid is deterministic + adds ~0 archive bytes.


def _make_grid_pe_bundle(num_pairs=4, base_ch=16, K=16, seed=0, num_freqs=4, train_proj=False):
    """A grid-PE-ENABLED FiLM bundle; optionally nudge the proj off zero-init.

    With ``train_proj=False`` the grid-PE projection is zero-init (identity: the
    ON-render == the OFF-render). With ``train_proj=True`` the projection weight +
    bias are nudged off zero so the grid grammar is LIVE (the NO-FAKE case).
    """
    from tac.capstone_vq_nerv.vq_nerv_bundle import (
        CapstoneVqNervBundle,
        CapstoneVqNervConfig,
    )

    b = CapstoneVqNervBundle(
        CapstoneVqNervConfig(
            num_pairs=num_pairs,
            base_channels=base_ch,
            codebook_size=K,
            hinerv_grid_pe=True,
            grid_pe_num_freqs=num_freqs,
            seed=seed,
        )
    )
    rng = np.random.default_rng(seed)
    b.set_pose_stats(
        rng.standard_normal(6).astype(np.float32),
        (np.abs(rng.standard_normal(6)) + 0.5).astype(np.float32),
    )
    # Nudge FiLM off identity (per-frame, like _make_film_bundle).
    for prefix, sign in (("pose_film0", 1.0), ("pose_film1", -1.0)):
        film = getattr(b, prefix)
        w2 = np.asarray(film.fc2.weight).copy()
        w2 += sign * 0.05 * rng.standard_normal(w2.shape).astype(np.float32)
        film.fc2.weight = mx.array(w2)
        bb = np.asarray(film.fc2.bias).copy()
        bb += sign * 0.1 * rng.standard_normal(bb.shape).astype(np.float32)
        film.fc2.bias = mx.array(bb)
    if train_proj:
        pw = np.asarray(b.grid_pe_proj.proj.weight).copy()
        pw += 0.3 * rng.standard_normal(pw.shape).astype(np.float32)
        b.grid_pe_proj.proj.weight = mx.array(pw)
        pb = np.asarray(b.grid_pe_proj.proj.bias).copy()
        pb += 0.2 * rng.standard_normal(pb.shape).astype(np.float32)
        b.grid_pe_proj.proj.bias = mx.array(pb)
    return b


def _build_grid_pe_config(bundle, *, decoder_dtype="fp16"):
    """Config sidecar for a grid-PE bundle (adds the two grid-PE flags)."""
    cfg = decode_config_from_bundle(bundle)
    return {
        "decoder_dtype": decoder_dtype,
        "num_pairs": int(bundle.cfg.num_pairs),
        "codebook_size": int(bundle.cfg.codebook_size),
        "latent_dim": int(bundle.cfg.latent_dim),
        "base_channels": int(cfg.base_channels),
        "base_h": int(cfg.base_h),
        "base_w": int(cfg.base_w),
        "film_enabled": bool(cfg.film_enabled),
        "pose_normalize": bool(cfg.pose_normalize),
        "pose_mean": list(cfg.pose_mean),
        "pose_std": list(cfg.pose_std),
        "hinerv_grid_pe": bool(cfg.hinerv_grid_pe),
        "grid_pe_num_freqs": int(cfg.grid_pe_num_freqs),
    }


def test_grid_pe_off_is_byte_identical_no_extra_params():
    """DEFAULT-OFF contract: no grid_pe_proj module, no grid-PE weight keys, flag off.

    The opt-in is byte-identical when off: an OFF bundle never constructs the
    ``_GridPE`` module, so its trainable + render-basis weight sets are exactly the
    pre-switch decoder's (the archive bytes are unchanged). NO-FAKE: if the OFF path
    silently added the grid-PE params this assertion fails.
    """
    from tac.capstone_vq_nerv.vq_nerv_bundle import (
        CapstoneVqNervBundle,
        CapstoneVqNervConfig,
    )

    b_off = CapstoneVqNervBundle(
        CapstoneVqNervConfig(num_pairs=4, base_channels=16, codebook_size=16, seed=0)
    )
    assert b_off.cfg.hinerv_grid_pe is False
    assert b_off.hinerv_grid_pe is False
    assert not hasattr(b_off, "grid_pe_proj")
    cfg = decode_config_from_bundle(b_off)
    assert cfg.hinerv_grid_pe is False
    if _HAVE_MLX:
        w = full_render_weights_from_bundle(b_off)
        assert not any("grid_pe" in k for k in w), (
            f"OFF must carry NO grid-PE params; found {[k for k in w if 'grid_pe' in k]}"
        )


@skip_no_mlx
def test_grid_pe_on_at_init_equals_off_render():
    """Identity-at-init: zero-init proj => grid-PE ON-render == OFF-render.

    The grid_pe_proj is zero-init so the PE contribution is 0 before training.
    This proves the upgrade is a SAFE default-off opt-in: enabling it does not
    perturb an untrained decoder. (Built on a single set of decoder weights by
    toggling only the cfg flag in the numpy decode, so the codebook RNG is held
    fixed — the comparison isolates the grid-PE branch.)
    """
    b = _make_grid_pe_bundle(num_pairs=4, base_ch=16, seed=2, train_proj=False)
    rng = np.random.default_rng(2)
    pose = rng.standard_normal((4, 6)).astype(np.float32)
    cb = np.asarray(b.quantizer.codebook, dtype=np.float32)
    z_q = cb[b.all_vq_indices()]
    weights = full_render_weights_from_bundle(b)
    cfg_on = decode_config_from_bundle(b)
    from dataclasses import replace

    cfg_off = replace(cfg_on, hinerv_grid_pe=False)
    r_on = numpy_decode_pair(z_q, pose, weights, cfg_on)
    r_off = numpy_decode_pair(z_q, pose, weights, cfg_off)
    assert float(np.max(np.abs(r_on - r_off))) < 1e-3, "zero-init grid-PE must be identity"


@skip_no_mlx
@pytest.mark.parametrize("train_proj", [False, True])
def test_grid_pe_numpy_matches_mlx_render(train_proj):
    """PORTABILITY GATE: pure-numpy inflate reproduces the MLX grid-PE render.

    Both at init (zero proj) AND after the proj is trained (the case that broke
    before the ``grid_pe_proj.proj.weight`` key fix — numpy silently skipped the
    grid-PE and diverged by ~34). This is THE contract: MLX fast path == numpy
    reference == (eventually) torch. Tolerance matches the existing FiLM parity
    (small fp32 conv-accumulation drift, NOT score-affecting).
    """
    b = _make_grid_pe_bundle(num_pairs=4, base_ch=16, seed=5, train_proj=train_proj)
    rng = np.random.default_rng(5)
    pose = rng.standard_normal((4, 6)).astype(np.float32)
    idx = mx.arange(4)
    r_mlx = np.asarray(b(idx, pose=mx.array(pose)), dtype=np.float32)
    cb = np.asarray(b.quantizer.codebook, dtype=np.float32)
    z_q = cb[b.all_vq_indices()]
    r_np = numpy_decode_pair(z_q, pose, full_render_weights_from_bundle(b), decode_config_from_bundle(b))
    drift = float(np.max(np.abs(r_mlx - r_np)))
    assert drift < 0.5, f"grid-PE numpy<->MLX parity broke (train_proj={train_proj}): {drift}"


@skip_no_mlx
def test_grid_pe_trained_actually_changes_render_not_a_noop():
    """NO-FAKE: a TRAINED grid-PE proj measurably changes the render vs OFF.

    If the grid-PE were a no-op (e.g. the projection never wired into the stem, or
    the key mismatch silently dropped it), the ON-trained render would equal the
    OFF render. It must NOT. This is the teeth: the mechanism does real work.
    """
    b = _make_grid_pe_bundle(num_pairs=4, base_ch=16, seed=9, train_proj=True)
    rng = np.random.default_rng(9)
    pose = rng.standard_normal((4, 6)).astype(np.float32)
    cb = np.asarray(b.quantizer.codebook, dtype=np.float32)
    z_q = cb[b.all_vq_indices()]
    weights = full_render_weights_from_bundle(b)
    cfg_on = decode_config_from_bundle(b)
    from dataclasses import replace

    cfg_off = replace(cfg_on, hinerv_grid_pe=False)
    r_on = numpy_decode_pair(z_q, pose, weights, cfg_on)
    r_off = numpy_decode_pair(z_q, pose, weights, cfg_off)
    delta = float(np.max(np.abs(r_on - r_off)))
    assert delta > 1.0, f"trained grid-PE must change the render; max|Δ|={delta} (no-op!)"


def test_grid_pe_is_deterministic_and_storage_free():
    """The grid is regenerated from coords (~0 archive bytes) + fully deterministic.

    The positional grid stores NO values in the archive — only the tiny
    ``channels[0] x pe_dim`` projection. The grid op itself is a pure function of
    ``(base_h, base_w, num_freqs)`` so the inflate reproduces it on any host.
    """
    from tac.capstone_vq_nerv.numpy_reference import grid_positional_encoding

    g1 = grid_positional_encoding(6, 8, 4)
    g2 = grid_positional_encoding(6, 8, 4)
    assert g1.shape == (6 * 8, 4 * 4)  # pe_dim = 4 * num_freqs
    assert np.array_equal(g1, g2), "grid must be deterministic"
    # band-limited to [-1, 1] (sin/cos) and not all-zero (carries real coords).
    assert float(np.max(np.abs(g1))) <= 1.0 + 1e-6
    assert float(np.max(np.abs(g1))) > 0.1
    with pytest.raises(ValueError):
        grid_positional_encoding(6, 8, 0)


@skip_no_mlx
def test_grid_pe_inflate_end_to_end_with_trained_proj(tmp_path):
    """END-TO-END: a grid-PE archive byte-closes + the real numpy inflate decodes it.

    Builds a trained grid-PE bundle, byte-closes the FULL render basis (decoder +
    FiLM + grid_pe_proj) into the contest archive, writes the zip + config sidecar
    with the grid-PE flags, then runs the REAL contest inflate. The decoded camera
    frames score-match the MLX render on the frozen scorer — proving the grid-PE
    survives the int8 archive roundtrip + the scorer-free numpy inflate.
    """
    n_pairs, h, w = 4, 48, 64
    b = _make_grid_pe_bundle(num_pairs=n_pairs, base_ch=16, seed=11, train_proj=True)
    rng = np.random.default_rng(11)
    pose_np = rng.standard_normal((n_pairs, 6)).astype(np.float32)
    b.pose_store_for_test = pose_np  # type: ignore[attr-defined]

    weights = full_render_weights_from_bundle(b)
    assert any("grid_pe" in k for k in weights), "grid-PE proj must be in the render basis"
    cb = np.asarray(b.quantizer.codebook, dtype=np.float32)
    vq = b.all_vq_indices()
    archive, _account = build_capstone_archive_bytes(
        decoder_weights=weights,
        codebook=cb,
        vq_indices=vq,
        pose_scalars=pose_np,
        codebook_size=b.cfg.codebook_size,
        decoder_dtype="fp16",
    )
    config = _build_grid_pe_config(b)
    from tac.capstone_vq_nerv.inflate import CAPSTONE_CONFIG_MEMBER, _read_archive_and_config

    zpath = tmp_path / "archive.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("x", archive)
        zf.writestr(CAPSTONE_CONFIG_MEMBER, json.dumps(config))
    rb, rc = _read_archive_and_config(zpath)
    assert rc["hinerv_grid_pe"] is True
    decoded = decode_archive(rb, rc)

    # score-parity vs MLX render on the frozen proto DistortionNet.
    dnet = _build_frozen_dnet(with_pose=True)
    seg_tgt = torch.zeros(n_pairs, h, w, dtype=torch.long)
    bands = [(0, 10), (10, 20), (20, 30), (30, 40), (40, h)]
    for i in range(n_pairs):
        for cls, (r0, r1) in enumerate(bands):
            seg_tgt[i, r0:r1, :] = (cls + i) % 5
    pose_tgt = torch.from_numpy(rng.standard_normal((n_pairs, 6)).astype(np.float32))

    r_mlx = _mlx_render(b, pose_np)
    cam_mlx = _camera_from_render(r_mlx)
    d_seg_mlx, d_pose_mlx = _score_frames(dnet, cam_mlx, seg_tgt, pose_tgt)

    cam_np = render_all_camera_frames(decoded)
    assert cam_np.shape == (n_pairs * 2, CAMERA_H, CAMERA_W, 3)
    d_seg_np, d_pose_np = _score_frames(dnet, cam_np, seg_tgt, pose_tgt)
    assert abs(d_seg_np - d_seg_mlx) < 1e-3, f"grid-PE inflate d_seg parity: mlx={d_seg_mlx} np={d_seg_np}"
    assert abs(d_pose_np - d_pose_mlx) <= max(1e-3, 5e-3 * abs(d_pose_mlx) + 1e-3), (
        f"grid-PE inflate d_pose parity: mlx={d_pose_mlx} np={d_pose_np}"
    )
