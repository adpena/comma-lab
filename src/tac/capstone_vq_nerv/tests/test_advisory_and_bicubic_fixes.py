# SPDX-License-Identifier: MIT
"""Behavioral tests for [A2] reloaded-int8 advisory + [A3] bicubic camera resize.

[A2] ``score_reloaded_int8_archive`` decodes the REAL byte-closed int8 archive
through the contest inflate path and re-scores it through the frozen scorer bridge
— the honest ``inflate.sh -> evaluate.py`` predictor (the live fp32 advisory hides
the int8 quant loss). REVERT-CATCH: the reloaded score reflects the int8-quantized
weights, so it tracks the contest decode, not the live render.

[A3] the inflate camera upscale is BICUBIC (PR95 ``score.py::_decoded_to_camera``),
matching PyTorch ``F.interpolate(mode='bicubic', align_corners=False)`` to fp32 eps.
REVERT-CATCH: a bilinear camera resize diverges from the torch bicubic reference by
more than the fp32-accumulation tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import mlx.core as mx

    _HAVE_MLX = True
except Exception:  # pragma: no cover
    mx = None  # type: ignore[assignment]
    _HAVE_MLX = False

skip_no_mlx = pytest.mark.skipif(not _HAVE_MLX, reason="mlx not available")

from tac.capstone_vq_nerv.numpy_reference import (  # noqa: E402
    bicubic_resize_to_nhwc,
    bilinear_resize_to_nhwc,
)

CAMERA_H, CAMERA_W = 874, 1164
RENDER_H, RENDER_W = 384, 512


# ===========================================================================
# [A3] BICUBIC camera resize matches PyTorch
# ===========================================================================


def test_a3_bicubic_matches_torch_interpolate_small():
    """``bicubic_resize_to_nhwc`` matches ``F.interpolate(mode='bicubic')`` to fp32 eps."""
    rng = np.random.default_rng(0)
    for H, W, TH, TW in [(8, 10, 17, 21), (12, 16, 24, 32), (6, 8, 13, 17)]:
        x = (rng.standard_normal((2, H, W, 3)) * 50 + 128).astype(np.float32)
        xt = torch.from_numpy(np.transpose(x, (0, 3, 1, 2)).copy())
        ref = np.transpose(
            F.interpolate(xt, size=(TH, TW), mode="bicubic", align_corners=False).numpy(),
            (0, 2, 3, 1),
        )
        out = bicubic_resize_to_nhwc(x, TH, TW)
        err = float(np.max(np.abs(out - ref)))
        assert out.shape == ref.shape
        assert err < 1e-3, f"bicubic drift {err} at {H}x{W}->{TH}x{TW}"


def test_a3_bicubic_matches_torch_at_camera_scale_uint8():
    """At the 384x512->camera scale, bicubic matches torch within 1 LSB after uint8."""
    rng = np.random.default_rng(1)
    x = (rng.standard_normal((1, RENDER_H, RENDER_W, 3)) * 40 + 128).astype(np.float32)
    xt = torch.from_numpy(np.transpose(x, (0, 3, 1, 2)).copy())
    ref = np.transpose(
        F.interpolate(
            xt, size=(CAMERA_H, CAMERA_W), mode="bicubic", align_corners=False
        ).numpy(),
        (0, 2, 3, 1),
    )
    out = bicubic_resize_to_nhwc(x, CAMERA_H, CAMERA_W)
    ref_u8 = np.clip(np.round(ref), 0, 255).astype(np.uint8)
    out_u8 = np.clip(np.round(out), 0, 255).astype(np.uint8)
    # within 1 LSB (the fp32 accumulation-order delta) and < 0.5% of pixels off-by-one.
    assert int(np.max(np.abs(out_u8.astype(int) - ref_u8.astype(int)))) <= 1
    assert float(np.mean(out_u8 != ref_u8)) < 0.01


def test_a3_bicubic_differs_from_bilinear():
    """Bicubic is NOT bilinear (REVERT-CATCH: the inflate must use the bicubic path)."""
    rng = np.random.default_rng(2)
    x = (rng.standard_normal((1, 32, 40, 3)) * 50 + 128).astype(np.float32)
    bc = bicubic_resize_to_nhwc(x, 70, 90)
    bl = bilinear_resize_to_nhwc(x, 70, 90)
    # bicubic vs bilinear differ materially on a textured input.
    assert float(np.max(np.abs(bc - bl))) > 1.0


def test_a3_inflate_camera_upscale_uses_bicubic():
    """The inflate runtime's camera upscale equals the bicubic reference (not bilinear).

    Renders a tiny capstone, runs ``render_all_camera_frames`` (the contest path),
    and compares its first camera frame against a torch BICUBIC reference of the
    same native render. A bilinear inflate would differ by > 1 LSB on many pixels.
    """
    if not _HAVE_MLX:
        pytest.skip("mlx not available")
    from tac.capstone_vq_nerv.inflate import decode_archive, render_all_camera_frames
    from tac.capstone_vq_nerv.numpy_reference import (
        decode_config_from_bundle,
        full_render_weights_from_bundle,
        numpy_decode_pair,
    )
    from tac.capstone_vq_nerv.vq_nerv_bundle import (
        CapstoneVqNervBundle,
        CapstoneVqNervConfig,
    )

    n_pairs = 2
    bundle = CapstoneVqNervBundle(
        CapstoneVqNervConfig(num_pairs=n_pairs, base_channels=16, codebook_size=16, seed=0)
    )
    pose = np.random.default_rng(0).standard_normal((n_pairs, 6)).astype(np.float32)
    bundle.set_pose_stats(pose.mean(0), pose.std(0))

    # build the archive + config (fp16 so the decode matches the native render exactly).
    import dataclasses

    from tac.capstone_vq_nerv.export import build_capstone_archive_bytes

    weights = full_render_weights_from_bundle(bundle)
    codebook = np.asarray(bundle.quantizer._codebook, dtype=np.float32)
    vq = bundle.all_vq_indices()
    payload, _account = build_capstone_archive_bytes(
        decoder_weights=weights, codebook=codebook, vq_indices=vq,
        pose_scalars=pose, codebook_size=int(codebook.shape[0]), decoder_dtype="fp16",
    )
    config = dataclasses.asdict(decode_config_from_bundle(bundle))
    config["num_pairs"] = n_pairs
    config["decoder_dtype"] = "fp16"

    decoded = decode_archive(payload, config)
    cam = render_all_camera_frames(decoded)  # (N*2, 874, 1164, 3) uint8

    # torch bicubic reference of the native numpy render for frame 0 (pair0, k=0).
    z_q = codebook[vq[:1]]
    native = numpy_decode_pair(z_q, pose[:1], weights, decoded["cfg"])  # (1,2,3,384,512)
    frame0 = native[0, 0]  # (3, 384, 512)
    xt = torch.from_numpy(frame0[None])  # (1,3,384,512)
    ref = F.interpolate(xt, size=(CAMERA_H, CAMERA_W), mode="bicubic", align_corners=False)
    ref_u8 = np.clip(np.round(ref[0].permute(1, 2, 0).numpy()), 0, 255).astype(np.uint8)

    # the inflate's bicubic camera frame matches the torch bicubic reference (<=1 LSB).
    diff = np.abs(cam[0].astype(int) - ref_u8.astype(int))
    assert int(diff.max()) <= 1, f"inflate camera upscale must be bicubic; max diff {diff.max()}"

    # REVERT-CATCH: a BILINEAR camera upscale would differ from the bicubic ref
    # on a meaningful fraction of pixels.
    frame0_nhwc = np.transpose(frame0, (1, 2, 0))[None]  # (1,384,512,3)
    bl = bilinear_resize_to_nhwc(frame0_nhwc, CAMERA_H, CAMERA_W)
    bl_u8 = np.clip(np.round(bl[0]), 0, 255).astype(np.uint8)
    bilinear_mismatch = float(np.mean(bl_u8 != ref_u8))
    assert bilinear_mismatch > 0.01, (
        "bilinear must differ from bicubic (else the test cannot detect a revert)"
    )


# ===========================================================================
# [A2] reloaded-int8 advisory scoring
# ===========================================================================


class _ColorProtoSeg(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        protos = torch.tensor(
            [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]],
            dtype=torch.float32,
        )
        self.c = nn.Conv2d(3, 5, 1)
        self.c.weight.data = protos.reshape(5, 3, 1, 1) / 128.0
        self.c.bias.data = -(protos**2).sum(1) / (2 * 128.0 * 128.0)

    def forward(self, x):
        return self.c(x * 255.0)


class _FrozenDNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.segnet = _ColorProtoSeg()
        self.posenet = None

    def preprocess_input(self, bhwc):
        last = bhwc[:, -1].permute(0, 3, 1, 2)
        return None, last / 255.0


def _frozen_dnet():
    dnet = _FrozenDNet().eval()
    for p in dnet.parameters():
        p.requires_grad = False
    return dnet


@skip_no_mlx
def test_a2_score_reloaded_int8_archive_runs_and_reflects_quantized_weights():
    """[A2] reloaded-int8 advisory scores the REAL int8 archive through the bridge.

    REVERT-CATCH: the reloaded-int8 d_seg is measured on the int8-quantized decode
    (the contest bytes), so it generally DIFFERS from the live fp32 render d_seg —
    the whole point of [A2] is that this gap is reported, not hidden.
    """
    from tac.capstone_vq_nerv.advisory import (
        ReloadedInt8Advisory,
        score_reloaded_int8_archive,
    )
    from tac.capstone_vq_nerv.capstone_trainer import (
        CapstoneTrainConfig,
        CapstoneTrainer,
    )
    from tac.capstone_vq_nerv.vq_nerv_bundle import (
        CapstoneVqNervBundle,
        CapstoneVqNervConfig,
    )
    from tac.mlx_pr95_port.score_bridge import TorchScorerBridge

    n_pairs, h, w = 4, 48, 64
    dnet = _frozen_dnet()
    seg_tgt = torch.zeros(n_pairs, h, w, dtype=torch.long)
    for i in range(n_pairs):
        for cls, (r0, r1) in enumerate([(0, 10), (10, 20), (20, 30), (30, 40), (40, h)]):
            seg_tgt[i, r0:r1, :] = (cls + i) % 5
    bridge = TorchScorerBridge(
        dnet, seg_tgt, None, seg_loss_form="ce_seg_loss",
        eval_roundtrip=True, scorer_hw=(h, w),
    )
    pose_store = np.zeros((n_pairs, 6), dtype=np.float32)
    bundle = CapstoneVqNervBundle(
        CapstoneVqNervConfig(num_pairs=n_pairs, base_channels=16, codebook_size=16, seed=0)
    )
    cfg = CapstoneTrainConfig(
        epochs=4, batch_size=4, eval_every=4, seed=0, muon_lr=3e-2, adamw_lr=2e-2,
        grad_clip=50.0, grad_clip_muon=50.0, cosine_lr_schedule=False,
        use_ema_for_eval=True,
    )
    trainer = CapstoneTrainer(bundle, bridge, pose_store, cfg)
    trainer.train()

    # build the REAL int8 archive payload + config (the EMA-shadow render basis).
    import dataclasses

    from tac.capstone_vq_nerv.export import build_capstone_archive_bytes
    from tac.capstone_vq_nerv.numpy_reference import decode_config_from_bundle

    weights = trainer.export_render_weights()
    codebook = np.asarray(bundle.quantizer._codebook, dtype=np.float32)
    vq = np.asarray(bundle.all_vq_indices(), dtype=np.int32)
    payload, _acct = build_capstone_archive_bytes(
        decoder_weights=weights, codebook=codebook, vq_indices=vq,
        pose_scalars=pose_store, codebook_size=int(codebook.shape[0]),
        decoder_dtype="int8",
    )
    config = dataclasses.asdict(decode_config_from_bundle(bundle))
    config["num_pairs"] = n_pairs
    config["decoder_dtype"] = "int8"

    reloaded = score_reloaded_int8_archive(payload, config, bridge)
    assert isinstance(reloaded, ReloadedInt8Advisory)
    assert reloaded.num_pairs == n_pairs
    # the reloaded-int8 d_seg is a real [0,1] argmax-disagreement rate.
    assert 0.0 <= reloaded.d_seg <= 1.0
    assert reloaded.d_pose == 0.0  # pose disabled in this proto
    d = reloaded.as_dict()
    assert d["reloaded_int8_d_seg"] == reloaded.d_seg
    assert d["reloaded_int8_num_pairs"] == n_pairs


@skip_no_mlx
def test_a2_reloaded_int8_tracks_int8_decode_not_live_fp32():
    """[A2] the reloaded-int8 advisory matches a direct int8-decode score, NOT live fp32.

    Builds an int8 archive, decodes it (the contest path), scores the decoded int8
    frames directly, and asserts ``score_reloaded_int8_archive`` returns THAT —
    i.e. the int8-quantized decode, the honest contest predictor (REVERT-CATCH: if
    it secretly scored the live fp32 render it would diverge from this direct check).
    """
    import dataclasses

    from tac.capstone_vq_nerv.advisory import score_reloaded_int8_archive
    from tac.capstone_vq_nerv.export import build_capstone_archive_bytes
    from tac.capstone_vq_nerv.inflate import decode_archive
    from tac.capstone_vq_nerv.numpy_reference import (
        decode_config_from_bundle,
        full_render_weights_from_bundle,
        numpy_decode_pair,
    )
    from tac.capstone_vq_nerv.vq_nerv_bundle import (
        CapstoneVqNervBundle,
        CapstoneVqNervConfig,
    )
    from tac.mlx_pr95_port.score_bridge import TorchScorerBridge

    n_pairs, h, w = 4, 48, 64
    dnet = _frozen_dnet()
    seg_tgt = torch.zeros(n_pairs, h, w, dtype=torch.long)
    for i in range(n_pairs):
        for cls, (r0, r1) in enumerate([(0, 10), (10, 20), (20, 30), (30, 40), (40, h)]):
            seg_tgt[i, r0:r1, :] = (cls + i) % 5
    bridge = TorchScorerBridge(
        dnet, seg_tgt, None, seg_loss_form="ce_seg_loss",
        eval_roundtrip=True, scorer_hw=(h, w),
    )
    pose_store = np.zeros((n_pairs, 6), dtype=np.float32)
    bundle = CapstoneVqNervBundle(
        CapstoneVqNervConfig(num_pairs=n_pairs, base_channels=16, codebook_size=16, seed=0)
    )
    bundle.set_pose_stats(pose_store.mean(0), pose_store.std(0))

    weights = full_render_weights_from_bundle(bundle)
    codebook = np.asarray(bundle.quantizer._codebook, dtype=np.float32)
    vq = np.asarray(bundle.all_vq_indices(), dtype=np.int32)
    payload, _acct = build_capstone_archive_bytes(
        decoder_weights=weights, codebook=codebook, vq_indices=vq,
        pose_scalars=pose_store, codebook_size=int(codebook.shape[0]),
        decoder_dtype="int8",
    )
    config = dataclasses.asdict(decode_config_from_bundle(bundle))
    config["num_pairs"] = n_pairs
    config["decoder_dtype"] = "int8"

    # direct int8 decode + score (the ground truth the advisory must reproduce).
    decoded = decode_archive(payload, config)
    direct_d_seg = 0.0
    nseen = 0
    for s in range(0, n_pairs, 8):
        e = min(s + 8, n_pairs)
        idx = np.arange(s, e)
        z_q = decoded["codebook"][decoded["vq_indices"][s:e]]
        render = numpy_decode_pair(z_q, decoded["pose"][s:e], decoded["weights"], decoded["cfg"])
        direct_d_seg += bridge.exact_d_seg(render, torch.from_numpy(idx.astype(np.int64))) * len(idx)
        nseen += len(idx)
    direct_d_seg /= nseen

    reloaded = score_reloaded_int8_archive(payload, config, bridge)
    assert reloaded.d_seg == pytest.approx(direct_d_seg, abs=1e-6), (
        "reloaded-int8 advisory must score the int8 decode (not the live fp32 render)"
    )
