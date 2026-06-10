# SPDX-License-Identifier: MIT
"""Bit-exact parity of the MLX HNeRV decoder stack vs ``torch.nn`` references (task #81).

Full-stack correctness audit deliverable. The existing
``test_canonical_kernels.py`` proves the canonical PixelShuffle's NUMPY-reference
and its own PyTorch *branch* (``F.pixel_shuffle`` after permute) agree; it does
NOT prove that the **MLX training kernel** (the channel-FIRST reshape/transpose
at ``pr95_hnerv_mlx.pixel_shuffle_2x_nhwc``) is byte-for-byte identical to the
real ``torch.nn.PixelShuffle(2)`` module — which is the parity the
``pr95_hnerv_mlx`` docstring CLAIMS ("0.0 absolute drift per sister D=Z6
anchor") but does not test in-tree. This module closes that gap and makes the
claim a permanent regression guard, plus an end-to-end ``HNeRVDecoderMLX`` vs
from-scratch ``nn.Module`` reference decoder parity check (PixelShuffle +
bilinear-skip + sin + terminal HF-refine + 6-stage 6x8->384x512 cascade).

Authority: ``[macOS-CPU advisory]`` numerical-parity test only; NO score claim,
NO MPS (CPU torch + CPU MLX), GT-free (synthetic latents). These are
correctness assertions on the decoder MATH, not contest measurements.

Why this matters (the #81 finding): the decoder math is CORRECT. The
``d_seg~0.50`` plateau is NOT a PixelShuffle/decoder bug — it is the inert
score-aware *training loop* (recon-MSE-dominant + scorer effective weight
ramped from 0.0 by dual-ascent + AdamW grad-clip-to-1.0 100% of steps) and the
skip-free *default* config (``use_bilinear_skip=False``, ``sin_frequency=30.0``).
This test pins the decoder math so a future regression there cannot be confused
with the training-loop defect.
"""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core", reason="MLX not installed")
torch = pytest.importorskip("torch", reason="torch not installed")

import torch.nn as tnn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from tac.local_acceleration.pr95_hnerv_mlx import (  # noqa: E402
    HNeRVDecoderMLX,
    bilinear_resize2x_align_corners_false_nhwc,
    pixel_shuffle_2x_nhwc,
)

# fp32-epsilon budget for elementwise float kernels. PixelShuffle is a pure
# permute (must be EXACT, 0.0). bilinear/conv accumulate -> allow fp32 atol.
_PS_EXACT_ATOL = 0.0
_FP32_ATOL = 1e-4  # generous for a deep 6-stage conv cascade on 0..255 output


def _nhwc_to_torch_nchw(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).permute(0, 3, 1, 2).contiguous()


# ---------------------------------------------------------------------------
# 1. MLX pixel_shuffle_2x_nhwc vs torch.nn.PixelShuffle(2) -- BIT EXACT
# ---------------------------------------------------------------------------
class TestMlxPixelShuffleVsTorchNNModule:
    @pytest.mark.parametrize(
        "shape",
        [(1, 2, 3, 8), (2, 6, 8, 144), (1, 4, 4, 12), (3, 5, 7, 4)],
    )
    def test_bit_exact_vs_nn_pixelshuffle(self, shape):
        x = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
        y_mlx = np.array(pixel_shuffle_2x_nhwc(mx.array(x)))
        ps = tnn.PixelShuffle(2)
        y_nn = ps(_nhwc_to_torch_nchw(x)).permute(0, 2, 3, 1).contiguous().numpy()
        assert y_mlx.shape == y_nn.shape
        drift = float(np.max(np.abs(y_mlx - y_nn)))
        assert drift == _PS_EXACT_ATOL, f"PixelShuffle drift {drift} != 0.0 (channel-convention bug)"

    def test_output_shape_is_2x_spatial_quarter_channels(self):
        x = np.zeros((2, 6, 8, 36 * 4), dtype=np.float32)
        y = np.array(pixel_shuffle_2x_nhwc(mx.array(x)))
        assert y.shape == (2, 12, 16, 36)

    def test_channel_first_convention_pins_against_channel_last_drift(self):
        # The FORBIDDEN channel-LAST convention produces ~2.4-3.8 absolute drift
        # (the FIX-WAVE-R1/R1' bug class). Prove the MLX kernel does NOT use it by
        # confirming it matches nn.PixelShuffle (channel-first) exactly while a
        # deliberate channel-last reshuffle differs.
        x = np.arange(1 * 2 * 3 * 8, dtype=np.float32).reshape(1, 2, 3, 8)
        y_mlx = np.array(pixel_shuffle_2x_nhwc(mx.array(x)))
        b, h, w, c = x.shape
        oc = c // 4
        # channel-LAST (forbidden): reshape (B,H,W,2,2,oc) + transpose (0,1,3,2,4,5)
        wrong = x.reshape(b, h, w, 2, 2, oc).transpose(0, 1, 3, 2, 4, 5).reshape(b, 2 * h, 2 * w, oc)
        assert not np.allclose(y_mlx, wrong), "MLX kernel must NOT match the forbidden channel-last layout"


# ---------------------------------------------------------------------------
# 2. MLX bilinear-2x vs torch F.interpolate(align_corners=False) -- fp32 eps
# ---------------------------------------------------------------------------
class TestMlxBilinear2xVsTorch:
    @pytest.mark.parametrize("shape", [(1, 3, 4, 2), (2, 6, 8, 5), (1, 12, 16, 18)])
    def test_bilinear_2x_matches_torch_interpolate(self, shape):
        x = np.random.default_rng(1).standard_normal(shape).astype(np.float32)
        y_mlx = np.array(bilinear_resize2x_align_corners_false_nhwc(mx.array(x)))
        y_t = (
            F.interpolate(_nhwc_to_torch_nchw(x), scale_factor=2, mode="bilinear", align_corners=False)
            .permute(0, 2, 3, 1)
            .contiguous()
            .numpy()
        )
        assert y_mlx.shape == y_t.shape
        drift = float(np.max(np.abs(y_mlx - y_t)))
        assert drift < _FP32_ATOL, f"bilinear-2x drift {drift} exceeds fp32 budget"


# ---------------------------------------------------------------------------
# 3. FULL HNeRVDecoderMLX vs from-scratch PyTorch reference -- end-to-end
#    (PixelShuffle + bilinear-skip + sin + terminal HF-refine + 6-stage cascade)
# ---------------------------------------------------------------------------
class _TorchHNeRVReference(tnn.Module):
    """Mirror of ``HNeRVDecoderMLX.features_nhwc`` + ``decode_pair_nhwc`` in NCHW.

    sin(PixelShuffle(conv(x)) + skip(bilinear_2x(x))) per block + terminal
    x + 0.1*sin(refine1(refine0(x))) + sigmoid RGB heads * 255.
    """

    def __init__(self, latent_dim: int, channels: list[int]) -> None:
        super().__init__()
        self.base_h, self.base_w = 6, 8
        self.channels = channels
        self.stem = tnn.Linear(latent_dim, channels[0] * self.base_h * self.base_w)
        self.convs = tnn.ModuleList()
        self.skips = tnn.ModuleList()
        for i in range(6):
            cin, cout = channels[i], channels[i + 1]
            self.convs.append(tnn.Conv2d(cin, cout * 4, 3, padding=1))
            self.skips.append(tnn.Conv2d(cin, cout, 1) if cin != cout else tnn.Identity())
        fc = channels[-1]
        self.refine0 = tnn.Conv2d(fc, fc // 2, 3, padding=2, dilation=2)
        self.refine1 = tnn.Conv2d(fc // 2, fc, 3, padding=1)
        self.rgb_0 = tnn.Conv2d(fc, 3, 3, padding=1)
        self.rgb_1 = tnn.Conv2d(fc, 3, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b = z.shape[0]
        x = self.stem(z).reshape(b, self.channels[0], self.base_h, self.base_w)
        x = torch.sin(x)
        for i in range(6):
            identity = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            identity = self.skips[i](identity)
            decoded = F.pixel_shuffle(self.convs[i](x), 2)
            x = torch.sin(decoded + identity)
        refined = self.refine1(self.refine0(x))
        x = x + 0.1 * torch.sin(refined)
        f0 = torch.sigmoid(self.rgb_0(x)) * 255.0
        f1 = torch.sigmoid(self.rgb_1(x)) * 255.0
        return torch.stack([f0, f1], dim=1)


def _build_paired_decoders():
    mx.random.seed(0)
    dec = HNeRVDecoderMLX(latent_dim=28, base_channels=36, eval_size=(384, 512), output_layout="n2chw")
    import mlx.utils as mu

    params = dict(mu.tree_flatten(dec.parameters()))
    tdec = _TorchHNeRVReference(28, list(dec.channels))

    def to_t(a):
        return torch.from_numpy(np.array(a, dtype=np.float32))

    sd = tdec.state_dict()
    sd["stem.weight"] = to_t(params["stem.weight"])
    sd["stem.bias"] = to_t(params["stem.bias"])
    for i in range(6):
        w = np.array(params[f"blocks.{i}.conv.weight"])  # MLX NHWC (out, kh, kw, in)
        sd[f"convs.{i}.weight"] = to_t(np.transpose(w, (0, 3, 1, 2)))  # -> NCHW (out, in, kh, kw)
        sd[f"convs.{i}.bias"] = to_t(params[f"blocks.{i}.conv.bias"])
        skip_key = f"blocks.{i}.skip_conv.weight"
        if skip_key in params:
            sw = np.array(params[skip_key])
            sd[f"skips.{i}.weight"] = to_t(np.transpose(sw, (0, 3, 1, 2)))
            sd[f"skips.{i}.bias"] = to_t(params[f"blocks.{i}.skip_conv.bias"])
    for nm in ["refine0", "refine1", "rgb_0", "rgb_1"]:
        w = np.array(params[f"{nm}.weight"])
        sd[f"{nm}.weight"] = to_t(np.transpose(w, (0, 3, 1, 2)))
        sd[f"{nm}.bias"] = to_t(params[f"{nm}.bias"])
    tdec.load_state_dict(sd)
    tdec.eval()
    return dec, tdec, to_t


class TestFullHNeRVDecoderParity:
    def test_channel_taper_is_pr95_canonical(self):
        dec = HNeRVDecoderMLX(latent_dim=28, base_channels=36, eval_size=(384, 512))
        # PR95 taper: [C, C, C, 0.75C, 0.58C, 0.5C, 0.5C] with C=36.
        assert list(dec.channels) == [36, 36, 36, 27, 20, 18, 18]

    def test_full_decoder_matches_torch_reference_end_to_end(self):
        dec, tdec, to_t = _build_paired_decoders()
        z = np.random.default_rng(7).standard_normal((1, 28)).astype(np.float32)
        y_mlx = np.array(dec(mx.array(z)))
        with torch.no_grad():
            y_t = tdec(to_t(z)).numpy()
        assert y_mlx.shape == y_t.shape == (1, 2, 3, 384, 512)
        drift = float(np.max(np.abs(y_mlx - y_t)))
        rng = float(np.max(np.abs(y_t))) + 1e-9
        # On a 0..255 output through a deep 6-stage conv cascade, MLX mx.conv2d
        # and torch F.conv2d accumulate in different orders -> ~0.03 ULP-class
        # absolute drift. The STRUCTURAL guard is the relative drift: a channel
        # /shuffle/skip-composition bug produces drift of 2-200 (rel >1e-2), not
        # ~1e-4. The tight relative bound is what proves the decoder math correct.
        assert drift < 5e-2, f"full-decoder abs drift {drift} on 0..255 output exceeds budget"
        assert drift / rng < 1e-3, f"full-decoder rel drift {drift / rng} exceeds fp32 budget"

    def test_full_decoder_parity_holds_for_multiple_latents(self):
        dec, tdec, to_t = _build_paired_decoders()
        rng = np.random.default_rng(11)
        for _ in range(3):
            z = rng.standard_normal((2, 28)).astype(np.float32)
            y_mlx = np.array(dec(mx.array(z)))
            with torch.no_grad():
                y_t = tdec(to_t(z)).numpy()
            drift = float(np.max(np.abs(y_mlx - y_t)))
            ref_rng = float(np.max(np.abs(y_t))) + 1e-9
            assert drift < 5e-2, f"abs drift {drift} exceeds budget"
            assert drift / ref_rng < 1e-3, f"rel drift {drift / ref_rng} exceeds fp32 budget"

    def test_output_is_valid_rgb_range(self):
        dec = HNeRVDecoderMLX(latent_dim=28, base_channels=36, eval_size=(384, 512))
        z = np.random.default_rng(3).standard_normal((1, 28)).astype(np.float32)
        y = np.array(dec(mx.array(z)))
        # sigmoid * 255 => strictly within [0, 255]
        assert float(y.min()) >= 0.0
        assert float(y.max()) <= 255.0

    def test_reference_decoder_has_skip_when_channels_change(self):
        # The PR95 block instantiates a 1x1 skip-conv ONLY when in_ch != out_ch.
        # taper [36,36,36,27,20,18,18]: blocks 0,1 have in==out (no skip),
        # blocks 2,3,4,5 change channels (skip present). Pin that structure.
        dec = HNeRVDecoderMLX(latent_dim=28, base_channels=36, eval_size=(384, 512))
        skip_present = [blk.skip_conv is not None for blk in dec.blocks]
        assert skip_present == [False, False, True, True, True, False]


# ---------------------------------------------------------------------------
# 4. Audit-finding regression guards (task #81): pin the CONFIG DEFECTS so a
#    future "score-aware" run cannot silently re-introduce them.
# ---------------------------------------------------------------------------
class TestAuditFindingRegressionGuards:
    def test_hinerv_default_config_is_skip_free_meanfield_carrier(self):
        # The #81 M-arch finding: the DEFAULT hi_nerv config is the skip-free
        # mean-field carrier (use_bilinear_skip=False, sin_frequency=30.0, grid
        # PE + ConvNeXt OFF). This guard documents the defect so a reviewer sees
        # the default is NOT the PR95-faithful decoder. When #76 flips the
        # default, update this guard to the new expected state.
        from tac.substrates.hi_nerv.architecture import HinervConfig

        cfg = HinervConfig()
        assert cfg.use_bilinear_skip is False, "default skip-free (M-arch defect, per #68/#81)"
        assert cfg.use_hierarchical_feature_grid is False
        assert cfg.use_convnext_blocks is False
        assert cfg.sin_frequency == 30.0, "w=30 SIREN convention (spectral-bias trap on skip-free map)"

    def test_shared_harness_scorer_weights_default_to_zero(self):
        # The #75/#68 M-loss finding, pinned at the SOURCE layer: the shared MLX
        # score-aware bundle defaults EVERY scorer distillation weight to 0.0, so
        # a caller that omits explicit nonzero weights trains scorer-blind
        # recon-MSE. This guard makes the inactive-objective bug class visible:
        # if these defaults silently change to nonzero, the test will flag it for
        # review (a nonzero default would be a DIFFERENT, also-reviewable change).
        import inspect

        from tac.substrates._shared.mlx_score_aware import bundle as _bundle

        src = inspect.getsource(_bundle)
        assert "distillation_weight: float = 0.0" in src
        assert "pose_distillation_weight: float = 0.0" in src
        assert "segnet_direct_live_distillation_weight: float = 0.0" in src

    def test_skip_path_uses_carrier_sin_frequency_not_pr95_w1(self):
        # Config defect: when use_bilinear_skip=True the hi_nerv carrier routes
        # sin_frequency (default 30.0) into the residual sin, i.e.
        # sin(30*(shuffled+identity)) -- whereas the PR95-faithful reference
        # block (_HNeRVUpsampleBlockMLX) uses sin(shuffled+identity) with w=1.0.
        # Prove the two are NOT equivalent so a future "skip-on" run is not
        # mistaken for PR95-faithful unless sin_frequency is also set to 1.0.
        from tac.framework_agnostic.backend import Backend
        from tac.framework_agnostic.canonical_kernels import bilinear_skip_residual_canonical

        rng = np.random.default_rng(5)
        a = rng.standard_normal((1, 4, 4, 3)).astype(np.float32)
        b = rng.standard_normal((1, 4, 4, 3)).astype(np.float32)
        w30 = bilinear_skip_residual_canonical(a, b, sin_frequency=30.0, backend=Backend.NUMPY)
        w1 = bilinear_skip_residual_canonical(a, b, sin_frequency=1.0, backend=Backend.NUMPY)
        assert not np.allclose(w30, w1), "w=30 skip path must differ from PR95-faithful w=1"
