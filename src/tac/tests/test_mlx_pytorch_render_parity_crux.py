# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the MLX<->PyTorch render-parity crux harness.

Every test exercises the REAL PR95-HNeRV carrier weights + the REAL PyTorch
reference decoder forward + the REAL MLX decoder forward. There are NO synthetic
fixtures, NO toy tensors, NO fabricated parity numbers. Each test would FAIL if
the corresponding fix were reverted (Slot EEE Class 2 behavioral discipline):

* ``test_first_divergent_op_is_a_conv_path`` fails if a structural layout bug
  (transpose/PixelShuffle-convention/bilinear) were introduced — the non-conv
  ops (sin0, interp, skip-1x1) must stay byte-stable.
* ``test_render_is_uint8_faithful`` fails if the render drift escaped the uint8
  floor.
* ``test_d_seg_identical_across_render_modes`` fails if render parity (source #1)
  actually moved the SegNet d_seg — the directive's premise. It does NOT.
* ``test_fixed_fp64_tightens_float_drift`` fails if the fp64 conv mode stopped
  reducing the float drift.
* ``test_carrier_default_render_mode_and_low_drift_override`` fails if the
  carrier loader stopped exposing the low-drift ``fixed_fp64`` override or if the
  default mode constant disappeared (the default is the fast ``optimized`` mode
  because it is already uint8-faithful + yields identical d_seg — fp64 buys zero
  d_seg for ~56x cost).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
REAL_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/lightning_batch"
    / "exact_eval_public_pr95_hnerv_muon_t4_fix2_20260504T0848Z/archive.zip"
)
REAL_VIDEO = REPO_ROOT / "upstream/videos/0.mkv"
_PYTORCH_REF = (
    REPO_ROOT
    / "experiments/results/public_pr_archive_kaggle_mirror"
    / "public_pr95_intake_20260505_auto/source/submissions/hnerv_muon/src/model.py"
)

try:  # MLX is required for the render path (Apple Silicon).
    import mlx.core as _mx  # noqa: F401

    _HAS_MLX = True
except Exception:  # pragma: no cover - environment dependent
    _HAS_MLX = False

requires_mlx = pytest.mark.skipif(not _HAS_MLX, reason="MLX not available")
requires_real_carrier = pytest.mark.skipif(
    not REAL_ARCHIVE.is_file(), reason="real PR95 carrier archive not present"
)
requires_pytorch_ref = pytest.mark.skipif(
    not _PYTORCH_REF.is_file(), reason="PyTorch reference decoder not present"
)
requires_video = pytest.mark.skipif(
    not REAL_VIDEO.is_file(), reason="real upstream/videos/0.mkv not present"
)


def _load_real_carrier():
    from tac.local_acceleration.pr95_hnerv_mlx import parse_pr95_public_archive_zip

    pkt = parse_pr95_public_archive_zip(REAL_ARCHIVE)
    lat = np.asarray(pkt.latents).astype(np.float32)
    return pkt.state_dict, lat


@requires_mlx
@requires_real_carrier
@requires_pytorch_ref
def test_first_divergent_op_is_a_conv_path() -> None:
    """The FIRST divergence is the first conv (b0_ps); non-conv ops are byte-stable.

    Reverting to a structural layout bug (e.g. a transpose error in the
    state_dict loader, a PixelShuffle convention flip, or an align_corners
    mismatch) would push ``sin0`` / ``b0_interp`` / ``b0_skip`` above the
    structural tolerance — this test asserts they stay byte-stable.
    """
    from tac.analysis.mlx_pytorch_render_parity_crux import localize_render_parity_crux

    sd, lat = _load_real_carrier()
    rep = localize_render_parity_crux(
        sd, lat[0], latent_dim=28, base_channels=36, conv_accumulation_mode="optimized"
    )
    layers = {d.layer: d for d in rep.per_layer}
    # Non-conv structural ops are byte-stable (no layout/convention bug).
    assert layers["sin0"].max_abs < 1.0e-6, layers["sin0"]
    assert layers["b0_interp"].max_abs < 1.0e-6, layers["b0_interp"]
    assert layers["b0_skip"].max_abs < 1.0e-6, layers["b0_skip"]
    # The first divergence is a conv path output (the PixelShuffle of conv).
    assert rep.first_divergent_layer in {"b0_ps", "b0_conv", "b0_out"}, (
        rep.first_divergent_layer
    )
    assert rep.crux_op == "conv2d_fp32_accumulation_order", rep.crux_op


@requires_mlx
@requires_real_carrier
@requires_pytorch_ref
def test_render_is_uint8_faithful() -> None:
    """The MLX render matches PyTorch within <=1 uint8 LSB on <0.01% of pixels.

    This is the render-faithfulness that actually matters (the contest inflate
    casts to uint8). Reverting the conv accumulation handling such that the
    drift escaped the uint8 floor would fail this.
    """
    from tac.analysis.mlx_pytorch_render_parity_crux import localize_render_parity_crux

    sd, lat = _load_real_carrier()
    for mode in ("optimized", "fixed_fp64"):
        rep = localize_render_parity_crux(
            sd, lat[0], latent_dim=28, base_channels=36, conv_accumulation_mode=mode
        )
        assert rep.final_frame_uint8_max_abs <= 1, (mode, rep.final_frame_uint8_max_abs)
        assert rep.final_frame_uint8_fraction_differ < 1.0e-4, (
            mode,
            rep.final_frame_uint8_fraction_differ,
        )
        # The float drift is bounded ~1e-3 (sigmoid*255-amplified conv drift).
        assert rep.final_frame_float_max_abs < 5.0e-3, (mode, rep.final_frame_float_max_abs)


@requires_mlx
@requires_real_carrier
@requires_pytorch_ref
def test_fixed_fp64_tightens_float_drift() -> None:
    """The canonical fp64 conv mode reduces the float drift vs the native path.

    Reverting the ``fixed_fp64`` accumulation so it no longer tightens the conv
    drift would fail this (defense-in-depth claim made concrete).
    """
    from tac.analysis.mlx_pytorch_render_parity_crux import localize_render_parity_crux

    sd, lat = _load_real_carrier()
    opt = localize_render_parity_crux(
        sd, lat[0], latent_dim=28, base_channels=36, conv_accumulation_mode="optimized"
    )
    fp64 = localize_render_parity_crux(
        sd, lat[0], latent_dim=28, base_channels=36, conv_accumulation_mode="fixed_fp64"
    )
    # fp64 accumulation reduces the intermediate conv drift; the b5_out feature
    # (deepest block) must be strictly tighter under fp64.
    opt_b5 = {d.layer: d for d in opt.per_layer}["b5_out"].max_abs
    fp64_b5 = {d.layer: d for d in fp64.per_layer}["b5_out"].max_abs
    assert fp64_b5 < opt_b5, (fp64_b5, opt_b5)


@requires_mlx
@requires_real_carrier
@requires_pytorch_ref
@requires_video
def test_d_seg_identical_across_render_modes() -> None:
    """Render parity (drift source #1) has ZERO impact on the SegNet d_seg.

    This is the directive's premise under test: the claim was that render-parity
    drift "roughly DOUBLED the distortion". It does NOT. The SegNet
    argmax-flip d_seg measured on the carrier vs real ground-truth is IDENTICAL
    whether the carrier is rendered with MLX-optimized, MLX-fixed_fp64, OR the
    PyTorch-fp32 faithful reference. The carrier distortion is the carrier R(D)
    + the eval-hardware axis (drift source #2), NOT the render parity.

    Reverting this finding (i.e. if render parity DID move d_seg) would fail the
    ``== pytest.approx`` assertions.
    """
    import torch

    from tac.analysis.inverse_steganalysis_linf_vs_l2_gate import measure_pair_d_seg_d_pose
    from tac.analysis.mlx_pytorch_render_parity_crux import (
        _import_pytorch_reference_decoder,
    )
    from tac.analysis.pr95_hnerv_linf_carrier import (
        _resize_pair_to,
        load_carrier_decoder,
        render_carrier_pair_bcthw,
    )
    from tac.analysis.score_exact_saliency import (
        decode_real_pairs,
        load_score_exact_scorers,
    )

    sd, lat = _load_real_carrier()
    gt = decode_real_pairs(str(REAL_VIDEO), 2, pair_stride=64, start_pair=0, device="cpu")
    posenet, segnet = load_score_exact_scorers("upstream", device="cpu")
    pair_indices = [0, 64]
    h, w = gt.shape[-2:]

    # PyTorch-fp32 faithful render of the same latents.
    decoder_cls = _import_pytorch_reference_decoder()
    pt = decoder_cls(latent_dim=28, base_channels=36)
    pt.load_state_dict(
        {k: torch.from_numpy(np.asarray(v).astype(np.float32)) for k, v in sd.items()}
    )
    pt.eval()

    def pt_render(z_row: np.ndarray) -> torch.Tensor:
        with torch.no_grad():
            return pt(
                torch.from_numpy(z_row.reshape(1, -1).astype(np.float32))
            ).float()

    dec_opt, _, _ = load_carrier_decoder(
        REAL_ARCHIVE, conv2d_accumulation_mode="optimized"
    )
    dec_fp64, _, _ = load_carrier_decoder(
        REAL_ARCHIVE, conv2d_accumulation_mode="fixed_fp64"
    )

    def measure(renderer) -> tuple[float, float]:
        ds = dp = 0.0
        for j, pi in enumerate(pair_indices):
            cp = _resize_pair_to(renderer(lat[pi]), h, w)
            d_seg, d_pose = measure_pair_d_seg_d_pose(posenet, segnet, gt[j : j + 1], cp)
            ds += d_seg
            dp += d_pose
        n = len(pair_indices)
        return ds / n, dp / n

    pt_seg, _ = measure(pt_render)
    opt_seg, _ = measure(lambda z: render_carrier_pair_bcthw(dec_opt, z))
    fp64_seg, _ = measure(lambda z: render_carrier_pair_bcthw(dec_fp64, z))

    # The d_seg (argmax-flip RATE) is IDENTICAL across all three renders. The
    # render-parity drift is sub-quantization for the SegNet argmax.
    assert opt_seg == pytest.approx(pt_seg, abs=1.0e-9), (opt_seg, pt_seg)
    assert fp64_seg == pytest.approx(pt_seg, abs=1.0e-9), (fp64_seg, pt_seg)


@requires_mlx
@requires_real_carrier
def test_carrier_default_render_mode_and_low_drift_override() -> None:
    """The carrier loader defaults to the fast mode + exposes the fp64 override.

    The default is ``optimized`` (already uint8-faithful + identical d_seg; fp64
    buys zero d_seg for ~56x cost). The byte-tightest ``fixed_fp64`` mode is
    available via the explicit kwarg and is auto-pinned to the MLX CPU device.
    Reverting so the low-drift override is no longer honored would fail this.
    """
    from tac.analysis.pr95_hnerv_linf_carrier import (
        CARRIER_RENDER_DEFAULT_CONV_MODE,
        CARRIER_RENDER_LOW_DRIFT_CONV_MODE,
        load_carrier_decoder,
        render_carrier_pair_bcthw,
    )

    assert CARRIER_RENDER_DEFAULT_CONV_MODE == "optimized"
    assert CARRIER_RENDER_LOW_DRIFT_CONV_MODE == "fixed_fp64"
    dec, lat, _ = load_carrier_decoder(REAL_ARCHIVE)
    assert dec.conv2d_accumulation_mode == "optimized"
    # Explicit low-drift override is honored AND renders (auto-pins MLX CPU for
    # fp64 conv, which Metal cannot accumulate).
    dec2, _, _ = load_carrier_decoder(
        REAL_ARCHIVE, conv2d_accumulation_mode="fixed_fp64"
    )
    assert dec2.conv2d_accumulation_mode == "fixed_fp64"
    pair = render_carrier_pair_bcthw(dec2, lat[0])
    assert tuple(pair.shape) == (1, 2, 3, 384, 512)


@requires_mlx
@requires_real_carrier
@requires_pytorch_ref
def test_report_is_non_promotable_and_serializable() -> None:
    """The parity report is never a score claim and round-trips to a dict."""
    from tac.analysis.mlx_pytorch_render_parity_crux import localize_render_parity_crux

    sd, lat = _load_real_carrier()
    rep = localize_render_parity_crux(sd, lat[0], latent_dim=28, base_channels=36)
    assert rep.score_claim is False
    assert rep.promotable is False
    assert rep.axis_tag == "[macOS-MLX vs PyTorch-CPU parity, exact-measured]"
    d = rep.as_dict()
    assert d["schema"] == "pr95_hnerv_mlx_pytorch_render_parity_crux.v1"
    assert d["crux_op"] == "conv2d_fp32_accumulation_order"
    assert isinstance(d["per_layer"], list) and len(d["per_layer"]) > 10


def test_harness_rejects_missing_pytorch_reference(tmp_path, monkeypatch) -> None:
    """The harness raises a clear error if the PyTorch reference is absent."""
    import tac.analysis.mlx_pytorch_render_parity_crux as crux

    monkeypatch.setattr(crux, "_PYTORCH_REFERENCE_MODEL_DIR", tmp_path / "nope")
    with pytest.raises(crux.RenderParityCruxError):
        crux._import_pytorch_reference_decoder()
