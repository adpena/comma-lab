# SPDX-License-Identifier: MIT
"""NO-FAKE tests for tac.margin_saliency_map.

These verify ACTUAL behavior (real autograd gradient flow through a tiny stub
SegNet, real argmax/margin math, real concentration arithmetic), NOT constants.
If the producer were replaced by ``return zeros``, the gradient-flow +
concentration tests would FAIL.
"""

from __future__ import annotations

import pytest
import torch

from tac.margin_saliency_map import (
    HNERV_STAGE_OUTPUT_HW,
    DecoderTensorSaliency,
    MarginSaliency,
    MarginSaliencyError,
    _gini,
    _stage_for_tensor_name,
    _topk_margin,
    compute_decoder_tensor_margin_saliency,
    compute_margin_saliency_map,
    saliency_boundary_concentration,
    verify_stage_saliency_ordering,
)


class _StubSegNet:
    """A minimal differentiable SegNet stub: slice last frame, bilinear to a
    small decision grid, then a fixed 1x1 conv to 5 classes. The conv weights
    make some pixels boundary (small margin) and some interior, with a REAL
    dependence on the input so autograd produces a non-trivial saliency."""

    def __init__(self, grid_hw=(8, 12), in_ch=3, classes=5):
        self.grid_hw = grid_hw
        torch.manual_seed(0)
        self.w = torch.randn(classes, in_ch, 1, 1)

    def preprocess_input(self, x):
        x = x[:, -1, ...]  # (B, C, H, W) last frame
        return torch.nn.functional.interpolate(x, size=self.grid_hw, mode="bilinear")

    def __call__(self, seg_in):
        return torch.nn.functional.conv2d(seg_in, self.w)  # (B, 5, H, W)


def _frame(h=16, w=24):
    torch.manual_seed(1)
    return torch.rand(3, h, w) * 255.0


def test_topk_margin_is_nonnegative_and_correct():
    logits = torch.tensor([[[[3.0]], [[1.0]], [[2.0]], [[0.0]], [[-1.0]]]])  # (1,5,1,1)
    m = _topk_margin(logits)
    assert m.shape == (1, 1, 1)
    assert float(m[0, 0, 0]) == pytest.approx(1.0)  # top1=3, top2=2 -> margin 1


def test_topk_margin_rejects_bad_shape():
    with pytest.raises(MarginSaliencyError):
        _topk_margin(torch.zeros(5, 8, 8))  # not (B, C, H, W)


def test_compute_returns_real_grids_on_decision_grid():
    seg = _StubSegNet(grid_hw=(8, 12))
    out = compute_margin_saliency_map(seg, _frame())
    assert isinstance(out, MarginSaliency)
    assert out.grid_hw == (8, 12)
    assert out.saliency.shape == (8, 12)
    assert out.margin.shape == (8, 12)
    # margin is top1-top2 >= 0 always
    assert float(out.margin.min()) >= 0.0


def test_saliency_is_real_gradient_not_zero():
    """The headline NO-FAKE guard: the saliency must be a genuine non-zero
    autograd gradient. If the producer returned zeros/constants this FAILS."""
    seg = _StubSegNet()
    out = compute_margin_saliency_map(seg, _frame())
    assert float(out.saliency.abs().sum()) > 0.0
    # not a constant map (a real gradient varies spatially)
    assert float(out.saliency.std()) > 0.0


def test_saliency_responds_to_input_change():
    """Different input frames produce different saliency (real dependence)."""
    seg = _StubSegNet()
    a = compute_margin_saliency_map(seg, _frame(16, 24)).saliency
    torch.manual_seed(99)
    b = compute_margin_saliency_map(seg, torch.rand(3, 16, 24) * 255.0).saliency
    assert not torch.allclose(a, b)


def test_flip_mask_changes_the_saliency():
    """Targeting a flip-pixel subset gives a different map than the global one."""
    seg = _StubSegNet(grid_hw=(8, 12))
    glob = compute_margin_saliency_map(seg, _frame())
    mask = torch.zeros(8, 12, dtype=torch.bool)
    mask[0:2, 0:2] = True
    targeted = compute_margin_saliency_map(seg, _frame(), flip_pixel_mask=mask)
    assert not torch.allclose(glob.saliency, targeted.saliency)


def test_flip_mask_shape_mismatch_raises():
    seg = _StubSegNet(grid_hw=(8, 12))
    with pytest.raises(MarginSaliencyError):
        compute_margin_saliency_map(
            seg, _frame(), flip_pixel_mask=torch.zeros(4, 4, dtype=torch.bool)
        )


def test_bad_frame_shape_raises():
    seg = _StubSegNet()
    with pytest.raises(MarginSaliencyError):
        compute_margin_saliency_map(seg, torch.zeros(16, 24))  # not (3, H, W)


def test_concentration_arithmetic_real():
    # construct a saliency where boundary (low-margin) pixels carry all mass
    margin = torch.tensor([[0.1, 0.1], [5.0, 5.0]])  # 2 boundary, 2 interior
    sal = torch.tensor([[10.0, 10.0], [0.0, 0.0]])  # all mass at boundary
    c = saliency_boundary_concentration(sal, margin, boundary_margin=0.5)
    assert c["frac_saliency_mass_in_boundary"] == pytest.approx(1.0)
    assert c["mean_saliency_interior"] == pytest.approx(0.0)
    assert c["boundary_over_interior_ratio"] == float("inf")
    assert c["frac_pixels_boundary"] == pytest.approx(0.5)


def test_concentration_uniform_case():
    margin = torch.tensor([[0.1, 0.1], [5.0, 5.0]])
    sal = torch.ones(2, 2)
    c = saliency_boundary_concentration(sal, margin, boundary_margin=0.5)
    assert c["boundary_over_interior_ratio"] == pytest.approx(1.0)
    assert c["frac_saliency_mass_in_boundary"] == pytest.approx(0.5)


def test_gini_extremes():
    assert _gini(torch.ones(100)) == pytest.approx(0.0, abs=1e-6)  # uniform
    spike = torch.zeros(100)
    spike[0] = 1.0
    assert _gini(spike) > 0.9  # concentrated
    assert _gini(torch.zeros(10)) == 0.0  # degenerate


def test_gini_empty():
    assert _gini(torch.tensor([])) == 0.0


# ---------------------------------------------------------------------------
# Per-decoder-tensor / per-stage saliency (the QAT bit-allocation prior)
# ---------------------------------------------------------------------------


class _StubDecoder(torch.nn.Module):
    """A tiny named-module decoder mirroring the HNeRV name geometry so the
    per-tensor producer can be exercised with REAL per-weight gradients. ``stem``
    lifts the 28-d latent to a small base grid; ``blocks.0`` then ``blocks.5``
    convolve toward the output; ``rgb_0``/``rgb_1`` are the two output heads. Only
    the names matter for the stage mapping; the math is real autograd."""

    def __init__(self, base_hw=(4, 6)):
        super().__init__()
        self.base_h, self.base_w = base_hw
        self.stem = torch.nn.Linear(28, 6 * self.base_h * self.base_w)
        self.blocks = torch.nn.ModuleList(
            [torch.nn.Conv2d(6, 6, 3, padding=1) for _ in range(6)]
        )
        self.rgb_0 = torch.nn.Conv2d(6, 3, 3, padding=1)
        self.rgb_1 = torch.nn.Conv2d(6, 3, 3, padding=1)

    def forward(self, z, head="rgb_0"):
        b = z.shape[0]
        x = torch.sin(self.stem(z).view(b, 6, self.base_h, self.base_w))
        for blk in self.blocks:
            x = torch.sin(blk(x))
        head_mod = self.rgb_0 if head == "rgb_0" else self.rgb_1
        return torch.sigmoid(head_mod(x)) * 255.0  # (b, 3, H, W)


def _stub_render(decoder, latent_row, score_head):
    out = decoder(latent_row, head=score_head)  # (1, 3, H, W)
    return out[0]  # (3, H, W) WITH grad


def _stub_latents(n=6):
    torch.manual_seed(3)
    return torch.rand(n, 28)


def test_stage_name_mapping():
    assert _stage_for_tensor_name("stem.weight") == "stem"
    assert _stage_for_tensor_name("blocks.4.weight") == "blocks.4"
    assert _stage_for_tensor_name("blocks.5.bias") == "blocks.5"
    assert _stage_for_tensor_name("rgb_0.weight") == "rgb_0"
    assert _stage_for_tensor_name("skips.2.weight") == "skips.2"
    # unknown -> first dotted component
    assert _stage_for_tensor_name("mystery.layer.weight") == "mystery"


def test_decoder_tensor_saliency_is_real_per_weight_gradient():
    """Headline NO-FAKE guard: per-tensor saliency is a genuine non-zero autograd
    gradient of the SegNet margin w.r.t. EACH decoder weight. If the producer
    returned zeros/constants this FAILS."""
    dec = _StubDecoder()
    seg = _StubSegNet(grid_hw=(8, 12))
    res = compute_decoder_tensor_margin_saliency(
        dec, _stub_latents(), seg, render_frame_chw=_stub_render, frame_indices=[0, 1, 2]
    )
    assert isinstance(res, DecoderTensorSaliency)
    assert res.n_frames == 3
    # every param tensor present and non-negative
    names = {n for n, _ in dec.named_parameters()}
    assert set(res.per_tensor) == names
    assert all(v >= 0.0 for v in res.per_tensor.values())
    # at least some tensors carry real (non-zero) saliency
    assert sum(v for v in res.per_tensor.values()) > 0.0
    # the ranking is NOT all-equal (a real gradient varies across tensors)
    vals = [v for _, v in res.ranking]
    assert max(vals) > min(vals)


def test_decoder_tensor_ranking_is_sorted_descending():
    dec = _StubDecoder()
    seg = _StubSegNet(grid_hw=(8, 12))
    res = compute_decoder_tensor_margin_saliency(
        dec, _stub_latents(), seg, render_frame_chw=_stub_render, frame_indices=[0, 1]
    )
    vals = [v for _, v in res.ranking]
    assert vals == sorted(vals, reverse=True)
    # ranking covers exactly the per_tensor keys
    assert {n for n, _ in res.ranking} == set(res.per_tensor)


def test_decoder_tensor_ranking_responds_to_zeroed_head():
    """A head that does NOT contribute to the scored frame must rank LOWER than
    the scored head — the ranking reflects the REAL score path, not a constant.
    Scoring through rgb_0 must give rgb_0.weight strictly more saliency than
    rgb_1.weight (which is off the rgb_0 graph entirely -> exactly zero)."""
    dec = _StubDecoder()
    seg = _StubSegNet(grid_hw=(8, 12))
    res = compute_decoder_tensor_margin_saliency(
        dec, _stub_latents(), seg, render_frame_chw=_stub_render,
        frame_indices=[0, 1, 2], score_head="rgb_0",
    )
    assert res.per_tensor["rgb_1.weight"] == pytest.approx(0.0)
    assert res.per_tensor["rgb_0.weight"] > 0.0


def test_decoder_tensor_normalized_divides_by_sqrt_numel():
    dec = _StubDecoder()
    seg = _StubSegNet(grid_hw=(8, 12))
    res = compute_decoder_tensor_margin_saliency(
        dec, _stub_latents(), seg, render_frame_chw=_stub_render, frame_indices=[0]
    )
    for name, raw in res.per_tensor.items():
        ne = res.tensor_numel[name]
        expected = raw / (ne ** 0.5) if ne > 0 else 0.0
        assert res.per_tensor_normalized[name] == pytest.approx(expected)


def test_decoder_tensor_per_stage_sums_match():
    dec = _StubDecoder()
    seg = _StubSegNet(grid_hw=(8, 12))
    res = compute_decoder_tensor_margin_saliency(
        dec, _stub_latents(), seg, render_frame_chw=_stub_render, frame_indices=[0, 1]
    )
    # per_stage mass == sum of its member tensors
    from collections import defaultdict

    recomputed = defaultdict(float)
    for name, v in res.per_tensor.items():
        recomputed[_stage_for_tensor_name(name)] += v
    for stage, mass in res.per_stage.items():
        assert mass == pytest.approx(recomputed[stage])


def test_decoder_tensor_default_frame_spread():
    """frame_indices=None picks a deterministic spread (<=8) across the sequence."""
    dec = _StubDecoder()
    seg = _StubSegNet(grid_hw=(8, 12))
    res = compute_decoder_tensor_margin_saliency(
        dec, _stub_latents(n=20), seg, render_frame_chw=_stub_render
    )
    assert res.n_frames == 8


def test_decoder_tensor_bad_latent_shape_raises():
    dec = _StubDecoder()
    seg = _StubSegNet(grid_hw=(8, 12))
    with pytest.raises(MarginSaliencyError):
        compute_decoder_tensor_margin_saliency(
            dec, torch.rand(28), seg, render_frame_chw=_stub_render
        )


def test_decoder_tensor_bad_frame_index_raises():
    dec = _StubDecoder()
    seg = _StubSegNet(grid_hw=(8, 12))
    with pytest.raises(MarginSaliencyError):
        compute_decoder_tensor_margin_saliency(
            dec, _stub_latents(n=3), seg, render_frame_chw=_stub_render, frame_indices=[5]
        )


def test_decoder_tensor_render_bad_shape_raises():
    dec = _StubDecoder()
    seg = _StubSegNet(grid_hw=(8, 12))

    def bad_render(decoder, latent_row, score_head):
        return decoder(latent_row, head=score_head)  # (1,3,H,W) not (3,H,W)

    with pytest.raises(MarginSaliencyError):
        compute_decoder_tensor_margin_saliency(
            dec, _stub_latents(), seg, render_frame_chw=bad_render, frame_indices=[0]
        )


def test_decoder_tensor_flip_mask_fn_changes_result():
    dec = _StubDecoder()
    seg = _StubSegNet(grid_hw=(8, 12))
    glob = compute_decoder_tensor_margin_saliency(
        dec, _stub_latents(), seg, render_frame_chw=_stub_render, frame_indices=[0, 1]
    )

    def mask_fn(logits):
        m = torch.zeros(logits.shape[-2], logits.shape[-1], dtype=torch.bool)
        m[0:2, 0:2] = True
        return m

    targeted = compute_decoder_tensor_margin_saliency(
        dec, _stub_latents(), seg, render_frame_chw=_stub_render,
        frame_indices=[0, 1], flip_pixel_mask_fn=mask_fn,
    )
    # restricting to a pixel subset changes the per-tensor mass
    assert glob.per_tensor != targeted.per_tensor


def test_decoder_tensor_flip_mask_fn_shape_mismatch_raises():
    dec = _StubDecoder()
    seg = _StubSegNet(grid_hw=(8, 12))

    def bad_mask_fn(logits):
        return torch.zeros(2, 2, dtype=torch.bool)  # wrong grid

    with pytest.raises(MarginSaliencyError):
        compute_decoder_tensor_margin_saliency(
            dec, _stub_latents(), seg, render_frame_chw=_stub_render,
            frame_indices=[0], flip_pixel_mask_fn=bad_mask_fn,
        )


def test_verify_stage_ordering_decision_band_wins():
    """When the decision-band stages carry more mass, the verdict is True."""
    per_stage = {"stem": 1.0, "blocks.0": 1.0, "blocks.5": 50.0, "rgb_0": 40.0}
    verdict = verify_stage_saliency_ordering(per_stage, HNERV_STAGE_OUTPUT_HW)
    assert verdict["ordering_matches_geometry"] is True
    assert "blocks.5" in verdict["decision_stages"]
    assert "stem" in verdict["blind_stages"]
    assert verdict["decision_band_mass_fraction"] > 0.5


def test_verify_stage_ordering_blind_band_wins():
    """When coarse/blind stages dominate, the verdict flags a mismatch."""
    per_stage = {"stem": 100.0, "blocks.0": 50.0, "blocks.5": 1.0}
    verdict = verify_stage_saliency_ordering(per_stage, HNERV_STAGE_OUTPUT_HW)
    assert verdict["ordering_matches_geometry"] is False
    assert verdict["decision_band_mass_fraction"] < 0.5
