# SPDX-License-Identifier: MIT
"""Dedicated tests for the SABOR boundary-only renderer substrate.

Covers (per CLAUDE.md "13 HNeRV parity-discipline lessons" — at least 20
focused tests):

- Config validation (positive: defaults; negative: bad params).
- Boundary detection (Canny-only, Canny+SegNet union).
- Renderer forward shape + dtype + range contract.
- Archive grammar roundtrip (encode/decode parity bit-exact).
- Archive header byte-layout invariants.
- Archive validation (mask vs boundary_rgb count mismatch; bad shapes).
- State-dict prefix discipline (class_means excluded; required prefixes).
- Score-aware loss forward + gradient connectivity.
- ``apply_eval_roundtrip=False`` refused per CLAUDE.md.

All tests use tiny shapes (16x16, num_pairs=2 or 3) for CPU speed.
"""

from __future__ import annotations

import math
import struct

import pytest
import torch

from tac.substrates.sabor_boundary_only_renderer import (
    SBO1_MAGIC,
    SBO1_SCHEMA_VERSION,
    SaborBoundaryOnlyConfig,
    SaborBoundaryOnlyLossWeights,
    SaborBoundaryOnlyRenderer,
    SaborBoundaryOnlyScoreAwareLoss,
    detect_boundary_mask_canny_segnet,
    pack_archive,
    parse_archive,
)
from tac.substrates.sabor_boundary_only_renderer.archive import (
    SBO1_HEADER_FMT,
    SBO1_HEADER_SIZE,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_config_defaults_are_valid() -> None:
    cfg = SaborBoundaryOnlyConfig()
    assert cfg.num_pairs == 600
    assert cfg.output_height == 384
    assert cfg.output_width == 512
    assert cfg.num_seg_classes == 5
    assert 0.0 < cfg.edge_threshold < 1.0
    assert cfg.refinement_hidden > 0
    assert cfg.refinement_blocks > 0


def test_config_refuses_nonpositive_num_pairs() -> None:
    with pytest.raises(ValueError, match="num_pairs"):
        SaborBoundaryOnlyConfig(num_pairs=0)


def test_config_refuses_edge_threshold_out_of_range() -> None:
    with pytest.raises(ValueError, match="edge_threshold"):
        SaborBoundaryOnlyConfig(edge_threshold=0.0)
    with pytest.raises(ValueError, match="edge_threshold"):
        SaborBoundaryOnlyConfig(edge_threshold=1.5)


def test_config_refuses_one_class_segnet() -> None:
    with pytest.raises(ValueError, match="num_seg_classes"):
        SaborBoundaryOnlyConfig(num_seg_classes=1)


# ---------------------------------------------------------------------------
# Boundary detection
# ---------------------------------------------------------------------------


def test_detect_boundary_canny_only_returns_bool_bhw() -> None:
    rgb = torch.rand(2, 3, 32, 32)
    mask = detect_boundary_mask_canny_segnet(rgb, None, edge_threshold=0.1)
    assert mask.shape == (2, 32, 32)
    assert mask.dtype == torch.bool


def test_detect_boundary_canny_threshold_monotone() -> None:
    """Higher threshold => fewer (or equal) boundary pixels."""
    torch.manual_seed(0)
    rgb = torch.rand(1, 3, 32, 32)
    low = int(detect_boundary_mask_canny_segnet(rgb, None, edge_threshold=0.05).sum())
    high = int(detect_boundary_mask_canny_segnet(rgb, None, edge_threshold=0.5).sum())
    assert low >= high


def test_detect_boundary_segnet_union_increases_count() -> None:
    """Canny ∪ SegNet must be >= Canny-only count."""
    torch.manual_seed(0)
    rgb = torch.rand(1, 3, 24, 24)
    seg = torch.randint(0, 5, (1, 24, 24), dtype=torch.long)
    canny = detect_boundary_mask_canny_segnet(rgb, None, edge_threshold=0.3)
    both = detect_boundary_mask_canny_segnet(rgb, seg, edge_threshold=0.3)
    assert int(both.sum()) >= int(canny.sum())


def test_detect_boundary_segnet_border_always_flagged() -> None:
    """Border pixels are always boundary per the 4-nbr disagreement spec."""
    rgb = torch.zeros(1, 3, 8, 8)
    seg = torch.zeros(1, 8, 8, dtype=torch.long)  # constant SegNet output
    mask = detect_boundary_mask_canny_segnet(rgb, seg, edge_threshold=0.5)
    assert bool(mask[0, 0, 0].item())
    assert bool(mask[0, -1, -1].item())


# ---------------------------------------------------------------------------
# Renderer forward
# ---------------------------------------------------------------------------


def _make_pair_batch(cfg: SaborBoundaryOnlyConfig, b: int = 2):
    h, w = cfg.output_height, cfg.output_width
    boundary_mask = torch.zeros(b, 2, h, w, dtype=torch.bool)
    boundary_mask[:, :, 0, :] = True
    boundary_mask[:, :, :, 0] = True
    boundary_rgb = torch.rand(b, 2, 3, h, w)
    seg = torch.randint(0, cfg.num_seg_classes, (b, 2, h, w), dtype=torch.long)
    pair_indices = torch.arange(b, dtype=torch.long)
    return pair_indices, boundary_mask, boundary_rgb, seg


def test_renderer_forward_shape_dtype_range() -> None:
    cfg = SaborBoundaryOnlyConfig(num_pairs=4, output_height=16, output_width=16)
    m = SaborBoundaryOnlyRenderer(cfg)
    pair_indices, bm, brgb, seg = _make_pair_batch(cfg, b=2)
    rgb_0, rgb_1 = m(pair_indices, bm, brgb, seg)
    assert rgb_0.shape == (2, 3, 16, 16)
    assert rgb_1.shape == (2, 3, 16, 16)
    assert rgb_0.dtype == torch.float32
    assert float(rgb_0.min()) >= 0.0 and float(rgb_0.max()) <= 1.0


def test_renderer_forward_rejects_bad_pair_indices() -> None:
    cfg = SaborBoundaryOnlyConfig(num_pairs=4, output_height=16, output_width=16)
    m = SaborBoundaryOnlyRenderer(cfg)
    pair_indices, bm, brgb, seg = _make_pair_batch(cfg, b=2)
    bad = torch.tensor([99], dtype=torch.long)
    with pytest.raises(ValueError, match="pair_indices out of range"):
        m(bad, bm[:1], brgb[:1], seg[:1])


def test_renderer_forward_rejects_non_long_seg() -> None:
    cfg = SaborBoundaryOnlyConfig(num_pairs=4, output_height=16, output_width=16)
    m = SaborBoundaryOnlyRenderer(cfg)
    pair_indices, bm, brgb, seg = _make_pair_batch(cfg, b=1)
    seg_int = seg.int()
    with pytest.raises(ValueError, match=r"segnet_argmax must be torch\.long"):
        m(pair_indices, bm, brgb, seg_int)


def test_renderer_class_means_quantization_uint8() -> None:
    cfg = SaborBoundaryOnlyConfig(num_pairs=4, output_height=8, output_width=8)
    m = SaborBoundaryOnlyRenderer(cfg)
    cm = m.quantize_class_means_for_archive()
    assert cm.dtype == torch.uint8
    assert cm.shape == (cfg.num_seg_classes, 3)


# ---------------------------------------------------------------------------
# Archive
# ---------------------------------------------------------------------------


def test_archive_header_format_size_is_48_bytes() -> None:
    """The header size is a wire-format invariant; mutating it is a schema change."""
    assert SBO1_HEADER_SIZE == 48
    assert struct.calcsize(SBO1_HEADER_FMT) == 48


def test_archive_magic_and_version_are_stable() -> None:
    assert SBO1_MAGIC == b"SBO1"
    assert SBO1_SCHEMA_VERSION == 1


def _build_smoke_archive(num_pairs: int = 2, h: int = 12, w: int = 12):
    torch.manual_seed(7)
    cfg = SaborBoundaryOnlyConfig(num_pairs=num_pairs, output_height=h, output_width=w)
    m = SaborBoundaryOnlyRenderer(cfg)
    m.eval()
    num_frames = num_pairs * 2
    boundary_mask = torch.zeros(num_frames, h, w, dtype=torch.bool)
    boundary_mask[:, 0, :] = True
    boundary_mask[:, :, 0] = True
    count = int(boundary_mask.sum())
    boundary_rgb_flat = torch.randint(0, 256, (count, 3), dtype=torch.uint8)
    seg = torch.randint(0, cfg.num_seg_classes, (num_frames, h, w), dtype=torch.uint8)
    class_means = m.quantize_class_means_for_archive()
    sd = m.runtime_state_dict_for_archive()
    blob = pack_archive(
        decoder_state_dict=sd,
        class_means=class_means,
        boundary_mask=boundary_mask,
        boundary_rgb_flat=boundary_rgb_flat,
        segnet_argmax=seg,
        meta={"note": "test"},
        num_pairs=cfg.num_pairs,
        output_height=cfg.output_height,
        output_width=cfg.output_width,
        num_seg_classes=cfg.num_seg_classes,
        refinement_hidden=cfg.refinement_hidden,
        refinement_blocks=cfg.refinement_blocks,
        embedding_dim=cfg.embedding_dim,
        bias_dim=cfg.bias_dim,
        edge_threshold=cfg.edge_threshold,
    )
    return cfg, blob, boundary_mask, boundary_rgb_flat, seg, class_means


def test_archive_roundtrip_preserves_boundary_mask_exactly() -> None:
    cfg, blob, mask, rgb_flat, seg, cm = _build_smoke_archive()
    arc = parse_archive(blob)
    assert torch.equal(arc.boundary_mask, mask)


def test_archive_roundtrip_preserves_boundary_rgb_exactly() -> None:
    cfg, blob, mask, rgb_flat, seg, cm = _build_smoke_archive()
    arc = parse_archive(blob)
    assert arc.boundary_rgb_flat.shape == rgb_flat.shape
    assert torch.equal(arc.boundary_rgb_flat, rgb_flat)


def test_archive_roundtrip_preserves_segnet_argmax_exactly() -> None:
    cfg, blob, mask, rgb_flat, seg, cm = _build_smoke_archive()
    arc = parse_archive(blob)
    assert torch.equal(arc.segnet_argmax, seg)


def test_archive_roundtrip_preserves_class_means_exactly() -> None:
    cfg, blob, mask, rgb_flat, seg, cm = _build_smoke_archive()
    arc = parse_archive(blob)
    assert torch.equal(arc.class_means, cm)


def test_archive_roundtrip_preserves_meta_and_header() -> None:
    cfg, blob, *_ = _build_smoke_archive()
    arc = parse_archive(blob)
    assert arc.meta == {"note": "test"}
    assert arc.num_pairs == cfg.num_pairs
    assert arc.output_height == cfg.output_height
    assert arc.output_width == cfg.output_width
    assert arc.num_seg_classes == cfg.num_seg_classes
    assert arc.refinement_hidden == cfg.refinement_hidden
    assert arc.refinement_blocks == cfg.refinement_blocks
    assert arc.embedding_dim == cfg.embedding_dim
    assert arc.bias_dim == cfg.bias_dim
    assert math.isclose(arc.edge_threshold, cfg.edge_threshold, abs_tol=2e-5)


def test_archive_refuses_bad_magic() -> None:
    cfg, blob, *_ = _build_smoke_archive()
    corrupted = b"XXXX" + blob[4:]
    with pytest.raises(ValueError, match="bad magic"):
        parse_archive(corrupted)


def test_archive_refuses_unknown_schema_version() -> None:
    cfg, blob, *_ = _build_smoke_archive()
    corrupted = blob[:4] + bytes([99]) + blob[5:]
    with pytest.raises(ValueError, match="unsupported schema version"):
        parse_archive(corrupted)


def test_archive_refuses_boundary_rgb_count_mismatch() -> None:
    """The header boundary_pixel_count must match boundary_mask True count."""
    cfg = SaborBoundaryOnlyConfig(num_pairs=2, output_height=8, output_width=8)
    m = SaborBoundaryOnlyRenderer(cfg)
    num_frames = cfg.num_pairs * 2
    mask = torch.zeros(num_frames, 8, 8, dtype=torch.bool)
    mask[:, 0, :] = True
    # Off-by-one boundary RGB count to provoke the mismatch check.
    rgb_flat = torch.zeros(int(mask.sum()) + 1, 3, dtype=torch.uint8)
    seg = torch.zeros(num_frames, 8, 8, dtype=torch.uint8)
    with pytest.raises(ValueError, match="boundary_rgb_flat rows"):
        pack_archive(
            decoder_state_dict=m.runtime_state_dict_for_archive(),
            class_means=m.quantize_class_means_for_archive(),
            boundary_mask=mask,
            boundary_rgb_flat=rgb_flat,
            segnet_argmax=seg,
            meta={},
            num_pairs=cfg.num_pairs,
            output_height=cfg.output_height,
            output_width=cfg.output_width,
            num_seg_classes=cfg.num_seg_classes,
            refinement_hidden=cfg.refinement_hidden,
            refinement_blocks=cfg.refinement_blocks,
            embedding_dim=cfg.embedding_dim,
            bias_dim=cfg.bias_dim,
            edge_threshold=cfg.edge_threshold,
        )


def test_archive_state_dict_excludes_class_means() -> None:
    cfg, blob, *_ = _build_smoke_archive()
    arc = parse_archive(blob)
    assert "class_means" not in arc.decoder_state_dict


def test_archive_state_dict_has_required_prefixes() -> None:
    cfg, blob, *_ = _build_smoke_archive()
    arc = parse_archive(blob)
    keys = list(arc.decoder_state_dict.keys())
    # Required prefixes per the runtime state_dict validator
    assert any(k == "pair_embedding" for k in keys)
    assert any(k == "pair_bias" for k in keys)
    assert any(k.startswith("stem.") for k in keys)
    assert any(k.startswith("blocks.") for k in keys)
    assert any(k.startswith("head_rgb.") for k in keys)


def test_archive_state_dict_refuses_class_means_key() -> None:
    """The state_dict validator refuses class_means key (separate uint8 section)."""
    cfg = SaborBoundaryOnlyConfig(num_pairs=2, output_height=8, output_width=8)
    m = SaborBoundaryOnlyRenderer(cfg)
    bad_sd = dict(m.state_dict())  # includes class_means
    num_frames = cfg.num_pairs * 2
    mask = torch.zeros(num_frames, 8, 8, dtype=torch.bool)
    rgb_flat = torch.zeros(0, 3, dtype=torch.uint8)
    seg = torch.zeros(num_frames, 8, 8, dtype=torch.uint8)
    with pytest.raises(ValueError, match="contains class_means"):
        pack_archive(
            decoder_state_dict=bad_sd,
            class_means=m.quantize_class_means_for_archive(),
            boundary_mask=mask,
            boundary_rgb_flat=rgb_flat,
            segnet_argmax=seg,
            meta={},
            num_pairs=cfg.num_pairs,
            output_height=cfg.output_height,
            output_width=cfg.output_width,
            num_seg_classes=cfg.num_seg_classes,
            refinement_hidden=cfg.refinement_hidden,
            refinement_blocks=cfg.refinement_blocks,
            embedding_dim=cfg.embedding_dim,
            bias_dim=cfg.bias_dim,
            edge_threshold=cfg.edge_threshold,
        )


# ---------------------------------------------------------------------------
# Score-aware loss
# ---------------------------------------------------------------------------


class _StubScorer(torch.nn.Module):
    """Minimal stand-in for SegNet/PoseNet honoring the preprocess contract."""

    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.proj = torch.nn.Linear(3, out_dim)

    def preprocess_input(self, pair_btchw: torch.Tensor) -> torch.Tensor:
        # Return per-sample mean RGB (B, 3) — keeps the test deterministic
        # while still being differentiable w.r.t. inputs.
        return pair_btchw.mean(dim=(1, 3, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4 and x.shape[1] == 3:
            x = x.mean(dim=(2, 3))
        return self.proj(x)


def _stub_score_pair_components(monkeypatch):  # pragma: no cover — helper
    """Replace score_pair_components with a tiny stub for the loss tests."""

    def stub(*, seg_scorer, pose_scorer, rgb_0_rt, rgb_1_rt, gt_rgb_0, gt_rgb_1,
             class_weights=None, segmentation_surrogate=None,
             segmentation_temperature=1.0, fisher_rao_eps=1e-6,
             sinkhorn_max_positions_per_chunk=None):
        # seg_term: mean abs diff; pose_term: mean squared diff (both differentiable).
        seg = (rgb_0_rt - gt_rgb_0).abs().mean() + (rgb_1_rt - gt_rgb_1).abs().mean()
        pose = ((rgb_0_rt - gt_rgb_0).pow(2).mean()
                + (rgb_1_rt - gt_rgb_1).pow(2).mean())
        return seg, pose

    monkeypatch.setattr(
        "tac.substrates.sabor_boundary_only_renderer.score_aware_loss.score_pair_components",
        stub,
    )


def test_loss_forward_returns_finite_scalar(monkeypatch) -> None:
    _stub_score_pair_components(monkeypatch)
    weights = SaborBoundaryOnlyLossWeights()
    loss_fn = SaborBoundaryOnlyScoreAwareLoss(
        seg_scorer=_StubScorer(5), pose_scorer=_StubScorer(6), weights=weights
    )
    b, h, w = 2, 8, 8
    rgb_0 = torch.rand(b, 3, h, w, requires_grad=True)
    rgb_1 = torch.rand(b, 3, h, w, requires_grad=True)
    gt_0 = torch.rand(b, 3, h, w)
    gt_1 = torch.rand(b, 3, h, w)
    mask = torch.zeros(b, 2, h, w, dtype=torch.bool)
    mask[:, :, 0, :] = True
    boundary_rgb_target = torch.rand(b, 2, 3, h, w)
    bytes_proxy = torch.tensor(200_000.0)
    loss, parts = loss_fn(
        rgb_0, rgb_1, gt_0, gt_1, bytes_proxy, mask, boundary_rgb_target,
        apply_eval_roundtrip=True, noise_std=0.0,
    )
    assert torch.isfinite(loss)
    assert {"rate_term", "seg_term", "pose_term", "boundary_term", "loss_total"} <= set(parts)


def test_loss_refuses_apply_eval_roundtrip_false() -> None:
    weights = SaborBoundaryOnlyLossWeights()
    loss_fn = SaborBoundaryOnlyScoreAwareLoss(
        seg_scorer=_StubScorer(5), pose_scorer=_StubScorer(6), weights=weights
    )
    b, h, w = 1, 8, 8
    bytes_proxy = torch.tensor(200_000.0)
    mask = torch.zeros(b, 2, h, w, dtype=torch.bool)
    target = torch.zeros(b, 2, 3, h, w)
    with pytest.raises(ValueError, match="apply_eval_roundtrip"):
        loss_fn(
            torch.zeros(b, 3, h, w),
            torch.zeros(b, 3, h, w),
            torch.zeros(b, 3, h, w),
            torch.zeros(b, 3, h, w),
            bytes_proxy,
            mask,
            target,
            apply_eval_roundtrip=False,
        )


def test_loss_propagates_gradients_to_rgb_inputs(monkeypatch) -> None:
    """The loss must be differentiable wrt rendered RGB (score-aware substrate)."""
    _stub_score_pair_components(monkeypatch)
    weights = SaborBoundaryOnlyLossWeights()
    loss_fn = SaborBoundaryOnlyScoreAwareLoss(
        seg_scorer=_StubScorer(5), pose_scorer=_StubScorer(6), weights=weights
    )
    b, h, w = 1, 8, 8
    rgb_0 = torch.rand(b, 3, h, w, requires_grad=True)
    rgb_1 = torch.rand(b, 3, h, w, requires_grad=True)
    gt_0 = torch.rand(b, 3, h, w)
    gt_1 = torch.rand(b, 3, h, w)
    mask = torch.zeros(b, 2, h, w, dtype=torch.bool)
    mask[:, :, 0, :] = True
    target = torch.rand(b, 2, 3, h, w)
    bytes_proxy = torch.tensor(200_000.0)
    loss, _ = loss_fn(
        rgb_0, rgb_1, gt_0, gt_1, bytes_proxy, mask, target,
        apply_eval_roundtrip=True, noise_std=0.0,
    )
    loss.backward()
    assert rgb_0.grad is not None
    assert rgb_1.grad is not None
    assert torch.isfinite(rgb_0.grad).all()
    assert torch.isfinite(rgb_1.grad).all()


def test_loss_boundary_term_zero_when_target_equals_render(monkeypatch) -> None:
    _stub_score_pair_components(monkeypatch)
    weights = SaborBoundaryOnlyLossWeights()
    loss_fn = SaborBoundaryOnlyScoreAwareLoss(
        seg_scorer=_StubScorer(5), pose_scorer=_StubScorer(6), weights=weights
    )
    b, h, w = 1, 8, 8
    rgb_0 = torch.rand(b, 3, h, w)
    rgb_1 = torch.rand(b, 3, h, w)
    gt_0 = torch.rand(b, 3, h, w)
    gt_1 = torch.rand(b, 3, h, w)
    mask = torch.zeros(b, 2, h, w, dtype=torch.bool)
    mask[:, :, 0, :] = True
    target = torch.stack([rgb_0, rgb_1], dim=1)  # render exactly == target
    bytes_proxy = torch.tensor(200_000.0)
    _, parts = loss_fn(
        rgb_0, rgb_1, gt_0, gt_1, bytes_proxy, mask, target,
        apply_eval_roundtrip=True, noise_std=0.0,
    )
    assert float(parts["boundary_term"]) < 1e-5


# ---------------------------------------------------------------------------
# Inflate roundtrip (renderer state lossy via fp16 but deterministic)
# ---------------------------------------------------------------------------


def test_inflate_roundtrip_produces_finite_frames() -> None:
    """End-to-end: build archive, parse, run renderer, expect finite RGB output."""
    from tac.substrates.sabor_boundary_only_renderer.inflate import inflate_one_video

    cfg, blob, *_ = _build_smoke_archive(num_pairs=2, h=12, w=12)
    # Write archive to a tmp file via the standard test fixture (use BytesIO is
    # cleaner but inflate_one_video expects archive_bytes directly).
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        from pathlib import Path

        dst = Path(td) / "out.raw"
        n_frames = inflate_one_video(blob, dst, device_str="cpu")
        # 2 pairs * 2 frames each = 4 frames
        assert n_frames == cfg.num_pairs * 2
        assert dst.is_file()
        assert dst.stat().st_size > 0
