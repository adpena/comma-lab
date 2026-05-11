"""Tests for src.tac.categorical_substrate (categorical full-RGB substrate)."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.tac.categorical_substrate import (
    ARCHIVE_GRAMMAR,
    CAMERA_H,
    CAMERA_W,
    CATEGORICAL_FORMAT_ID,
    CATEGORICAL_FORMAT_VERSION,
    CATEGORICAL_MAGIC,
    CategoricalRenderer,
    CategoricalSubstrateConfig,
    CodebookCollapseError,
    NUM_CLASSES,
    SEGNET_IN_H,
    SEGNET_IN_W,
    _class_entropy,
    _eval_roundtrip_uint8_clamp,
    export_to_archive,
    parse_archive_sections,
    train_step,
)


# ── Constants + grammar ──────────────────────────────────────────────────


def test_constants():
    assert CAMERA_H == 874
    assert CAMERA_W == 1164
    assert SEGNET_IN_H == 384
    assert SEGNET_IN_W == 512
    assert NUM_CLASSES == 5
    assert CATEGORICAL_MAGIC == b"CATG"
    assert CATEGORICAL_FORMAT_ID == 0x51
    assert CATEGORICAL_FORMAT_VERSION == 1


def test_archive_grammar_well_formed():
    assert ARCHIVE_GRAMMAR["format_id"] == CATEGORICAL_FORMAT_ID
    assert ARCHIVE_GRAMMAR["magic"] == "CATG"
    names = [s["name"] for s in ARCHIVE_GRAMMAR["sections"]]
    assert names == ["header", "meta", "renderer_state", "palette", "tokens"]


# ── Config ───────────────────────────────────────────────────────────────


def test_config_defaults_ok():
    cfg = CategoricalSubstrateConfig()
    assert cfg.num_pairs == 600
    assert cfg.num_classes == NUM_CLASSES
    assert cfg.palette_dim == 8
    assert cfg.shading_channels == 16
    assert cfg.cuda_required is True


def test_config_rejects_zero_pairs():
    with pytest.raises(ValueError, match="num_pairs must be positive"):
        CategoricalSubstrateConfig(num_pairs=0)


def test_config_rejects_wrong_num_classes():
    with pytest.raises(ValueError, match="num_classes pinned at 5"):
        CategoricalSubstrateConfig(num_classes=4)


def test_config_rejects_small_palette_dim():
    with pytest.raises(ValueError, match="palette_dim must be >= 3"):
        CategoricalSubstrateConfig(palette_dim=2)


def test_config_rejects_zero_shading_channels():
    with pytest.raises(ValueError, match="shading_channels must be positive"):
        CategoricalSubstrateConfig(shading_channels=0)


def test_config_rejects_negative_collapse_floor():
    with pytest.raises(ValueError, match="codebook_collapse_floor must be >= 0"):
        CategoricalSubstrateConfig(codebook_collapse_floor=-0.1)


def test_config_is_frozen():
    cfg = CategoricalSubstrateConfig()
    with pytest.raises((AttributeError, TypeError)):
        cfg.num_pairs = 100  # type: ignore[misc]


# ── Renderer ─────────────────────────────────────────────────────────────


def test_renderer_forward_shape():
    cfg = CategoricalSubstrateConfig(num_pairs=4)
    renderer = CategoricalRenderer(cfg)
    tokens = torch.zeros(2, SEGNET_IN_H, SEGNET_IN_W, dtype=torch.long)
    idx = torch.tensor([0, 1], dtype=torch.long)
    out = renderer(tokens, idx)
    assert out.shape == (2, 2, 3, CAMERA_H, CAMERA_W)  # (B, 2 frames, RGB, H, W)


def test_renderer_rejects_non_long_tokens():
    cfg = CategoricalSubstrateConfig(num_pairs=4)
    renderer = CategoricalRenderer(cfg)
    tokens = torch.zeros(2, SEGNET_IN_H, SEGNET_IN_W, dtype=torch.float32)
    idx = torch.tensor([0, 1], dtype=torch.long)
    with pytest.raises(TypeError, match="tokens must be long"):
        renderer(tokens, idx)


def test_renderer_rejects_non_long_idx():
    cfg = CategoricalSubstrateConfig(num_pairs=4)
    renderer = CategoricalRenderer(cfg)
    tokens = torch.zeros(2, SEGNET_IN_H, SEGNET_IN_W, dtype=torch.long)
    idx = torch.tensor([0, 1], dtype=torch.float32)
    with pytest.raises(TypeError, match="idx must be long"):
        renderer(tokens, idx)


def test_renderer_rejects_wrong_dim():
    cfg = CategoricalSubstrateConfig(num_pairs=4)
    renderer = CategoricalRenderer(cfg)
    tokens = torch.zeros(2, 5, SEGNET_IN_H, SEGNET_IN_W, dtype=torch.long)
    idx = torch.tensor([0, 1], dtype=torch.long)
    with pytest.raises(ValueError, match=r"tokens must be \(B, H, W\)"):
        renderer(tokens, idx)


def test_renderer_output_pixel_range():
    cfg = CategoricalSubstrateConfig(num_pairs=4)
    renderer = CategoricalRenderer(cfg)
    tokens = torch.randint(0, NUM_CLASSES, (2, SEGNET_IN_H, SEGNET_IN_W), dtype=torch.long)
    idx = torch.tensor([0, 1], dtype=torch.long)
    out = renderer(tokens, idx)
    assert out.min().item() >= 0.0
    assert out.max().item() <= 255.0


def test_renderer_grad_through_tokens_indirect():
    """Although tokens are categorical (no gradient), the renderer is differentiable
    in its OWN weights — this is what we need for joint training."""
    cfg = CategoricalSubstrateConfig(num_pairs=4)
    renderer = CategoricalRenderer(cfg)
    tokens = torch.zeros(1, SEGNET_IN_H, SEGNET_IN_W, dtype=torch.long)
    idx = torch.tensor([0], dtype=torch.long)
    out = renderer(tokens, idx)
    out.sum().backward()
    # All conv weights should have non-None grad
    for name, p in renderer.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"missing grad for {name}"


def test_renderer_param_count_is_compact():
    """Categorical substrate should be smaller than ANR (no master+slave duplication)."""
    cfg = CategoricalSubstrateConfig(num_pairs=600)
    renderer = CategoricalRenderer(cfg)
    total = sum(p.numel() for p in renderer.parameters())
    assert total < 50_000, f"categorical param count {total} too large"


def test_renderer_coord_cache_invalidation():
    """Coord cache rebuilds on H/W change (sanity for multi-resolution support)."""
    cfg = CategoricalSubstrateConfig(num_pairs=4)
    renderer = CategoricalRenderer(cfg)
    tokens1 = torch.zeros(1, SEGNET_IN_H, SEGNET_IN_W, dtype=torch.long)
    idx = torch.tensor([0], dtype=torch.long)
    _ = renderer(tokens1, idx)
    cached_hw1 = renderer._cached_hw
    # Different resolution (artificially smaller to test invalidation)
    tokens2 = torch.zeros(1, 32, 32, dtype=torch.long)
    _ = renderer(tokens2, idx)
    assert renderer._cached_hw != cached_hw1


# ── Class entropy + codebook-collapse guard ──────────────────────────────


def test_class_entropy_uniform_max():
    """Uniform distribution over K classes → entropy = ln(K)."""
    import math
    K = 5
    # Construct tokens with equal counts per class
    tokens = torch.tensor([0, 1, 2, 3, 4] * 100, dtype=torch.long)
    entropy = _class_entropy(tokens, K)
    assert abs(entropy.item() - math.log(K)) < 1e-4


def test_class_entropy_single_class_zero():
    """All one class → entropy = 0."""
    tokens = torch.zeros(100, dtype=torch.long)
    entropy = _class_entropy(tokens, 5)
    # Approaches zero (we clamp log(0) → -inf*0 = clamp_safe)
    assert entropy.item() < 1e-6


def test_class_entropy_empty_raises():
    with pytest.raises(ValueError, match="cannot compute entropy"):
        _class_entropy(torch.zeros(0, dtype=torch.long), 5)


def test_codebook_collapse_error_inherits_runtime_error():
    err = CodebookCollapseError("test")
    assert isinstance(err, RuntimeError)


# ── Archive export / parse ───────────────────────────────────────────────


def test_export_archive_roundtrip():
    cfg = CategoricalSubstrateConfig(num_pairs=4)
    renderer = CategoricalRenderer(cfg)
    tokens_bin = b"\xAB" * 100

    blob, sha = export_to_archive(
        config=cfg, renderer=renderer, tokens_bin=tokens_bin,
    )
    assert isinstance(blob, bytes)
    assert len(sha) == 64

    sections = parse_archive_sections(blob)
    assert sections["_header"]["magic"] == CATEGORICAL_MAGIC
    assert sections["_header"]["format_id"] == CATEGORICAL_FORMAT_ID
    assert sections["_header"]["num_pairs"] == 4
    assert sections["tokens"] == tokens_bin


def test_export_determinism():
    cfg = CategoricalSubstrateConfig(num_pairs=4)
    torch.manual_seed(42)
    renderer = CategoricalRenderer(cfg)
    blob1, sha1 = export_to_archive(config=cfg, renderer=renderer, tokens_bin=b"x" * 50)
    blob2, sha2 = export_to_archive(config=cfg, renderer=renderer, tokens_bin=b"x" * 50)
    assert blob1 == blob2
    assert sha1 == sha2


def test_export_rejects_non_bytes_tokens():
    cfg = CategoricalSubstrateConfig(num_pairs=4)
    renderer = CategoricalRenderer(cfg)
    with pytest.raises(TypeError, match="tokens_bin must be bytes-like"):
        export_to_archive(config=cfg, renderer=renderer, tokens_bin="bad")  # type: ignore[arg-type]


def test_parse_rejects_short_blob():
    with pytest.raises(ValueError, match="archive too short"):
        parse_archive_sections(b"\x00" * 4)


def test_parse_rejects_bad_magic():
    import struct as _struct
    bad_header = _struct.pack(
        "<4sHHII", b"XXXX", CATEGORICAL_FORMAT_ID,
        CATEGORICAL_FORMAT_VERSION, 4, 0,
    )
    blob = bad_header + b"\x00\x00\x00\x00" * 4
    with pytest.raises(ValueError, match="magic mismatch"):
        parse_archive_sections(blob)


def test_parse_rejects_bad_format_id():
    import struct as _struct
    bad = _struct.pack(
        "<4sHHII", CATEGORICAL_MAGIC, 0x99,
        CATEGORICAL_FORMAT_VERSION, 4, 0,
    )
    blob = bad + b"\x00\x00\x00\x00" * 4
    with pytest.raises(ValueError, match="format_id mismatch"):
        parse_archive_sections(blob)


def test_parse_rejects_truncated_section():
    import struct as _struct
    header = _struct.pack(
        "<4sHHII", CATEGORICAL_MAGIC, CATEGORICAL_FORMAT_ID,
        CATEGORICAL_FORMAT_VERSION, 4, 0,
    )
    blob = header + _struct.pack("<I", 9999) + b"\x00"
    with pytest.raises(ValueError, match="archive truncated inside section"):
        parse_archive_sections(blob)


def test_parse_rejects_trailing_bytes():
    """Roundtrip then append junk → trailing-byte detection."""
    cfg = CategoricalSubstrateConfig(num_pairs=2)
    renderer = CategoricalRenderer(cfg)
    blob, _ = export_to_archive(
        config=cfg, renderer=renderer, tokens_bin=b"x" * 10,
    )
    with pytest.raises(ValueError, match="trailing bytes"):
        parse_archive_sections(blob + b"\x00\x00")


# ── Eval roundtrip helper ────────────────────────────────────────────────


def test_eval_roundtrip_clamps():
    # torch uses banker's rounding (round-half-to-even); 100.5 → 100.
    rgb = torch.tensor([[-5.0, 100.6, 256.0]])
    out = _eval_roundtrip_uint8_clamp(rgb)
    assert torch.allclose(out, torch.tensor([[0.0, 101.0, 255.0]]))


def test_eval_roundtrip_gradient_flows():
    rgb = torch.tensor([[50.0, 100.0]], requires_grad=True)
    out = _eval_roundtrip_uint8_clamp(rgb)
    out.sum().backward()
    assert torch.allclose(rgb.grad, torch.tensor([[1.0, 1.0]]))


# ── train_step contract ─────────────────────────────────────────────────


class _MockSegNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3 * 384 * 512, 5 * 384 * 512)

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        from torch.nn import functional as F_
        last = x[:, -1, ...]
        return F_.interpolate(last, size=(384, 512), mode="bilinear",
                              align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        return self.linear(x.reshape(B, -1)).reshape(B, 5, 384, 512)


class _MockPoseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3 * 384 * 512 * 2, 6)

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        from torch.nn import functional as F_
        B, F_pp, C, H, W = x.shape
        flat = x.reshape(B * F_pp, C, H, W)
        resized = F_.interpolate(flat, size=(384, 512), mode="bilinear",
                                  align_corners=False)
        return resized.reshape(B, F_pp, C, 384, 512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        return self.linear(x.reshape(B, -1))


def test_train_step_rejects_eval_roundtrip_false():
    cfg = CategoricalSubstrateConfig(num_pairs=2)
    renderer = CategoricalRenderer(cfg)
    tokens = torch.randint(0, 5, (1, SEGNET_IN_H, SEGNET_IN_W), dtype=torch.long)
    gt = torch.zeros(1, 2, 3, CAMERA_H, CAMERA_W)
    idx = torch.tensor([0], dtype=torch.long)
    with pytest.raises(ValueError, match="eval_roundtrip=False is forbidden"):
        train_step(
            renderer=renderer, pair_indices=idx, tokens=tokens, gt_pairs_uint8=gt,
            scorer_seg=_MockSegNet(), scorer_pose=_MockPoseNet(),
            seg_surrogate=lambda a, b: (a - b).pow(2).mean(),
            pose_surrogate=lambda a, b: (a - b).pow(2).mean(),
            lambda_seg=1.0, lambda_pose=1.0,
            codebook_collapse_floor=0.4, eval_roundtrip=False,
        )


def test_train_step_codebook_collapse_guard_fires():
    """All-zeros tokens → entropy near 0 → CodebookCollapseError raised."""
    cfg = CategoricalSubstrateConfig(num_pairs=2)
    renderer = CategoricalRenderer(cfg)
    tokens = torch.zeros(1, SEGNET_IN_H, SEGNET_IN_W, dtype=torch.long)  # all class 0
    gt = torch.zeros(1, 2, 3, CAMERA_H, CAMERA_W)
    idx = torch.tensor([0], dtype=torch.long)
    with pytest.raises(CodebookCollapseError, match="Class-entropy"):
        train_step(
            renderer=renderer, pair_indices=idx, tokens=tokens, gt_pairs_uint8=gt,
            scorer_seg=_MockSegNet(), scorer_pose=_MockPoseNet(),
            seg_surrogate=lambda a, b: (a - b).pow(2).mean(),
            pose_surrogate=lambda a, b: (a - b).pow(2).mean(),
            lambda_seg=1.0, lambda_pose=1.0,
            codebook_collapse_floor=0.4,
        )


def test_train_step_codebook_collapse_passes_with_diverse_tokens():
    """Uniform tokens → high entropy → no collapse error."""
    cfg = CategoricalSubstrateConfig(num_pairs=2)
    renderer = CategoricalRenderer(cfg)
    tokens = torch.randint(0, 5, (1, SEGNET_IN_H, SEGNET_IN_W), dtype=torch.long)
    gt = torch.zeros(1, 2, 3, CAMERA_H, CAMERA_W)
    idx = torch.tensor([0], dtype=torch.long)
    result = train_step(
        renderer=renderer, pair_indices=idx, tokens=tokens, gt_pairs_uint8=gt,
        scorer_seg=_MockSegNet(), scorer_pose=_MockPoseNet(),
        seg_surrogate=lambda a, b: (a - b).pow(2).mean(),
        pose_surrogate=lambda a, b: (a - b).pow(2).mean(),
        lambda_seg=1.0, lambda_pose=1.0,
        codebook_collapse_floor=0.4,
    )
    assert "class_entropy" in result
    assert result["class_entropy"].item() > 0.4


def test_train_step_rejects_wrong_token_shape():
    cfg = CategoricalSubstrateConfig(num_pairs=2)
    renderer = CategoricalRenderer(cfg)
    tokens = torch.randint(0, 5, (1, 100, 100), dtype=torch.long)
    gt = torch.zeros(1, 2, 3, CAMERA_H, CAMERA_W)
    idx = torch.tensor([0], dtype=torch.long)
    with pytest.raises(ValueError, match="tokens spatial shape"):
        train_step(
            renderer=renderer, pair_indices=idx, tokens=tokens, gt_pairs_uint8=gt,
            scorer_seg=_MockSegNet(), scorer_pose=_MockPoseNet(),
            seg_surrogate=lambda a, b: (a - b).pow(2).mean(),
            pose_surrogate=lambda a, b: (a - b).pow(2).mean(),
            lambda_seg=1.0, lambda_pose=1.0,
            codebook_collapse_floor=0.4,
        )


def test_train_step_returns_diffable_loss():
    torch.manual_seed(0)
    cfg = CategoricalSubstrateConfig(num_pairs=2)
    renderer = CategoricalRenderer(cfg)
    tokens = torch.randint(0, 5, (1, SEGNET_IN_H, SEGNET_IN_W), dtype=torch.long)
    gt = torch.full((1, 2, 3, CAMERA_H, CAMERA_W), 128.0)
    idx = torch.tensor([0], dtype=torch.long)
    result = train_step(
        renderer=renderer, pair_indices=idx, tokens=tokens, gt_pairs_uint8=gt,
        scorer_seg=_MockSegNet(), scorer_pose=_MockPoseNet(),
        seg_surrogate=lambda a, b: (a - b).pow(2).mean(),
        pose_surrogate=lambda a, b: (a - b).pow(2).mean(),
        lambda_seg=1.0, lambda_pose=1.0,
        codebook_collapse_floor=0.4,
    )
    assert result["loss"].grad_fn is not None
    result["loss"].backward()


# ── HNeRV parity ────────────────────────────────────────────────────────


def test_lesson5_full_rgb_to_camera_resolution():
    cfg = CategoricalSubstrateConfig(num_pairs=2)
    renderer = CategoricalRenderer(cfg)
    tokens = torch.zeros(1, SEGNET_IN_H, SEGNET_IN_W, dtype=torch.long)
    idx = torch.tensor([0], dtype=torch.long)
    out = renderer(tokens, idx)
    assert out.shape[-2:] == (CAMERA_H, CAMERA_W)


def test_lesson3_monolithic_single_blob():
    cfg = CategoricalSubstrateConfig(num_pairs=2)
    renderer = CategoricalRenderer(cfg)
    blob, _ = export_to_archive(config=cfg, renderer=renderer, tokens_bin=b"\x00" * 10)
    assert isinstance(blob, bytes)


def test_lesson11_sha_changes_on_input_change():
    cfg = CategoricalSubstrateConfig(num_pairs=2)
    renderer = CategoricalRenderer(cfg)
    _, sha_a = export_to_archive(config=cfg, renderer=renderer, tokens_bin=b"AAA")
    _, sha_b = export_to_archive(config=cfg, renderer=renderer, tokens_bin=b"BBB")
    assert sha_a != sha_b


def test_no_mps_fallback_in_config():
    cfg = CategoricalSubstrateConfig()
    assert cfg.cuda_required is True
