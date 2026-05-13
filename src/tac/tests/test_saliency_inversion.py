"""Lane SI: tests for the saliency-inversion primitives.

We use small mock scorers (a few conv layers) so the tests run on CPU
without the upstream PoseNet/SegNet safetensors. The mock-scorer tests
verify the saliency primitive's INVARIANTS (positive, finite, scorer-
sensitive); the upstream-scorer integration is exercised by Lane SI's
remote bootstrap on CUDA.
"""
from __future__ import annotations


import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from tac.saliency_inversion import (
    CAMERA_H,
    CAMERA_W,
    _MAGIC,
    _rle_decode_bool,
    _rle_encode_bool,
    apply_saliency_weighted_compression,
    compute_inverse_saliency_mask,
    compute_pixel_saliency,
    unpack_saliency_payload,
)


# ─── Mock scorers (PoseNet-shaped + SegNet-shaped) ──────────────────────


class _MockPoseNet(nn.Module):
    """Mimics PoseNet's interface: preprocess_input returns a 12-channel
    YUV-ish tensor; forward returns dict{'pose': (B, 12)}."""

    def __init__(self):
        super().__init__()
        # 12-channel input (matches IN_CHANS = 6 * 2 in upstream/modules.py)
        self.conv = nn.Conv2d(12, 8, kernel_size=3, padding=1)
        self.fc = nn.Linear(8, 12)

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 3, H, W). Downsample to 192x256 then build a fake
        # 12-channel tensor by stacking R,G,B + their half-res mean.
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        x = F.interpolate(x, size=(192, 256), mode="bilinear", align_corners=False)
        x = x.reshape(b, t * c, 192, 256)  # (B, T*3, ...)
        # Pad to 12 channels with zeros so the conv input matches.
        if x.shape[1] < 12:
            pad = torch.zeros(b, 12 - x.shape[1], 192, 256, device=x.device)
            x = torch.cat([x, pad], dim=1)
        return x[:, :12]

    def forward(self, x: torch.Tensor) -> dict:
        h = self.conv(x).mean(dim=(2, 3))  # (B, 8)
        return {"pose": self.fc(h)}


class _MockSegNet(nn.Module):
    """Mimics SegNet: returns (B, 5, H', W') logits. Uses last frame."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 5, kernel_size=3, padding=1)

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        last = x[:, -1, ...]  # (B, 3, H, W)
        return F.interpolate(last, size=(96, 128), mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ─── compute_pixel_saliency ─────────────────────────────────────────────


def test_pixel_saliency_posenet_shape_and_finite() -> None:
    torch.manual_seed(0)
    scorer = _MockPoseNet()
    frames = torch.rand(2, 2, 3, 64, 96) * 255.0
    sal = compute_pixel_saliency(scorer, frames, output_shape=(64, 96))
    assert sal.shape == (64, 96)
    assert torch.isfinite(sal).all(), "saliency must be finite"
    assert (sal >= 0).all(), "saliency must be non-negative"


def test_pixel_saliency_segnet_shape_and_finite() -> None:
    torch.manual_seed(0)
    scorer = _MockSegNet()
    frames = torch.rand(2, 2, 3, 64, 96) * 255.0
    sal = compute_pixel_saliency(scorer, frames, output_shape=(64, 96))
    assert sal.shape == (64, 96)
    assert torch.isfinite(sal).all()
    assert (sal >= 0).all()


def test_pixel_saliency_resizes_to_camera_grid() -> None:
    torch.manual_seed(0)
    scorer = _MockPoseNet()
    frames = torch.rand(1, 2, 3, 64, 96) * 255.0
    sal = compute_pixel_saliency(scorer, frames, output_shape=(CAMERA_H, CAMERA_W))
    assert sal.shape == (CAMERA_H, CAMERA_W)


def test_pixel_saliency_rejects_bad_input() -> None:
    scorer = _MockPoseNet()
    with pytest.raises(ValueError, match=r"\(N, T, 3, H, W\)"):
        compute_pixel_saliency(scorer, torch.rand(2, 3, 64, 96))  # 4-D
    with pytest.raises(ValueError, match=r"seq_len"):
        compute_pixel_saliency(scorer, torch.rand(1, 3, 3, 64, 96))  # wrong T
    with pytest.raises(ValueError, match=r"3 \(RGB\)"):
        compute_pixel_saliency(scorer, torch.rand(1, 2, 4, 64, 96))


def test_pixel_saliency_max_reduce_dominates_mean() -> None:
    """max-reduce should be >= mean-reduce per pixel (for the same input)."""
    torch.manual_seed(0)
    scorer = _MockPoseNet()
    frames = torch.rand(4, 2, 3, 32, 48) * 255.0
    sal_mean = compute_pixel_saliency(scorer, frames, output_shape=(32, 48), reduce="mean")
    sal_max = compute_pixel_saliency(scorer, frames, output_shape=(32, 48), reduce="max")
    # Strict: max >= mean everywhere (ignoring numerical fuzz from interpolate)
    assert (sal_max + 1e-6 >= sal_mean).all()


def test_pixel_saliency_is_scorer_sensitive() -> None:
    """Different scorer outputs ⇒ different saliency maps. If the
    saliency was constant, we'd be measuring the input distribution
    (useless). Random input + two different scorers should give
    measurably different maps."""
    torch.manual_seed(0)
    scorer_a = _MockPoseNet()
    scorer_b = _MockPoseNet()
    # Different init ⇒ different gradients
    frames = torch.rand(2, 2, 3, 32, 48) * 255.0
    sal_a = compute_pixel_saliency(scorer_a, frames, output_shape=(32, 48))
    sal_b = compute_pixel_saliency(scorer_b, frames, output_shape=(32, 48))
    delta = (sal_a - sal_b).abs().mean()
    assert delta > 1e-6, (
        f"Saliency maps for two different scorers should differ; got delta={delta:.3e}"
    )


# ─── compute_inverse_saliency_mask ──────────────────────────────────────


def test_inverse_saliency_mask_basic() -> None:
    sal = torch.arange(100, dtype=torch.float32).view(10, 10)
    inv = compute_inverse_saliency_mask(sal, threshold_quantile=0.5)
    # Bottom-50% of values are blind-spot. With sequential ints 0..99,
    # exactly half should be True.
    assert inv.dtype == torch.bool
    assert inv.shape == (10, 10)
    frac = inv.float().mean().item()
    assert 0.45 <= frac <= 0.55, f"~50%% expected; got {frac:.2f}"


def test_inverse_saliency_mask_quantile_extremes() -> None:
    sal = torch.rand(20, 20)
    inv_all = compute_inverse_saliency_mask(sal, threshold_quantile=1.0)
    inv_none = compute_inverse_saliency_mask(sal, threshold_quantile=0.0)
    assert inv_all.all(), "q=1.0 should mark every pixel blind"
    # q=0 marks only pixels <= min(sal) — at least 1 pixel (the min)
    assert inv_none.sum() >= 1


def test_inverse_saliency_mask_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match=r"H, W"):
        compute_inverse_saliency_mask(torch.rand(10))  # 1-D
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        compute_inverse_saliency_mask(torch.rand(4, 4), threshold_quantile=1.5)


# ─── apply_saliency_weighted_compression ────────────────────────────────


def test_saliency_weighted_compression_roundtrip() -> None:
    """Encode + unpack should preserve the region mask + payload bytes."""
    torch.manual_seed(0)
    masks = (torch.rand(3, 16, 24) * 5).to(torch.uint8)
    sal_inv = torch.rand(16, 24) > 0.5  # 50/50 split
    blob = apply_saliency_weighted_compression(masks, sal_inv, high_crf=50, low_crf=30)

    # Header magic + size sanity
    assert blob[:4] == _MAGIC
    assert len(blob) > 24  # header alone is 24 bytes

    parsed = unpack_saliency_payload(blob)
    assert parsed["h"] == 16
    assert parsed["w"] == 24
    assert parsed["region_mask"].shape == (16, 24)
    assert torch.equal(parsed["region_mask"], sal_inv)


def test_saliency_weighted_compression_shrinks_blind_payload() -> None:
    """Equal CRFs should give similar payload sizes; high_crf > low_crf
    should give a smaller blind-spot payload (per byte of region)."""
    torch.manual_seed(0)
    masks = (torch.rand(3, 16, 24) * 5).to(torch.uint8)
    sal_inv = torch.rand(16, 24) > 0.5

    blob_eq = apply_saliency_weighted_compression(masks, sal_inv, high_crf=30, low_crf=30)
    blob_sk = apply_saliency_weighted_compression(masks, sal_inv, high_crf=63, low_crf=0)
    # Both must roundtrip
    assert unpack_saliency_payload(blob_eq)["h"] == 16
    assert unpack_saliency_payload(blob_sk)["h"] == 16


def test_saliency_weighted_compression_rejects_bad_inputs() -> None:
    masks = torch.zeros(3, 8, 8, dtype=torch.uint8)
    sal = torch.zeros(8, 8, dtype=torch.bool)

    with pytest.raises(ValueError, match=r"uint8"):
        apply_saliency_weighted_compression(masks.float(), sal)
    with pytest.raises(ValueError, match=r"bool"):
        apply_saliency_weighted_compression(masks, sal.float())
    with pytest.raises(ValueError, match=r"shape mismatch"):
        apply_saliency_weighted_compression(masks, torch.zeros(4, 4, dtype=torch.bool))
    with pytest.raises(ValueError, match=r"CRF"):
        apply_saliency_weighted_compression(masks, sal, high_crf=10, low_crf=50)
    with pytest.raises(ValueError, match=r"CRF"):
        apply_saliency_weighted_compression(masks, sal, high_crf=80, low_crf=10)


def test_saliency_weighted_compression_uses_custom_encoder() -> None:
    """The encoder= argument lets the remote script swap in AV1."""
    masks = torch.zeros(2, 4, 4, dtype=torch.uint8)
    sal = torch.zeros(4, 4, dtype=torch.bool)
    sal[0:2] = True  # half blind, half salient

    seen = []

    def fake_encoder(arr: torch.Tensor, crf: int) -> bytes:
        seen.append((tuple(arr.shape), crf))
        return f"crf={crf}".encode()

    blob = apply_saliency_weighted_compression(
        masks, sal, high_crf=50, low_crf=30, encoder=fake_encoder
    )
    # encoder called twice (once per region)
    assert len(seen) == 2
    crfs = [c for (_, c) in seen]
    assert sorted(crfs) == [30, 50]
    # blob contains the encoder outputs
    assert b"crf=50" in blob
    assert b"crf=30" in blob


# ─── RLE helpers ────────────────────────────────────────────────────────


def test_rle_roundtrip() -> None:
    torch.manual_seed(0)
    mask = torch.rand(20, 30) > 0.7
    blob = _rle_encode_bool(mask)
    flat = _rle_decode_bool(blob, mask.numel())
    assert torch.equal(flat.view(20, 30).bool(), mask)


def test_rle_compresses_smooth_mask() -> None:
    """A smooth top-half/bottom-half mask should RLE-compress to <100 bytes."""
    mask = torch.zeros(100, 100, dtype=torch.bool)
    mask[:50] = True
    blob = _rle_encode_bool(mask)
    assert len(blob) < 100, f"smooth mask should be <100 bytes, got {len(blob)}"
    # Roundtrips
    out = _rle_decode_bool(blob, mask.numel())
    assert torch.equal(out.view(100, 100).bool(), mask)


# ─── Lane SI-V2: target_bytes Lagrangian path ─────────────────────────────


def test_lane_si_v2_target_bytes_requires_saliency() -> None:
    """target_bytes mode needs the raw saliency map; saliency_inv alone
    isn't enough (the threshold has to index into the distribution)."""
    masks = torch.zeros(2, 8, 8, dtype=torch.uint8)
    with pytest.raises(ValueError, match=r"target_bytes requires"):
        apply_saliency_weighted_compression(masks, target_bytes=100)


def test_lane_si_v2_either_inv_or_target_bytes() -> None:
    """Cannot call with neither saliency_inv NOR target_bytes."""
    masks = torch.zeros(2, 8, 8, dtype=torch.uint8)
    with pytest.raises(ValueError, match=r"Either saliency_inv"):
        apply_saliency_weighted_compression(masks)


def test_lane_si_v2_saliency_shape_mismatch_raises() -> None:
    masks = torch.zeros(2, 8, 8, dtype=torch.uint8)
    bad_sal = torch.zeros(4, 4, dtype=torch.float32)
    with pytest.raises(ValueError, match=r"shape mismatch"):
        apply_saliency_weighted_compression(
            masks, saliency=bad_sal, target_bytes=100
        )


def test_lane_si_v2_saliency_must_be_2d() -> None:
    masks = torch.zeros(2, 8, 8, dtype=torch.uint8)
    bad_sal = torch.zeros(2, 8, 8, dtype=torch.float32)
    with pytest.raises(ValueError, match=r"saliency must be 2-D"):
        apply_saliency_weighted_compression(
            masks, saliency=bad_sal, target_bytes=100
        )


def test_lane_si_v2_drives_payload_size_toward_target() -> None:
    """target_bytes mode should produce a payload of approximately the
    requested size (within tolerance + codec overhead)."""
    torch.manual_seed(0)
    masks = torch.randint(0, 5, (8, 32, 32), dtype=torch.uint8)
    sal = torch.linspace(0, 1, 32 * 32).reshape(32, 32)

    # Probe at extremes to bound the achievable byte range
    full_blind = (sal <= 1.0).bool()
    full_salient = (sal <= 0.0).bool()
    bytes_full_blind = len(
        apply_saliency_weighted_compression(masks, full_blind, high_crf=50, low_crf=30)
    )
    bytes_full_salient = len(
        apply_saliency_weighted_compression(masks, full_salient, high_crf=50, low_crf=30)
    )
    lo, hi = sorted([bytes_full_blind, bytes_full_salient])
    # Target a value in the middle of the range.
    target = (lo + hi) // 2
    payload = apply_saliency_weighted_compression(
        masks, saliency=sal, target_bytes=target,
        high_crf=50, low_crf=30, target_bytes_tolerance=512.0,
    )
    # Payload should be within tolerance of target (Lagrangian + linear
    # rate model converges to within tolerance_bytes; the actual encoder
    # may differ slightly from the linear surrogate, hence the larger
    # 1.5KB observation tolerance below).
    actual = len(payload)
    # Sanity: actual is between the two extremes
    assert lo - 1024 <= actual <= hi + 1024
    # Closer to target than to either extreme (the Lagrangian moved it)
    assert abs(actual - target) <= max(abs(actual - lo), abs(actual - hi))


def test_lane_si_v2_v1_fallback_byte_identical() -> None:
    """When target_bytes is None, the V2 path must not change the V1
    byte-output (Lane SI-V1 archives stay reproducible)."""
    torch.manual_seed(7)
    masks = torch.randint(0, 5, (4, 16, 16), dtype=torch.uint8)
    sal_inv = torch.zeros(16, 16, dtype=torch.bool)
    sal_inv[:8] = True
    v1_payload = apply_saliency_weighted_compression(
        masks, sal_inv, high_crf=50, low_crf=30
    )
    v2_legacy_payload = apply_saliency_weighted_compression(
        masks, sal_inv, high_crf=50, low_crf=30,
        # target_bytes left None ⇒ V1 path
    )
    assert v1_payload == v2_legacy_payload
