"""Tests for ``tac.foveation_field`` — telescopic foveation field sidecar.

Per CLAUDE.md HNeRV parity discipline lessons 8 (eval_roundtrip-aware),
10 (no-op detector), 11 (no-op detector enforced), and the "Bugs must be
permanently fixed AND self-protected against" non-negotiable. These tests
cover encode/decode round-trip, byte-budget enforcement, no-op detection,
warp differentiability, and the MPS-rejection rule.
"""
from __future__ import annotations


import pytest
import torch

from tac.foveation_field import (
    DEFAULT_DELTA_SCALE,
    DEFAULT_N_GAUSS,
    FOVEATION_FIELD_FORMAT_ID,
    FOVEATION_FIELD_MAGIC,
    MAX_ENCODED_BYTES,
    NO_OP_DISPLACEMENT_THRESHOLD,
    FoveationField,
    compute_foveation_byte_budget,
    compute_foveation_warp,
    decode_foveation_field,
    encode_foveation_field,
    is_no_op,
)


def _make_field(n_frames: int = 8, n_gauss: int = DEFAULT_N_GAUSS, seed: int = 42) -> FoveationField:
    g = torch.Generator().manual_seed(seed)
    centers = 0.4 + 0.2 * torch.rand(n_frames, n_gauss, 2, generator=g)  # in (0.4, 0.6)²
    log_sigma = torch.log(torch.full((n_frames, n_gauss), 0.1))
    log_amp = torch.log(torch.full((n_frames, n_gauss), 0.02))
    return FoveationField(centers=centers, log_sigma=log_sigma, log_amp=log_amp)


def _make_no_op_field(n_frames: int = 4) -> FoveationField:
    """Build a field with effectively zero amplitude → no displacement."""
    centers = torch.full((n_frames, 2, 2), 0.5)
    log_sigma = torch.log(torch.full((n_frames, 2), 0.1))
    log_amp = torch.log(torch.full((n_frames, 2), 1e-8))  # tiny amp
    return FoveationField(centers=centers, log_sigma=log_sigma, log_amp=log_amp)


def test_field_validate_rejects_bad_shapes():
    bad = FoveationField(
        centers=torch.zeros(3, 2),  # missing trailing dim
        log_sigma=torch.zeros(3, 2),
        log_amp=torch.zeros(3, 2),
    )
    with pytest.raises(ValueError, match="centers must have shape"):
        bad.validate()


def test_field_validate_rejects_nan_inputs():
    field = _make_field()
    field.centers[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        field.validate()


def test_field_validate_rejects_shape_mismatch():
    bad = FoveationField(
        centers=torch.zeros(3, 4, 2),
        log_sigma=torch.zeros(3, 5),  # wrong G
        log_amp=torch.zeros(3, 4),
    )
    with pytest.raises(ValueError, match="log_sigma shape mismatch"):
        bad.validate()


def test_encode_decode_round_trip_small():
    """Verify encode → decode reconstructs the field within int8 quant error."""
    field = _make_field(n_frames=8, n_gauss=2)
    blob = encode_foveation_field(field, delta_scale=DEFAULT_DELTA_SCALE)
    decoded = decode_foveation_field(blob)
    assert decoded.n_frames == field.n_frames
    assert decoded.n_gauss == field.n_gauss
    # Anchor (frame 0) must be exact within fp16 precision.
    centers_err_0 = (decoded.centers[0] - field.centers[0]).abs().max().item()
    assert centers_err_0 < 1e-3, f"frame-0 centers fp16 error too large: {centers_err_0}"
    # Subsequent frames bounded by int8 delta quant error (cumulative).
    max_err = (decoded.centers - field.centers).abs().max().item()
    # Per-frame quant error is bounded by delta_scale; cumulative over n_frames-1.
    bound = DEFAULT_DELTA_SCALE * (field.n_frames - 1) * 1.5  # 1.5 = safety
    assert max_err < bound, f"cumulative quant error {max_err} > bound {bound}"


def test_encode_decode_round_trip_single_frame():
    """Single-frame field (no deltas) round-trips cleanly via fp16 anchor only."""
    field = _make_field(n_frames=1, n_gauss=2)
    blob = encode_foveation_field(field)
    decoded = decode_foveation_field(blob)
    assert decoded.n_frames == 1
    assert (decoded.centers - field.centers).abs().max().item() < 1e-3


def test_encode_blob_starts_with_magic_and_format_id():
    field = _make_field(n_frames=4, n_gauss=2)
    blob = encode_foveation_field(field)
    assert blob[0] == FOVEATION_FIELD_MAGIC
    assert blob[1] == FOVEATION_FIELD_FORMAT_ID


def test_encode_enforces_budget():
    """Massive field overshoots ≤ 500 B budget and must raise."""
    big = _make_field(n_frames=1200, n_gauss=8)
    with pytest.raises(ValueError, match="exceeds|> budget|> 500"):
        encode_foveation_field(big, enforce_budget=True)


def test_encode_enforces_budget_disabled_returns_oversized_blob():
    big = _make_field(n_frames=1200, n_gauss=8)
    blob = encode_foveation_field(big, enforce_budget=False)
    assert len(blob) > 0
    # Decoding the oversized blob still works.
    decoded = decode_foveation_field(blob)
    assert decoded.n_frames == 1200


def test_decode_rejects_bad_magic():
    field = _make_field()
    blob = bytearray(encode_foveation_field(field))
    blob[0] = 0x00  # corrupt magic
    with pytest.raises(ValueError, match="magic mismatch"):
        decode_foveation_field(bytes(blob))


def test_decode_rejects_bad_format_id():
    field = _make_field()
    blob = bytearray(encode_foveation_field(field))
    blob[1] = 0x99  # wrong format_id
    with pytest.raises(ValueError, match="format_id mismatch"):
        decode_foveation_field(bytes(blob))


def test_decode_rejects_truncated_blob():
    field = _make_field()
    blob = encode_foveation_field(field)
    truncated = blob[:5]
    with pytest.raises(ValueError):
        decode_foveation_field(truncated)


def test_decode_rejects_trailing_bytes():
    field = _make_field()
    blob = encode_foveation_field(field) + b"\xde\xad"
    with pytest.raises(ValueError, match="trailing bytes"):
        decode_foveation_field(blob)


def test_warp_identity_when_no_op_field():
    """Near-zero amplitude → warped ~ input (bilinear half-pixel shift only).

    Bilinear grid_sample with ``align_corners=False`` introduces a deterministic
    half-pixel offset between source and target grids — even when the warp is
    the identity, the output is the source linearly interpolated at the
    pixel-centre grid, which differs from the source values at the pixel
    corners by < 1 sample. We bound the warp error by the per-pixel range of
    the input image (max grad between neighbour pixels).
    """
    torch.manual_seed(7)
    field = _make_no_op_field(n_frames=2)
    # Slowly varying image so the bilinear half-pixel offset is small.
    h, w = 32, 48
    decoded_rgb = (
        torch.linspace(0, 1, h).unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(2, 3, h, w)
        * 50.0
        + 128.0
    ).contiguous()
    warped = compute_foveation_warp(field, decoded_rgb)
    assert warped.shape == decoded_rgb.shape
    # Per-pixel slope ~ 50/32 = 1.56; half-pixel shift bounded by ~ slope.
    err = (warped - decoded_rgb).abs().max().item()
    assert err < 2.5, f"no-op field caused large warp error: {err}"


def test_warp_changes_output_when_active_field():
    """Active field must produce measurable warp on a non-uniform image."""
    field = _make_field(n_frames=2, n_gauss=2)
    # Use larger amp to ensure visible displacement.
    field.log_amp = torch.log(torch.full_like(field.log_amp, 0.5))
    # Non-uniform image so the warp shifts pixels.
    decoded_rgb = torch.arange(0, 2 * 3 * 32 * 48, dtype=torch.float32).reshape(2, 3, 32, 48)
    warped = compute_foveation_warp(field, decoded_rgb)
    diff = (warped - decoded_rgb).abs().mean().item()
    assert diff > 1.0, f"active warp should change image; got mean diff {diff}"


def test_warp_is_differentiable():
    """grid_sample-based warp must allow gradient flow into field params."""
    field = _make_field(n_frames=2, n_gauss=2)
    field.centers = field.centers.detach().requires_grad_(True)
    field.log_sigma = field.log_sigma.detach().requires_grad_(True)
    field.log_amp = field.log_amp.detach().requires_grad_(True)
    decoded_rgb = torch.rand(2, 3, 16, 24, requires_grad=False) * 255.0
    warped = compute_foveation_warp(field, decoded_rgb)
    loss = warped.mean()
    loss.backward()
    assert field.centers.grad is not None
    assert field.log_sigma.grad is not None
    assert field.log_amp.grad is not None
    # At least one gradient component must be non-zero.
    assert field.centers.grad.abs().sum().item() > 0.0


def test_warp_rejects_mps_device():
    field = _make_field(n_frames=1, n_gauss=1)
    decoded_rgb = torch.rand(1, 3, 16, 16)
    with pytest.raises(ValueError, match="MPS is forbidden"):
        compute_foveation_warp(field, decoded_rgb, device="mps")


def test_warp_rejects_wrong_n_frames():
    field = _make_field(n_frames=3, n_gauss=2)
    decoded_rgb = torch.rand(4, 3, 16, 16)  # mismatched T
    with pytest.raises(ValueError, match="n_frames"):
        compute_foveation_warp(field, decoded_rgb)


def test_warp_rejects_wrong_rgb_shape():
    field = _make_field(n_frames=2, n_gauss=2)
    decoded_rgb = torch.rand(2, 4, 16, 16)  # wrong channel count
    with pytest.raises(ValueError, match="\\(T, 3, H, W\\)"):
        compute_foveation_warp(field, decoded_rgb)


def test_compute_byte_budget_matches_encode_length():
    field = _make_field(n_frames=8, n_gauss=2)
    budget = compute_foveation_byte_budget(field)
    blob = encode_foveation_field(field, enforce_budget=False)
    assert budget == len(blob)


def test_byte_budget_under_500_for_typical_config():
    """Per CLAUDE.md / operator constraint: 600-pair-friendly tiny field fits."""
    field = _make_field(n_frames=8, n_gauss=2)
    blob = encode_foveation_field(field)
    assert len(blob) <= MAX_ENCODED_BYTES


def test_is_no_op_detects_zero_amp_field():
    """Field with effectively zero amp must be flagged no-op."""
    field = _make_no_op_field()
    assert is_no_op(field)


def test_is_no_op_returns_false_for_active_field():
    field = _make_field(n_frames=2, n_gauss=4)
    field.log_amp = torch.log(torch.full_like(field.log_amp, 0.5))
    assert not is_no_op(field)


def test_no_op_threshold_constant_is_small():
    """Sanity-check the no-op threshold is small enough to trigger on real
    no-op fields but not on legitimate small warps."""
    assert NO_OP_DISPLACEMENT_THRESHOLD > 0
    assert NO_OP_DISPLACEMENT_THRESHOLD < 1e-2
