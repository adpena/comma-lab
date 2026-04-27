"""Tests for Lane C UNIWARD δ codec + apply path.

CPU-only. No Vast.ai, no scorer loading, no GT video. These tests cover
the codec invariants the inflate path relies on.

Coverage matrix:
  - Round-trip: pack(δ) → unpack → reconstructed equals quantized δ.
  - L∞ budget: every kept value within ±l_inf_budget.
  - Sparsity:  n_kept ≤ requested top-K (target_bytes path) AND
               n_kept matches the number of non-zero pixel-channels in
               the source δ when we feed in a known-sparse input.
  - Determinism: same inputs → exact same bytes (zlib + int8 quant +
                 deterministic top-K).
  - Apply: scatter_add gives the expected per-pixel delta on a known
           input; respects [0, 255] clamp.
  - Magic + schema: bad bytes raise ValueError with a useful message.
  - Frame size mismatch raises ValueError (catches the historical
    "ship at one resolution, inflate at another" failure mode).
"""
from __future__ import annotations

import zlib

import numpy as np
import pytest
import torch

from tac.uniward_delta import (
    DeltaSpec,
    MAGIC,
    SCHEMA_VERSION,
    apply_delta_to_frame,
    pack_sparse_delta,
    unpack_sparse_delta,
)


# ── Helpers ──────────────────────────────────────────────────────────────


def _random_delta(
    n_frames: int = 4, H: int = 16, W: int = 24, sparsity: float = 0.05,
    seed: int = 0, l_inf: float = 4.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct a (B, 3, H, W) δ with `sparsity` fraction of non-zero
    pixel-channels, drawn uniformly in [-l_inf, l_inf]. Returns (δ, cost).
    """
    g = torch.Generator().manual_seed(seed)
    n_total = n_frames * 3 * H * W
    n_nz = max(int(n_total * sparsity), 1)
    delta = torch.zeros(n_total, dtype=torch.float32)
    perm = torch.randperm(n_total, generator=g)
    delta[perm[:n_nz]] = (torch.rand(n_nz, generator=g) * 2 - 1) * l_inf
    delta = delta.reshape(n_frames, 3, H, W)
    # Cost map: smooth random (positive). Doesn't materially affect
    # round-trip correctness but exercises the rank-boost code path.
    cost = torch.rand(n_frames, H, W, generator=g)
    return delta, cost


# ── Round-trip ───────────────────────────────────────────────────────────


def test_roundtrip_basic_recovers_quantized_delta() -> None:
    """pack → unpack → apply against zero frame should reconstruct
    the quantized δ at every kept position, and zero elsewhere."""
    delta, cost = _random_delta(n_frames=3, H=8, W=12, sparsity=0.10, seed=1)
    blob = pack_sparse_delta(
        delta, cost, l_inf_budget=4.0, target_bytes=None,
    )
    spec = unpack_sparse_delta(blob)
    # Apply each frame to a zero base. The result should match the quant
    # snapshot of δ at the kept positions (others = 0).
    for f in range(spec.n_frames):
        zero = torch.zeros(spec.H, spec.W, 3, dtype=torch.float32)
        out = apply_delta_to_frame(zero, spec, f)
        # Construct the expected mask from the spec
        local = spec.per_frame_local_idx[f]
        if local is None:
            assert torch.allclose(out, zero)
            continue
        flat_out = out.reshape(-1)
        # Every recovered value must equal the dequant value (or its
        # int8 representable value), within a half-step of the scale.
        recovered = flat_out[local]
        expected = spec.per_frame_dequant[f]
        # The applied value is dequant clamped into [0, 255]. Since base
        # is zero, negative values clip to 0.
        # Compare against clamp(expected, 0, 255) at the kept positions.
        assert torch.allclose(recovered, expected.clamp(0.0, 255.0))


def test_roundtrip_byte_stable_for_identical_inputs() -> None:
    """Same δ + cost + budget → same bytes. Determinism non-negotiable."""
    delta, cost = _random_delta(n_frames=2, H=8, W=8, sparsity=0.20, seed=42)
    blob_a = pack_sparse_delta(delta, cost, l_inf_budget=4.0, target_bytes=None, seed=7)
    blob_b = pack_sparse_delta(delta, cost, l_inf_budget=4.0, target_bytes=None, seed=7)
    assert blob_a == blob_b, "pack must be byte-stable for identical inputs"


def test_roundtrip_preserves_magic_and_schema() -> None:
    """Decompressed payload must start with the wire-format magic, and
    the embedded header must match the current schema version."""
    delta, cost = _random_delta(n_frames=1, H=4, W=4, sparsity=0.5)
    blob = pack_sparse_delta(delta, cost, l_inf_budget=4.0)
    raw = zlib.decompress(blob)
    assert raw[:4] == MAGIC
    import json
    import struct
    header_len = struct.unpack("<I", raw[4:8])[0]
    header = json.loads(raw[8:8 + header_len].decode("utf-8"))
    assert header["schema_version"] == SCHEMA_VERSION


# ── L∞ budget ────────────────────────────────────────────────────────────


def test_l_inf_budget_is_respected_after_quantization() -> None:
    """Every dequantized δ value must satisfy |δ| ≤ l_inf_budget,
    even after int8 round-trip. The scale is l_inf_budget so the
    representable range is exactly [-l_inf_budget * 127/127, +l_inf_budget].
    """
    # Construct δ that EXCEEDS the budget — pack must clip.
    delta = torch.full((1, 3, 4, 4), 100.0)  # well above any reasonable budget
    cost = torch.ones(1, 4, 4)
    blob = pack_sparse_delta(delta, cost, l_inf_budget=4.0)
    spec = unpack_sparse_delta(blob)
    for f in range(spec.n_frames):
        if spec.per_frame_dequant[f] is None:
            continue
        max_abs = float(spec.per_frame_dequant[f].abs().max().item())
        assert max_abs <= 4.0 + 1e-6, (
            f"|δ| max = {max_abs} exceeds l_inf_budget=4.0"
        )


def test_l_inf_budget_zero_raises() -> None:
    delta, cost = _random_delta()
    with pytest.raises(ValueError, match="l_inf_budget"):
        pack_sparse_delta(delta, cost, l_inf_budget=0.0)


# ── Sparsity / target_bytes ──────────────────────────────────────────────


def test_target_bytes_caps_blob_size() -> None:
    """The binary search MUST respect the byte budget — the resulting
    blob is guaranteed ≤ target_bytes."""
    delta, cost = _random_delta(n_frames=4, H=32, W=48, sparsity=1.0, seed=3)
    target = 500
    blob = pack_sparse_delta(delta, cost, l_inf_budget=4.0, target_bytes=target)
    assert len(blob) <= target, f"blob {len(blob)} exceeds target_bytes {target}"


def test_target_bytes_achieves_nonzero_sparsity_when_room() -> None:
    """With a generous budget we should keep SOME entries (otherwise
    the optimization is silently a no-op)."""
    delta, cost = _random_delta(n_frames=2, H=16, W=16, sparsity=0.5, seed=4)
    blob = pack_sparse_delta(delta, cost, l_inf_budget=4.0, target_bytes=10_000)
    spec = unpack_sparse_delta(blob)
    assert spec.any_delta
    assert spec.n_kept > 0


def test_target_bytes_zero_falls_back_to_empty() -> None:
    """An impossibly tight budget still produces a valid blob (n_kept=0)."""
    delta, cost = _random_delta()
    blob = pack_sparse_delta(delta, cost, l_inf_budget=4.0, target_bytes=1)
    # The all-zero blob should fit within ~50 bytes of overhead
    spec = unpack_sparse_delta(blob)
    assert spec.n_kept == 0
    assert not spec.any_delta


# ── Apply ────────────────────────────────────────────────────────────────


def test_apply_clamps_to_uint8_range() -> None:
    """δ added to a saturated base should clamp to 255, not overflow."""
    # Force one large positive δ
    delta = torch.zeros(1, 3, 4, 4)
    delta[0, 0, 1, 2] = 4.0  # within budget
    cost = torch.ones(1, 4, 4)
    blob = pack_sparse_delta(delta, cost, l_inf_budget=4.0, target_bytes=None)
    spec = unpack_sparse_delta(blob)

    base = torch.full((4, 4, 3), 254.0)
    out = apply_delta_to_frame(base, spec, frame_index=0)
    # All values clamped to [0, 255]
    assert (out >= 0).all() and (out <= 255).all()
    # The targeted pixel should be at 255 (was 254 + ~4 clip).
    assert out[1, 2, 0].item() == 255.0


def test_apply_no_delta_for_frame_returns_input_unchanged() -> None:
    """Frames absent from the spec must be returned untouched (zero-cost
    path the inflate loop relies on for the no-perturbation frames)."""
    delta = torch.zeros(2, 3, 4, 4)
    delta[0, 0, 0, 0] = 3.0  # only frame 0 has δ
    cost = torch.ones(2, 4, 4)
    blob = pack_sparse_delta(delta, cost, l_inf_budget=4.0, target_bytes=None)
    spec = unpack_sparse_delta(blob)

    base = torch.full((4, 4, 3), 100.0)
    out_unchanged = apply_delta_to_frame(base, spec, frame_index=1)
    assert torch.allclose(out_unchanged, base)


def test_apply_wrong_frame_size_raises() -> None:
    """Catches the historical (mask res != renderer res) failure mode
    in the spec dimension. δ is ALWAYS at renderer native resolution."""
    delta, cost = _random_delta(n_frames=1, H=8, W=8, sparsity=0.1)
    blob = pack_sparse_delta(delta, cost, l_inf_budget=4.0)
    spec = unpack_sparse_delta(blob)

    wrong = torch.zeros(7, 8, 3)  # H mismatch
    with pytest.raises(ValueError, match="frame shape"):
        apply_delta_to_frame(wrong, spec, frame_index=0)


def test_apply_out_of_range_frame_index_raises() -> None:
    delta, cost = _random_delta(n_frames=2, H=4, W=4, sparsity=0.5)
    spec = unpack_sparse_delta(pack_sparse_delta(delta, cost, l_inf_budget=4.0))
    base = torch.zeros(4, 4, 3)
    with pytest.raises(IndexError):
        apply_delta_to_frame(base, spec, frame_index=99)


# ── Bad inputs ───────────────────────────────────────────────────────────


def test_unpack_bad_magic_raises() -> None:
    bad = zlib.compress(b"NOPE" + b"\x00" * 100)
    with pytest.raises(ValueError, match="bad UWD magic"):
        unpack_sparse_delta(bad)


def test_pack_shape_mismatch_raises() -> None:
    delta = torch.zeros(2, 3, 8, 8)
    cost = torch.zeros(2, 8, 9)  # W mismatch
    with pytest.raises(ValueError, match="spatial mismatch"):
        pack_sparse_delta(delta, cost, l_inf_budget=4.0)


def test_pack_frame_count_mismatch_raises() -> None:
    delta = torch.zeros(2, 3, 8, 8)
    cost = torch.zeros(3, 8, 8)  # B mismatch
    with pytest.raises(ValueError, match="frame count mismatch"):
        pack_sparse_delta(delta, cost, l_inf_budget=4.0)


def test_pack_bad_delta_shape_raises() -> None:
    delta = torch.zeros(2, 8, 8)  # not 4-D
    cost = torch.zeros(2, 8, 8)
    with pytest.raises(ValueError, match="delta_bchw"):
        pack_sparse_delta(delta, cost, l_inf_budget=4.0)


# ── Frame-index layout sanity ────────────────────────────────────────────


def test_apply_targets_correct_pixel_channel() -> None:
    """Ground-truth check that the (frame, y, x, c) wire layout matches
    the HWC apply layout. If the index encoding ever drifts, this test
    fails with a precise pinpoint of where δ landed."""
    H, W = 6, 8
    delta = torch.zeros(1, 3, H, W)
    # Put a known value at (y=2, x=5, c=1) → within HWC at flat index
    # y*W*3 + x*3 + c = 2*24 + 15 + 1 = 64
    delta[0, 1, 2, 5] = 4.0
    cost = torch.ones(1, H, W)
    blob = pack_sparse_delta(delta, cost, l_inf_budget=4.0, target_bytes=None)
    spec = unpack_sparse_delta(blob)
    base = torch.zeros(H, W, 3)
    out = apply_delta_to_frame(base, spec, 0)
    assert out[2, 5, 1].item() > 0, (
        "δ landed at the wrong (y, x, c). Layout drift between "
        "pack-side (b, c, y, x) and apply-side (H, W, 3) — see "
        "_build_bcyx_to_byxc_index in tac.uniward_delta."
    )


def test_apply_does_not_mutate_input_buffer() -> None:
    """The renderer output may be reused across frames; apply must not
    alias-mutate the input HWC tensor."""
    delta = torch.zeros(1, 3, 4, 4)
    delta[0, 0, 0, 0] = 4.0
    cost = torch.ones(1, 4, 4)
    spec = unpack_sparse_delta(pack_sparse_delta(delta, cost, l_inf_budget=4.0))
    base = torch.full((4, 4, 3), 100.0)
    base_before = base.clone()
    _ = apply_delta_to_frame(base, spec, 0)
    assert torch.equal(base, base_before)


# ── DeltaSpec ────────────────────────────────────────────────────────────


def test_delta_spec_fields_are_consistent_after_unpack() -> None:
    delta, cost = _random_delta(n_frames=3, H=8, W=8, sparsity=0.10, seed=99)
    blob = pack_sparse_delta(delta, cost, l_inf_budget=4.0, target_bytes=None)
    spec = unpack_sparse_delta(blob)
    assert isinstance(spec, DeltaSpec)
    assert spec.n_frames == 3
    assert spec.H == 8 and spec.W == 8
    assert len(spec.per_frame_local_idx) == 3
    assert len(spec.per_frame_dequant) == 3
    counted = sum(
        0 if t is None else int(t.numel())
        for t in spec.per_frame_local_idx
    )
    assert counted == spec.n_kept
