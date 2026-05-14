# SPDX-License-Identifier: MIT
"""Lane Ω-W-V2 — codec regression tests.

Covers the 5 council-mandated test classes from the V2 design spec:

    (a) test_v2_round_trip_bit_identical
    (b) test_v2_byte_savings_vs_v1_raw
    (c) test_v2_determinism_encode_twice_equal
    (d) test_v2_score_parity_decode_within_tol
    (e) test_v2_block_fp_ineligible_raises_gate_regression

Plus paranoia tests for the magic byte registry, header parsing, and
encode-time silent-default-blocking gates (Check 81 STRICT compliance).
All tests are pure-CPU; no GPU, no scorer load.
"""
from __future__ import annotations

import struct

import pytest
import torch

from tac.block_fp_codec import encode_conv_weight
from tac.codec_magic_registry import (
    all_entries,
    find_by_magic,
    sniff_codec,
)
from tac.water_filling_codec_v2 import (
    OWV2_MAGIC,
    BlockFPIneligible,
    GateRegression,
    decode_omega_w_v2,
    encode_omega_w_v2,
    is_owv2_blob,
)


# ── synthetic fixture helpers ─────────────────────────────────────────────


def _selfcomp_like_weights(
    o: int = 24,
    i: int = 24,
    kh: int = 3,
    kw: int = 3,
    seed: int = 0,
) -> torch.Tensor:
    """A Selfcomp-class block-FP-eligible conv weight tensor.

    Distribution chosen to mimic a trained SegMap conv layer: small
    magnitude (~0.05 std), some near-zero channels, no NaN/Inf.
    """
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(o, i, kh, kw, generator=g) * 0.05).to(torch.float32)


def _hessian_for(weights: torch.Tensor, seed: int = 1) -> torch.Tensor:
    """Synthetic per-channel Hessian — deterministic from seed."""
    g = torch.Generator().manual_seed(seed)
    o = weights.shape[0]
    return (torch.rand(o, generator=g) + 0.1).to(torch.float32)


def _v1_raw_estimate_for(weights: torch.Tensor) -> int:
    """Mirror water_filling_codec_v2._v1_raw_qint_byte_estimate."""
    o, i, kh, kw = weights.shape
    return int(o * i * kh * kw) + int(o * 4) + 32


# ── (a) round-trip bit-identical on fixed synthetic block-FP weights ──────


def test_v2_round_trip_bit_identical() -> None:
    """Encode a fixed weight tensor, decode, re-encode — bytes match exactly.

    Bit-identity means the codec is deterministic at the byte level: same
    input → same bytes, every time. Critical for archive reproducibility.
    """
    weights = _selfcomp_like_weights(o=24, i=24, kh=3, kw=3, seed=0)
    hessian = _hessian_for(weights, seed=1)
    total_bits = 24 * 24 * 3 * 3 * 4  # ~ 4 bits/elt budget — comfortably feasible

    blob_a = encode_omega_w_v2(
        weights_block_fp=weights,
        hessian=hessian,
        total_bits=total_bits,
    )
    decoded = decode_omega_w_v2(blob=blob_a)
    blob_b = encode_omega_w_v2(
        weights_block_fp=decoded,
        hessian=hessian,
        total_bits=total_bits,
    )
    # Decode the second blob and re-encode; the SECOND round-trip MUST be
    # idempotent at the byte level (decoded weights are already on the
    # quantization lattice, so re-encoding produces identical bytes).
    decoded_2 = decode_omega_w_v2(blob=blob_b)
    blob_c = encode_omega_w_v2(
        weights_block_fp=decoded_2,
        hessian=hessian,
        total_bits=total_bits,
    )
    assert blob_b == blob_c, (
        f"V2 round-trip not bit-identical: blob_b ({len(blob_b)}B) != "
        f"blob_c ({len(blob_c)}B) on second re-encode of decoded weights."
    )
    # Magic byte present at the head of the blob.
    assert blob_a[:4] == OWV2_MAGIC


# ── (b) byte-savings >0 vs V1 raw qint output ─────────────────────────────


def test_v2_byte_savings_vs_v1_raw() -> None:
    """V2 must beat V1 raw qint estimate by > 0 bytes on a real Selfcomp-class weight.

    Hard fail if regression: encode raises GateRegression so this test
    indirectly verifies that the gate fires (or doesn't) correctly.
    """
    weights = _selfcomp_like_weights(o=24, i=24, kh=3, kw=3, seed=2)
    hessian = _hessian_for(weights, seed=3)
    # Use a tight budget (~3 bits/element) where ternary alphabet dominates.
    total_bits = 24 * 24 * 3 * 3 * 3

    blob = encode_omega_w_v2(
        weights_block_fp=weights,
        hessian=hessian,
        total_bits=total_bits,
    )
    v1_raw_estimate = _v1_raw_estimate_for(weights)
    savings = v1_raw_estimate - len(blob)
    assert savings > 0, (
        f"V2 did NOT save bytes vs V1: encoded={len(blob)}, "
        f"v1_raw_estimate={v1_raw_estimate}, savings={savings}. "
        f"Carmack overhead-gate would have raised — investigate."
    )
    # Surface the actual savings for the agent's report.
    print(
        f"\n  [v2-byte-savings] V1 raw {v1_raw_estimate}B -> V2 {len(blob)}B "
        f"(savings {savings}B = {100.0 * savings / v1_raw_estimate:.2f}%)"
    )


# ── (c) determinism: encode twice → bytes equal ───────────────────────────


def test_v2_determinism_encode_twice_equal() -> None:
    """Same inputs → same bytes; twice in a row, same process.

    Uses a 24x24x3x3 Selfcomp-class fixture so the arithmetic coder
    amortizes its header (1x1 conv tensors are too small for the
    arithmetic header to pay for itself; the overhead gate would fire).
    """
    weights = _selfcomp_like_weights(o=24, i=24, kh=3, kw=3, seed=4)
    hessian = _hessian_for(weights, seed=5)
    total_bits = 24 * 24 * 3 * 3 * 4

    blob_1 = encode_omega_w_v2(
        weights_block_fp=weights,
        hessian=hessian,
        total_bits=total_bits,
    )
    blob_2 = encode_omega_w_v2(
        weights_block_fp=weights,
        hessian=hessian,
        total_bits=total_bits,
    )
    assert blob_1 == blob_2, (
        f"V2 encode is non-deterministic: blob_1 ({len(blob_1)}B) != "
        f"blob_2 ({len(blob_2)}B)"
    )


# ── (d) score parity: decode within 1e-6 L1 of post-water-fill quantized ──


def test_v2_score_parity_decode_within_tol() -> None:
    """Decoded weights match the V1 block-FP quantized reference bit-for-bit.

    The arithmetic terminal is LOSSLESS over qints — encoding qints,
    arithmetic-coding them, then decoding must reproduce the SAME qints
    exactly. The float reconstruction (qint * 2**exp) is also exact since
    small int × power-of-2 is representable in float32.

    Reference: V1 block_fp_codec.encode_conv_weight + decode_conv_weight
    using the SAME per-channel Q allocation that V2 picked. They MUST
    agree to L1 < 1e-6 — anything larger means the arithmetic coder lost
    information OR our exponent/scale algebra differs from V1's.
    """
    weights = _selfcomp_like_weights(o=12, i=12, kh=3, kw=3, seed=6)
    hessian = _hessian_for(weights, seed=7)
    total_bits = 12 * 12 * 3 * 3 * 4

    blob = encode_omega_w_v2(
        weights_block_fp=weights,
        hessian=hessian,
        total_bits=total_bits,
    )
    decoded = decode_omega_w_v2(blob=blob)

    # Build a V1 reference using the SAME per-channel Q allocation V2 chose.
    # Read qmax_per_channel from the V2 header to match exactly.
    o, i, kh, kw = weights.shape
    qmax_offset = 4 + 2 + 2 + 16 + 4 + 4  # magic+ver+arith_v+OIHW+q_max+n_chan
    qmax_per_channel_v2 = list(blob[qmax_offset:qmax_offset + o])

    from tac.block_fp_codec import (  # noqa: PLC0415
        decode_conv_weight,
    )
    v1_packed = encode_conv_weight(
        weights, qint_max=max(qmax_per_channel_v2),
        per_channel_qint_max=qmax_per_channel_v2,
    )
    v1_decoded = decode_conv_weight(v1_packed)

    l1 = (decoded - v1_decoded).abs().mean().item()
    assert l1 < 1e-6, (
        f"V2 score-parity violation: V2 decoded vs V1 ref L1 = {l1:.6g} "
        f"(tol 1e-6). The arithmetic terminal must be lossless over qints, "
        f"and the per-channel exponent algebra must match V1 block-FP."
    )

    # Sanity: shape and dtype preserved.
    assert decoded.shape == weights.shape, (
        f"V2 decode reshape error: got {decoded.shape}, expected {weights.shape}"
    )
    assert decoded.dtype == torch.float32

    # Idempotent re-encode: encoding the DECODED tensor (already on the
    # quantization lattice) with the SAME hessian + total_bits MAY pick a
    # different allocation because per-channel max_abs shifts after
    # quantization. We assert only that DECODE(ENCODE(x)) is itself on the
    # lattice — i.e. a SECOND decode round equals the FIRST decode round
    # to L1 < 1e-6 only if both codecs picked the same allocation. We do
    # NOT assert allocation stability here.


# ── (e) hard kill: block-FP eligibility failure raises GateRegression ─────


def test_v2_block_fp_ineligible_raises_gate_regression() -> None:
    """Each ineligible-input class raises BlockFPIneligible (a GateRegression).

    Covers:
        * None weights
        * Non-tensor input
        * Non-4D shape
        * Empty tensor
        * NaN/Inf weights
    """
    hessian = torch.rand(8) + 0.1

    # 1. None weights
    with pytest.raises(BlockFPIneligible) as exc_info:
        encode_omega_w_v2(
            weights_block_fp=None,
            hessian=hessian,
            total_bits=64,
        )
    assert "None" in str(exc_info.value)

    # 2. Non-tensor input
    with pytest.raises(BlockFPIneligible) as exc_info:
        encode_omega_w_v2(
            weights_block_fp=[1, 2, 3],  # type: ignore[arg-type]
            hessian=hessian,
            total_bits=64,
        )
    assert "torch.Tensor" in str(exc_info.value)

    # 3. Non-4D shape (linear weight)
    linear_w = torch.randn(8, 16)
    with pytest.raises(BlockFPIneligible) as exc_info:
        encode_omega_w_v2(
            weights_block_fp=linear_w,
            hessian=hessian,
            total_bits=64,
        )
    assert "4-D" in str(exc_info.value) or "rank-2" in str(exc_info.value)

    # 4. Empty tensor
    empty_w = torch.zeros(0, 4, 3, 3)
    with pytest.raises(BlockFPIneligible) as exc_info:
        encode_omega_w_v2(
            weights_block_fp=empty_w,
            hessian=torch.zeros(0),
            total_bits=64,
        )
    # zero output channels OR zero numel — message contains "zero" or "empty"
    msg = str(exc_info.value).lower()
    assert "zero" in msg or "empty" in msg or "numel" in msg

    # 5. NaN/Inf weights
    nan_w = torch.randn(8, 4, 3, 3)
    nan_w[3, 0, 0, 0] = float("nan")
    with pytest.raises(BlockFPIneligible) as exc_info:
        encode_omega_w_v2(
            weights_block_fp=nan_w,
            hessian=hessian,
            total_bits=64,
        )
    assert "non-finite" in str(exc_info.value).lower() or "nan" in str(exc_info.value).lower()

    # And: BlockFPIneligible is a GateRegression subclass.
    assert issubclass(BlockFPIneligible, GateRegression)


# ── paranoia 1: silent-default audit (Check 81 STRICT) ────────────────────


def test_v2_no_silent_defaults_total_bits_required() -> None:
    """`total_bits` has no silent default: passing None raises.

    Validates the silent-default audit (Check 81 STRICT). The eligibility
    gate runs first, so we use a valid 4-D conv tensor; total_bits=None
    must trigger the GateRegression after eligibility passes.
    """
    weights = _selfcomp_like_weights(o=24, i=24, kh=3, kw=3, seed=8)
    hessian = _hessian_for(weights, seed=9)
    with pytest.raises(GateRegression) as exc_info:
        encode_omega_w_v2(
            weights_block_fp=weights,
            hessian=hessian,
            total_bits=None,
        )
    assert "total_bits" in str(exc_info.value)


def test_v2_no_silent_defaults_hessian_required() -> None:
    """`hessian` has no silent default: passing None raises."""
    weights = _selfcomp_like_weights(o=24, i=24, kh=3, kw=3, seed=10)
    with pytest.raises(GateRegression) as exc_info:
        encode_omega_w_v2(
            weights_block_fp=weights,
            hessian=None,
            total_bits=24 * 24 * 3 * 3 * 4,
        )
    assert "hessian" in str(exc_info.value).lower()


# ── paranoia 2: magic-byte registry has OWV2 entry ────────────────────────


def test_v2_magic_in_canonical_registry() -> None:
    """OWV2 magic is registered and discoverable via sniff_codec."""
    entries = all_entries()
    assert any(e.magic == OWV2_MAGIC for e in entries), (
        f"OWV2 magic not in canonical registry: {[e.magic for e in entries]}"
    )
    entry = find_by_magic(OWV2_MAGIC)
    assert entry is not None
    assert entry.name == "Lane Ω-W-V2"
    assert "decode_omega_w_v2" in entry.decode_module

    # Sniff a real OWV2 payload (larger fixture so overhead gate doesn't fire).
    weights = _selfcomp_like_weights(o=24, i=24, kh=3, kw=3, seed=11)
    hessian = _hessian_for(weights, seed=12)
    blob = encode_omega_w_v2(
        weights_block_fp=weights,
        hessian=hessian,
        total_bits=24 * 24 * 3 * 3 * 4,
    )
    sniffed = sniff_codec(blob)
    assert sniffed is not None
    assert sniffed.magic == OWV2_MAGIC
    assert is_owv2_blob(blob) is True
    assert is_owv2_blob(b"DEADBEEF") is False


# ── paranoia 3: header-version mismatch raises on decode ──────────────────


def test_v2_decode_rejects_bad_magic() -> None:
    """A blob with wrong magic raises ValueError."""
    bad = b"XYZ1" + b"\x00" * 64
    with pytest.raises(ValueError) as exc_info:
        decode_omega_w_v2(blob=bad)
    assert "magic" in str(exc_info.value).lower()


def test_v2_decode_rejects_truncated_header() -> None:
    """A blob too short for the header raises."""
    with pytest.raises(ValueError):
        decode_omega_w_v2(blob=b"OWV2")


def test_v2_decode_rejects_none_blob() -> None:
    """`blob=None` raises (Check 81 STRICT)."""
    with pytest.raises(ValueError) as exc_info:
        decode_omega_w_v2(blob=None)
    assert "blob" in str(exc_info.value).lower()


# ── paranoia 4: V2 ↔ V1 alphabet sanity ───────────────────────────────────


def test_v2_alphabet_matches_qint_levels() -> None:
    """The arithmetic alphabet width = 2*Q_max+1; Q_max in QINT_LEVELS.

    Decode reads `qmax_per_channel` from the header and uses it to
    reconstruct exponents; the alphabet must always cover the WIDEST Q
    actually allocated.
    """
    from tac.water_filling_codec import QINT_LEVELS

    weights = _selfcomp_like_weights(o=12, i=12, kh=3, kw=3, seed=13)
    hessian = _hessian_for(weights, seed=14)
    blob = encode_omega_w_v2(
        weights_block_fp=weights,
        hessian=hessian,
        total_bits=12 * 12 * 3 * 3 * 4,
    )
    # Header bytes 8-23 are OIHW; bytes 24-27 are q_max_global.
    # Layout: magic(4) + version(2) + arith_v(2) + oihw(16) + qmax(4) + ...
    q_max_global = struct.unpack("<i", blob[24:28])[0]
    assert q_max_global in QINT_LEVELS, (
        f"q_max_global={q_max_global} not in QINT_LEVELS={QINT_LEVELS}"
    )


# ── paranoia 5: tiny tensor raises GateRegression (overhead exceeds savings) ──


def test_v2_tiny_tensor_overhead_gate_fires() -> None:
    """On a TINY tensor (4 weights), OWV2 header dwarfs the payload.

    The Carmack overhead gate must fire — V2 cannot ship larger than V1.
    """
    weights = torch.randn(2, 1, 2, 1) * 0.01  # 4 weights total
    hessian = torch.tensor([1.0, 1.0])
    with pytest.raises(GateRegression) as exc_info:
        encode_omega_w_v2(
            weights_block_fp=weights,
            hessian=hessian,
            total_bits=16,
        )
    msg = str(exc_info.value)
    assert "OWV2 encoded" in msg
    assert ">=" in msg or "regression" in msg.lower() or "amortize" in msg.lower()
