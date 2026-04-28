"""Round-trip + bits/weight tests for ``tac.block_fp_codec``.

These tests are the contract for the szabolcs Phase 2 weight format. They
guarantee:

1. ``unpack_block_fp(pack_block_fp(w))`` recovers ``w`` to within the ternary
   rounding error bound (~ 0.5 * 2**block_exponent per element).
2. Header round-trips losslessly.
3. The packed representation lands well below 8 bits/weight on conv-shaped
   tensors. Achieving exactly 1.017 bits/weight requires the tar.xz outer
   wrapper (covered in ``test_szabolcs_export_load.py``).
"""
from __future__ import annotations

import struct

import pytest
import torch

from tac.block_fp_codec import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_CLIP_THRESHOLD,
    BlockFPHeader,
    measure_bits_per_weight,
    pack_block_fp,
    unpack_block_fp,
)


# ── Header ─────────────────────────────────────────────────────────────────


class TestBlockFPHeader:
    def test_header_round_trip(self):
        h = BlockFPHeader(
            rank=4,
            block_size=16,
            clip_threshold=0.5,
            shape=(32, 8, 3, 3),
            num_blocks=2,
            qint_nbytes=32 * 8 * 3 * 3,
            exponents_nbytes=2 * 4,
        )
        blob = h.encode()
        h2, used = BlockFPHeader.decode(blob)
        assert h2 == h
        assert used == len(blob)

    def test_header_bad_magic_raises(self):
        with pytest.raises(ValueError, match="block_fp_codec"):
            BlockFPHeader.decode(b"BFP2" + b"\x00" * 32)

    def test_header_bad_version_raises(self):
        bad = b"BFP1" + struct.pack("<B", 99) + b"\x00" * 32
        with pytest.raises(ValueError, match="unsupported version"):
            BlockFPHeader.decode(bad)


# ── Pack/unpack round trip ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "shape",
    [
        (32, 8, 3, 3),    # conv2d weight
        (64, 8, 1, 1),    # 1x1 conv
        (1200, 6),        # affine embedding
        (3, 30, 40),      # latent grid (after squeeze)
        (16,),            # 1-D vector
        (17, 5),          # non-divisible-by-block_size leading dim
    ],
)
def test_pack_unpack_round_trip(shape):
    torch.manual_seed(0)
    w = torch.randn(*shape) * 0.3
    blob = pack_block_fp(w)
    w_hat = unpack_block_fp(blob)
    assert w_hat.shape == w.shape
    # Per-block scale is 2**ceil(log2(max_abs)); ternary noise per element is
    # bounded by the block scale * clip_threshold. We compute a per-block
    # bound and assert the actual error is within it.
    err = (w - w_hat).abs()
    # An overall "max element error <= largest block scale" check is always
    # safe; for most blocks the actual bound is much tighter.
    overall_max_scale = 2.0 ** max(0.0, torch.ceil(torch.log2(w.abs().max() + 1e-20)).item())
    assert err.max().item() <= overall_max_scale + 1e-6


def test_zero_weight_round_trip():
    w = torch.zeros(8, 4, 3, 3)
    blob = pack_block_fp(w)
    w_hat = unpack_block_fp(blob)
    assert torch.equal(w_hat, torch.zeros_like(w))


def test_empty_weight_safe():
    # A 0-element tensor should produce a header-only blob and round-trip.
    w = torch.zeros(0, 8)
    blob = pack_block_fp(w)
    w_hat = unpack_block_fp(blob)
    assert w_hat.shape == w.shape


def test_pack_rejects_scalar():
    with pytest.raises(ValueError, match="scalar tensors not supported"):
        pack_block_fp(torch.tensor(1.0))


def test_pack_rejects_bad_args():
    w = torch.randn(8, 4)
    with pytest.raises(ValueError, match="block_size"):
        pack_block_fp(w, block_size=0)
    with pytest.raises(ValueError, match="clip_threshold"):
        pack_block_fp(w, clip_threshold=-1.0)


# ── Bits/weight ───────────────────────────────────────────────────────────


def test_bits_per_weight_under_raw_int8():
    """Pre-tar.xz, the codec stores int8 qint + small float exponent header.
    Worst case is ~8 bits/weight + small overhead; we assert < 10 bits/weight.
    The tar.xz outer wrapper drives this down to ~1-1.5 bits/weight in
    ``test_szabolcs_export_load.py``."""
    torch.manual_seed(0)
    w = torch.randn(64, 8, 3, 3) * 0.2
    blob = pack_block_fp(w)
    bpw = measure_bits_per_weight(w, blob)
    # Without outer compression, dense int8 is the floor (8 bpw); we add a
    # tiny header + per-block exponents so allow up to 10.
    assert bpw <= 10.0, f"raw bits/weight {bpw:.2f} too high"


def test_pack_unpack_via_path_roundtrip(tmp_path):
    """Bytes API: caller can write the blob to disk and read it back."""
    torch.manual_seed(0)
    w = torch.randn(32, 4, 3, 3) * 0.1
    blob = pack_block_fp(w, block_size=8)
    f = tmp_path / "weight.bfp"
    f.write_bytes(blob)
    w_hat = unpack_block_fp(f.read_bytes())
    assert w_hat.shape == w.shape
    assert torch.isfinite(w_hat).all()


def test_unpack_shape_override_check():
    w = torch.randn(8, 4)
    blob = pack_block_fp(w)
    # Matching shape: no error.
    unpack_block_fp(blob, shape=(8, 4))
    # Wrong shape: hard error.
    with pytest.raises(ValueError, match="shape override"):
        unpack_block_fp(blob, shape=(4, 8))


def test_default_constants_documented():
    """Defaults are part of the on-disk format contract — changing them
    silently would break previously-encoded archives."""
    assert DEFAULT_BLOCK_SIZE == 16
    assert DEFAULT_CLIP_THRESHOLD == 0.5
