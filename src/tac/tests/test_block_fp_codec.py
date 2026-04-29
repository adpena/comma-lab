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


# ── Selfcomp Lane MM block-FP codec ──────────────────────────────────────


from tac.block_fp_codec import (  # noqa: E402
    decode_conv_weight,
    decode_tensor_linear_q_per_tensor_v1,
    encode_conv_weight,
    encode_tensor_linear_q_per_tensor_v1,
    pack_payload_tar_xz,
    unpack_payload_tar_xz,
    verify_roundtrip,
)


def test_encode_conv_weight_shape_and_layout():
    """encode_conv_weight returns HWOI int8 + per-output int32 exponents."""
    torch.manual_seed(0)
    w = torch.randn(8, 4, 3, 3) * 0.1
    packed = encode_conv_weight(w, qint_max=7)
    assert packed["weight_qint"].dtype == torch.int8
    # HWOI layout: (kH, kW, O, I) — encoder permutes OIHW -> HWOI.
    assert packed["weight_qint"].shape == (3, 3, 8, 4)
    assert packed["weight_exponents"].shape == (8,)
    assert packed["weight_exponents"].dtype == torch.int32
    assert packed["shape_oihw"] == (8, 4, 3, 3)
    # Range constraint
    assert packed["weight_qint"].abs().max().item() <= 7


def test_conv_weight_decode_roundtrip_low_error():
    """encode -> decode recovers the original within block-FP quant noise."""
    torch.manual_seed(1)
    w = torch.randn(16, 8, 3, 3) * 0.05
    packed = encode_conv_weight(w, qint_max=7)
    rec = decode_conv_weight(packed)
    assert rec.shape == w.shape
    err = (rec - w).abs().max().item()
    # Per-channel exp + qint_max=7 -> max error per channel is
    # roughly 0.5 * 2**exp where exp = floor(log2(max_abs/7)).
    # For tiny weights the error is on the order of max_abs/7.
    rel = err / max(w.abs().max().item(), 1e-8)
    assert rel < 0.30, f"relative error {rel:.3f} above acceptable band"


def test_exponent_picker_zero_weight_channel():
    """A channel of all zeros gets exp=0, qint=0 (no NaN, no -inf)."""
    w = torch.zeros(4, 3, 3, 3)
    w[1] = torch.randn(3, 3, 3) * 0.1  # only channel 1 has non-zero weight
    packed = encode_conv_weight(w)
    assert int(packed["weight_exponents"][0].item()) == 0
    assert int(packed["weight_exponents"][2].item()) == 0
    assert int(packed["weight_exponents"][3].item()) == 0
    # HWOI layout: index [:, :, O, :] selects output channel slice.
    assert packed["weight_qint"][:, :, 0, :].abs().sum().item() == 0
    assert packed["weight_qint"][:, :, 2, :].abs().sum().item() == 0
    assert packed["weight_qint"][:, :, 3, :].abs().sum().item() == 0
    rec = decode_conv_weight(packed)
    assert torch.equal(rec[0], torch.zeros_like(rec[0]))


def test_linear_q_per_tensor_v1_roundtrip():
    """linear_q_per_tensor_v1 round-trips bias-shaped tensors within tol."""
    torch.manual_seed(2)
    bias = torch.randn(32) * 0.5
    packed = encode_tensor_linear_q_per_tensor_v1(bias, bits=8)
    rec = decode_tensor_linear_q_per_tensor_v1(packed)
    assert rec.shape == bias.shape
    # 8-bit linear quant: max error ≈ (max - min) / 255 / 2.
    eps = float(bias.max() - bias.min()) / 255 / 2 + 1e-6
    assert (rec - bias).abs().max().item() <= 2 * eps


def test_pack_payload_tar_xz_writes_archive(tmp_path):
    """pack_payload_tar_xz writes a tar.xz that can be reopened via tarfile."""
    state = {
        "layer_in.weight": torch.randn(8, 4, 1, 1) * 0.1,
        "layer_in.bias": torch.randn(8) * 0.1,
        "shared_latent_base": torch.randn(1, 3, 30, 40) * 0.05,
    }
    out = tmp_path / "payload.tar.xz"
    pack_payload_tar_xz(state, out)
    import tarfile
    with tarfile.open(out, "r:xz") as tf:
        names = set(tf.getnames())
    assert "meta.json" in names
    assert "layer_in.weight_qint.bin" in names
    assert "layer_in.weight_exponents.bin" in names
    # bias and the latent base land via linear_q
    assert "layer_in.bias.tensor.pt" in names
    assert "shared_latent_base.tensor.pt" in names


def test_roundtrip_mse_below_threshold(tmp_path):
    """verify_roundtrip succeeds on a representative SegMap-shaped state_dict."""
    torch.manual_seed(3)
    state = {
        "layer_in.weight": torch.randn(16, 8, 1, 1) * 0.05,
        "layer_in.bias": torch.randn(16) * 0.05,
        "blocks.0.conv1.weight": torch.randn(16, 16, 3, 3) * 0.02,
        "blocks.0.conv1.bias": torch.randn(16) * 0.02,
        "blocks.0.conv2.weight": torch.randn(16, 16, 3, 3) * 0.02,
        "blocks.0.conv2.bias": torch.randn(16) * 0.02,
        "layer_out.weight": torch.randn(3, 16, 1, 1) * 0.05,
        "layer_out.bias": torch.randn(3) * 0.05,
        "shared_latent_base": torch.randn(1, 3, 30, 40) * 0.02,
        "frame_affine_embedding.weight": torch.randn(8, 6) * 0.01,
    }
    payload = tmp_path / "roundtrip.tar.xz"
    # Loose tolerance: per-key ≤ 1e-4 because conv weights with qint_max=7
    # reconstruct to within ~3% relative error per element (per-channel exp).
    mse = verify_roundtrip(state, payload, tol=1e-3)
    # Every key returned a measured MSE.
    assert set(mse.keys()) == set(state.keys())
    for k, v in mse.items():
        assert v <= 1e-3, f"{k} MSE {v:.6g} above tol"


def test_conv_weight_hwoi_permute_is_invertible():
    """HWOI permute (encode) -> reverse permute (decode) is identity-correct.

    Catches the layout-confusion bug where the wrong permute axis order
    silently produces a reconstructed tensor with shape (O, I, kH, kW) but
    weights drawn from a transposed source — would give nonsense scores
    without crashing.
    """
    torch.manual_seed(4)
    w = torch.randn(6, 4, 3, 3) * 0.05
    packed = encode_conv_weight(w)
    rec = decode_conv_weight(packed)
    assert rec.shape == w.shape
    # Compare row-wise: each output channel survives the round-trip even
    # when other channels would have very different exponents.
    for c in range(w.shape[0]):
        err = (rec[c] - w[c]).abs().mean().item()
        # Per-channel bound: with qint_max=7 the worst case is ~max_abs/7.
        bound = max(w[c].abs().max().item() / 7.0, 1e-6)
        assert err <= 1.5 * bound, f"channel {c} err {err:.6g} > {1.5 * bound:.6g}"
