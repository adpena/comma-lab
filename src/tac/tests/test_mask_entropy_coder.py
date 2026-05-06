"""Roundtrip tests for ``tac.mask_entropy_coder``.

Check 46 contract: lossless 5-class mask codec must satisfy
``decode(encode(x)) == x`` exactly (atol=0). The module already exposes
a self-test (`test_roundtrip` function) but it runs at full 1200×384×512
resolution, which is too slow for CI. These tests run small synthetic
inputs targeting the same code paths.
"""

from __future__ import annotations

import tempfile
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest
import torch

from tac.mask_entropy_coder import (
    MAGIC,
    VERSION,
    decode_masks_entropy,
    encode_masks_entropy,
)


def _make_synthetic_masks(N: int, H: int, W: int, seed: int = 0) -> torch.Tensor:
    """Build spatially-coherent 5-class masks similar to driving scenes."""
    rng = np.random.RandomState(seed)
    base = np.zeros((H, W), dtype=np.uint8)
    base[: H // 4, :] = 3  # sky
    base[H // 4 : H // 2, :] = 2  # buildings
    for col in (W // 4, W // 2, 3 * W // 4):
        base[H // 2 :, max(0, col - 1) : col + 2] = 1  # lane markings
    base[3 * H // 4 :, : W // 3] = 4  # vehicle
    masks = np.empty((N, H, W), dtype=np.uint8)
    masks[0] = base
    for i in range(1, N):
        masks[i] = masks[i - 1].copy()
        # Add a few random pixel changes per frame to stress the sparse path.
        n_changes = rng.randint(1, 10)
        for _ in range(n_changes):
            r = rng.randint(0, H)
            c = rng.randint(0, W)
            masks[i, r, c] = rng.randint(0, 5)
    return torch.from_numpy(masks.astype(np.int64))


def _write_mskv(path: Path, *, N: int, H: int, W: int, method: int, raw: bytes, backend_id: int = 0) -> None:
    path.write_bytes(
        MAGIC
        + struct.pack("<BBIHHBI", VERSION, backend_id, N, H, W, method, len(raw))
        + zlib.compress(raw, 9)
    )


def test_mask_entropy_lossless_full_delta_path() -> None:
    """Static-ish masks favour the full-frame delta path; must roundtrip exactly."""
    masks = _make_synthetic_masks(N=8, H=32, W=48, seed=1)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.msk"
        encode_masks_entropy(masks, path, backend="lzma")
        decoded = decode_masks_entropy(path)
    assert torch.equal(decoded, masks), (
        "mask_entropy_coder is documented as lossless; encode/decode must be "
        "bit-exact (atol=0)."
    )


def test_mask_entropy_lossless_sparse_path() -> None:
    """High-change-rate input favours sparse path; must still roundtrip exactly."""
    rng = np.random.RandomState(2)
    H, W, N = 24, 32, 6
    masks_np = rng.randint(0, 5, size=(N, H, W), dtype=np.uint8)
    masks = torch.from_numpy(masks_np.astype(np.int64))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sparse.msk"
        encode_masks_entropy(masks, path, backend="lzma")
        decoded = decode_masks_entropy(path)
    assert torch.equal(decoded, masks)


def test_mask_entropy_zlib_backend_roundtrip() -> None:
    """Both LZMA and zlib backends must be lossless."""
    masks = _make_synthetic_masks(N=4, H=16, W=24, seed=3)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "zlib.msk"
        encode_masks_entropy(masks, path, backend="zlib")
        decoded = decode_masks_entropy(path)
    assert torch.equal(decoded, masks)


def test_mask_entropy_invalid_magic_raises() -> None:
    """Decoder must reject corrupt files (catches truncation regressions)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bad.msk"
        path.write_bytes(b"XXXX" + b"\x00" * 32)
        with pytest.raises(ValueError, match="Invalid magic"):
            decode_masks_entropy(path)


def test_mask_entropy_encode_rejects_out_of_domain_classes(tmp_path: Path) -> None:
    path = tmp_path / "bad.msk"

    with pytest.raises(ValueError, match="class ids must be in"):
        encode_masks_entropy(np.array([[[-1]]], dtype=np.int16), path)
    with pytest.raises(ValueError, match="class ids must be in"):
        encode_masks_entropy(np.array([[[5]]], dtype=np.int16), path)
    with pytest.raises(ValueError, match="unsupported mask entropy backend"):
        encode_masks_entropy(np.array([[[0]]], dtype=np.uint8), path, backend="brotli")


def test_mask_entropy_decode_rejects_malformed_header_fields(tmp_path: Path) -> None:
    short = tmp_path / "short.msk"
    short.write_bytes(MAGIC + b"\x04")
    with pytest.raises(ValueError, match="truncated MSKV header"):
        decode_masks_entropy(short)

    bad_backend = tmp_path / "bad_backend.msk"
    _write_mskv(bad_backend, N=1, H=1, W=1, method=0, raw=b"\x00", backend_id=9)
    with pytest.raises(ValueError, match="unsupported backend id 9"):
        decode_masks_entropy(bad_backend)

    bad_method = tmp_path / "bad_method.msk"
    _write_mskv(bad_method, N=1, H=1, W=1, method=9, raw=b"\x00")
    with pytest.raises(ValueError, match="unsupported method id 9"):
        decode_masks_entropy(bad_method)


def test_mask_entropy_decode_rejects_malformed_payloads(tmp_path: Path) -> None:
    bad_class = tmp_path / "bad_class.msk"
    _write_mskv(bad_class, N=1, H=1, W=1, method=0, raw=b"\x05")
    with pytest.raises(ValueError, match="class ids outside mask domain"):
        decode_masks_entropy(bad_class)

    rle_overshoot = tmp_path / "rle_overshoot.msk"
    _write_mskv(rle_overshoot, N=1, H=1, W=2, method=1, raw=b"\x00\x03")
    with pytest.raises(ValueError, match="RLE run exceeds frame size"):
        decode_masks_entropy(rle_overshoot)

    sparse_trailing = tmp_path / "sparse_trailing.msk"
    _write_mskv(sparse_trailing, N=1, H=1, W=1, method=1, raw=b"\x00\x01junk")
    with pytest.raises(ValueError, match="trailing bytes"):
        decode_masks_entropy(sparse_trailing)

    compressed_trailing = tmp_path / "compressed_trailing.msk"
    compressed_trailing.write_bytes(
        MAGIC
        + struct.pack("<BBIHHBI", VERSION, 0, 1, 1, 1, 0, 1)
        + zlib.compress(b"\x00", 9)
        + b"junk"
    )
    with pytest.raises(ValueError, match="trailing bytes after zlib stream"):
        decode_masks_entropy(compressed_trailing)
