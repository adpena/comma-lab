"""Custom entropy coder for 5-class segmentation masks.

Exploits three properties of our data that AV1 cannot fully leverage:
1. Extreme class imbalance (~60% road, ~20% sky, sparse lane/vehicle)
2. Massive spatial coherence (contiguous same-class blocks)
3. Near-zero temporal change (most pixels identical frame-to-frame)

Encoding strategy:
    Frame 0: RLE on raster-scan order (huge contiguous class runs)
    Frames 1..N-1: sparse delta -- only changed pixel (position, value) pairs
    Final: entire pre-processed stream compressed with LZMA

The sparse delta representation is key: for ~0.5-2% pixel change rate,
we store only ~1K-4K (position, value) pairs per frame instead of 196K pixels.
"""
from __future__ import annotations

import io
import lzma
import struct
import zlib
from pathlib import Path

import numpy as np
import torch

# Magic bytes to identify our format
MAGIC = b"MSKV"
VERSION = 3
NUM_CLASSES = 5


def _rle_encode_bytes(data: np.ndarray) -> bytes:
    """RLE encode a flat uint8 array into bytes.

    Format: (value:u8, count:varint) pairs.
    Optimized for the keyframe where huge contiguous runs are typical.
    """
    buf = io.BytesIO()
    n = len(data)
    if n == 0:
        return buf.getvalue()

    i = 0
    while i < n:
        val = data[i]
        run = 1
        while i + run < n and data[i + run] == val:
            run += 1
        buf.write(bytes([val]))
        _write_varint(buf, run)
        i += run
    return buf.getvalue()


def _rle_decode_bytes(data: bytes, expected_len: int) -> np.ndarray:
    """Decode RLE bytes back to flat uint8 array."""
    result = np.empty(expected_len, dtype=np.uint8)
    pos = 0
    idx = 0
    while idx < expected_len:
        val = data[pos]
        pos += 1
        run, pos = _read_varint(data, pos)
        result[idx : idx + run] = val
        idx += run
    return result


def _write_varint(buf: io.BytesIO, value: int) -> None:
    """Write unsigned LEB128 varint."""
    while value >= 0x80:
        buf.write(bytes([(value & 0x7F) | 0x80]))
        value >>= 7
    buf.write(bytes([value & 0x7F]))


def _read_varint(data: bytes, pos: int) -> tuple[int, int]:
    """Read unsigned LEB128 varint, return (value, new_pos)."""
    value = 0
    shift = 0
    while True:
        b = data[pos]
        pos += 1
        value |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
    return value, pos


def _encode_sparse_delta(prev: np.ndarray, curr: np.ndarray) -> bytes:
    """Encode frame delta as sparse list of (flat_index, new_value) pairs.

    For ~0.5% change rate on 384x512: ~980 changed pixels.
    Each stored as (varint_gap, 3-bit_value), heavily compressible.

    We store gaps between consecutive changed positions (delta-of-deltas)
    rather than absolute positions. Gaps are smaller and compress better.
    """
    flat_prev = prev.ravel()
    flat_curr = curr.ravel()
    changed_indices = np.where(flat_prev != flat_curr)[0]
    num_changed = len(changed_indices)

    buf = io.BytesIO()
    _write_varint(buf, num_changed)

    if num_changed == 0:
        return buf.getvalue()

    # Delta-encode positions (gaps between consecutive changed indices)
    # First position stored as-is, subsequent as gap from previous
    prev_idx = 0
    values = flat_curr[changed_indices]

    for i in range(num_changed):
        gap = int(changed_indices[i]) - prev_idx
        prev_idx = int(changed_indices[i])
        _write_varint(buf, gap)
        # Value is 0-4, fits in 3 bits, but write as byte for simplicity
        # (LZMA will handle the entropy coding of this small alphabet)
        buf.write(bytes([values[i]]))

    return buf.getvalue()


def _decode_sparse_delta(prev: np.ndarray, delta_bytes: bytes, offset: int) -> tuple[np.ndarray, int]:
    """Decode sparse delta, return (new_frame, bytes_consumed)."""
    curr = prev.copy()
    flat = curr.ravel()

    pos = offset
    num_changed, pos = _read_varint(delta_bytes, pos)

    if num_changed == 0:
        return curr, pos

    abs_idx = 0
    for _ in range(num_changed):
        gap, pos = _read_varint(delta_bytes, pos)
        abs_idx += gap
        val = delta_bytes[pos]
        pos += 1
        flat[abs_idx] = val

    return flat.reshape(prev.shape), pos


def encode_masks_entropy(
    masks: torch.Tensor | np.ndarray,
    output_path: str | Path,
    backend: str = "lzma",
) -> int:
    """Encode segmentation masks to a compact binary bitstream.

    Args:
        masks: (N, H, W) uint8/long tensor or ndarray with values 0-4
        output_path: path for output binary file
        backend: "lzma" (smaller) or "zlib" (faster)

    Returns:
        File size in bytes
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(masks, torch.Tensor):
        masks_np = masks.cpu().numpy().astype(np.uint8)
    else:
        masks_np = np.asarray(masks, dtype=np.uint8)

    N, H, W = masks_np.shape

    # Build raw payload
    payload = io.BytesIO()

    # Keyframe: RLE-encoded
    kf = _rle_encode_bytes(masks_np[0].ravel())
    _write_varint(payload, len(kf))
    payload.write(kf)

    # Delta frames: sparse encoding
    for i in range(1, N):
        delta = _encode_sparse_delta(masks_np[i - 1], masks_np[i])
        _write_varint(payload, len(delta))
        payload.write(delta)

    raw_payload = payload.getvalue()

    # Compress
    if backend == "lzma":
        compressed = lzma.compress(raw_payload, format=lzma.FORMAT_XZ, preset=9)
        backend_id = 1
    elif backend == "zlib":
        compressed = zlib.compress(raw_payload, 9)
        backend_id = 0
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # Write file
    with open(output_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<B", VERSION))
        f.write(struct.pack("<B", backend_id))
        f.write(struct.pack("<I", N))
        f.write(struct.pack("<H", H))
        f.write(struct.pack("<H", W))
        f.write(struct.pack("<I", len(raw_payload)))
        f.write(compressed)

    size = output_path.stat().st_size
    raw_size = N * H * W
    print(
        f"[mask_entropy] Encoded {N} masks ({H}x{W}) -> {output_path} "
        f"({size:,} bytes, ratio={size / raw_size:.6f})"
    )
    return size


def decode_masks_entropy(
    input_path: str | Path,
) -> torch.Tensor:
    """Decode masks from entropy-coded binary bitstream.

    Args:
        input_path: path to .msk binary file

    Returns:
        (N, H, W) long tensor with values in [0, NUM_CLASSES)
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Mask file not found: {input_path}")

    with open(input_path, "rb") as f:
        data = f.read()

    pos = 0

    magic = data[pos : pos + 4]
    pos += 4
    if magic != MAGIC:
        raise ValueError(f"Invalid magic: {magic!r}")

    version = data[pos]
    pos += 1
    if version != VERSION:
        raise ValueError(f"Unknown version: {version}")

    backend_id = data[pos]
    pos += 1

    N = struct.unpack_from("<I", data, pos)[0]; pos += 4
    H = struct.unpack_from("<H", data, pos)[0]; pos += 2
    W = struct.unpack_from("<H", data, pos)[0]; pos += 2
    uncompressed_size = struct.unpack_from("<I", data, pos)[0]; pos += 4

    compressed = data[pos:]

    if backend_id == 1:
        raw = lzma.decompress(compressed, format=lzma.FORMAT_XZ)
    elif backend_id == 0:
        raw = zlib.decompress(compressed)
    else:
        raise ValueError(f"Unknown backend_id: {backend_id}")

    assert len(raw) == uncompressed_size

    frame_size = H * W
    masks = np.empty((N, H, W), dtype=np.uint8)

    # Decode keyframe
    rpos = 0
    kf_len, rpos = _read_varint(raw, rpos)
    kf_data = raw[rpos : rpos + kf_len]
    rpos += kf_len
    masks[0] = _rle_decode_bytes(kf_data, frame_size).reshape(H, W)

    # Decode delta frames
    for i in range(1, N):
        delta_len, rpos = _read_varint(raw, rpos)
        delta_data = raw[rpos : rpos + delta_len]
        rpos += delta_len
        # Parse sparse delta from beginning of delta_data
        masks[i], _ = _decode_sparse_delta(masks[i - 1], delta_data, 0)

    result = torch.from_numpy(masks.astype(np.int64))
    print(f"[mask_entropy] Decoded {N} masks ({H}x{W}) from {input_path}")
    return result


def test_roundtrip(
    num_frames: int = 1200,
    H: int = 384,
    W: int = 512,
) -> dict:
    """Test encode/decode roundtrip on synthetic mask data.

    Two scenarios:
    1. "realistic": 0.5% pixel change rate (typical real driving data)
    2. "worst_case": 2% random pixel change rate (stress test)
    """
    import tempfile
    import time

    results = {}

    for scenario, change_rate in [("realistic", 0.005), ("worst_case", 0.02)]:
        rng = np.random.RandomState(42)

        # Spatially coherent base frame
        base = np.zeros((H, W), dtype=np.uint8)
        base[: int(H * 0.25), :] = 3          # sky
        base[int(H * 0.25) : int(H * 0.4), :] = 2  # undrivable
        base[int(H * 0.4) :, :] = 0           # road
        for col_center in [W // 4, W // 2, 3 * W // 4]:
            base[int(H * 0.4) :, col_center - 2 : col_center + 2] = 1  # lanes
        for vx, vy in [(W // 3, int(H * 0.55)), (2 * W // 3, int(H * 0.6))]:
            base[vy - 15 : vy + 15, vx - 20 : vx + 20] = 4  # vehicles

        masks = np.empty((num_frames, H, W), dtype=np.uint8)
        masks[0] = base
        pixels_per_change = int(change_rate * H * W)
        for i in range(1, num_frames):
            masks[i] = masks[i - 1].copy()
            rows = rng.randint(0, H, size=pixels_per_change)
            cols = rng.randint(0, W, size=pixels_per_change)
            vals = rng.randint(0, NUM_CLASSES, size=pixels_per_change).astype(np.uint8)
            masks[i, rows, cols] = vals

        masks_tensor = torch.from_numpy(masks.astype(np.int64))

        with tempfile.TemporaryDirectory() as tmpdir:
            for backend in ["lzma", "zlib"]:
                key = f"{scenario}_{backend}"
                out_path = Path(tmpdir) / f"test_{key}.msk"

                t0 = time.time()
                size = encode_masks_entropy(masks_tensor, out_path, backend=backend)
                enc_time = time.time() - t0

                t0 = time.time()
                decoded = decode_masks_entropy(out_path)
                dec_time = time.time() - t0

                match = torch.equal(masks_tensor, decoded)
                raw_size = num_frames * H * W
                av1_est = 33_000

                results[key] = {
                    "size_bytes": size,
                    "raw_size": raw_size,
                    "ratio": size / raw_size,
                    "vs_av1_pct": round(100 * size / av1_est, 1),
                    "lossless": match,
                    "encode_sec": round(enc_time, 2),
                    "decode_sec": round(dec_time, 2),
                }
                print(
                    f"\n[test] {key}: {size:,} bytes "
                    f"(ratio={size / raw_size:.6f}, "
                    f"~{100 * size / av1_est:.0f}% of AV1, "
                    f"enc={enc_time:.1f}s dec={dec_time:.1f}s "
                    f"lossless={match})"
                )

    return results


if __name__ == "__main__":
    results = test_roundtrip()
    print("\n=== Summary ===")
    for key, r in results.items():
        status = "PASS" if r["lossless"] else "FAIL"
        print(
            f"  {key}: {r['size_bytes']:,} bytes | "
            f"{r['vs_av1_pct']}% of AV1 | "
            f"enc={r['encode_sec']}s dec={r['decode_sec']}s | {status}"
        )
