"""Custom entropy coder for 5-class segmentation masks.

Exploits three properties of our data that AV1 cannot fully leverage:
1. Extreme class imbalance (~60% road, ~20% sky, sparse lane/vehicle)
2. Massive spatial coherence (contiguous same-class blocks)
3. Near-zero temporal change (most pixels identical frame-to-frame)

Strategy: two-pass encoding picks the smaller of two representations:

  Method A (full-frame delta): encode frame 0 raw, then delta frames where
    0=unchanged, (class+1)=changed. The stream is >95% zeros. LZMA's
    dictionary coder compresses long zero runs to near-zero cost.

  Method B (sparse delta): for very low change rates, store only
    (gap, value) pairs for changed pixels. Better when <0.5% changes.

Both methods share the same LZMA backend. The encoder picks whichever
produces smaller output, recorded in a 1-bit flag per method.
"""

from __future__ import annotations

import io
import lzma
import struct
import zlib
from pathlib import Path

import numpy as np
import torch

MAGIC = b"MSKV"
VERSION = 4
NUM_CLASSES = 5
HEADER_SIZE = 4 + 1 + 1 + 4 + 2 + 2 + 1 + 4  # 19 bytes


def _write_varint(buf: io.BytesIO, value: int) -> None:
    """Write unsigned LEB128 varint."""
    while value >= 0x80:
        buf.write(bytes([(value & 0x7F) | 0x80]))
        value >>= 7
    buf.write(bytes([value & 0x7F]))


def _read_varint_from(data: bytes, pos: int) -> tuple[int, int]:
    """Read unsigned LEB128 varint, return (value, new_pos)."""
    value = 0
    shift = 0
    while True:
        if pos >= len(data):
            raise ValueError(f"Truncated varint at position {pos}")
        b = data[pos]
        pos += 1
        value |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
    return value, pos


# ---------- Method A: full-frame delta ----------


def _build_fulldelta_payload(masks_np: np.ndarray) -> bytes:
    """Build payload: frame 0 raw + delta frames (0=same, class+1=changed)."""
    N = masks_np.shape[0]
    parts = [masks_np[0].tobytes()]
    for i in range(1, N):
        prev = masks_np[i - 1]
        curr = masks_np[i]
        delta = np.where(prev == curr, np.uint8(0), curr + np.uint8(1))
        parts.append(delta.tobytes())
    return b"".join(parts)


def _decode_fulldelta_payload(raw: bytes, N: int, H: int, W: int) -> np.ndarray:
    """Decode full-frame delta payload."""
    frame_size = H * W
    masks = np.empty((N, H, W), dtype=np.uint8)
    masks[0] = np.frombuffer(raw, dtype=np.uint8, count=frame_size, offset=0).reshape(H, W)
    for i in range(1, N):
        delta = np.frombuffer(raw, dtype=np.uint8, count=frame_size, offset=i * frame_size).reshape(H, W)
        masks[i] = np.where(delta == 0, masks[i - 1], delta - 1)
    return masks


# ---------- Method B: sparse delta ----------


def _build_sparse_payload(masks_np: np.ndarray) -> bytes:
    """Build payload: RLE keyframe + sparse deltas (gap-coded positions)."""
    N, H, W = masks_np.shape
    buf = io.BytesIO()

    # Keyframe: RLE
    flat0 = masks_np[0].ravel()
    n = len(flat0)
    i = 0
    while i < n:
        val = flat0[i]
        run = 1
        while i + run < n and flat0[i + run] == val:
            run += 1
        buf.write(bytes([val]))
        _write_varint(buf, run)
        i += run

    # Delta frames: sparse (count, gap+value pairs)
    for fi in range(1, N):
        flat_prev = masks_np[fi - 1].ravel()
        flat_curr = masks_np[fi].ravel()
        changed = np.where(flat_prev != flat_curr)[0]
        _write_varint(buf, len(changed))
        prev_idx = 0
        for ci in changed:
            _write_varint(buf, int(ci) - prev_idx)
            buf.write(bytes([flat_curr[ci]]))
            prev_idx = int(ci)

    return buf.getvalue()


def _decode_sparse_payload(raw: bytes, N: int, H: int, W: int) -> np.ndarray:
    """Decode sparse payload."""
    frame_size = H * W
    masks = np.empty((N, H, W), dtype=np.uint8)

    # Keyframe: RLE decode
    pos = 0
    frame0 = np.empty(frame_size, dtype=np.uint8)
    idx = 0
    while idx < frame_size:
        val = raw[pos]
        pos += 1
        run, pos = _read_varint_from(raw, pos)
        frame0[idx : idx + run] = val
        idx += run
    masks[0] = frame0.reshape(H, W)

    # Delta frames
    for fi in range(1, N):
        masks[fi] = masks[fi - 1].copy()
        flat = masks[fi].ravel()
        num_changed, pos = _read_varint_from(raw, pos)
        abs_idx = 0
        for _ in range(num_changed):
            gap, pos = _read_varint_from(raw, pos)
            abs_idx += gap
            flat[abs_idx] = raw[pos]
            pos += 1

    return masks


# ---------- Public API ----------


def encode_masks_entropy(
    masks: torch.Tensor | np.ndarray,
    output_path: str | Path,
    backend: str = "lzma",
) -> int:
    """Encode segmentation masks to a compact binary bitstream.

    Tries both full-frame delta and sparse delta, picks the smaller one.

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

    # Build both payloads
    payload_full = _build_fulldelta_payload(masks_np)
    payload_sparse = _build_sparse_payload(masks_np)

    # Compress both and pick smaller
    compress = (
        (lambda d: lzma.compress(d, format=lzma.FORMAT_XZ, preset=6))
        if backend == "lzma"
        else (lambda d: zlib.compress(d, 9))
    )
    backend_id = 1 if backend == "lzma" else 0

    comp_full = compress(payload_full)
    comp_sparse = compress(payload_sparse)

    if len(comp_full) <= len(comp_sparse):
        method = 0  # full-frame delta
        compressed = comp_full
        raw_size = len(payload_full)
    else:
        method = 1  # sparse delta
        compressed = comp_sparse
        raw_size = len(payload_sparse)

    # Write file
    with open(output_path, "wb") as f:
        f.write(MAGIC)  # 4
        f.write(struct.pack("<B", VERSION))  # 1
        f.write(struct.pack("<B", backend_id))  # 1
        f.write(struct.pack("<I", N))  # 4
        f.write(struct.pack("<H", H))  # 2
        f.write(struct.pack("<H", W))  # 2
        f.write(struct.pack("<B", method))  # 1
        f.write(struct.pack("<I", raw_size))  # 4
        f.write(compressed)

    size = output_path.stat().st_size
    total_pixels = N * H * W
    method_name = "full-delta" if method == 0 else "sparse"
    print(
        f"[mask_entropy] Encoded {N} masks ({H}x{W}) -> {output_path} "
        f"({size:,} bytes, ratio={size / total_pixels:.6f}, method={method_name})"
    )
    return size


def decode_masks_entropy(
    input_path: str | Path,
) -> torch.Tensor:
    """Decode masks from entropy-coded binary bitstream.

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
    N = struct.unpack_from("<I", data, pos)[0]
    pos += 4
    H = struct.unpack_from("<H", data, pos)[0]
    pos += 2
    W = struct.unpack_from("<H", data, pos)[0]
    pos += 2
    method = data[pos]
    pos += 1
    raw_size = struct.unpack_from("<I", data, pos)[0]
    pos += 4

    compressed = data[pos:]

    if backend_id == 1:
        raw = lzma.decompress(compressed, format=lzma.FORMAT_XZ)
    else:
        raw = zlib.decompress(compressed)

    assert len(raw) == raw_size, f"Size mismatch: {len(raw)} vs {raw_size}"

    if method == 0:
        masks = _decode_fulldelta_payload(raw, N, H, W)
    else:
        masks = _decode_sparse_payload(raw, N, H, W)

    result = torch.from_numpy(masks.astype(np.int64))
    print(f"[mask_entropy] Decoded {N} masks ({H}x{W}) from {input_path}")
    return result


# ---------- Test ----------


def test_roundtrip(
    num_frames: int = 1200,
    H: int = 384,
    W: int = 512,
    av1_est: int = 33_000,
) -> dict:
    """Test roundtrip on synthetic data with varying temporal change rates."""
    import tempfile
    import time

    results = {}
    scenarios = [
        "static",  # zero change (identical frames)
        "boundary",  # coherent boundary shifts (realistic driving)
        "random_0.5pct",  # 0.5% random scatter (adversarial)
        "random_2pct",  # 2% random scatter (extreme adversarial)
    ]

    for scenario in scenarios:
        rng = np.random.RandomState(42)

        # Spatially coherent base frame
        base = np.zeros((H, W), dtype=np.uint8)
        base[: int(H * 0.25), :] = 3
        base[int(H * 0.25) : int(H * 0.4), :] = 2
        for col in [W // 4, W // 2, 3 * W // 4]:
            base[int(H * 0.4) :, col - 2 : col + 2] = 1
        for vx, vy in [(W // 3, int(H * 0.55)), (2 * W // 3, int(H * 0.6))]:
            base[vy - 15 : vy + 15, vx - 20 : vx + 20] = 4

        masks = np.empty((num_frames, H, W), dtype=np.uint8)
        masks[0] = base

        for i in range(1, num_frames):
            masks[i] = masks[i - 1].copy()

            if scenario == "static":
                pass  # no changes

            elif scenario == "boundary":
                # Realistic: shift class boundaries by 1 pixel per frame
                # This simulates camera motion causing boundary movement
                shift = rng.choice([-1, 0, 1])
                if shift != 0:
                    # Shift the sky/undrivable boundary
                    sky_row = int(H * 0.25) + (i * shift) % 5 - 2
                    sky_row = max(10, min(int(H * 0.35), sky_row))
                    masks[i, sky_row - 1 : sky_row + 1, :] = masks[i, sky_row + 1, :]
                    # Shift lane markings horizontally
                    for col in [W // 4, W // 2, 3 * W // 4]:
                        jitter = rng.randint(-1, 2)
                        c = col + jitter
                        if 2 <= c < W - 2:
                            masks[i, int(H * 0.4) :, c - 2 : c + 2] = 1

            elif scenario == "random_0.5pct":
                ppc = int(0.005 * H * W)
                rows = rng.randint(0, H, size=ppc)
                cols = rng.randint(0, W, size=ppc)
                vals = rng.randint(0, NUM_CLASSES, size=ppc).astype(np.uint8)
                masks[i, rows, cols] = vals

            elif scenario == "random_2pct":
                ppc = int(0.02 * H * W)
                rows = rng.randint(0, H, size=ppc)
                cols = rng.randint(0, W, size=ppc)
                vals = rng.randint(0, NUM_CLASSES, size=ppc).astype(np.uint8)
                masks[i, rows, cols] = vals

        masks_tensor = torch.from_numpy(masks.astype(np.int64))

        with tempfile.TemporaryDirectory() as tmpdir:
            for backend in ["lzma"]:
                key = f"{scenario}_{backend}"
                out_path = Path(tmpdir) / f"test_{key}.msk"

                t0 = time.time()
                size = encode_masks_entropy(masks_tensor, out_path, backend=backend)
                enc_time = time.time() - t0

                t0 = time.time()
                decoded = decode_masks_entropy(out_path)
                dec_time = time.time() - t0

                match = torch.equal(masks_tensor, decoded)
                raw_pixels = num_frames * H * W

                results[key] = {
                    "size_bytes": size,
                    "ratio": size / raw_pixels,
                    "vs_av1_pct": round(100 * size / av1_est, 1),
                    "lossless": match,
                    "encode_sec": round(enc_time, 2),
                    "decode_sec": round(dec_time, 2),
                }
                print(
                    f"\n[test] {key}: {size:,} bytes "
                    f"({100 * size / av1_est:.0f}% of AV1, "
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
            f"ratio={r['ratio']:.6f} | "
            f"enc={r['encode_sec']}s dec={r['decode_sec']}s | {status}"
        )
