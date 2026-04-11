#!/usr/bin/env python3
"""Benchmark mask encoding strategies: AV1 vs VVC vs lossless alternatives.

Generates synthetic 5-class segmentation masks (1200 frames, 384x512)
matching our pipeline, then encodes with every available codec and measures:
  - File size (bytes)
  - Encode time
  - Decode time
  - Class accuracy (fraction of pixels that round-trip perfectly)

Usage:
    python experiments/benchmark_mask_codecs.py
"""
from __future__ import annotations

import io
import os
import struct
import subprocess
import sys
import tempfile
import time
import zlib
from pathlib import Path

import numpy as np

# ── Constants matching our pipeline ──────────────────────────────────────────
NUM_FRAMES = 300  # Use 300 for fast benchmarking, scale results to estimate 1200
H, W = 384, 512
NUM_CLASSES = 5
SCALE_FACTOR = 255 // (NUM_CLASSES - 1)  # 63
FPS = 20

RESULTS: list[dict] = []


def generate_realistic_masks(n_frames: int = NUM_FRAMES, h: int = H, w: int = W) -> np.ndarray:
    """Generate synthetic 5-class masks with realistic spatial structure.

    Creates masks that look like driving scene segmentation:
    - Road (class 0): bottom half
    - Sky (class 4): top quarter
    - Buildings/objects (classes 1-3): scattered blocks

    Temporal coherence: each frame drifts slightly from the previous one.
    """
    print(f"Generating {n_frames} realistic masks ({h}x{w}, {NUM_CLASSES} classes)...")
    rng = np.random.default_rng(42)
    masks = np.zeros((n_frames, h, w), dtype=np.int64)

    # Base template: road at bottom, sky at top
    template = np.zeros((h, w), dtype=np.int64)
    template[:h // 4, :] = 4          # sky
    template[h // 4:h // 2, :] = 2    # mid-ground
    template[h // 2:, :] = 0          # road

    # Add some structure (lane markings, buildings)
    template[h // 4:h * 3 // 4, :w // 8] = 1      # left building
    template[h // 4:h * 3 // 4, w * 7 // 8:] = 3  # right building

    masks[0] = template.copy()

    for i in range(1, n_frames):
        masks[i] = masks[i - 1].copy()
        # Randomly perturb ~2% of pixels (simulates motion)
        n_flip = int(h * w * 0.02)
        rows = rng.integers(0, h, size=n_flip)
        cols = rng.integers(0, w, size=n_flip)
        masks[i, rows, cols] = rng.integers(0, NUM_CLASSES, size=n_flip)

    print(f"  Class distribution: {np.bincount(masks.ravel(), minlength=NUM_CLASSES)}")
    return masks


def scale_to_gray(masks: np.ndarray) -> np.ndarray:
    """Scale class labels to byte range: 0->0, 1->63, 2->127, 3->191, 4->255."""
    return (masks * SCALE_FACTOR).clip(0, 255).astype(np.uint8)


def gray_to_classes(pixels: np.ndarray) -> np.ndarray:
    """Invert gray scaling back to class labels."""
    return np.round(pixels.astype(np.float32) / SCALE_FACTOR).clip(0, NUM_CLASSES - 1).astype(np.int64)


def measure_accuracy(original: np.ndarray, decoded: np.ndarray) -> float:
    """Fraction of pixels with correct class after round-trip."""
    return float(np.mean(original == decoded))


def record(name: str, size: int, accuracy: float, enc_time: float, dec_time: float):
    """Record benchmark result."""
    RESULTS.append({
        "name": name,
        "size_bytes": size,
        "size_kb": size / 1024,
        "accuracy": accuracy,
        "encode_s": enc_time,
        "decode_s": dec_time,
    })
    print(f"  {name}: {size:,} bytes ({size/1024:.1f} KB), accuracy={accuracy:.6f}, "
          f"enc={enc_time:.2f}s, dec={dec_time:.2f}s")


# ═══════════════════════════════════════════════════════════════════════════════
# Codec implementations
# ═══════════════════════════════════════════════════════════════════════════════

def bench_av1(masks: np.ndarray, crf: int, label_suffix: str = "",
              disable_filters: bool = False) -> None:
    """Benchmark AV1 encoding via libsvtav1."""
    pixels = scale_to_gray(masks)
    raw = pixels.tobytes()
    n, h, w = masks.shape

    svt_params = []
    if disable_filters:
        svt_params = ["-svtav1-params", "enable-restoration=0:enable-cdef=0"]

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        out_path = f.name

    try:
        cmd = [
            "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{w}x{h}", "-pix_fmt", "gray", "-r", str(FPS),
            "-i", "pipe:0",
            "-c:v", "libsvtav1", "-crf", str(crf), "-preset", "6",
            *svt_params,
            "-pix_fmt", "yuv420p", "-an", out_path,
        ]
        t0 = time.time()
        proc = subprocess.run(cmd, input=raw, capture_output=True, timeout=300)
        enc_time = time.time() - t0
        if proc.returncode != 0:
            print(f"  AV1 CRF={crf} FAILED: {proc.stderr[-500:]}")
            return

        size = os.path.getsize(out_path)

        # Decode
        cmd_dec = [
            "ffmpeg", "-i", out_path, "-f", "rawvideo", "-pix_fmt", "gray",
            "-v", "error", "pipe:1",
        ]
        t0 = time.time()
        dec = subprocess.run(cmd_dec, capture_output=True, timeout=300)
        dec_time = time.time() - t0

        decoded_pixels = np.frombuffer(dec.stdout, dtype=np.uint8).reshape(n, h, w)
        decoded_classes = gray_to_classes(decoded_pixels)
        acc = measure_accuracy(masks, decoded_classes)

        filt_str = " (no filters)" if disable_filters else ""
        record(f"AV1 CRF={crf}{filt_str}{label_suffix}", size, acc, enc_time, dec_time)
    finally:
        os.unlink(out_path)


def bench_vvc(masks: np.ndarray, qp: int) -> None:
    """Benchmark VVC/H.266 encoding via vvencapp + vvdecapp."""
    pixels = scale_to_gray(masks)
    n, h, w = masks.shape

    with tempfile.TemporaryDirectory() as tmpdir:
        yuv_path = os.path.join(tmpdir, "masks.yuv")
        vvc_path = os.path.join(tmpdir, "masks.266")
        dec_yuv_path = os.path.join(tmpdir, "decoded.yuv")

        # Convert gray to YUV400 (grayscale): just write raw bytes
        # vvencapp supports yuv400 format directly
        with open(yuv_path, "wb") as f:
            f.write(pixels.tobytes())

        # Encode with vvencapp
        cmd_enc = [
            "vvencapp",
            "-i", yuv_path,
            "-s", f"{w}x{h}",
            "-c", "yuv400",
            "-r", str(FPS),
            "-f", str(n),
            "--preset", "medium",
            "-q", str(qp),
            "--qpa", "0",  # disable perceptual QP adaptation (we want uniform)
            "-o", vvc_path,
        ]
        t0 = time.time()
        proc = subprocess.run(cmd_enc, capture_output=True, timeout=600)
        enc_time = time.time() - t0

        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="replace")
            print(f"  VVC QP={qp} encode FAILED: {stderr[-500:]}")
            return

        size = os.path.getsize(vvc_path)

        # Decode with vvdecapp
        cmd_dec = [
            "vvdecapp",
            "-b", vvc_path,
            "-o", dec_yuv_path,
        ]
        t0 = time.time()
        dec = subprocess.run(cmd_dec, capture_output=True, timeout=300)
        dec_time = time.time() - t0

        if dec.returncode != 0:
            stderr = dec.stderr.decode("utf-8", errors="replace")
            print(f"  VVC QP={qp} decode FAILED: {stderr[-500:]}")
            return

        # Read decoded YUV400
        decoded_raw = np.fromfile(dec_yuv_path, dtype=np.uint8)
        expected_size = n * h * w
        if decoded_raw.size < expected_size:
            print(f"  VVC QP={qp} decoded size mismatch: {decoded_raw.size} vs {expected_size}")
            return
        decoded_pixels = decoded_raw[:expected_size].reshape(n, h, w)
        decoded_classes = gray_to_classes(decoded_pixels)
        acc = measure_accuracy(masks, decoded_classes)

        record(f"VVC QP={qp} (yuv400)", size, acc, enc_time, dec_time)


def bench_vvc_yuv420(masks: np.ndarray, qp: int) -> None:
    """Benchmark VVC with yuv420p (for fair comparison with AV1 which uses 420)."""
    pixels = scale_to_gray(masks)
    n, h, w = masks.shape

    with tempfile.TemporaryDirectory() as tmpdir:
        yuv_path = os.path.join(tmpdir, "masks.yuv")
        vvc_path = os.path.join(tmpdir, "masks.266")
        dec_yuv_path = os.path.join(tmpdir, "decoded.yuv")

        # Convert gray to YUV420: Y=gray, U=V=128 (neutral chroma)
        yuv_frames = []
        for i in range(n):
            y = pixels[i]
            u = np.full((h // 2, w // 2), 128, dtype=np.uint8)
            v = np.full((h // 2, w // 2), 128, dtype=np.uint8)
            yuv_frames.append(y.tobytes() + u.tobytes() + v.tobytes())

        with open(yuv_path, "wb") as f:
            for frame in yuv_frames:
                f.write(frame)

        cmd_enc = [
            "vvencapp",
            "-i", yuv_path,
            "-s", f"{w}x{h}",
            "-c", "yuv420",
            "-r", str(FPS),
            "-f", str(n),
            "--preset", "medium",
            "-q", str(qp),
            "--qpa", "0",
            "-o", vvc_path,
        ]
        t0 = time.time()
        proc = subprocess.run(cmd_enc, capture_output=True, timeout=600)
        enc_time = time.time() - t0

        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="replace")
            print(f"  VVC YUV420 QP={qp} encode FAILED: {stderr[-500:]}")
            return

        size = os.path.getsize(vvc_path)

        # Decode
        cmd_dec = [
            "vvdecapp",
            "-b", vvc_path,
            "-o", dec_yuv_path,
        ]
        t0 = time.time()
        dec = subprocess.run(cmd_dec, capture_output=True, timeout=300)
        dec_time = time.time() - t0

        if dec.returncode != 0:
            stderr = dec.stderr.decode("utf-8", errors="replace")
            print(f"  VVC YUV420 QP={qp} decode FAILED: {stderr[-500:]}")
            return

        # Read decoded YUV420 — only Y plane matters
        decoded_raw = np.fromfile(dec_yuv_path, dtype=np.uint8)
        frame_size_420 = h * w + 2 * (h // 2) * (w // 2)  # Y + U + V
        decoded_pixels = np.zeros((n, h, w), dtype=np.uint8)
        for i in range(n):
            offset = i * frame_size_420
            decoded_pixels[i] = decoded_raw[offset:offset + h * w].reshape(h, w)

        decoded_classes = gray_to_classes(decoded_pixels)
        acc = measure_accuracy(masks, decoded_classes)

        record(f"VVC QP={qp} (yuv420)", size, acc, enc_time, dec_time)


def bench_rle_zlib(masks: np.ndarray) -> None:
    """Custom RLE + zlib: run-length encode each row, then compress."""
    n, h, w = masks.shape
    t0 = time.time()

    # RLE encode all frames
    rle_data = bytearray()
    # Header: frame count, height, width
    rle_data.extend(struct.pack("<III", n, h, w))

    for frame_idx in range(n):
        for row in range(h):
            row_data = masks[frame_idx, row]
            # RLE: (value, count) pairs, value is 0-4, count fits in uint16
            runs = []
            current_val = row_data[0]
            count = 1
            for col in range(1, w):
                if row_data[col] == current_val and count < 65535:
                    count += 1
                else:
                    runs.append((current_val, count))
                    current_val = row_data[col]
                    count = 1
            runs.append((current_val, count))

            # Pack: num_runs (uint16), then (val: uint8, count: uint16) per run
            rle_data.extend(struct.pack("<H", len(runs)))
            for val, cnt in runs:
                rle_data.extend(struct.pack("<BH", val, cnt))

    enc_time = time.time() - t0

    # Compress with zlib
    compressed = zlib.compress(bytes(rle_data), level=9)
    size = len(compressed)

    # Decode
    t0 = time.time()
    decompressed = zlib.decompress(compressed)
    buf = memoryview(decompressed)
    offset = 0
    dn, dh, dw = struct.unpack_from("<III", buf, offset)
    offset += 12
    decoded = np.zeros((dn, dh, dw), dtype=np.int64)

    for frame_idx in range(dn):
        for row in range(dh):
            num_runs, = struct.unpack_from("<H", buf, offset)
            offset += 2
            col = 0
            for _ in range(num_runs):
                val, cnt = struct.unpack_from("<BH", buf, offset)
                offset += 3
                decoded[frame_idx, row, col:col + cnt] = val
                col += cnt

    dec_time = time.time() - t0
    acc = measure_accuracy(masks, decoded)
    record("RLE + zlib (level 9)", size, acc, enc_time, dec_time)


def bench_raw_zlib(masks: np.ndarray) -> None:
    """Raw class labels compressed with zlib."""
    t0 = time.time()
    raw = masks.astype(np.uint8).tobytes()
    compressed = zlib.compress(raw, level=9)
    enc_time = time.time() - t0
    size = len(compressed)

    t0 = time.time()
    decompressed = zlib.decompress(compressed)
    decoded = np.frombuffer(decompressed, dtype=np.uint8).reshape(masks.shape).astype(np.int64)
    dec_time = time.time() - t0

    acc = measure_accuracy(masks, decoded)
    record("Raw + zlib (level 9)", size, acc, enc_time, dec_time)


def bench_delta_zlib(masks: np.ndarray) -> None:
    """Delta encoding (temporal) + zlib. First frame stored raw, rest as diffs."""
    n, h, w = masks.shape
    t0 = time.time()

    buf = bytearray()
    buf.extend(struct.pack("<III", n, h, w))
    # First frame
    buf.extend(masks[0].astype(np.uint8).tobytes())
    # Delta frames (wrapping difference mod 256)
    for i in range(1, n):
        delta = (masks[i].astype(np.uint8).astype(np.int16) - masks[i - 1].astype(np.uint8).astype(np.int16))
        # Most deltas are 0 for our temporally coherent masks
        buf.extend(delta.astype(np.int8).tobytes())

    compressed = zlib.compress(bytes(buf), level=9)
    enc_time = time.time() - t0
    size = len(compressed)

    # Decode
    t0 = time.time()
    decompressed = zlib.decompress(compressed)
    arr = np.frombuffer(decompressed[12:12 + h * w], dtype=np.uint8).reshape(h, w).astype(np.int64)
    decoded = np.zeros_like(masks)
    decoded[0] = arr
    offset = 12 + h * w
    for i in range(1, n):
        delta = np.frombuffer(decompressed[offset:offset + h * w], dtype=np.int8).reshape(h, w).astype(np.int64)
        decoded[i] = np.clip(decoded[i - 1] + delta, 0, NUM_CLASSES - 1)
        offset += h * w
    dec_time = time.time() - t0

    acc = measure_accuracy(masks, decoded)
    record("Delta + zlib (level 9)", size, acc, enc_time, dec_time)


def bench_png_zip(masks: np.ndarray) -> None:
    """PNG sequence packed into a zip file."""
    import zipfile
    from PIL import Image

    n, h, w = masks.shape
    pixels = scale_to_gray(masks)

    t0 = time.time()
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
        zip_path = f.name

    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            for i in range(n):
                img = Image.fromarray(pixels[i], mode="L")
                buf = io.BytesIO()
                img.save(buf, format="PNG", optimize=True)
                zf.writestr(f"{i:04d}.png", buf.getvalue())
        enc_time = time.time() - t0
        size = os.path.getsize(zip_path)

        # Decode
        t0 = time.time()
        decoded = np.zeros_like(masks)
        with zipfile.ZipFile(zip_path, "r") as zf:
            for i in range(n):
                data = zf.read(f"{i:04d}.png")
                img = Image.open(io.BytesIO(data))
                decoded[i] = gray_to_classes(np.array(img))
        dec_time = time.time() - t0

        acc = measure_accuracy(masks, decoded)
        record("PNG sequence + zip", size, acc, enc_time, dec_time)
    finally:
        os.unlink(zip_path)


def bench_palettized_png_zip(masks: np.ndarray) -> None:
    """Palettized (indexed color) PNG sequence in zip — optimal for 5-class data."""
    import zipfile
    from PIL import Image

    n, h, w = masks.shape

    t0 = time.time()
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
        zip_path = f.name

    # Create a 5-color palette
    palette = [0] * 768  # 256 * 3
    for c in range(NUM_CLASSES):
        palette[c * 3] = c * SCALE_FACTOR
        palette[c * 3 + 1] = c * SCALE_FACTOR
        palette[c * 3 + 2] = c * SCALE_FACTOR

    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            for i in range(n):
                img = Image.fromarray(masks[i].astype(np.uint8), mode="P")
                img.putpalette(palette)
                buf = io.BytesIO()
                img.save(buf, format="PNG", optimize=True)
                zf.writestr(f"{i:04d}.png", buf.getvalue())
        enc_time = time.time() - t0
        size = os.path.getsize(zip_path)

        # Decode
        t0 = time.time()
        decoded = np.zeros_like(masks)
        with zipfile.ZipFile(zip_path, "r") as zf:
            for i in range(n):
                data = zf.read(f"{i:04d}.png")
                img = Image.open(io.BytesIO(data))
                decoded[i] = np.array(img).astype(np.int64)
        dec_time = time.time() - t0

        acc = measure_accuracy(masks, decoded)
        record("Palettized PNG + zip", size, acc, enc_time, dec_time)
    finally:
        os.unlink(zip_path)


def bench_apng(masks: np.ndarray) -> None:
    """Animated PNG (APNG) — lossless, good for palettized."""
    from PIL import Image

    n, h, w = masks.shape
    pixels = scale_to_gray(masks)

    with tempfile.NamedTemporaryFile(suffix=".apng", delete=False) as f:
        out_path = f.name

    try:
        t0 = time.time()
        frames = [Image.fromarray(pixels[i], mode="L") for i in range(n)]
        frames[0].save(out_path, save_all=True, append_images=frames[1:],
                       duration=1000 // FPS, loop=0, format="PNG")
        enc_time = time.time() - t0
        size = os.path.getsize(out_path)

        # Decode
        t0 = time.time()
        img = Image.open(out_path)
        decoded = np.zeros_like(masks)
        for i in range(n):
            try:
                img.seek(i)
                decoded[i] = gray_to_classes(np.array(img.convert("L")))
            except EOFError:
                break
        dec_time = time.time() - t0

        acc = measure_accuracy(masks, decoded)
        record("APNG (animated PNG)", size, acc, enc_time, dec_time)
    finally:
        os.unlink(out_path)


def bench_h265(masks: np.ndarray, crf: int) -> None:
    """Benchmark H.265/HEVC for comparison."""
    pixels = scale_to_gray(masks)
    raw = pixels.tobytes()
    n, h, w = masks.shape

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        out_path = f.name

    try:
        cmd = [
            "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{w}x{h}", "-pix_fmt", "gray", "-r", str(FPS),
            "-i", "pipe:0",
            "-c:v", "libx265", "-crf", str(crf), "-preset", "medium",
            "-x265-params", "no-sao=1",  # disable sample adaptive offset (like AV1 filter disable)
            "-pix_fmt", "yuv420p", "-an", out_path,
        ]
        t0 = time.time()
        proc = subprocess.run(cmd, input=raw, capture_output=True, timeout=300)
        enc_time = time.time() - t0
        if proc.returncode != 0:
            print(f"  H.265 CRF={crf} FAILED: {proc.stderr[-500:]}")
            return

        size = os.path.getsize(out_path)

        # Decode
        cmd_dec = [
            "ffmpeg", "-i", out_path, "-f", "rawvideo", "-pix_fmt", "gray",
            "-v", "error", "pipe:1",
        ]
        t0 = time.time()
        dec = subprocess.run(cmd_dec, capture_output=True, timeout=300)
        dec_time = time.time() - t0

        decoded_pixels = np.frombuffer(dec.stdout, dtype=np.uint8).reshape(n, h, w)
        decoded_classes = gray_to_classes(decoded_pixels)
        acc = measure_accuracy(masks, decoded_classes)

        record(f"H.265 CRF={crf} (no SAO)", size, acc, enc_time, dec_time)
    finally:
        os.unlink(out_path)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    masks = generate_realistic_masks()
    raw_size = masks.size  # 1200 * 384 * 512 = 235,929,600 bytes (as uint8)
    print(f"\nRaw mask size: {raw_size:,} bytes ({raw_size / 1024 / 1024:.1f} MB)\n")

    print("=" * 80)
    print("BENCHMARKING MASK CODECS")
    print("=" * 80)

    # ── AV1 (current pipeline) ──────────────────────────────────────────────
    print("\n── AV1 (libsvtav1) ──")
    bench_av1(masks, crf=20, label_suffix="")
    bench_av1(masks, crf=20, disable_filters=True)
    bench_av1(masks, crf=15, disable_filters=True)
    bench_av1(masks, crf=10, disable_filters=True)

    # ── H.265 (reference) ──────────────────────────────────────────────────
    print("\n── H.265 (libx265) ──")
    bench_h265(masks, crf=20)
    bench_h265(masks, crf=15)

    # ── VVC (H.266) ────────────────────────────────────────────────────────
    print("\n── VVC/H.266 (vvencapp) ──")
    # QP range roughly maps: VVC QP ~= AV1 CRF + offset
    # For masks, test QP 20-32
    bench_vvc(masks, qp=32)
    bench_vvc(masks, qp=27)
    bench_vvc(masks, qp=22)
    bench_vvc(masks, qp=17)

    print("\n── VVC/H.266 YUV420 (vvencapp) ──")
    bench_vvc_yuv420(masks, qp=27)
    bench_vvc_yuv420(masks, qp=22)

    # ── Lossless alternatives ──────────────────────────────────────────────
    print("\n── Lossless / custom codecs ──")
    bench_raw_zlib(masks)
    bench_delta_zlib(masks)
    bench_rle_zlib(masks)

    print("\n── PNG-based ──")
    bench_png_zip(masks)
    bench_palettized_png_zip(masks)
    bench_apng(masks)

    # ── Summary table ──────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print(f"{'Method':<35} {'Size (KB)':>10} {'Accuracy':>10} {'Enc (s)':>10} {'Dec (s)':>10} {'vs AV1 CRF20':>14}")
    print("-" * 100)

    # Sort by size
    RESULTS.sort(key=lambda x: x["size_bytes"])

    baseline_size = next((r["size_bytes"] for r in RESULTS if "AV1 CRF=20 " in r["name"] and "no filters" not in r["name"]), RESULTS[0]["size_bytes"])

    for r in RESULTS:
        ratio = r["size_bytes"] / baseline_size if baseline_size > 0 else 0
        acc_str = f"{r['accuracy']:.6f}" if r["accuracy"] < 1.0 else "LOSSLESS"
        print(f"{r['name']:<35} {r['size_kb']:>9.1f} {acc_str:>10} {r['encode_s']:>9.2f} {r['decode_s']:>9.2f} {ratio:>13.2f}x")

    print("\n" + "=" * 100)
    print("NOTE: Accuracy < 1.0 means some pixels changed class after encode/decode round-trip.")
    print("      For our pipeline, accuracy > 0.999 is acceptable (masks feed neural renderer).")
    print(f"      Raw uncompressed size: {raw_size/1024:.1f} KB")

    # Find best lossy and best lossless
    lossless = [r for r in RESULTS if r["accuracy"] >= 1.0]
    near_lossless = [r for r in RESULTS if r["accuracy"] >= 0.999]

    if lossless:
        best_ll = min(lossless, key=lambda x: x["size_bytes"])
        print(f"\n  BEST LOSSLESS: {best_ll['name']} at {best_ll['size_kb']:.1f} KB")

    if near_lossless:
        best_nl = min(near_lossless, key=lambda x: x["size_bytes"])
        print(f"  BEST NEAR-LOSSLESS (>99.9%): {best_nl['name']} at {best_nl['size_kb']:.1f} KB")

    best_overall = min(RESULTS, key=lambda x: x["size_bytes"])
    print(f"  SMALLEST OVERALL: {best_overall['name']} at {best_overall['size_kb']:.1f} KB (accuracy={best_overall['accuracy']:.6f})")


if __name__ == "__main__":
    main()
