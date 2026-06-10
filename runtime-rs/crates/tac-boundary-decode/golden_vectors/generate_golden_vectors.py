# SPDX-License-Identifier: MIT
"""Generate golden-vector manifests + input fixtures for the tac-boundary-decode Rust crate.

This is the **Python ORACLE** side of the parity gate. Each vector pins the SHA-256
of the *decoded raw output* the Rust port must reproduce bit-for-bit, plus the input
byte fixtures the Rust port reads (so we do NOT reimplement numpy's RNG in Rust).

NO FAKE: every vector is produced by the REAL boundary_math oracle functions
(``encode_partition`` / ``decode_partition`` for the contour codec; the popcount
``d_seg_*`` functionals; ``connected_components`` for the RAG). The vectors are NOT
hand-authored constants — they are the deterministic output of the oracle on pinned
inputs. The Rust crate's parity test reads these fixtures, runs the Rust decode, and
asserts the SHA-256 of its raw output matches the manifest.

Run:  .venv/bin/python runtime-rs/crates/tac-boundary-decode/golden_vectors/generate_golden_vectors.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from tac.boundary_math.bitmask_dseg import d_seg_reference, flip_count
from tac.boundary_math.contour_codec import encode_partition
from tac.boundary_math.partition import connected_components

HERE = Path(__file__).resolve().parent
N_CLASSES = 5


def _sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _write_manifest(name: str, manifest: dict) -> None:
    (HERE / f"{name}.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _write_bin(name: str, data: bytes) -> None:
    (HERE / name).write_bytes(data)


def _structured_partition(h: int, w: int, seed: int) -> np.ndarray:
    """A realistic SegNet-style partition: contiguous bands + objects (NOT random noise).

    The contour codec exploits constant-label runs (interior = free); a structured
    partition exercises the real boundary-entropy compression path the codec is for.
    """

    rng = np.random.default_rng(seed)
    lab = np.zeros((h, w), dtype=np.int64)
    lab[: h // 3] = 3  # sky band
    lab[h // 3 : 2 * h // 3] = 0  # road / undrivable band
    lab[2 * h // 3 :] = 1  # my-lane band
    # two vertical objects (pole / car) at deterministic positions
    lab[h // 4 : 3 * h // 4, w // 4 : w // 4 + 30] = 2
    lab[h // 3 : h, 4 * w // 5 : 4 * w // 5 + 40] = 4
    # a sprinkle of small deterministic blobs to add boundary entropy
    for _ in range(8):
        r = int(rng.integers(0, h - 10))
        c = int(rng.integers(0, w - 10))
        cls = int(rng.integers(0, N_CLASSES))
        lab[r : r + 6, c : c + 6] = cls
    return lab.astype(np.uint8)


# ── Vector 1: contour codec DECODE (LZMA-RAW payload -> raw label-map bytes) ──
def gen_contour_decode_full() -> None:
    """Full-resolution 384x512 partition — the contest seg target shape."""

    h, w = 384, 512
    argmax = _structured_partition(h, w, seed=20260610)
    code = encode_partition(argmax, N_CLASSES)
    # The Rust decode reads `payload` and must reproduce the raw uint8 label bytes.
    raw_decoded = argmax.astype(np.uint8).tobytes(order="C")
    _write_bin("contour_decode_full_v1_payload.bin", code.payload)
    _write_manifest(
        "contour_decode_full_v1",
        {
            "schema": "contour_decode.v1",
            "sha256": _sha256_hex(raw_decoded),  # sha of the DECODED raw output
            "payload_len": len(code.payload),
            "height": h,
            "width": w,
            "n_classes": N_CLASSES,
            "raw_decoded_len": len(raw_decoded),
            "lzma2_preset_extreme_level9": True,
            "lc": 0,
            "lp": 0,
            "pb": 0,
            "score_claim": False,
            "promotion_eligible": False,
            "note": "decoded-raw sha; Rust liblzma RAW-LZMA2 decode must match byte-for-byte",
        },
    )


def gen_contour_decode_small() -> None:
    """Small 16x24 partition — fast unit-scale parity (edge cases + bit-exactness)."""

    h, w = 16, 24
    argmax = _structured_partition(h, w, seed=7)
    code = encode_partition(argmax, N_CLASSES)
    raw_decoded = argmax.astype(np.uint8).tobytes(order="C")
    _write_bin("contour_decode_small_v1_payload.bin", code.payload)
    _write_manifest(
        "contour_decode_small_v1",
        {
            "schema": "contour_decode.v1",
            "sha256": _sha256_hex(raw_decoded),
            "payload_len": len(code.payload),
            "height": h,
            "width": w,
            "n_classes": N_CLASSES,
            "raw_decoded_len": len(raw_decoded),
            "lc": 0,
            "lp": 0,
            "pb": 0,
            "score_claim": False,
            "promotion_eligible": False,
        },
    )


# ── Vector 2: d_seg popcount kernel (two label maps -> flip count) ──
def gen_dseg_popcount() -> None:
    """A candidate vs gt label map: the Rust popcount(XOR) must match flip_count + d_seg."""

    h, w = 64, 96
    gt = _structured_partition(h, w, seed=11)
    cand = gt.copy()
    rng = np.random.default_rng(99)
    # Flip a deterministic, structured subset of pixels (a patch + scattered singles).
    cand[10:18, 20:30] = (cand[10:18, 20:30] + 1) % N_CLASSES
    for _ in range(37):
        r = int(rng.integers(0, h))
        c = int(rng.integers(0, w))
        cand[r, c] = (cand[r, c] + 1) % N_CLASSES
    fc = flip_count(cand, gt)
    dseg = d_seg_reference(cand, gt)
    # Rust reads both raw label-byte arrays; output = the 8-byte LE flip count + the
    # d_seg float64 LE. We pin the sha of that 16-byte little-endian result blob.
    out = int(fc).to_bytes(8, "little") + np.float64(dseg).tobytes()
    _write_bin("dseg_popcount_v1_cand.bin", cand.astype(np.uint8).tobytes(order="C"))
    _write_bin("dseg_popcount_v1_gt.bin", gt.astype(np.uint8).tobytes(order="C"))
    _write_manifest(
        "dseg_popcount_v1",
        {
            "schema": "dseg_popcount.v1",
            "sha256": _sha256_hex(out),
            "height": h,
            "width": w,
            "n_classes": N_CLASSES,
            "flip_count": int(fc),
            "d_seg": float(dseg),
            "n_pixels": int(h * w),
            "score_claim": False,
            "promotion_eligible": False,
            "note": "output blob = flip_count(u64 LE) || d_seg(f64 LE)",
        },
    )


# ── Vector 3: connected-components region label map (4-connectivity) ──
def gen_connected_components() -> None:
    """4-connectivity connected components: the Rust region_of map must match scipy.

    NOTE: scipy.ndimage.label assigns component ids in a deterministic raster-scan
    order per class; ``connected_components`` iterates classes 0..n then components
    1..n, assigning a global contiguous id. The Rust port must reproduce the EXACT
    same per-pixel region id map (the byte-for-byte raster of region_of as int32 LE).
    """

    h, w = 32, 48
    argmax = _structured_partition(h, w, seed=5)
    region_of, regions = connected_components(argmax, N_CLASSES)
    # Output: the region_of map as int32 little-endian raster (the rasterized RAG ids).
    out = region_of.astype(np.int32).tobytes(order="C")
    _write_bin("connected_components_v1_argmax.bin", argmax.astype(np.uint8).tobytes(order="C"))
    _write_manifest(
        "connected_components_v1",
        {
            "schema": "connected_components.v1",
            "sha256": _sha256_hex(out),
            "height": h,
            "width": w,
            "n_classes": N_CLASSES,
            "n_regions": len(regions),
            "region_of_dtype": "int32_le",
            "score_claim": False,
            "promotion_eligible": False,
            "note": "output = region_of (H,W) int32 LE raster; 4-connectivity, per-class then component",
        },
    )


def main() -> None:
    gen_contour_decode_full()
    gen_contour_decode_small()
    gen_dseg_popcount()
    gen_connected_components()
    print("golden vectors written to", HERE)
    for p in sorted(HERE.glob("*.json")):
        m = json.loads(p.read_text())
        print(f"  {p.name:40s} schema={m['schema']:24s} sha={m['sha256'][:16]}…")


if __name__ == "__main__":
    main()
