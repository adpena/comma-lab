# SPDX-License-Identifier: MIT
"""Build + decode-verify the full-n600 V10 JXL-lossless archive (RATE-CRUSH R0).

Chain: C1 prepare chunks (sha-custodied) -> exact uint8 planes -> production
archive with y codec ``jxl-lossless-plane.v1`` -> read-back -> parse (deep
fail-closed per-plane SHA custody) -> two-plane expansion -> byte equality
against the source planes AND the frozen C1 full-custody Y0/Y1 SHAs that the
officially-scored capstone row was measured on.

Byte identity of the decoded planes pins the distortion terms to the official
report exactly; only the rate term moves.  The implied S printed here is an
arithmetic consequence of that identity, NOT a new score measurement.

Axis: [macOS-CPU local rate measurement] NON-PROMOTABLE; score_claim=false.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from tac.codec.v10_jxl_plane_codec import CODEC_ID as JXL_PLANE_Y_CODEC_ID
from tac.witness_dsl.v10_production_receiver import (
    build_production_archive,
    decode_y_plane_pair,
    parse_packet,
)
from tac.witness_dsl.v10_two_plane_timing_receiver import (
    FULL_PAIR_COUNT,
    FULL_Y0_SHA256,
    FULL_Y1_SHA256,
    _validate_packet,
)

H, W, C = 384, 512, 3
CHUNK_PAIRS = 12
SCORE_NORMALIZER = 37_545_489
# Official capstone report components (600 samples, upstream/evaluate.py, cpu):
# <SSD>/evidence/c1_two_plane_receiver_20260719/capstone_eval/report.txt
OFFICIAL_D_SEG = 0.00015196
OFFICIAL_D_POSE = 0.00010184


def _load_chunk(chunks_dir: Path, index: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
    manifest = json.loads((chunks_dir / f"chunk-{index:04d}.manifest.json").read_text())
    y0_raw = (chunks_dir / f"chunk-{index:04d}.y0.bin").read_bytes()
    y1_raw = (chunks_dir / f"chunk-{index:04d}.y1.bin").read_bytes()
    if hashlib.sha256(y0_raw).hexdigest() != manifest["y0_sha256"]:
        raise SystemExit(f"chunk {index}: y0 sha custody failure")
    if hashlib.sha256(y1_raw).hexdigest() != manifest["y1_sha256"]:
        raise SystemExit(f"chunk {index}: y1 sha custody failure")
    pair_ids = [int(p) for p in manifest["pair_ids"]]
    n = len(pair_ids)
    return (
        np.frombuffer(y0_raw, dtype=np.uint8).reshape(n, H, W, C),
        np.frombuffer(y1_raw, dtype=np.uint8).reshape(n, H, W, C),
        pair_ids,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--effort", type=int, default=9)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    chunk_count = FULL_PAIR_COUNT // CHUNK_PAIRS
    all_y0, all_y1, all_ids = [], [], []
    for index in range(chunk_count):
        y0, y1, pair_ids = _load_chunk(args.chunks_dir, index)
        all_y0.append(y0)
        all_y1.append(y1)
        all_ids.extend(pair_ids)
    y0 = np.ascontiguousarray(np.concatenate(all_y0))
    y1 = np.ascontiguousarray(np.concatenate(all_y1))
    if all_ids != list(range(FULL_PAIR_COUNT)):
        raise SystemExit("chunk pair ids are not exactly 0..599")
    y0_sha = hashlib.sha256(y0.tobytes(order="C")).hexdigest()
    y1_sha = hashlib.sha256(y1.tobytes(order="C")).hexdigest()
    if y0_sha != FULL_Y0_SHA256 or y1_sha != FULL_Y1_SHA256:
        raise SystemExit("concatenated planes differ from the frozen C1 full custody")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = args.output_dir / "archive.zip"
    t0 = time.time()
    result = build_production_archive(
        y1,
        archive_path=archive_path,
        camera_height=874,
        camera_width=1164,
        y_codec_id=JXL_PLANE_Y_CODEC_ID,
        frame0_y_planes=y0,
        jxl_effort=args.effort,
        jxl_workers=args.workers,
    )
    build_seconds = time.time() - t0

    # Independent read-back decode-verify (fresh parse, deep per-plane custody).
    t0 = time.time()
    import zipfile

    with zipfile.ZipFile(archive_path) as archive:
        packet_bytes = archive.read("0.bin")
    parsed = parse_packet(packet_bytes)
    _validate_packet(parsed)
    pair = decode_y_plane_pair(parsed)
    decode_seconds = time.time() - t0
    if not (np.array_equal(pair.frame0, y0) and np.array_equal(pair.frame1, y1)):
        raise SystemExit("decoded planes differ from custody source — byte identity FAILED")
    decoded_y0_sha = hashlib.sha256(pair.frame0.tobytes(order="C")).hexdigest()
    decoded_y1_sha = hashlib.sha256(pair.frame1.tobytes(order="C")).hexdigest()
    if decoded_y0_sha != FULL_Y0_SHA256 or decoded_y1_sha != FULL_Y1_SHA256:
        raise SystemExit("decoded plane SHAs differ from frozen C1 custody")

    rate = 25.0 * result.archive_bytes / SCORE_NORMALIZER
    implied_s = 100.0 * OFFICIAL_D_SEG + (10.0 * OFFICIAL_D_POSE) ** 0.5 + rate
    receipt = {
        "schema": "v10_ratecrush_jxl_n600_archive.v1",
        "axis": "[macOS-CPU local rate measurement] NON-PROMOTABLE",
        "score_claim": False,
        "promotion_eligible": False,
        "y_codec_id": JXL_PLANE_Y_CODEC_ID,
        "jxl_effort": args.effort,
        "pair_count": FULL_PAIR_COUNT,
        "archive_path": str(result.archive_path),
        "archive_bytes": result.archive_bytes,
        "archive_sha256": result.archive_sha256,
        "packet_bytes": result.packet_bytes,
        "packet_sha256": result.packet_sha256,
        "bytes_per_pair": result.archive_bytes / FULL_PAIR_COUNT,
        "decode_exact_vs_custody_source": True,
        "decoded_y0_sha256": decoded_y0_sha,
        "decoded_y1_sha256": decoded_y1_sha,
        "frozen_custody_match": True,
        "rate_term": rate,
        "official_d_seg": OFFICIAL_D_SEG,
        "official_d_pose": OFFICIAL_D_POSE,
        "implied_s_from_byte_identity": implied_s,
        "implied_s_note": (
            "distortion terms pinned by decoded-plane byte identity to the officially-scored "
            "C1 planes; implied S is arithmetic, not a new evaluate.py row"
        ),
        "build_seconds": round(build_seconds, 1),
        "readback_decode_seconds": round(decode_seconds, 1),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=1, sort_keys=True))
    print(json.dumps(receipt, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
