#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""R6CAL: measure the description-compression floor of the ms2r_r3 solved records.

Walks the real ``chunk-*.predictor.bin`` predictor records, decomposes every pair's
byte budget into bootstrap / descriptor / residual / framing, measures empirical
entropy, races real general-purpose coders on sampled chunks, and prices the whole
description against the #603 planning box and the exact score dual.

Every number is measured from the artifacts on disk. Nothing is extrapolated.
No score claim: this prices bytes, it does not evaluate distortion.
"""

from __future__ import annotations

import argparse
import bz2
import collections
import hashlib
import json
import lzma
import struct
import zlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "r6cal_description_compression_floor.v1"
CHUNK_PREFIX = struct.Struct("<8sHB3xIIIH")
PAIR_PREFIX = struct.Struct("<IB3xIII32s32s32s32s")
PREDICTOR_MAGIC = b"TACV10PR"
PREDICTOR_MODES = {0: "PREVIOUS_PLANE_COPY", 1: "AFFINE6_Q12", 2: "SPATIAL_SMOOTH_121"}

SEG_SITES = 117_964_800          # 600 pairs x 512 x 384 scorer sites
UNCOMPRESSED_BYTES = 37_545_489  # upstream/videos denominator, measured
SCORE_LAMBDA_BYTE = 25.0 / UNCOMPRESSED_BYTES
SCORE_LAMBDA_SEG_ERROR = 100.0 / SEG_SITES


def _walk_records(chunk_path: Path) -> tuple[dict[str, int], list[dict[str, int]]]:
    blob = chunk_path.read_bytes()
    magic, _version, _tag, pair_count, height, width, channels = CHUNK_PREFIX.unpack_from(blob, 0)
    if magic != PREDICTOR_MAGIC:
        raise SystemExit(f"{chunk_path}: unexpected magic {magic!r}")
    geometry = {"height": int(height), "width": int(width), "channels": int(channels), "file_bytes": len(blob)}
    cursor = CHUNK_PREFIX.size
    records: list[dict[str, int]] = []
    for _ in range(int(pair_count)):
        pair_id, mode, boot, desc, resid, *_ = PAIR_PREFIX.unpack_from(blob, cursor)
        cursor += PAIR_PREFIX.size + boot + desc + resid
        records.append({
            "pair_id": int(pair_id), "mode": int(mode), "bootstrap": int(boot),
            "descriptor": int(desc), "residual": int(resid), "header": PAIR_PREFIX.size,
        })
    if cursor != len(blob):
        raise SystemExit(f"{chunk_path}: {len(blob) - cursor} unaccounted trailing bytes")
    return geometry, records


def _step_from_name(chunk_path: Path) -> int:
    """Parse the quantisation step out of ``chunk-NNNN.qS.predictor.bin``, fail-closed.

    An ``else``-catches-everything classifier would silently file an unknown family
    (e.g. a future ``q1``) under an existing step and corrupt the selected-record join.
    """
    parts = chunk_path.name.split(".")
    tags = [part for part in parts if part.startswith("q") and part[1:].isdigit()]
    if len(tags) != 1:
        raise SystemExit(f"{chunk_path}: expected exactly one qN step tag in the filename, found {tags}")
    return int(tags[0][1:])


def _entropy(sample: bytes) -> dict[str, float]:
    data = np.frombuffer(sample, dtype=np.uint8)
    counts = np.bincount(data, minlength=256).astype(np.float64)
    probs = counts[counts > 0] / data.size
    h0 = float(-(probs * np.log2(probs)).sum())
    pairs = data[:-1].astype(np.int32) * 256 + data[1:].astype(np.int32)
    joint = np.bincount(pairs, minlength=65536).astype(np.float64).reshape(256, 256)
    context = joint.sum(axis=1)
    live = context > 0
    conditional = joint[live] / context[live][:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        per_context = np.where(conditional > 0, -conditional * np.log2(conditional), 0.0).sum(axis=1)
    h1 = float(((context[live] / joint.sum()) * per_context).sum())
    return {"h0_bits_per_byte": h0, "h1_bits_per_byte": h1, "distinct_symbols": int((counts > 0).sum())}


def _race_coders(sample: bytes) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [{"coder": "RAW", "bytes": len(sample)}]
    rows.append({"coder": "zlib-9", "bytes": len(zlib.compress(sample, 9))})
    rows.append({"coder": "bz2-9", "bytes": len(bz2.compress(sample, 9))})
    rows.append({"coder": "lzma-9e", "bytes": len(lzma.compress(sample, preset=9 | lzma.PRESET_EXTREME))})
    try:
        import brotli
        rows.append({"coder": "brotli-q11", "bytes": len(brotli.compress(sample, quality=11))})
    except ImportError:
        rows.append({"coder": "brotli-q11", "bytes": None, "note": "brotli unavailable"})
    try:
        import zstandard
        rows.append({"coder": "zstd-19", "bytes": len(zstandard.ZstdCompressor(level=19).compress(sample))})
    except ImportError:
        rows.append({"coder": "zstd-19", "bytes": None, "note": "zstandard unavailable"})
    raw = len(sample)
    for row in rows:
        if row.get("bytes") is not None:
            row["ratio_vs_raw"] = row["bytes"] / raw
            row["delta_vs_raw"] = row["bytes"] - raw
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rate-dir", type=Path, required=True, help="stage_checkpoints/01_rate")
    parser.add_argument("--candidate-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--coder-sample-chunks", type=int, default=2)
    parser.add_argument("--box-bytes", type=int, action="append", default=[],
                        help="planning box byte target to price against. Repeatable.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    boxes = args.box_bytes or [154_524, 200_000]

    candidate = json.loads(args.candidate_json.read_text())
    selected = {row["pair_id"]: row["selected_step"] for row in candidate["selected_record_rows"]}
    chunks = sorted(args.rate_dir.glob("chunk-*.predictor.bin"))
    if not chunks:
        raise SystemExit(f"no predictor chunks under {args.rate_dir}")

    all_totals: collections.Counter[str] = collections.Counter()
    sel_totals: collections.Counter[str] = collections.Counter()
    per_step: dict[int, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    modes: collections.Counter[str] = collections.Counter()
    geometry: dict[str, int] = {}
    for chunk in chunks:
        step = _step_from_name(chunk)
        chunk_geometry, records = _walk_records(chunk)
        if geometry and {k: chunk_geometry[k] for k in ("height", "width", "channels")} != {
            k: geometry[k] for k in ("height", "width", "channels")
        }:
            raise SystemExit(f"{chunk}: geometry differs from earlier chunks; plane denominator is not uniform")
        geometry = chunk_geometry
        all_totals["file_bytes"] += geometry["file_bytes"]
        all_totals["chunk_header"] += CHUNK_PREFIX.size
        for record in records:
            modes[PREDICTOR_MODES.get(record["mode"], str(record["mode"]))] += 1
            for field in ("bootstrap", "descriptor", "residual", "header"):
                all_totals[field] += record[field]
            all_totals["records"] += 1
            if selected.get(record["pair_id"]) == step:
                for field in ("bootstrap", "descriptor", "residual", "header"):
                    sel_totals[field] += record[field]
                    per_step[step][field] += record[field]
                sel_totals["records"] += 1
                per_step[step]["records"] += 1

    plane_values = geometry["height"] * geometry["width"] * geometry["channels"]
    sel_payload = sum(sel_totals[f] for f in ("bootstrap", "descriptor", "residual", "header"))
    n_sel = sel_totals["records"]
    archive_bytes = int(candidate["archive"]["bytes"])

    entropy_rows, coder_rows = [], []
    for chunk in chunks[: max(0, args.coder_sample_chunks)]:
        sample = chunk.read_bytes()
        entropy_rows.append({"chunk": chunk.name, "bytes": len(sample), **_entropy(sample)})
        coder_rows.append({"chunk": chunk.name, "rows": _race_coders(sample)})

    break_even = SCORE_LAMBDA_SEG_ERROR / SCORE_LAMBDA_BYTE
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_axis": "[measured-on-artifact bytes; no distortion claim]",
        "source": {
            "rate_dir": str(args.rate_dir),
            "candidate_json": str(args.candidate_json),
            "candidate_sha256": hashlib.sha256(args.candidate_json.read_bytes()).hexdigest(),
            "chunk_files": len(chunks),
            "geometry": geometry,
        },
        "predictor_mode_histogram": dict(modes),
        "all_records_budget": dict(all_totals),
        "selected_records_budget": {
            **{k: int(v) for k, v in sel_totals.items()},
            "payload_bytes": sel_payload,
            "bytes_per_pair": sel_payload / n_sel,
            "share_bootstrap": sel_totals["bootstrap"] / sel_payload,
            "share_residual": sel_totals["residual"] / sel_payload,
            "bits_per_plane_value_bootstrap": sel_totals["bootstrap"] * 8 / (n_sel * plane_values),
            "bits_per_plane_value_residual": sel_totals["residual"] * 8 / (n_sel * plane_values),
            "bits_per_plane_value_total": sel_payload * 8 / (n_sel * plane_values),
        },
        "per_step_budget": {
            str(step): {**{k: int(v) for k, v in counts.items()},
                        "bytes_per_pair": sum(counts[f] for f in ("bootstrap", "descriptor", "residual", "header")) / counts["records"]}
            for step, counts in sorted(per_step.items())
        },
        "entropy": entropy_rows,
        "coder_race": coder_rows,
        "score_duals": {
            "lambda_byte_score_per_byte": SCORE_LAMBDA_BYTE,
            "lambda_seg_error_score_per_error": SCORE_LAMBDA_SEG_ERROR,
            "score_optimal_bytes_per_corrected_seg_error": break_even,
        },
        "box_pricing": [
            {
                "box_bytes": box,
                "archive_bytes": archive_bytes,
                "compression_factor_required": archive_bytes / box,
                "rate_term_now": 25.0 * archive_bytes / UNCOMPRESSED_BYTES,
                "rate_term_at_box": 25.0 * box / UNCOMPRESSED_BYTES,
                "bytes_per_pair_at_box": box / n_sel,
                "bits_per_plane_value_at_box": box * 8 / (n_sel * plane_values),
            }
            for box in boxes
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({"out": str(args.out),
                      "selected_payload_bytes": sel_payload,
                      "bits_per_plane_value_total": report["selected_records_budget"]["bits_per_plane_value_total"],
                      "score_optimal_bytes_per_error": break_even}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
