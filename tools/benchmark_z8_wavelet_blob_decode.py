#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Benchmark Z8 wavelet-blob decode throughput on a byte-closed archive."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from tac.substrates.z8_hierarchical_predictive_coding.archive import parse_archive
from tac.substrates.z8_hierarchical_predictive_coding.canonical_quadruple_binding import (
    parse_pair_blobs_from_wavelet_blob,
    summarize_wavelet_blob_detail_codecs,
)

NON_PROMOTABLE_MARKERS: dict[str, Any] = {
    "evidence_grade": "macOS-CPU-advisory",
    "axis_tag": "[macOS-CPU advisory]",
    "score_claim": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "promotable": False,
}


def benchmark_archive_decode(
    archive_path: Path,
    *,
    repeat: int,
    auth_eval_window_seconds: float,
) -> dict[str, Any]:
    if repeat <= 0:
        raise ValueError("repeat must be positive")
    archive_bytes = archive_path.read_bytes()
    arc = parse_archive(archive_bytes)
    summary = summarize_wavelet_blob_detail_codecs(arc.wavelet_coeffs_blob)
    timings: list[float] = []
    pair_count = 0
    for _ in range(repeat):
        start = time.perf_counter()
        pairs = parse_pair_blobs_from_wavelet_blob(arc.wavelet_coeffs_blob)
        elapsed = time.perf_counter() - start
        timings.append(elapsed)
        pair_count = len(pairs)
    best = min(timings)
    mean = sum(timings) / len(timings)
    return {
        "schema": "z8_wavelet_blob_decode_benchmark.v1",
        "purpose": "Decode-throughput budget check for Z8 detail entropy codecs.",
        **NON_PROMOTABLE_MARKERS,
        "archive_path": archive_path.as_posix(),
        "archive_bytes": len(archive_bytes),
        "wavelet_blob_bytes": len(arc.wavelet_coeffs_blob),
        "pair_count": int(pair_count),
        "repeat": int(repeat),
        "decode_seconds_best": float(best),
        "decode_seconds_mean": float(mean),
        "pairs_per_second_best": float(pair_count / best) if best > 0 else None,
        "auth_eval_window_seconds": float(auth_eval_window_seconds),
        "auth_eval_window_fraction_best": (
            float(best / auth_eval_window_seconds) if auth_eval_window_seconds > 0 else None
        ),
        "detail_codec_summary": summary,
        "blockers": [],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--auth-eval-window-seconds", type=float, default=1800.0)
    parser.add_argument("--out-json", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = benchmark_archive_decode(
        args.archive,
        repeat=int(args.repeat),
        auth_eval_window_seconds=float(args.auth_eval_window_seconds),
    )
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
