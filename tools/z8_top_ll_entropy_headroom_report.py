#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Measure Z8 top-LL payload compression headroom.

After Z8 detail coefficients are quantized/entropy-coded, the next binding
wavelet-payload surface is the per-pair top-LL float32 approximation stored in
each pair blob. This read-only advisory report measures top-LL operating
points: raw float32, byte-shuffled float32, direct quantized top-LL, and
frame1-as-residual-from-frame0 quantized coding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import brotli  # type: ignore[import-not-found]
import numpy as np

from tac.substrates.z8_hierarchical_predictive_coding.archive import parse_archive
from tac.substrates.z8_hierarchical_predictive_coding.canonical_quadruple_binding import (
    _DETAIL_CODEC_NAMES,
    _DETAIL_CODEC_QI16_DENSE,
    _DETAIL_CODEC_ZZ16_BYTEPLANE,
    _encode_f32_byteshuffle_payload,
    _encode_zz16_byteplane,
    parse_pair_blobs_from_wavelet_blob,
)

NON_PROMOTABLE_MARKERS: dict[str, Any] = {
    "axis_tag": "[macOS-CPU advisory]",
    "evidence_grade": "macOS-CPU-advisory",
    "score_claim": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "promotable": False,
}


def _brotli_len(data: bytes, *, quality: int = 1) -> int:
    return len(brotli.compress(data, quality=quality))


def _entropy_bytes(values: np.ndarray) -> float:
    flat = np.asarray(values).reshape(-1)
    if flat.size == 0:
        return 0.0
    _, counts = np.unique(flat, return_counts=True)
    probs = counts.astype(np.float64) / float(flat.size)
    return float(-np.sum(probs * np.log2(probs)) * flat.size / 8.0)


@dataclass(frozen=True)
class TopLLSurface:
    name: str
    values: np.ndarray
    reconstruction_base: np.ndarray | None = None


def _fast_quantized_payload(q: np.ndarray) -> tuple[int, bytes]:
    dense = np.asarray(q, dtype="<i2").tobytes(order="C")
    byteplane = _encode_zz16_byteplane(q)
    choices = [
        (_DETAIL_CODEC_QI16_DENSE, dense),
        (_DETAIL_CODEC_ZZ16_BYTEPLANE, byteplane),
    ]
    return min(choices, key=lambda item: (_brotli_len(item[1], quality=1), len(item[1])))


def _quantize_measure(surface: TopLLSurface, step: float) -> dict[str, Any]:
    values = np.asarray(surface.values, dtype=np.float32)
    q = np.rint(np.nan_to_num(values / np.float32(step))).clip(-32768, 32767).astype("<i2")
    method, payload = _fast_quantized_payload(q)
    dequant = q.astype(np.float32) * np.float32(step)
    if surface.reconstruction_base is not None:
        recon = np.asarray(surface.reconstruction_base, dtype=np.float32) + dequant
        target = np.asarray(surface.reconstruction_base, dtype=np.float32) + values
        distortion = float(np.mean((recon - target) ** 2))
    else:
        distortion = float(np.mean((dequant - values) ** 2))
    return {
        "quant_step": float(step),
        "method": _DETAIL_CODEC_NAMES.get(method, f"method_{method}"),
        "payload_bytes": len(payload),
        "payload_brotli_bytes": _brotli_len(payload),
        "order0_symbol_floor_bytes": round(_entropy_bytes(q), 1),
        "distortion_mse": distortion,
        "nonzero_fraction": float(np.count_nonzero(q) / q.size) if q.size else 0.0,
        "bytes_per_value": round(len(payload) / q.size, 6) if q.size else 0.0,
        "brotli_bytes_per_value": round(_brotli_len(payload) / q.size, 6) if q.size else 0.0,
    }


def _surface_report(surface: TopLLSurface, *, quant_steps: list[float]) -> dict[str, Any]:
    values = np.asarray(surface.values, dtype=np.float32)
    raw = values.tobytes(order="C")
    byte_shuffle = _encode_f32_byteshuffle_payload(values)
    quant = [_quantize_measure(surface, step) for step in quant_steps]
    best_by_brotli = min(quant, key=lambda row: int(row["payload_brotli_bytes"]))
    return {
        "surface": surface.name,
        "shape": list(values.shape),
        "value_count": int(values.size),
        "abs_mean": float(np.mean(np.abs(values))) if values.size else 0.0,
        "abs_max": float(np.max(np.abs(values))) if values.size else 0.0,
        "raw_f32_bytes": len(raw),
        "raw_f32_brotli_bytes": _brotli_len(raw),
        "f32_byteshuffle_brotli_bytes": _brotli_len(byte_shuffle),
        "quant_sweep": quant,
        "best_quantized_by_brotli": best_by_brotli,
    }


def build_report(*, archive_path: Path, quant_steps: list[float], max_pairs: int | None) -> dict[str, Any]:
    archive_bytes = archive_path.read_bytes()
    parsed = parse_archive(archive_bytes)
    pyramids = parse_pair_blobs_from_wavelet_blob(parsed.wavelet_coeffs_blob)
    if max_pairs is not None:
        pyramids = pyramids[: int(max_pairs)]
    frame0 = np.stack([np.asarray(p["frame_0_top_ll"], dtype=np.float32) for p in pyramids], axis=0)
    frame1 = np.stack([np.asarray(p["frame_1_top_ll"], dtype=np.float32) for p in pyramids], axis=0)
    residual = frame1 - frame0
    surfaces = [
        TopLLSurface("frame0_top_ll", frame0),
        TopLLSurface("frame1_top_ll", frame1),
        TopLLSurface("both_frames_top_ll_concat", np.concatenate([frame0, frame1], axis=0)),
        TopLLSurface("frame1_minus_frame0_residual", residual, reconstruction_base=frame0),
    ]
    surface_reports = [_surface_report(surface, quant_steps=quant_steps) for surface in surfaces]
    copy_frame0_mse = float(np.mean((frame1 - frame0) ** 2))
    return {
        "schema": "z8_top_ll_entropy_headroom_report.v1",
        "tool": "tools/z8_top_ll_entropy_headroom_report.py",
        **NON_PROMOTABLE_MARKERS,
        "archive_path": str(archive_path),
        "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "archive_total_bytes": len(archive_bytes),
        "wavelet_blob_bytes": len(parsed.wavelet_coeffs_blob),
        "pairs_measured": len(pyramids),
        "total_pairs_in_archive": parsed.num_pairs,
        "quant_steps": quant_steps,
        "brotli_probe_quality": 1,
        "frame1_copy_from_frame0_zero_byte_mse": copy_frame0_mse,
        "surfaces": surface_reports,
        "interpretation": {
            "binding_surface": "top-LL float32 approximation after detail payload collapse",
            "direct_quantization": "quantize top-LL values and entropy-code symbols",
            "conditional_residual": "store frame1 top-LL residual conditioned on exact frame0 top-LL",
            "promotion_rule": "advisory only until a materializer emits byte-closed archive and full-video local replay accepts",
        },
    }


def _parse_quant_steps(raw: str) -> list[float]:
    vals = [float(x) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("at least one quant step is required")
    for value in vals:
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"invalid quant step: {value}")
    return vals


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument(
        "--quant-steps",
        default="0.00390625,0.0078125,0.015625,0.03125,0.0625,0.125,0.25",
    )
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--out-json", required=True, type=Path)
    args = parser.parse_args(argv)

    report = build_report(
        archive_path=args.archive.resolve(),
        quant_steps=_parse_quant_steps(args.quant_steps),
        max_pairs=args.max_pairs,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "archive_bytes": report["archive_total_bytes"],
                "pairs_measured": report["pairs_measured"],
                "frame1_copy_from_frame0_zero_byte_mse": report[
                    "frame1_copy_from_frame0_zero_byte_mse"
                ],
                "best_by_surface": {
                    row["surface"]: {
                        "quant_step": row["best_quantized_by_brotli"]["quant_step"],
                        "payload_brotli_bytes": row["best_quantized_by_brotli"][
                            "payload_brotli_bytes"
                        ],
                        "distortion_mse": row["best_quantized_by_brotli"]["distortion_mse"],
                    }
                    for row in report["surfaces"]
                },
                "out_json": str(args.out_json),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
