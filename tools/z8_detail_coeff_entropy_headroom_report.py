#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Z8 wavelet detail-coefficient entropy-headroom report — measurable $0 diagnostic.

`[macOS-CPU advisory]` NON-PROMOTABLE per CLAUDE.md "MPS auth eval is NOISE" +
Catalog #127/#192/#317/#323/#341. This tool makes NO contest-score claim. It is a
READ-ONLY diagnostic that measures, on a REAL byte-closed Z8HPC1 archive, how many
bytes the per-subband wavelet detail-coefficient coding leaves on the table — i.e.
the ``(fp/raw → brotli) − (quantize + entropy-code) − (Shannon floor)`` gap table.

WHY THIS EXISTS (operator directive 2026-05-31, "provide brotli precisely what it
needs with no signal loss" + "we can super optimize this part"): the Z8HPC1 wavelet
detail blob default schema stores each detail coefficient as **raw float32** and asks
brotli to compress it. The low mantissa bits of an analog detail coefficient are
essentially random, so brotli cannot exploit LZ77 / Huffman / context structure and
stores the survivors at ~4 bytes/coeff. This report quantifies the headroom that a
quantize → zigzag → per-subband-mode (dense int16 / zero-RLE / byteplane / range)
codec recovers, at MATCHED distortion, per subband, on REAL coefficients.

SISTER-DISJOINT (Catalog #340): this is a NEW read-only tool. It does NOT modify the
v2 pair-blob codec (``canonical_quadruple_binding.py``, codex's active lane). It
CONSUMES the canonical read-only decode + encoder primitives so every measurement is
apples-to-apples with the deployed pipeline (no re-implemented strawman encoder).

The Shannon floor is reported in two forms:
  - order-0:  H0/8 bytes/coeff over the full quantized symbol stream.
  - structured: (H_bin(p_nonzero) + p_nonzero·H_nonzero)/8 bytes/coeff — the floor a
    zero-RLE + zigzag coder operationally targets on a dead-zone-sparse subband.

Usage:
    .venv/bin/python tools/z8_detail_coeff_entropy_headroom_report.py \
        --archive experiments/results/z8_joint_p18_p19_deadzone_rate_attack/baseline/byte_closed_archive/0.bin \
        --num-pairs 8 --quant-steps 0.5,1.0,2.0,4.0 \
        --out-json .omx/research/z8_detail_entropy_headroom_<utc>.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Read-only canonical decode + encoder reuse (NOT modified; apples-to-apples with the
# deployed v2 pair-blob codec). Importing the canonical encoders avoids a re-implemented
# strawman that would not match the codec codex is tuning.
from tac.substrates.z8_hierarchical_predictive_coding.archive import parse_archive
from tac.substrates.z8_hierarchical_predictive_coding.canonical_quadruple_binding import (
    _DETAIL_CODEC_NAMES,
    _encode_f32_byteshuffle_payload,
    _encode_qi16_constriction_range,
    _encode_quantized_detail_payload,
    parse_pair_blobs_from_wavelet_blob,
)

_BROTLI_QUALITY = 11  # PR95-family L32 canonical (matches the deployed pair-blob path).

# Non-promotable custody markers (Catalog #127/#192/#317/#323/#341).
NON_PROMOTABLE_MARKERS: dict[str, Any] = {
    "evidence_grade": "macOS-CPU-advisory",
    "axis_tag": "[macOS-CPU advisory]",
    "score_claim": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "promotable": False,
}


def _brotli_len(payload: bytes) -> int:
    import brotli  # type: ignore[import-not-found]

    return len(brotli.compress(payload, quality=_BROTLI_QUALITY))


def _shannon_bits_per_symbol(counts: np.ndarray, total: int) -> float:
    """Order-0 Shannon entropy H = -sum p log2 p over a symbol histogram."""
    if total <= 0:
        return 0.0
    probs = counts[counts > 0].astype(np.float64) / float(total)
    return float(-np.sum(probs * np.log2(probs)))


def _structured_floor_bits_per_coeff(q: np.ndarray) -> float:
    """Floor a zero-RLE + zigzag coder targets on a sparse subband.

    = H_bin(p_nonzero)   [occupancy mask, bits/coeff]
      + p_nonzero * H_nonzero   [value entropy over nonzeros, amortized over all coeffs]
    """
    n = q.size
    if n == 0:
        return 0.0
    nz_mask = q != 0
    n_nz = int(np.count_nonzero(nz_mask))
    p_nz = n_nz / n
    # Binary entropy of the occupancy.
    h_bin = (
        -(p_nz * math.log2(p_nz) + (1.0 - p_nz) * math.log2(1.0 - p_nz))
        if 0.0 < p_nz < 1.0
        else 0.0
    )
    # Value entropy over nonzeros (amortized over ALL coeffs => * p_nz).
    if n_nz > 0:
        nz_vals = q[nz_mask]
        _, nz_counts = np.unique(nz_vals, return_counts=True)
        h_nz = _shannon_bits_per_symbol(nz_counts, n_nz)
    else:
        h_nz = 0.0
    return h_bin + p_nz * h_nz


@dataclass
class _QuantMeasurement:
    quant_step: float
    method_name: str
    payload_bytes_per_coeff: float
    payload_brotli_bytes_per_coeff: float
    static_range_bytes_per_coeff: float | None
    shannon_floor_order0_bpc: float
    shannon_floor_structured_bpc: float
    distortion_mse: float
    nonzero_fraction: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "quant_step": self.quant_step,
            "live_codec_method": self.method_name,
            "live_codec_bytes_per_coeff": round(self.payload_bytes_per_coeff, 5),
            "live_codec_brotli_bytes_per_coeff": round(self.payload_brotli_bytes_per_coeff, 5),
            "static_range_bytes_per_coeff": (
                round(self.static_range_bytes_per_coeff, 5)
                if self.static_range_bytes_per_coeff is not None
                else None
            ),
            "shannon_floor_order0_bytes_per_coeff": round(self.shannon_floor_order0_bpc, 5),
            "shannon_floor_structured_bytes_per_coeff": round(self.shannon_floor_structured_bpc, 5),
            "distortion_mse": float(self.distortion_mse),
            "nonzero_fraction": round(self.nonzero_fraction, 5),
        }


@dataclass
class _SubbandReport:
    key: str
    n_coeffs: int
    coeff_abs_mean: float
    coeff_abs_max: float
    raw_f32_bytes_per_coeff: float  # = 4.0 (uncompressed reference)
    raw_f32_brotli_bytes_per_coeff: float  # current deployed default cost
    byteshuffle_brotli_bytes_per_coeff: float  # lossless preconditioner cost
    quant: list[_QuantMeasurement] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "subband": self.key,
            "n_coeffs": self.n_coeffs,
            "coeff_abs_mean": float(self.coeff_abs_mean),
            "coeff_abs_max": float(self.coeff_abs_max),
            "raw_f32_uncompressed_bytes_per_coeff": self.raw_f32_bytes_per_coeff,
            "current_raw_f32_brotli_bytes_per_coeff": round(self.raw_f32_brotli_bytes_per_coeff, 5),
            "byteshuffle_brotli_bytes_per_coeff": round(self.byteshuffle_brotli_bytes_per_coeff, 5),
            "quant_sweep": [m.as_dict() for m in self.quant],
        }


def measure_subband(
    coeffs: np.ndarray,
    *,
    quant_steps: list[float],
    measure_static_range: bool,
    static_range_sample_cap: int,
    key: str,
) -> _SubbandReport:
    """Measure a single aggregated subband's coding headroom on REAL coefficients."""
    coeffs = np.ascontiguousarray(coeffs.astype(np.float32, copy=False).reshape(-1, 1, 1))
    # Reshape to (H,W,C)-like 3D the canonical encoders expect: use (n,1,1).
    n = int(coeffs.size)
    abs_c = np.abs(coeffs)

    raw_bytes = coeffs.tobytes(order="C")
    raw_brotli = _brotli_len(raw_bytes)
    byteshuffle = _brotli_len(_encode_f32_byteshuffle_payload(coeffs))

    rep = _SubbandReport(
        key=key,
        n_coeffs=n,
        coeff_abs_mean=float(abs_c.mean()) if n else 0.0,
        coeff_abs_max=float(abs_c.max()) if n else 0.0,
        raw_f32_bytes_per_coeff=4.0,
        raw_f32_brotli_bytes_per_coeff=raw_brotli / n if n else 0.0,
        byteshuffle_brotli_bytes_per_coeff=byteshuffle / n if n else 0.0,
    )

    for step in quant_steps:
        # Reuse the canonical v2 encoder: it picks the best of dense/RLE/byteplane
        # (and range when symbol-gated). This is the LIVE codec's per-subband choice.
        method, payload = _encode_quantized_detail_payload(coeffs, quantization_step=step)
        payload_brotli = _brotli_len(payload)

        # Real quantize round-trip for distortion + entropy (matches the encoder math).
        q = np.rint(
            np.nan_to_num(coeffs.astype(np.float32) / np.float32(step))
        ).clip(-32768, 32767).astype(np.int64)
        dequant = q.astype(np.float32) * np.float32(step)
        distortion = float(np.mean((dequant - coeffs.astype(np.float32)) ** 2)) if n else 0.0

        _, counts = np.unique(q, return_counts=True)
        h0 = _shannon_bits_per_symbol(counts, n)
        h_struct = _structured_floor_bits_per_coeff(q)
        nz_frac = float(np.count_nonzero(q) / n) if n else 0.0

        static_range_bpc: float | None = None
        if measure_static_range and n > 0:
            # Measure the native Rust-backed constriction range path on a capped sample
            # to quantify the residual gap between the selected live mode and a
            # fractional-bit entropy coder.
            sample = q.astype("<i2", copy=False)
            if sample.size > static_range_sample_cap:
                sample = sample[:static_range_sample_cap]
            try:
                rng_bytes = len(_encode_qi16_constriction_range(sample.reshape(-1, 1, 1)))
                static_range_bpc = rng_bytes / sample.size
            except Exception:  # pragma: no cover - defensive; range coder is fixture-grade
                static_range_bpc = None

        rep.quant.append(
            _QuantMeasurement(
                quant_step=step,
                method_name=_DETAIL_CODEC_NAMES.get(method, f"method_{method}"),
                payload_bytes_per_coeff=len(payload) / n if n else 0.0,
                payload_brotli_bytes_per_coeff=payload_brotli / n if n else 0.0,
                static_range_bytes_per_coeff=static_range_bpc,
                shannon_floor_order0_bpc=h0 / 8.0,
                shannon_floor_structured_bpc=h_struct / 8.0,
                distortion_mse=distortion,
                nonzero_fraction=nz_frac,
            )
        )
    return rep


def build_report(
    *,
    archive_path: Path,
    num_pairs: int,
    quant_steps: list[float],
    measure_static_range: bool,
    static_range_sample_cap: int,
) -> dict[str, Any]:
    archive_bytes = archive_path.read_bytes()
    parsed = parse_archive(archive_bytes)
    pyramids = parse_pair_blobs_from_wavelet_blob(parsed.wavelet_coeffs_blob)
    total_pairs = len(pyramids)
    use_pairs = pyramids[: min(num_pairs, total_pairs)]

    # Aggregate coefficients per (level_idx, orientation) across the sampled pairs
    # and both frames. Deepest-first per the canonical serializer (level 0 = coarsest
    # detail, adjacent to LL).
    buckets: dict[str, list[np.ndarray]] = {}
    for pyr in use_pairs:
        for frame_key in ("frame_0_details", "frame_1_details"):
            details = pyr.get(frame_key, [])
            for level_idx, detail in enumerate(details):
                for orient in ("lh", "hl", "hh"):
                    sub = np.asarray(getattr(detail, orient), dtype=np.float32)
                    if sub.ndim == 4 and sub.shape[0] == 1:
                        sub = sub[0]
                    buckets.setdefault(f"L{level_idx}_{orient}", []).append(sub.reshape(-1))

    subbands: list[_SubbandReport] = []
    for key in sorted(buckets):
        coeffs = np.concatenate(buckets[key])
        subbands.append(
            measure_subband(
                coeffs,
                quant_steps=quant_steps,
                measure_static_range=measure_static_range,
                static_range_sample_cap=static_range_sample_cap,
                key=key,
            )
        )

    # Headline: at each quant step, the aggregate detail-band bytes/coeff under the
    # current default (raw f32 -> brotli) vs the live v2 codec, plus the distortion.
    total_coeffs = sum(sb.n_coeffs for sb in subbands)
    current_total_bytes = sum(
        sb.raw_f32_brotli_bytes_per_coeff * sb.n_coeffs for sb in subbands
    )
    headline_by_step: list[dict[str, Any]] = []
    for i, step in enumerate(quant_steps):
        v2_total = sum(sb.quant[i].payload_brotli_bytes_per_coeff * sb.n_coeffs for sb in subbands)
        floor_total = sum(
            sb.quant[i].shannon_floor_structured_bpc * sb.n_coeffs for sb in subbands
        )
        # Coefficient-magnitude-weighted mean distortion across subbands.
        dist = (
            sum(sb.quant[i].distortion_mse * sb.n_coeffs for sb in subbands) / total_coeffs
            if total_coeffs
            else 0.0
        )
        headroom = current_total_bytes - v2_total
        headline_by_step.append(
            {
                "quant_step": step,
                "current_detail_bytes": round(current_total_bytes, 1),
                "v2_codec_detail_bytes": round(v2_total, 1),
                "structured_shannon_floor_detail_bytes": round(floor_total, 1),
                "headroom_bytes": round(headroom, 1),
                "headroom_fraction": round(headroom / current_total_bytes, 4)
                if current_total_bytes
                else 0.0,
                "mean_distortion_mse": dist,
                "v2_vs_floor_gap_bytes": round(v2_total - floor_total, 1),
            }
        )

    return {
        "schema": "z8_detail_coeff_entropy_headroom_report.v1",
        "tool": "z8_detail_coeff_entropy_headroom_report",
        "purpose": (
            "Measurable per-subband entropy headroom on the REAL Z8HPC1 wavelet "
            "detail blob: (raw f32 -> brotli) vs (quantize + per-subband mode) vs "
            "Shannon floor, at matched distortion. Grounds the v2 pair-blob codec's "
            "per-subband mode + step selection (codex's lane). Read-only; no codec edit."
        ),
        **NON_PROMOTABLE_MARKERS,
        "archive_path": str(archive_path),
        "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "archive_total_bytes": len(archive_bytes),
        "wavelet_blob_bytes": len(parsed.wavelet_coeffs_blob),
        "total_pairs_in_archive": total_pairs,
        "pairs_measured": len(use_pairs),
        "total_detail_coeffs_measured": total_coeffs,
        "quant_steps": quant_steps,
        "static_range_measured": measure_static_range,
        "headline_by_quant_step": headline_by_step,
        "per_subband": [sb.as_dict() for sb in subbands],
        "interpretation": {
            "current_default": "detail blob default schema = raw float32 -> brotli q=11",
            "headroom_definition": "current_detail_bytes - v2_codec_detail_bytes at matched distortion",
            "v2_vs_floor_gap": "how far the live per-subband mode is above the structured Shannon floor",
            "static_range_note": (
                "static_range_bytes_per_coeff measures the native Rust-backed constriction "
                "range coder on a capped sample; compare it against the selected live mode "
                "to decide whether fractional-bit coding beats Brotli-aware RLE/byte-plane modes"
            ),
            "no_signal_loss": "quantization is the ONLY lossy step; all coders round-trip bijectively; LL stays float32",
        },
    }


def _parse_quant_steps(raw: str) -> list[float]:
    steps = [float(x) for x in raw.split(",") if x.strip()]
    if not steps:
        raise ValueError("at least one quant step required")
    for s in steps:
        if not (s > 0.0 and math.isfinite(s)):
            raise ValueError(f"quant step must be finite positive: {s}")
    return steps


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--archive",
        default=(
            "experiments/results/z8_joint_p18_p19_deadzone_rate_attack/"
            "baseline/byte_closed_archive/0.bin"
        ),
        help="Path to a byte-closed Z8HPC1 0.bin archive.",
    )
    ap.add_argument("--num-pairs", type=int, default=8, help="Number of pairs to sample.")
    ap.add_argument("--quant-steps", default="0.5,1.0,2.0,4.0", help="Comma-separated quant steps Δ.")
    ap.add_argument(
        "--measure-static-range",
        action="store_true",
        help=(
            "Also measure the native constriction range coder on a capped sample "
            "(flag name retained for backwards-compatible reports)."
        ),
    )
    ap.add_argument("--static-range-sample-cap", type=int, default=20000)
    ap.add_argument("--out-json", default=None, help="Optional path to write the report JSON.")
    args = ap.parse_args(argv)

    archive_path = Path(args.archive).resolve()
    if not archive_path.is_file():
        ap.error(f"archive not found: {archive_path}")

    report = build_report(
        archive_path=archive_path,
        num_pairs=int(args.num_pairs),
        quant_steps=_parse_quant_steps(args.quant_steps),
        measure_static_range=bool(args.measure_static_range),
        static_range_sample_cap=int(args.static_range_sample_cap),
    )

    print(f"[z8-headroom] archive={archive_path.name} [macOS-CPU advisory] NON-PROMOTABLE", flush=True)
    print(
        f"[z8-headroom] {report['pairs_measured']}/{report['total_pairs_in_archive']} pairs, "
        f"{report['total_detail_coeffs_measured']:,} detail coeffs, "
        f"wavelet_blob={report['wavelet_blob_bytes']:,}B",
        flush=True,
    )
    print("[z8-headroom] headline (detail-band bytes for the sampled pairs):", flush=True)
    for h in report["headline_by_quant_step"]:
        print(
            f"    Δ={h['quant_step']:<5} current={h['current_detail_bytes']:>12,.0f}B "
            f"v2={h['v2_codec_detail_bytes']:>12,.0f}B "
            f"floor={h['structured_shannon_floor_detail_bytes']:>12,.0f}B "
            f"headroom={h['headroom_fraction']*100:>6.1f}% "
            f"dist={h['mean_distortion_mse']:.3e}",
            flush=True,
        )
    print("[z8-headroom] per-subband current vs best-v2 bytes/coeff (Δ-swept):", flush=True)
    for sb in report["per_subband"]:
        best = min(sb["quant_sweep"], key=lambda m: m["live_codec_brotli_bytes_per_coeff"])
        print(
            f"    {sb['subband']:<10} n={sb['n_coeffs']:>9,} "
            f"|c|~{sb['coeff_abs_mean']:.4f} "
            f"current={sb['current_raw_f32_brotli_bytes_per_coeff']:.3f} bpc "
            f"-> v2[{best['live_codec_method']}]={best['live_codec_brotli_bytes_per_coeff']:.3f} bpc "
            f"(floor={best['shannon_floor_structured_bytes_per_coeff']:.3f}, "
            f"Δ={best['quant_step']}, nz={best['nonzero_fraction']*100:.1f}%)",
            flush=True,
        )

    if args.out_json:
        out = Path(args.out_json).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True))
        print(f"[z8-headroom] wrote {out}", flush=True)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
