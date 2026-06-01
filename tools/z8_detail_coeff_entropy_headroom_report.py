#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""CLI wrapper for Z8 wavelet detail-coefficient entropy-headroom reports.

The reusable implementation lives in
``tac.substrates.z8_hierarchical_predictive_coding.detail_entropy_headroom`` so
queue runners and allocators can consume the same report without importing a
``tools/`` script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tac.substrates.z8_hierarchical_predictive_coding.detail_entropy_headroom import (
    _parse_quant_steps,
    _parse_workers,
    build_report,
)


def build_parser() -> argparse.ArgumentParser:
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
    ap.add_argument(
        "--quant-steps",
        default="0.5,1.0,2.0,4.0",
        help="Comma-separated quant steps Delta.",
    )
    ap.add_argument(
        "--measure-static-range",
        action="store_true",
        help=(
            "Also measure the native constriction range coder on a capped sample "
            "(flag name retained for backwards-compatible reports)."
        ),
    )
    ap.add_argument("--static-range-sample-cap", type=int, default=20000)
    ap.add_argument(
        "--workers",
        default="1",
        help="Subband measurement workers. Use 'auto' to saturate independent subbands.",
    )
    ap.add_argument("--out-json", default=None, help="Optional path to write the report JSON.")
    return ap


def _print_summary(report: dict[str, object], archive_path: Path) -> None:
    print(f"[z8-headroom] archive={archive_path.name} [macOS-CPU advisory] NON-PROMOTABLE", flush=True)
    print(
        f"[z8-headroom] {report['pairs_measured']}/{report['total_pairs_in_archive']} pairs, "
        f"{report['total_detail_coeffs_measured']:,} detail coeffs, "
        f"wavelet_blob={report['wavelet_blob_bytes']:,}B",
        flush=True,
    )
    print(f"[z8-headroom] workers={report['workers']}", flush=True)
    print("[z8-headroom] headline (detail-band bytes for the sampled pairs):", flush=True)
    for h in report["headline_by_quant_step"]:
        print(
            f"    Delta={h['quant_step']:<5} current={h['current_detail_bytes']:>12,.0f}B "
            f"v2={h['v2_codec_detail_bytes']:>12,.0f}B "
            f"floor={h['structured_shannon_floor_detail_bytes']:>12,.0f}B "
            f"headroom={h['headroom_fraction']*100:>6.1f}% "
            f"dist={h['mean_distortion_mse']:.3e}",
            flush=True,
        )
    print("[z8-headroom] per-subband current vs best-v2 bytes/coeff (Delta-swept):", flush=True)
    for sb in report["per_subband"]:
        best = min(sb["quant_sweep"], key=lambda m: m["live_codec_brotli_bytes_per_coeff"])
        print(
            f"    {sb['subband']:<10} n={sb['n_coeffs']:>9,} "
            f"|c|~{sb['coeff_abs_mean']:.4f} "
            f"current={sb['current_raw_f32_brotli_bytes_per_coeff']:.3f} bpc "
            f"-> v2[{best['live_codec_method']}]={best['live_codec_brotli_bytes_per_coeff']:.3f} bpc "
            f"(floor={best['shannon_floor_structured_bytes_per_coeff']:.3f}, "
            f"Delta={best['quant_step']}, nz={best['nonzero_fraction']*100:.1f}%)",
            flush=True,
        )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    archive_path = Path(args.archive).resolve()
    if not archive_path.is_file():
        parser.error(f"archive not found: {archive_path}")

    report = build_report(
        archive_path=archive_path,
        num_pairs=int(args.num_pairs),
        quant_steps=_parse_quant_steps(args.quant_steps),
        measure_static_range=bool(args.measure_static_range),
        static_range_sample_cap=int(args.static_range_sample_cap),
        workers=_parse_workers(args.workers),
    )
    _print_summary(report, archive_path)

    if args.out_json:
        out = Path(args.out_json).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True))
        print(f"[z8-headroom] wrote {out}", flush=True)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
