#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Probe the compact SNeRV step-map packet coder on deterministic maps."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.analysis.snerv_step_map_coder import (  # noqa: E402
    encode_step_maps,
    encode_step_maps_adaptive,
    encode_step_maps_waterfill,
)


def _default_out() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return REPO_ROOT / f".omx/research/snerv_step_map_coder_probe_{stamp}.json"


def _smooth_maps(count: int, h: int, w: int) -> list[np.ndarray]:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = 0.5 + 0.2 * np.sin(xx / 12.0) + 0.1 * np.cos(yy / 9.0)
    return [np.exp2(base + i * 0.02).astype(np.float32) for i in range(count)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-count", type=int, default=24)
    parser.add_argument("--h", type=int, default=48)
    parser.add_argument("--w", type=int, default=64)
    parser.add_argument(
        "--bins",
        default="128,64,16,4",
        help="Comma-separated log2 quantizer bin counts to probe.",
    )
    parser.add_argument(
        "--adaptive-portfolios",
        default="128,64,4;64,16,4;16,4",
        help="Semicolon-separated adaptive bin portfolios.",
    )
    parser.add_argument(
        "--constant-importance-quantile",
        type=float,
        default=None,
        help="Optional low-importance quantile to encode as constant-fill maps.",
    )
    parser.add_argument(
        "--waterfill-target-bits",
        type=float,
        default=6.0,
        help="Target average bits/coefficient for the waterfill precision ladder.",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    maps = _smooth_maps(args.map_count, args.h, args.w)
    bin_values = _parse_bins(args.bins)
    packets = [encode_step_maps(maps, bins=bins) for bins in bin_values]
    importance = np.linspace(0.0, 1.0, len(maps), dtype=np.float64)
    adaptive_packets = [
        encode_step_maps_adaptive(
            maps,
            map_importance=importance,
            bin_choices=tuple(portfolio),
            constant_importance_quantile=args.constant_importance_quantile,
        )
        for portfolio in _parse_portfolios(args.adaptive_portfolios)
    ]
    waterfill_packet = encode_step_maps_waterfill(
        maps,
        map_importance=importance,
        target_bits_per_coeff=args.waterfill_target_bits,
    )
    payload = {
        "schema": "snerv_step_map_coder_probe.v3",
        "axis_tag": "[codec-unit:false-authority]",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "map_count": args.map_count,
        "shape": [args.h, args.w],
        "packets": [packet.as_jsonable() for packet in packets],
        "adaptive_bundle_packets": [packet.as_jsonable() for packet in adaptive_packets],
        "waterfill_packet": waterfill_packet.as_jsonable(),
        "waterfill_target_bits_per_coeff": args.waterfill_target_bits,
        "constant_importance_quantile": args.constant_importance_quantile,
        "bundling_note": (
            "adaptive bundle groups maps by assigned precision, sharing one "
            "group subpacket per bins value instead of one packet per map; "
            "optional constant groups are header-only run-length fills; "
            "waterfill extends the same grammar with fp16-protected groups"
        ),
        "next_action": (
            "use score saliency to protect sensitive maps with fp16/int8 while "
            "collapsing insensitive maps to int4/int2/constant before decoder-fit work"
        ),
    }
    out_path = Path(args.out) if args.out else _default_out()
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print("[SNeRV step-map coder probe] false-authority")
    for packet in packets:
        print(
            "  "
            f"bins={packet.bins:3d} bits/code={packet.bits_per_code} "
            f"packed={packet.packed_code_bytes} B packet={packet.total_bytes} B "
            f"fp32-lzma={packet.fp32_lzma_baseline_bytes} B "
            f"max_rel_err={packet.max_relative_error:.6f}"
        )
    for packet in adaptive_packets:
        print(
            "  "
            f"adaptive_bundle packet={packet.total_bytes} B "
            f"groups={[(g['bins'], len(g['map_indices'])) for g in packet.groups]} "
            f"max_rel_err={packet.max_relative_error:.6f}"
        )
    print(
        "  "
        f"waterfill packet={waterfill_packet.total_bytes} B "
        f"groups={[(g.get('precision_label'), len(g['map_indices'])) for g in waterfill_packet.groups]} "
        f"max_rel_err={waterfill_packet.max_relative_error:.6f}"
    )
    print(f"  wrote {out_path}")
    return 0


def _parse_bins(raw: str) -> list[int]:
    out = []
    for chunk in raw.split(","):
        value = int(chunk.strip())
        if value < 2 or value > 256:
            raise ValueError("--bins values must be in [2, 256]")
        out.append(value)
    if not out:
        raise ValueError("at least one --bins value is required")
    return out


def _parse_portfolios(raw: str) -> list[list[int]]:
    portfolios = []
    for spec in raw.split(";"):
        spec = spec.strip()
        if not spec:
            continue
        portfolios.append(_parse_bins(spec))
    if not portfolios:
        raise ValueError("at least one --adaptive-portfolios entry is required")
    return portfolios


if __name__ == "__main__":
    raise SystemExit(main())
