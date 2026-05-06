#!/usr/bin/env python3
"""Forensic stack predictor for PR106-anchor lanes.

Given:
  - apogee_intN bit-width N (∈ {4, 5, 6, 7, 8}; 7 is Pareto-dominated, skip)
  - subset of sidechannels: {latent, yshift, lrl1}

Outputs:
  - expected total archive bytes
  - per-layer score contribution (rate Δ + distortion Δ)
  - predicted score band tagged as prediction-only, noncanonical evidence
  - explicit dispatch blockers

Sources of truth:
  - apogee_intN bands: tools/apogee_intN_pareto.py historical forensic bands
  - sidechannel deltas: lane registry + memory paradigm thread predictions
  - PR106 baseline: 0.20945673, 186,239 bytes

This tool never emits exact-eval readiness. Its outputs are prediction-only
forensics until a scorer-basin parity gate, contest-faithful distortion model,
or exact CUDA evidence validates the exact candidate archive bytes.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

PR106_BASELINE_SCORE = 0.20945673
PR106_BASELINE_BYTES = 186239
RATE_DENOM = 37545489.0  # contest formula: rate = 25 * bytes / RATE_DENOM
DISPATCH_BLOCKERS = [
    "prediction_only_stack_estimate",
    "missing_exact_candidate_archive",
    "missing_contest_faithful_distortion_model",
    "missing_scorer_basin_parity_gate",
    "sidechannel_deltas_not_exact_cuda_stack_evidence",
]


@dataclass(frozen=True)
class ApogeeBand:
    bits: int
    archive_bytes: int
    rate_delta: float  # vs PR106
    predicted_band_low: float
    predicted_band_high: float
    risk: str
    pareto_dominated: bool = False


@dataclass(frozen=True)
class SidechannelBand:
    name: str
    overhead_bytes: int  # added on top of inner archive
    predicted_distortion_delta: float  # negative = improvement
    band_low_offset: float  # range around point estimate
    band_high_offset: float
    rate_only_delta: float  # = 25 * overhead_bytes / RATE_DENOM
    requires_inner_layers: tuple[str, ...] = ()  # gate dependencies


APOGEE_BANDS: dict[int, ApogeeBand] = {
    4: ApogeeBand(bits=4, archive_bytes=109996, rate_delta=-0.0508,
                  predicted_band_low=0.155, predicted_band_high=0.180,
                  risk="HIGH"),
    5: ApogeeBand(bits=5, archive_bytes=154555, rate_delta=-0.0211,
                  predicted_band_low=0.180, predicted_band_high=0.196,
                  risk="MEDIUM"),
    6: ApogeeBand(bits=6, archive_bytes=170450, rate_delta=-0.0105,
                  predicted_band_low=0.190, predicted_band_high=0.204,
                  risk="LOW"),
    7: ApogeeBand(bits=7, archive_bytes=205158, rate_delta=+0.0126,
                  predicted_band_low=0.198, predicted_band_high=0.208,
                  risk="VERY LOW", pareto_dominated=True),
    8: ApogeeBand(bits=8, archive_bytes=187731, rate_delta=+0.0010,
                  predicted_band_low=0.196, predicted_band_high=0.207,
                  risk="ALMOST LOSSLESS"),
}


def _rate_only_delta(overhead_bytes: int) -> float:
    return 25.0 * overhead_bytes / RATE_DENOM


SIDECHANNELS: dict[str, SidechannelBand] = {
    "latent": SidechannelBand(
        name="latent_sidecar",
        overhead_bytes=23,  # CPU-smoke empirical: 186131 → 186262
        predicted_distortion_delta=-0.00218,
        band_low_offset=-0.001,
        band_high_offset=+0.001,
        rate_only_delta=_rate_only_delta(23),
    ),
    "yshift": SidechannelBand(
        name="yshift_sidechannel",
        overhead_bytes=44,  # CPU-smoke empirical: 186131 → 186283 = +44 (mode=zero, brotli-43B + 9B-wrapper)
        predicted_distortion_delta=-0.001,
        band_low_offset=-0.0005,
        band_high_offset=+0.0005,
        rate_only_delta=_rate_only_delta(44),
        requires_inner_layers=("latent",),  # gated per paradigm thread
    ),
    "lrl1": SidechannelBand(
        name="lrl1_sidechannel",
        overhead_bytes=50,  # CPU-smoke empirical: 186239 → 186289 (zero-correction)
        predicted_distortion_delta=-0.0015,
        band_low_offset=-0.0005,
        band_high_offset=+0.0005,
        rate_only_delta=_rate_only_delta(50),
        requires_inner_layers=("latent", "yshift"),
    ),
}


def predict(bits: int, sidechannels: list[str]) -> dict:
    if bits not in APOGEE_BANDS:
        raise ValueError(f"unsupported bits={bits}; choose from {sorted(APOGEE_BANDS)}")
    base = APOGEE_BANDS[bits]
    if base.pareto_dominated:
        warning = f"int{bits} is PARETO-DOMINATED by int8 (more bytes, no distortion gain)"
    else:
        warning = None

    # validate gate dependencies
    chosen = list(sidechannels)
    for sc_name in sidechannels:
        if sc_name not in SIDECHANNELS:
            raise ValueError(f"unknown sidechannel '{sc_name}'; choose from {sorted(SIDECHANNELS)}")
        sc = SIDECHANNELS[sc_name]
        for required in sc.requires_inner_layers:
            if required not in chosen:
                raise ValueError(
                    f"sidechannel '{sc_name}' requires inner layer '{required}' "
                    f"first; current chain: {chosen}"
                )

    total_overhead = sum(SIDECHANNELS[n].overhead_bytes for n in sidechannels)
    total_bytes = base.archive_bytes + total_overhead
    total_rate_delta = base.rate_delta + sum(SIDECHANNELS[n].rate_only_delta for n in sidechannels)
    total_distortion_delta = sum(SIDECHANNELS[n].predicted_distortion_delta for n in sidechannels)

    # midpoint of base band as point estimate, then layer in sidechannel deltas.
    # Sidechannel rate overhead must be folded in (was missing pre-2026-05-05 audit).
    sc_rate_overhead = sum(SIDECHANNELS[n].rate_only_delta for n in sidechannels)
    base_midpoint = 0.5 * (base.predicted_band_low + base.predicted_band_high)
    point_estimate = base_midpoint + total_distortion_delta + sc_rate_overhead
    band_low = (base.predicted_band_low + total_distortion_delta + sc_rate_overhead
                + sum(SIDECHANNELS[n].band_low_offset for n in sidechannels))
    band_high = (base.predicted_band_high + total_distortion_delta + sc_rate_overhead
                 + sum(SIDECHANNELS[n].band_high_offset for n in sidechannels))

    return {
        "config": {
            "apogee_bits": bits,
            "sidechannels": sidechannels,
        },
        "warning": warning,
        "evidence_semantics": "prediction_only_forensic",
        "ready_for_exact_eval_dispatch": False,
        "dispatch_blockers": list(DISPATCH_BLOCKERS),
        "predicted_archive_bytes": total_bytes,
        "predicted_total_rate_delta": total_rate_delta,
        "predicted_total_distortion_delta": total_distortion_delta,
        "predicted_score_point_estimate": point_estimate,
        "predicted_score_band": [band_low, band_high],
        "predicted_beats_pr106": point_estimate < PR106_BASELINE_SCORE,
        "beats_pr106": False,
        "beats_pr106_claim_valid": False,
        "improvement_over_pr106": PR106_BASELINE_SCORE - point_estimate,
        "per_layer_breakdown": {
            f"apogee_int{bits}": {
                "bytes": base.archive_bytes,
                "rate_delta": base.rate_delta,
                "band": [base.predicted_band_low, base.predicted_band_high],
                "risk": base.risk,
            },
            **{
                f"sidechannel_{n}": {
                    "overhead_bytes": SIDECHANNELS[n].overhead_bytes,
                    "rate_delta": SIDECHANNELS[n].rate_only_delta,
                    "predicted_distortion_delta": SIDECHANNELS[n].predicted_distortion_delta,
                }
                for n in sidechannels
            },
        },
    }


def _format_human(result: dict) -> str:
    cfg = result["config"]
    lines = [
        "═══════════════════════════════════════════════════════════════════",
        "PR106-stack predictor",
        "═══════════════════════════════════════════════════════════════════",
        f"PR106 baseline:                  {PR106_BASELINE_SCORE:.6f}    {PR106_BASELINE_BYTES:>7,} bytes",
        f"Config: apogee_int{cfg['apogee_bits']} + sidechannels {cfg['sidechannels']}",
        "",
    ]
    if result["warning"]:
        lines.append(f"⚠️  WARNING: {result['warning']}")
        lines.append("")
    lines.append("Per-layer breakdown:")
    for name, info in result["per_layer_breakdown"].items():
        lines.append(f"  {name}")
        for k, v in info.items():
            lines.append(f"    {k}: {v}")
    lines.extend([
        "",
        f"Predicted total archive: {result['predicted_archive_bytes']:>7,} bytes",
        f"Predicted total rate Δ:  {result['predicted_total_rate_delta']:+.6f}",
        f"Predicted total distortion Δ: {result['predicted_total_distortion_delta']:+.6f}",
        "",
        f"Predicted score (point):   {result['predicted_score_point_estimate']:.6f}",
        f"Predicted score (band):    [{result['predicted_score_band'][0]:.6f}, {result['predicted_score_band'][1]:.6f}]",
        f"Predicted below PR106 ({PR106_BASELINE_SCORE:.6f})? {'YES' if result['predicted_beats_pr106'] else 'NO'}",
        f"Predicted delta vs PR106:  {result['improvement_over_pr106']:+.6f}",
        "",
        "ready_for_exact_eval_dispatch=false",
        "Evidence semantics: prediction_only_forensic",
        "Dispatch blockers: " + ", ".join(result["dispatch_blockers"]),
    ])
    if cfg["sidechannels"]:
        lines.extend([
            "",
            "Operator action: do not dispatch from this prediction table.",
            "Build an exact candidate archive and pass an explicit distortion/parity gate first.",
        ])
    else:
        lines.extend([
            "",
            f"apogee_int{cfg['apogee_bits']} alone remains blocked for score dispatch.",
            "Use tools/dispatch_dryrun_apogee_intN.py --allow-forensic-byte-only for local parser checks only.",
        ])
    lines.append("═══════════════════════════════════════════════════════════════════")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bits", type=int, required=True,
                        choices=[4, 5, 6, 7, 8],
                        help="apogee_intN bit-width")
    parser.add_argument("--sidechannels", nargs="*", default=[],
                        choices=["latent", "yshift", "lrl1"],
                        help="sidechannels to stack on top (in order: latent → yshift → lrl1)")
    parser.add_argument("--json", action="store_true",
                        help="output JSON instead of human-readable")
    parser.add_argument("--all", action="store_true",
                        help="enumerate all bits + sidechannel combinations and print Pareto-optimal")
    args = parser.parse_args(argv)

    if args.all:
        # enumerate all valid combinations
        results = []
        for bits in [4, 5, 6, 8]:  # skip Pareto-dominated 7
            for sc_combo in [[], ["latent"], ["latent", "yshift"], ["latent", "yshift", "lrl1"]]:
                try:
                    r = predict(bits, sc_combo)
                    results.append(r)
                except ValueError:
                    continue
        # sort by point-estimate ascending (best first)
        results.sort(key=lambda r: r["predicted_score_point_estimate"])
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print("All combinations sorted by forensic predicted score (not dispatch-ready):\n")
            for i, r in enumerate(results[:12], 1):
                cfg = r["config"]
                sc = "+".join(cfg["sidechannels"]) if cfg["sidechannels"] else "alone"
                print(f"  {i:2d}. int{cfg['apogee_bits']} {sc:<28s} → {r['predicted_score_point_estimate']:.6f} "
                      f"(predicted Δ {r['improvement_over_pr106']:+.6f}, {r['predicted_archive_bytes']:>7,} bytes, ready=false)")
        return 0

    result = predict(args.bits, args.sidechannels)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(_format_human(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
