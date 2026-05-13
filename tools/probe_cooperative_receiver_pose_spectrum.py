#!/usr/bin/env python3
"""Probe the L2 SAR/coherent-integration pose-spectrum hypothesis.

This is a zero-spend planning probe. It reads precomputed PoseNet targets and
asks whether the six pose dimensions concentrate in a small temporal-frequency
set. If they do, a future byte-closed pose sidecar can store sparse rFFT
coefficients instead of independent per-pair pose records.

No score authority is created here: the output is forced through the canonical
proxy false-authority contract.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from tac.optimization.proxy_candidate_contract import apply_proxy_evidence_boundary
from tac.scorer_targets import load_posenet_targets

POSE_SPECTRUM_SCHEMA = "tac_cooperative_receiver_pose_spectrum_probe_v1"


def _threshold_k(sorted_energy: torch.Tensor, threshold: float) -> int:
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")
    if sorted_energy.numel() == 0:
        return 0
    cumsum = torch.cumsum(sorted_energy, dim=0)
    hits = torch.nonzero(cumsum >= threshold, as_tuple=False)
    return int(hits[0].item() + 1) if hits.numel() else int(sorted_energy.numel())


def analyze_pose_targets_tensor(
    targets: torch.Tensor,
    *,
    low_frequency_fraction: float = 0.10,
    thresholds: Sequence[float] = (0.90, 0.95, 0.99),
) -> dict[str, Any]:
    """Return a proxy-safe temporal spectrum report for ``(pairs, 6)`` targets."""

    if targets.ndim != 2 or targets.shape[1] != 6:
        raise ValueError(f"targets must have shape (num_pairs, 6), got {tuple(targets.shape)}")
    if targets.shape[0] < 2:
        raise ValueError("at least two pose pairs are required for a spectrum probe")
    if not torch.isfinite(targets).all():
        raise ValueError("targets contain non-finite values")
    if not 0.0 < low_frequency_fraction <= 1.0:
        raise ValueError("low_frequency_fraction must be in (0, 1]")

    y = targets.detach().to(device="cpu", dtype=torch.float64)
    centered = y - y.mean(dim=0, keepdim=True)
    spectrum = torch.fft.rfft(centered, dim=0)
    power_per_bin = (spectrum.real.square() + spectrum.imag.square()).sum(dim=1)
    total_power = float(power_per_bin.sum().item())
    num_bins = int(power_per_bin.numel())
    low_bins = max(1, math.ceil(num_bins * low_frequency_fraction))

    normalized = torch.zeros_like(power_per_bin) if total_power <= 0.0 else power_per_bin / total_power
    sorted_energy, sorted_indices = torch.sort(normalized, descending=True)

    recommended_top_k = {
        f"{threshold:.2f}": _threshold_k(sorted_energy, threshold)
        for threshold in thresholds
    }
    k95 = recommended_top_k.get("0.95", _threshold_k(sorted_energy, 0.95))
    # One frequency carries six complex fp16 coefficients plus a uint16 index.
    sparse_fft_bytes_95 = int(k95 * (2 + 6 * 2 * 2) + 16)

    top_decile_bins = max(1, math.ceil(num_bins * 0.10))
    low_frequency_energy_fraction = float(normalized[:low_bins].sum().item())
    top_decile_energy_fraction = float(sorted_energy[:top_decile_bins].sum().item())
    strongest_bins = [
        {
            "frequency_bin": int(idx.item()),
            "energy_fraction": float(sorted_energy[pos].item()),
        }
        for pos, idx in enumerate(sorted_indices[: min(12, num_bins)])
    ]

    row = {
        "schema": POSE_SPECTRUM_SCHEMA,
        "num_pairs": int(targets.shape[0]),
        "pose_dims": 6,
        "rfft_bins": num_bins,
        "low_frequency_fraction": low_frequency_fraction,
        "low_frequency_bins": low_bins,
        "total_centered_power": total_power,
        "low_frequency_energy_fraction": low_frequency_energy_fraction,
        "top_decile_energy_fraction": top_decile_energy_fraction,
        "recommended_top_k_coefficients": recommended_top_k,
        "estimated_sparse_fft_bytes_at_95pct": sparse_fft_bytes_95,
        "strongest_frequency_bins": strongest_bins,
        "hypothesis_supported": (
            low_frequency_energy_fraction >= 0.60 or top_decile_energy_fraction >= 0.80
        ),
        "campaign_id": "l2_sar_coherent_pose_spectrum",
        "evidence_semantics": "zero_spend_pose_target_spectrum_proxy",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_blockers": [
            "pose_codec_section_missing",
            "byte_closed_archive_missing",
            "paired_exact_eval_missing",
        ],
    }
    return apply_proxy_evidence_boundary(row)


def load_targets_tensor(path: str | Path) -> torch.Tensor:
    loaded = load_posenet_targets(path)
    if loaded is None:
        raise FileNotFoundError(f"PoseNet targets not found or invalid: {path}")
    targets = loaded.get("targets")
    if not isinstance(targets, torch.Tensor):
        raise ValueError("loaded posenet targets missing tensor field 'targets'")
    return targets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pose-targets",
        type=Path,
        default=Path("experiments/posenet_targets.bin"),
        help="Path to precomputed posenet_targets.bin.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/cooperative_receiver/pose_spectrum_l2.json"),
        help="JSON output path.",
    )
    parser.add_argument(
        "--low-frequency-fraction",
        type=float,
        default=0.10,
        help="Fraction of rFFT bins counted as low-frequency.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    targets = load_targets_tensor(args.pose_targets)
    report = analyze_pose_targets_tensor(
        targets,
        low_frequency_fraction=args.low_frequency_fraction,
    )
    report["pose_targets_path"] = str(args.pose_targets)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "wrote cooperative_receiver_pose_spectrum "
        f"pairs={report['num_pairs']} low_freq={report['low_frequency_energy_fraction']:.6f} "
        f"top_decile={report['top_decile_energy_fraction']:.6f} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "POSE_SPECTRUM_SCHEMA",
    "analyze_pose_targets_tensor",
    "build_parser",
    "load_targets_tensor",
    "main",
]
