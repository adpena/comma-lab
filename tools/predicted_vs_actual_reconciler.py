#!/usr/bin/env python3
"""Predicted-vs-actual reconciliation for apogee_intN dispatches.

Joins:
  predicted: experiments/results/apogee_int*_repack_*/repack_metadata.json
             (predicted_score_band emitted by the producer at compress time)
  actual:    experiments/results/lane_apogee_int*_*/eval/contest_auth_eval.json
             (final_score from CUDA dispatch — when the operator has launched
             the lane via scripts/remote_lane_apogee_intN.sh)

Reports per-bits:
  - predicted band [low, high] (from manifest)
  - actual final_score (from contest_auth_eval.json, if present)
  - in-band? (✓ if low ≤ actual ≤ high; ✗ otherwise)
  - rate component beats PR106 baseline 0.20946? (from rate_score_delta)
  - device + samples (advisory tag check)

Pairs with:
  tools/apogee_intN_pareto.py  — pre-dispatch decision matrix
  tools/score_dashboard.py      — sorted view of every contest_auth_eval

Usage:
  .venv/bin/python tools/predicted_vs_actual_reconciler.py
  .venv/bin/python tools/predicted_vs_actual_reconciler.py --json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PR106_BASELINE_SCORE = 0.20945673


@dataclass
class ReconciledRow:
    bits: int
    predicted_low: float
    predicted_high: float
    archive_size_bytes: int
    distortion_risk: str
    rate_score_delta: float
    actual_score: float | None
    actual_path: str | None
    in_band: bool | None
    beats_pr106: bool | None
    device: str | None
    samples: int | None


def _parse_band(band_str: str) -> tuple[float, float]:
    inner = band_str.strip().lstrip("[").rstrip("]")
    lo_str, hi_str = inner.split(",")
    return float(lo_str.strip()), float(hi_str.strip())


def _find_actual_for_bits(repo_root: Path, bits: int) -> tuple[float | None, str | None, str | None, int | None]:
    """Find the latest contest_auth_eval.json for a given apogee_intN bits config.

    Looks under experiments/results/ for any directory matching
    `lane_apogee_int{bits}_*` containing an `eval/contest_auth_eval.json`
    OR a top-level `contest_auth_eval.json`. Returns the score from the
    most-recently-modified JSON found.
    """
    candidates: list[Path] = []
    base = repo_root / "experiments" / "results"
    if not base.is_dir():
        return None, None, None, None
    for lane_dir in base.glob(f"lane_apogee_int{bits}*"):
        for sub in lane_dir.rglob("contest_auth_eval*.json"):
            candidates.append(sub)
    if not candidates:
        return None, None, None, None
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    try:
        data = json.loads(latest.read_text())
    except (json.JSONDecodeError, OSError):
        return None, str(latest.relative_to(repo_root)), None, None
    score = data.get("final_score") or data.get("score") or data.get("total_score")
    if score is not None:
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = None
    device = data.get("device", "?")
    samples = data.get("samples") or data.get("n_samples")
    return score, str(latest.relative_to(repo_root)), str(device), samples


def reconcile(repo_root: Path) -> list[ReconciledRow]:
    rows: list[ReconciledRow] = []
    base = repo_root / "experiments" / "results"
    if not base.is_dir():
        return rows
    for manifest in base.glob("apogee_int*_repack_*/repack_metadata.json"):
        try:
            data = json.loads(manifest.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        try:
            lo, hi = _parse_band(data["predicted_score_band"])
        except (KeyError, ValueError):
            continue
        bits = int(data["bits"])
        actual, actual_path, device, samples = _find_actual_for_bits(repo_root, bits)
        in_band = (lo <= actual <= hi) if actual is not None else None
        beats_pr106 = (actual < PR106_BASELINE_SCORE) if actual is not None else None
        rows.append(ReconciledRow(
            bits=bits,
            predicted_low=lo,
            predicted_high=hi,
            archive_size_bytes=int(data["archive_size_bytes"]),
            distortion_risk=str(data["distortion_risk"]),
            rate_score_delta=float(data["rate_component_score_delta_vs_pr106"]),
            actual_score=actual,
            actual_path=actual_path,
            in_band=in_band,
            beats_pr106=beats_pr106,
            device=device,
            samples=samples,
        ))
    rows.sort(key=lambda r: r.bits)
    return rows


def _format_table(rows: list[ReconciledRow]) -> str:
    if not rows:
        return "(no apogee_intN repack manifests found — run experiments/repack_pr106_with_intN_block_fp.py first)"
    out: list[str] = []
    out.append(f"PR106 baseline: 0.{int(PR106_BASELINE_SCORE * 1e8):08d}, 186,239 bytes")
    out.append("")
    header = f"{'bits':>4}  {'predicted band':<18}  {'actual':>8}  {'in band?':<8}  {'beats PR106?':<13}  {'risk':<16}  {'device':<6}  evidence"
    out.append(header)
    out.append("-" * len(header))
    n_landed = 0
    n_in_band = 0
    n_beats = 0
    for r in rows:
        actual_str = f"{r.actual_score:.5f}" if r.actual_score is not None else "(pending)"
        if r.in_band is True:
            inband_str = "✓ in"
            n_in_band += 1
        elif r.in_band is False:
            inband_str = "✗ OUT"
        else:
            inband_str = "—"
        if r.beats_pr106 is True:
            beats_str = "✓ YES (frontier!)"
            n_beats += 1
        elif r.beats_pr106 is False:
            beats_str = "✗ no"
        else:
            beats_str = "—"
        if r.actual_score is not None:
            n_landed += 1
        # Mark non-CUDA device with asterisk per CLAUDE.md
        device_str = r.device if r.device else "—"
        if device_str not in ("cuda", "—", "?", None) and not device_str.endswith("*"):
            device_str = device_str + "*"
        evidence = r.actual_path or "(no contest_auth_eval.json yet)"
        # Truncate evidence path
        if len(evidence) > 60:
            evidence = "…" + evidence[-59:]
        out.append(
            f"{r.bits:>4}  [{r.predicted_low:.3f}, {r.predicted_high:.3f}]   "
            f"{actual_str:>8}  {inband_str:<8}  {beats_str:<13}  "
            f"{r.distortion_risk:<16}  {(device_str or '—'):<6}  {evidence}"
        )
    out.append("")
    out.append(f"Summary: {n_landed}/{len(rows)} bits configs have actual scores; "
              f"{n_in_band}/{n_landed} in predicted band; "
              f"{n_beats}/{n_landed} beat PR106 baseline ({PR106_BASELINE_SCORE:.5f})")
    if n_landed == 0:
        out.append("")
        out.append("No actual scores yet. Operator action: run a Pareto-frontier dispatch from")
        out.append("`tools/apogee_intN_pareto.py` output, then re-run this reconciler.")
    out.append("(* = non-CUDA device; treat as advisory only per CLAUDE.md MPS-auth-eval-is-NOISE)")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true",
                        help="Machine-readable JSON output instead of human table.")
    args = parser.parse_args(argv)
    rows = reconcile(REPO_ROOT)
    if args.json:
        out = {
            "pr106_baseline_score": PR106_BASELINE_SCORE,
            "n_configs": len(rows),
            "n_landed": sum(1 for r in rows if r.actual_score is not None),
            "n_in_band": sum(1 for r in rows if r.in_band is True),
            "n_beats_pr106": sum(1 for r in rows if r.beats_pr106 is True),
            "rows": [asdict(r) for r in rows],
        }
        print(json.dumps(out, indent=2))
    else:
        print(_format_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
