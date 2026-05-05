#!/usr/bin/env python3
"""apogee_intN Pareto frontier analysis: which bit-width to dispatch?

Reads every `experiments/results/apogee_int*_repack_*/repack_metadata.json`,
sorts by archive size, marks Pareto-dominated configs (smaller AND lower-risk
exists), and prints a one-glance dispatch decision matrix with operator-ready
one-liners per non-dominated config.

A config is Pareto-dominated when there exists another config with BOTH:
  - smaller archive_size_bytes (better rate)
  - same-or-lower distortion_risk class (better-or-equal distortion)

Risk ordering (low → high):
  ALMOST LOSSLESS < VERY LOW < LOW < MEDIUM < HIGH

Usage:
    .venv/bin/python tools/apogee_intN_pareto.py
    .venv/bin/python tools/apogee_intN_pareto.py --json   # machine-readable
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PR106_BASELINE_SCORE = 0.20945673
RISK_ORDER = ["ALMOST LOSSLESS", "VERY LOW", "LOW", "MEDIUM", "HIGH"]


@dataclass
class ApogeeRow:
    bits: int
    archive_path: str
    archive_size_bytes: int
    delta_bytes: int
    rate_score_delta: float
    distortion_risk: str
    predicted_band: str
    rel_err_pct: float
    pareto_dominated_by: int | None
    predicted_low: float
    predicted_high: float

    def risk_rank(self) -> int:
        return RISK_ORDER.index(self.distortion_risk) if self.distortion_risk in RISK_ORDER else len(RISK_ORDER)


def _parse_band(band_str: str) -> tuple[float, float]:
    inner = band_str.strip().lstrip("[").rstrip("]")
    lo_str, hi_str = inner.split(",")
    return float(lo_str.strip()), float(hi_str.strip())


def scan(repo_root: Path) -> list[ApogeeRow]:
    rows: list[ApogeeRow] = []
    for manifest in (repo_root / "experiments" / "results").glob("apogee_int*_repack_*/repack_metadata.json"):
        try:
            data = json.loads(manifest.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        try:
            lo, hi = _parse_band(data["predicted_score_band"])
        except (KeyError, ValueError):
            continue
        rows.append(ApogeeRow(
            bits=int(data["bits"]),
            archive_path=str(Path(data["archive_path"]).relative_to(repo_root))
                        if Path(data["archive_path"]).is_relative_to(repo_root)
                        else data["archive_path"],
            archive_size_bytes=int(data["archive_size_bytes"]),
            delta_bytes=int(data["delta_bytes"]),
            rate_score_delta=float(data["rate_component_score_delta_vs_pr106"]),
            distortion_risk=str(data["distortion_risk"]),
            predicted_band=str(data["predicted_score_band"]),
            rel_err_pct=float(data["rel_err_pct_per_weight"]),
            pareto_dominated_by=None,
            predicted_low=lo,
            predicted_high=hi,
        ))
    # Sort ascending by bits for stable display
    rows.sort(key=lambda r: r.bits)
    # Mark Pareto-dominated configs
    for i, r in enumerate(rows):
        for j, other in enumerate(rows):
            if i == j:
                continue
            if other.archive_size_bytes < r.archive_size_bytes and other.risk_rank() <= r.risk_rank():
                r.pareto_dominated_by = other.bits
                break
    return rows


def _format_table(rows: list[ApogeeRow]) -> str:
    if not rows:
        return "(no apogee_intN repack manifests found)"
    out_lines: list[str] = []
    out_lines.append(f"PR106 baseline: 0.{int(PR106_BASELINE_SCORE * 1e8):08d}, 186,239 bytes")
    out_lines.append("")
    header = f"{'bits':>4}  {'bytes':>8}  {'delta':>8}  {'rate Δ':>9}  {'risk':<16}  {'predicted band':<18}  {'pareto':<10}"
    out_lines.append(header)
    out_lines.append("-" * len(header))
    for r in rows:
        if r.pareto_dominated_by is not None:
            pareto_str = f"DOM by int{r.pareto_dominated_by}"
        else:
            pareto_str = "FRONTIER"
        out_lines.append(
            f"{r.bits:>4}  {r.archive_size_bytes:>8,}  {r.delta_bytes:>+8,}  "
            f"{r.rate_score_delta:>+9.4f}  {r.distortion_risk:<16}  {r.predicted_band:<18}  {pareto_str:<10}"
        )
    out_lines.append("")
    out_lines.append("=== DISPATCH ONE-LINERS (Pareto-frontier configs only) ===")
    out_lines.append("")
    for r in rows:
        if r.pareto_dominated_by is not None:
            continue
        beats_baseline = r.predicted_high < PR106_BASELINE_SCORE
        marker = "🎯" if beats_baseline else "  "
        out_lines.append(f"{marker} int{r.bits}  predicted [{r.predicted_low:.3f}, {r.predicted_high:.3f}]  "
                        f"({'beats' if beats_baseline else 'may match/exceed'} PR106 0.20946):")
        out_lines.append(
            f"    APOGEE_INTN_BITS={r.bits} .venv/bin/python scripts/launch_lane_on_vastai.py full \\\n"
            f"      --lane-script scripts/remote_lane_apogee_intN.sh \\\n"
            f"      --label lane_apogee_int{r.bits}_pr106 \\\n"
            f"      --predicted-band {r.predicted_low} {r.predicted_high} \\\n"
            f"      --estimated-cost 0.30 --council-priority 1 --max-dph 0.30"
        )
        out_lines.append("")
    out_lines.append("(All dispatches use the same scripts/remote_lane_apogee_intN.sh wrapper;")
    out_lines.append(" APOGEE_INTN_BITS=N env var picks bit-width — magic byte = 0xA0 | bits.)")
    return "\n".join(out_lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true",
                        help="Machine-readable JSON output instead of human table.")
    args = parser.parse_args(argv)

    rows = scan(REPO_ROOT)
    if args.json:
        out = {
            "pr106_baseline_score": PR106_BASELINE_SCORE,
            "pr106_baseline_bytes": 186239,
            "n_configs": len(rows),
            "n_pareto_frontier": sum(1 for r in rows if r.pareto_dominated_by is None),
            "rows": [asdict(r) for r in rows],
        }
        print(json.dumps(out, indent=2))
    else:
        print(_format_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
