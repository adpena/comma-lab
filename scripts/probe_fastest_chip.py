#!/usr/bin/env python3
"""Probe Vast.ai supply for the fastest available training chip.

Per the FAST CHIP DIRECTIVE (feedback_fast_chip_directive_no_waiting_20260501.md):
H100 SXM is the new default for time-critical dispatches; T4/4090 only by
explicit opt-in for short (<1h) training. Wall-clock to contest-CUDA score is
the optimization, not $/hr.

Iterates the canonical chip preference list:
    H100_SXM > H100 > H200 > A100_SXM4_80 > A100_SXM4_40 > A100_PCIE
    > RTX_5090 > RTX_4090 > RTX_4080 > A10G > T4

Returns the cheapest offer in the highest-tier with at least one match.

Usage:
    python scripts/probe_fastest_chip.py --max-dph 3.0 --min-disk-gb 60
    python scripts/probe_fastest_chip.py --print-json --max-dph 3.0
    python scripts/probe_fastest_chip.py --tier H100_SXM --max-dph 2.0

Companion to:
    - scripts/launch_lane_on_vastai.py (caller for create_instance)
    - tools/vast_preemption_watchdog.py (auto-redispatch path)
    - PCC9 check_training_dispatches_use_fast_chip (preflight enforcement)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# Canonical fastest → slowest chip preference. Updated 2026-05-01.
# Per CLAUDE.md FAST CHIP DIRECTIVE: H100 SXM is the new default for any
# training that takes > 1h. T4 only for final A++ promotion.
CANONICAL_CHIP_PREFERENCE = [
    "H100_SXM",
    "H100",
    "H200",
    "A100_SXM4_80",
    "A100_SXM4_40",
    "A100_PCIE",
    "RTX_5090",
    "RTX_4090",
    "RTX_4080",
    "A10G",
    "T4",
]


@dataclass
class Offer:
    """One Vast.ai offer that matched the search."""

    offer_id: int
    gpu_name: str
    dph_total: float
    disk_space_gb: int
    inet_down_mbps: int
    reliability: float
    geolocation: str
    cuda_max_good: float

    def to_dict(self) -> dict:
        return {
            "offer_id": self.offer_id,
            "gpu_name": self.gpu_name,
            "dph_total": self.dph_total,
            "disk_space_gb": self.disk_space_gb,
            "inet_down_mbps": self.inet_down_mbps,
            "reliability": self.reliability,
            "geolocation": self.geolocation,
            "cuda_max_good": self.cuda_max_good,
        }


def _resolve_vastai_binary() -> Path:
    """Mirror launch_lane_on_vastai.py's lookup pattern."""
    repo_root = Path(__file__).resolve().parent.parent
    candidates = [
        repo_root / ".venv" / "bin" / "vastai",
        Path(shutil.which("vastai") or ""),
    ]
    for c in candidates:
        if c and c.exists():
            return c
    raise SystemExit(
        "vastai CLI not found. Install via .venv/bin/pip install vastai "
        "or activate a venv with vastai on PATH."
    )


def _search_one_chip(
    vastai: Path,
    chip: str,
    *,
    min_reliability: float,
    max_dph: float,
    min_disk_gb: int,
    min_cuda_vers: float,
    timeout_s: int,
) -> list[Offer]:
    """Run `vastai search offers` for ONE chip name. Empty list on no match."""
    # Per CLAUDE.md "Forbidden Vast.ai create without disk + cuda_vers gate":
    # always include cuda_vers >= 12.4 (or operator override) so the cu124
    # torch wheel actually loads. Otherwise CPU-fallback silent → invalid score.
    query = (
        f"gpu_name={chip} "
        f"reliability>{min_reliability} "
        f"inet_down>200 "
        f"disk_space>{min_disk_gb} "
        f"num_gpus=1 "
        f"cuda_vers>={min_cuda_vers} "
        # Geolocation skip per launch_lane_on_vastai.py (poor inet from KR).
        f"geolocation!=KR"
    )
    cmd = [str(vastai), "search", "offers", query, "-o", "dph", "--raw"]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s, check=False,
        )
    except subprocess.TimeoutExpired:
        return []
    if result.returncode != 0:
        # `vastai search` returns 0 even on no results, so a non-zero rc
        # indicates a real CLI error — emit to stderr but don't raise.
        sys.stderr.write(
            f"  [warn] vastai search for {chip!r} failed: "
            f"{result.stderr.strip()[:200]}\n"
        )
        return []
    try:
        raw = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []
    out: list[Offer] = []
    for o in raw:
        try:
            dph = float(o.get("dph_total", 99.0))
        except (TypeError, ValueError):
            continue
        if dph > max_dph:
            continue
        out.append(Offer(
            offer_id=int(o["id"]),
            gpu_name=str(o.get("gpu_name", chip)),
            dph_total=dph,
            disk_space_gb=int(o.get("disk_space", 0)),
            inet_down_mbps=int(o.get("inet_down", 0)),
            reliability=float(o.get("reliability2", o.get("reliability", 0.0))),
            geolocation=str(o.get("geolocation", "?")),
            cuda_max_good=float(o.get("cuda_max_good", 0.0)),
        ))
    return out


def probe(
    *,
    chip_preference: list[str] | None = None,
    max_dph: float = 3.0,
    min_disk_gb: int = 60,
    min_reliability: float = 0.96,
    min_cuda_vers: float = 12.4,
    only_tier: str | None = None,
    timeout_s: int = 30,
) -> list[Offer]:
    """Return the offers from the highest-tier chip with at least one match.

    If only_tier is set, search ONLY that chip (no walk-down).
    Otherwise iterate CANONICAL_CHIP_PREFERENCE and stop at first non-empty
    result. The returned list is sorted ascending by dph_total.

    Returns empty list if no chip in the preference yields any match.
    """
    vastai = _resolve_vastai_binary()
    chips = [only_tier] if only_tier else (chip_preference or CANONICAL_CHIP_PREFERENCE)
    for chip in chips:
        offers = _search_one_chip(
            vastai, chip,
            min_reliability=min_reliability,
            max_dph=max_dph,
            min_disk_gb=min_disk_gb,
            min_cuda_vers=min_cuda_vers,
            timeout_s=timeout_s,
        )
        if offers:
            offers.sort(key=lambda o: o.dph_total)
            return offers
    return []


def _format_table(offers: list[Offer]) -> str:
    if not offers:
        return "(no offers matched)\n"
    lines = [
        f"{'offer_id':>10}  {'gpu':<14}  {'$/hr':>6}  {'disk':>5}  "
        f"{'mbps':>5}  {'rel':>5}  {'cuda':>5}  geo",
    ]
    for o in offers:
        lines.append(
            f"{o.offer_id:>10}  {o.gpu_name:<14}  {o.dph_total:>6.3f}  "
            f"{o.disk_space_gb:>5}  {o.inet_down_mbps:>5}  "
            f"{o.reliability:>5.2f}  {o.cuda_max_good:>5.1f}  {o.geolocation}"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--max-dph", type=float, default=3.0,
                        help="Max $/hr (default 3.0; 4.0+ for SXM variants)")
    parser.add_argument("--min-disk-gb", type=int, default=60,
                        help="Min instance disk (default 60GB per Bug Class #6)")
    parser.add_argument("--min-reliability", type=float, default=0.96)
    parser.add_argument("--min-cuda-vers", type=float, default=12.4,
                        help="Min CUDA driver. <12.4 silently CPU-falls "
                             "back the cu124 torch wheel.")
    parser.add_argument("--tier", type=str, default=None,
                        help="Search ONLY this chip name (no walk-down).")
    parser.add_argument("--print-json", action="store_true",
                        help="Emit offers as JSON to stdout (for piping into "
                             "launch_lane_on_vastai.py).")
    parser.add_argument("--max-results", type=int, default=10,
                        help="Limit table output (default 10).")
    args = parser.parse_args()

    if os.environ.get("PROBE_FASTEST_CHIP_DRY_RUN"):
        # Allows the unit test to run without hitting the live vastai API.
        sample = Offer(
            offer_id=99999, gpu_name="H100_SXM", dph_total=1.80,
            disk_space_gb=200, inet_down_mbps=1000, reliability=0.99,
            geolocation="US", cuda_max_good=12.6,
        )
        offers = [sample]
    else:
        offers = probe(
            max_dph=args.max_dph,
            min_disk_gb=args.min_disk_gb,
            min_reliability=args.min_reliability,
            min_cuda_vers=args.min_cuda_vers,
            only_tier=args.tier,
        )

    if args.print_json:
        print(json.dumps([o.to_dict() for o in offers[:args.max_results]], indent=2))
    else:
        if not offers:
            print(
                "[probe-fastest-chip] no offers matched. "
                f"max_dph={args.max_dph} min_disk_gb={args.min_disk_gb} "
                f"min_cuda_vers={args.min_cuda_vers}",
                file=sys.stderr,
            )
            return 1
        print(f"[probe-fastest-chip] top tier: {offers[0].gpu_name}")
        print(_format_table(offers[:args.max_results]))

    return 0 if offers else 1


if __name__ == "__main__":
    sys.exit(main())
