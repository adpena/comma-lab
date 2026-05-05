#!/usr/bin/env python3
"""All-lanes preflight: run every available dispatch dry-run in sequence.

Single command for full pre-dispatch confidence across every launch-ready
PR106-stacking lane. Each sub-tool runs at $0 cost (CPU-only); failures
in any one are reported but DON'T block the others (each lane stands alone).

Currently runs:
  Lane #1: tools/dispatch_dryrun_apogee_intN.py --all-pareto-frontier
           (4 Pareto-frontier bits configs: 4, 5, 6, 8 — int7 dominated)
  Lane #2: tools/dispatch_dryrun_omega_w_v3.py
           (Stages 1+3 + parser-roundtrip; Stages 2+4 require CUDA)

Lane SJ-KL is intentionally NOT included — its end-to-end local mode
requires real prepared pair tensors (heavy fixture). Its codec library
is covered by 26 tests in test_sjkl_basis.py + the legacy-artifact
forensic drift test landed earlier this session.

Exit code: 0 if every lane PASSES, non-zero count of FAILED lanes otherwise.

Usage:
  .venv/bin/python tools/all_lanes_preflight.py
  .venv/bin/python tools/all_lanes_preflight.py -v   # verbose (PASS lines too)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "tools"

LANES = [
    {
        "name": "apogee_intN (PR106 HNeRV signed-intN block-FP)",
        "tool": TOOLS / "dispatch_dryrun_apogee_intN.py",
        "args": ["--all-pareto-frontier"],
    },
    {
        "name": "Lane Ω-W-V3 (water-fill v2 → PR106 HNeRV decoder)",
        "tool": TOOLS / "dispatch_dryrun_omega_w_v3.py",
        "args": [],
    },
]


def _run_lane(lane: dict, verbose: bool) -> tuple[bool, str]:
    args = [sys.executable, str(lane["tool"])] + lane["args"]
    if verbose:
        args.append("--verbose")
    proc = subprocess.run(args, capture_output=True, text=True)
    return proc.returncode == 0, proc.stdout + proc.stderr


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Forward --verbose to each dry-run (shows PASS lines).")
    args = parser.parse_args(argv)

    # Verify all sub-tools exist before running any
    for lane in LANES:
        if not lane["tool"].is_file():
            print(f"FATAL: missing sub-tool {lane['tool'].relative_to(REPO)}",
                  file=sys.stderr)
            return 2

    n_passed = 0
    n_failed = 0
    summary_lines: list[str] = []

    for i, lane in enumerate(LANES, start=1):
        bar = "═" * 70
        print(f"\n{bar}\nLANE #{i}: {lane['name']}\n{bar}")
        passed, output = _run_lane(lane, verbose=args.verbose)
        print(output.rstrip())
        if passed:
            n_passed += 1
            summary_lines.append(f"  ✓ Lane #{i}: {lane['name']} — PASSED")
        else:
            n_failed += 1
            summary_lines.append(f"  ✗ Lane #{i}: {lane['name']} — FAILED")

    bar = "═" * 70
    print(f"\n{bar}\nALL-LANES PREFLIGHT SUMMARY\n{bar}\n")
    for line in summary_lines:
        print(line)
    print()
    if n_failed == 0:
        print(f"ALL {n_passed} LANES GO — every dispatch dry-run passed.")
        print("Operator one-liners from `tools/apogee_intN_pareto.py` + "
              "`bash scripts/remote_lane_omega_w_v3_pr106.sh` will succeed at GPU time.")
        return 0
    print(f"{n_failed} of {n_passed + n_failed} LANES FAILED — DO NOT DISPATCH the failed lanes.")
    print("See per-lane output above for specific check failures.")
    return n_failed


if __name__ == "__main__":
    raise SystemExit(main())
