#!/usr/bin/env python3
"""Cross-series INTERACTION analyzer for witness training dynamics (task #312 Phase C).

Read-only. Consumes a witness run dir (``run.log`` + ``costate_shadow.jsonl``), computes
windowed lagged cross-correlations + lead/lag structure between every telemetry series
(per-term losses, d_seg/d_pose, λ, schedule/octave, gnorm, blob bytes), and prints (i) a
ranked synergy report and (ii) ADVISORY fine-tune recommendations with the evidence chain.
``--json`` emits the machine-readable report; ``--jsonl-out`` appends the synergy rows to a
file for the costate DECIDE surface / dashboard. NO trainer changes, NO score claims —
every row is [macOS advisory] NON-PROMOTABLE; pointer 0.19110 UNMOVED.

Usage:
  .venv/bin/python tools/witness_dynamics_analyzer.py --run-dir <dir>
  .venv/bin/python tools/witness_dynamics_analyzer.py --run-dir <dir> --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from tac.witness_control.dynamics_analyzer import analyze  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, help="witness run dir (run.log + shadow jsonl)")
    ap.add_argument("--window", type=int, default=40, help="epochs per correlation window")
    ap.add_argument("--max-lag", type=int, default=8, help="max lead/lag in epochs")
    ap.add_argument("--min-abs-corr", type=float, default=0.3,
                    help="drop pairs below this |full-support correlation|")
    ap.add_argument("--top", type=int, default=15, help="how many interactions to print")
    ap.add_argument("--json", action="store_true", help="emit machine-readable report to stdout")
    ap.add_argument("--jsonl-out", default=None,
                    help="append the synergy report as ONE json line to this file")
    args = ap.parse_args(argv)

    rep = analyze(args.run_dir, window=args.window, max_lag=args.max_lag,
                  min_abs_corr=args.min_abs_corr)
    obj = rep.to_obj()

    if args.jsonl_out:
        with open(args.jsonl_out, "a") as fh:
            fh.write(json.dumps(obj) + "\n")
        print(f"[dynamics] appended synergy report -> {args.jsonl_out}", file=sys.stderr)

    if args.json:
        print(json.dumps(obj, indent=2))
        return 0

    print(f"[dynamics] {rep.run_dir}")
    print(f"  series={rep.n_series}  grid_epochs={rep.n_grid}  "
          f"interactions={len(rep.interactions)}  axis={obj['axis']}")
    print("  top interactions (lag>0 => first LEADS second):")
    if not rep.interactions:
        print("    (none above threshold — series too short or uncorrelated)")
    for row in rep.interactions[: args.top]:
        a, b = row["pair"]
        print(f"    {a:<22} ~ {b:<22} r={row['correlation']:+.2f} lag={row['lag']:+d} "
              f"stab={row['stability']:.2f} lead={row['lead']}")
    print("  ADVISORY recommendations (NON-PROMOTABLE; feed costate DECIDE):")
    if not rep.recommendations:
        print("    (none — no strong+stable actionable interaction)")
    for r in rep.recommendations:
        print(f"    [{r['action']}]")
        print(f"        {r['rationale']}")
    print(f"  {obj['pointer']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
