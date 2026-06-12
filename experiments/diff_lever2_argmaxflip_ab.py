#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Diff the Lever-2 (argmax-flip seg surrogate) arm against the live CE basin.

Reads BOTH per-epoch trajectory JSONLs, extracts the EVALUATED rows (the ones
carrying the exact CPU-authority d_seg / d_pose / rate / score), and prints a
matched-global-epoch comparison from the SHARED fork epoch onward. Both arms
start from the IDENTICAL init (the frozen forkpoint), so any d_seg gap at matched
epochs is the lever's effect.

Authority: ``[contest-CPU advisory]`` NON-PROMOTABLE — these are in-basin advisory
d_seg numbers on a sub-frontier base_ch=20 archive, NOT a frontier/score claim
(the rate term here is the small in-basin build_archive size, not the PR110
frontier archive). The A/B answers ONE question: does the score-domain seg
surrogate lower d_seg vs vendored CE on THIS substrate, from the same init?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_CONTROL = Path("experiments/results/torch_vehicle_full_mps_basin_bc20_n600/torch_vehicle_trajectory.jsonl")
_FORK_EPOCH = 426  # the global epoch the forkpoint was frozen at


def _eval_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [r for r in rows if r.get("evaluated") and r.get("d_seg") is not None]


def _nearest(rows: list[dict], target_ep: int) -> dict | None:
    cand = [r for r in rows if r.get("d_seg") is not None]
    if not cand:
        return None
    return min(cand, key=lambda r: abs(r["global_epoch"] - target_ep))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lever-out", type=Path, required=True,
                   help="the lever arm out-dir (contains torch_vehicle_trajectory.jsonl)")
    p.add_argument("--control", type=Path, default=_CONTROL)
    p.add_argument("--fork-epoch", type=int, default=_FORK_EPOCH)
    args = p.parse_args(argv)

    lever = _eval_rows(args.lever_out / "torch_vehicle_trajectory.jsonl")
    control = _eval_rows(args.control)

    print("=" * 78)
    print("LEVER-2 A/B — score-domain argmax-flip seg surrogate vs vendored CE")
    print(f"  fork epoch = {args.fork_epoch} (shared init); [contest-CPU advisory] NON-PROMOTABLE")
    print("=" * 78)

    def _fmt(rows: list[dict], label: str) -> None:
        print(f"\n{label} evaluated rows (from fork epoch {args.fork_epoch}):")
        print(f"  {'global_ep':>9} {'d_seg':>11} {'d_pose':>11} {'rate':>9} {'score':>10} {'bytes':>8}")
        post = [r for r in rows if r["global_epoch"] >= args.fork_epoch - 5]
        if not post:
            print("    (no evaluated rows at/after the fork yet)")
        for r in post:
            print(f"  {r['global_epoch']:>9} {r['d_seg']:>11.6f} "
                  f"{(r.get('d_pose') or 0):>11.6f} {(r.get('rate') or 0):>9.5f} "
                  f"{(r.get('score') or 0):>10.5f} {r.get('archive_bytes')!s:>8}")

    _fmt(control, "CONTROL (CE)")
    _fmt(lever, "LEVER (soft_cosine)")

    # Matched comparison: for each lever eval epoch, the nearest control eval epoch.
    lever_post = [r for r in lever if r["global_epoch"] >= args.fork_epoch - 5]
    print("\n" + "-" * 78)
    print("MATCHED-EPOCH DIFF (lever d_seg vs nearest-control d_seg):")
    print(f"  {'lever_ep':>9} {'d_seg_lev':>11} {'~ctrl_ep':>9} {'d_seg_ctrl':>11} "
          f"{'Δd_seg':>11} {'Δd_pose':>11} {'verdict':>10}")
    any_row = False
    for lr in lever_post:
        cr = _nearest(control, lr["global_epoch"])
        if cr is None:
            continue
        any_row = True
        dseg_lev = lr["d_seg"]
        dseg_ctrl = cr["d_seg"]
        delta = dseg_lev - dseg_ctrl
        dpose_lev = lr.get("d_pose") or 0.0
        dpose_ctrl = cr.get("d_pose") or 0.0
        dpose_delta = dpose_lev - dpose_ctrl
        verdict = "LEVER<CE" if delta < 0 else ("tie" if abs(delta) < 1e-6 else "CE<LEVER")
        print(f"  {lr['global_epoch']:>9} {dseg_lev:>11.6f} {cr['global_epoch']:>9} "
              f"{dseg_ctrl:>11.6f} {delta:>+11.6f} {dpose_delta:>+11.6f} {verdict:>10}")
    if not any_row:
        print("    (lever has no evaluated rows yet — re-run after the first eval lands)")
    print("-" * 78)
    print("NOTE: in-basin advisory d_seg; rate term is the small base_ch=20 build_archive")
    print("size (NOT the PR110 frontier archive). Frontier UNMOVED; this gates, never IS,")
    print("a paired contest-CPU+CUDA exact eval.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
