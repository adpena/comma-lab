#!/usr/bin/env python3
"""Read realized-vs-projected rungs off a live ``ddm_jg3`` solve checkpoint.

The jg2 S2 spec requires realized-vs-projected printed at n = 3 / 12 / 48 / 150 /
600.  This reads them off the SAME run rather than launching five runs, which is
only legitimate because ``ddm_jg3_joint_solve`` visits pairs in a **seeded
permutation**: with a shuffled order every prefix of the run is an unbiased random
sample of the field, so rung ``n`` is a genuine n-sample.  With the natural sorted
order it would be a contiguous pair prefix, which ``ddm_bp2``/``ddm_na2`` measured
as a DIFFERENT POPULATION (pose prefixes 2.54-4.21x harder, seg prefixes 0.95-0.97x
easier) -- the canonical false-negative shape.

Axis: ``[macOS-CPU advisory]`` · ``score_claim=false`` · ``promotable=false``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))

import ddm_jg3_joint_solve as jg3  # noqa: E402

DEFAULT_RUNGS = (3, 12, 48, 150, 300, 600)


def load_rows(checkpoint: Path) -> list[dict]:
    rows = []
    for line in checkpoint.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def rung_table(
    rows: list[dict], rungs=DEFAULT_RUNGS, measured_archive_delta_bytes: int | None = None
) -> list[dict]:
    out = []
    for n in rungs:
        if len(rows) < n:
            continue
        head = rows[:n]
        repaired = sum(r["repaired"] for r in head)
        tokens = sum(r["tokens_changed"] for r in head)
        if tokens == 0:
            continue
        projection = jg3.project(
            repaired,
            tokens,
            n,
            measured_archive_delta_bytes=measured_archive_delta_bytes,
        )
        out.append(
            {
                "n": n,
                "flips_before": sum(r["flips_before"] for r in head),
                "repaired": repaired,
                "tokens": tokens,
                "yield": repaired / tokens,
                "repair_fraction": repaired / max(sum(r["flips_before"] for r in head), 1),
                "pairs_with_zero_accept": sum(
                    1 for r in head if r["tokens_changed"] == 0
                ),
                "packing_residual_max": max(r["packing_residual_max"] for r in head),
                "seconds_per_pair": sum(r["seconds"] for r in head) / n,
                "projected_S": projection["projected_S"],
                "net_delta_S": projection["net_delta_S"],
                "clears_sub_015": projection["clears_sub_015"],
                "rate_source": projection["rate_source"],
            }
        )
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--measured-archive-delta-bytes", type=int, default=None)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)

    rows = load_rows(Path(args.checkpoint))
    table = rung_table(
        rows, measured_archive_delta_bytes=args.measured_archive_delta_bytes
    )
    print(f"pairs solved so far: {len(rows)}")
    print(
        f"{'n':>5} {'repaired':>9} {'tokens':>7} {'yield':>7} {'repair%':>8} "
        f"{'netS':>11} {'projS':>10} {'clears':>7} {'s/pair':>7}"
    )
    for row in table:
        print(
            f"{row['n']:>5} {row['repaired']:>9} {row['tokens']:>7} "
            f"{row['yield']:>7.3f} {row['repair_fraction'] * 100:>7.1f}% "
            f"{row['net_delta_S']:>+11.6f} {row['projected_S']:>10.6f} "
            f"{row['clears_sub_015']!s:>7} {row['seconds_per_pair']:>7.1f}"
        )
    print()
    print(f"break-even yield (rate cancels seg): {jg3.break_even_yield():.4f}")
    print("axis [macOS-CPU advisory] · score_claim=false · promotable=false")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(table, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
