"""ddm_ren2: per-pair seg comparison between a refit checkpoint and the control.

Three questions the population mean cannot answer, all answered from the per-pair rows
the price producer already retains:

1. **Direction.** How many of the 600 pairs did the refit improve, damage, leave alone?
   A refit that damages a few pairs badly and improves many is a different object from
   one that damages a few and improves none, and only the second closes the axis.
2. **Cells.** ``ddm_rw1``'s unit is argmax cells; the admit bar is 23.6 repaired cells.
   Gross cells moved vs NET cells repaired separates "the refit moved nothing" from
   "the refit moved a lot and it cancelled".
3. **The per-pair ORACLE bound.** A weight change has no per-pair admission lever
   (``ddm_ren1`` §4.2): it moves all 600 frames at once.  But a successor could ship a
   per-pair SELECTOR (600 bits = 75 B) choosing the better renderer per pair -- a
   receiver change, and therefore not this arm's to fire.  The oracle bound prices that
   door before anyone builds it: if picking the better of {shipped, refit} per pair
   still cannot clear the bar, the door is shut for this candidate and nobody has to
   spend a receiver change to learn it.

No score claim, no promotion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CELLS_PER_PAIR = 384 * 512
N_PAIRS = 600
BYTE_TO_SCORE = 25.0 / 37_545_489
SELECTOR_BYTES = 75  # 600 bits, the cheapest possible per-pair selector


def load_rows(path: Path) -> dict[int, float]:
    rows: dict[int, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[int(row["pair"])] = float(row["d_seg"])
    return rows


def load_cells(path: Path) -> dict[int, int]:
    cells: dict[int, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("cells_moved") is not None:
                cells[int(row["pair"])] = int(row["cells_moved"])
    return cells


def report(candidate: Path, control: Path) -> dict:
    cand = load_rows(candidate)
    ctrl = load_rows(control)
    shared = sorted(set(cand) & set(ctrl))
    if not shared:
        raise SystemExit("candidate and control share no pairs")
    c = np.array([cand[p] for p in shared], dtype=np.float64)
    s = np.array([ctrl[p] for p in shared], dtype=np.float64)
    delta_cells = (c - s) * CELLS_PER_PAIR
    cells = load_cells(candidate)
    oracle = np.minimum(c, s)
    oracle_cut = (s.mean() - oracle.mean()) / s.mean()
    oracle_delta_s = 100.0 * (oracle.mean() - s.mean()) + SELECTOR_BYTES * BYTE_TO_SCORE
    return {
        "candidate_rows": str(candidate),
        "control_rows": str(control),
        "pairs": len(shared),
        "is_n600": len(shared) == N_PAIRS,
        "candidate_d_seg": float(c.mean()),
        "control_d_seg": float(s.mean()),
        "cut_fraction": float((s.mean() - c.mean()) / s.mean()),
        "net_cells_repaired": float(-delta_cells.sum()),
        "pairs_improved": int((c < s).sum()),
        "pairs_worse": int((c > s).sum()),
        "pairs_unchanged": int((c == s).sum()),
        "worst_pair_delta_cells": float(delta_cells.max()),
        "best_pair_delta_cells": float(delta_cells.min()),
        "gross_cells_moved_vs_shipped_argmax": (
            int(sum(cells.values())) if len(cells) == len(shared) else None
        ),
        "net_over_gross": (
            float(abs(delta_cells.sum()) / sum(cells.values()))
            if len(cells) == len(shared) and sum(cells.values())
            else None
        ),
        "per_pair_oracle": {
            "d_seg": float(oracle.mean()),
            "cut_fraction_vs_control": float(oracle_cut),
            "selector_bytes_assumed": SELECTOR_BYTES,
            "delta_s_including_selector": float(oracle_delta_s),
            "clears_admit_bar": bool(oracle_delta_s < -2e-5),
            "note": (
                "an UPPER bound on any per-pair selection of {shipped, refit}: it uses "
                "the true per-pair minimum, which no shippable selector can beat, and "
                "charges only the 600-bit selector.  A receiver change, so not this "
                "arm's to fire -- priced here so nobody builds it to find out."
            ),
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-rows", required=True, type=Path)
    parser.add_argument("--control-rows", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    payload = report(args.candidate_rows, args.control_rows)
    text = json.dumps(payload, indent=1, sort_keys=True) + "\n"
    if args.out is not None:
        tmp = args.out.with_suffix(args.out.suffix + ".tmp")
        tmp.write_text(text)
        tmp.replace(args.out)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
