#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Register the exchange-ratio noise-floor canonical equation (ddm_xr1, #1248) into the JSONL
registry (``.omx/state/canonical_equations_registry.jsonl``) so the EQUATIONS leg of the triality
carries the statistic the campaign had been closing rows against without ever measuring.

Equation: ``exchange_ratio_noise_floor_v1`` — sigma_B = 0 over three physical RC64 null
re-encodes, so the entire byte<->distortion exchange-ratio noise floor is the PAIR-level
bootstrap: +/-200 B on JBP1 row A's -2,950 B rate credit, and +/-0.00024 S on FCD3's realized
+0.00194 S (interval excludes zero — the win-win cone stays refused).

Producer: experiments/ddm_xr1_exchange_ratio_noise_floor.py.
Consumers: the ddm_xr1 and ddm_rn1 near-win acceptance ledgers.

CONTAINMENT: pure build + JSONL append; NO scorer, NO Modal, NO Metal, NO launch. The own-vehicle
frontier is UNMOVED — this is apparatus, not a lever. Idempotent: re-running appends another
'registered' event; ``query_equations`` returns the latest payload per equation_id.

Usage:
    .venv/bin/python tools/register_exchange_ratio_noise_floor_equation_20260903.py [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tac.canonical_equations.exchange_ratio_noise_floor_20260903 import (  # noqa: E402
    build_exchange_ratio_noise_floor_v1,
)
from tac.canonical_equations.registry import register_canonical_equation  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="build + validate but do NOT append to the registry",
    )
    args = ap.parse_args(argv)

    eq = build_exchange_ratio_noise_floor_v1()
    print(f"built {eq.equation_id}: {eq.one_line_summary}")
    print(f"  producers={eq.canonical_producers}")
    print(f"  consumers={eq.canonical_consumers}")
    print(f"  anchors={[anchor.anchor_id for anchor in eq.empirical_anchors]}")
    if args.dry_run:
        print("DRY-RUN: not registered.")
        return 0
    register_canonical_equation(
        eq,
        agent="ddm_xr1_exchange_ratio_noise_floor_20260903",
        notes=(
            "MAIN's #1248 estimand, instrumented and measured: physical sigma_B from three "
            "complete RC64 null re-encodes, plus seeded 200-resample PAIR-level bootstraps of "
            "JBP1 row A's byte credit and FCD3's realized dS. Scorer-free (retained per-pair "
            "receipts only); advisory axis; NON-PROMOTABLE; pointer unmoved."
        ),
    )
    print(f"registered {eq.equation_id} into .omx/state/canonical_equations_registry.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
